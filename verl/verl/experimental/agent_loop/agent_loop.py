# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Agent framework for multi-turn rollout and agentic reinforcement learning.
- AgentLoopBase: coroutine based abstract base class for agent loop.
  - SingleTurnAgentLoop: single turn agent loop.
  - ToolAgentLoop: ReAct agent loop with tool calling, with user defined tools.
- AgentLoopWorker: worker class for running agent loop coroutines in parallel.
- AgentLoopManager: manager class for running agent loop workers in parallel.

AgentLoopManager is one specific agent-framework implementation in verl,
and is designed to be fully replaceable by other agent frameworks such as:
- NVIDIA Nemo-Gym
- AWS Bedrock AgentCore
- SWE-agent
- ...
"""

import asyncio
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Optional
from uuid import uuid4

import hydra
import numpy as np
import ray
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.agent_loop.utils import resolve_config_path
from verl.protocol import DataProto
from verl.tools.tool_registry import load_all_tools
from verl.trainer.distillation import is_distillation_enabled
from verl.utils.chat_template import apply_chat_template, initialize_system_prompt
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import RLHFDataset, get_dataset_class
from verl.utils.model import compute_position_id_with_mask
from verl.utils.profiler import simple_timer
from verl.utils.ray_utils import auto_await, get_event_loop
from verl.utils.rollout_trace import (
    RolloutTraceConfig,
    rollout_trace_attr,
)
from verl.utils.tokenizer import (
    build_multimodal_processor_inputs,
    get_processor_token_id,
    normalize_token_ids,
)
from verl.workers.config import (
    HFModelConfig,
    RolloutConfig,
)
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_ROUTING_CACHE_SIZE = 10000


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

    generate_sequences: float = 0.0
    tool_calls: float = 0.0
    compute_score: float = 0.0
    num_preempted: int = -1  # -1 means not available


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    response_logprobs: Optional[list[float]] = None
    """Log probabilities for the response tokens."""
    routed_experts: Optional[Any] = None
    """Routed experts for the total tokens."""
    multi_modal_data: Optional[dict[str, Any]] = None
    """Multi-modal data for multi-modal tools."""
    reward_score: Optional[float] = None
    """Reward score for the trajectory."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""
    mm_processor_kwargs: Optional[dict[str, Any]] = None
    """Processor/backend kwargs that must stay aligned across rollout and training paths."""

    def as_dict(self) -> dict[str, Any]:
        """Convert agent loop output to a dictionary."""
        output = self.model_dump(exclude_unset=True)

        output["prompts"] = torch.tensor(output.pop("prompt_ids"), dtype=torch.int64)
        output["responses"] = torch.tensor(output.pop("response_ids"), dtype=torch.int64)
        output["response_mask"] = torch.tensor(output.pop("response_mask"), dtype=torch.int64)

        response_logprobs = output.pop("response_logprobs", None)
        if response_logprobs is not None:
            output["rollout_log_probs"] = torch.tensor(response_logprobs, dtype=torch.float32)

        routed_experts = output.pop("routed_experts", None)
        if routed_experts is not None:
            output["routed_experts"] = torch.tensor(routed_experts, dtype=torch.int64)

        # rm_scores: reward score for each token
        reward_score = output.pop("reward_score", None)
        if reward_score is not None:
            rm_scores = torch.zeros_like(output["response_mask"], dtype=torch.float32)
            rm_scores[-1] = reward_score
            output["rm_scores"] = rm_scores

        teacher_ids, teacher_logprobs = (
            output["extra_fields"].pop("teacher_ids", None),
            output["extra_fields"].pop("teacher_logprobs", None),
        )
        if teacher_ids is not None:
            output["teacher_ids"] = teacher_ids
        if teacher_logprobs is not None:
            output["teacher_logprobs"] = teacher_logprobs
        return output


class _InternalAgentLoopOutput(AgentLoopOutput):
    """Internal agent loop output with padded sequences."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_ids: torch.Tensor
    """Padded response token ids."""
    input_ids: torch.Tensor
    """Padded input ids(prompt_ids + response_ids)."""
    position_ids: torch.Tensor
    """Padded position ids."""
    response_mask: torch.Tensor
    """Padded response mask."""
    attention_mask: torch.Tensor
    """Padded attention mask."""
    response_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities for the response tokens."""
    teacher_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities from teacher model for prompt/response tokens."""
    teacher_ids: Optional[torch.Tensor] = None
    """Padded token ids corresponding to the teacher log probabilities."""
    attack_teacher_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities from attack teacher model (negative hint prompt)."""
    routed_experts: Optional[torch.Tensor] = None
    """Padded routed experts for the total tokens."""
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    """Multi-modal inputs for processors (e.g. pixel_values, image_grid_thw, video_grid_thw)."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class DictConfigWrap:
    """Wrapper for DictConfig to avoid hydra.utils.instantiate recursive resolve."""

    def __init__(self, config: DictConfig):
        self.config = config


class ToolListWrap:
    """Wraps a tool list so ``hydra.utils.instantiate`` doesn't recursively
    resolve its elements (which would demote them to ``DictConfig``)."""

    def __init__(self, tools: list):
        self.tools = tools


class AgentLoopBase(ABC):
    """An agent loop takes an input message, chat with OpenAI compatible LLM server and interact with various
    environments.

    Args:
        trainer_config (DictConfig): whole config for main entrypoint.
        server_manager (LLMServerClient): OpenAI compatible LLM server manager.
        tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
        processor (AutoProcessor): Processor for process messages.
        dataset_cls (type[Dataset]): Dataset class for creating dataset, Defaults to RLHFDataset.
        data_config (DictConfigWrap): Dataset config.
    """

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: LLMServerClient,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        dataset_cls: type[RLHFDataset],
        data_config: DictConfigWrap,
        **kwargs,
    ):
        self.config = trainer_config.config
        self.rollout_config = self.config.actor_rollout_ref.rollout
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.dataset_cls = dataset_cls
        self.data_config = data_config.config
        self.apply_chat_template_kwargs = self.data_config.get("apply_chat_template_kwargs", {})
        self.mm_processor_kwargs = self.data_config.get("mm_processor_kwargs", {})
        processing_class = self.processor if self.processor is not None else self.tokenizer
        self.system_prompt = initialize_system_prompt(processing_class, **self.apply_chat_template_kwargs)
        self.loop = get_event_loop()

    def _get_mm_processor_kwargs(self, audio_data: Optional[list[Any]] = None) -> dict[str, Any]:
        mm_processor_kwargs = dict(self.mm_processor_kwargs or {})
        if audio_data is not None and "sampling_rate" not in mm_processor_kwargs:
            sampling_rate = getattr(getattr(self.processor, "feature_extractor", None), "sampling_rate", None)
            if sampling_rate is not None:
                mm_processor_kwargs["sampling_rate"] = int(sampling_rate)
        return mm_processor_kwargs

    async def process_vision_info(self, messages: list[dict]) -> dict:
        """Backward-compatible wrapper for multi-modal extraction."""
        return await self.process_multi_modal_info(messages)

    async def process_multi_modal_info(self, messages: list[dict]) -> dict:
        """Extract images, videos and audios from messages.

        Args:
            messages (list[dict]): Input messages.

        Returns:
            dict: Multi-modal data with keys like "images", "videos" and "audios".
        """
        multi_modal_data = {}
        if self.processor is not None:
            image_patch_size = getattr(getattr(self.processor, "image_processor", None), "patch_size", 14)
            if hasattr(self.dataset_cls, "process_multi_modal_info"):
                images, videos, audios = await self.dataset_cls.process_multi_modal_info(
                    messages, image_patch_size=image_patch_size, config=self.data_config
                )
            else:
                images, videos = await self.dataset_cls.process_vision_info(
                    messages, image_patch_size=image_patch_size, config=self.data_config
                )
                audios = None
            if images is not None:
                multi_modal_data["images"] = images
            if videos is not None:
                multi_modal_data["videos"] = videos
            if audios is not None:
                multi_modal_data["audios"] = audios

        return multi_modal_data

    async def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        images: list[Image.Image] = None,
        videos: list[tuple[torch.Tensor, dict]] = None,
        audios: list[Any] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        remove_system_prompt: bool = False,
    ):
        """Apply chat template to messages with optional tools, images, and videos.

        Args:
            messages (list[dict]): Input messages.
            tools (list[dict], optional): Tools schemas. Defaults to None.
            images (list[Image.Image], optional): Input images. Defaults to None.
            videos (list[tuple[torch.Tensor, dict]], optional): Input videos. Defaults to None.
            remove_system_prompt (bool, optional): Whether to remove system prompt. Defaults to False.

        Returns:
            list[int]: Prompt token ids.
        """
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: apply_chat_template(
                    self.processor,
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )

            model_inputs = build_multimodal_processor_inputs(
                self.processor,
                text=[raw_prompt],
                images=images,
                videos=videos,
                audio=audios,
                mm_processor_kwargs=mm_processor_kwargs
                if mm_processor_kwargs is not None
                else self._get_mm_processor_kwargs(audios),
            )
            prompt_ids = normalize_token_ids(model_inputs.pop("input_ids"))
        else:
            tokenized_prompt = await self.loop.run_in_executor(
                None,
                lambda: apply_chat_template(
                    self.tokenizer,
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
            prompt_ids = normalize_token_ids(tokenized_prompt)

        if remove_system_prompt:
            prompt_ids = prompt_ids[len(self.system_prompt) :]

        return prompt_ids

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run agent loop to interact with LLM server and environment.

        Args:
            sampling_params (Dict[str, Any]): LLM sampling params.
            **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

        Returns:
            AgentLoopOutput: Agent loop output.
        """
        raise NotImplementedError


"""Agent loop registry: key is agent_name, value is a dict of agent loop config
used by hydra.utils.instantiate to initialize agent loop instance.

https://hydra.cc/docs/advanced/instantiate_objects/overview/
"""
_agent_loop_registry: dict[str, dict] = {}


def register(agent_name: str):
    """Register agent loop class."""

    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator


class AgentLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop.

    Args:
        config (DictConfig): whole config for main entrypoint.
        llm_client (LLMServerClient): Client for the LLM server.
        teacher_client (dict[str, LLMServerClient]): Client for multiple teacher servers.
        reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming reward computation.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: dict[str, LLMServerClient] = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.config = config
        self.llm_client = llm_client
        self.teacher_client = teacher_client
        self.reward_loop_worker_handles = reward_loop_worker_handles

        rollout_config, model_config = config.actor_rollout_ref.rollout, config.actor_rollout_ref.model
        self.rollout_config: RolloutConfig = omega_conf_to_dataclass(rollout_config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config)

        self.dataset_cls = get_dataset_class(config.data)
        self.tokenizer = self.model_config.tokenizer
        self.processor = self.model_config.processor
        self.mm_processor_kwargs = config.data.get("mm_processor_kwargs", {})

        # Online policy distillation
        self.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})

        self.distillation_enabled = is_distillation_enabled(config.distillation)
        if self.distillation_enabled:
            from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager

            self.teacher_key: str = config.distillation.teacher_key
            self.teacher_server_manager = AsyncTeacherLLMServerManager(
                config=config,
                teacher_client=teacher_client,
            )

        # Load tools once per worker; each trajectory just reuses self.tools.
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        function_tool_path = self.rollout_config.multi_turn.function_tool_path
        self.tools = load_all_tools(
            tool_config_path=resolve_config_path(tool_config_path) if tool_config_path else None,
            function_tool_path=resolve_config_path(function_tool_path) if function_tool_path else None,
        )

        # Load custom agent loop implementations from config path
        agent_loop_config_path = self.rollout_config.agent.agent_loop_config_path
        if agent_loop_config_path:
            resolved_path = resolve_config_path(agent_loop_config_path)
            agent_loop_configs = OmegaConf.load(resolved_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config
        if self.model_config.get("custom_chat_template", None) is not None:
            if self.model_config.processor is not None:
                self.model_config.processor.chat_template = self.model_config.custom_chat_template
            self.model_config.tokenizer.chat_template = self.model_config.custom_chat_template

        trace_config = self.rollout_config.trace
        RolloutTraceConfig.init(
            self.rollout_config.trace.project_name,
            self.rollout_config.trace.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
            trace_config.get("max_samples_per_step_per_worker", None),
        )

    def _get_mm_processor_kwargs(self, audio_data: Optional[list[Any]] = None) -> dict[str, Any]:
        """Return multimodal processor kwargs with audio sampling-rate defaults."""
        mm_processor_kwargs = dict(self.mm_processor_kwargs or {})
        if audio_data is not None and "sampling_rate" not in mm_processor_kwargs:
            sampling_rate = getattr(getattr(self.processor, "feature_extractor", None), "sampling_rate", None)
            if sampling_rate is not None:
                mm_processor_kwargs["sampling_rate"] = int(sampling_rate)
        return mm_processor_kwargs

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        config = self.rollout_config
        validate = batch.meta_info.get("validate", False)
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
            params["top_p"] = 1.0
            params["top_k"] = -1
            params["temperature"] = 0

        # override sampling params for validation
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker

        # For n rollouts per sample, we trace all n rollouts for selected samples
        # Note: This sampling happens per-worker, so total traces = max_samples_per_worker * num_workers * n
        if max_samples_per_worker is not None:
            unique_sample_indices = np.unique(index)
            if max_samples_per_worker < len(unique_sample_indices):
                selected_samples = set(
                    np.random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist()
                )
                traced_indices = set(i for i in range(len(batch)) if index[i] in selected_samples)
            else:
                traced_indices = set(range(len(batch)))
        else:
            traced_indices = set(range(len(batch)))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        # NOTE: __do_sample__ is an internal per-sample override used by REMAX combined rollout.
        # Do not forward it to concrete agent loops, which may reject unknown kwargs.
        per_sample_do_sample = batch.non_tensor_batch.get("__do_sample__")
        tasks = []
        for i in range(len(batch)):
            trace_this_sample = i in traced_indices
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items() if k != "__do_sample__"}
            sample_sampling_params = dict(sampling_params)
            if not validate and per_sample_do_sample is not None and not bool(per_sample_do_sample[i]):
                apply_greedy_sampling_params(sample_sampling_params)
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(sample_sampling_params, trajectory_info[i], trace=trace_this_sample, **kwargs)
                )
            )
        outputs = await asyncio.gather(*tasks)

        output = self._postprocess(
            outputs, input_non_tensor_batch=batch.non_tensor_batch, validate=batch.meta_info.get("validate", False)
        )
        return output

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ) -> _InternalAgentLoopOutput:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=trace,
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.llm_client,
                tokenizer=self.tokenizer,
                processor=self.processor,
                dataset_cls=self.dataset_cls,
                data_config=DictConfigWrap(self.config.data),
                tools=ToolListWrap(self.tools),
            )
            output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)
            return await self._agent_loop_postprocess(output, trajectory["validate"], **kwargs)

    async def _agent_loop_postprocess(self, output, validate, **kwargs) -> _InternalAgentLoopOutput:
        """Perform post-processing operations on the output of each individual agent loop."""
        output.extra_fields["raw_prompt"] = kwargs["raw_prompt"]

        # Some AgentLoop may have already computed the reward score, e.g SWE-agent.

        # NOTE: consistent with the legacy batch version of generate_sequences that existed in the
        # deprecated vLLM SPMD rollout implementation.
        # prompt_ids: left padded with zeros (e.g., [0,0,0,0,1,2,3,4])
        # response_ids: right padded with zeros (e.g., [5,6,7,8,0,0,0,0])
        # input_ids: concatenation of prompt + response
        # Mask:
        # For example, if the prompt is [1,2,3,4] and the response is [5,6,7,(tool start)8,9(tool end),10,11,12]
        # - prompt_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [0,0,0,0,1,1,1,1]
        # - response_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
        # attention_mask: concatenation of prompt_attention_mask and response_attention_mask
        #   e.g., [0,0,0,0,1,1,1,1(prompt),1,1,1,1,1,1,1,1,1,1,1,0,0,0,0(response)]
        # - response_mask: 1s for LLM generated tokens, 0 for tool response/padding tokens
        #   e.g., [1,1,1,1,1,1,1,(tool start),0,0(tool end),1,1,0,0,0,0]
        # - position_ids: sequential positions for tokens, starting at 0
        #   e.g., [0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,0,0,0]

        # TODO(wuxibin): remove padding and use tensordict.
        self.tokenizer.padding_side = "left"
        prompt_output = self.tokenizer.pad(
            {"input_ids": output.prompt_ids},
            padding="max_length",
            max_length=self.rollout_config.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if prompt_output["input_ids"].dim() == 1:
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
            prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

        self.tokenizer.padding_side = "right"
        response_output = self.tokenizer.pad(
            {"input_ids": output.response_ids},
            padding="max_length",
            max_length=self.rollout_config.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if response_output["input_ids"].dim() == 1:
            response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
            response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

        response_mask_output = self.tokenizer.pad(
            {"input_ids": output.response_mask},
            padding="max_length",
            max_length=self.rollout_config.response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        if response_mask_output["input_ids"].dim() == 1:
            response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

        response_logprobs = None
        if output.response_logprobs is not None:
            pad_size = self.rollout_config.response_length - len(output.response_logprobs)
            response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

        response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
        attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
        input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

        routed_experts = None
        if output.routed_experts is not None:
            total_length = input_ids.shape[1]
            length, layer_num, topk_num = output.routed_experts.shape
            if isinstance(output.routed_experts, np.ndarray):
                routed_experts_array = output.routed_experts
                if not routed_experts_array.flags.writeable:
                    routed_experts_array = routed_experts_array.copy()
                experts_tensor = torch.from_numpy(routed_experts_array)
            elif isinstance(output.routed_experts, torch.Tensor):
                experts_tensor = output.routed_experts
            else:
                raise TypeError(f"Unsupported type for routed_experts: {type(output.routed_experts)}")
            routed_experts = torch.zeros(1, total_length, layer_num, topk_num, dtype=experts_tensor.dtype)

            # Calculate start position: left padding means original prompt starts at the end
            start_pos = prompt_output["input_ids"].shape[1] - len(output.prompt_ids)
            end_pos = min(start_pos + length, total_length)

            # Add boundary checks for robustness
            if start_pos < 0 or end_pos > total_length:
                raise ValueError(
                    f"Invalid position range: start_pos={start_pos}, end_pos={end_pos}, total_length={total_length}"
                )

            routed_experts[:, start_pos:end_pos] = experts_tensor.unsqueeze(0)

        multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
        position_ids = self._compute_position_ids(
            input_ids,
            attention_mask,
            multi_modal_inputs,
            output.mm_processor_kwargs
            if output.mm_processor_kwargs is not None
            else self._get_mm_processor_kwargs(
                output.multi_modal_data.get("audios") if output.multi_modal_data else None
            ),
        )
        await self._compute_score([output], kwargs=kwargs)
        await self._compute_teacher_logprobs(
            output,
            prompt_ids=output.prompt_ids,
            response_ids=output.response_ids,
            validate=validate,
            sample_kwargs=kwargs,
        )
        teacher_ids, teacher_logprobs = (
            output.extra_fields.pop("teacher_ids", None),
            output.extra_fields.pop("teacher_logprobs", None),
        )
        teacher_prompt_len = output.extra_fields.pop("teacher_prompt_len", None)
        attack_teacher_logprobs_raw = output.extra_fields.pop("attack_teacher_logprobs", None)
        attack_teacher_prompt_len = output.extra_fields.pop("attack_teacher_prompt_len", None)
        if teacher_ids is not None and teacher_logprobs is not None:
            # TODO(wuxibin): remove padding and use tensordict.
            from verl.experimental.teacher_loop.teacher_manager import _pad_teacher_outputs

            student_prompt_len = len(output.prompt_ids)
            response_len = len(output.response_ids)

            if teacher_prompt_len is not None:
                # Positive self-distillation: teacher has a different (longer) prompt.
                # Extract only the response logprobs from teacher output.
                # teacher_logprobs[i] = logprob of token[i+1], so response tokens'
                # logprobs are at positions [teacher_prompt_len-1 : teacher_prompt_len+response_len-1].
                resp_start = teacher_prompt_len - 1
                resp_end = teacher_prompt_len + response_len - 1
                teacher_response_logprobs = teacher_logprobs[resp_start:resp_end]
                teacher_response_ids = teacher_ids[resp_start:resp_end]

                # We need to place the teacher's response logprobs into a padded tensor
                # that matches the student's (prompt_width + response_width) layout, so
                # downstream code (remove_pad -> no_padding_2_padding) extracts the right
                # positions.
                #
                # no_padding_2_padding extracts values[P-1 : P+R-1] (left-shift by 1),
                # where P = student_prompt_len, R = response_len.  So we must place:
                #   - teacher_response_logprobs[0] at the P-th position (index P-1)
                #   - teacher_response_logprobs[1:] at indices P .. P+R-2
                #
                # Build a flat tensor of length (P + R) with dummy prompt entries,
                # then pass through _pad_teacher_outputs to get (prompt_width + response_width).
                prompt_width = prompt_output["input_ids"].shape[1]
                response_width = response_output["input_ids"].shape[1]

                # Place logprobs so that values[P-1] = teacher_response_logprobs[0], etc.
                flat_logprobs = torch.zeros(
                    (student_prompt_len + response_len,) + teacher_response_logprobs.shape[1:],
                    dtype=teacher_response_logprobs.dtype,
                )
                # teacher_response_logprobs has length response_len,
                # flat_logprobs[student_prompt_len - 1 :] has length response_len + 1,
                # so we slice to match exactly.
                flat_logprobs[student_prompt_len - 1 : student_prompt_len - 1 + response_len] = teacher_response_logprobs

                flat_ids = torch.full(
                    (student_prompt_len + response_len,) + teacher_response_ids.shape[1:],
                    self.tokenizer.pad_token_id,
                    dtype=teacher_response_ids.dtype,
                )
                flat_ids[student_prompt_len - 1 : student_prompt_len - 1 + response_len] = teacher_response_ids

                teacher_ids, teacher_logprobs = _pad_teacher_outputs(
                    flat_ids,
                    flat_logprobs,
                    prompt_width=prompt_width,
                    response_width=response_width,
                    prompt_length=student_prompt_len,
                    response_length=response_len,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            else:
                teacher_ids, teacher_logprobs = _pad_teacher_outputs(
                    teacher_ids,
                    teacher_logprobs,
                    prompt_width=prompt_output["input_ids"].shape[1],
                    response_width=response_output["input_ids"].shape[1],
                    prompt_length=student_prompt_len,
                    response_length=response_len,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        # Pad attack teacher log-probs if present (negative self-distillation).
        attack_teacher_logprobs = None
        if attack_teacher_logprobs_raw is not None:
            from verl.experimental.teacher_loop.teacher_manager import _pad_teacher_outputs

            student_prompt_len = len(output.prompt_ids)
            response_len = len(output.response_ids)
            prompt_width = prompt_output["input_ids"].shape[1]
            response_width = response_output["input_ids"].shape[1]

            # Attack teacher has a different prompt, extract response logprobs.
            resp_start = attack_teacher_prompt_len - 1
            resp_end = attack_teacher_prompt_len + response_len - 1
            attack_response_logprobs = attack_teacher_logprobs_raw[resp_start:resp_end]

            flat_attack_logprobs = torch.zeros(
                (student_prompt_len + response_len,) + attack_response_logprobs.shape[1:],
                dtype=attack_response_logprobs.dtype,
            )
            flat_attack_logprobs[student_prompt_len - 1 : student_prompt_len - 1 + response_len] = attack_response_logprobs

            # We don't need attack_teacher_ids for loss computation, only logprobs.
            # Create a dummy flat_ids for _pad_teacher_outputs.
            flat_ids_dummy = torch.full(
                (student_prompt_len + response_len, 1),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
            )
            _, attack_teacher_logprobs = _pad_teacher_outputs(
                flat_ids_dummy,
                flat_attack_logprobs,
                prompt_width=prompt_width,
                response_width=response_width,
                prompt_length=student_prompt_len,
                response_length=response_len,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        return _InternalAgentLoopOutput(
            prompt_ids=prompt_output["input_ids"],
            response_ids=response_output["input_ids"],
            input_ids=input_ids,
            position_ids=position_ids,
            response_mask=response_mask,
            attention_mask=attention_mask,
            response_logprobs=response_logprobs,
            routed_experts=routed_experts,
            multi_modal_inputs=multi_modal_inputs,
            multi_modal_data=output.multi_modal_data,
            mm_processor_kwargs=output.mm_processor_kwargs,
            teacher_logprobs=teacher_logprobs,
            teacher_ids=teacher_ids,
            attack_teacher_logprobs=attack_teacher_logprobs,
            reward_score=output.reward_score,
            num_turns=output.num_turns,
            metrics=output.metrics,
            extra_fields=output.extra_fields,
        )

    def _compute_multi_modal_inputs(self, output, input_ids) -> dict[str, torch.Tensor]:
        """Compute multi-modal inputs with image, video and audio."""
        multi_modal_inputs = {}
        if self.processor is None:
            return multi_modal_inputs

        multi_modal_data = output.multi_modal_data or {}
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)

        multi_modal_inputs = build_multimodal_processor_inputs(
            self.processor,
            text=[current_text],
            images=images,
            videos=videos,
            audio=audios,
            mm_processor_kwargs=output.mm_processor_kwargs
            if output.mm_processor_kwargs is not None
            else self._get_mm_processor_kwargs(audios),
        )
        multi_modal_inputs.pop("input_ids", None)
        multi_modal_inputs.pop("attention_mask", None)

        # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
        # because np.array() only keeps the keys for BatchFeature.
        multi_modal_inputs = dict(multi_modal_inputs.convert_to_tensors("pt"))
        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            images_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0])
            multi_modal_inputs["images_seqlens"] = images_seqlens
        return multi_modal_inputs

    def _compute_position_ids(
        self,
        input_ids,
        attention_mask,
        multi_modal_inputs,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Compute position ids for multi-modal inputs."""
        if self.processor is None:
            return compute_position_id_with_mask(attention_mask)  # (1, seq_len)

        multi_modal_kwargs = {
            "image_grid_thw": multi_modal_inputs.get("image_grid_thw"),
            "video_grid_thw": multi_modal_inputs.get("video_grid_thw"),
        }
        # For transformers>=5.3.0, mm_token_type_ids is only used to calculate position ids.
        if multi_modal_inputs.pop("mm_token_type_ids", None) is not None:
            mm_token_type_ids = torch.zeros_like(input_ids)
            image_token_id = get_processor_token_id(self.processor, "image")
            video_token_id = get_processor_token_id(self.processor, "video")
            if image_token_id is not None:
                mm_token_type_ids[0][input_ids[0] == image_token_id] = 1
            if video_token_id is not None:
                mm_token_type_ids[0][input_ids[0] == video_token_id] = 2
            multi_modal_kwargs["mm_token_type_ids"] = mm_token_type_ids

        # Model's get_rope_index has been dynamically bind to the processor.
        vision_position_ids, _ = self.processor.get_rope_index(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **multi_modal_kwargs,
        )
        vision_position_ids = vision_position_ids.transpose(0, 1)  # (3, 1, seq_len) => (1, 3, seq_len)

        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
        text_position_ids = text_position_ids.unsqueeze(0)
        position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  # (1, 4, seq_length)
        return position_ids

    async def _compute_score(self, outputs: list[AgentLoopOutput], kwargs: dict) -> None:
        """Compute reward score for all outputs in a trajectory; assigns result to outputs[-1]."""
        enable_async_reward = self.reward_loop_worker_handles is not None

        final_output = outputs[-1]
        if final_output.reward_score is None and enable_async_reward:
            timing = {}
            with simple_timer("compute_score", timing):
                all_prompts, all_responses, all_input_ids, all_attention_mask, all_position_ids = [], [], [], [], []
                for output in outputs:
                    prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
                    responses = torch.tensor(output.response_ids, dtype=torch.int64)
                    input_ids = torch.cat([prompts, responses], dim=0)
                    attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
                    multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
                    position_ids = self._compute_position_ids(
                        input_ids.unsqueeze(0),
                        attention_mask.unsqueeze(0),
                        multi_modal_inputs,
                        output.mm_processor_kwargs
                        if output.mm_processor_kwargs is not None
                        else self._get_mm_processor_kwargs(
                            output.multi_modal_data.get("audios") if output.multi_modal_data else None
                        ),
                    ).squeeze(0)
                    all_prompts.append(prompts)
                    all_responses.append(responses)
                    all_input_ids.append(input_ids)
                    all_attention_mask.append(attention_mask)
                    all_position_ids.append(position_ids)

                n = len(outputs)
                batch = TensorDict(
                    {
                        "prompts": torch.nn.utils.rnn.pad_sequence(all_prompts, batch_first=True, padding_value=0),
                        "responses": torch.nn.utils.rnn.pad_sequence(all_responses, batch_first=True, padding_value=0),
                        "attention_mask": torch.nn.utils.rnn.pad_sequence(
                            all_attention_mask, batch_first=True, padding_value=0
                        ),
                        "input_ids": torch.nn.utils.rnn.pad_sequence(all_input_ids, batch_first=True, padding_value=0),
                        "position_ids": torch.nn.utils.rnn.pad_sequence(
                            all_position_ids, batch_first=True, padding_value=0
                        ),
                    },
                    batch_size=n,
                )
                non_tensor_batch = {
                    **{k: np.array([v] * n) for k, v in kwargs.items()},
                    "__num_turns__": np.array([o.num_turns for o in outputs]),
                    "tool_extra_fields": np.array([o.extra_fields for o in outputs], dtype=object),
                    "prompt_len": np.array([len(o.prompt_ids) for o in outputs]),
                    "response_len": np.array([len(o.response_ids) for o in outputs]),
                }

                data = DataProto(
                    batch=batch,
                    non_tensor_batch=non_tensor_batch,
                )
                selected_reward_loop_worker_handle = random.choice(self.reward_loop_worker_handles)
                result = await selected_reward_loop_worker_handle.compute_score.remote(data)
                final_output.reward_score = result["reward_score"]
                final_output.extra_fields["reward_extra_info"] = result["reward_extra_info"]
            final_output.metrics.compute_score = timing["compute_score"]

    async def compute_canon_teacher_logprobs_batch(
        self,
        batch: "DataProto",
        teacher_prompts: list[list[dict]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute teacher logprobs for a batch with custom (consensus-conditioned) prompts.

        Returns teacher_logprobs and teacher_ids in the same padded format as normal
        distillation: [bsz, prompt_width + response_width, topk].
        """
        import asyncio
        from verl.experimental.teacher_loop.teacher_manager import _pad_teacher_outputs

        bsz = len(batch)
        prompt_width = batch.batch["prompts"].shape[1]
        response_width = batch.batch["responses"].shape[1]
        attention_mask = batch.batch["attention_mask"]

        async def _compute_one(idx):
            # Get student prompt/response lengths from attention_mask
            student_prompt_len = int(attention_mask[idx, :prompt_width].sum().item())
            response_mask = batch.batch["response_mask"][idx]
            response_len = int(response_mask.sum().item())

            response_ids = batch.batch["responses"][idx]
            response_ids_valid = response_ids[:response_len].tolist()

            teacher_prompt_raw = teacher_prompts[idx]
            teacher_prompt_ids = self.tokenizer.apply_chat_template(
                teacher_prompt_raw, add_generation_prompt=True,
                **self.apply_chat_template_kwargs,
            )
            if hasattr(teacher_prompt_ids, "input_ids"):
                teacher_prompt_ids = teacher_prompt_ids["input_ids"]
            if isinstance(teacher_prompt_ids, dict):
                teacher_prompt_ids = teacher_prompt_ids["input_ids"]
            if not isinstance(teacher_prompt_ids, list):
                teacher_prompt_ids = list(teacher_prompt_ids)

            sequence_ids = teacher_prompt_ids + response_ids_valid
            raw_ids, raw_logprobs = await self.teacher_server_manager.compute_teacher_logprobs_single(
                sequence_ids=sequence_ids,
                multi_modal_data=None,
                mm_processor_kwargs=None,
                routing_key=None,
            )
            # Extract response portion
            teacher_prompt_len = len(teacher_prompt_ids)
            resp_start = teacher_prompt_len - 1
            resp_end = teacher_prompt_len + response_len - 1
            resp_logprobs = raw_logprobs[resp_start:resp_end]
            resp_ids = raw_ids[resp_start:resp_end]
            if resp_logprobs.dim() == 1:
                resp_logprobs = resp_logprobs.unsqueeze(-1)
            if resp_ids.dim() == 1:
                resp_ids = resp_ids.unsqueeze(-1)

            # Build flat tensor [student_prompt_len + response_len, topk] with logprobs
            # placed so that no_padding_2_padding can extract correctly
            flat_logprobs = torch.zeros(
                (student_prompt_len + response_len,) + resp_logprobs.shape[1:],
                dtype=resp_logprobs.dtype,
            )
            flat_logprobs[student_prompt_len - 1 : student_prompt_len - 1 + response_len] = resp_logprobs

            flat_ids = torch.full(
                (student_prompt_len + response_len,) + resp_ids.shape[1:],
                self.tokenizer.pad_token_id,
                dtype=resp_ids.dtype,
            )
            flat_ids[student_prompt_len - 1 : student_prompt_len - 1 + response_len] = resp_ids

            # Pad to [1, prompt_width + response_width, topk]
            padded_ids, padded_logprobs = _pad_teacher_outputs(
                flat_ids,
                flat_logprobs,
                prompt_width=prompt_width,
                response_width=response_width,
                prompt_length=student_prompt_len,
                response_length=response_len,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            return padded_ids, padded_logprobs  # each [1, prompt_width+response_width, topk]

        tasks = [asyncio.create_task(_compute_one(i)) for i in range(bsz)]
        results = await asyncio.gather(*tasks)
        all_ids = torch.cat([r[0] for r in results], dim=0)
        all_logprobs = torch.cat([r[1] for r in results], dim=0)
        return all_logprobs, all_ids  # [bsz, prompt_width+response_width, topk]

    async def _compute_teacher_logprobs(
        self,
        output: AgentLoopOutput,
        prompt_ids: list[int],
        response_ids: list[int],
        validate: bool,
        sample_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Compute teacher logprobs for single sample."""
        # CANON mode: skip teacher logprobs during initial rollout; computed post-consensus
        canon_mode = getattr(getattr(self.config, "distillation", None), "canon_mode", False)
        if canon_mode and not validate:
            return

        if self.distillation_enabled and not validate:
            routing_key = None
            if sample_kwargs is not None:
                routing_value = sample_kwargs.get(self.teacher_key)
                if routing_value is not None:
                    # Non-tensor batch values arrive as 0-d numpy objects / arrays; normalize to Python.
                    routing_key = routing_value.item() if hasattr(routing_value, "item") else routing_value

            # Support different teacher prompt for positive self-distillation.
            # If `teacher_prompt` is present in non_tensor_batch, tokenize it
            # and use teacher_prompt_ids + response_ids as the teacher input.
            teacher_prompt_ids = None
            if sample_kwargs is not None:
                teacher_prompt_raw = sample_kwargs.get("teacher_prompt")
                if teacher_prompt_raw is not None:
                    # teacher_prompt may be a numpy array of dicts or a list of dicts
                    if hasattr(teacher_prompt_raw, "item"):
                        teacher_prompt_raw = teacher_prompt_raw.item()
                    # Convert numpy array to list for apply_chat_template
                    if hasattr(teacher_prompt_raw, "tolist"):
                        teacher_prompt_raw = teacher_prompt_raw.tolist()
                    if isinstance(teacher_prompt_raw, list):
                        teacher_prompt_ids = self.tokenizer.apply_chat_template(
                            teacher_prompt_raw, add_generation_prompt=True,
                            **self.apply_chat_template_kwargs,
                        )
                        if hasattr(teacher_prompt_ids, "input_ids"):
                            # BatchEncoding or similar object
                            teacher_prompt_ids = teacher_prompt_ids["input_ids"]
                        if isinstance(teacher_prompt_ids, dict):
                            teacher_prompt_ids = teacher_prompt_ids["input_ids"]
                        if not isinstance(teacher_prompt_ids, list):
                            teacher_prompt_ids = list(teacher_prompt_ids)

            if teacher_prompt_ids is not None:
                sequence_ids = teacher_prompt_ids + response_ids
            else:
                sequence_ids = prompt_ids + response_ids

            teacher_ids, teacher_logprobs = await self.teacher_server_manager.compute_teacher_logprobs_single(
                sequence_ids=sequence_ids,
                multi_modal_data=output.multi_modal_data,
                mm_processor_kwargs=output.mm_processor_kwargs,
                routing_key=routing_key,
            )
            output.extra_fields["teacher_ids"] = teacher_ids
            output.extra_fields["teacher_logprobs"] = teacher_logprobs
            if teacher_prompt_ids is not None:
                output.extra_fields["teacher_prompt_len"] = len(teacher_prompt_ids)

            # Support attack teacher prompt for negative self-distillation.
            # Two modes:
            #   1. Offline (static): `attack_teacher_prompt` present in dataset — use as-is.
            #   2. Online (dynamic): `online_nsd_attack_prompt_template` present in config —
            #      decode student rollout, call teacher.generate() to produce an attack prompt,
            #      then compute teacher log-probs under that attack prompt.
            if sample_kwargs is not None:
                attack_prompt_raw = sample_kwargs.get("attack_teacher_prompt")
                if attack_prompt_raw is not None:
                    if hasattr(attack_prompt_raw, "item"):
                        attack_prompt_raw = attack_prompt_raw.item()
                    if hasattr(attack_prompt_raw, "tolist"):
                        attack_prompt_raw = attack_prompt_raw.tolist()
                    if isinstance(attack_prompt_raw, list):
                        attack_prompt_ids = self.tokenizer.apply_chat_template(
                            attack_prompt_raw, add_generation_prompt=True,
                            **self.apply_chat_template_kwargs,
                        )
                        if hasattr(attack_prompt_ids, "input_ids"):
                            attack_prompt_ids = attack_prompt_ids["input_ids"]
                        if isinstance(attack_prompt_ids, dict):
                            attack_prompt_ids = attack_prompt_ids["input_ids"]
                        if not isinstance(attack_prompt_ids, list):
                            attack_prompt_ids = list(attack_prompt_ids)

                        attack_sequence_ids = attack_prompt_ids + response_ids
                        _, attack_logprobs = await self.teacher_server_manager.compute_teacher_logprobs_single(
                            sequence_ids=attack_sequence_ids,
                            multi_modal_data=output.multi_modal_data,
                            mm_processor_kwargs=output.mm_processor_kwargs,
                            routing_key=routing_key,
                        )
                        output.extra_fields["attack_teacher_logprobs"] = attack_logprobs
                        output.extra_fields["attack_teacher_prompt_len"] = len(attack_prompt_ids)

            # Online NSD: dynamically generate attack prompt from student rollout via teacher.
            # Triggered when distillation.online_nsd=True in config.
            # Takes priority over any static attack_teacher_prompt in the dataset.
            online_nsd_cfg = getattr(getattr(self.config, "distillation", None), "online_nsd", False)
            if self.distillation_enabled and not validate and online_nsd_cfg:
                # Decode student rollout text
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                # Get problem text from the prompt field in sample_kwargs
                problem_raw = sample_kwargs.get("prompt") if sample_kwargs is not None else None
                if problem_raw is not None:
                    if hasattr(problem_raw, "item"):
                        problem_raw = problem_raw.item()
                    if hasattr(problem_raw, "tolist"):
                        problem_raw = problem_raw.tolist()
                    # Extract problem text from the last user message
                    problem_text = ""
                    if isinstance(problem_raw, list):
                        for msg in reversed(problem_raw):
                            if isinstance(msg, dict) and msg.get("role") == "user":
                                problem_text = msg.get("content", "")
                                break
                    # Strip leading "Problem: " if present
                    if problem_text.startswith("Problem: "):
                        problem_text = problem_text[len("Problem: "):]

                    if not problem_text:
                        return

                    # Build meta-prompt for teacher to generate attack instruction
                    attack_meta_prompt = (
                        "You are an expert Math Educator and AI Prompt Engineer. "
                        "Your task is to analyze the following math problem and a student's existing solution, "
                        "then generate a \"Targeted Attack Prompt\" that exploits the exact reasoning steps "
                        "the student used to cause a highly plausible cognitive error.\n\n"
                        "The student's solution reveals HOW they solved this problem — use that to craft an "
                        "attack targeting their specific reasoning steps.\n\n"
                        "Anatomy of the Targeted Attack Prompt:\n"
                        "1. Persona: Must start exactly with \"You are a student who...\". Describe a specific "
                        "bad habit that would corrupt the EXACT step where this student's reasoning is most fragile.\n"
                        "2. Trigger: Reference the TYPE of reasoning the student used (NOT specific numbers or "
                        "variables from this problem).\n"
                        "3. Flawed Execution: Instruct a shortcut that mirrors the student's approach but "
                        "introduces a subtle error.\n"
                        "4. Fatal Omission: Forbid the specific verification the student performed correctly.\n\n"
                        f"Problem: {problem_text}\n\n"
                        f"Student's Existing Solution:\n{response_text}\n\n"
                        "Output ONLY the \"Targeted Attack Prompt\". "
                        "Start your response with \"You are a student who...\". Keep it concise (2-3 sentences)."
                    )
                    meta_prompt_ids = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": attack_meta_prompt}],
                        add_generation_prompt=True,
                        **self.apply_chat_template_kwargs,
                    )
                    if hasattr(meta_prompt_ids, "input_ids"):
                        meta_prompt_ids = meta_prompt_ids["input_ids"]
                    if isinstance(meta_prompt_ids, dict):
                        meta_prompt_ids = meta_prompt_ids["input_ids"]
                    if not isinstance(meta_prompt_ids, list):
                        meta_prompt_ids = list(meta_prompt_ids)

                    # Call teacher to generate the attack instruction text
                    teacher_key = self.teacher_server_manager._resolve_teacher_key(routing_key)
                    teacher_raw_client = self.teacher_client[teacher_key]
                    gen_output = await teacher_raw_client.generate(
                        request_id=uuid4().hex,
                        prompt_ids=meta_prompt_ids,
                        sampling_params={"max_tokens": 400, "temperature": 0.7, "top_p": 0.9},
                    )
                    attack_instruction = self.tokenizer.decode(
                        gen_output.token_ids, skip_special_tokens=True
                    ).strip()

                    # Validate: must start with "You are a student who"
                    if attack_instruction.startswith("You are a student who"):
                        # Build the attack prompt (same format as offline sol_aware)
                        student_suffix = r"Let's think step by step and output the final answer within \boxed{}."
                        attack_content = (
                            f"Problem: {problem_text}\n\n"
                            f"{attack_instruction}\n\n"
                            f"Now solve the problem following this instruction:\n\n{student_suffix}"
                        )
                        online_attack_ids = self.tokenizer.apply_chat_template(
                            [{"role": "user", "content": attack_content}],
                            add_generation_prompt=True,
                            **self.apply_chat_template_kwargs,
                        )
                        if hasattr(online_attack_ids, "input_ids"):
                            online_attack_ids = online_attack_ids["input_ids"]
                        if isinstance(online_attack_ids, dict):
                            online_attack_ids = online_attack_ids["input_ids"]
                        if not isinstance(online_attack_ids, list):
                            online_attack_ids = list(online_attack_ids)

                        attack_sequence_ids = online_attack_ids + response_ids
                        _, attack_logprobs = await self.teacher_server_manager.compute_teacher_logprobs_single(
                            sequence_ids=attack_sequence_ids,
                            multi_modal_data=output.multi_modal_data,
                            mm_processor_kwargs=output.mm_processor_kwargs,
                            routing_key=routing_key,
                        )
                        output.extra_fields["attack_teacher_logprobs"] = attack_logprobs
                        output.extra_fields["attack_teacher_prompt_len"] = len(online_attack_ids)
                        # Stats for wandb logging (panel: "Attack prompt generation solution")
                        output.extra_fields["online_nsd_rollout_len"] = len(response_ids)
                        output.extra_fields["online_nsd_attack_instr_len"] = len(gen_output.token_ids)
                        output.extra_fields["online_nsd_attack_total_len"] = len(online_attack_ids) + len(response_ids)
                        output.extra_fields["online_nsd_attack_success"] = 1
                    else:
                        # Attack instruction failed validation — log failure stats
                        output.extra_fields["online_nsd_rollout_len"] = len(response_ids)
                        output.extra_fields["online_nsd_attack_instr_len"] = len(gen_output.token_ids)
                        output.extra_fields["online_nsd_attack_total_len"] = 0
                        output.extra_fields["online_nsd_attack_success"] = 0

    def _postprocess(
        self,
        inputs: list[_InternalAgentLoopOutput],
        input_non_tensor_batch: dict | None = None,
        validate: bool = False,
    ) -> DataProto:
        """Process the padded outputs from _run_agent_loop and combine them into a batch."""
        # Convert lists back to tensors and stack them to create a batch.
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)
        if inputs[0].routed_experts is not None:
            optional_outputs["routed_experts"] = torch.cat([input.routed_experts for input in inputs], dim=0)
        if inputs[0].teacher_logprobs is not None and inputs[0].teacher_ids is not None:
            optional_outputs["teacher_logprobs"] = torch.cat([input.teacher_logprobs for input in inputs], dim=0)
            optional_outputs["teacher_ids"] = torch.cat([input.teacher_ids for input in inputs], dim=0)
        # For online NSD some samples may have attack_teacher_logprobs=None (validation failure).
        # Fall back to teacher_logprobs as a neutral substitute so the batch stays consistent;
        # the divergence_gated_sigmoid gate (π_atk > π_ref) will suppress those samples' loss.
        _atk_lps = [inp.attack_teacher_logprobs for inp in inputs]
        if any(lp is not None for lp in _atk_lps):
            _ref = next(lp for lp in _atk_lps if lp is not None)
            _atk_lps_filled = [
                lp if lp is not None else torch.zeros_like(_ref)
                for lp in _atk_lps
            ]
            optional_outputs["attack_teacher_logprobs"] = torch.cat(_atk_lps_filled, dim=0)
        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                # position_ids: [bsz, 3, prompt_length + response_length] or [bsz, prompt_length + response_length]
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            prompt_length = prompt_ids.size(1)
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }
        if self.reward_loop_worker_handles is None and input_non_tensor_batch:
            non_tensor_batch.update(input_non_tensor_batch)

        # add reward_extra_info to non_tensor_batch
        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = list(reward_extra_infos[0].keys())
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        # Add multi_modal_inputs to non_tensor_batch if any samples have them
        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]
        # Collect extra fields from all inputs and convert them to np.ndarray
        # Keep a stable set of keys so downstream batch concat stays consistent across agent loops.
        extra_fields = {}
        default_extra_keys = {
            "turn_scores",
            "tool_rewards",
            "min_global_steps",
            "max_global_steps",
            "extras",
        }
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields) | default_extra_keys
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
            extra_fields[key] = temp_arr

        non_tensor_batch.update(extra_fields)

        # Only include reward_extra_keys in meta_info if rm_scores is in batch
        # This avoids conflicts when reward_tensor is merged later in ray_trainer.py
        if "rm_scores" in batch.keys():
            meta_info = {"metrics": metrics, "reward_extra_keys": reward_extra_keys}
        else:
            meta_info = {"metrics": metrics}

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info=meta_info,
        )


async def get_trajectory_info(step, index, validate):
    """Get trajectory info.

    Args:
        step (int): global steps in the trainer.
        index (list): form datastore extra_info.index column.
        validate (bool): whether is a validate step.

    Returns:
        list: trajectory.
    """
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


class AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers.

    Args:
        config (DictConfig): whole config for main entrypoint.
        llm_client (LLMServerClient): Client for the LLM server.
        teacher_client (dict[str, LLMServerClient]): Client for multiple teacher servers.
        reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming reward computation.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: dict[str, LLMServerClient] = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.config = config
        self.rollout_config = config.actor_rollout_ref.rollout
        self.model_config = config.actor_rollout_ref.model
        self.llm_client = llm_client
        self.teacher_client = teacher_client
        self.reward_loop_worker_handles = reward_loop_worker_handles

        if not hasattr(self, "agent_loop_workers_class"):
            self.agent_loop_workers_class = ray.remote(AgentLoopWorker)

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create agent loop manager."""
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        return instance

    async def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.rollout_config.agent.num_workers

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"agent_loop_worker_{i}" + f"_{uuid4().hex[:8]}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(
                    self.config,
                    self.llm_client,
                    self.teacher_client,
                    self.reward_loop_worker_handles,
                )
            )

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = await asyncio.gather(
            *[
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        output = DataProto.concat(outputs)

        # calculate performance metrics
        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)

        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    @auto_await
    async def compute_canon_teacher_logprobs(self, batch: "DataProto", teacher_prompts: list) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute teacher logprobs for CANON consensus-conditioned prompts.

        Splits across workers for parallelism.
        Returns (teacher_logprobs, teacher_ids), both [bsz, prompt_width+response_width, topk].
        """
        bsz = len(batch)
        num_workers = len(self.agent_loop_workers)
        chunk_size = (bsz + num_workers - 1) // num_workers

        tasks = []
        for w_idx, worker in enumerate(self.agent_loop_workers):
            start = w_idx * chunk_size
            end = min(start + chunk_size, bsz)
            if start >= end:
                break
            chunk_batch = batch.slice(start, end)
            chunk_prompts = teacher_prompts[start:end]
            tasks.append(worker.compute_canon_teacher_logprobs_batch.remote(chunk_batch, chunk_prompts))

        results = await asyncio.gather(*tasks)
        all_logprobs = torch.cat([r[0] for r in results], dim=0)
        all_ids = torch.cat([r[1] for r in results], dim=0)
        return all_logprobs, all_ids

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        t_compute_score = np.array([metric["compute_score"] for chunk in metrics for metric in chunk])
        num_preempted = np.array([metric["num_preempted"] for chunk in metrics for metric in chunk])
        timing["agent_loop/num_preempted/min"] = num_preempted.min()
        timing["agent_loop/num_preempted/max"] = num_preempted.max()
        timing["agent_loop/num_preempted/mean"] = num_preempted.mean()
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()
        timing["agent_loop/compute_score/min"] = t_compute_score.min()
        timing["agent_loop/compute_score/max"] = t_compute_score.max()
        timing["agent_loop/compute_score/mean"] = t_compute_score.mean()

        # batch sequence generation is bounded by the slowest sample
        slowest = np.argmax(t_generate_sequences + t_tool_calls + t_compute_score)
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/compute_score"] = t_compute_score[slowest]
        timing["agent_loop/slowest/num_preempted"] = num_preempted[slowest]

        if "attention_mask" in output.batch:
            attention_mask = output.batch["attention_mask"][slowest]
            timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
            timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing
