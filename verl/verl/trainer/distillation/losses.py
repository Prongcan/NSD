# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
from tensordict import TensorDict

from verl.base_config import BaseConfig
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, DistillationConfig, DistillationLossConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding

DistillationLossFn = Callable[
    [
        ActorConfig,  # actor_config
        DistillationConfig,  # distillation_config
        dict,  # model_output
        TensorDict,  # micro batch input
    ],
    tuple[torch.Tensor, dict[str, Any]],
]


def is_distillation_enabled(config: Optional[DistillationConfig]) -> bool:
    """Check if distillation is enabled based on the provided configuration."""
    if config is None:
        return False
    return config.enabled


@dataclass
class DistillationLossSettings(BaseConfig):
    """
    Settings for a distillation loss function to be registered.

    Args:
        names (str | list[str]): Name(s) to register the distillation loss function under.
        use_topk (bool): Whether the loss function uses top-k log probabilities.
        use_estimator (bool): Whether the loss function uses single-sample KL estimators.
    """

    names: str | list[str] = field(default_factory=list)
    use_topk: bool = False
    use_estimator: bool = False

    _mutable_fields = {"names"}

    def __post_init__(self):
        self.names = [self.names] if isinstance(self.names, str) else self.names
        if sum([self.use_topk, self.use_estimator]) != 1:
            raise ValueError(
                f"Expected only one of use_estimator, use_topk, but got {self.use_estimator=}, {self.use_topk=}."
            )


DISTILLATION_LOSS_REGISTRY: dict[str, DistillationLossFn] = {}
DISTILLATION_SETTINGS_REGISTRY: dict[str, DistillationLossSettings] = {}


def register_distillation_loss(
    loss_settings: DistillationLossSettings,
) -> Callable[[DistillationLossFn], DistillationLossFn]:
    """Register a distillation loss function with the given name."""

    def decorator(func: DistillationLossFn) -> DistillationLossFn:
        for name in loss_settings.names:
            if name in DISTILLATION_LOSS_REGISTRY:
                raise ValueError(f"Distillation loss function with name '{name}' is already registered.")
            DISTILLATION_LOSS_REGISTRY[name] = func
            DISTILLATION_SETTINGS_REGISTRY[name] = loss_settings
        return func

    return decorator


def get_distillation_loss_fn(loss_name: str) -> DistillationLossFn:
    """Get the distillation loss function with a given name."""
    if loss_name not in DISTILLATION_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_LOSS_REGISTRY.keys())}"
        )
    return DISTILLATION_LOSS_REGISTRY[loss_name]


def get_distillation_loss_settings(loss_name: str) -> DistillationLossSettings:
    """Get the distillation loss settings with a given name."""
    if loss_name not in DISTILLATION_SETTINGS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_SETTINGS_REGISTRY.keys())}"
        )
    return DISTILLATION_SETTINGS_REGISTRY[loss_name]


def compute_distillation_loss_range(
    distillation_losses: torch.Tensor, response_mask: torch.Tensor
) -> dict[str, Metric]:
    """Compute min and max distillation loss over valid response tokens."""
    if response_mask.is_nested:
        distillation_losses_response = distillation_losses[response_mask.bool().to_padded_tensor(False)]
    else:
        distillation_losses_response = distillation_losses[response_mask.bool()]
    return {
        "distillation/loss_min": Metric(AggregationType.MIN, distillation_losses_response.min()),
        "distillation/loss_max": Metric(AggregationType.MAX, distillation_losses_response.max()),
    }


def compute_topk_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    data: TensorDict,
    student_logits: torch.Tensor,
    data_format: str,
) -> torch.Tensor:
    """Compute the topk loss in logit processor.

    Returns:
    - distillation_losses: (bsz, seqlen/cp_size)
    - student_mass: (bsz, seqlen/cp_size)
    - teacher_mass: (bsz, seqlen/cp_size)
    """
    match config.strategy:
        # VeOmni uses FSDP2 internally, so its loss computation is identical to FSDP.
        case "fsdp" | "veomni":
            import verl.trainer.distillation.fsdp.losses as fsdp_losses

            distillation_loss_fn = fsdp_losses.compute_forward_kl_topk
        case "megatron":
            import verl.trainer.distillation.megatron.losses as megatron_losses

            distillation_loss_fn = megatron_losses.compute_forward_kl_topk
        case _:
            raise NotImplementedError(f"Unsupported strategy: {config.strategy=}")

    outputs = distillation_loss_fn(
        student_logits=student_logits,
        teacher_topk_log_probs=data["teacher_logprobs"],
        teacher_topk_ids=data["teacher_ids"],
        config=distillation_config,
        data_format=data_format,
    )

    expected_shape = student_logits.shape[:2]
    for k, v in outputs.items():
        assert v.shape == expected_shape, f"Expected shape {expected_shape}, but got {v.shape} for {k=}."

    return outputs


def distillation_ppo_loss(
    config: ActorConfig,
    distillation_config: Optional[DistillationConfig],
    model_output: dict = None,
    data: TensorDict = None,
    dp_group=None,
    student_logits: torch.Tensor = None,
    data_format: str = "thd",
):
    """Loss function used both for logit processor and final policy loss.
    - student_logits is not None, compute the topk loss in logit processor.
    - student_logits is None, compute final policy loss.

    [split sequence across sp/cp groups]
                   |
    [model forward and output logits: (bsz, seqlen/cp_size, vocab_size/tp_size)]
                   |
    [logits processor compute topk loss: (bsz, seqlen/cp_size)]
                   |
    [all gather topk loss across sp/cp groups: (bsz, seqlen)]
                   |
    [combine topk loss with policy loss]

    Args:
        config: Actor configuration.
        distillation_config: Distillation configuration.
        model_output: Model output, including log_probs, entropy.
        data: Micro input batch, contains
          - teacher_logprobs: (bsz, seqlen, topk)
          - teacher_ids: (bsz, seqlen, topk)
        student_logits: (bsz, seqlen/cp_size, vocab_size/tp_size).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - student_logits is not None, return the topk loss tensor (bsz, seqlen/cp_size).
    - student_logits is None, return the final policy loss scalar and metrics.
    """

    # Called as logits processor
    if student_logits is not None:
        return compute_topk_loss(config, distillation_config, data, student_logits, data_format)

    # Called as final policy loss
    distillation_loss_config = distillation_config.distillation_loss
    distill_loss, distill_metrics = distillation_loss(config, distillation_config, model_output, data)
    policy_loss, policy_metrics = ppo_loss(config, model_output, data, dp_group)
    if not distillation_loss_config.use_task_rewards:
        policy_loss = 0.0

    # Combine distillation with policy loss
    policy_metrics.update(distill_metrics)
    distillation_loss_coef = (
        distillation_loss_config.distillation_loss_coef if distillation_loss_config.use_task_rewards else 1.0
    )
    policy_loss += distill_loss * distillation_loss_coef
    policy_metrics["distillation/loss"] = Metric(value=distill_loss, aggregation=AggregationType.SUM)

    return policy_loss, policy_metrics


def distillation_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics.

    Returns:
    - distillation_loss: Aggregated distillation loss scalar.
    - distillation_metrics: Dictionary of metrics.
    """
    assert distillation_config is not None
    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_loss_fn = get_distillation_loss_fn(loss_config.loss_mode)
    distillation_losses, distillation_metrics = distillation_loss_fn(
        config=config,
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )
    response_mask = data["response_mask"]
    loss_agg_mode = config.loss_agg_mode

    distillation_metrics.update(
        compute_distillation_loss_range(distillation_losses=distillation_losses, response_mask=response_mask)
    )
    if loss_config.loss_max_clamp is not None:
        # clamping min is for k1 loss which can be negative
        distillation_losses = distillation_losses.clamp(min=-loss_config.loss_max_clamp, max=loss_config.loss_max_clamp)

    if loss_config.use_policy_gradient:
        # Use negative distillation loss as reward, as done by https://thinkingmachines.ai/blog/on-policy-distillation/.
        policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
        for k, v in config.global_batch_info.items():
            loss_config.global_batch_info[k] = v
        log_prob = no_padding_2_padding(model_output["log_probs"], data)
        old_log_prob = data["old_log_probs"]
        if old_log_prob.is_nested:
            old_log_prob = data["old_log_probs"].to_padded_tensor(0.0)
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        rollout_is_weights = data.get("rollout_is_weights", None)
        distillation_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=-distillation_losses.detach(),
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=loss_config,
            rollout_is_weights=rollout_is_weights,
        )
        pg_metrics = {f"distillation/{k[len('actor/') :]}": v for k, v in pg_metrics.items()}
        distillation_metrics.update(pg_metrics)
    else:
        # Directly backpropagate distillation loss as a supervised loss, as in https://arxiv.org/abs/2306.13649.
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        loss_mask = response_mask
        supervised_clip_ratio = loss_config.get("supervised_clip_ratio", None)
        if supervised_clip_ratio is not None:
            # Trust-region mask: only update tokens whose policy hasn't drifted too far from rollout.
            # Prevents runaway entropy growth in pure supervised mode (no PPO-clip otherwise).
            log_prob = no_padding_2_padding(model_output["log_probs"], data)
            old_log_prob = data["old_log_probs"]
            if old_log_prob.is_nested:
                old_log_prob = old_log_prob.to_padded_tensor(0.0)
            ratio = torch.exp((log_prob - old_log_prob).clamp(min=-20.0, max=20.0))
            # One-sided: only skip tokens that have been sufficiently pushed down (ratio < 1-ε).
            # Tokens with ratio > 1+ε still need to be pushed down, so they must not be masked.
            trust_mask = ratio >= (1.0 - supervised_clip_ratio)
            loss_mask = loss_mask & trust_mask
            distillation_metrics["distillation/supervised_clip_frac"] = (
                (~trust_mask & response_mask).float().sum() / response_mask.float().sum().clamp(min=1)
            ).item()
        distillation_loss = agg_loss(
            loss_mat=distillation_losses,
            loss_mask=loss_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )

    return distillation_loss, distillation_metrics


@register_distillation_loss(DistillationLossSettings(names=["forward_kl_topk"], use_topk=True))  # type: ignore[arg-type]
def compute_forward_kl_topk(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute forward KL distillation loss and related metrics using top-k log probabilities.

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    # topk loss has been computed in logits processor
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    student_mass = no_padding_2_padding(model_output["student_mass"], data)
    teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert distillation_losses.shape == student_mass.shape == teacher_mass.shape == response_mask_bool.shape

    # Log amount of mass in the top-k log probabilities for both student and teacher.
    student_mass = student_mass[response_mask_bool]
    teacher_mass = teacher_mass[response_mask_bool]
    distillation_metrics = {
        "distillation/student_mass": student_mass.mean().item(),
        "distillation/student_mass_min": Metric(AggregationType.MIN, student_mass.min()),
        "distillation/student_mass_max": Metric(AggregationType.MAX, student_mass.max()),
        "distillation/teacher_mass": teacher_mass.mean().item(),
        "distillation/teacher_mass_min": Metric(AggregationType.MIN, teacher_mass.min()),
        "distillation/teacher_mass_max": Metric(AggregationType.MAX, teacher_mass.max()),
    }

    # Due to use of top-k, student and teacher distributions don't sum to 1 -> divergences can be negative.
    distillation_losses = distillation_losses.clamp_min(0.0)

    return distillation_losses, distillation_metrics


@register_distillation_loss(
    DistillationLossSettings(names=["divergence_gated"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_distillation_loss_divergence_gated(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute divergence-gated unlikelihood distillation loss.

    Loss = beta * KL_topk(student || ref) + alpha * Push
      KL penalty: top-k forward KL divergence (prevents drift from reference model)
      Push: max(0, pi_atk(a) - pi_ref(a)) * (-log(1 - pi_theta(a)))  (divergence-gated unlikelihood)

    The reference model serves as an anchor (prevent drift), NOT a learning target.
    All learning signal comes from the Push term (push away from attack teacher's bad patterns).

    When use_topk is enabled and distillation_losses are pre-computed in logits processor,
    those are used as the KL penalty. Otherwise falls back to k1 estimator.

    Requires:
      - data["teacher_logprobs"]: reference teacher log-probs (pi_ref)
      - data["attack_teacher_logprobs"]: attack teacher log-probs (pi_atk)

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    loss_config: DistillationLossConfig = distillation_config.distillation_loss

    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    ref_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    atk_log_probs = no_padding_2_padding(data["attack_teacher_logprobs"], data).squeeze(-1)

    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()

    assert ref_log_probs.shape == student_log_probs.shape == atk_log_probs.shape == response_mask_bool.shape

    # Clamp log-probs for numerical stability
    if loss_config.log_prob_min_clamp is not None:
        student_log_probs = student_log_probs.clamp(min=loss_config.log_prob_min_clamp)
        ref_log_probs = ref_log_probs.clamp(min=loss_config.log_prob_min_clamp)
        atk_log_probs = atk_log_probs.clamp(min=loss_config.log_prob_min_clamp)

    # Convert to probabilities
    pi_student = student_log_probs.exp()
    pi_ref = ref_log_probs.exp()
    pi_atk = atk_log_probs.exp()

    # KL penalty: use top-k KL from logits processor if available, otherwise estimator
    if "distillation_losses" in model_output:
        # Top-k KL was pre-computed in the logits processor using full student logits
        kl_penalty = no_padding_2_padding(model_output["distillation_losses"], data)
        if kl_penalty.shape != student_log_probs.shape:
            # Shape mismatch due to sequence parallel, use k1 fallback
            kl_penalty = student_log_probs - ref_log_probs
    else:
        kl_mode = loss_config.get("kl_mode", "k1")
        if kl_mode == "forward":
            kl_penalty = pi_ref * (ref_log_probs - student_log_probs)
        else:
            kl_penalty = student_log_probs - ref_log_probs

    # Push: divergence-gated unlikelihood
    # alpha * max(0, pi_atk(a) - pi_ref(a)) * (-log(1 - pi_theta(a)))
    divergence_gate = (pi_atk - pi_ref).clamp(min=0.0)
    # Clamp pi_student away from 1.0 to avoid log(0)
    unlikelihood = -(1.0 - pi_student.clamp(max=1.0 - 1e-7)).log()
    push = loss_config.alpha * divergence_gate * unlikelihood

    distillation_losses = loss_config.kl_coeff * kl_penalty + push

    metrics = {
        "distillation/kl_penalty_mean": kl_penalty[response_mask_bool].mean().item(),
        "distillation/push_mean": push[response_mask_bool].mean().item(),
        "distillation/gate_mean": divergence_gate[response_mask_bool].mean().item(),
        "distillation/gate_active_ratio": (divergence_gate[response_mask_bool] > 0).float().mean().item(),
        "distillation/abs_loss": Metric(AggregationType.MEAN, distillation_losses[response_mask_bool].abs().mean()),
    }
    return distillation_losses, metrics


@register_distillation_loss(
    DistillationLossSettings(names=["divergence_gated_log"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_distillation_loss_divergence_gated_log(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute divergence-gated unlikelihood distillation loss with log-space gate.

    Loss = beta * KL_topk(student || ref) + alpha * Push
      KL penalty: top-k forward KL divergence (prevents drift from reference model)
      Push: max(0, log(pi_atk(a)) - log(pi_ref(a))) * (-log(1 - pi_theta(a)))  (log-space divergence-gated unlikelihood)

    Difference from divergence_gated: uses log-space gate instead of probability-space.
    This is more sensitive to relative differences between teachers.

    The reference model serves as an anchor (prevent drift), NOT a learning target.
    All learning signal comes from the Push term (push away from attack teacher's bad patterns).

    When use_topk is enabled and distillation_losses are pre-computed in logits processor,
    those are used as the KL penalty. Otherwise falls back to k1 estimator.

    Requires:
      - data["teacher_logprobs"]: reference teacher log-probs (pi_ref)
      - data["attack_teacher_logprobs"]: attack teacher log-probs (pi_atk)

    Returns:
      - distillation_losses: (bsz, resp_len)
      - distillation_metrics: Dictionary of metrics.
    """
    loss_config: DistillationLossConfig = distillation_config.distillation_loss

    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    ref_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    atk_log_probs = no_padding_2_padding(data["attack_teacher_logprobs"], data).squeeze(-1)

    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()

    assert ref_log_probs.shape == student_log_probs.shape == atk_log_probs.shape == response_mask_bool.shape

    # Clamp log-probs for numerical stability
    if loss_config.log_prob_min_clamp is not None:
        student_log_probs = student_log_probs.clamp(min=loss_config.log_prob_min_clamp)
        ref_log_probs = ref_log_probs.clamp(min=loss_config.log_prob_min_clamp)
        atk_log_probs = atk_log_probs.clamp(min=loss_config.log_prob_min_clamp)

    # Convert to probabilities for student (needed for unlikelihood)
    pi_student = student_log_probs.exp()

    # KL penalty: use top-k KL from logits processor if available, otherwise estimator
    if "distillation_losses" in model_output:
        # Top-k KL was pre-computed in the logits processor using full student logits
        kl_penalty = no_padding_2_padding(model_output["distillation_losses"], data)
        if kl_penalty.shape != student_log_probs.shape:
            # Shape mismatch due to sequence parallel, use k1 fallback
            kl_penalty = student_log_probs - ref_log_probs
    else:
        kl_mode = loss_config.get("kl_mode", "k1")
        if kl_mode == "forward":
            kl_penalty = pi_ref * (ref_log_probs - student_log_probs)
        else:
            kl_penalty = student_log_probs - ref_log_probs

    # Push: log-space divergence-gated unlikelihood
    # alpha * max(0, log(pi_atk(a)) - log(pi_ref(a))) * (-log(1 - pi_theta(a)))
    # Use log-space gate: more sensitive to relative differences
    divergence_gate = (atk_log_probs - ref_log_probs).clamp(min=0.0)
    # Clamp pi_student away from 1.0 to avoid log(0)
    unlikelihood = -(1.0 - pi_student.clamp(max=1.0 - 1e-7)).log()
    push = loss_config.alpha * divergence_gate * unlikelihood

    distillation_losses = loss_config.kl_coeff * kl_penalty + push

    metrics = {
        "distillation/kl_penalty_mean": kl_penalty[response_mask_bool].mean().item(),
        "distillation/push_mean": push[response_mask_bool].mean().item(),
        "distillation/gate_mean": divergence_gate[response_mask_bool].mean().item(),
        "distillation/gate_active_ratio": (divergence_gate[response_mask_bool] > 0).float().mean().item(),
        "distillation/abs_loss": Metric(AggregationType.MEAN, distillation_losses[response_mask_bool].abs().mean()),
    }
    return distillation_losses, metrics


@register_distillation_loss(
    DistillationLossSettings(names=["divergence_gated_sigmoid"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_distillation_loss_divergence_gated_sigmoid(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute divergence-gated sigmoid-unlikelihood distillation loss.

    Loss = beta * KL_topk(student || ref) + alpha * Push
      KL penalty: top-k forward KL divergence (prevents drift from reference model)
      Push: max(0, pi_atk(a) - pi_ref(a)) * (1 / (2 - pi_theta(a)))  (divergence-gated sigmoid unlikelihood)

    The sigmoid unlikelihood is: sigmoid(-log(1 - pi)) = 1 / (1 + (1 - pi)) = 1 / (2 - pi)
    This provides smoother gradients than -log(1 - pi), especially near pi -> 1.

    The reference model serves as an anchor (prevent drift), NOT a learning target.
    All learning signal comes from the Push term (push away from attack teacher's bad patterns).

    When use_topk is enabled and distillation_losses are pre-computed in logits processor,
    those are used as the KL penalty. Otherwise falls back to k1 estimator.

    Requires:
      - data["teacher_logprobs"]: reference teacher log-probs (pi_ref)
      - data["attack_teacher_logprobs"]: attack teacher log-probs (pi_atk)

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    loss_config: DistillationLossConfig = distillation_config.distillation_loss

    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    ref_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    atk_log_probs = no_padding_2_padding(data["attack_teacher_logprobs"], data).squeeze(-1)

    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()

    assert ref_log_probs.shape == student_log_probs.shape == atk_log_probs.shape == response_mask_bool.shape

    # Clamp log-probs for numerical stability
    if loss_config.log_prob_min_clamp is not None:
        student_log_probs = student_log_probs.clamp(min=loss_config.log_prob_min_clamp)
        ref_log_probs = ref_log_probs.clamp(min=loss_config.log_prob_min_clamp)
        atk_log_probs = atk_log_probs.clamp(min=loss_config.log_prob_min_clamp)

    # Convert to probabilities
    pi_student = student_log_probs.exp()
    pi_ref = ref_log_probs.exp()
    pi_atk = atk_log_probs.exp()

    # KL penalty: use top-k KL from logits processor if available, otherwise estimator
    if "distillation_losses" in model_output:
        # Top-k KL was pre-computed in the logits processor using full student logits
        kl_penalty = no_padding_2_padding(model_output["distillation_losses"], data)
        if kl_penalty.shape != student_log_probs.shape:
            # Shape mismatch due to sequence parallel, use k1 fallback
            kl_penalty = student_log_probs - ref_log_probs
    else:
        kl_mode = loss_config.get("kl_mode", "k1")
        if kl_mode == "forward":
            # Importance-weighted forward KL approximation over the sampled token:
            # π_ref(a) * (log π_ref(a) - log π_c(a))
            # Gradient w.r.t. log π_c: -π_ref(a), which pushes π_c toward π_ref
            # (bidirectional: pushes up when π_c < π_ref, pushes down when π_c > π_ref)
            kl_penalty = pi_ref * (ref_log_probs - student_log_probs)
        else:
            # K1 estimator: log π_c - log π_ref
            kl_penalty = student_log_probs - ref_log_probs

    # Push: divergence-gated sigmoid unlikelihood
    # alpha * max(0, pi_atk(a) - pi_ref(a)) * (1 / (2 - pi_theta(a)))
    # sigmoid(-log(1 - pi)) = 1 / (1 + (1 - pi)) = 1 / (2 - pi)
    divergence_gate = (pi_atk - pi_ref).clamp(min=0.0)
    # Clamp pi_student away from 1.0 to avoid division by zero
    pi_student_clamped = pi_student.clamp(max=1.0 - 1e-7)
    sigmoid_unlikelihood = 1.0 / (2.0 - pi_student_clamped)
    push = loss_config.alpha * divergence_gate * sigmoid_unlikelihood

    distillation_losses = loss_config.kl_coeff * kl_penalty + push

    metrics = {
        "distillation/kl_penalty_mean": kl_penalty[response_mask_bool].mean().item(),
        "distillation/push_mean": push[response_mask_bool].mean().item(),
        "distillation/gate_mean": divergence_gate[response_mask_bool].mean().item(),
        "distillation/gate_active_ratio": (divergence_gate[response_mask_bool] > 0).float().mean().item(),
        "distillation/abs_loss": Metric(AggregationType.MEAN, distillation_losses[response_mask_bool].abs().mean()),
    }
    return distillation_losses, metrics


@register_distillation_loss(
    DistillationLossSettings(names=["jsd"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_jsd_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute Jensen-Shannon Divergence loss (token-level, single-sample estimator).

    JSD(p || q) = 0.5 * KL(p || m) + 0.5 * KL(q || m), m = 0.5*(p + q)
    Bounded in [0, log(2)]. Symmetric and numerically stable.

    Uses single-sample log-probs (student samples action a):
      student_lp = log π_student(a)
      teacher_lp = log π_teacher(a)
      m_lp = log(0.5 * exp(student_lp) + 0.5 * exp(teacher_lp))
      JSD ≈ 0.5 * (student_lp - m_lp) + 0.5 * (teacher_lp - m_lp)
           = 0.5 * (student_lp + teacher_lp - 2 * m_lp)
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape

    # m = 0.5 * (p + q) in log-space via logsumexp
    # log(m) = log(0.5 * exp(lp_s) + 0.5 * exp(lp_t))
    #         = logsumexp(lp_s + log(0.5), lp_t + log(0.5))
    log_half = torch.tensor(-0.6931471805599453, device=student_log_probs.device, dtype=student_log_probs.dtype)
    log_m = torch.logaddexp(student_log_probs + log_half, teacher_log_probs + log_half)

    # JSD = 0.5 * KL(student || m) + 0.5 * KL(teacher || m)
    # Single-sample: KL(p || m) at sampled token ≈ log p(a) - log m(a)
    kl_student_m = student_log_probs - log_m  # always >= 0
    kl_teacher_m = teacher_log_probs - log_m  # always >= 0
    jsd = 0.5 * kl_student_m + 0.5 * kl_teacher_m

    # Clamp for numerical safety (should already be in [0, log2])
    jsd = jsd.clamp(min=0.0, max=0.7)

    metrics = {
        "distillation/jsd_mean": Metric(AggregationType.MEAN, jsd[response_mask_bool].mean()),
        "distillation/kl_student_m": Metric(AggregationType.MEAN, kl_student_m[response_mask_bool].mean()),
        "distillation/kl_teacher_m": Metric(AggregationType.MEAN, kl_teacher_m[response_mask_bool].mean()),
    }
    return jsd, metrics


@register_distillation_loss(
    DistillationLossSettings(names=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_distillation_loss_reverse_kl_estimator(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics using single-sample KL estimators.

    Uses the kl_penalty function from core_algos which supports various KL divergence
    estimators: "kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3".

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape

    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_losses = kl_penalty(
        logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty=loss_config.loss_mode
    )
    # Since k1 can be negative, log the mean absolute loss.
    metrics = {
        "distillation/abs_loss": Metric(AggregationType.MEAN, distillation_losses[response_mask_bool].abs().mean()),
    }
    return distillation_losses, metrics
