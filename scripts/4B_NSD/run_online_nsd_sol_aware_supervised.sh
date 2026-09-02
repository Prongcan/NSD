#!/usr/bin/env bash
# Online Negative Self-Distillation (Online NSD) | Supervised Distillation
# Each training step: student generates rollout → teacher generates sol-aware attack on-the-fly
# → NSD loss directly backpropagated (no policy gradient / advantage function)
#
# KEY DIFFERENCE from run_online_nsd_sol_aware.sh:
#   use_policy_gradient=False  → loss directly backpropped (supervised distillation)
#   kl_mode=forward            → importance-weighted forward KL: π_ref*(log π_ref - log π_c)
#                                 bidirectional restoring force vs K1 (prevents entropy collapse)
#
# GPU layout (6 GPUs total):
#   GPU 0-3: actor + rollout (tp=2 rollout, FSDP actor)
#   GPU 4-5: teacher (tp=2, frozen Qwen3-4B)

set -xeuo pipefail

PYTHON=${PYTHON:-python3}

export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

export WANDB_API_KEY="${WANDB_API_KEY}"  # set via: export WANDB_API_KEY=your_key
export WANDB_ENTITY="personl"
export WANDB_MODE=online

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_PATH="${CHECKPOINT_ROOT}/hf_models/Qwen3-4B"

NNODES=1
NGPUS_PER_NODE=4
TEACHER_WORLD_SIZE=2

distillation_loss_mode=divergence_gated_sigmoid
use_policy_gradient=False
distillation_topk=32
alpha=1.0

train_batch_size=32
ppo_mini_batch_size=32
max_prompt_length=512
max_response_length=4096
ppo_max_token_len_per_gpu=6144

actor_lr=1e-6
lr_warmup_steps_ratio=0.1

rollout_tp=2
rollout_gpu_mem_util=0.25
teacher_tp=2
# Teacher needs extra capacity for generation (attack prompt) + logprob computation
teacher_gpu_mem_util=0.8
teacher_max_model_len=8192

total_epochs=2
save_freq=50
test_freq=20

project_name=meng_verl_nsd_math
experiment_name=qwen3_4b_online_nsd_sol_aware_supervised
default_local_dir=${CHECKPOINT_ROOT}/${project_name}/${experiment_name}

DATA_DIR=${PROJECT_ROOT}/data
train_data=${DATA_DIR}/math_online_nsd/train.parquet
test_data=${DATA_DIR}/math_online_nsd/test_200.parquet

train_files="['$train_data']"
val_files="['$test_data']"

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
    +data.apply_chat_template_kwargs.enable_thinking=False
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${lr_warmup_steps_ratio}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.default_local_dir=${default_local_dir}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=True
    trainer.resume_mode=${resume_mode:-auto}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

EXTRA=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    distillation.nnodes=${NNODES}
    distillation.teacher_models.teacher_model.model_path="$MODEL_PATH"
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp}
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    distillation.teacher_models.teacher_model.inference.max_model_len=${teacher_max_model_len}
    distillation.distillation_loss.loss_mode=${distillation_loss_mode}
    distillation.distillation_loss.topk=${distillation_topk}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}
    +distillation.distillation_loss.alpha=${alpha}
    +distillation.distillation_loss.kl_coeff=0.01
    +distillation.distillation_loss.kl_mode=forward
    +distillation.distillation_loss.supervised_clip_ratio=0.2
    distillation.distillation_loss.loss_max_clamp=0.5
    distillation.distillation_loss.log_prob_min_clamp=-10.0
    # Enable online NSD: teacher generates sol-aware attack from student rollout at each step
    +distillation.online_nsd=True
)

LOG_DIR="${SCRIPT_DIR}/logs_online_nsd_sol_aware_supervised"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/nsd_train_$(date +%Y%m%d_%H%M%S).log"

echo "=== Online NSD Supervised (sol-aware, forward KL, no policy gradient) ==="
echo "=== Logging to: ${LOG_FILE} ==="
echo "=== To monitor: tail -f ${LOG_FILE} ==="

$PYTHON -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@" 2>&1 | tee "${LOG_FILE}"
