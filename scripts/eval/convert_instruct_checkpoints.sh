#!/bin/bash
# Convert instruct NSD checkpoints to HuggingFace format

set -xe

source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl-distill

cd ${PROJECT_ROOT}/scripts/eval

# Output directory for HF checkpoints
OUTPUT_BASE="${CHECKPOINT_ROOT}/hf_models/qwen3_4b_instruct_nsd_full_math_every_prompt_sigmoid_hf"
mkdir -p "$OUTPUT_BASE"

# Base model
BASE_MODEL="${CHECKPOINT_ROOT}/hf_models/Qwen3-4B-Instruct-2507"

# FSDP checkpoints to convert
FSDP_BASE="${CHECKPOINT_ROOT}/meng_verl_nsd_math/qwen3_4b_instruct_nsd_full_math_every_prompt_sigmoid"

STEPS=(40 80 120 160 200)

echo ""
echo "========================================"
echo "Converting Instruct Checkpoints to HF"
echo "========================================"
echo ""

for step in "${STEPS[@]}"; do
    fsdp_path="$FSDP_BASE/global_step_$step"
    output_path="$OUTPUT_BASE/global_step_$step"

    echo "=== Converting step $step ==="

    if [ -d "$output_path" ] && [ -f "$output_path/config.json" ]; then
        echo "Already converted, skipping..."
        continue
    fi

    mkdir -p "$output_path"

    python convert_fsdp_to_hf_v4.py \
        --fsdp-path "$fsdp_path" \
        --output-dir "$output_path" \
        --base-model "$BASE_MODEL"

    echo "=== Step $step Done ==="
    echo ""
done

echo ""
echo "========================================"
echo "Conversion Complete!"
echo "========================================"
echo ""
echo "HF checkpoints saved to: $OUTPUT_BASE"
