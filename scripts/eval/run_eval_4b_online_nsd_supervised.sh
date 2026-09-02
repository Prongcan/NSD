#!/bin/bash
# Evaluate Online NSD Supervised (sol-aware, forward KL) checkpoints
# Steps: 100, 200, 250, 300, 400
# Convert FSDP → HF (serial), then eval in parallel on GPU 0-5 (TP=2, 3 jobs at a time)
# Datasets: AIME 2024, AIME 2025, HMMT Feb 2025

set -euo pipefail

PYTHON=${PYTHON:-python3}
CONVERT=${PROJECT_ROOT}/scripts/eval/convert_fsdp_to_hf_v8.py
EVAL=${PROJECT_ROOT}/scripts/eval/eval_aime_pass8.py

export LIBRARY_PATH="${CONDA_PREFIX:-}/lib/stubs:${CONDA_PREFIX:-}/lib64/stubs:/usr/local/cuda/lib64/stubs:/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CUDA_HOME=~/miniconda3
export PATH=$CUDA_HOME/bin:$PATH

BASE_DIR="${PROJECT_ROOT}"
FSDP_BASE="${CHECKPOINT_ROOT}/meng_verl_nsd_math/qwen3_4b_online_nsd_sol_aware_supervised"
HF_BASE="${CHECKPOINT_ROOT}/hf_models/4b_online_nsd_supervised_eval"
BASE_MODEL="${CHECKPOINT_ROOT}/hf_models/Qwen3-4B"

AIME_DATA="$BASE_DIR/data/aime/all.parquet"
HMMT_DATA="$BASE_DIR/data/hmmt/hmmt_feb_2025.parquet"

AIME2024_OUT="$BASE_DIR/eval_results/4b_online_nsd_supervised_aime2024_pass8"
AIME2025_OUT="$BASE_DIR/eval_results/4b_online_nsd_supervised_aime2025_pass8"
HMMT2025_OUT="$BASE_DIR/eval_results/4b_online_nsd_supervised_hmmt2025_pass8"

CONVERT_LOGS="$BASE_DIR/eval_results/4b_online_nsd_supervised_aime2024_pass8/logs/convert"
mkdir -p "$AIME2024_OUT/logs" "$AIME2025_OUT/logs" "$HMMT2025_OUT/logs" "$CONVERT_LOGS" "$HF_BASE"

STEPS=(400)

# ── Step 1: Convert FSDP → HF (serial) ───────────────────────────────────────
echo "========================================"
echo "Converting FSDP checkpoints to HF format"
echo "========================================"

for step in "${STEPS[@]}"; do
    HF_OUT="$HF_BASE/hf_step${step}"
    if [ -d "$HF_OUT" ] && [ -f "$HF_OUT/config.json" ]; then
        echo "[$(date +%H:%M:%S)] hf_step${step} already exists, skipping"
        continue
    fi
    echo "[$(date +%H:%M:%S)] Converting step${step}..."
    $PYTHON "$CONVERT" \
        --fsdp-path "$FSDP_BASE/global_step_${step}" \
        --output-dir "$HF_OUT" \
        --base-model "$BASE_MODEL" \
        > "$CONVERT_LOGS/convert_step${step}.log" 2>&1
    echo "[$(date +%H:%M:%S)] step${step} done"
done

echo "All conversions done."

# ── Helper: run eval on 2 GPUs ───────────────────────────────────────────────
eval_pass8() {
    local gpus=$1
    local model_path=$2
    local label=$3
    local data_path=$4
    local year=$5
    local output_dir=$6

    local mname
    mname=$(basename "$model_path")
    local summary="$output_dir/eval_aime_y${year}_${mname}_pass8_summary.json"

    if [ -f "$summary" ]; then
        echo "[$(date +%H:%M:%S)] $label: already done, skipping"
        return
    fi

    local log="$output_dir/logs/${label}.log"
    echo "[$(date +%H:%M:%S)] Starting $label (year=$year, GPUs $gpus)..."
    CUDA_VISIBLE_DEVICES=$gpus $PYTHON "$EVAL" \
        --model-path "$model_path" \
        --data-path "$data_path" \
        --year "$year" \
        --output-dir "$output_dir" \
        --num-samples 8 \
        --temperature 0.6 \
        --top-p 0.95 \
        --top-k 20 \
        --min-p 0.0 \
        --max-tokens 38912 \
        --max-model-len 40960 \
        --tensor-parallel-size 2 \
        --gpu-memory-util 0.85 \
        --use-chat-template \
        --disable-thinking \
        > "$log" 2>&1
    echo "[$(date +%H:%M:%S)] Done: $label"
}

# ── Step 2: AIME 2024 ────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "AIME 2024"
echo "========================================"

eval_pass8 0,1 "$HF_BASE/hf_step400" "step400_aime2024" "$AIME_DATA" 2024 "$AIME2024_OUT"

# ── Step 3: AIME 2025 ────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "AIME 2025"
echo "========================================"

eval_pass8 0,1 "$HF_BASE/hf_step400" "step400_aime2025" "$AIME_DATA" 2025 "$AIME2025_OUT"

# ── Step 4: HMMT Feb 2025 ────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "HMMT Feb 2025"
echo "========================================"

eval_pass8 0,1 "$HF_BASE/hf_step400" "step400_hmmt2025" "$HMMT_DATA" 2025 "$HMMT2025_OUT"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Summary"
echo "========================================"

$PYTHON << 'PYEOF'
import json, pandas as pd, os

BASE = "${PROJECT_ROOT}"
DIRS = {
    "aime2024": f"{BASE}/eval_results/4b_online_nsd_supervised_aime2024_pass8",
    "aime2025": f"{BASE}/eval_results/4b_online_nsd_supervised_aime2025_pass8",
    "hmmt":     f"{BASE}/eval_results/4b_online_nsd_supervised_hmmt2025_pass8",
}

def avg8(out_dir, year, mname):
    sf = f"{out_dir}/eval_aime_y{year}_{mname}_pass8_summary.json"
    pf = f"{out_dir}/eval_aime_y{year}_{mname}_pass8.parquet"
    if not os.path.exists(sf):
        return None, None
    d = json.load(open(sf))
    df = pd.read_parquet(pf)
    avg = df['num_correct_samples'].sum() / (len(df) * 8) * 100
    return d['pass_at_k'], avg

for title, key, year in [("AIME 2024", "aime2024", 2024), ("AIME 2025", "aime2025", 2025), ("HMMT Feb 2025", "hmmt", 2025)]:
    print(f"\n=== {title} ===")
    for step in [400]:
        p8, a8 = avg8(DIRS[key], year, f"hf_step{step}")
        if p8 is not None:
            print(f"  step{step:3d}: pass@8={p8:.4f}  avg@8={a8:.2f}%")
        else:
            print(f"  step{step:3d}: MISSING")
PYEOF

echo ""
echo "All done!"
