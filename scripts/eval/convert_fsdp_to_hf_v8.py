"""
Convert FSDP checkpoint (world_size=8) to HuggingFace format for vLLM inference.
Auto-detects world size from shard filenames.
"""

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_fsdp_shards(fsdp_checkpoint_path, output_path, base_model):
    print(f"Merging {fsdp_checkpoint_path}...")

    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    fsdp_path = Path(fsdp_checkpoint_path) / "actor"
    rank_files = sorted(fsdp_path.glob("model_world_size_*_rank_*.pt"))

    if not rank_files:
        raise FileNotFoundError(f"No model shard files found in {fsdp_path}")

    world_size = len(rank_files)
    print(f"Found {world_size} shard files")

    # Sort by rank number
    rank_files = sorted(rank_files, key=lambda f: int(f.stem.split("rank_")[-1]))

    all_shards = []
    for rank_file in rank_files:
        print(f"Loading {rank_file.name}...")
        checkpoint = torch.load(rank_file, map_location="cpu", weights_only=False)

        state_dict = checkpoint
        for key in ["model", "state_dict", "module"]:
            if key in state_dict:
                state_dict = state_dict[key]
                break

        all_shards.append(state_dict)

    print(f"Loaded {len(all_shards)} shards")

    merged_state_dict = {}
    model_state_dict = model.state_dict()

    for key in model_state_dict.keys():
        if key in all_shards[0]:
            raw_shards = [shard[key] for shard in all_shards]

            # DTensors carry placement metadata - use it
            first = raw_shards[0]
            if type(first).__name__ == 'DTensor':
                placements = first.placements
                shard_dim = placements[0].dim if hasattr(placements[0], 'dim') else 0
                shards = [s.to_local().cpu() for s in raw_shards]
                merged = torch.cat(shards, dim=shard_dim)
            else:
                shards = [s.cpu() if hasattr(s, 'cpu') else s for s in raw_shards]
                expected_shape = model_state_dict[key].shape
                if shards[0].shape[0] * world_size == expected_shape[0]:
                    merged = torch.cat(shards, dim=0)
                else:
                    merged = shards[0]

            expected_shape = model_state_dict[key].shape
            if merged.shape != expected_shape:
                print(f"Warning: Shape mismatch for {key}: got {merged.shape}, expected {expected_shape}")
                if shards[0].shape == expected_shape:
                    merged = shards[0]
                else:
                    print(f"  Skipping {key}")
                    continue

            merged_state_dict[key] = merged
        else:
            print(f"Warning: Key {key} not found in checkpoint")

    print(f"Merged state dict has {len(merged_state_dict)} keys")

    print("Loading state dict into model...")
    missing, unexpected = model.load_state_dict(merged_state_dict, strict=False)
    if missing:
        print(f"Missing {len(missing)} keys: {missing[:5]}...")
    if unexpected:
        print(f"Unexpected {len(unexpected)} keys")

    print(f"Saving to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, max_shard_size="2GB")
    tokenizer.save_pretrained(output_path)
    print(f"Conversion complete! Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsdp-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-1.7B")
    args = parser.parse_args()

    merge_fsdp_shards(args.fsdp_path, args.output_dir, args.base_model)
