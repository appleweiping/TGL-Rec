"""Run observation experiment: test if base Qwen3-8B uses temporal order.

Usage:
    python scripts/run_observation.py \
        --domain beauty \
        --model-path /home/ajifang/models/Qwen/Qwen3-8B \
        --data-dir data/beauty_valid \
        --output-dir outputs/observation/beauty/ \
        --limit 20 \
        --variants base,shuffled,reversed,recent_only

This is Block 1 of the experiment plan. If shuffling/reversing history
does NOT significantly hurt performance, it proves the LLM ignores
temporal order — validating our research premise.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm4rec.io.artifacts import ensure_dir, read_jsonl, write_json, write_jsonl


OBSERVATION_PROMPT_TEMPLATE = """You are a recommendation system. Given a user's interaction history, rank the candidate items from most to least likely to be the user's next interaction.

{history_section}

Candidates:
{candidates_section}

Rank all candidates from most likely (1) to least likely. Output ONLY the ranking as a numbered list."""


def format_history(items: list[dict], variant: str) -> str:
    if variant == "base":
        pass  # keep original order
    elif variant == "shuffled":
        items = items.copy()
        random.shuffle(items)
    elif variant == "reversed":
        items = list(reversed(items))
    elif variant == "recent_only":
        items = items[-3:]
    elif variant == "no_time":
        items = [{"title": it.get("title", it.get("item_id")), "item_id": it.get("item_id")} for it in items]
        return "User's interaction history:\n" + "\n".join(
            f"{i+1}. {it['title']}" for i, it in enumerate(items)
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    lines = []
    for i, it in enumerate(items):
        title = it.get("title", it.get("item_id", f"item_{i}"))
        ts_info = ""
        if "timestamp_text" in it:
            ts_info = f" (on {it['timestamp_text']})"
        elif "relative_time" in it:
            ts_info = f" ({it['relative_time']})"
        lines.append(f"{i+1}. {title}{ts_info}")

    return "User's interaction history (chronological order):\n" + "\n".join(lines)


def run_observation_variant(
    *,
    variant: str,
    examples: list[dict],
    model_path: str,
    output_dir: Path,
    limit: int | None = None,
    seed: int = 42,
    quantize_4bit: bool = False,
) -> dict:
    """Run one observation variant and save results."""

    random.seed(seed)
    variant_dir = output_dir / variant
    ensure_dir(variant_dir)

    if limit:
        examples = examples[:limit]

    print(f"[obs-{variant}] Running {len(examples)} examples...")
    t0 = time.time()

    # Lazy import to avoid loading model until needed
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        print("[obs] transformers/torch not available. Generating prompts only.")
        prompts = []
        for ex in examples:
            history_items = ex.get("history_items", [])
            candidates = ex.get("candidate_items", [])
            history_text = format_history(history_items, variant)
            cand_text = "\n".join(f"{i+1}. {c.get('title', c.get('item_id', ''))}" for i, c in enumerate(candidates))
            prompt = OBSERVATION_PROMPT_TEMPLATE.format(
                history_section=history_text,
                candidates_section=cand_text,
            )
            prompts.append({"user_id": ex.get("user_id"), "prompt": prompt})
        write_jsonl(variant_dir / "prompts.jsonl", prompts)
        write_json(variant_dir / "status.json", {"status": "prompts_only", "n": len(prompts)})
        return {"status": "prompts_only", "n": len(prompts)}

    print(f"[obs-{variant}] Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    load_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if quantize_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()

    predictions = []
    for idx, ex in enumerate(examples):
        if idx % 10 == 0:
            print(f"[obs-{variant}] {idx}/{len(examples)}")

        history_items = ex.get("history_items", [])
        candidates = ex.get("candidate_items", [])
        history_text = format_history(history_items, variant)
        cand_text = "\n".join(
            f"{i+1}. {c.get('title', c.get('item_id', ''))}"
            for i, c in enumerate(candidates)
        )
        prompt = OBSERVATION_PROMPT_TEMPLATE.format(
            history_section=history_text,
            candidates_section=cand_text,
        )

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        predictions.append({
            "user_id": ex.get("user_id"),
            "variant": variant,
            "ground_truth": ex.get("target_item"),
            "raw_output": response,
            "candidate_items": [c.get("item_id", "") for c in candidates],
        })

    elapsed = time.time() - t0
    write_jsonl(variant_dir / "predictions.jsonl", predictions)

    metrics = _compute_observation_metrics(predictions)
    metrics["variant"] = variant
    metrics["n_examples"] = len(predictions)
    metrics["elapsed_seconds"] = elapsed
    write_json(variant_dir / "metrics.json", metrics)

    print(f"[obs-{variant}] Done in {elapsed:.1f}s. MRR={metrics.get('MRR', 'N/A')}")
    return metrics


def _compute_observation_metrics(predictions: list[dict]) -> dict:
    """Compute ranking metrics from raw LLM outputs."""
    # Simplified metric computation - parse ranking from output
    mrr_sum = 0.0
    hr5 = 0
    hr10 = 0
    valid = 0

    for pred in predictions:
        gt = pred.get("ground_truth", "")
        candidates = pred.get("candidate_items", [])
        output = pred.get("raw_output", "")

        ranked = _parse_ranking(output, candidates)
        if not ranked:
            continue
        valid += 1

        if gt in ranked:
            rank = ranked.index(gt) + 1
            mrr_sum += 1.0 / rank
            if rank <= 5:
                hr5 += 1
            if rank <= 10:
                hr10 += 1

    n = max(valid, 1)
    return {
        "MRR": mrr_sum / n,
        "HR@5": hr5 / n,
        "HR@10": hr10 / n,
        "parse_success_rate": valid / max(len(predictions), 1),
        "valid_predictions": valid,
    }


def _parse_ranking(output: str, candidates: list[str]) -> list[str]:
    """Best-effort parse of LLM ranking output."""
    lines = output.strip().split("\n")
    ranked = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Try to match "1. item_title" or just numbered items
        for cand in candidates:
            if cand in line and cand not in ranked:
                ranked.append(cand)
                break
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="Run observation experiment")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--variants", default="base,shuffled,reversed,recent_only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantize-4bit", action="store_true", help="Use 4-bit quantization (fits in ~6GB)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    data_dir = Path(args.data_dir)
    ranking_file = data_dir / "ranking_valid.jsonl"
    if not ranking_file.exists():
        print(f"[obs] ERROR: {ranking_file} not found")
        sys.exit(1)

    examples = list(read_jsonl(ranking_file))
    print(f"[obs] Loaded {len(examples)} examples from {ranking_file}")

    variants = [v.strip() for v in args.variants.split(",")]
    all_metrics = {}

    for variant in variants:
        metrics = run_observation_variant(
            variant=variant,
            examples=examples,
            model_path=args.model_path,
            output_dir=output_dir,
            limit=args.limit,
            seed=args.seed,
            quantize_4bit=args.quantize_4bit,
        )
        all_metrics[variant] = metrics

    write_json(output_dir / "observation_summary.json", all_metrics)

    print("\n=== Observation Summary ===")
    print(f"{'Variant':<15} {'MRR':<8} {'HR@5':<8} {'HR@10':<8}")
    print("-" * 40)
    for variant, m in all_metrics.items():
        print(f"{variant:<15} {m.get('MRR', 0):<8.4f} {m.get('HR@5', 0):<8.4f} {m.get('HR@10', 0):<8.4f}")

    # Key diagnostic: is shuffled ≈ base?
    if "base" in all_metrics and "shuffled" in all_metrics:
        base_mrr = all_metrics["base"].get("MRR", 0)
        shuf_mrr = all_metrics["shuffled"].get("MRR", 0)
        diff = abs(base_mrr - shuf_mrr)
        if diff < 0.02:
            print(f"\n⚠ OBSERVATION CONFIRMED: Shuffling barely affects MRR (diff={diff:.4f})")
            print("  → LLM ignores temporal order. Our temporal evidence approach is justified.")
        else:
            print(f"\n⚠ OBSERVATION WEAK: Shuffling affects MRR by {diff:.4f}")
            print("  → LLM may already use some temporal signal. Investigate further.")


if __name__ == "__main__":
    main()
