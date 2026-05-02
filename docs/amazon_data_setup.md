# Amazon Reviews 2023 Multidomain Setup

Phase 8B adapts local Amazon Reviews 2023 domain dumps into the unified project schema. It does not
run training, evaluation, API calls, LoRA, or paper-scale experiments.

Expected raw directories:

```text
data/raw/amazon_reviews_2023_beauty
data/raw/amazon_reviews_2023_digital_music
data/raw/amazon_reviews_2023_handmade
data/raw/amazon_reviews_2023_health
data/raw/amazon_reviews_2023_video_games
```

Domain mapping:

- `amazon_reviews_2023_beauty` -> `Beauty`
- `amazon_reviews_2023_digital_music` -> `Digital_Music`
- `amazon_reviews_2023_handmade` -> `Handmade`
- `amazon_reviews_2023_health` -> `Health`
- `amazon_reviews_2023_video_games` -> `Video_Games`

Inspect schemas first:

```bash
python scripts/check_amazon_schema.py --config configs/datasets/amazon_reviews_2023.yaml
```

Create a sampled conversion for readiness/smoke checks:

```bash
python scripts/prepare_amazon_multidomain.py --config configs/datasets/amazon_multidomain_sampled.yaml --dry-run
python scripts/prepare_amazon_multidomain.py --config configs/datasets/amazon_multidomain_sampled.yaml --materialize
```

The sampled output is written to:

```text
data/processed/amazon_multidomain_sampled/interactions.jsonl
data/processed/amazon_multidomain_sampled/items.jsonl
data/processed/amazon_multidomain_sampled/conversion_report.json
```

The full paper-scale converted paths are:

```text
data/raw/amazon_multidomain/interactions.jsonl
data/raw/amazon_multidomain/items.jsonl
```

Do not materialize the full conversion automatically if it may be large. Ask for confirmation first.
The converter drops reviews with missing user/item/timestamp, deduplicates by
`user_id,item_id,timestamp,domain`, deduplicates items by `item_id,domain`, preserves domain labels,
and reports missing metadata/text coverage in `conversion_report.json`.

Supported source formats are `.jsonl`, `.jsonl.gz`, `.csv`, and `.parquet`. Parquet requires
`pyarrow`; without it the schema checker/converter reports a clear dependency issue.
