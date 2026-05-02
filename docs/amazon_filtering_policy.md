# Amazon Filtering Policy

Phase 8D preserves the converted full Amazon Reviews 2023 multidomain data as the immutable
conversion artifact:

```text
data/raw/amazon_multidomain/interactions.jsonl
data/raw/amazon_multidomain/items.jsonl
```

Paper-ready Amazon data is derived under `data/processed/amazon_multidomain_filtered/`. The
filtered files are not raw data, and configs must not describe them as unfiltered full conversion
outputs.

## Policy

The initial paper-ready Amazon dataset uses iterative k-core filtering with `k=3` for both users
and items. The filter is applied before split construction, candidate generation, model training,
or evaluation. Each retained user has enough timestamped interactions to support leave-one-out
splitting, and retained items must keep usable text fields for retrieval and prompt construction.

The `k=3` setting is the initial policy because Phase 8C readiness showed many users with fewer
than three interactions. A three-interaction minimum supports train/history, validation, and test
holdout construction while retaining more multidomain coverage than a stricter threshold. The
`k=5` configs are kept for sensitivity and robustness checks, but they are not the default launch
dataset unless the protocol is explicitly changed.

Filtering preserves domain labels and records per-domain retention. Reports include input/output
users, items, interactions, timestamp ranges, retained ratios, threshold violations, and a raw-file
snapshot confirming that the converted raw files were not modified.

Pilot and paper results must report the filtering strategy, k values, retained ratios, per-domain
counts, and readiness status. Filtering statistics are descriptive provenance; they must not be
presented as evidence that model quality improved.

## Commands

```bash
python scripts/filter_amazon_multidomain.py --config configs/datasets/amazon_multidomain_filtered_k3.yaml --dry-run
python scripts/filter_amazon_multidomain.py --config configs/datasets/amazon_multidomain_filtered_iterative_k3.yaml --dry-run
python scripts/filter_amazon_multidomain.py --config configs/datasets/amazon_multidomain_filtered_iterative_k3.yaml --materialize
python scripts/check_dataset_readiness.py --config configs/datasets/amazon_multidomain_filtered_iterative_k3.yaml
```

The canonical Phase 8D report is:

```text
data/processed/amazon_multidomain_filtered/filtering_report.json
```
