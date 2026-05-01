"""Command line entry points for TGLRec."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tglrec.data.amazon import preprocess_amazon_reviews_2023
from tglrec.data.movielens import preprocess_movielens_1m
from tglrec.eval.history_perturbations import (
    DEFAULT_HISTORY_PERTURBATIONS,
    run_history_perturbation_diagnostics,
)
from tglrec.eval.semantic_transition_stress import run_semantic_transition_stress
from tglrec.eval.tdig_recall import DEFAULT_TDIG_RECALL_SCORE_FIELD, run_tdig_candidate_recall
from tglrec.graph.tdig import build_tdig_artifact
from tglrec.models.sanity_baselines import DEFAULT_KS, run_sanity_baselines
from tglrec.utils.config import load_config
from tglrec.utils.seeds import set_global_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tglrec")
    parser.add_argument("--version", action="store_true", help="print package version")
    subparsers = parser.add_subparsers(dest="command")

    check_config = subparsers.add_parser("check-config", help="load and validate a YAML config")
    check_config.add_argument("path", type=Path, help="path to YAML config")

    preprocess = subparsers.add_parser("preprocess", help="dataset preprocessing commands")
    preprocess_sub = preprocess.add_subparsers(dest="dataset", required=True)
    ml1m = preprocess_sub.add_parser(
        "movielens-1m", help="preprocess MovieLens-1M ratings into temporal splits"
    )
    ml1m.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="directory containing ratings.dat and movies.dat, or a parent containing ml-1m/",
    )
    ml1m.add_argument("--zip-path", type=Path, default=None, help="path to ml-1m.zip")
    ml1m.add_argument(
        "--download",
        action="store_true",
        help="download ml-1m.zip from the official GroupLens URL if raw files are absent",
    )
    ml1m.add_argument(
        "--download-dir",
        type=Path,
        default=Path("data/raw/movielens_1m"),
        help="where downloaded raw files are stored",
    )
    ml1m.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/datasets/movielens_1m"),
        help="processed dataset output directory",
    )
    ml1m.add_argument("--min-user-interactions", type=int, default=5)
    ml1m.add_argument("--min-item-interactions", type=int, default=5)
    ml1m.add_argument("--global-train-ratio", type=float, default=0.8)
    ml1m.add_argument("--global-val-ratio", type=float, default=0.1)
    ml1m.add_argument("--seed", type=int, default=2026)

    amazon = preprocess_sub.add_parser(
        "amazon-reviews-2023",
        help="preprocess a local Amazon Reviews 2023 category file into temporal splits",
    )
    amazon.add_argument(
        "--reviews-path",
        type=Path,
        required=True,
        help="local review file (.jsonl, .jsonl.gz, .json.gz, .ndjson, or .csv)",
    )
    amazon.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="optional local item metadata file with parent_asin records",
    )
    amazon.add_argument("--category", default=None, help="category label recorded in metadata")
    amazon.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/datasets/amazon_reviews_2023"),
        help="processed dataset output directory",
    )
    amazon.add_argument("--user-col", default="user_id")
    amazon.add_argument("--item-col", default="parent_asin")
    amazon.add_argument("--item-fallback-col", default="asin")
    amazon.add_argument("--timestamp-col", default=None)
    amazon.add_argument("--rating-col", default="rating")
    amazon.add_argument("--metadata-item-col", default="parent_asin")
    amazon.add_argument("--min-rating", type=float, default=None)
    amazon.add_argument("--min-user-interactions", type=int, default=5)
    amazon.add_argument("--min-item-interactions", type=int, default=5)
    amazon.add_argument("--global-train-ratio", type=float, default=0.8)
    amazon.add_argument("--global-val-ratio", type=float, default=0.1)
    amazon.add_argument(
        "--keep-duplicate-user-items",
        action="store_true",
        help="keep repeated user-item events; default collapses to the first observed event",
    )
    amazon.add_argument(
        "--allow-same-timestamp-user-events",
        action="store_true",
        help="allow ambiguous same-user same-timestamp ordering for exploratory preprocessing",
    )
    amazon.add_argument(
        "--source-file-url",
        default=None,
        help="optional exact source URL for the local review file, recorded in metadata",
    )
    amazon.add_argument(
        "--metadata-source-url",
        default=None,
        help="optional exact source URL for the local metadata file, recorded in metadata",
    )
    amazon.add_argument(
        "--hf-revision",
        default=None,
        help="optional Hugging Face dataset revision or commit, recorded in metadata",
    )
    amazon.add_argument("--seed", type=int, default=2026)

    graph = subparsers.add_parser("graph", help="temporal graph construction commands")
    graph_sub = graph.add_subparsers(dest="graph_command", required=True)
    build_tdig = graph_sub.add_parser(
        "build-tdig",
        help="build a train-only temporal directed item graph from processed interactions",
    )
    build_tdig.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("artifacts/datasets/movielens_1m"),
        help="processed dataset directory containing interactions.csv",
    )
    build_tdig.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/graphs/movielens_1m_tdig"),
        help="TDIG artifact output directory",
    )
    build_tdig.add_argument(
        "--split-name",
        choices=["temporal_leave_one_out", "global_time"],
        default="temporal_leave_one_out",
    )
    build_tdig.add_argument(
        "--train-split-label",
        default="train",
        help="split label used as graph evidence; default is train",
    )
    build_tdig.add_argument(
        "--strict-before-timestamp",
        type=int,
        default=None,
        help=(
            "optional as-of cutoff; exclude train events at or after this prediction timestamp"
        ),
    )
    build_tdig.add_argument(
        "--include-same-timestamp-transitions",
        action="store_true",
        help=(
            "include directed transitions between same-user events with identical timestamps; "
            "disabled by default because their temporal order may be arbitrary"
        ),
    )

    evaluate = subparsers.add_parser("evaluate", help="evaluation commands")
    evaluate_sub = evaluate.add_subparsers(dest="evaluator", required=True)
    sanity = evaluate_sub.add_parser(
        "sanity-baselines", help="evaluate local popularity and item-kNN baselines"
    )
    sanity.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("artifacts/datasets/movielens_1m"),
        help="processed dataset directory containing interactions.csv and items.csv",
    )
    sanity.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="run output directory; defaults to runs/<timestamp>-sanity-baselines",
    )
    sanity.add_argument(
        "--split-name",
        choices=["temporal_leave_one_out", "global_time"],
        default="temporal_leave_one_out",
    )
    sanity.add_argument("--eval-split", choices=["val", "test"], default="test")
    sanity.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    sanity.add_argument("--item-knn-neighbors", type=int, default=50)
    sanity.add_argument(
        "--item-knn-max-history-items",
        type=int,
        default=100,
        help="number of recent unique train-history items used for item-kNN scoring; 0 uses all",
    )
    sanity.add_argument(
        "--cooccurrence-history-window",
        type=int,
        default=200,
        help="recent unique items per user used when updating co-occurrence; 0 uses all",
    )
    sanity.add_argument(
        "--include-seen",
        action="store_true",
        help="rank previously seen training items instead of filtering them",
    )
    sanity.add_argument(
        "--no-validation-history",
        action="store_true",
        help="for test evaluation, do not add each user's validation event as prior history",
    )
    sanity.add_argument("--seed", type=int, default=2026)

    history_diag = evaluate_sub.add_parser(
        "history-perturbations",
        help="evaluate original and perturbed histories for sanity baselines",
    )
    history_diag.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("artifacts/datasets/movielens_1m"),
        help="processed dataset directory containing interactions.csv and items.csv",
    )
    history_diag.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="run output directory; defaults to runs/<timestamp>-history-perturbations",
    )
    history_diag.add_argument(
        "--split-name",
        choices=["temporal_leave_one_out", "global_time"],
        default="temporal_leave_one_out",
    )
    history_diag.add_argument("--eval-split", choices=["val", "test"], default="test")
    history_diag.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    history_diag.add_argument(
        "--perturbations",
        choices=list(DEFAULT_HISTORY_PERTURBATIONS),
        nargs="+",
        default=list(DEFAULT_HISTORY_PERTURBATIONS),
        help="history variants to evaluate; original is always included",
    )
    history_diag.add_argument("--item-knn-neighbors", type=int, default=50)
    history_diag.add_argument(
        "--item-knn-max-history-items",
        type=int,
        default=100,
        help="number of recent unique train-history items used for item-kNN scoring; 0 uses all",
    )
    history_diag.add_argument(
        "--cooccurrence-history-window",
        type=int,
        default=200,
        help="recent unique items per user used when updating co-occurrence; 0 uses all",
    )
    history_diag.add_argument(
        "--include-seen",
        action="store_true",
        help="rank previously seen training items instead of filtering them",
    )
    history_diag.add_argument(
        "--no-validation-history",
        action="store_true",
        help="for test evaluation, do not add each user's validation event as prior history",
    )
    history_diag.add_argument("--seed", type=int, default=2026)

    tdig_recall = evaluate_sub.add_parser(
        "tdig-candidate-recall",
        aliases=["tdig-recall"],
        help="evaluate direct-transition TDIG candidate recall with strict as-of train evidence",
    )
    tdig_recall.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("artifacts/datasets/movielens_1m"),
        help="processed dataset directory containing interactions.csv and items.csv",
    )
    tdig_recall.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="run output directory; defaults to runs/<timestamp>-tdig-candidate-recall",
    )
    tdig_recall.add_argument(
        "--split-name",
        choices=["temporal_leave_one_out", "global_time"],
        default="temporal_leave_one_out",
    )
    tdig_recall.add_argument("--eval-split", choices=["val", "test"], default="test")
    tdig_recall.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    tdig_recall.add_argument(
        "--max-history-items",
        type=int,
        default=20,
        help="number of recent user history events used as TDIG source items; 0 uses all",
    )
    tdig_recall.add_argument(
        "--score-field",
        choices=["support", "transition_probability", "lift", "pmi"],
        default=DEFAULT_TDIG_RECALL_SCORE_FIELD,
        help="edge statistic used to rank TDIG candidates across source history items",
    )
    tdig_recall.add_argument(
        "--per-source-top-k",
        type=int,
        default=50,
        help="number of direct TDIG candidates retained per source history item before aggregation",
    )
    tdig_recall.add_argument(
        "--aggregation",
        choices=["max", "sum"],
        default="max",
        help="how to combine scores for candidates reached from multiple source history items",
    )
    tdig_recall.add_argument(
        "--gap-bucket",
        choices=["same_session", "within_1d", "within_1w", "within_1m", "long_gap"],
        default=None,
        help="optional time-gap bucket support used as the candidate score",
    )
    tdig_recall.add_argument(
        "--include-seen",
        action="store_true",
        help="allow previously seen items in generated TDIG candidates",
    )
    tdig_recall.add_argument(
        "--no-validation-history",
        action="store_true",
        help="for test evaluation, do not use each user's validation event as a source history item",
    )
    tdig_recall.add_argument(
        "--include-same-timestamp-transitions",
        action="store_true",
        help="include same-user same-timestamp transitions in the as-of TDIG state",
    )
    tdig_recall.add_argument("--seed", type=int, default=2026)

    stress = evaluate_sub.add_parser(
        "semantic-transition-stress",
        aliases=["stress-candidates"],
        help="build semantic-vs-transition hard candidate sets with as-of TDIG evidence",
    )
    stress.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("artifacts/datasets/movielens_1m"),
        help="processed dataset directory containing interactions.csv and items.csv",
    )
    stress.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="run output directory; defaults to runs/<timestamp>-semantic-transition-stress",
    )
    stress.add_argument(
        "--split-name",
        choices=["temporal_leave_one_out", "global_time"],
        default="temporal_leave_one_out",
    )
    stress.add_argument("--eval-split", choices=["val", "test"], default="test")
    stress.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    stress.add_argument(
        "--max-history-items",
        type=int,
        default=20,
        help="number of recent user history events used as source items; 0 uses all",
    )
    stress.add_argument(
        "--score-field",
        choices=["support", "transition_probability", "lift", "pmi"],
        default=DEFAULT_TDIG_RECALL_SCORE_FIELD,
        help="edge statistic used to rank TDIG transition candidates",
    )
    stress.add_argument(
        "--per-source-top-k",
        type=int,
        default=50,
        help="number of direct TDIG candidates retained per source history item",
    )
    stress.add_argument(
        "--aggregation",
        choices=["max", "sum"],
        default="max",
        help="how to combine TDIG scores for candidates reached from multiple source items",
    )
    stress.add_argument(
        "--gap-bucket",
        choices=["same_session", "within_1d", "within_1w", "within_1m", "long_gap"],
        default=None,
        help="optional time-gap bucket support used as the transition score",
    )
    stress.add_argument(
        "--include-seen",
        action="store_true",
        help="allow previously seen items in generated hard negatives",
    )
    stress.add_argument(
        "--no-validation-history",
        action="store_true",
        help="for test evaluation, do not use each user's validation event as source history",
    )
    stress.add_argument(
        "--include-same-timestamp-transitions",
        action="store_true",
        help="include same-user same-timestamp transitions in the as-of TDIG state",
    )
    stress.add_argument(
        "--max-eval-cases",
        type=int,
        default=None,
        help="optional deterministic prefix of eval cases for engineering smoke runs",
    )
    stress.add_argument("--seed", type=int, default=2026)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = "tglrec " + " ".join(argv) if argv is not None else " ".join(sys.argv)
    if args.version:
        from tglrec import __version__

        print(__version__)
        return 0
    if args.command == "check-config":
        config = load_config(args.path)
        print(f"loaded {args.path}: top-level keys={sorted(config)}")
        return 0
    if args.command == "preprocess" and args.dataset == "movielens-1m":
        set_global_seed(args.seed)
        result = preprocess_movielens_1m(
            output_dir=args.output_dir,
            raw_dir=args.raw_dir,
            zip_path=args.zip_path,
            download=args.download,
            download_dir=args.download_dir,
            min_user_interactions=args.min_user_interactions,
            min_item_interactions=args.min_item_interactions,
            global_train_ratio=args.global_train_ratio,
            global_val_ratio=args.global_val_ratio,
            seed=args.seed,
        )
        print(f"processed MovieLens-1M: {result.output_dir}")
        print(f"interactions={result.num_interactions} users={result.num_users} items={result.num_items}")
        return 0
    if args.command == "preprocess" and args.dataset == "amazon-reviews-2023":
        set_global_seed(args.seed)
        result = preprocess_amazon_reviews_2023(
            reviews_path=args.reviews_path,
            metadata_path=args.metadata_path,
            category=args.category,
            output_dir=args.output_dir,
            user_col=args.user_col,
            item_col=args.item_col,
            item_fallback_col=args.item_fallback_col,
            timestamp_col=args.timestamp_col,
            rating_col=args.rating_col,
            metadata_item_col=args.metadata_item_col,
            min_rating=args.min_rating,
            min_user_interactions=args.min_user_interactions,
            min_item_interactions=args.min_item_interactions,
            global_train_ratio=args.global_train_ratio,
            global_val_ratio=args.global_val_ratio,
            deduplicate_user_items=not args.keep_duplicate_user_items,
            allow_same_timestamp_user_events=args.allow_same_timestamp_user_events,
            source_file_url=args.source_file_url,
            metadata_source_url=args.metadata_source_url,
            hf_revision=args.hf_revision,
            seed=args.seed,
            command=command,
        )
        print(f"processed Amazon Reviews 2023: {result.output_dir}")
        print(f"interactions={result.num_interactions} users={result.num_users} items={result.num_items}")
        return 0
    if args.command == "graph" and args.graph_command == "build-tdig":
        result = build_tdig_artifact(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            split_name=args.split_name,
            train_split_label=args.train_split_label,
            strict_before_timestamp=args.strict_before_timestamp,
            include_same_timestamp_transitions=args.include_same_timestamp_transitions,
            command=command,
        )
        print(f"wrote TDIG artifact: {result.output_dir}")
        print(f"edges={result.num_edges} transitions={result.num_transitions}")
        return 0
    if args.command == "evaluate" and args.evaluator == "sanity-baselines":
        set_global_seed(args.seed)
        result = run_sanity_baselines(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            split_name=args.split_name,
            eval_split=args.eval_split,
            ks=tuple(args.ks),
            item_knn_neighbors=args.item_knn_neighbors,
            item_knn_max_history_items=args.item_knn_max_history_items,
            cooccurrence_history_window=args.cooccurrence_history_window,
            use_validation_history_for_test=not args.no_validation_history,
            exclude_seen=not args.include_seen,
            seed=args.seed,
            command=command,
        )
        print(f"wrote sanity baseline run: {result.output_dir}")
        print(f"eval_cases={result.num_cases}")
        for baseline, metrics in result.metrics.items():
            metric_text = " ".join(f"{name}={value:.6f}" for name, value in sorted(metrics.items()))
            print(f"{baseline}: {metric_text}")
        return 0
    if args.command == "evaluate" and args.evaluator == "history-perturbations":
        set_global_seed(args.seed)
        result = run_history_perturbation_diagnostics(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            split_name=args.split_name,
            eval_split=args.eval_split,
            ks=tuple(args.ks),
            perturbations=tuple(args.perturbations),
            item_knn_neighbors=args.item_knn_neighbors,
            item_knn_max_history_items=args.item_knn_max_history_items,
            cooccurrence_history_window=args.cooccurrence_history_window,
            use_validation_history_for_test=not args.no_validation_history,
            exclude_seen=not args.include_seen,
            seed=args.seed,
            command=command,
        )
        print(f"wrote history perturbation run: {result.output_dir}")
        print(f"eval_cases={result.num_cases}")
        for baseline, baseline_metrics in result.metrics.items():
            for perturbation, metrics in baseline_metrics.items():
                metric_text = " ".join(f"{name}={value:.6f}" for name, value in sorted(metrics.items()))
                print(f"{baseline}/{perturbation}: {metric_text}")
        return 0
    if args.command == "evaluate" and args.evaluator in {"tdig-candidate-recall", "tdig-recall"}:
        set_global_seed(args.seed)
        result = run_tdig_candidate_recall(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            split_name=args.split_name,
            eval_split=args.eval_split,
            ks=tuple(args.ks),
            max_history_items=args.max_history_items,
            per_source_top_k=args.per_source_top_k,
            score_field=args.score_field,
            aggregation=args.aggregation,
            gap_bucket_name=args.gap_bucket,
            use_validation_history_for_test=not args.no_validation_history,
            exclude_seen=not args.include_seen,
            include_same_timestamp_transitions=args.include_same_timestamp_transitions,
            seed=args.seed,
            command=command,
        )
        print(f"wrote TDIG candidate recall run: {result.output_dir}")
        print(f"eval_cases={result.num_cases}")
        metric_text = " ".join(f"{name}={value:.6f}" for name, value in sorted(result.metrics.items()))
        print(f"tdig_direct: {metric_text}")
        return 0
    if args.command == "evaluate" and args.evaluator in {
        "semantic-transition-stress",
        "stress-candidates",
    }:
        set_global_seed(args.seed)
        result = run_semantic_transition_stress(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            split_name=args.split_name,
            eval_split=args.eval_split,
            ks=tuple(args.ks),
            max_history_items=args.max_history_items,
            per_source_top_k=args.per_source_top_k,
            score_field=args.score_field,
            aggregation=args.aggregation,
            gap_bucket_name=args.gap_bucket,
            use_validation_history_for_test=not args.no_validation_history,
            exclude_seen=not args.include_seen,
            include_same_timestamp_transitions=args.include_same_timestamp_transitions,
            max_eval_cases=args.max_eval_cases,
            seed=args.seed,
            command=command,
        )
        print(f"wrote semantic-transition stress run: {result.output_dir}")
        print(f"eval_cases={result.num_cases}")
        metric_text = " ".join(
            f"{name}={value:.6f}" for name, value in sorted(result.metrics.items())
        )
        print(f"semantic_transition_stress: {metric_text}")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
