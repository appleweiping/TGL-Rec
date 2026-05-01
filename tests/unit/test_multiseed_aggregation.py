from llm4rec.experiments.multiseed import aggregate_seed_metrics


def test_multiseed_aggregation_mean_std():
    rows = [{"Recall@5": 1.0}, {"Recall@5": 0.0}]
    aggregate = aggregate_seed_metrics(rows, metric_names=["Recall@5"])
    assert aggregate[0]["mean"] == 0.5
    assert aggregate[0]["num_seeds"] == 2
