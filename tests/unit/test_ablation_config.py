from llm4rec.methods.ablation import REQUIRED_ABLATIONS, ablation_switches, build_ablation_configs


def test_all_required_ablations_are_configurable():
    base = {"method": {"name": "time_graph_evidence_rec"}, "ablation": {}}
    configs = build_ablation_configs(base, names=REQUIRED_ABLATIONS)
    assert [cfg["method"]["ablation_name"] for cfg in configs] == REQUIRED_ABLATIONS
    assert all(cfg["method"]["reportable"] is False for cfg in configs)


def test_w_o_transition_edges_disables_transition_switch():
    switches = ablation_switches("w_o_transition_edges")
    assert switches.use_transition_edges is False
    assert switches.use_time_window_edges is True
