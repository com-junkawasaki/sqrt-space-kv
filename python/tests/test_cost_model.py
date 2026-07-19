import pytest

from sqrt_space_kv import cost_model as cm


def test_optimal_block_tree_height_keep_indices(golden):
    for case in golden["optimal_block_tree_height_keep_indices"]:
        s = case["s"]
        b = cm.optimal_block(s)
        assert b == case["b"], f"optimal_block({s})"
        h = cm.tree_height(s, b)
        assert h == case["h"], f"tree_height({s}, {b})"
        keep = cm.keep_indices(s, b)
        assert keep == case["keep"], f"keep_indices({s}, {b})"
        # keep_indices() with no explicit b must match too
        assert cm.keep_indices(s) == case["keep"]


def test_strategy_cost_matches_golden(golden):
    for case in golden["strategy_cost"]:
        strat = case["strategy"]
        s = case["s"]
        expected = case["result"]
        got = cm.strategy_cost(strat, s)
        assert got["kv_cells"] == expected["kv-cells"], strat
        assert got["weight_frac"] == pytest.approx(expected["weight-frac"]), strat
        assert got["recompute_per_token"] == pytest.approx(expected["recompute-per-token"]), strat
        assert got["fidelity"] == expected["fidelity"], strat
        assert got["bytes"] == expected["bytes"], strat


def test_strategy_catalog_has_six_strategies():
    assert set(cm.STRATEGY_CATALOG) == {
        "full-kv",
        "sliding-window",
        "sqrt-checkpoint",
        "block-respecting",
        "expert-page",
        "sqrt-plus-page",
    }


def test_sweep_and_compare_to_baseline_smoke():
    rows = cm.sweep([128, 4096])
    assert len(rows) == 2 * len(cm.STRATEGY_CATALOG)
    cmp = cm.compare_to_baseline(4096)
    assert len(cmp) == len(cm.STRATEGY_CATALOG)
    full_kv_row = next(r for r in cmp if r["strategy"] == "full-kv")
    assert full_kv_row["cells_saved"] == 0
