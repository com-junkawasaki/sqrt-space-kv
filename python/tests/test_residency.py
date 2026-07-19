from sqrt_space_kv import residency as res


def test_apply_residency_kv_matches_golden(golden):
    for case in golden["residency"]:
        keep = case["keep"]
        k_in = case["k_in"]
        k2, v2 = res.apply_residency_kv(k_in, k_in, keep, case["seq_axis"])
        assert k2 == case["k_expected"], case["label"]
        assert v2 == case["k_expected"], case["label"]


def test_apply_residency_seq():
    xs = list(range(10))
    assert res.apply_residency_seq(xs, [0, 3, 9]) == [0, 3, 9]
    # out-of-range indices are dropped, not errored
    assert res.apply_residency_seq(xs, [-1, 0, 100]) == [0]


def test_apply_plan_to_kv_full_kv_is_noop():
    k = [[1, 2, 3]]
    v = [[4, 5, 6]]
    k2, v2 = res.apply_plan_to_kv(k, v, keep=[0], strategy="full-kv")
    assert k2 is k
    assert v2 is v
