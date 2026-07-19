from sqrt_space_kv import attention as attn


def test_block_stream_matches_online_softmax_within_language(golden):
    for case in golden["attention"]:
        q, k, v, scale, block = case["q"], case["k"], case["v"], case["scale"], case["block"]
        full = attn.online_softmax_attn(q, k, v, scale)
        blk = attn.block_stream_attn(q, k, v, scale, block)
        delta = max(abs(a - b) for a, b in zip(full, blk))
        assert delta < 1e-9, case["label"]


def test_matches_golden_reference_values(golden):
    for case in golden["attention"]:
        q, k, v, scale, block = case["q"], case["k"], case["v"], case["scale"], case["block"]
        full = attn.online_softmax_attn(q, k, v, scale)
        blk = attn.block_stream_attn(q, k, v, scale, block)
        for a, b in zip(full, case["online_softmax_attn"]):
            assert abs(a - b) < 1e-6, case["label"]
        for a, b in zip(blk, case["block_stream_attn"]):
            assert abs(a - b) < 1e-6, case["label"]
