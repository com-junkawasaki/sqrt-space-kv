//! Golden-fixture conformance tests.
//!
//! Root-level `tests/fixtures/golden.json`, shared with the Python port --
//! do not vendor a per-language copy, that would recreate the exact
//! cross-language drift problem this fixture exists to prevent.

use serde_json::Value;
use std::fs;

fn load_golden() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/fixtures/golden.json");
    let text = fs::read_to_string(path).expect("read golden.json");
    serde_json::from_str(&text).expect("parse golden.json")
}

fn f64_arr(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn u64_arr(v: &Value) -> Vec<u64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap())
        .collect()
}

fn matrix(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().unwrap().iter().map(f64_arr).collect()
}

#[test]
fn optimal_block_tree_height_keep_indices_match_golden() {
    let golden = load_golden();
    for case in golden["optimal_block_tree_height_keep_indices"]
        .as_array()
        .unwrap()
    {
        let s = case["s"].as_u64().unwrap();
        let expected_b = case["b"].as_u64().unwrap();
        let expected_h = case["h"].as_u64().unwrap();
        let expected_keep = u64_arr(&case["keep"]);

        let b = sqrt_space_kv::optimal_block(s);
        assert_eq!(b, expected_b, "optimal_block({s})");
        let h = sqrt_space_kv::tree_height(s, b);
        assert_eq!(h, expected_h, "tree_height({s}, {b})");
        let keep = sqrt_space_kv::keep_indices(s);
        assert_eq!(keep, expected_keep, "keep_indices({s})");
    }
}

#[test]
fn strategy_cost_matches_golden() {
    let golden = load_golden();
    for case in golden["strategy_cost"].as_array().unwrap() {
        let strat = case["strategy"].as_str().unwrap();
        let s = case["s"].as_u64().unwrap();
        let expected = &case["result"];
        let got = sqrt_space_kv::strategy_cost(strat, s).unwrap();
        assert_eq!(
            got.kv_cells,
            expected["kv-cells"].as_u64().unwrap(),
            "{strat}"
        );
        assert!(
            (got.weight_frac - expected["weight-frac"].as_f64().unwrap()).abs() < 1e-9,
            "{strat}"
        );
        assert!(
            (got.recompute_per_token - expected["recompute-per-token"].as_f64().unwrap()).abs()
                < 1e-9,
            "{strat}"
        );
        assert_eq!(
            got.fidelity,
            expected["fidelity"].as_str().unwrap(),
            "{strat}"
        );
        assert_eq!(got.bytes, expected["bytes"].as_u64().unwrap(), "{strat}");
    }
}

#[test]
fn strategy_catalog_has_six_strategies() {
    assert_eq!(sqrt_space_kv::STRATEGY_NAMES.len(), 6);
}

#[test]
fn attention_matches_golden() {
    let golden = load_golden();
    for case in golden["attention"].as_array().unwrap() {
        let q = f64_arr(&case["q"]);
        let k = matrix(&case["k"]);
        let v = matrix(&case["v"]);
        let scale = case["scale"].as_f64().unwrap();
        let block = case["block"].as_u64().unwrap();
        let label = case["label"].as_str().unwrap();

        let full = sqrt_space_kv::online_softmax_attn(&q, &k, &v, scale);
        let blk = sqrt_space_kv::block_stream_attn(&q, &k, &v, scale, block);

        let delta = full
            .iter()
            .zip(blk.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(delta < 1e-9, "{label}: within-language delta={delta}");

        let expected_full = f64_arr(&case["online_softmax_attn"]);
        let expected_blk = f64_arr(&case["block_stream_attn"]);
        for (a, b) in full.iter().zip(expected_full.iter()) {
            assert!((a - b).abs() < 1e-6, "{label}");
        }
        for (a, b) in blk.iter().zip(expected_blk.iter()) {
            assert!((a - b).abs() < 1e-6, "{label}");
        }
    }
}

#[test]
fn residency_matches_golden() {
    let golden = load_golden();
    for case in golden["residency"].as_array().unwrap() {
        let keep = u64_arr(&case["keep"]);
        // k_in / k_expected are [H][S][D] with H=1
        let k_in: Vec<Vec<Vec<f64>>> = case["k_in"]
            .as_array()
            .unwrap()
            .iter()
            .map(matrix)
            .collect();
        let expected: Vec<Vec<Vec<f64>>> = case["k_expected"]
            .as_array()
            .unwrap()
            .iter()
            .map(matrix)
            .collect();

        let (k2, v2) = sqrt_space_kv::apply_residency_kv_hsd(&k_in, &k_in, &keep);
        assert_eq!(k2, expected, "{}", case["label"]);
        assert_eq!(v2, expected, "{}", case["label"]);
    }
}
