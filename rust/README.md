# sqrt-space-kv (Rust)

Sqrt-Checkpoint KV-cache residency: cost models, keep-set computation, and
exact reference attention. Zero dependencies.

Full documentation, honest tradeoffs, and links to the paper / Python port:
see the [repository README](https://github.com/com-junkawasaki/sqrt-space-kv#readme).

## Install

```bash
cargo add sqrt-space-kv
```

## Quickstart

```rust
let s = 4096;
let keep = sqrt_space_kv::keep_indices(s);
println!("{} of {} kept", keep.len(), s);

let cost = sqrt_space_kv::strategy_cost("sqrt-checkpoint", s).unwrap();
println!("{} cells, fidelity={}", cost.kv_cells, cost.fidelity);
```

## Development

```bash
cargo test
cargo clippy -- -D warnings
```
