# sqrt-space-kv (Python)

Sqrt-Checkpoint KV-cache residency: cost models, keep-set computation, and
exact reference attention. Zero runtime dependencies.

Full documentation, honest tradeoffs, and links to the paper / Rust port:
see the [repository README](../README.md).

## Install

```bash
pip install sqrt-space-kv
```

## Quickstart

```python
from sqrt_space_kv import keep_indices, strategy_cost

s = 4096
keep = keep_indices(s)          # token indices to keep resident
print(len(keep), "of", s, "kept")

cost = strategy_cost("sqrt-checkpoint", s)
print(cost["kv_cells"], cost["fidelity"])
```

## Development

```bash
python -m pip install -e ".[dev]"
pytest -q
```
