# Contributing

## Running the test suites

```bash
# Python
cd python
python -m pip install -e ".[dev]"
pytest -q

# Rust
cd rust
cargo test
cargo clippy -- -D warnings
cargo fmt --check
```

Both suites are checked against the single shared fixture at
`tests/fixtures/golden.json` -- do not add a per-language copy of it.

## Why no FFI / shared binary

`python/` and `rust/` are independent, hand-written ports of the same
canonical implementation (`gftdcojp/cloud-murakumo`'s `sqrt_space.cljc` /
`kv_runtime.cljc`), not bindings generated from a shared Rust core (no
PyO3/maturin/uniffi). This matches how this project's monorepo handles
cross-language reuse elsewhere: keep one canonical reference implementation,
hand-port the pure math to each target language, and use a golden-fixture
conformance test to catch drift instead of a shared binary. See the
repo-local ADR at `90-docs/adr/2607190900-python-rust-library-port.md` for the reasoning.

## Regenerating the golden fixture

The fixture is generated from `cloud-murakumo` (a separate, private repo)
and vendored here by hand -- there is no CI job that pulls from a private
repo into this public one. If the canonical `.cljc` implementation changes:

1. In a `cloud-murakumo` checkout, run:
   ```bash
   clj -M:sqrt-kv-fixtures /path/to/golden.json
   ```
   (alias defined in `cloud-murakumo`'s `deps.edn`, source in
   `src/cloud_murakumo/sqrt_space_kv_fixtures.clj`)
2. Copy the output to this repo's `tests/fixtures/golden.json`.
3. Run both test suites; fix any port that no longer matches.
4. Bump this repo's version and note the change in `CHANGELOG.md`.

## Adding a new ported function

1. Read the function in `cloud-murakumo`'s `sqrt_space.cljc` / `kv_runtime.cljc`.
2. Port it to `python/src/sqrt_space_kv/` and `rust/src/` with matching
   semantics (snake_case field names in both; strategy names and fidelity
   labels stay as the literal hyphenated strings used by the canonical
   implementation, e.g. `"sqrt-checkpoint"`, `"exact-with-recompute"`).
3. Add or extend a case in `cloud-murakumo`'s fixture-dump script covering
   the new function, regenerate `golden.json` (see above), and add a
   matching assertion in both `python/tests/` and `rust/tests/golden.rs`.

## A note from the initial port

The first Python translation of `block_stream_attn` had a real bug: it
never carried `m_new` into the next loop iteration's `m`, silently
discarding all prior blocks' contributions. The golden-fixture test caught
it immediately (delta ~0.13 against the reference, versus the ~1e-16 the
canonical implementation gets). This is exactly the failure mode the
fixture approach exists to catch -- a port that looks reasonable, compiles,
and runs, but is quietly wrong on multi-block inputs.
