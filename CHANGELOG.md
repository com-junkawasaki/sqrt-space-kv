# Changelog

All notable changes to the `python/` and `rust/` library packages are
documented here. Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added

- Initial Python (`sqrt-space-kv` on PyPI, not yet published) and Rust
  (`sqrt-space-kv` on crates.io, not yet published) ports of the
  sqrt-checkpoint KV-cache residency cost models, keep-set computation,
  residency application, and exact reference attention.
- Golden-fixture conformance testing (`tests/fixtures/golden.json`,
  generated from `cloud-murakumo`'s canonical `.cljc`) covering both ports.
- `python/` package: pure Python (`>=3.11`), zero runtime dependencies.
- `rust/` crate: pure `std`, zero runtime dependencies.
