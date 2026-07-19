# Release checklist

## Pre-flight (first release only, or on rename)

Neither registry has been published to from this project before. Verify
name availability before anything else -- do not assume `sqrt-space-kv` is
free on either registry:

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://pypi.org/pypi/sqrt-space-kv/json
curl -s -o /dev/null -w "%{http_code}\n" https://crates.io/api/v1/crates/sqrt-space-kv
```

Expect `404` on both. If either returns `200`, STOP and pick a fallback
name before continuing -- that would also mean revisiting the PyPI import
name / crate name together, not just one of them.

## Python

```bash
cd python
python -m pip install -e ".[dev]"
pytest -q
python -m build --sdist --wheel -o ../tmp/dist
python -m venv ../tmp/release-venv
../tmp/release-venv/bin/pip install ../tmp/dist/sqrt_space_kv-*.whl
../tmp/release-venv/bin/python -c "import sqrt_space_kv; print(sqrt_space_kv.keep_indices(4096))"
twine check ../tmp/dist/*
twine upload ../tmp/dist/*        # manual, not automated -- first-ever PyPI publish from this project
```

## Rust

```bash
cd rust
cargo test
cargo clippy -- -D warnings
cargo publish --dry-run
cargo publish                     # manual, not automated -- first-ever crates.io publish from this project
```

## Tag and release

- Tag `vX.Y.Z`, push after CI is green on `main`.
- Attach the Python wheel + sdist to the GitHub release.
- Include the `CHANGELOG.md` section for this version in the release notes.

This intentionally does not automate publishing to PyPI/crates.io on a
tag push -- for a first-ever publish to either registry from this project,
a manual, deliberate step is the safer default. Automate later once the
process has been run manually at least once successfully.
