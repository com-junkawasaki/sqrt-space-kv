# Release checklist

## Pre-flight (first release only, or on rename)

Verify name availability before anything else -- do not assume
`sqrt-space-kv` is free on either registry:

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://pypi.org/pypi/sqrt-space-kv/json
curl -s -A "sqrt-space-kv-name-check" -o /dev/null -w "%{http_code}\n" https://crates.io/api/v1/crates/sqrt-space-kv
```

Expect `404` on both (confirmed 2026-07-19). If either returns `200`, STOP
and pick a fallback name before continuing -- that would also mean
revisiting the PyPI import name / crate name together, not just one of them.

## Python -- via PyPI Trusted Publishing (no stored token)

`.github/workflows/publish.yml` runs the test suite, builds sdist+wheel,
and publishes via [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
(GitHub Actions OIDC -- no API token stored anywhere, ever) whenever a
`v*` tag is pushed.

**One-time setup, before the first tag** (PyPI web UI, ~1 minute, cannot
be done via CLI -- this is intentionally gated behind a logged-in PyPI
session): at https://pypi.org/manage/account/publishing/, add a pending
publisher with:

| Field | Value |
|---|---|
| PyPI project name | `sqrt-space-kv` |
| Owner | `com-junkawasaki` |
| Repository name | `sqrt-space-kv` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

After that, releasing is just:

```bash
cd python
python -m pip install -e ".[dev]"
pytest -q                                    # sanity check locally before tagging
git tag v0.1.0 && git push origin v0.1.0     # CI takes it from here
```

Local dry-run smoke test (already verified 2026-07-19, re-run if the
package structure changes):

```bash
python -m build --sdist --wheel -o ../tmp/dist
python -m venv ../tmp/release-venv
../tmp/release-venv/bin/pip install ../tmp/dist/sqrt_space_kv-*.whl
../tmp/release-venv/bin/python -c "import sqrt_space_kv; print(sqrt_space_kv.keep_indices(4096))"
```

## Rust -- manual (crates.io has no trusted-publishing-from-fork-free path here yet)

`cargo login` itself requires a one-time browser step (GitHub OAuth at
https://crates.io/me) to generate a token -- there is no CLI-only path to
create it. Once you have a token:

```bash
cargo login <token>     # one-time per machine, stores in ~/.cargo/credentials.toml
cd rust
cargo test
cargo clippy -- -D warnings
cargo publish --dry-run   # already verified 2026-07-19 with --allow-dirty
cargo publish
```

## Tag and release

- Tag `vX.Y.Z`, push after CI is green on `main` -- this also triggers
  the Python publish workflow above.
- `cargo publish` separately (not tag-triggered; run by hand per above).
- Attach the Python wheel + sdist to the GitHub release.
- Include the `CHANGELOG.md` section for this version in the release notes.
