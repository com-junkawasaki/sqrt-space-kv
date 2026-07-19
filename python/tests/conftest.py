import json
import pathlib

import pytest

# Root-level, shared with the Rust port -- do not duplicate this file per
# language, that would recreate the exact cross-language drift problem this
# fixture exists to prevent.
FIXTURE_PATH = pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "golden.json"


@pytest.fixture(scope="session")
def golden():
    return json.loads(FIXTURE_PATH.read_text())
