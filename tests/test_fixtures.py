import json
from pathlib import Path

from tests.fixtures.make_fixtures import FIXTURE_DATES, make_fixtures

FIXTURES = Path(__file__).parent / "fixtures"


def test_fixtures_are_reproducible(tmp_path):
    make_fixtures(tmp_path)
    for name in [*(f"raw/sms-call-internet-mi-{d}.txt" for d in FIXTURE_DATES), "golden.json"]:
        assert (tmp_path / name).read_bytes() == (FIXTURES / name).read_bytes(), name


def test_golden_contains_planned_cases():
    golden = json.loads((FIXTURES / "golden.json").read_text(encoding="utf-8"))
    assert len(golden["cache_values"]) >= 3
    assert len(golden["missing_gaps"]) == 2
    assert sorted(g["length"] for g in golden["missing_gaps"]) == [1, 4]
    assert golden["zero_cells"], "구역 행만 빠진 사례(→ 0)가 있어야 함"
    assert golden["metrics"]["mae"] == 1.0
