"""Generate small fake raw files + golden values (plan 4.5 실행 자산, seed 0).

Run: uv run python -m tests.fixtures.make_fixtures
"""

import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

FIXTURE_DATES = ("2013-11-04", "2013-11-05", "2013-11-06")
SQUARES = {1: 10.0, 2: 100.0, 5000: 200.0, 10000: 1.0}  # square_id -> base level
COUNTRIES = (39, 33)
SLOTS = 144
GAPS = {"2013-11-04": [50], "2013-11-06": [80, 81, 82, 83]}  # slots missing for every cell
ZERO = {"2013-11-05": (10000, [10, 11, 12])}  # slots where only this cell has no rows
GOLDEN_KEYS = [(5000, "2013-11-04", 0), (2, "2013-11-05", 70), (1, "2013-11-06", 143)]
OUT = Path(__file__).parent


def slot_time(day: str, slot: int) -> datetime:
    local = datetime.fromisoformat(day).replace(tzinfo=ZoneInfo("Europe/Rome"))
    return (local + timedelta(minutes=10 * slot)).astimezone(ZoneInfo("UTC"))


def fmt(value: float | None) -> str:
    return "" if value is None else repr(round(value, 6))


def make_fixtures(out: Path = OUT) -> None:
    rng = random.Random(0)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    totals: dict[tuple[int, str, int], float] = {}
    empty_internet_keys = []

    for day in FIXTURE_DATES:
        rows = []
        for slot in range(SLOTS):
            if slot in GAPS.get(day, []):
                continue
            ms = int(slot_time(day, slot).timestamp() * 1000)
            for square, base in SQUARES.items():
                zero_square, zero_slots = ZERO.get(day, (None, []))
                if square == zero_square and slot in zero_slots:
                    continue
                for cc in COUNTRIES:
                    if cc == 33 and rng.random() < 0.5:
                        continue
                    internet = base + rng.uniform(0, 50) if cc == 39 else rng.uniform(0, 1)
                    if cc == 33 and rng.random() < 0.2:
                        internet = None
                        empty_internet_keys.append((square, day, slot))
                    totals[(square, day, slot)] = totals.get((square, day, slot), 0.0) + (
                        0.0 if internet is None else round(internet, 6)
                    )
                    sms = [rng.uniform(0, 5) if rng.random() < 0.7 else None for _ in range(4)]
                    rows.append(
                        "\t".join([str(square), str(ms), str(cc), *map(fmt, sms), fmt(internet)])
                    )
        rng.shuffle(rows)
        path = out / "raw" / f"sms-call-internet-mi-{day}.txt"
        path.write_text("\n".join(rows) + "\n", encoding="ascii", newline="\n")

    keys = GOLDEN_KEYS + empty_internet_keys[:1]
    golden = {
        "top_square_id": 5000,
        "cache_values": [
            {"square_id": s, "time_utc": slot_time(d, t).isoformat(), "internet": totals[(s, d, t)]}
            for s, d, t in keys
        ],
        "missing_gaps": [
            {"start_utc": slot_time(d, slots[0]).isoformat(), "length": len(slots)}
            for d, slots in GAPS.items()
        ],
        "zero_cells": [
            {"square_id": s, "time_utc": slot_time(d, t).isoformat()}
            for d, (s, slots) in ZERO.items()
            for t in slots
        ],
        "present_times_per_day": {d: SLOTS - len(GAPS.get(d, [])) for d in FIXTURE_DATES},
        "metrics": {
            "y_true": [1.0, 2.0, 3.0, 4.0],
            "y_pred": [1.0, 3.0, 2.0, 6.0],
            "mae": 1.0,
            "rmse": math.sqrt(1.5),
        },
    }
    (out / "golden.json").write_text(
        json.dumps(golden, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    make_fixtures()
