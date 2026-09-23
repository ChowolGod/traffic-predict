"""F-10 README (plan 2 Should): links work, limitations are stated."""

import re

from tp import config

README = config.ROOT / "README.md"


def test_readme_relative_links_exist():
    text = README.read_text(encoding="utf-8")
    links = [t for t in re.findall(r"\]\(([^)]+)\)", text) if not t.startswith("http")]
    assert "results/test/metrics.md" in links
    experiment_links = [t for t in links if t.startswith("experiments/EXP-")]
    assert len(experiment_links) == len(list((config.ROOT / "experiments").glob("EXP-*_*")))
    missing = [t for t in links if not (config.ROOT / t.split("#")[0]).exists()]
    assert missing == []


def test_readme_states_the_limitations():
    text = README.read_text(encoding="utf-8")
    for phrase in ("val은 선택용이라 편향됨", "모든 모델은 train으로만 학습", "11/01, 12/07, 12/08",
                   "성탄 직전 쇼핑 주간", "부트스트랩 블록이 7개뿐"):  # fmt: skip
        assert phrase in text, phrase
