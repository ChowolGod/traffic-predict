"""Argument parsing, command dispatch and exit codes (M-16, plan 4.1)."""

import argparse
import logging
import sys
from collections.abc import Callable

from tp.config import PHASES
from tp.errors import TPError

log = logging.getLogger("tp")

# Filled in as each command is implemented (I-01..I-05).
HANDLERS: dict[str, Callable[[argparse.Namespace], None]] = {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tp", description="밀라노 트래픽 예측 실험")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="I-01 캐시·구역·시계열 생성")
    p.add_argument("--phase", required=True, choices=PHASES)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("run", help="I-02 실험 1개 실행")
    p.add_argument("--config", required=True)
    p.add_argument("--retry", action="store_true")

    p = sub.add_parser("sweep", help="I-03 키 하나의 값 목록으로 자식 실험 실행")
    p.add_argument("--parent", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--values", required=True)
    p.add_argument("--start-id", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--hypothesis", required=True)

    sub.add_parser("results", help="I-04 결과 표·그림 재생성")

    p = sub.add_parser("test", help="I-05 test 구간 1회 평가")
    p.add_argument("--final", required=True)
    p.add_argument("--confirm", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    handler = HANDLERS.get(args.command)
    if handler is None:
        print(f"아직 구현되지 않은 명령: {args.command}", file=sys.stderr)
        return 2
    try:
        handler(args)
    except TPError as err:
        print(str(err), file=sys.stderr)
        return 1
    except Exception:
        log.exception("예상하지 못한 오류")
        return 2
    return 0
