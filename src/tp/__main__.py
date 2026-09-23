"""Entry point. Thread env vars must be set before numpy/torch are imported (plan 3.3)."""

import os
import sys
from pathlib import Path

import yaml

DEFAULT_THREADS = 4


def configure_threads(argv: list[str]) -> None:
    threads = DEFAULT_THREADS
    if "--config" in argv:
        index = argv.index("--config") + 1
        if index < len(argv) and Path(argv[index]).is_file():
            cfg = yaml.safe_load(Path(argv[index]).read_text(encoding="utf-8")) or {}
            threads = (cfg.get("runtime") or {}).get("threads", DEFAULT_THREADS)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)


if __name__ == "__main__":
    configure_threads(sys.argv[1:])
    from tp.cli import main

    sys.exit(main())
