"""TPError: the only exception type the CLI turns into exit code 1 (plan 4.3)."""

KNOWN_CODES = frozenset(
    {
        "E-1001", "E-1002", "E-1003",
        "E-2001", "E-2002", "E-2003",
        "E-3001", "E-3002", "E-3003",
        "E-4001", "E-4002", "E-4003", "E-4004", "E-4005", "E-4006", "E-4007",
    }
)  # fmt: skip


class TPError(Exception):
    def __init__(self, code: str, message: str):
        if code not in KNOWN_CODES:
            raise ValueError(f"unknown error code: {code}")
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")
