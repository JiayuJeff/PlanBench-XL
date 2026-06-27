from __future__ import annotations

import sys


MESSAGE = "This evaluation script is still under construction. We will provide an updated version as soon as possible."


def main() -> int:
    print(MESSAGE, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
