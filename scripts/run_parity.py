#!/usr/bin/env python3
"""Run the numerical parity test suite against karpathy/nanoGPT reference."""

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent


def main():
    print("Running Neural Blueprint Studio nanoGPT Numerical Parity Suite on CPU...")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(ROOT_DIR / "server" / "tests" / "parity"),
        "-v",
    ]
    env = dict(PYTHONPATH=str(ROOT_DIR / "server"))
    res = subprocess.run(cmd, env=env)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
