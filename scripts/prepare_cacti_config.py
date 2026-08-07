from __future__ import annotations

import argparse
import re
from pathlib import Path


def replace_one(text: str, pattern: str, replacement: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"expected exactly one CACTI config match for {pattern!r}, got {count}")
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.source)
    output = Path(args.output)
    text = source.read_text(encoding="utf-8")
    text = replace_one(text, r"^-size \(bytes\) \d+\s*$", "-size (bytes) 1048576")
    text = replace_one(text, r"^-block size \(bytes\) \d+\s*$", "-block size (bytes) 64")
    text = replace_one(text, r"^-associativity \d+\s*$", "-associativity 1")
    text = replace_one(text, r"^-read-write port \d+\s*$", "-read-write port 1")
    text = replace_one(text, r"^-exclusive read port \d+\s*$", "-exclusive read port 0")
    text = replace_one(text, r"^-exclusive write port \d+\s*$", "-exclusive write port 0")
    text = replace_one(text, r"^-UCA bank count \d+\s*$", "-UCA bank count 1")
    text = replace_one(text, r"^-technology \(u\) [0-9.]+\s*$", "-technology (u) 0.032")
    text = replace_one(text, r"^-output/input bus width \d+\s*$", "-output/input bus width 512")
    text = replace_one(text, r'^-cache type "[^"]+"\s*$', '-cache type "ram"')
    text = replace_one(text, r'^-Print input parameters - "[^"]+"\s*$', '-Print input parameters - "false"')
    output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
