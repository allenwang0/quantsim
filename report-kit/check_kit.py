"""Fail if a vendored report kit differs from the hashes pinned in KIT_VERSION.

Usage: python3 report-kit/check_kit.py [--canonical DIR]
  --canonical DIR  also require byte-identical files in DIR (the canonical kit).
Exit 0 when every file matches; 1 otherwise. Standard library only.
"""

import argparse
import hashlib
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinned(directory):
    lines = (directory / "KIT_VERSION").read_text(encoding="utf-8").splitlines()
    version, entries = lines[0].strip(), {}
    for line in lines[1:]:
        if line.strip():
            digest, name = line.split(maxsplit=1)
            entries[name.strip()] = digest
    return version, entries


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--canonical", type=pathlib.Path)
    args = parser.parse_args()
    version, entries = pinned(HERE)
    problems = []
    for name, digest in entries.items():
        path = HERE / name
        if not path.exists():
            problems.append(f"{name}: missing")
        elif sha256(path) != digest:
            problems.append(f"{name}: sha256 {sha256(path)[:12]} != pinned {digest[:12]}")
        if args.canonical and path.exists():
            other = args.canonical / name
            if not other.exists() or other.read_bytes() != path.read_bytes():
                problems.append(f"{name}: differs from {other}")
    for problem in problems:
        print(f"FAIL {problem}")
    print(f"report-kit {version}: {len(entries) - len({p.split(':')[0] for p in problems})}/{len(entries)} files match")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
