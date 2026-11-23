"""
Run the offline test suite and record its summary in results/tests.json.

    python scripts/record_tests.py

The report quotes the pass count from this file, so it is a recorded run at a
named commit, not a number typed into the page. Live and browser tests are
deselected, as in CI's main test step.
"""

import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARKERS = "not live and not browser"


def main() -> int:
    command = [sys.executable, "-m", "pytest", "-m", MARKERS, "-q", "-p", "no:cacheprovider"]
    started = datetime.now(timezone.utc).replace(microsecond=0)
    run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    summary = run.stdout.strip().splitlines()[-1]
    counts = {key: int(n) for n, key in re.findall(r"(\d+) (\w+)", summary)}
    seconds = re.search(r"in ([\d.]+)s", summary)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    record = {
        "command": f'pytest -m "{MARKERS}" -q',
        "sha": sha.stdout.strip(),
        "date": started.isoformat().replace("+00:00", "Z"),
        "python": platform.python_version(),
        "passed": counts.get("passed", 0),
        "failed": counts.get("failed", 0),
        "errors": counts.get("error", 0) + counts.get("errors", 0),
        "skipped": counts.get("skipped", 0),
        "deselected": counts.get("deselected", 0),
        "seconds": float(seconds.group(1)) if seconds else None,
        "summary": summary.strip("= "),
    }
    out = ROOT / "results" / "tests.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Wrote {out}: {record['summary']}")
    return run.returncode


if __name__ == "__main__":
    sys.exit(main())
