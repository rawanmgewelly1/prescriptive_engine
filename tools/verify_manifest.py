"""Verify stored file hashes for this organized research snapshot."""
from pathlib import Path
import csv
import hashlib

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "FILE_MANIFEST_SHA256.csv"

def main():
    failures = []
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        entries = list(csv.DictReader(f))
    for entry in entries:
        path = ROOT / entry["relative_path"]
        if not path.is_file():
            failures.append("MISSING: " + entry["relative_path"])
            continue
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1048576), b""):
                h.update(chunk)
        if h.hexdigest() != entry["sha256"]:
            failures.append("CHANGED: " + entry["relative_path"])
    print(f"Verified {len(entries) - len(failures)}/{len(entries)} listed files")
    for failure in failures:
        print(failure)
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())
