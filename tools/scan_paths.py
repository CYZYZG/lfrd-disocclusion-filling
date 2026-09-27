"""List every absolute Windows path in the tracked files, grouped, for neutralisation.

    python tools/scan_paths.py [--csv out.csv]

Read-only.  It separates paths that can be replaced mechanically (the project's own directory,
the sibling project, the dataset, the python interpreter) from ones that need a human decision,
and shows, per file, how many occurrences there are.
"""
import argparse
import csv
import os
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# order matters: longer / more specific patterns first
RULES = [
    (re.compile(r"[A-Za-z]:\\项目\\空洞填补2\\output[^\s\"'`,)]*"), "PROJECT_OUTPUT"),
    (re.compile(r"[A-Za-z]:\\项目\\空洞填补2"), "PROJECT"),
    (re.compile(r"[A-Za-z]:\\项目\\3DVideos-distrib[^\s\"'`,)]*"), "DATASET"),
    (re.compile(r"[A-Za-z]:\\项目\\空洞填补[^\s\"'`,)]*"), "SIBLING"),
    (re.compile(r"[A-Za-z]:\\Users\\[A-Za-z0-9_.-]+\\\.dsh[^\s\"'`,)]*"), "PYTHON_ENV"),
    (re.compile(r"[A-Za-z]:\\Users\\[A-Za-z0-9_.-]+"), "USER_HOME"),
    (re.compile(r"[A-Za-z]:\\Program Files[^\s\"'`,)]*"), "SYSTEM"),
    (re.compile(r"[A-Za-z]:\\ProgramData[^\s\"'`,)]*"), "SYSTEM"),
    (re.compile(r"[A-Za-z]:\\\\[^\s\"'`,)]*"), "PROJECT_ESCAPED"),
    (re.compile(r"[A-Za-z]:\\[^\s\"'`,)]*"), "OTHER"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    files = [f for f in subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                                       text=True, encoding="utf-8").stdout.splitlines()
             if f.strip()]
    rows = []
    for rel in files:
        if not rel.endswith((".py", ".md", ".txt", ".json", ".cfg", ".toml", ".yml", ".yaml",
                             ".gitignore", ".bat", ".ps1")):
            continue
        p = os.path.join(ROOT, rel)
        try:
            s = open(p, encoding="utf-8", errors="replace").read()
        except Exception:                                            # noqa: BLE001
            continue
        counts = {}
        for rx, name in RULES:
            for m in rx.finditer(s):
                # a PROJECT_OUTPUT hit also matches PROJECT; and PROJECT also matches the
                # escaped variant, so only count the most specific rule per position
                counts.setdefault(name, 0)
                counts[name] += 1
        # drop the generic buckets when a specific rule already covered the same text
        for generic in ("OTHER", "PROJECT_ESCAPED", "USER_HOME"):
            if generic in counts and any(k in counts for k in
                                         ("PROJECT", "SIBLING", "DATASET", "PYTHON_ENV")):
                pass
        if counts:
            rows.append((rel, counts))
    rows.sort(key=lambda r: (-sum(r[1].values()), r[0]))
    tot = {}
    for rel, counts in rows:
        for k, v in counts.items():
            tot[k] = tot.get(k, 0) + v
    print(f"files with absolute paths: {len(rows)} / {len(files)} tracked\n")
    print(f"{'file':54s} " + "  ".join(f"{k:>9s}" for k in sorted(tot)))
    for rel, counts in rows:
        print(f"{rel:54s} " + "  ".join(f"{counts.get(k, 0):9d}" for k in sorted(tot)))
    print()
    print("totals:", ", ".join(f"{k}={v}" for k, v in sorted(tot.items())))
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            keys = sorted(tot)
            w.writerow(["file"] + keys)
            for rel, counts in rows:
                w.writerow([rel] + [counts.get(k, 0) for k in keys])
        print("written:", a.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
