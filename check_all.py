"""Run every test file and every step's on-disk assertions; print one summary.

    python check_all.py                # tests + existing run artefacts
    python check_all.py --run ba54_f000

Exit code is non-zero if any check fails.
"""
import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
TESTS = ["tests/test_step1_2.py", "tests/test_step3_4.py", "tests/test_step5_6.py"]


def run(cmd, timeout=3600):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ba54_f000")
    ap.add_argument("--skip_tests", action="store_true")
    a = ap.parse_args()

    total_pass = total_fail = 0
    print("=" * 78)
    print("1) test files")
    print("=" * 78)
    if not a.skip_tests:
        for t in TESTS:
            p = os.path.join(ROOT, t)
            if not os.path.isfile(p):
                print(f"  [MISSING] {t}")
                total_fail += 1
                continue
            rc, out, err = run([PY, p])
            npass = len(re.findall(r"\[CHECK\]\[PASS\]|^ok ", out, re.M)) + \
                len(re.findall(r"\bPASS\b", out))
            nfail = len(re.findall(r"\[CHECK\]\[FAIL\]|\bFAIL\b", out))
            # trust the script's own summary line if present
            totals = re.findall(r"\[TEST\]\s*\S+:\s*(\d+) passed,\s*(\d+) failed", out)
            if not totals:
                totals = re.findall(r"tests/\S+:\s*(\d+)/(\d+) passed,\s*(\d+) failed", out)
                if totals:
                    p_, t_, f_ = totals[0]
                    npass, nfail = int(p_), int(f_)
            else:
                npass, nfail = (sum(int(a) for a, _ in totals),
                                sum(int(b) for _, b in totals))
            if not totals:
                m = re.search(r"ALL CHECKS PASSED", out)
                if m:
                    npass = len(re.findall(r"\[ok  \]", out))
                    nfail = 0
                m2 = re.search(r"FAILED:\s*(\d+) check", out)
                if m2:
                    nfail = int(m2.group(1))
            else:
                m = re.search(r"ALL CHECKS PASSED \((\d+) ok, (\d+) FAIL\)", out)
                if m:
                    npass, nfail = int(m.group(1)), int(m.group(2))
            total_pass += npass
            total_fail += nfail
            print(f"  {t:26s} exit={rc}  pass={npass}  fail={nfail}")
            if rc != 0 or nfail:
                tail = "\n      ".join((out or "").strip().splitlines()[-8:])
                print(f"      FAILED output tail:\n      {tail}")
                if err:
                    print(f"      stderr: {err.strip()[:400]}")

    print()
    print("=" * 78)
    print(f"2) on-disk stage assertions for run '{a.run}'")
    print("=" * 78)
    run_root = io_utils.run_dir(a.run, create=False)
    # Multi-frame runs now give every frame its OWN run directory (<run>__cam<src>-<dst>-f###)
    # and keep a per-frame copy in <run>/cam<src>-cam<dst>-f###/, so the stage dirs of the
    # top-level run may legitimately be empty.  Prefer the per-frame copies.
    frame_dirs = []
    if os.path.isdir(run_root):
        frame_dirs = sorted(d for d in os.listdir(run_root)
                            if d.startswith("cam") and
                            os.path.isdir(os.path.join(run_root, d)))
    stage_dirs = ["10_preproc", "20_warp", "30_class", "40_removal", "50_fill", "60_final"]
    if frame_dirs:
        print(f"   (per-frame copies found: {len(frame_dirs)}; checking all of them)")
    for sd in stage_dirs:
        npass = nfail = 0
        checked = 0
        candidates = ([os.path.join(run_root, fd, sd) for fd in frame_dirs]
                      if frame_dirs else [os.path.join(run_root, sd)])
        if not frame_dirs:
            candidates.append(os.path.join(run_root, sd))
        for base in candidates:
            p = os.path.join(base, "checks.txt")
            if not os.path.isfile(p):
                continue
            checked += 1
            with open(p, "r", encoding="utf-8") as fh:
                body = fh.read()
            npass += body.count("[CHECK][PASS]")
            nfail += body.count("[CHECK][FAIL]")
            if nfail:
                for line in body.splitlines():
                    if "[CHECK][FAIL]" in line:
                        print(f"      {os.path.basename(os.path.dirname(base))}: {line}")
        total_pass += npass
        total_fail += nfail
        if checked == 0:
            print(f"  {sd:12s} (no checks.txt)")
        else:
            flag = "OK " if nfail == 0 else "FAIL"
            print(f"  {sd:12s} {flag} pass={npass} fail={nfail}  ({checked} frame dirs)")

    print()
    print("=" * 78)
    print(f"TOTAL: {total_pass} passed, {total_fail} failed")
    print("=" * 78)
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())
