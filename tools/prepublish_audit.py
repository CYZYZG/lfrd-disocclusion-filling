"""Pre-publication audit: what exactly would become public?

    python tools/prepublish_audit.py

Checks the TRACKED files (what a push would publish) for
  * licence-restricted material (the paper, the dataset)
  * credentials / tokens / absolute personal paths
  * large files and the total size
  * the .gitignore coverage of the things that must NOT be published
and prints the git history size too.
"""
import os
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git(*args):
    r = subprocess.run(["git"] + list(args), cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return r.stdout


SECRET_PATTERNS = [
    (r"gho_[A-Za-z0-9]{20,}", "GitHub OAuth token"),
    (r"ghp_[A-Za-z0-9]{20,}", "GitHub PAT"),
    (r"github_pat_[A-Za-z0-9_]{20,}", "GitHub fine-grained PAT"),
    (r"glpat-[A-Za-z0-9\-_]{15,}", "GitLab token"),
    (r"AKIA[0-9A-Z]{16}", "AWS access key"),
    (r"sk-[A-Za-z0-9]{20,}", "OpenAI-style key"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "private key"),
    (r"(?i)password\s*[:=]\s*['\"][^'\"]{4,}", "hard-coded password"),
    (r"(?i)api[_-]?key\s*[:=]\s*['\"][^'\"]{8,}", "hard-coded API key"),
]
FORBIDDEN = [
    (r"\.pdf$", "论文 PDF（IEEE 版权）"),
    (r"^paper_figs/", "论文截图（IEEE 版权）"),
    (r"^paper_text\.txt$", "论文正文提取"),
    (r"^output/", "运行产物（体积大，可重生成）"),
    (r"^input/", "数据集副本"),
    (r"\.onnx$|\.pth$|\.tflite$|\.caffemodel$|\.h5$", "模型权重"),
    (r"^\.env$|\.pem$", "环境/证书文件"),
]


def main():
    files = [f for f in git("ls-files").splitlines() if f.strip()]
    print(f"tracked files: {len(files)}")
    total = 0
    sizes = []
    for rel in files:
        p = os.path.join(ROOT, rel)
        sz = os.path.getsize(p) if os.path.isfile(p) else 0
        total += sz
        sizes.append((sz, rel))
    print(f"tracked size : {total / 1024:.0f} KB")
    print()
    print("--- largest tracked files ---")
    for sz, rel in sorted(sizes, reverse=True)[:10]:
        print(f"  {sz / 1024:9.1f} KB  {rel}")

    print()
    print("--- forbidden material in the tracked set ---")
    bad = []
    for rel in files:
        for pat, why in FORBIDDEN:
            if re.search(pat, rel):
                bad.append((rel, why))
    if bad:
        for rel, why in bad:
            print(f"  !! {rel}  ({why})")
    else:
        print("  none -- the paper, the dataset, the weights and output/ are all untracked")

    print()
    print("--- credentials / personal paths inside tracked content ---")
    hits = []
    for rel in files:
        p = os.path.join(ROOT, rel)
        if not os.path.isfile(p) or os.path.getsize(p) > 4 * 1024 * 1024:
            continue
        if rel.endswith((".png", ".jpg", ".jpeg", ".npy", ".npz")):
            continue
        try:
            s = open(p, encoding="utf-8", errors="replace").read()
        except Exception:                                            # noqa: BLE001
            continue
        for pat, why in SECRET_PATTERNS:
            for m in re.finditer(pat, s):
                hits.append((rel, why, m.group(0)[:40]))
    if hits:
        for rel, why, frag in hits[:20]:
            print(f"  !! {rel}: {why}  [{frag}]")
        print(f"  total {len(hits)}")
    else:
        print("  none")

    print()
    print("--- absolute personal paths in tracked content (privacy / portability) ---")
    pathpat = re.compile(r"[A-Za-z]:\\\\?[^\s\"'<>|]{3,}|[A-Za-z]:\\[^\s\"'<>|]{3,}")
    pathhits = {}
    for rel in files:
        p = os.path.join(ROOT, rel)
        if not os.path.isfile(p) or not rel.endswith((".py", ".md", ".txt", ".json")):
            continue
        try:
            s = open(p, encoding="utf-8", errors="replace").read()
        except Exception:                                            # noqa: BLE001
            continue
        for m in pathpat.finditer(s):
            t = m.group(0)
            if re.match(r"(?i)^[A-Z]:\\(windows|program files|programdata)", t):
                continue
            pathhits.setdefault(t, 0)
            pathhits[t] += 1
    top = sorted(pathhits.items(), key=lambda kv: -kv[1])[:12]
    for t, c in top:
        print(f"  {c:4d}x  {t}")
    print(f"  distinct paths: {len(pathhits)}  "
          f"(the user's own machine paths appear in docs and tools; that is normal for a "
          f"reproduction repo but they do leak the local layout)")

    print()
    print("--- .gitignore coverage ---")
    gi = open(os.path.join(ROOT, ".gitignore"), encoding="utf-8").read()
    for pat, why in FORBIDDEN:
        key = pat.strip("^$").split("|")[0].replace("\\", "")
        print(f"  {'OK ' if key in gi else '?? '} {why:24s} (pattern like '{key}')")

    print()
    print("--- history ---")
    n = git("rev-list", "--count", "HEAD").strip()
    print(f"  commits: {n}")
    print("  branch : " + git("rev-parse", "--abbrev-ref", "HEAD").strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
