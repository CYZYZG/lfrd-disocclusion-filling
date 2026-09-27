"""Tiny assertion reporter used by every step script (`[CHECK] ...` output).

Usage
-----
    rep = Reporter("step1")
    rep.check("marked pixels > 0", int(er.sum()) > 0, f"{int(er.sum())} px")
    rep.finish()          # prints the summary, raises SystemExit(1) on failure
"""
import sys

from . import io_utils


class Reporter:
    def __init__(self, title, log_path=None):
        self.title = title
        self.lines = []
        self.n_pass = 0
        self.n_fail = 0
        self.log_path = log_path

    def log(self, msg=""):
        s = str(msg)
        print(s)
        self.lines.append(s)
        return s

    def info(self, msg):
        return self.log(msg)

    def check(self, name, ok, detail=""):
        ok = bool(ok)
        tag = "PASS" if ok else "FAIL"
        line = f"[CHECK][{tag}] {self.title} :: {name}" + (f"  ({detail})" if detail else "")
        self.lines.append(line)
        print(line)
        if ok:
            self.n_pass += 1
        else:
            self.n_fail += 1
        return ok

    def summary(self):
        return f"[CHECK] {self.title}: {self.n_pass} passed, {self.n_fail} failed"

    def finish(self, save=True, extra=None):
        if extra:
            self.log(extra)
        self.log(self.summary())
        if save and self.log_path:
            io_utils.write_text(self.log_path, "\n".join(self.lines) + "\n")
        if self.n_fail:
            sys.exit(1)
        return self.n_pass
