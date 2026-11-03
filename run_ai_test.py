#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the ai_function_enhancement.test statements against a local seekdb and
print the output of every statement, so it can be compared with the official
.result file."""
import subprocess
import sys
import re

TEST = "/workspace/seekdb/tools/deploy/mysql_test/test_suite/ai_function/t/ai_function_enhancement.test"
PORT = sys.argv[1] if len(sys.argv) > 1 else "2881"
USER = "root"
HOST = "127.0.0.1"


def statements(path):
    out = []
    buf = []
    for raw in open(path, encoding="utf-8", errors="replace").read().split("\n"):
        line = raw.rstrip()
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("--"):
            continue
        if re.match(r"^(connection|let|exec|disable_|enable_|shutdown|sleep|source)\b", s):
            continue
        buf.append(line)
        if s.endswith(";"):
            out.append("\n".join(buf).strip())
            buf = []
    if buf:
        out.append("\n".join(buf).strip())
    return out


def run(sql):
    p = subprocess.run(
        ["mysql", "-h" + HOST, "-P" + PORT, "-u" + USER, "-Dtest", "--batch", "--raw", "-e", sql],
        capture_output=True, text=True, timeout=300)
    return p.returncode, p.stdout, p.stderr


def main():
    for sql in statements(TEST):
        rc, out, err = run(sql)
        print("=" * 70)
        print("SQL:", " ".join(sql.split()))
        if rc != 0:
            print("ERROR rc=%d\n%s%s" % (rc, out, err))
        else:
            print(out.rstrip())
    print("=" * 70)
    print("DONE")


if __name__ == "__main__":
    main()
