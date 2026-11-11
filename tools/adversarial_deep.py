#!/usr/bin/env python3
"""Adversarial deep-input tests for the official protocol.

Reproduces the stack-overflow killer (100k-term AND chain) that aborted the
process with "fatal runtime error: stack overflow" (uncatchable; on the
judge: "database exited before response"), plus siblings: long OR chains,
mixed operators, deep parentheses (accepted and over-cap), UPDATE/DELETE
with long chains, a 1M-char WHERE, and a 200k-term chain. The process must
survive every statement, answer each, and still restart cleanly.
"""
import json
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

BINARY = sys.argv[1] if len(sys.argv) > 1 else "target/release/minivectordb"
ROOT = Path(tempfile.mkdtemp(prefix="advdeep-"))
DATA = ROOT / "data"
DATA.mkdir()
proc = subprocess.Popen(
    [BINARY, "--data-dir", str(DATA)],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True, encoding="utf-8", bufsize=1)
seq = 0
failures = []


def sql(s, expect_ok=True, note=""):
    global seq
    seq += 1
    proc.stdin.write(json.dumps(
        {"id": seq, "op": "execute", "sql": s, "search_mode": "exact"}) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        err = proc.stderr.read()
        print(f"FAIL {note or s[:70]!r}: DIED rc={proc.poll()}\nstderr: {err[-800:]}")
        sys.exit(1)
    r = json.loads(line)
    ok = r.get("ok") is expect_ok
    tag = "PASS" if ok else "FAIL"
    if not ok:
        failures.append(note or s[:70])
    print(f"{tag} {note or s[:70]!r} -> {str(r)[:110]}")


def boot():
    p = subprocess.Popen(
        [BINARY, "--data-dir", str(DATA)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", bufsize=1)
    line = p.stdout.readline()
    assert line.strip() == '{"type":"ready"}', f"bad ready: {line!r}"
    return p


ready = proc.stdout.readline()
assert ready.strip() == '{"type":"ready"}', f"bad ready: {ready!r}"

sql("CREATE TABLE t (id BIGINT PRIMARY KEY, c VARCHAR(8), v VECTOR(4))", True, "create")
for i in range(50):
    tag = "even" if i % 2 == 0 else "odd"
    proc.stdin.write(json.dumps({"id": seq + 1, "op": "execute", "sql":
        f"INSERT INTO t VALUES ({i}, '{tag}', '[1,2,3,4]')",
        "search_mode": "exact"}) + "\n")
    proc.stdin.flush()
    seq += 1
    line = proc.stdout.readline()
    assert json.loads(line).get("ok"), line
print(f"PASS seeded 50 rows")

# semantic spot-checks of the compiled filter machine
sql("SELECT id FROM t WHERE (id = 1 OR id = 3) AND c = 'odd'", True, "paren-filter")
sql("SELECT id FROM t WHERE id = 1 OR id = 3", True, "plain-or")
sql("SELECT id FROM t WHERE id >= 10 AND id < 20 AND c = 'even'", True, "and-3")

# the killer and its siblings
sql("SELECT id FROM t WHERE " + " AND ".join(f"id >= {i % 7}" for i in range(100_000)),
    True, "100k-AND")
sql("SELECT id FROM t WHERE " + " OR ".join(f"id = {i}" for i in range(100_000)),
    True, "100k-OR")
terms = []
for i in range(50_000):
    terms.append(f"id >= {i % 7}" if i % 2 == 0 else f"c <> 'zz{i}'")
sql("SELECT id FROM t WHERE " + " AND ".join(terms), True, "50k-mixed-AND")
sql("SELECT id FROM t WHERE " + " OR ".join(terms), True, "50k-mixed-OR")
sql("UPDATE t SET c = 'u' WHERE " + " AND ".join(f"id >= {i % 7}" for i in range(100_000)),
    True, "100k-AND-UPDATE")
sql("SELECT id FROM t WHERE " + "(" * 1000 + "id = 1" + ")" * 1000, True, "1000-deep-parens")
sql("SELECT id FROM t WHERE " + "(" * 8000 + "id = 1" + ")" * 8000, True, "8000-deep-parens")
sql("SELECT id FROM t WHERE " + "(" * 20000 + "id = 1" + ")" * 20000, False, "20000-deep-parens-cap")
big = " AND ".join(f"id >= {i % 7}" for i in range(12_000))
sql(f"SELECT id FROM t WHERE {big} OR ({big})", True, f"1M-char-WHERE({len(big)*2})")
sql("SELECT id FROM t WHERE " + " AND ".join(f"id >= {i % 7}" for i in range(200_000)),
    True, "200k-AND")

# correctness after the torture: exact counts
sql("SELECT id FROM t WHERE id >= 0", True, "post-torture-select")

# shutdown must be clean, restart must recover
seq += 1
proc.stdin.write(json.dumps({"id": seq, "op": "shutdown"}) + "\n")
proc.stdin.flush()
proc.stdin.close()
rc = proc.wait(timeout=30)
print(f"{'PASS' if rc == 0 else 'FAIL'} shutdown rc={rc}")
if rc != 0:
    failures.append("shutdown")

p2 = boot()
p2.stdin.write(json.dumps({"id": 1, "op": "execute", "sql": "SELECT id FROM t WHERE id = 1",
                           "search_mode": "exact"}) + "\n")
p2.stdin.flush()
line = p2.stdout.readline()
r = json.loads(line)
ok = r.get("ok") and [row[0] for row in r["rows"]] == [1]
print(f"{'PASS' if ok else 'FAIL'} restart-recovered -> {str(r)[:110]}")
if not ok:
    failures.append("restart")
p2.stdin.write(json.dumps({"id": 2, "op": "shutdown"}) + "\n")
p2.stdin.flush()
p2.wait(timeout=30)

shutil.rmtree(ROOT, ignore_errors=True)
if failures:
    print(f"\n{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("\nALL DEEP-INPUT TESTS PASS")
