#!/usr/bin/env python3
"""Panic hunt: with panic=unwind the catch_unwind in the official driver
turns any residual panic into an internal_error RESPONSE instead of an
abort. Feed pathological inputs; flag every internal_error (a real panic)
and any process death. Under the old panic=abort every one of these killed
the process on the platform ("database exited before response").
"""
import json
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

BINARY = sys.argv[1] if len(sys.argv) > 1 else "target/release/minivectordb"
ROOT = Path(tempfile.mkdtemp(prefix="panichunt-"))
DATA = ROOT / "data"
DATA.mkdir()
proc = subprocess.Popen(
    [BINARY, "--data-dir", str(DATA)],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True, encoding="utf-8", bufsize=1)
seq = 0
panics = []
bad = []

I64_MAX = 9223372036854775807
I64_MIN = -9223372036854775808


def sql(s, mode="exact", note=None):
    global seq
    seq += 1
    proc.stdin.write(json.dumps(
        {"id": seq, "op": "execute", "sql": s, "search_mode": mode}) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        err = proc.stderr.read()
        print(f"DEATH on {note or s[:90]!r} rc={proc.poll()}\nstderr: {err[-600:]}")
        sys.exit(2)
    r = json.loads(line)
    code = (r.get("error") or {}).get("code", "")
    if code == "internal_error":
        panics.append(note or s[:90])
        print(f"PANIC {note or s[:90]!r} -> internal_error")
    return r


ready = proc.stdout.readline()
assert ready.strip() == '{"type":"ready"}'

sql("CREATE TABLE t (id BIGINT PRIMARY KEY, c VARCHAR(8), v VECTOR(4))")
sql("CREATE TABLE z (id BIGINT PRIMARY KEY, s VARCHAR(2), v VECTOR(0))")
sql("CREATE TABLE big (id BIGINT PRIMARY KEY, c VARCHAR(10000), v VECTOR(1024))")

# --- numeric extremes ---
for val, note in [
    (str(I64_MAX), "i64max"), (str(I64_MIN), "i64min"),
    ("9223372036854775808", "i64max+1"), ("-9223372036854775809", "i64min-1"),
    ("18446744073709551616", "2^64"), ("9" * 500, "500digits"),
    ("0000000000000001", "leading-zeros"), ("+7", "plus-sign"), ("-0", "minus-zero"),
    ("0x10", "hex"), ("1e3", "sci-int"), ("--5", "double-minus"),
]:
    r = sql(f"INSERT INTO t VALUES ({val}, 'x', '[1,2,3,4]')", note=f"int:{note}")
    if val in (str(I64_MAX), str(I64_MIN), "0000000000000001"):
        pass  # may legitimately succeed
    if note in ("i64max+1", "i64min-1", "2^64", "500digits", "hex", "sci-int",
                "double-minus", "plus-sign") and r.get("ok"):
        bad.append(f"int:{note} expected error, got ok")
sql(f"SELECT id FROM t WHERE id = {I64_MAX}", note="where-i64max")
sql(f"SELECT id FROM t WHERE id > {I64_MIN} LIMIT 3", note="where-i64min")
sql("SELECT id FROM t ORDER BY id DESC LIMIT 2", note="order-desc")

# --- vector literals ---
for vec, note in [
    ("[1e400]", "inf"), ("[nan]", "nan"), ("[inf]", "inf-lit"),
    ("[-inf]", "neg-inf"), ("[1e-400]", "subnormal"), ("[]", "empty"),
    ("[ ]", "space-empty"), ("[1,]", "trailing-comma"), ("[,1]", "lead-comma"),
    ("[1 2]", "space-sep"), ("[+1,2,3,4]", "plus"), ("[.5,2,3,4]", "dot5"),
    ("[5.,2,3,4]", "dot-end"), ("[1.5e2,2,3,4]", "sci"), ("[0.1,0.2,0.3,0.4]", "ok"),
    ("[1,2,3]", "dim3-on-4"), ("[1,2,3,4,5]", "dim5-on-4"),
    ("[" + ",".join(["1"] * 1024) + "]", "dim1024-on-4"),
]:
    sql(f"INSERT INTO t VALUES (900, 'x', '{vec}')", note=f"vec:{note}")
sql("INSERT INTO z VALUES (1, '', '[]')", note="vec0-empty")
sql("INSERT INTO z VALUES (2, 'ab', '[0]')", note="vec0-nonempty")

# --- strings ---
for s_val, note in [("", "empty"), ("''", "escaped-quote"), ("中中", "2-cjk"),
                    ("a中b", "mixed"), ("😂", "emoji"), ("a''b", "quote-mid")]:
    sql(f"INSERT INTO z VALUES (10{len(note)}, '{s_val}', '[]')", note=f"str:{note}")
sql("INSERT INTO big VALUES (1, '" + "x" * 9999 + "', '[" + ",".join(["0.5"] * 1024) + "]')",
    note="big-row")
sql("SELECT id FROM big WHERE c = '" + "x" * 9999 + "'", note="big-where")

# --- SELECT / ORDER BY shapes ---
v4 = "'[1,2,3,4]'"
sql(f"SELECT id, L2_DISTANCE(v, {v4}) AS d FROM t ORDER BY d LIMIT 3", note="l2-order")
sql(f"SELECT id, L2_DISTANCE(v, {v4}) FROM t LIMIT 3", note="l2-noalias")
sql(f"SELECT id FROM t ORDER BY L2_DISTANCE(v, {v4}) LIMIT 3", note="l2-hidden")
sql(f"SELECT id, L2_DISTANCE(v, {v4}) AS d FROM t ORDER BY d DESC LIMIT 2", note="l2-desc")
sql(f"SELECT id, L2_DISTANCE(v, {v4}) AS d FROM t LIMIT 2", note="l2-limit-only")
sql(f"SELECT id, L2_DISTANCE(v, {v4}) AS d FROM t ORDER BY d", note="l2-order-nolimit")
sql("SELECT * FROM t LIMIT 2", note="select-star")
sql("SELECT id FROM t ORDER BY id ASC, c DESC LIMIT 4", note="multi-order")
sql("SELECT id FROM t LIMIT 0", note="limit0")
sql("SELECT id FROM t LIMIT 99999999999999999999", note="limit-overflow")
sql("SELECT id FROM t LIMIT -1", note="limit-neg")
sql("SeLeCt Id FrOm t WhErE id = 1", note="weird-case")
sql("SELECT id FROM t WHERE c = 'a;b'", note="semicolon-in-string")
sql("SELECT id FROM t WHERE (id = 1 OR id = 2) AND c LIKE 'x'", note="like-op")

# --- transactions ---
sql("BEGIN", note="begin")
sql("BEGIN", note="nested-begin")
sql("ROLLBACK", note="rollback")
sql("COMMIT", note="commit-without-begin")
sql("BEGIN", note="begin2")
sql("INSERT INTO t VALUES (999, 'tx', '[9,9,9,9]')", note="tx-insert")
sql("COMMIT", note="tx-commit")
sql("UPDATE t SET id = 5 WHERE id = 1", note="pk-update")
sql("UPDATE t SET c = 'u' WHERE id = 1", note="update")
sql("UPDATE t SET v = '[1,1,1,1]' WHERE c = 'u'", note="update-vec")
sql("DELETE FROM t WHERE id = 999", note="delete")
sql("DELETE FROM t", note="delete-all")

# --- DDL edges ---
sql("CREATE TABLE t (id BIGINT PRIMARY KEY)", note="dup-table")
sql("CREATE TABLE bad1 (id BIGINT, c VARCHAR(1))", note="no-pk")
sql("CREATE TABLE bad2 (id BIGINT PRIMARY KEY, v1 VECTOR(2), v2 VECTOR(2))", note="two-vec")
sql("CREATE TABLE bad3 (c VARCHAR(1) PRIMARY KEY, id BIGINT)", note="wrong-pk-type")
sql("CREATE TABLE n0 (id BIGINT PRIMARY KEY, v VECTOR(0))", note="vec0-table")
sql("CREATE INDEX ix ON t (id)", note="scalar-index")
sql("CREATE VECTOR INDEX iv ON t (v)", note="vec-index")
sql("CREATE VECTOR INDEX iv ON t (v)", note="dup-vec-index")
sql("CREATE VECTOR INDEX iv2 ON nosuch (v)", note="index-no-table")

# --- ann mode paths (IVF present now) ---
sql(f"SELECT id, L2_DISTANCE(v, {v4}) AS d FROM t ORDER BY d LIMIT 3", mode="ann",
    note="ann-l2")
sql("SELECT id FROM t WHERE id >= 0 LIMIT 2", mode="ann", note="ann-filter")

# --- misc ---
sql("INSERT INTO t (id, c, v) VALUES (1, 'a', '[1,2,3,4]')", note="full-collist")
sql("INSERT INTO t (id) VALUES (2)", note="partial-collist")
sql("INSERT INTO nosuch VALUES (1)", note="no-table")
sql("SELECT id FROM nosuch", note="select-no-table")
sql("DROP TABLE t", note="drop")
sql("ALTER TABLE t ADD c2 VARCHAR(1)", note="alter")
sql("SELECT COUNT(*) FROM t", note="count")
sql("SELECT id FROM t WHERE id = NULL", note="null")
sql("", note="empty-sql")
sql("   ", note="whitespace-sql")
sql(";", note="semicolon-only")

# shutdown + restart check
seq += 1
proc.stdin.write(json.dumps({"id": seq, "op": "shutdown"}) + "\n")
proc.stdin.flush()
proc.stdin.close()
rc = proc.wait(timeout=30)
p2 = subprocess.Popen([BINARY, "--data-dir", str(DATA)], stdin=subprocess.PIPE,
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                      text=True, encoding="utf-8", bufsize=1)
line = p2.stdout.readline()
ok = line.strip() == '{"type":"ready"}'
print(f"{'PASS' if ok else 'FAIL'} restart-ready")
p2.stdin.write(json.dumps({"id": 1, "op": "execute", "sql": "SELECT id FROM t LIMIT 1",
                           "search_mode": "exact"}) + "\n")
p2.stdin.flush()
print("restart-query:", p2.stdout.readline().strip()[:80])
p2.stdin.write(json.dumps({"id": 2, "op": "shutdown"}) + "\n")
p2.stdin.flush()
p2.wait(timeout=30)

shutil.rmtree(ROOT, ignore_errors=True)
print(f"\npanics found: {len(panics)}")
for p in panics:
    print("  -", p)
if bad:
    print("semantic surprises:")
    for b in bad:
        print("  -", b)
print("DONE", "CLEAN" if not panics and not bad else "ISSUES")
