#!/usr/bin/env python3
"""ANN/IVF hammer: the public cases never exercise search_mode:"ann", so the
IVF probe path is our least-tested code. Flag internal_error responses (a
real panic — deadly under panic=abort, which all three failed submissions
ran with) and any process death.
"""
import json
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

BINARY = sys.argv[1] if len(sys.argv) > 1 else "target/release/minivectordb"
ROOT = Path(tempfile.mkdtemp(prefix="annhunt-"))
DATA = ROOT / "data"
DATA.mkdir()
proc = subprocess.Popen(
    [BINARY, "--data-dir", str(DATA)],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True, encoding="utf-8", bufsize=1)
seq = 0
issues = []


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
    if (r.get("error") or {}).get("code") == "internal_error":
        issues.append(note or s[:90])
        print(f"PANIC {note or s[:90]!r}")
    return r


def restart():
    global proc, seq
    seq += 1
    proc.stdin.write(json.dumps({"id": seq, "op": "shutdown"}) + "\n")
    proc.stdin.flush()
    proc.stdin.close()
    rc = proc.wait(timeout=30)
    assert rc == 0, f"shutdown rc={rc}"
    proc = subprocess.Popen(
        [BINARY, "--data-dir", str(DATA)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", bufsize=1)
    line = proc.stdout.readline()
    assert line.strip() == '{"type":"ready"}', f"bad ready {line!r}"


def vec(rng_dims, seed=1, mag=1.0):
    import random
    rng = random.Random(seed)
    return "'[" + ",".join(f"{rng.uniform(-mag, mag):.6g}" for _ in range(rng_dims)) + "]'"


assert proc.stdout.readline().strip() == '{"type":"ready"}'

# empty table ann
sql("CREATE TABLE e (id BIGINT PRIMARY KEY, c VARCHAR(4), v VECTOR(4))")
sql(f"SELECT id, L2_DISTANCE(v, {vec(4)}) AS d FROM e ORDER BY d LIMIT 3", "ann", "ann-empty")
sql("CREATE VECTOR INDEX ei ON e (v)", note="index-empty-table")
sql(f"SELECT id, L2_DISTANCE(v, {vec(4)}) AS d FROM e ORDER BY d LIMIT 3", "ann", "ann-empty-indexed")

# 1-row + ties
sql(f"INSERT INTO e VALUES (1, 'a', {vec(4, 7)})")
sql(f"SELECT id, L2_DISTANCE(v, {vec(4)}) AS d FROM e ORDER BY d LIMIT 5", "ann", "ann-one-row-topk-gt")
sql(f"INSERT INTO e VALUES (2, 'a', '[1,1,1,1]'), (3, 'a', '[1,1,1,1]'), (4, 'a', '[1,1,1,1]')")
sql(f"SELECT id, L2_DISTANCE(v, '[1,1,1,1]') AS d FROM e ORDER BY d LIMIT 3", "ann", "ann-ties")

# filter matching nothing (probe-set growth path)
sql(f"SELECT id, L2_DISTANCE(v, {vec(4)}) AS d FROM e WHERE c = 'zzz' ORDER BY d LIMIT 2",
    "ann", "ann-filter-empty")
sql(f"SELECT id, L2_DISTANCE(v, {vec(4)}) AS d FROM e WHERE id = 4 AND c = 'a' ORDER BY d LIMIT 1",
    "ann", "ann-filter-narrow")

# delete-all then ann
sql("DELETE FROM e")
sql(f"SELECT id, L2_DISTANCE(v, {vec(4)}) AS d FROM e ORDER BY d LIMIT 3", "ann", "ann-after-delete-all")

# incremental index maintenance: index first, bulk insert after
sql("CREATE TABLE g (id BIGINT PRIMARY KEY, c VARCHAR(4), v VECTOR(8))")
sql("CREATE VECTOR INDEX gi ON g (v)")
for batch in range(6):
    vals = ",".join(
        f"({batch * 40 + i}, '{'ab' if i % 2 else 'cd'}', {vec(8, batch * 100 + i)})"
        for i in range(40))
    sql(f"INSERT INTO g VALUES {vals}", note=f"bulk-{batch}")
sql(f"SELECT id, L2_DISTANCE(v, {vec(8, 999)}) AS d FROM g ORDER BY d LIMIT 10", "ann", "ann-bulk")
sql(f"SELECT id, L2_DISTANCE(v, {vec(8, 999)}) AS d FROM g WHERE c = 'ab' ORDER BY d LIMIT 7",
    "ann", "ann-bulk-filtered")
# exact/ann agreement on same query
r_exact = sql(f"SELECT id, L2_DISTANCE(v, {vec(8, 999)}) AS d FROM g ORDER BY d LIMIT 10", "exact")
r_ann = sql(f"SELECT id, L2_DISTANCE(v, {vec(8, 999)}) AS d FROM g ORDER BY d LIMIT 10", "ann")
if [row[0] for row in r_exact["rows"]] != [row[0] for row in r_ann["rows"]]:
    issues.append("ann-vs-exact id mismatch (rank/float issue)")

# update vectors under index, then ann
sql("UPDATE g SET v = '[7,7,7,7,7,7,7,7]' WHERE id < 5", note="update-vec-indexed")
sql(f"SELECT id, L2_DISTANCE(v, '[7,7,7,7,7,7,7,7]') AS d FROM g ORDER BY d LIMIT 4", "ann",
    "ann-after-update")
# delete many under index
sql("DELETE FROM g WHERE id >= 200", note="bulk-delete-indexed")
sql(f"SELECT id, L2_DISTANCE(v, {vec(8, 555)}) AS d FROM g ORDER BY d LIMIT 5", "ann", "ann-after-del")

# index created AFTER data (build path)
sql("CREATE TABLE h (id BIGINT PRIMARY KEY, c VARCHAR(4), v VECTOR(4))")
vals = ",".join(f"({i}, 'q', {vec(4, 5000 + i)})" for i in range(120))
sql(f"INSERT INTO h VALUES {vals}", note="h-load")
sql("CREATE VECTOR INDEX hi ON h (v)", note="index-after-data")
sql(f"SELECT id, L2_DISTANCE(v, {vec(4, 777)}) AS d FROM h ORDER BY d LIMIT 6", "ann", "ann-built-index")
sql(f"SELECT id, L2_DISTANCE(v, {vec(4, 777)}) AS d FROM h WHERE c='q' ORDER BY d LIMIT 6", "ann",
    "ann-built-filtered")

# ann without any index (fallback)
sql("CREATE TABLE ni (id BIGINT PRIMARY KEY, v VECTOR(4))")
sql(f"INSERT INTO ni VALUES (1, {vec(4, 3)}), (2, {vec(4, 4)})")
sql(f"SELECT id, L2_DISTANCE(v, {vec(4, 5)}) AS d FROM ni ORDER BY d LIMIT 1", "ann", "ann-no-index")

# dim-0 vectors
sql("CREATE TABLE z (id BIGINT PRIMARY KEY, v VECTOR(0))")
sql("INSERT INTO z VALUES (1, '[]'), (2, '[]')")
sql("SELECT id, L2_DISTANCE(v, '[]') AS d FROM z ORDER BY d LIMIT 2", "ann", "ann-dim0")
sql("SELECT id, L2_DISTANCE(v, '[]') AS d FROM z ORDER BY d LIMIT 2", "exact", "exact-dim0")

# restart: durable index, rebuild, ann again
restart()
sql(f"SELECT id, L2_DISTANCE(v, {vec(8, 999)}) AS d FROM g ORDER BY d LIMIT 9", "ann",
    "ann-after-restart")
sql(f"SELECT id, L2_DISTANCE(v, {vec(4, 777)}) AS d FROM h ORDER BY d LIMIT 6", "ann",
    "ann-h-after-restart")
r_exact = sql(f"SELECT id, L2_DISTANCE(v, {vec(8, 999)}) AS d FROM g ORDER BY d LIMIT 9", "exact")
r_ann = sql(f"SELECT id, L2_DISTANCE(v, {vec(8, 999)}) AS d FROM g ORDER BY d LIMIT 9", "ann")
if [row[0] for row in r_exact["rows"]] != [row[0] for row in r_ann["rows"]]:
    issues.append("ann-vs-exact mismatch after restart")

# ann with non-distance ORDER BY / plain select in ann mode
sql("SELECT id FROM g WHERE id < 100 ORDER BY id DESC LIMIT 3", "ann", "ann-mode-plain")
sql("SELECT * FROM g LIMIT 2", "ann", "ann-mode-star")

# huge-magnitude vectors (f32 accumulator overflow corner)
sql("CREATE TABLE m (id BIGINT PRIMARY KEY, v VECTOR(4))")
sql("INSERT INTO m VALUES (1, '[3e38,3e38,3e38,3e38]'), (2, '[-3e38,-3e38,-3e38,-3e38]')")
sql("SELECT id, L2_DISTANCE(v, '[3e38,3e38,3e38,3e38]') AS d FROM m ORDER BY d LIMIT 2", "ann",
    "ann-huge-mag")
sql("SELECT id, L2_DISTANCE(v, '[3e38,3e38,3e38,3e38]') AS d FROM m ORDER BY d LIMIT 2", "exact",
    "exact-huge-mag")

seq += 1
proc.stdin.write(json.dumps({"id": seq, "op": "shutdown"}) + "\n")
proc.stdin.flush()
proc.stdin.close()
rc = proc.wait(timeout=30)
print(f"shutdown rc={rc}")
shutil.rmtree(ROOT, ignore_errors=True)
print(f"\nissues: {len(issues)}")
for i in issues:
    print("  -", i)
print("DONE", "CLEAN" if not issues else "ISSUES")
