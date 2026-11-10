#!/usr/bin/env python3
"""Generate a big-load official-protocol workload as JSONL on stdout."""
import json
import random
import struct
import sys

rows = int(sys.argv[1]) if len(sys.argv) > 1 else 300000
batch = 500
rng = random.Random(600081)


def f32(v):
    return struct.unpack("<f", struct.pack("<f", v))[0]


def vec(dim):
    return "[" + ",".join(f"{f32(rng.uniform(-1, 1)):.8g}" for _ in range(dim)) + "]"


reqs = []
seq = 0
def add(sql, mode="exact"):
    global seq
    seq += 1
    reqs.append({"id": seq, "op": "execute", "sql": sql, "search_mode": mode})

add("CREATE TABLE t (id BIGINT PRIMARY KEY, category VARCHAR(4), v VECTOR(128))")
for base in range(0, rows, batch):
    vals = ",".join(
        f"({pk},'{'even' if pk % 2 == 0 else 'odd'}','{vec(128)}')"
        for pk in range(base, min(base + batch, rows)))
    add("INSERT INTO t VALUES " + vals)
add("CREATE VECTOR INDEX iv ON t(v)")
add(f"SELECT id, L2_DISTANCE(v,'{vec(128)}') AS d FROM t WHERE category='even' ORDER BY d LIMIT 10")
add(f"SELECT id, L2_DISTANCE(v,'{vec(128)}') AS d FROM t ORDER BY d LIMIT 10", "ann")
add("SELECT id FROM t WHERE id = 0")
seq += 1
reqs.append({"id": seq, "op": "shutdown"})
for r in reqs:
    print(json.dumps(r, ensure_ascii=False))
sys.stderr.write(f"generated {len(reqs)} requests for {rows} rows\n")
