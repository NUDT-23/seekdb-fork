#!/usr/bin/env python3
"""Small local benchmark matching the published 10k x 128 / 1000-query shape."""

import argparse
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
import struct
import subprocess
import tempfile
import time


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def vec_literal(vector):
    return "[" + ",".join(repr(x) for x in vector) + "]"


def l2(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--mode", choices=["exact", "ann"], default="exact")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    rng = random.Random(600081)
    rows = []
    for pk in range(args.rows):
        rows.append((pk, pk % 2, [f32(rng.uniform(-1, 1)) for _ in range(args.dim)]))
    queries = []
    for index in range(args.queries):
        filtered = index % 2 == 1
        query = [f32(rng.uniform(-1, 1)) for _ in range(args.dim)]
        sql = "SELECT id, L2_DISTANCE(v,'%s') AS distance FROM t%s ORDER BY distance LIMIT 10" % (
            vec_literal(query),
            " WHERE category='even'" if filtered else "",
        )
        queries.append((query, filtered, sql))

    truth = {}
    for index in list(range(5)) + list(range(5, 10)):
        query, filtered, _ = queries[index]
        candidates = [row for row in rows if not filtered or row[1] == 0]
        expected = sorted((l2(vector, query), pk) for pk, _, vector in candidates)[:10]
        truth[index] = [pk for _, pk in expected]

    data_dir = Path(tempfile.mkdtemp(prefix="minivectordb-bench-"))
    proc = subprocess.Popen(
        [str(args.binary.resolve()), "--data-dir", str(data_dir)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )

    def request(sql):
        proc.stdin.write(sql + "\n")
        proc.stdin.flush()
        response = proc.stdout.readline()
        if not response:
            raise RuntimeError(f"database exited: {proc.poll()}")
        value = json.loads(response)
        if not value.get("ok"):
            raise RuntimeError((sql[:200], value))
        return value

    assert json.loads(proc.stdout.readline()).get("ready") is True
    request("CREATE TABLE t (id BIGINT PRIMARY KEY, category VARCHAR(4), v VECTOR(%d))" % args.dim)
    write_start = time.perf_counter()
    for base in range(0, len(rows), 100):
        values = []
        for pk, category, vector in rows[base:base + 100]:
            values.append("(%d,'%s','%s')" % (pk, "even" if category == 0 else "odd", vec_literal(vector)))
        result = request("INSERT INTO t VALUES " + ",".join(values))
        assert result.get("affected_rows") == len(values)
    write_seconds = time.perf_counter() - write_start
    request("CREATE VECTOR INDEX t_v ON t(v)")
    if args.mode == "ann":
        request("--mode ann")

    query_start = time.perf_counter()
    recalls = []
    for index, (_, filtered, sql) in enumerate(queries):
        result = request(sql)
        ids = [row[0] for row in result["rows"]]
        assert len(ids) == 10 and len(set(ids)) == 10
        if filtered:
            assert all(pk % 2 == 0 for pk in ids)
        if index in truth:
            expected = truth[index]
            recalls.append(len(set(ids) & set(expected)) / 10)
    query_seconds = time.perf_counter() - query_start
    status = Path(f"/proc/{proc.pid}/status").read_text()
    memory_status = {
        key: int(line.split()[1]) // 1024
        for key in ("VmHWM", "VmRSS")
        for line in status.splitlines()
        if line.startswith(key + ":")
    }
    proc.terminate()
    proc.wait(timeout=20)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    disk_bytes = sum(path.stat().st_size for path in data_dir.rglob("*") if path.is_file())
    metrics = {
        "rows": args.rows,
        "dim": args.dim,
        "queries": args.queries,
        "mode": args.mode,
        "write_seconds": round(write_seconds, 6),
        "query_seconds": round(query_seconds, 6),
        "queries_per_second": round(args.queries / query_seconds, 3),
        "sample_recall_at_10": round(sum(recalls) / len(recalls), 6),
        "max_rss_mib": round(usage.ru_maxrss / 1024, 3),
        "db_vm_hwm_mib": memory_status.get("VmHWM"),
        "db_vm_rss_mib": memory_status.get("VmRSS"),
        "data_file_bytes": disk_bytes,
        "data_dir": str(data_dir),
    }
    print(json.dumps(metrics, indent=2))
    if not args.keep:
        shutil.rmtree(data_dir)


if __name__ == "__main__":
    main()
