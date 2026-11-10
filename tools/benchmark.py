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

    # Full Recall@10 truth for every query (not just a sample): a mode that
    # cannot hold 0.95 must fail loudly here instead of hiding in metrics.
    truth = {}
    for index, (query, filtered, _) in enumerate(queries):
        candidates = [row for row in rows if not filtered or row[1] == 0]
        ranked = sorted((l2(vector, query), pk) for pk, _, vector in candidates)[:10]
        truth[index] = {pk for _, pk in ranked}

    data_dir = Path(tempfile.mkdtemp(prefix="minivectordb-bench-"))
    proc = subprocess.Popen(
        [str(args.binary.resolve()), "--data-dir", str(data_dir)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )

    # Official protocol: ready object first, then JSONL requests.
    ready = proc.stdout.readline()
    assert json.loads(ready) == {"type": "ready"}, ready
    seq = 0

    def request(sql):
        nonlocal seq
        seq += 1
        proc.stdin.write(json.dumps(
            {"id": seq, "op": "execute", "sql": sql, "search_mode": args.mode},
            ensure_ascii=False) + "\n")
        proc.stdin.flush()
        response = proc.stdout.readline()
        if not response:
            raise RuntimeError(f"database exited: {proc.poll()}")
        value = json.loads(response)
        if value.get("ok") is not True:
            raise RuntimeError((sql[:200], value))
        return value

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

    query_start = time.perf_counter()
    hits = []
    for index, (_, filtered, sql) in enumerate(queries):
        result = request(sql)
        ids = [row[0] for row in result["rows"]]
        assert len(ids) == 10 and len(set(ids)) == 10
        if filtered:
            assert all(pk % 2 == 0 for pk in ids)
        hits.append(len(set(ids) & truth[index]) / 10)
    query_seconds = time.perf_counter() - query_start
    recall_unfiltered = sum(hits[0::2]) / (len(hits) // 2)
    recall_filtered = sum(hits[1::2]) / (len(hits) // 2)
    if args.mode == "ann":
        assert recall_unfiltered >= 0.95 and recall_filtered >= 0.95, (
            "recall below 0.95", recall_unfiltered, recall_filtered)
    status = Path(f"/proc/{proc.pid}/status").read_text()
    memory_status = {
        key: int(line.split()[1]) // 1024
        for key in ("VmHWM", "VmRSS")
        for line in status.splitlines()
        if line.startswith(key + ":")
    }
    # Official shutdown: respond, exit 0, nothing further on stdout.
    seq += 1
    proc.stdin.write(json.dumps({"id": seq, "op": "shutdown"}) + "\n")
    proc.stdin.flush()
    shutdown = json.loads(proc.stdout.readline())
    assert shutdown == {"id": seq, "ok": True}, shutdown
    proc.wait(timeout=20)
    assert proc.returncode == 0, proc.returncode
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
        "sample_recall_at_10": round(sum(hits) / len(hits), 6),
        "recall_at_10_unfiltered": round(recall_unfiltered, 6),
        "recall_at_10_filtered": round(recall_filtered, 6),
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
