#!/usr/bin/env python3
"""Adversarial stress matrix: hunt for process-death bugs the hidden cases hit.

Each workload runs the binary via the official protocol in a fresh data dir.
A workload FAILS if the process exits non-zero or closes stdout while a
response is still expected ("database exited before response" on the judge).
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

BINARY = sys.argv[1] if len(sys.argv) > 1 else "target/release/minivectordb"


def f32(v):
    return struct.unpack("<f", struct.pack("<f", v))[0]


class Db:
    def __init__(self, root, name):
        self.dir = Path(root) / name
        self.dir.mkdir(parents=True)
        self.proc = None
        self.seq = 0

    def start(self):
        self.proc = subprocess.Popen(
            [BINARY, "--data-dir", str(self.dir)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)
        line = self.proc.stdout.readline()
        assert json.loads(line) == {"type": "ready"}, f"bad ready: {line!r}"

    def sql(self, sql, mode="exact"):
        self.seq += 1
        self.proc.stdin.write(json.dumps(
            {"id": self.seq, "op": "execute", "sql": sql, "search_mode": mode}) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            rc = self.proc.poll()
            err = self.proc.stderr.read()
            raise RuntimeError(f"PROCESS DIED rc={rc}\nstderr tail: {err[-3000:]}")
        return json.loads(line)

    def shutdown(self):
        self.seq += 1
        self.proc.stdin.write(json.dumps({"id": self.seq, "op": "shutdown"}) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            rc = self.proc.poll()
            err = self.proc.stderr.read()
            raise RuntimeError(f"DIED AT SHUTDOWN rc={rc}\nstderr tail: {err[-3000:]}")
        self.proc.stdin.close()
        self.proc.wait(timeout=15)
        if self.proc.returncode != 0:
            raise RuntimeError(f"shutdown rc={self.proc.returncode}")

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()


def vec(dim, gen):
    return "[" + ",".join(f"{f32(gen.uniform(-1, 1)):.9g}" for _ in range(dim)) + "]"


def w_large_table_ivf(root):
    """50k rows x 128 dim, vector index, ann + exact TopK, restart mid-way."""
    import random
    rng = random.Random(1)
    db = Db(root, "big-ivf")
    db.start()
    db.sql("CREATE TABLE t (id BIGINT PRIMARY KEY, category VARCHAR(8), v VECTOR(128))")
    for base in range(0, 50000, 1000):
        values = ",".join(
            f"({pk},'c{pk%7}','{vec(128, rng)}')" for pk in range(base, base + 1000))
        r = db.sql("INSERT INTO t VALUES " + values)
        assert r.get("ok"), (base, r)
    db.sql("CREATE VECTOR INDEX iv ON t(v)")
    q = vec(128, rng)
    r = db.sql(f"SELECT id, L2_DISTANCE(v,'{q}') AS d FROM t WHERE category='c3' ORDER BY d LIMIT 10", "ann")
    assert r.get("ok") and len(r["rows"]) == 10, r
    r = db.sql(f"SELECT id FROM t ORDER BY id LIMIT 3")
    assert r.get("ok"), r
    db.shutdown()
    db.start()  # restart with big data
    r = db.sql(f"SELECT id, L2_DISTANCE(v,'{q}') AS d FROM t ORDER BY d LIMIT 5", "ann")
    assert r.get("ok") and len(r["rows"]) == 5, r
    db.shutdown()


def w_huge_dim(root):
    """VECTOR(1024) plus extreme f32 values."""
    db = Db(root, "huge-dim")
    db.start()
    db.sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(1024))")
    big = "[" + ",".join(["3.0e38"] * 1024) + "]"
    r = db.sql(f"INSERT INTO t VALUES (1,'{big}')")
    print("  huge-vector insert:", "ok" if r.get("ok") else r)
    q = "[" + ",".join(["1.0"] * 1024) + "]"
    r = db.sql(f"SELECT id, L2_DISTANCE(v,'{q}') AS d FROM t ORDER BY d LIMIT 1")
    print("  huge-vector distance:", "ok" if r.get("ok") else r)
    sub = "[" + ",".join(["1.4e-45"] * 1024) + "]"
    r = db.sql(f"INSERT INTO t VALUES (2,'{sub}')")
    assert r.get("ok"), r
    neg = "[" + ",".join(["-0.0"] * 1024) + "]"
    r = db.sql(f"INSERT INTO t VALUES (3,'{neg}')")
    assert r.get("ok"), r
    r = db.sql("SELECT id, L2_DISTANCE(v,'[0.0]') AS d FROM t ORDER BY d LIMIT 3")  # dim mismatch -> error not death
    assert r.get("ok") is False, r
    r = db.sql(f"SELECT v FROM t ORDER BY id")  # vector projection as JSON array
    assert r.get("ok"), r
    db.shutdown()


def w_mega_insert(root):
    """One INSERT statement with 20k rows."""
    import random
    rng = random.Random(2)
    db = Db(root, "mega-insert")
    db.start()
    db.sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(64))")
    values = ",".join(f"({pk},'{vec(64, rng)}')" for pk in range(20000))
    r = db.sql("INSERT INTO t VALUES " + values)
    assert r.get("ok") and r.get("affected_rows") == 20000, r
    r = db.sql("SELECT id, L2_DISTANCE(v,'[0.5]') AS d FROM t ORDER BY d LIMIT 10")  # dim mismatch
    assert r.get("ok") is False, r
    db.shutdown()


def w_many_tables(root):
    db = Db(root, "many-tables")
    db.start()
    for i in range(100):
        r = db.sql(f"CREATE TABLE t{i} (id BIGINT PRIMARY KEY, v VECTOR(4))")
        assert r.get("ok"), (i, r)
        r = db.sql(f"INSERT INTO t{i} VALUES (1,'[1,2,3,4]')")
        assert r.get("ok"), (i, r)
    for i in range(100):
        r = db.sql(f"CREATE VECTOR INDEX iv{i} ON t{i}(v)")
        assert r.get("ok"), (i, r)
    r = db.sql("SELECT id FROM t99 WHERE id=1")
    assert r.get("ok"), r
    db.shutdown()


def w_txn_churn(root):
    """Transactions mixed with DDL, updates of vectors, deletes, rollbacks."""
    import random
    rng = random.Random(3)
    db = Db(root, "txn-churn")
    db.start()
    db.sql("CREATE TABLE t (id BIGINT PRIMARY KEY, c VARCHAR(4), v VECTOR(8))")
    for base in range(0, 5000, 500):
        values = ",".join(f"({pk},'a','{vec(8, rng)}')" for pk in range(base, base + 500))
        db.sql("INSERT INTO t VALUES " + values)
    db.sql("BEGIN")
    for pk in range(0, 5000, 997):
        r = db.sql(f"UPDATE t SET v='{vec(8, rng)}', c='z' WHERE id={pk}")
        assert r.get("ok"), r
    r = db.sql("SELECT COUNT(*) FROM t")  # unsupported -> error
    assert r.get("ok") is False, r
    db.sql("ROLLBACK")
    db.sql("BEGIN")
    db.sql("DELETE FROM t WHERE id < 1000")
    db.sql("COMMIT")
    r = db.sql(f"SELECT id, L2_DISTANCE(v,'{vec(8, rng)}') AS d FROM t ORDER BY d LIMIT 10", "ann")
    assert r.get("ok") and len(r["rows"]) == 10, r
    db.sql("CREATE TABLE t2 (id BIGINT PRIMARY KEY, v VECTOR(8))")
    db.sql("BEGIN")
    db.sql("INSERT INTO t2 VALUES (1,'[0,0,0,0,0,0,0,0]')")
    db.sql("ROLLBACK")
    r = db.sql("SELECT id FROM t2")
    assert r.get("ok") and r["rows"] == [], r
    db.shutdown()


def w_boundary_sql(root):
    """Weird-but-legal SQL shapes: comments, semicolons, empty, unicode pk."""
    db = Db(root, "boundary-sql")
    db.start()
    db.sql("CREATE TABLE 你好 (id BIGINT PRIMARY KEY, 名字 VARCHAR(8), v VECTOR(2));")
    r = db.sql("INSERT INTO 你好 VALUES (-9223372036854775808, '最小值🚀', '[3,4]');")
    assert r.get("ok"), r
    r = db.sql("SELECT id, 名字 FROM 你好 WHERE id = -9223372036854775808")
    assert r.get("ok") and r["rows"] == [[-9223372036854775808, "最小值🚀"]], r
    r = db.sql("select id from 你好 order by id desc limit 1")
    assert r.get("ok"), r
    r = db.sql("SeLeCt iD FrOm 你好")
    assert r.get("ok"), r
    r = db.sql("SELECT id FROM 你好 WHERE id > -9223372036854775808 AND id < 0 OR 名字 = 'x'")
    assert r.get("ok"), r
    r = db.sql("UPDATE 你好 SET v='[0,0]' WHERE id=-9223372036854775808")
    assert r.get("ok"), r
    db.shutdown()


def w_crash_restart_cycle(root):
    """SIGKILL mid-write then repeated recovery + queries."""
    import random
    rng = random.Random(4)
    db = Db(root, "crash-cycle")
    db.start()
    db.sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(32))")
    for base in range(0, 3000, 500):
        values = ",".join(f"({pk},'{vec(32, rng)}')" for pk in range(base, base + 500))
        db.sql("INSERT INTO t VALUES " + values)
    db.proc.kill()  # hard kill, no shutdown
    db.proc.wait(timeout=10)
    db.start()
    r = db.sql(f"SELECT id, L2_DISTANCE(v,'{vec(32, rng)}') AS d FROM t ORDER BY d LIMIT 10")
    assert r.get("ok") and len(r["rows"]) == 10, r
    db.proc.kill()
    db.proc.wait(timeout=10)
    db.start()
    r = db.sql("SELECT id FROM t ORDER BY id LIMIT 1")
    assert r.get("ok"), r
    db.shutdown()


WORKLOADS = [w_large_table_ivf, w_huge_dim, w_mega_insert, w_many_tables,
             w_txn_churn, w_boundary_sql, w_crash_restart_cycle]


def main():
    root = tempfile.mkdtemp(prefix="stress-")
    failed = []
    try:
        for w in WORKLOADS:
            name = w.__name__
            try:
                w(root)
                print(f"PASS {name}")
            except RuntimeError as exc:
                print(f"FAIL {name}: {exc}")
                failed.append(name)
            except Exception as exc:
                print(f"ERROR {name}: {type(exc).__name__}: {exc}")
                failed.append(name)
    finally:
        if not failed:
            shutil.rmtree(root, ignore_errors=True)
        else:
            print(f"data kept: {root}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
