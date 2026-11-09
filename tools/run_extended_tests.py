#!/usr/bin/env python3
"""Extended functional, crash-recovery and exact-vs-ANN checks.

The expected TopK results are computed independently in Python from float32
values; no expected output is shared with the Rust implementation.
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
import tempfile


class Db:
    def __init__(self, binary, directory, env=None):
        self.binary = str(binary)
        self.directory = Path(directory)
        self.env = env or {}
        self.proc = None

    def start(self):
        env = os.environ.copy()
        env.update(self.env)
        self.proc = subprocess.Popen(
            [self.binary, "--data-dir", str(self.directory)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env,
        )
        ready = json.loads(self.proc.stdout.readline())
        assert ready.get("ready") is True, ready
        return self

    def sql(self, text):
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        assert line, (text, self.proc.poll())
        return json.loads(line)

    def ok(self, text, affected=None):
        result = self.sql(text)
        assert result.get("ok") is True, (text, result)
        if affected is not None:
            assert result.get("affected_rows") == affected, (text, result)
        return result

    def error(self, text, code=None):
        result = self.sql(text)
        assert result.get("ok") is False, (text, result)
        assert result.get("error") or result.get("code"), (text, result)
        if code is not None:
            assert result.get("error") == code, (text, result)
        return result

    def rows(self, text):
        result = self.ok(text)
        return result["rows"]

    def restart(self, kill=False):
        self.stop(kill=kill)
        self.start()

    def stop(self, kill=False):
        if self.proc and self.proc.poll() is None:
            if kill:
                self.proc.kill()
            else:
                self.proc.terminate()
            self.proc.wait(timeout=10)
        if self.proc:
            if self.proc.stdin:
                self.proc.stdin.close()
            if self.proc.stdout:
                self.proc.stdout.close()
        self.proc = None


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def validation_test(binary, root):
    db = Db(binary, root / "validation").start()
    try:
        db.ok("CREATE TABLE t (id BIGINT PRIMARY KEY, s VARCHAR(3), v VECTOR(2))")
        db.ok("INSERT INTO t VALUES (1, 'abc', '[1,2]')", 1)
        db.error("INSERT INTO t VALUES (2, 'ok', '[NaN,0]')", "invalid_vector")
        db.error("INSERT INTO t VALUES (2, 'ok', '[Infinity,0]')", "invalid_vector")
        db.error("INSERT INTO t VALUES (2, 'ok', '[-inf,0]')", "invalid_vector")
        db.error("INSERT INTO t VALUES (2, 'ok', '[3.5e38,0]')", "invalid_vector")
        db.error("INSERT INTO t VALUES (2, 'ok', '[1]')", "dimension_mismatch")
        db.error("INSERT INTO t VALUES (2, 'ok', '[]')", "dimension_mismatch")
        db.error("INSERT INTO t VALUES (2, '四个字符', '[1,2]')", "string_too_long")
        db.error("INSERT INTO t VALUES (9223372036854775808, 'ok', '[1,2]')", "value_overflow")
        db.ok("INSERT INTO t VALUES (-9223372036854775808, '最小', '[0,0]')", 1)
        assert db.rows("SELECT id, s FROM t ORDER BY id DESC") == [[1, "abc"], [-9223372036854775808, "最小"]]
        db.error("SELECT id FROM t WHERE v = '[1,2]'")
        db.error("SELECT id, L2_DISTANCE(v, '[0]') AS distance FROM t ORDER BY distance LIMIT 1", "dimension_mismatch")
        db.error("INSERT INTO t VALUES (3, 'ok', '[1,2]'), (1, 'dup', '[1,2]')", "duplicate_key")
        assert db.rows("SELECT id FROM t ORDER BY id") == [[-9223372036854775808], [1]]
    finally:
        db.stop()


def transaction_test(binary, root):
    db = Db(binary, root / "transactions").start()
    try:
        db.ok("CREATE TABLE t (id BIGINT PRIMARY KEY, s VARCHAR(8), v VECTOR(2))")
        db.ok("INSERT INTO t VALUES (1,'一','[1,0]'),(2,'二','[2,0]'),(3,'三','[3,0]')", 3)
        db.ok("BEGIN")
        db.ok("UPDATE t SET s='改', v='[9,0]' WHERE id=1", 1)
        db.ok("DELETE FROM t WHERE id=2", 1)
        db.ok("INSERT INTO t VALUES (4,'四','[4,0]')", 1)
        assert db.rows("SELECT id FROM t ORDER BY id") == [[1], [3], [4]]
        db.ok("ROLLBACK")
        assert db.rows("SELECT id, s FROM t ORDER BY id") == [[1, "一"], [2, "二"], [3, "三"]]
        db.ok("BEGIN")
        db.ok("CREATE TABLE transient (id BIGINT PRIMARY KEY)")
        db.ok("INSERT INTO transient VALUES (7)", 1)
        assert db.rows("SELECT id FROM transient") == [[7]]
        db.ok("ROLLBACK")
        db.error("SELECT id FROM transient", "table_not_found")
        db.ok("BEGIN")
        db.ok("CREATE TABLE durable (id BIGINT PRIMARY KEY, v VECTOR(2))")
        db.ok("INSERT INTO durable VALUES (7,'[3,4]')", 1)
        db.ok("COMMIT")
        db.restart(kill=True)
        assert db.rows("SELECT id FROM durable") == [[7]]
        assert db.rows("SELECT id, L2_DISTANCE(v,'[0,0]') AS distance FROM durable ORDER BY distance LIMIT 1") == [[7, 5.0]]
        db.error("COMMIT", "invalid_transaction_state")
    finally:
        db.stop()


def crash_test(binary, root):
    directory = root / "crash"
    db = Db(binary, directory).start()
    db.ok("CREATE TABLE t (id BIGINT PRIMARY KEY, s VARCHAR(8))")
    db.ok("INSERT INTO t VALUES (1,'old')", 1)
    db.ok("BEGIN")
    db.ok("UPDATE t SET s='new' WHERE id=1", 1)
    db.ok("INSERT INTO t VALUES (2,'tx')", 1)
    db.restart(kill=True)
    assert db.rows("SELECT id,s FROM t ORDER BY id") == [[1, "old"]]
    db.ok("INSERT INTO t VALUES (3,'durable')", 1)
    db.restart(kill=True)
    for _ in range(3):
        assert db.rows("SELECT id,s FROM t ORDER BY id") == [[1, "old"], [3, "durable"]]
        db.restart(kill=True)
    db.stop()
    # Simulate a process dying halfway through its next WAL append.
    with open(directory / "wal.log", "ab") as wal:
        wal.write(b"\x40\x00\x00\x00broken-tail")
    db.start()
    assert db.rows("SELECT id,s FROM t ORDER BY id") == [[1, "old"], [3, "durable"]]
    db.ok("INSERT INTO t VALUES (4,'after')", 1)
    db.restart(kill=True)
    assert db.rows("SELECT id FROM t ORDER BY id") == [[1], [3], [4]]
    db.stop()


def distance(a, b):
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))


def random_topk_test(binary, root):
    rng = random.Random(20261109)
    env = {
        "MINIVEC_CHECKPOINT_MIN_BYTES": "1",
        "MINIVEC_CHECKPOINT_LIVE_FACTOR": "0",
        "MINIVEC_VECTOR_ARENA_MB": "0",
    }
    db = Db(binary, root / "random-topk", env).start()
    rows = {}
    dim = 8
    try:
        db.ok("CREATE TABLE t (id BIGINT PRIMARY KEY, category VARCHAR(4), v VECTOR(8))")
        for base in range(0, 120, 20):
            values = []
            for pk in range(base, base + 20):
                vector = [f32(rng.uniform(-5, 5)) for _ in range(dim)]
                category = "even" if pk % 2 == 0 else "odd"
                rows[pk] = (category, vector)
                literal = "[" + ",".join(repr(x) for x in vector) + "]"
                values.append(f"({pk},'{category}','{literal}')")
            db.ok("INSERT INTO t VALUES " + ",".join(values), 20)
        db.ok("CREATE VECTOR INDEX t_v ON t(v)")
        db.ok("UPDATE t SET v='[9,9,9,9,9,9,9,9]' WHERE id=5", 1)
        rows[5] = (rows[5][0], [9.0] * dim)
        db.ok("DELETE FROM t WHERE id=6 OR id=7", 2)
        rows.pop(6)
        rows.pop(7)
        db.restart(kill=True)

        def check_queries():
            for query_index in range(40):
                query = [f32(rng.uniform(-5, 5)) for _ in range(dim)]
                literal = "[" + ",".join(repr(x) for x in query) + "]"
                filtered = query_index % 2 == 1
                candidates = [
                    (pk, value) for pk, value in rows.items()
                    if not filtered or value[0] == "even"
                ]
                expected = sorted(
                    ((distance(vector, query), pk) for pk, (_, vector) in candidates),
                    key=lambda item: (item[0], item[1]),
                )[:10]
                where = " WHERE category='even'" if filtered else ""
                actual = db.rows(
                    "SELECT id, L2_DISTANCE(v,'%s') AS distance FROM t%s ORDER BY distance LIMIT 10"
                    % (literal, where)
                )
                assert [row[0] for row in actual] == [pk for _, pk in expected], (query_index, actual, expected)
                for actual_row, (expected_distance, _) in zip(actual, expected):
                    assert math.isclose(actual_row[1], expected_distance, rel_tol=1e-5, abs_tol=1e-5)

        check_queries()
        db.ok("--mode ann")
        check_queries()
        db.ok("--mode exact")
    finally:
        db.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve()
    root = Path(tempfile.mkdtemp(prefix="minivectordb-extended-"))
    tests = [validation_test, transaction_test, crash_test, random_topk_test]
    try:
        for test in tests:
            test(binary, root)
            print("PASS", test.__name__)
        print(f"{len(tests)}/{len(tests)} extended test groups passed")
        shutil.rmtree(root)
        return 0
    except Exception:
        print("extended test data kept at", root, file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
