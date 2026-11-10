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
        self.seq = 0
        self.mode = "exact"

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
        # Official protocol (STDIO_PROTOCOL.md §2): exact ready object after
        # recovery, before the first request.
        line = self.proc.stdout.readline()
        assert json.loads(line) == {"type": "ready"}, line
        return self

    def sql(self, text, mode=None):
        self.seq += 1
        request = {"id": self.seq, "op": "execute", "sql": text,
                   "search_mode": mode or self.mode}
        self.proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        assert line, (text, self.proc.poll())
        response = json.loads(line)
        assert response.get("id") == self.seq, (text, response)
        assert isinstance(response.get("ok"), bool), (text, response)
        return response

    def directive(self, text):
        """Mode directives from the old .test files now map to the per-request
        search_mode field instead of a wire directive."""
        if text == "--mode ann":
            self.mode = "ann"
        elif text == "--mode exact":
            self.mode = "exact"
        else:
            raise ValueError(f"unsupported directive: {text}")

    def ok(self, text, affected=None):
        result = self.sql(text)
        assert result.get("ok") is True, (text, result)
        if affected is not None:
            assert result.get("affected_rows") == affected, (text, result)
        return result

    def error(self, text, code=None):
        result = self.sql(text)
        assert result.get("ok") is False, (text, result)
        if code is not None:
            assert result["error"]["code"] == code, (text, result)
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
                self.proc.wait(timeout=10)
            else:
                # Official shutdown (§4): respond then exit 0 with no
                # further stdout. Force-kill only if it hangs.
                self.seq += 1
                self.proc.stdin.write(
                    json.dumps({"id": self.seq, "op": "shutdown"}) + "\n")
                self.proc.stdin.flush()
                line = self.proc.stdout.readline()
                assert line, "no shutdown response"
                response = json.loads(line)
                assert response == {"id": self.seq, "ok": True}, response
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=10)
                assert self.proc.returncode == 0, self.proc.returncode
                extra = self.proc.stdout.read()
                assert not extra.strip(), f"extra stdout after shutdown: {extra!r}"
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
        db.directive("--mode ann")
        check_queries()
        db.directive("--mode exact")
    finally:
        db.stop()


def boundaries_test(binary, root):
    db = Db(binary, root / "boundaries").start()
    try:
        db.ok("CREATE TABLE t (id BIGINT PRIMARY KEY, category VARCHAR(4), v VECTOR(2))")
        db.ok("INSERT INTO t VALUES (-5,'甲','[5,0]'),(1,'乙','[1,0]'),(2,'乙','[2,0]')", 3)
        # SELECT of the vector column itself (rendered as a numeric array).
        raw = db.sql("SELECT v FROM t WHERE id=1")
        assert isinstance(raw["rows"][0][0], list) and len(raw["rows"][0][0]) == 2
        # LIMIT edge cases.
        assert db.rows("SELECT id FROM t ORDER BY id LIMIT 0") == []
        assert len(db.rows("SELECT id FROM t ORDER BY id LIMIT 99")) == 3
        # Zero-match mutations still report affected_rows 0 and succeed.
        db.ok("UPDATE t SET category='x' WHERE id=999", 0)
        db.ok("DELETE FROM t WHERE id=999", 0)
        # OR filters.
        assert db.rows("SELECT id FROM t WHERE id=-5 OR category='乙' ORDER BY id") == [[-5], [1], [2]]
        # Distance DESC ordering with pk tie-break.
        assert db.rows("SELECT id, L2_DISTANCE(v,'[0,0]') AS distance FROM t ORDER BY distance DESC LIMIT 2") == [
            [-5, 5.0], [2, 2.0]
        ]
        # VARCHAR exactly at the limit is accepted; one over is rejected.
        db.ok("INSERT INTO t VALUES (3,'四字测试','[0,1]')", 1)
        db.error("INSERT INTO t VALUES (4,'五个字符串','[0,1]')", "string_too_long")
        # DDL edge cases: second vector column, missing primary key,
        # non-BIGINT primary key, duplicate table.
        db.error("CREATE TABLE bad1 (id BIGINT PRIMARY KEY, a VECTOR(2), b VECTOR(2))", "unsupported_sql")
        db.error("CREATE TABLE bad2 (id BIGINT)", "unsupported_sql")
        db.error("CREATE TABLE bad3 (s VARCHAR(4) PRIMARY KEY)", "unsupported_sql")
        db.error("CREATE TABLE t (id BIGINT PRIMARY KEY)", "table_exists")
        # Primary key column cannot be updated.
        db.error("UPDATE t SET id=9 WHERE id=1", "unsupported_sql")
        # Nested transactions are rejected, and a failed BEGIN leaves the
        # outer transaction usable.
        db.ok("BEGIN")
        db.error("BEGIN", "invalid_transaction_state")
        db.ok("UPDATE t SET category='内' WHERE id=1", 1)
        db.ok("ROLLBACK")
        assert db.rows("SELECT category FROM t WHERE id=1") == [["乙"]]
        # Errors keep the session usable afterwards.
        db.error("SELECT id FROM missing", "table_not_found")
        assert db.rows("SELECT id FROM t WHERE id=1") == [[1]]
    finally:
        db.stop()


def ivf_recall_test(binary, root):
    """IVF path coverage on a table large enough to trigger clustering.

    Checks ann-mode results against brute-force truth for recall, ID
    validity, no duplicates and filter correctness, including incremental
    maintenance after update/delete/insert and a restart rebuild.
    """
    rng = random.Random(600081)
    n = 5000
    dim = 16
    rows = {}
    db = Db(binary, root / "ivf-recall").start()
    try:
        db.ok("CREATE TABLE t (id BIGINT PRIMARY KEY, category VARCHAR(4), v VECTOR(%d))" % dim)
        for base in range(0, n, 250):
            values = []
            for pk in range(base, base + 250):
                vector = [f32(rng.gauss(0, 1)) for _ in range(dim)]
                rows[pk] = ("even" if pk % 2 == 0 else "odd", vector)
                values.append("(%d,'%s','%s')" % (pk, rows[pk][0], "[" + ",".join(repr(x) for x in vector) + "]"))
            db.ok("INSERT INTO t VALUES " + ",".join(values), 250)
        db.ok("CREATE VECTOR INDEX t_v ON t(v)")
        # incremental maintenance before querying
        db.ok("UPDATE t SET v='[%s]' WHERE id=10" % ",".join(repr(f32(rng.gauss(5, 1))) for _ in range(dim)), 1)
        db.ok("DELETE FROM t WHERE id >= 20 AND id < 40", 20)
        for pk in range(20, 40):
            rows.pop(pk)
        db.directive("--mode ann")

        def truth(query, filtered):
            candidates = [item for item in rows.items() if not filtered or item[1][0] == "even"]
            ranked = sorted(
                ((distance(vector, query), pk) for pk, (_, vector) in candidates),
                key=lambda item: (item[0], item[1]),
            )
            return ranked[:10]

        recalls = []
        for index in range(30):
            query = [f32(rng.gauss(0, 1)) for _ in range(dim)]
            literal = "[" + ",".join(repr(x) for x in query) + "]"
            filtered = index % 2 == 1
            where = " WHERE category='even'" if filtered else ""
            result = db.rows(
                "SELECT id, L2_DISTANCE(v,'%s') AS distance FROM t%s ORDER BY distance LIMIT 10"
                % (literal, where)
            )
            ids = [row[0] for row in result]
            assert len(ids) == 10, (index, ids)
            assert len(set(ids)) == 10, (index, ids)
            assert all(pk in rows for pk in ids), (index, ids)
            if filtered:
                assert all(pk % 2 == 0 for pk in ids), (index, ids)
            expected = truth(query, filtered)
            recalls.append(len(set(ids) & {pk for _, pk in expected}) / 10)
            # Distances must match truth for any overlapping id.
            by_pk = {pk: dist for dist, pk in expected}
            for pk, dist in zip(ids, (row[1] for row in result)):
                if pk in by_pk:
                    assert math.isclose(dist, by_pk[pk], rel_tol=5e-4, abs_tol=5e-4), (index, pk, dist, by_pk[pk])
        overall = sum(recalls) / len(recalls)
        assert overall >= 0.95, ("recall below 0.95", overall, recalls)
        # restart rebuilds the index before ready; results stay correct
        db.restart(kill=True)
        hits = []
        for _ in range(10):
            query = [f32(rng.gauss(0, 1)) for _ in range(dim)]
            result = db.rows(
                "SELECT id FROM t ORDER BY L2_DISTANCE(v,'%s') ASC LIMIT 10"
                % ("[" + ",".join(repr(x) for x in query) + "]")
            )
            expected = {pk for _, pk in truth(query, False)}
            hits.append(len({row[0] for row in result} & expected) / 10)
        assert sum(hits) / len(hits) >= 0.9, ("post-restart recall", hits)
        return overall
    finally:
        db.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve()
    root = Path(tempfile.mkdtemp(prefix="minivectordb-extended-"))
    tests = [validation_test, transaction_test, crash_test, random_topk_test, boundaries_test, ivf_recall_test]
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
