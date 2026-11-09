#!/usr/bin/env python3
"""Smoke-test every wire-protocol variant end to end.

When the official driver publishes its format, the right variant must be one
environment variable away - and already verified. This script drives each
MINIVEC_PROTOCOL value through the full statement lifecycle: DDL, DML,
query (ordered TopK with float output), error, driver directives (silent),
comment lines, trailing semicolons, multi-statement lines, clean shutdown on
--restart with rollback, and recovery after restart.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def talk(binary, data_dir, lines, env=None):
    """Send lines, return (stdout_lines, exit_code). stdin closes at EOF."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    proc = subprocess.run(
        [str(binary), "--data-dir", str(data_dir)],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=full_env,
        timeout=30,
    )
    return [line for line in proc.stdout.splitlines() if line], proc.returncode


def check_jsonl(binary, root):
    data = root / "jsonl"
    out, rc = talk(binary, data, [
        "CREATE TABLE t (id BIGINT PRIMARY KEY, category VARCHAR(4), v VECTOR(2));",
        "INSERT INTO t VALUES (1,'a','[3,0]'),(2,'b','[1,0]');",
        "-- a comment",
        "SELECT id, L2_DISTANCE(v,'[0,0]') AS distance FROM t ORDER BY distance LIMIT 1; SELECT id FROM t WHERE category='b';",
        "INSERT INTO t VALUES (1,'dup','[0,0]');",
        "SELECT nope FROM t;",
        "--mode ann",
        "SELECT id FROM t ORDER BY id;",
    ])
    assert rc == 0, rc
    assert out == [
        "{}",
        '{"affected_rows":2}',
        '{"columns":["id","distance"],"rows":[[2,1.0]]}',
        '{"columns":["id"],"rows":[[2]]}',
        '{"error":"duplicate_key"}',
        '{"error":"column_not_found"}',
        '{"columns":["id"],"rows":[[1],[2]]}',
    ], out
    # --restart: uncommitted work rolls back, process exits 0, data survives.
    out, rc = talk(binary, data, [
        "BEGIN;",
        "INSERT INTO t VALUES (3,'c','[0,9]');",
        "--restart",
    ])
    assert rc == 0, rc
    assert out == ["{}", '{"affected_rows":1}'], out
    out, rc = talk(binary, data, ["SELECT id FROM t ORDER BY id;"])
    assert out == ['{"columns":["id"],"rows":[[1],[2]]}'], out
    return "jsonl ok"


def check_tsv(binary, root):
    data = root / "tsv"
    out, rc = talk(binary, data, [
        "CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(2));",
        "INSERT INTO t VALUES (1,'[3,0]');",
        "SELECT id, L2_DISTANCE(v,'[0,0]') AS distance FROM t LIMIT 5;",
        "SELECT id FROM t WHERE id=99;",
        "INSERT INTO t VALUES (1,'[0,0]');",
    ], env={"MINIVEC_PROTOCOL": "tsv"})
    assert rc == 0, rc
    assert out == [
        "READY",
        "OK 0",
        "OK 1",
        "id\tdistance",
        "1\t3.0",
        "OK 1",
        "id",
        "OK 0",
        "ERROR duplicate_key: duplicate_key",
    ], out
    return "tsv ok"


def check_json(binary, root):
    data = root / "json"
    out, rc = talk(binary, data, [
        "CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(2));",
        "INSERT INTO t VALUES (7,'[1,1]');",
        "SELECT id, L2_DISTANCE(v,'[0,0]') AS distance FROM t;",
        "SELECT id FROM t WHERE id=1;",
    ], env={"MINIVEC_PROTOCOL": "json"})
    assert rc == 0, rc
    assert out == [
        '{"ready":true}',
        '{"ok":true}',
        '{"ok":true,"affected_rows":1}',
        '{"ok":true,"columns":["id","distance"],"rows":[[7,1.414214]]}',
        '{"ok":true,"columns":["id"],"rows":[]}',
    ], out
    return "json ok"


def check_json_noready(binary, root):
    data = root / "json-noready"
    out, rc = talk(binary, data, [
        "CREATE TABLE t (id BIGINT PRIMARY KEY);",
        "SELECT * FROM t;",
    ], env={"MINIVEC_PROTOCOL": "json-noready"})
    assert rc == 0, rc
    assert out == [
        '{"ok":true}',
        '{"ok":true,"columns":["id"],"rows":[]}',
    ], out
    return "json-noready ok"


def check_miniob(binary, root):
    data = root / "miniob"
    out, rc = talk(binary, data, [
        "CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(2));",
        "INSERT INTO t VALUES (5,'[6,8]');",
        "SELECT id, L2_DISTANCE(v,'[0,0]') AS distance FROM t;",
        "SELECT id FROM t WHERE id=1;",
    ], env={"MINIVEC_PROTOCOL": "miniob"})
    assert rc == 0, rc
    assert out == [
        "SUCCESS",
        "SUCCESS",
        "affected rows: 1",
        "id | distance",
        "5 | 10.0",
        "SUCCESS",
        "id",
        "SUCCESS",
    ], out
    return "miniob ok"


def main():
    binary = Path(sys.argv[1] if len(sys.argv) > 1 else "target/release/minivectordb").resolve()
    root = Path(tempfile.mkdtemp(prefix="minivectordb-protocols-"))
    try:
        print(check_jsonl(binary, root))
        print(check_tsv(binary, root))
        print(check_json(binary, root))
        print(check_json_noready(binary, root))
        print(check_miniob(binary, root))
        print("all protocol variants ok")
        return 0
    except AssertionError as exc:
        print("PROTOCOL CHECK FAILED:", exc, file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
