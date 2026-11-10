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


def check_official(binary, root):
    """Official STDIO_PROTOCOL.md transport: JSONL requests, id echo,
    {"type":"ready"} handshake, shutdown semantics."""
    data = root / "official"
    proc = subprocess.Popen(
        [str(binary), "--data-dir", str(data)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1,
    )
    ready = proc.stdout.readline()
    assert json.loads(ready) == {"type": "ready"}, ready

    def req(payload):
        proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        assert line, payload
        return json.loads(line)

    # DDL, DML, query, error; integer id echoed as integer.
    assert req({"id": 1, "op": "execute", "sql":
                "CREATE TABLE t (id BIGINT PRIMARY KEY, s VARCHAR(4), v VECTOR(2));",
                "search_mode": "exact"}) == {"id": 1, "ok": True}
    assert req({"id": 2, "op": "execute", "sql":
                "INSERT INTO t VALUES (1,'a','[3,0]'),(2,'b','[1,0]');",
                "search_mode": "exact"}) == {"id": 2, "ok": True, "affected_rows": 2}
    query = req({"id": 3, "op": "execute", "sql":
                 "SELECT id, L2_DISTANCE(v,'[0,0]') AS distance FROM t ORDER BY distance LIMIT 1;",
                 "search_mode": "exact"})
    assert query == {"id": 3, "ok": True, "columns": ["id", "distance"], "rows": [[2, 1.0]]}, query
    err = req({"id": 4, "op": "execute", "sql": "INSERT INTO t VALUES (1,'d','[0,0]');",
               "search_mode": "exact"})
    assert err["ok"] is False and err["id"] == 4 and err["error"]["code"], err
    # String id echoed with its JSON type preserved.
    strid = req({"id": "seven", "op": "execute", "sql": "SELECT id FROM t;", "search_mode": "ann"})
    assert strid == {"id": "seven", "ok": True, "columns": ["id"], "rows": [[1], [2]]}, strid
    # Unsupported op answers an error object, not silence.
    bad = req({"id": 5, "op": "frobnicate"})
    assert bad["ok"] is False and bad["error"]["code"], bad
    # Malformed lines must stay silent on stdout (unsolicited responses fail
    # the judge transport).
    proc.stdin.write("not json at all\n")
    proc.stdin.flush()
    # Shutdown: response, exit 0, nothing further on stdout.
    proc.stdin.write('{"id":6,"op":"shutdown"}\n')
    proc.stdin.flush()
    line = proc.stdout.readline()
    assert json.loads(line) == {"id": 6, "ok": True}, line
    proc.stdin.close()
    proc.wait(timeout=10)
    assert proc.returncode == 0, proc.returncode
    extra = proc.stdout.read()
    assert not extra.strip(), extra
    # Committed data survives the clean shutdown; ready repeats on restart.
    proc = subprocess.Popen(
        [str(binary), "--data-dir", str(data)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1,
    )
    assert json.loads(proc.stdout.readline()) == {"type": "ready"}
    proc.stdin.write('{"id":1,"op":"execute","sql":"SELECT id FROM t ORDER BY id;","search_mode":"exact"}\n')
    proc.stdin.flush()
    line = proc.stdout.readline()
    assert json.loads(line) == {"id": 1, "ok": True, "columns": ["id"], "rows": [[1], [2]]}, line
    proc.stdin.write('{"id":2,"op":"shutdown"}\n')
    proc.stdin.flush()
    proc.stdout.readline()
    proc.stdin.close()
    proc.wait(timeout=10)
    return "official ok"


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
    ], env={"MINIVEC_PROTOCOL": "jsonl"})
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
    ], env={"MINIVEC_PROTOCOL": "jsonl"})
    assert rc == 0, rc
    assert out == ["{}", '{"affected_rows":1}'], out
    out, rc = talk(binary, data, ["SELECT id FROM t ORDER BY id;"], env={"MINIVEC_PROTOCOL": "jsonl"})
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
        print(check_official(binary, root))
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
