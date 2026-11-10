#!/usr/bin/env python3
"""Grammar fuzzer over the official protocol: valid-ish SQL plus mutations.

Goal: find any input that kills the process (panic/abort). Any stderr line
containing 'panicked at' is reported with the exact source location.
"""
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

BINARY = sys.argv[1] if len(sys.argv) > 1 else "target/release/minivectordb"
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000

IDENTS = ["t", "t2", "docs", "你好", "a" * 60, "T", "select", "id", "v", "c"]
COLS = ["id", "v", "c", "nope", "ID", "V", "", "x y"]
VECS = ["'[1,2]'", "'[0]'", "'[]'", "'[1,2,3,4,5,6,7,8]'", "'[1e400]'", "'[NaN,1]'",
        "'[-0.0,-0.0]'", "'[3e38,3e38]'", "'[1,2'", "''", "'1,2'", "[1,2]"]
NUMS = ["0", "1", "-1", "9223372036854775807", "-9223372036854775808",
        "9223372036854775808", "1.5", "1e10", "0x10", "007", "'1'"]
STRS = ["'a'", "''", "'四个字符长串'", "'it''s'", "'\\'", "'[1,2]'", "nul"]


def gen(rng):
    kind = rng.randrange(12)
    t = rng.choice(IDENTS)
    c = rng.choice(COLS)
    v = rng.choice(VECS)
    n = rng.choice(NUMS)
    s = rng.choice(STRS)
    if kind == 0:
        return f"CREATE TABLE {t} (id BIGINT PRIMARY KEY, c VARCHAR(4), v VECTOR({rng.choice([2, 4, 8, 128])}))"
    if kind == 1:
        return f"INSERT INTO {t} VALUES ({n}, {s}, {v})"
    if kind == 2:
        return f"INSERT INTO {t} (id, c, v) VALUES ({n}, {s}, {v})"
    if kind == 3:
        return f"INSERT INTO {t} (id) VALUES ({n})"
    if kind == 4:
        return f"SELECT {c} FROM {t}"
    if kind == 5:
        return f"SELECT id, L2_DISTANCE({c}, {v}) AS d FROM {t} WHERE {c} {rng.choice(['=', '<', '>', '<=', '>=', '!='])} {n} ORDER BY {rng.choice(['d', 'id', c])} {rng.choice(['ASC', 'DESC', ''])} LIMIT {rng.choice(['0', '1', '10', '-1', n])}"
    if kind == 6:
        return f"UPDATE {t} SET v={v}, c={s} WHERE id {rng.choice(['=', '<'], )} {n}" if rng.random() < 0.5 else f"UPDATE {t} SET id={n}"
    if kind == 7:
        return f"DELETE FROM {t} WHERE {c} = {n}"
    if kind == 8:
        return rng.choice(["BEGIN", "COMMIT", "ROLLBACK"])
    if kind == 9:
        return f"CREATE VECTOR INDEX {rng.choice(['iv', 'iv2', t])} ON {t}({rng.choice(['v', c])})"
    if kind == 10:
        base = f"SELECT * FROM {t} WHERE {c} = {s}"
        return rng.choice([base, base.upper(), base + ";", base + " ;", "  " + base, base[: rng.randrange(len(base))] if base else base])
    return rng.choice([
        "SELECT id FROM t ORDER BY v",
        "SELECT id FROM t ORDER BY v LIMIT 2",
        "SELECT v FROM t ORDER BY id",
        "SELECT id, L2_DISTANCE(v, '[1,2]') AS distance FROM t ORDER BY distance DESC LIMIT 3",
        "SELECT id FROM t WHERE v = '[1,2]'",
        "DROP TABLE t", "ALTER TABLE t ADD c VARCHAR(2)",
        "SELECT COUNT(*) FROM t", "SELECT DISTINCT id FROM t",
        "SELECT id FROM t1 JOIN t2 ON 1=1",
        "INSERT INTO t VALUES (NULL, 'a', '[1,2]')",
        "SELECT L2_DISTANCE(v, '[1,2]') FROM t",
        "SELECT id, L2_DISTANCE(c, '[1,2]') AS d FROM t",
    ])


def main():
    rng = random.Random(600081)
    root = Path(tempfile.mkdtemp(prefix="fuzz-"))
    proc = subprocess.Popen(
        [BINARY, "--data-dir", str(root)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)
    assert json.loads(proc.stdout.readline()) == {"type": "ready"}
    seq = 0
    killed_at = None
    stderr_tail = []
    try:
        for i in range(ROUNDS):
            sql = gen(rng)
            seq += 1
            proc.stdin.write(json.dumps(
                {"id": seq, "op": "execute", "sql": sql, "search_mode": rng.choice(["exact", "ann", "weird"])}) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            if not line:
                killed_at = (i, sql)
                break
            resp = json.loads(line)
            assert resp.get("id") == seq, (sql, resp)
            assert isinstance(resp.get("ok"), bool), (sql, resp)
    finally:
        if proc.poll() is None:
            proc.stdin.close()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        stderr = proc.stderr.read()
    if killed_at:
        print(f"PROCESS DIED at iteration {killed_at[0]} sql={killed_at[1]!r}")
        print("STDERR:", stderr[-4000:])
        return 1
    panics = [ln for ln in stderr.splitlines() if "panicked" in ln or "RUST_BACKTRACE" in ln]
    if panics:
        print("PANICS IN STDERR:", panics[:10])
        print(stderr[-3000:])
        return 1
    print(f"OK: {ROUNDS} fuzzed statements, process alive, exit={proc.returncode}")
    import shutil
    shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
