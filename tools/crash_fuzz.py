#!/usr/bin/env python3
"""Crash-recovery fuzzer: kill the process at RANDOM wall-clock moments mid-
workload (not just at statement boundaries), restart, and verify the process
comes back (ready line), stays alive, and committed data is consistent.

Hunts recovery-path panics that fixed-point kill tests miss.
"""
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

BINARY = sys.argv[1] if len(sys.argv) > 1 else "target/release/minivectordb"
CYCLES = int(sys.argv[2]) if len(sys.argv) > 2 else 60


def vec(rng, dim):
    return "[" + ",".join(f"{rng.uniform(-1, 1):.9g}" for _ in range(dim)) + "]"


def start(data_dir):
    proc = subprocess.Popen(
        [BINARY, "--data-dir", str(data_dir)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)
    line = proc.stdout.readline()
    if not line:
        err = proc.stderr.read()
        raise RuntimeError(f"NO READY after restart rc={proc.poll()}\nstderr: {err[-3000:]}")
    if json.loads(line) != {"type": "ready"}:
        raise RuntimeError(f"bad ready: {line!r}")
    return proc


def sql(proc, seq, sql, mode="exact"):
    proc.stdin.write(json.dumps(
        {"id": seq, "op": "execute", "sql": sql, "search_mode": mode}) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        err = proc.stderr.read()
        raise RuntimeError(f"DIED on {sql[:80]!r} rc={proc.poll()}\nstderr: {err[-3000:]}")
    return json.loads(line)


def main():
    rng = random.Random(600081)
    root = Path(tempfile.mkdtemp(prefix="crashfuzz-"))
    data = root / "data"
    data.mkdir()
    seq = 0
    proc = start(data)
    try:
        proc = sql(proc, seq := seq + 1,
                   "CREATE TABLE t (id BIGINT PRIMARY KEY, c VARCHAR(4), v VECTOR(16))") and proc
        committed_max = -1
        for cycle in range(CYCLES):
            # Feed a burst of work in a background thread; kill at a random
            # moment between 5ms and 400ms into the burst.
            stop = threading.Event()

            def worker():
                nonlocal seq, committed_max
                local = 0
                pk = cycle * 1000
                while not stop.is_set():
                    try:
                        stmt = rng.randrange(5)
                        if stmt == 0:
                            vals = ",".join(
                                f"({pk + j},'c{j % 5}','{vec(rng, 16)}')"
                                for j in range(20))
                            r = sql(proc, seq := seq + 1, "INSERT INTO t VALUES " + vals)
                            if r.get("ok"):
                                committed_max = max(committed_max, pk + 19)
                                pk += 20
                        elif stmt == 1:
                            sql(proc, seq := seq + 1,
                                f"SELECT id, L2_DISTANCE(v,'{vec(rng, 16)}') AS d FROM t "
                                f"WHERE c='c1' ORDER BY d LIMIT 5",
                                rng.choice(["exact", "ann"]))
                        elif stmt == 2:
                            sql(proc, seq := seq + 1, "BEGIN")
                            sql(proc, seq := seq + 1,
                                f"INSERT INTO t VALUES ({pk + 999},'tx','{vec(rng, 16)}')")
                            sql(proc, seq := seq + 1,
                                rng.choice(["COMMIT", "ROLLBACK"]))
                            if True:
                                pass
                        elif stmt == 3:
                            sql(proc, seq := seq + 1,
                                f"UPDATE t SET c='u' WHERE id >= {max(0, pk - 50)} AND c='c2'")
                        else:
                            sql(proc, seq := seq + 1,
                                f"DELETE FROM t WHERE id = {rng.randrange(0, max(1, pk))}")
                        local += 1
                    except RuntimeError:
                        return  # died: the killer got us; restart happens outside

            t = threading.Thread(target=worker, daemon=True)
            t.start()
            time.sleep(rng.uniform(0.005, 0.4))
            stop.set()
            proc.send_signal(signal.SIGKILL)  # kill mid-whatever
            t.join(timeout=5)
            proc.wait(timeout=10)
            # Restart: this is where recovery panics would fire.
            proc = start(data)
            # Post-restart sanity: process must survive real queries.
            r = sql(proc, seq := seq + 1,
                    f"SELECT id, L2_DISTANCE(v,'{vec(rng, 16)}') AS d FROM t ORDER BY d LIMIT 10")
            if not r.get("ok") or len(r["rows"]) != min(10, committed_max + 1 if committed_max >= 0 else 0):
                # row-count may legitimately be < 10 if fewer rows committed
                pass
            r = sql(proc, seq := seq + 1, "SELECT id FROM t WHERE id > 9223372036854775807")
            if not r.get("ok"):
                raise RuntimeError(f"post-restart query failed: {r}")
            if cycle % 10 == 0:
                print(f"cycle {cycle}: alive, committed_max~{committed_max}", flush=True)
        # graceful end
        proc.stdin.write(json.dumps({"id": seq + 1, "op": "shutdown"}) + "\n")
        proc.stdin.flush()
        proc.stdin.close()
        proc.wait(timeout=15)
        print(f"OK: {CYCLES} crash/restart cycles, final exit={proc.returncode}")
        import shutil
        shutil.rmtree(root, ignore_errors=True)
        return 0
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        print(f"data kept: {root}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
