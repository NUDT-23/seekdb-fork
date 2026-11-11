#!/usr/bin/env python3
"""Judge simulator: run plausible hidden-case workloads against the Docker
image under the EXACT resource limits of the official reference harness
(STDIO_PROTOCOL.md: --network=none --cpus=2 --memory=2g --memory-swap=2g
--pids-limit=128, 180s budget per case). Reports per-case wall time, exit
code (137 = OOM-kill) and response integrity — hunting the platform's
"database exited before response".
"""
import json
import os
import random
import struct
import subprocess
import sys
import time
from pathlib import Path

IMAGE = sys.argv[1] if len(sys.argv) > 1 else "minivectordb:official"
DIM = int(sys.argv[2]) if len(sys.argv) > 2 else 128
ROWS = int(sys.argv[3]) if len(sys.argv) > 3 else 100_000
BUDGET = 600  # protocol_check/config.json case_timeout (the 180s shell demo is only a smoke test)


def f32(v):
    return struct.unpack("<f", struct.pack("<f", v))[0]


def vec(rng, dim):
    return "[" + ",".join(f"{f32(rng.uniform(-1, 1)):.8g}" for _ in range(dim)) + "]"


def build_workload(rows, dim):
    rng = random.Random(600081)
    reqs = []
    seq = 0

    def add(sql, mode="exact"):
        nonlocal seq
        seq += 1
        reqs.append({"id": seq, "op": "execute", "sql": sql, "search_mode": mode})

    add(f"CREATE TABLE t (id BIGINT PRIMARY KEY, category VARCHAR(4), v VECTOR({dim}))")
    t0 = time.time()
    batch = 500
    for base in range(0, rows, batch):
        vals = ",".join(
            f"({pk},'{'even' if pk % 2 == 0 else 'odd'}','{vec(rng, dim)}')"
            for pk in range(base, min(base + batch, rows)))
        add("INSERT INTO t VALUES " + vals)
    add("CREATE VECTOR INDEX iv ON t (v)")
    for i in range(10):
        add(f"SELECT id, L2_DISTANCE(v,'{vec(rng, dim)}') AS d FROM t "
            f"WHERE category='even' ORDER BY d LIMIT 10")
    add(f"SELECT id, L2_DISTANCE(v,'{vec(rng, dim)}') AS d FROM t ORDER BY d LIMIT 10", "ann")
    add("UPDATE t SET category='u' WHERE id >= %d AND id < %d" % (rows - 2000, rows))
    add("DELETE FROM t WHERE id >= %d AND id < %d" % (rows - 1000, rows))
    add("SELECT id FROM t WHERE id = %d" % (rows // 2))
    add("SELECT id FROM t WHERE id >= 0 LIMIT 5")
    seq += 1
    reqs.append({"id": seq, "op": "shutdown"})
    return reqs


def run_case(name, reqs):
    """One docker run = one case, EXACT flags of the official reference
    harness (STDIO_PROTOCOL.md self-check): --read-only, --cap-drop=ALL,
    --security-opt=no-new-privileges, --memory-swap=2g (swap OFF), bind
    mount at /data with chmod 777."""
    payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in reqs)
    data_dir = Path.home() / f"jsim-{name}"
    if data_dir.exists():
        subprocess.run(["rm", "-rf", str(data_dir)], check=True)
    data_dir.mkdir(parents=True)
    os.chmod(data_dir, 0o777)
    cmd = ["docker", "run", "--rm", "-i", "--network=none", "--cpus=2",
           "--memory=2g", "--memory-swap=2g", "--pids-limit=128",
           "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
           "--mount", f"type=bind,src={data_dir},dst=/data", IMAGE]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, input=payload, capture_output=True,
                              text=True, encoding="utf-8", timeout=BUDGET + 30)
        dur = time.time() - t0
        lines = [l for l in proc.stdout.splitlines() if l.strip()]
        rc = proc.returncode
        ready_ok = bool(lines) and lines[0].strip() == '{"type":"ready"}'
        n_resp = len(lines) - 1
        last = lines[-1] if lines else ""
        ok_count = sum(1 for l in lines[1:] if json.loads(l).get("ok"))
        verdict = "PASS" if (rc == 0 and ready_ok and n_resp == len(reqs)
                             and ok_count == len(reqs)) else "FAIL"
        oom = " **OOM-KILL**" if rc == 137 else ""
        print(f"{verdict} {name}: {dur:.1f}s rc={rc}{oom} ready={ready_ok} "
              f"responses={n_resp}/{len(reqs)} ok={ok_count} last={last[:60]}")
        if verdict == "FAIL" and proc.stderr:
            print(f"  stderr tail: {proc.stderr[-400:]}")
        subprocess.run(["rm", "-rf", str(data_dir)], check=False)
        return verdict == "PASS", dur
    except subprocess.TimeoutExpired:
        dur = time.time() - t0
        print(f"FAIL {name}: HARD TIMEOUT >{BUDGET+30}s (judge budget {BUDGET}s)")
        return False, dur


def run_streamed(name, rows, dim):
    """Lockstep streaming: one request -> one response, judge-style, with the
    exact judge docker flags. For payloads too big to build in memory."""
    import threading
    data_dir = Path.home() / f"jsim-{name}"
    if data_dir.exists():
        subprocess.run(["rm", "-rf", str(data_dir)], check=True)
    data_dir.mkdir(parents=True)
    os.chmod(data_dir, 0o777)
    cmd = ["docker", "run", "--rm", "-i", "--network=none", "--cpus=2",
           "--memory=2g", "--memory-swap=2g", "--pids-limit=128",
           "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
           "--mount", f"type=bind,src={data_dir},dst=/data", IMAGE]
    rng = random.Random(600081)
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8",
                            bufsize=1)
    phase_t = {}

    def phase(label):
        class Ctx:
            def __enter__(s):
                s.t = time.time()
            def __exit__(s, *a):
                phase_t[label] = phase_t.get(label, 0) + time.time() - s.t
        return Ctx()

    ready = proc.stdout.readline()
    assert ready.strip() == '{"type":"ready"}', f"bad ready {ready!r}"

    def req(obj, timeout=170):
        proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            err = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"DIED mid-case rc={proc.poll()} stderr={err[-300:]}")
        return json.loads(line)

    seq = 0

    def sql(s, mode="exact"):
        nonlocal seq
        seq += 1
        return req({"id": seq, "op": "execute", "sql": s, "search_mode": mode})

    total_bytes = 0
    # Reuse 256 deterministic vector literals. SQL bytes and database memory
    # behavior stay realistic, while the harness does not spend minutes
    # generating hundreds of millions of random floats.
    vecs = [vec(rng, dim) for _ in range(256)]
    with phase("create"):
        sql(f"CREATE TABLE t (id BIGINT PRIMARY KEY, category VARCHAR(4), v VECTOR({dim}))")
    with phase("load"):
        batch = 500
        for base in range(0, rows, batch):
            vals = ",".join(
                f"({pk},'{'even' if pk % 2 == 0 else 'odd'}','{vecs[pk % 256]}')"
                for pk in range(base, min(base + batch, rows)))
            stmt = "INSERT INTO t VALUES " + vals
            total_bytes += len(stmt)
            r = sql(stmt)
            if not r.get("ok"):
                print(f"  load batch failed at {base}: {str(r)[:120]}")
                break
    with phase("index"):
        sql("CREATE VECTOR INDEX iv ON t (v)")
    with phase("query"):
        for i in range(2):
            sql(f"SELECT id, L2_DISTANCE(v,'{vec(rng, dim)}') AS d FROM t "
                f"WHERE category='even' ORDER BY d LIMIT 10")
        # ANN must also fall back to bounded scanning instead of building a
        # second full in-memory vector copy for the IVF index.
        sql(f"SELECT id, L2_DISTANCE(v,'{vec(rng, dim)}') AS d FROM t "
            f"WHERE category='odd' ORDER BY d LIMIT 10", "ann")
    seq += 1
    with phase("shutdown"):
        r = req({"id": seq, "op": "shutdown"})
    proc.stdin.close()
    rc = proc.wait(timeout=60)
    with phase("restart"):
        if rc == 0:
            # Reuse the same /data mount exactly like the judge's --restart.
            # Startup must stream multi-GB checkpoint files, not read_to_end.
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                    bufsize=1)
            ready = proc.stdout.readline()
            if ready.strip() != '{"type":"ready"}':
                raise RuntimeError(f"no ready after big-data restart: {ready!r}")
    with phase("recovery-query"):
        if rc == 0:
            sql(f"SELECT id, L2_DISTANCE(v,'{vec(rng, dim)}') AS d FROM t "
                f"WHERE category='even' ORDER BY d LIMIT 10")
            sql(f"SELECT id, L2_DISTANCE(v,'{vec(rng, dim)}') AS d FROM t "
                f"WHERE category='odd' ORDER BY d LIMIT 10", "ann")
            seq += 1
            r = req({"id": seq, "op": "shutdown"})
            proc.stdin.close()
            rc = proc.wait(timeout=60)
    dur = time.time() - t0
    print(f"{'PASS' if rc == 0 and r.get('ok') else 'FAIL'} {name}: total={dur:.1f}s "
          f"rc={rc} phases={{{', '.join(f'{k}:{v:.1f}s' for k, v in phase_t.items())}}} "
          f"sql_mb={total_bytes/1e6:.0f}")
    subprocess.run(["rm", "-rf", str(data_dir)], check=False)
    return dur < BUDGET and rc == 0


def main():
    if len(sys.argv) > 4 and sys.argv[4] == "stream":
        ok = run_streamed(f"stream-{sys.argv[3]}x{sys.argv[2]}", int(sys.argv[3]), int(sys.argv[2]))
        return 0 if ok else 1
    print(f"image={IMAGE} dim={DIM} rows={ROWS}")
    reqs = build_workload(ROWS, DIM)
    print(f"workload: {len(reqs)} requests")
    ok, dur = run_case(f"load-{ROWS}x{DIM}", reqs)
    print(f"total {dur:.1f}s (budget {BUDGET}s)")
    return 0 if ok and dur < BUDGET else 1


if __name__ == "__main__":
    sys.exit(main())
