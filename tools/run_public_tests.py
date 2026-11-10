#!/usr/bin/env python3
"""Local driver for the public MiniVectorDB mysqltest-style samples.

Speaks the official STDIO_PROTOCOL.md transport (JSONL requests/responses,
`{"type":"ready"}` handshake, shutdown op) and checks the package's result
assertions with stricter shape rules than the official protocol_check:
response id must echo the request id, ok must be the right boolean, and no
unexpected fields may ride along.
"""

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


class CaseFailure(Exception):
    pass


class Session:
    def __init__(self, binary, data_dir):
        self.proc = subprocess.Popen(
            [str(binary), "--data-dir", str(data_dir)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        ready = self.proc.stdout.readline()
        if json.loads(ready) != {"type": "ready"}:
            raise CaseFailure(f"bad ready line: {ready!r}")
        self.seq = 0
        self.mode = "exact"

    def execute(self, sql):
        self.seq += 1
        request = {"id": self.seq, "op": "execute", "sql": sql, "search_mode": self.mode}
        self.proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise CaseFailure(f"database closed stdout after {sql!r}; rc={self.proc.poll()}")
        try:
            response = json.loads(line)
        except Exception as exc:
            raise CaseFailure(f"non-JSON response to {sql!r}: {line!r}: {exc}")
        if response.get("id") != self.seq:
            raise CaseFailure(f"{sql}: response id {response.get('id')!r} != {self.seq}")
        if not isinstance(response.get("ok"), bool):
            raise CaseFailure(f"{sql}: missing boolean ok: {response}")
        return response

    def set_mode(self, mode):
        self.mode = mode

    def shutdown(self):
        """Official restart/shutdown: respond, exit 0, nothing further."""
        self.seq += 1
        self.proc.stdin.write(json.dumps({"id": self.seq, "op": "shutdown"}) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise CaseFailure(f"no shutdown response; rc={self.proc.poll()}")
        response = json.loads(line)
        if response != {"id": self.seq, "ok": True}:
            raise CaseFailure(f"bad shutdown response: {response!r}")
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            raise CaseFailure("database did not exit after shutdown")
        if self.proc.returncode != 0:
            raise CaseFailure(f"shutdown exit code {self.proc.returncode}")
        extra = self.proc.stdout.read()
        if extra.strip():
            raise CaseFailure(f"extra stdout after shutdown: {extra!r}")

    def close(self):
        if self.proc.poll() is None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=5)
        else:
            if self.proc.stdin:
                self.proc.stdin.close()
        if self.proc.stdout:
            self.proc.stdout.close()


def equal_value(actual, expected):
    if isinstance(expected, float) or isinstance(actual, float):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            return False
        if not isinstance(expected, (int, float)) or isinstance(expected, bool):
            return False
        return math.isclose(float(actual), float(expected), rel_tol=1e-5, abs_tol=1e-5)
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            equal_value(a, e) for a, e in zip(actual, expected)
        )
    return actual == expected


def equal_rows(actual, expected, unordered):
    if len(actual) != len(expected):
        return False
    if not unordered:
        return all(equal_value(a, e) for a, e in zip(actual, expected))
    remaining = list(actual)
    for wanted in expected:
        for index, candidate in enumerate(remaining):
            if equal_value(candidate, wanted):
                remaining.pop(index)
                break
        else:
            return False
    return not remaining


def check_response(actual, expected, sql):
    if "error" in expected:
        if actual.get("ok") is not False:
            raise CaseFailure(f"{sql}: expected error {expected['error']}, got {actual}")
        code = actual.get("error", {}).get("code")
        if not code:
            raise CaseFailure(f"{sql}: empty error code: {actual}")
        if code != expected["error"]:
            raise CaseFailure(f"{sql}: expected error {expected['error']}, got {code}")
        if set(actual) != {"id", "ok", "error"}:
            raise CaseFailure(f"{sql}: unexpected fields alongside error: {actual}")
        return
    if actual.get("ok") is not True:
        raise CaseFailure(f"{sql}: expected success, got {actual}")
    allowed = {"id", "ok", "affected_rows", "columns", "rows"}
    if set(actual) - allowed:
        raise CaseFailure(f"{sql}: unexpected fields: {set(actual) - allowed}")
    if "affected_rows" in expected and actual.get("affected_rows") != expected["affected_rows"]:
        raise CaseFailure(
            f"{sql}: affected_rows expected {expected['affected_rows']}, got {actual.get('affected_rows')}"
        )
    if "columns" in expected and actual.get("columns") != expected["columns"]:
        raise CaseFailure(f"{sql}: columns expected {expected['columns']}, got {actual.get('columns')}")
    if "rows" in expected:
        actual_rows = actual.get("rows")
        if not isinstance(actual_rows, list) or not equal_rows(
            actual_rows, expected["rows"], expected.get("unordered", False)
        ):
            raise CaseFailure(
                f"{sql}: rows mismatch\nexpected={json.dumps(expected['rows'], ensure_ascii=False)}"
                f"\nactual={json.dumps(actual_rows, ensure_ascii=False)}"
            )


def run_case(binary, manifest_dir, case, work_root):
    data_dir = work_root / case["id"]
    data_dir.mkdir(parents=True)
    test_path = manifest_dir / case["test"]
    result_path = manifest_dir / case["result"]
    lines = test_path.read_text(encoding="utf-8").splitlines()
    expected_results = json.loads(result_path.read_text(encoding="utf-8"))
    statements = [line.strip() for line in lines if line.strip()]
    sql_count = sum(not line.startswith("--") for line in statements)
    if sql_count != len(expected_results):
        raise CaseFailure(f"assertion count {len(expected_results)} != SQL count {sql_count}")
    session = Session(binary, data_dir)
    assertion = 0
    try:
        for statement in statements:
            if statement == "--restart":
                session.shutdown()
                session = Session(binary, data_dir)
                continue
            if statement.startswith("--mode "):
                session.set_mode(statement.split()[1])
                continue
            if statement.startswith("--"):
                raise CaseFailure(f"unknown driver directive: {statement}")
            response = session.execute(statement)
            check_response(response, expected_results[assertion], statement)
            assertion += 1
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    binary = args.binary.resolve()
    manifest_path = args.manifest.resolve()
    manifest_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    work_root = Path(tempfile.mkdtemp(prefix="minivectordb-public-"))
    passed = 0
    try:
        for case in manifest["cases"]:
            try:
                run_case(binary, manifest_dir, case, work_root)
            except Exception as exc:
                print(f"FAIL {case['id']}: {exc}", file=sys.stderr)
                print(f"data kept at {work_root}", file=sys.stderr)
                return 1
            passed += 1
            print(f"PASS {case['id']}")
        print(f"{passed}/{len(manifest['cases'])} public cases passed")
        if not args.keep:
            shutil.rmtree(work_root)
        return 0
    finally:
        if args.keep:
            print(f"data kept at {work_root}")


if __name__ == "__main__":
    sys.exit(main())
