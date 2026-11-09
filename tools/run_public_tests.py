#!/usr/bin/env python3
"""Local driver for the public MiniVectorDB mysqltest-style samples.

This is deliberately independent from the database implementation. It speaks
our interim one-line-JSON protocol and checks the package's result assertions.
The official wire protocol will replace only the transport, not these checks.
"""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


class CaseFailure(Exception):
    pass


def start(binary, data_dir):
    proc = subprocess.Popen(
        [str(binary), "--data-dir", str(data_dir)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    ready = proc.stdout.readline()
    try:
        value = json.loads(ready)
    except Exception as exc:
        raise CaseFailure(f"bad ready line {ready!r}: {exc}")
    if value.get("ready") is not True:
        raise CaseFailure(f"missing ready marker: {ready!r}")
    return proc


def request(proc, text):
    proc.stdin.write(text + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise CaseFailure(f"database closed stdout after {text!r}; rc={proc.poll()}")
    try:
        return json.loads(line)
    except Exception as exc:
        raise CaseFailure(f"non-JSON response to {text!r}: {line!r}: {exc}")


def stop(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    if proc.stdin:
        proc.stdin.close()
    if proc.stdout:
        proc.stdout.close()


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
        code = actual.get("error") or actual.get("code")
        if not code:
            raise CaseFailure(f"{sql}: empty error code: {actual}")
        if code != expected["error"]:
            raise CaseFailure(f"{sql}: expected error {expected['error']}, got {code}")
        return
    if actual.get("ok") is not True:
        raise CaseFailure(f"{sql}: expected success, got {actual}")
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
    proc = start(binary, data_dir)
    assertion = 0
    try:
        for statement in statements:
            if statement == "--restart":
                stop(proc)
                proc = start(binary, data_dir)
                continue
            if statement.startswith("--mode "):
                response = request(proc, statement)
                if response.get("ok") is not True:
                    raise CaseFailure(f"mode switch failed: {response}")
                continue
            if statement.startswith("--"):
                raise CaseFailure(f"unknown driver directive: {statement}")
            response = request(proc, statement)
            check_response(response, expected_results[assertion], statement)
            assertion += 1
    finally:
        stop(proc)


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
