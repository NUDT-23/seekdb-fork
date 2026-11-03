#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 mysqltest 的方式（同一个连接、顺序执行）重放官方 ai_function_enhancement.test。

用法：
  run_single.py 2881 10000000   # 用默认 10s 会话超时，复现平台行为
  run_single.py 2881 60000000   # 把会话超时放大到 60s，看是否只是 DDL 慢

输出：每条语句的返回情况（--force 让失败后继续），便于与官方 .result 比对。
"""
import re
import subprocess
import sys

TEST = "/workspace/seekdb/tools/deploy/mysql_test/test_suite/ai_function/t/ai_function_enhancement.test"
DB = "ai_fx"
PORT = sys.argv[1] if len(sys.argv) > 1 else "2881"
TIMEOUT = sys.argv[2] if len(sys.argv) > 2 else "10000000"
MYSQL = ["mysql", "-h127.0.0.1", "-P" + PORT, "-uroot", "--force", "--batch", "--raw", "-v"]


def statements(path):
    out, buf = [], []
    for raw in open(path, encoding="utf-8", errors="replace").read().split("\n"):
        line = raw.rstrip()
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("--"):
            continue
        if re.match(r"^(connection|let|exec|disable_|enable_|shutdown|sleep|source|delimiter)\b", s):
            continue
        if s.startswith("while") or s.startswith("if ") or s == "}" or s.startswith("dec "):
            continue
        buf.append(line)
        if s.endswith(";"):
            out.append("\n".join(buf).strip())
            buf = []
    if buf:
        out.append("\n".join(buf).strip())
    return out


def run(sql, label):
    p = subprocess.run(MYSQL, input=sql, capture_output=True, text=True, timeout=900)
    print("======== %s ========" % label)
    print(p.stdout.strip())
    if p.stderr.strip():
        print("---- stderr ----")
        print(p.stderr.strip()[:2000])
    return p


stmts = statements(TEST)
print("statements: %d" % len(stmts))

head = ("set ob_query_timeout=%s;\n"
        "drop database if exists %s;\n"
        "create database %s;\n"
        "use %s;\n" % (TIMEOUT, DB, DB, DB))

# 第一次：模型不存在，DROP 直接返回 11118
run(head + ";\n".join(stmts) + ";\n", "run1 (fresh) timeout=%s" % TIMEOUT)
# 第二次：上一轮可能残留模型，DROP 会真正走一次 RS DDL —— 平台上的超时就发生在这一步
run(head + ";\n".join(stmts) + ";\n", "run2 (models may exist)")
