#!/bin/bash
echo "--- hostname: $(hostname)  time: $(date '+%F %T')"
echo "--- setup2.log tail:"
tail -n 8 /tmp/setup2.log 2>/dev/null || echo "no setup2.log"
echo "--- running procs (git/build/curl):"
ps aux | grep -E "git|build.py|amd_env|rustup" | grep -v grep | head -8
echo "--- workspace:"
ls /workspace/ 2>/dev/null
ls /workspace/bcr 2>/dev/null | head -3
echo "--- seekdb dir:"
ls /workspace/seekdb 2>/dev/null | head -8
ls /workspace/seekdb/deps/3rd 2>/dev/null | head -5
echo "--- rustup: $(command -v rustup || echo MISSING)  bazel cache: $(ls ~/.cache/bazel/binaries/9.3.0/ 2>/dev/null | head -2)"
echo "--- disk:"
df -h /workspace | tail -1
