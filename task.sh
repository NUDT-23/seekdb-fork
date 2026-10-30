#!/bin/bash
echo "=== TASK 1: diagnose + unstick BCR $(date '+%F %T') ==="
echo "host: $(hostname)"
echo "--- setup2.log tail:"
tail -n 6 /tmp/setup2.log 2>/dev/null || echo "(no log)"
echo "--- proc state:"
ps aux | grep -E "git|amd_env|build.py|rustup" | grep -v grep | head -12
echo "--- workspace:"
du -sh /workspace/bcr 2>/dev/null
ls /workspace/bcr 2>/dev/null | head -3
echo "--- toolchain:"
echo "rustup: $(command -v rustup || echo MISSING)"
echo "cargo:  $(command -v cargo 2>/dev/null || echo MISSING)"
echo "bazel cache: $(ls ~/.cache/bazel/binaries/9.3.0/ 2>/dev/null | head -2)"
echo "deps/3rd: $(ls /workspace/seekdb/deps/3rd 2>/dev/null | head -3)"
echo "--- kill hung clone:"
pkill -f "bazel-central-registry" 2>/dev/null && echo "killed bcr clone" || echo "no bcr clone proc"
pkill -f "amd_env.sh" 2>/dev/null && echo "killed amd_env" || echo "no amd_env proc"
sleep 1
echo "--- start BCR via codeload tarball:"
rm -rf /workspace/bcr /tmp/bcr.tgz /tmp/bcr.status
nohup bash -c 'curl -sL --max-time 1800 https://codeload.github.com/bazelbuild/bazel-central-registry/tar.gz/refs/heads/main -o /tmp/bcr.tgz && mkdir -p /workspace/bcr && tar -xzf /tmp/bcr.tgz -C /workspace/bcr --strip-components=1 && echo DONE > /tmp/bcr.status || echo FAIL > /tmp/bcr.status' >/dev/null 2>&1 &
echo "started (status in /tmp/bcr.status)"
echo "=== TASK 1 END ==="
