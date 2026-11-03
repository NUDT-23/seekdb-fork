#!/bin/bash
echo "===== OS ====="
. /etc/os-release 2>/dev/null; echo "$PRETTY_NAME  $(uname -r)"
echo "===== CPU/MEM ====="
nproc; free -g | head -2
echo "===== DISK ====="
df -h /workspace /tmp / 2>/dev/null | grep -v Filesystem
echo "===== /opt ====="
ls -1 /opt
echo "===== deps caches ====="
for d in /opt/*deps* /opt/*cache* /root/.cache/bazel /workspace/.cache; do
  [ -e "$d" ] && { echo "-- $d"; ls -1 "$d" | head -10; du -sh "$d" 2>/dev/null; }
done
echo "===== toolchain ====="
for b in bazel bazelisk gcc g++ cmake ninja ccache bison flex python3 rustc cargo; do
  printf "%-10s %s\n" "$b" "$(command -v $b || echo MISSING)"
done
echo "===== existing seekdb ====="
ls -d /workspace/seekdb /root/seekdb /opt/seekdb 2>/dev/null
find / -maxdepth 4 -name "BUILD.bazel" -path "*seekdb*" 2>/dev/null | head -3
echo "===== workspace ====="
ls -1 /workspace | head -20
echo "===== env hints ====="
env | grep -iE "http_proxy|https_proxy|no_proxy|bazel|seekdb" | head
echo "===== sshd ====="
(pgrep -a sshd | head -3) || echo "no sshd"
echo "===== net ====="
for u in https://api.github.com https://codeload.github.com https://gitee.com; do
  printf "%-32s %s\n" "$u" "$(curl -s -o /dev/null -m 8 -w '%{http_code}' $u)"
done
echo "===== github speed (20s) ====="
timeout 20 curl -s -o /tmp/spd.bin "https://codeload.github.com/NUDT-23/seekdb-fork/tar.gz/refs/heads/dev-env" && ls -l /tmp/spd.bin | awk '{print $5" bytes in <=20s"}'
echo "===== done ====="
