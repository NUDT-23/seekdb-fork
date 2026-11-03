#!/bin/bash
echo "===== deps cache ====="
du -sh /opt/seekdb-deps-cache 2>/dev/null
ls -1 /opt/seekdb-deps-cache | head
for d in /opt/seekdb-deps-cache/*/; do echo "-- $d"; ls -1 "$d" | head -8; done
echo "===== /opt/hips ====="
ls -1 /opt/hips 2>/dev/null | head -20
echo "===== scripts ====="
ls -1 /opt/seekdb-notebook
find /opt -maxdepth 3 \( -name "*.sh" -o -name "bazel*" -o -name "*.tar.gz" \) 2>/dev/null | head -20
echo "===== seekdb binary ====="
find / -maxdepth 6 -type f -name "seekdb" 2>/dev/null | head -5
find / -maxdepth 6 -type f -name "observer" 2>/dev/null | head -5
echo "===== java ====="
(command -v java && java -version 2>&1 | head -2) || echo "no java"
echo "===== git ====="
command -v git; git --version
echo "===== env full ====="
env | grep -E "SEEDB|CACHE|PROXY|PATH=" | head -12
echo "===== speed test 15s ====="
curl -s -o /tmp/spd.bin --max-time 15 -w "downloaded=%{size_download} speed=%{speed_download}B/s\n" "https://codeload.github.com/NUDT-23/seekdb-fork/tar.gz/refs/heads/dev-env" || echo "curl failed rc=$?"
echo "===== done ====="
