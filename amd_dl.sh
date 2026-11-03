#!/bin/bash
# Download the seekdb source tarball for the competition branch, retrying until
# the archive is complete. Runs in the background, progress goes to the log.
cd /workspace || exit 1
URL="https://codeload.github.com/NUDT-23/seekdb-fork/tar.gz/refs/heads/$1"
OUT="/workspace/src.tgz"
rm -f "$OUT"
for i in $(seq 1 8); do
  echo "[$(date +%H:%M:%S)] try $i start, current size $(stat -c%s "$OUT" 2>/dev/null || echo 0)"
  curl -sL --retry 2 --retry-delay 3 -C - "$URL" -o "$OUT"
  rc=$?
  sz=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
  echo "[$(date +%H:%M:%S)] try $i done rc=$rc size=$sz"
  if tar -tzf "$OUT" >/dev/null 2>&1; then
    echo "TGZ_OK size=$sz"
    exit 0
  fi
  echo "archive incomplete, retrying"
  sleep 3
done
echo "TGZ_FAIL"
exit 1
