#!/bin/bash
# Unpack the source, install the prebuilt dependencies and start the release build.
set -x
cd /workspace || exit 1
if [ ! -f src.tgz ]; then echo "NO_TGZ"; exit 1; fi
if [ ! -d seekdb ]; then
  tar -xzf src.tgz || { echo "UNTAR_FAIL"; exit 1; }
  mv seekdb-fork-* seekdb || true
fi
cd /workspace/seekdb || exit 1
if [ ! -d deps/3rd ]; then
  cp -a /opt/seekdb-deps-cache/*/3rd deps/3rd || { echo "DEPS_FAIL"; exit 1; }
fi
printf "" > /root/.cargo/config.toml
./build.sh release --make > /workspace/build.log 2>&1
echo "BUILD_RC=$?"
tail -5 /workspace/build.log
ls -l /workspace/seekdb/build_release/src/observer/seekdb 2>/dev/null
