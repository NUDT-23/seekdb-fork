#!/bin/bash
# One-shot build setup for the seekdb AI function work on the AMD cloud box.
#   bash /tmp/build.sh
# Steps are idempotent: everything that already exists is skipped, so the script
# can be re-run after a network failure.
set -x
export RUSTUP_DIST_SERVER=https://static.rust-lang.org
export RUSTUP_UPDATE_ROOT=https://static.rust-lang.org/rustup
REF="${BUILD_REF:-feat/ai-enhancement}"
# github.com is heavily throttled from some clouds, a gitee mirror of the fork can
# be used instead:  SRC_REPO=https://gitee.com/<user>/seekdb-fork.git bash build.sh
SRC_REPO="${SRC_REPO:-https://github.com/NUDT-23/seekdb-fork.git}"

echo "############ STEP 0: source ############"
cd /workspace || exit 1
if [ ! -f /workspace/seekdb/BUILD.bazel ]; then
  rm -rf /workspace/seekdb
  for i in 1 2 3 4 5; do
    git clone --depth 1 -b "$REF" "$SRC_REPO" /workspace/seekdb && break
    echo "CLONE_RETRY $i"
    rm -rf /workspace/seekdb
    sleep 5
  done
fi
[ -f /workspace/seekdb/BUILD.bazel ] && echo SRC_OK || { echo SRC_FAIL; exit 1; }
cd /workspace/seekdb || exit 1

echo "############ STEP 1: host tools ############"
command -v rpm2cpio >/dev/null 2>&1 || apt-get install -y rpm2cpio cpio >/dev/null 2>&1
echo "rpm2cpio=$(command -v rpm2cpio) cpio=$(command -v cpio)"

echo "############ STEP 2: bazel downloader patches ############"
sed -i 's#releases.bazel.build/{version}/release#github.com/bazelbuild/bazel/releases/download#' bazel/binaries.json
cat > bazel/downloader.cfg <<'EOF'
rewrite (?:mirror\.bazel\.build/)?github\.com/(.*) https://github.com/$1
allow github.com
allow codeload.github.com
allow raw.githubusercontent.com
allow www.netlib.org
allow archives.boost.io
block *
EOF
sed -i 's|sparse+https://rsproxy.cn/index/|sparse+https://index.crates.io/|g' deps/init/prepare.py bazel/prepare_cargo.py
python3 - <<'PYEOF'
p = "/workspace/seekdb/deps/init/prepare.py"
s = open(p).read()
s = s.replace("'rpm2cpio'", "'/usr/bin/rpm2cpio'").replace("'cpio'", "'/usr/bin/cpio'")
open(p, "w").write(s)
print("prepare.py patched")
PYEOF

echo "############ STEP 3: local bazel central registry ############"
if [ ! -d /workspace/bcr/modules ]; then
  rm -rf /workspace/bcr
  for i in 1 2 3; do
    git clone --depth 1 https://github.com/bazelbuild/bazel-central-registry.git /workspace/bcr && break
    echo "BCR_RETRY $i"
    rm -rf /workspace/bcr
    sleep 5
  done
fi
[ -d /workspace/bcr/modules ] && echo BCR_OK || echo BCR_FAIL

echo "############ STEP 4: bazel binary ############"
BAZEL_BIN="$HOME/.cache/bazel/binaries/9.3.0/bazel-9.3.0-linux-x86_64"
mkdir -p "$(dirname "$BAZEL_BIN")"
if [ ! -x "$BAZEL_BIN" ]; then
  curl -sL --max-time 900 -o "$BAZEL_BIN" \
    https://github.com/bazelbuild/bazel/releases/download/9.3.0/bazel-9.3.0-linux-x86_64
  chmod +x "$BAZEL_BIN"
fi
ls -l "$BAZEL_BIN"

echo "############ STEP 5: rustup ############"
if ! command -v rustup >/dev/null 2>&1; then
  curl -sSL --max-time 900 https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init -o /tmp/rustup-init
  chmod +x /tmp/rustup-init
  /tmp/rustup-init -y --profile minimal --default-toolchain none --no-modify-path
fi
echo "rustup: $(command -v rustup)"

echo "############ STEP 6: build.py init ############"
if [ ! -d deps/3rd ]; then
  for i in 1 2 3; do
    ./build.py init && break
    echo "INIT_RETRY $i"
    sleep 5
  done
fi
[ -d deps/3rd ] && echo INIT_OK || echo INIT_FAIL

echo "############ STEP 7: libaio ############"
apt-get install -y libaio1t64 >/dev/null 2>&1
if [ -f /usr/lib/x86_64-linux-gnu/libaio.so.1t64.0.2 ]; then
  rm -f /usr/lib/x86_64-linux-gnu/libaio.so.1
  cp /usr/lib/x86_64-linux-gnu/libaio.so.1t64.0.2 /usr/lib/x86_64-linux-gnu/libaio.so.1
  ldconfig
fi

echo "############ STEP 8: build seekdb ############"
./build.py build seekdb --registry=file:///workspace/bcr > /workspace/build.log 2>&1
rc=$?
echo "BUILD_EXIT=$rc"
if [ "$rc" != "0" ]; then
  grep -nE "error:" /workspace/build.log | head -20
else
  ls -l build/bin/src/observer/seekdb
fi
echo "############ ALL DONE $(date '+%F %T') ############"
