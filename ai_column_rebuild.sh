#!/bin/bash
# One-shot rebuild of the seekdb workspace for quiz-3 (AI COLUMN) on the AMD cloud container.
# Usage: bash <(curl -sL "https://api.github.com/repos/NUDT-23/seekdb-fork/contents/ai_column_rebuild.sh?ref=dev-env")
set -x
export RUSTUP_DIST_SERVER=https://static.rust-lang.org
export RUSTUP_UPDATE_ROOT=https://static.rust-lang.org/rustup
BRANCH=feat/ai-column-2026

echo "############ STEP 0: source ($BRANCH) ############"
mkdir -p /workspace
cd /workspace
rm -rf /workspace/seekdb
curl -sL --max-time 1800 "https://codeload.github.com/NUDT-23/seekdb-fork/tar.gz/refs/heads/$BRANCH" -o /tmp/s.tgz \
  && mkdir -p /workspace/seekdb \
  && tar -xzf /tmp/s.tgz -C /workspace/seekdb --strip-components=1 \
  && echo SRC_OK || { echo SRC_FAIL; exit 1; }
cd /workspace/seekdb || { echo NO_SRC_DIR; exit 1; }

echo "############ STEP 0.5: host tools ############"
apt-get install -y rpm2cpio cpio >/dev/null 2>&1
echo "rpm2cpio=$(command -v rpm2cpio) cpio=$(command -v cpio)"

echo "############ STEP 1: bazel downloader + cargo index patches ############"
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
grep -n "usr/bin/rpm2cpio\|usr/bin/cpio" deps/init/prepare.py | head -4

echo "############ STEP 2: BCR via codeload tarball ############"
if [ ! -d /workspace/bcr/modules ]; then
  rm -rf /workspace/bcr
  curl -sL --max-time 1800 https://codeload.github.com/bazelbuild/bazel-central-registry/tar.gz/refs/heads/main -o /tmp/bcr.tgz \
    && mkdir -p /workspace/bcr \
    && tar -xzf /tmp/bcr.tgz -C /workspace/bcr --strip-components=1 \
    && echo BCR_OK || echo BCR_FAIL
else
  echo BCR_REUSE
fi

echo "############ STEP 3: bazel binary ############"
BAZEL_BIN="$HOME/.cache/bazel/binaries/9.3.0/bazel-9.3.0-linux-x86_64"
mkdir -p "$(dirname "$BAZEL_BIN")"
if [ ! -x "$BAZEL_BIN" ]; then
  curl -sL --max-time 900 https://github.com/bazelbuild/bazel/releases/download/9.3.0/bazel-9.3.0-linux-x86_64 -o "$BAZEL_BIN" && chmod +x "$BAZEL_BIN"
fi
echo "bazel=$(ls -la "$BAZEL_BIN" 2>/dev/null)"

echo "############ STEP 4: rustup ############"
if ! command -v rustup >/dev/null 2>&1; then
  curl -sSL --max-time 900 https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init -o /tmp/rustup-init
  chmod +x /tmp/rustup-init
  /tmp/rustup-init -y --profile minimal --default-toolchain none --no-modify-path
fi
echo "rustup: $(command -v rustup)"

echo "############ STEP 5: build.py init ############"
if [ ! -d deps/3rd ]; then
  ./build.py init && echo INIT_OK || echo INIT_FAIL
else
  echo INIT_REUSE
fi

echo "############ STEP 6: build seekdb ############"
./build.py build seekdb --registry=file:///workspace/bcr
echo "BUILD_EXIT=$?"
ls -la build/bin/src/observer/seekdb 2>/dev/null || echo BIN_MISSING
echo "############ BUILD PHASE DONE $(date '+%F %T') ############"
