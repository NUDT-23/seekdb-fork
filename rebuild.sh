#!/bin/bash
# One-shot rebuild of the seekdb AI-function workspace on the AMD cloud container.
# Usage: bash <(curl -sL "https://api.github.com/repos/NUDT-23/seekdb-fork/contents/rebuild.sh?ref=dev-env")
set -x
export RUSTUP_DIST_SERVER=https://static.rust-lang.org
export RUSTUP_UPDATE_ROOT=https://static.rust-lang.org/rustup

echo "############ STEP 0: source ############"
mkdir -p /workspace
cd /workspace
if [ ! -f /workspace/seekdb/BUILD.bazel ]; then
  rm -rf /workspace/seekdb
  curl -sL --max-time 1800 https://codeload.github.com/NUDT-23/seekdb-fork/tar.gz/refs/heads/feat/ai-enhancement -o /tmp/s.tgz \
    && mkdir -p /workspace/seekdb \
    && tar -xzf /tmp/s.tgz -C /workspace/seekdb --strip-components=1 \
    && echo SRC_OK || echo SRC_FAIL
fi
cd /workspace/seekdb || { echo NO_SRC_DIR; exit 1; }
ls -la | head -5

echo "############ STEP 1: bazel downloader patches ############"
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
echo PATCH_OK

echo "############ STEP 2: BCR via codeload tarball ############"
if [ ! -d /workspace/bcr/modules ]; then
  rm -rf /workspace/bcr
  curl -sL --max-time 1800 https://codeload.github.com/bazelbuild/bazel-central-registry/tar.gz/refs/heads/main -o /tmp/bcr.tgz \
    && mkdir -p /workspace/bcr \
    && tar -xzf /tmp/bcr.tgz -C /workspace/bcr --strip-components=1 \
    && echo BCR_OK || echo BCR_FAIL
fi
ls /workspace/bcr | head -3

echo "############ STEP 3: bazel binary cache ############"
BAZEL_BIN="$HOME/.cache/bazel/binaries/9.3.0/bazel-9.3.0-linux-x86_64"
mkdir -p "$(dirname "$BAZEL_BIN")"
if [ ! -x "$BAZEL_BIN" ]; then
  curl -sL --max-time 900 https://github.com/bazelbuild/bazel/releases/download/9.3.0/bazel-9.3.0-linux-x86_64 -o "$BAZEL_BIN" && chmod +x "$BAZEL_BIN"
fi
ls -la "$BAZEL_BIN" 2>/dev/null || echo BAZEL_BIN_MISSING

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
fi
ls deps/3rd 2>/dev/null | head -5

echo "############ STEP 6: build seekdb ############"
./build.py build seekdb --registry=file:///workspace/bcr
echo "BUILD_EXIT=$?"
ls -la build/bin/src/observer/seekdb 2>/dev/null || echo BIN_MISSING
echo "############ ALL DONE $(date '+%F %T') ############"
