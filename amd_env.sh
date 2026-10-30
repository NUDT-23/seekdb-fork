#!/bin/bash
# Rebuild the seekdb build environment on the AMD cloud container.
# Usage inside the container:  bash /tmp/amd_env.sh
set -e

export RUSTUP_DIST_SERVER=https://static.rust-lang.org
export RUSTUP_UPDATE_ROOT=https://static.rust-lang.org/rustup

# 0) source
cd /workspace
if [ ! -f /workspace/seekdb/BUILD.bazel ]; then
  git clone --depth 1 https://github.com/oceanbase/seekdb.git /workspace/seekdb
fi
cd /workspace/seekdb

# 1) Bazel binary from GitHub release (releases.bazel.build is proxy-blocked)
sed -i 's#releases.bazel.build/{version}/release#github.com/bazelbuild/bazel/releases/download#' bazel/binaries.json

# 2) Downloader: talk to github.com directly (gh-proxy.com is blocked)
cat > bazel/downloader.cfg <<'EOF'
rewrite (?:mirror\.bazel\.build/)?github\.com/(.*) https://github.com/$1
allow github.com
allow codeload.github.com
allow raw.githubusercontent.com
allow www.netlib.org
allow archives.boost.io
block *
EOF

# 3) Cargo sparse index (rsproxy.cn is blocked)
sed -i 's|sparse+https://rsproxy.cn/index/|sparse+https://index.crates.io/|g' \
  deps/init/prepare.py bazel/prepare_cargo.py

# 4) Local Bazel Central Registry (bcr.bazel.build is blocked)
[ -d /workspace/bcr ] || git clone --depth 1 \
  https://github.com/bazelbuild/bazel-central-registry.git /workspace/bcr

# 5) Pre-seed the Bazel launcher cache (build.py verifies sha256 itself)
BAZEL_BIN="$HOME/.cache/bazel/binaries/9.3.0/bazel-9.3.0-linux-x86_64"
mkdir -p "$(dirname "$BAZEL_BIN")"
if [ ! -x "$BAZEL_BIN" ]; then
  curl -sL https://github.com/bazelbuild/bazel/releases/download/9.3.0/bazel-9.3.0-linux-x86_64 -o "$BAZEL_BIN"
  chmod +x "$BAZEL_BIN"
fi

# 6) rustup (sh.rustup.rs / rsproxy installer are blocked, use the static binary)
if ! command -v rustup >/dev/null 2>&1; then
  curl -sSL https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init -o /tmp/rustup-init
  chmod +x /tmp/rustup-init
  /tmp/rustup-init -y --profile minimal --default-toolchain none --no-modify-path
fi

# 7) prepare dependencies (rust toolchain + deps/3rd)
if [ ! -d /workspace/seekdb/deps/3rd ]; then
  ./build.py init
fi

echo "ENV_READY"
echo "build: cd /workspace/seekdb && ./build.py build seekdb --registry=file:///workspace/bcr"
