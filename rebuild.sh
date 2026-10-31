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

echo "############ STEP 0.5: host tools ############"
# rpm2cpio/cpio are required by deps/init/prepare.py to unpack dependency rpms
apt-get install -y rpm2cpio cpio >/dev/null 2>&1
echo "rpm2cpio=$(command -v rpm2cpio) cpio=$(command -v cpio)"

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

# prepare.py invokes rpm2cpio/cpio by bare name; the build.py subprocess PATH does not
# resolve them, so pin absolute paths (patch is idempotent).
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
fi

echo "############ STEP 3: bazel binary cache ############"
BAZEL_BIN="$HOME/.cache/bazel/binaries/9.3.0/bazel-9.3.0-linux-x86_64"
mkdir -p "$(dirname "$BAZEL_BIN")"
if [ ! -x "$BAZEL_BIN" ]; then
  curl -sL --max-time 900 https://github.com/bazelbuild/bazel/releases/download/9.3.0/bazel-9.3.0-linux-x86_64 -o "$BAZEL_BIN" && chmod +x "$BAZEL_BIN"
fi

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

echo "############ STEP 5.5: libaio.so.1 for seekdb runtime ############"
# seekdb links against libaio.so.1; Ubuntu 24.04 only ships libaio.so.1t64 (different
# soname) and the packaged libaio.so.1 is a symlink that the runtime refuses to load.
apt-get install -y libaio1t64 >/dev/null 2>&1
if [ -f /usr/lib/x86_64-linux-gnu/libaio.so.1t64.0.2 ]; then
  rm -f /usr/lib/x86_64-linux-gnu/libaio.so.1
  cp /usr/lib/x86_64-linux-gnu/libaio.so.1t64.0.2 /usr/lib/x86_64-linux-gnu/libaio.so.1
  ldconfig
  echo "libaio.so.1 installed as regular file"
fi

echo "############ STEP 6: build seekdb ############"
./build.py build seekdb --registry=file:///workspace/bcr
echo "BUILD_EXIT=$?"
ls -la build/bin/src/observer/seekdb 2>/dev/null || echo BIN_MISSING

echo "############ STEP 7: mock AI service ############"
curl -sL -H "Accept: application/vnd.github.raw" "https://api.github.com/repos/NUDT-23/seekdb-fork/contents/mock_ai_service.py?ref=dev-env" -o /tmp/mock_ai.py
(nohup python3 /tmp/mock_ai.py > /tmp/mock_ai.log 2>&1 &)
sleep 2
curl -s http://127.0.0.1:8000/health; echo

echo "############ STEP 8: start seekdb (default params) ############"
# NOTE: do NOT pass --parameter here; an invalid/unsupported combination makes seekdb
# exit with -4147 (OB_INVALID_CONFIG). Default parameters work on this container.
rm -rf /workspace/sd_run; mkdir -p /workspace/sd_run; cd /workspace/sd_run
(nohup /workspace/seekdb/build/bin/src/observer/seekdb --nodaemon -P 2881 --base-dir /workspace/sd_run > /workspace/sd_run/console.log 2>&1 &)
sleep 45
/workspace/seekdb/deps/3rd/u01/obclient/bin/obclient -h127.0.0.1 -P2881 -uroot -e 'select 1' 2>&1 | head -3

echo "############ ALL DONE $(date '+%F %T') ############"
