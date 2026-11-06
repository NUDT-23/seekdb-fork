#!/bin/bash
# One-shot rebuild for quiz-3 (AI COLUMN) on a fresh AMD cloud container.
# Encodes every workaround discovered on 2026-11-06:
#   - HTTP/1.1 for all GitHub downloads (proxy throttles HTTP/2 badly)
#   - rpm/cpio required by deps/init (Ubuntu 24 has no rpm2cpio package)
#   - cargo must NOT point at rsproxy.cn (unreachable -> 403 tunnel failed)
#   - rust toolchain pinned to 1.98.1, native target only (saves ~20 min)
#   - CMake VerifyGlobs scans 7.4GB deps and hangs make -> disabled
set -x
export RUSTUP_DIST_SERVER=https://static.rust-lang.org
export RUSTUP_UPDATE_ROOT=https://static.rust-lang.org/rustup
export PATH=/root/.cargo/bin:/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin:/sbin
BRANCH=feat/ai-column-2026

echo "############ STEP 0: source ($BRANCH) ############"
mkdir -p /workspace
cd /workspace
rm -rf /workspace/seekdb
curl --http1.1 -sL --max-time 1800 "https://codeload.github.com/NUDT-23/seekdb-fork/tar.gz/refs/heads/$BRANCH" -o /tmp/s.tgz \
  && mkdir -p /workspace/seekdb \
  && tar -xzf /tmp/s.tgz -C /workspace/seekdb --strip-components=1 \
  && echo SRC_OK || { echo SRC_FAIL; exit 1; }
cd /workspace/seekdb || { echo NO_SRC_DIR; exit 1; }

echo "############ STEP 1: host tools ############"
apt-get update > /tmp/apt_up.log 2>&1
apt-get install -y rpm cpio > /tmp/apt_inst.log 2>&1
echo "rpm2cpio=$(command -v rpm2cpio) cpio=$(command -v cpio)"

echo "############ STEP 2: cargo config (no unreachable mirror) ############"
printf '[net]\ngit-fetch-with-cli = true\n' > /root/.cargo/config.toml
cat /root/.cargo/config.toml

echo "############ STEP 3: rustup ############"
if ! command -v rustup > /dev/null 2>&1; then
  curl --http1.1 -sSL --max-time 400 https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init -o /tmp/rustup-init
  chmod +x /tmp/rustup-init
  /tmp/rustup-init -y --profile minimal --default-toolchain none --no-modify-path
fi
echo "rustup: $(command -v rustup)"

echo "############ STEP 4: pin toolchain (native target only) ############"
cat > /workspace/seekdb/rust/rust-toolchain.toml <<'EOF'
[toolchain]
channel = "1.98.1"
profile = "minimal"
targets = ["x86_64-unknown-linux-gnu"]
EOF

echo "############ STEP 5: dependency init ############"
./build.sh init
echo "INIT_EXIT=$?"

echo "############ STEP 6: CMake configure ############"
./build.sh release
echo "CONFIG_EXIT=$?"

echo "############ STEP 7: disable VerifyGlobs (scans 7.4GB deps) ############"
echo '# verifyglobs disabled' > /workspace/seekdb/build_release/CMakeFiles/VerifyGlobs.cmake
python3 - <<'PY'
p1 = "/workspace/seekdb/build_release/Makefile"
p2 = "/workspace/seekdb/build_release/CMakeFiles/Makefile2"
old = ("\t$(CMAKE_COMMAND) -P /workspace/seekdb/build_release/CMakeFiles/VerifyGlobs.cmake\n"
       "\t$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)"
       " --check-build-system CMakeFiles/Makefile.cmake 0\n")
new = "\t@true\n"
for p in (p1, p2):
    s = open(p).read()
    if old in s:
        open(p, "w").write(s.replace(old, new, 1))
        print("patched", p)
PY

echo "############ STEP 8: build ############"
cd /workspace/seekdb/build_release
make -j128 seekdb
echo "BUILD_EXIT=$?"
ls -la /workspace/seekdb/build_release/src/observer/seekdb 2>/dev/null || echo BIN_MISSING
echo "############ DONE $(date '+%F %T') ############"
