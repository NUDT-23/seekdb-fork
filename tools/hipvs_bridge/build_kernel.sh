#!/usr/bin/env bash
# Build the exact-kNN kernel sidecar library (libseekdb_sk_exact.so) from
# kernels/sk_exact_knn.hip.  Called from the seekdb CMake build when
# OB_BUILD_CUVS=ON.  It DELIBERATELY never returns failure: on any toolchain
# problem the sidecar is simply left absent and the bridge falls back to
# cuvsCagraBuild at runtime - a kernel build hiccup can never fail the
# seekdb build itself.
#
#   build_kernel.sh <input.hip> <output.so>
set -u

IN=${1:-}
OUT=${2:-}
if [ -z "$IN" ] || [ -z "$OUT" ] || [ ! -f "$IN" ]; then
  echo "build_kernel: nothing to do (IN='$IN' OUT='$OUT')" >&2
  exit 0
fi
HIPVS_ROOT=${HIPVS_ROOT:-/opt/hipvs}
ROCM_ROOT=${ROCM_ROOT:-/opt/rocm}
HIPCC=${HIPCC:-$ROCM_ROOT/bin/hipcc}

OUT_DIR=$(dirname "$OUT")
ERR_FILE=$OUT_DIR/sk_kernel_build.err
mkdir -p "$OUT_DIR" 2>/dev/null || true
rm -f "$ERR_FILE" 2>/dev/null || true

# Primary path: the native hipcc of the eval machine (official ROCm).
# The offload arch is pinned: W7900D-class parts are gfx1100, and an
# unset/policy default could target the wrong silicon (kernels then fail
# to launch at runtime and the bridge silently falls back).
if [ -x "$HIPCC" ]; then
  if "$HIPCC" -O2 -fPIC -shared --offload-arch=gfx1100 "$IN" \
       -Wl,-soname,libseekdb_sk_exact.so \
       -o "$OUT" 2> "$ERR_FILE"; then
    echo "build_kernel: built $OUT"
    exit 0
  fi
  echo "build_kernel: hipcc failed, trying clang++ hip front-end:" >&2
  cat "$ERR_FILE" >&2 2>/dev/null || true
fi

# Fallback: clang++ driving the HIP front-end explicitly (rescue for
# toolchains whose hipcc wrapper is unusable).
CLANGXX=""
for c in "$ROCM_ROOT/lib/llvm/bin/clang++" "$ROCM_ROOT/llvm/bin/clang++"; do
  if [ -x "$c" ]; then CLANGXX=$c; break; fi
done
CLANG_INC=$(ls -d "$ROCM_ROOT"/lib/llvm/lib/clang/*/include 2>/dev/null | tail -1)

if [ -n "$CLANGXX" ]; then
  EXTRA_INC=""
  [ -n "$CLANG_INC" ] && EXTRA_INC="-I$CLANG_INC"
  # shellcheck disable=SC2086
  if "$CLANGXX" -O2 -fPIC -shared -x hip -nocudainc \
       -I"$ROCM_ROOT/include" $EXTRA_INC \
       -include __clang_hip_runtime_wrapper.h \
       --offload-arch=gfx1100 \
       "$IN" -L"$ROCM_ROOT/lib" -lamdhip64 \
       -Wl,-soname,libseekdb_sk_exact.so \
       -o "$OUT" 2>> "$ERR_FILE"; then
    echo "build_kernel: built $OUT (clang++ hip front-end)"
    exit 0
  fi
  echo "build_kernel: clang++ hip front-end failed:" >&2
  cat "$ERR_FILE" >&2 2>/dev/null || true
fi

echo "build_kernel: WARNING - $OUT not built; the bridge will use the cuvsCagraBuild fallback" >&2
exit 0
