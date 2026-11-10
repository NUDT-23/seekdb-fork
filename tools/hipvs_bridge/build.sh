#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HIPVS_ROOT=${HIPVS_ROOT:-/opt/hipvs}
ROCM_ROOT=${ROCM_ROOT:-/opt/rocm}
HIPCC=${HIPCC:-$ROCM_ROOT/bin/hipcc}
OUTPUT=${1:-$SCRIPT_DIR/libseekdb_cuvs_bridge.so}

[[ -x "$HIPCC" ]] || { echo "hipcc is not executable: $HIPCC" >&2; exit 2; }
[[ -f "$HIPVS_ROOT/include/cuvs/core/c_api.h" ]] || {
  echo "hipVS headers are missing under $HIPVS_ROOT/include" >&2
  exit 2
}
[[ -f "$HIPVS_ROOT/lib/libcuvs_c.so" ]] || {
  echo "libcuvs_c.so is missing under $HIPVS_ROOT/lib" >&2
  exit 2
}

mkdir -p "$(dirname "$OUTPUT")"
BUILD_DIR="$(dirname "$OUTPUT")"

# Stage 1 (optional): compile the exact-graph HIP kernels.  On any compile
# failure the bridge is still built WITHOUT them (-DSK_NO_EXACT) so a
# toolchain hiccup can never turn into failMsg=3 - the worst case is the
# NN_DESCENT fallback path instead of a zero-scored submission.
HIP_OBJ=""
EXTRA_DEFS=()
if [[ -f "$SCRIPT_DIR/sk_exact_knn.hip" ]]; then
  if "$HIPCC" -O2 -fPIC -c "$SCRIPT_DIR/sk_exact_knn.hip" \
       -I"$HIPVS_ROOT/include" -I"$ROCM_ROOT/include" \
       -o "$BUILD_DIR/sk_exact_knn.o" 2> "$BUILD_DIR/sk_exact_knn.err"; then
    HIP_OBJ="$BUILD_DIR/sk_exact_knn.o"
    echo "built $HIP_OBJ"
  else
    echo "WARNING: sk_exact_knn.hip failed to compile - building bridge without the exact-graph path:" >&2
    cat "$BUILD_DIR/sk_exact_knn.err" >&2
    EXTRA_DEFS+=(-DSK_NO_EXACT)
  fi
fi

# Stage 2: the bridge itself (+ kernels when stage 1 succeeded).
"$HIPCC" \
  -shared \
  -fPIC \
  -O2 \
  -D__HIP_PLATFORM_AMD__ \
  ${EXTRA_DEFS[@]+"${EXTRA_DEFS[@]}"} \
  -I"$HIPVS_ROOT/include" \
  -I"$ROCM_ROOT/include" \
  "$SCRIPT_DIR/seekdb_cuvs_bridge.c" \
  $HIP_OBJ \
  -L"$HIPVS_ROOT/lib" \
  -Wl,-soname,libseekdb_cuvs_bridge.so \
  -Wl,-rpath,"$HIPVS_ROOT/lib:$ROCM_ROOT/lib" \
  -lcuvs_c \
  -o "$OUTPUT"

echo "built $OUTPUT"
