#!/bin/bash
# Sync the changed sources of NUDT-23/seekdb-fork:feat/ai-enhancement into
# /workspace/seekdb and rebuild seekdb incrementally.
#
#   bash /tmp/sync.sh            # sync + build
#   bash /tmp/sync.sh --src-only # sync only
#
# Only the files touched by the AI function work are transferred, the build tree
# on the cloud box already has everything else.
set -u
REF="${SYNC_REF:-feat/ai-enhancement}"
SRC=/workspace/seekdb
API="https://api.github.com/repos/NUDT-23/seekdb-fork/contents"

FILES="
src/share/ai_service/ob_ai_service_struct.h
src/share/ai_service/ob_ai_service_struct.cpp
src/sql/engine/expr/ob_expr_ai/ob_ai_func.h
src/sql/engine/expr/ob_expr_ai/ob_ai_func_client.h
src/sql/engine/expr/ob_expr_ai/ob_ai_func_client.cpp
src/sql/engine/expr/ob_expr_ai/ob_ai_func_utils.h
src/sql/engine/expr/ob_expr_ai/ob_ai_func_utils.cpp
src/sql/engine/expr/ob_expr_ai/ob_expr_ai_prompt.h
src/sql/engine/expr/ob_expr_ai/ob_expr_ai_prompt.cpp
src/sql/engine/expr/ob_expr_ai/ob_expr_ai_complete.h
src/sql/engine/expr/ob_expr_ai/ob_expr_ai_complete.cpp
src/sql/engine/expr/ob_expr_ai/ob_expr_ai_embed.h
src/sql/engine/expr/ob_expr_ai/ob_expr_ai_embed.cpp
src/sql/engine/expr/ob_expr_ai/ob_expr_ai_rerank.h
src/sql/engine/expr/ob_expr_ai/ob_expr_ai_rerank.cpp
"

fail=0
n=0
for f in $FILES; do
  dst="$SRC/$f"
  mkdir -p "$(dirname "$dst")"
  code=$(curl -s -m 120 -o "$dst.new" -w "%{http_code}" -H "Accept: application/vnd.github.raw" "$API/$f?ref=$REF")
  if [ "$code" = "200" ] && [ -s "$dst.new" ]; then
    mv "$dst.new" "$dst"
    n=$((n + 1))
  else
    echo "SYNC_FAIL $code $f"
    rm -f "$dst.new"
    fail=1
  fi
done
echo "SYNCED=$n FAIL=$fail"
[ "$fail" = "0" ] || exit 1
[ "${1:-}" = "--src-only" ] && exit 0

cd "$SRC" || exit 1
# keep the build fixes that rebuild.sh applied to this working tree
./build.py build seekdb --registry=file:///workspace/bcr > /workspace/build.log 2>&1
rc=$?
echo "BUILD_EXIT=$rc"
if [ "$rc" != "0" ]; then
  grep -nE "error:|Error:|ERROR:" /workspace/build.log | head -30
  echo "---- context ----"
  grep -nE -A 3 "error:" /workspace/build.log | head -60
else
  ls -l build/bin/src/observer/seekdb
fi
