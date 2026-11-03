#!/bin/bash
# 在 AMD 实例上准备 seekdb 构建环境（源码解压 -> 依赖复制 -> 编译）
set -x

BR="${1:-feat/ai-enhancement-2026}"
WS=/workspace
CACHE=$(ls -d /opt/seekdb-deps-cache/*/ 2>/dev/null | head -1)

mkdir -p $WS

# 0) 上传的包可能落在 notebook 目录
for cand in $WS/src.tgz $WS/seekdb_hips_notebook/src.tgz; do
  if [ -s "$cand" ]; then mv "$cand" $WS/src.tgz; break; fi
done

# 1) 源码（本地已有可用包则跳过下载）
if ! tar -tzf $WS/src.tgz >/dev/null 2>&1; then
  for i in 1 2 3; do
    curl -sL --retry 2 --max-time 900 -o $WS/src.tgz \
      "https://codeload.github.com/NUDT-23/seekdb-fork/tar.gz/refs/heads/${BR}"
    if tar -tzf $WS/src.tgz >/dev/null 2>&1; then break; fi
    echo "download retry $i"; sleep 3
  done
fi

if [ ! -s $WS/seekdb/CMakeLists.txt ]; then
  rm -rf $WS/seekdb && mkdir -p $WS/seekdb
  tar -xzf $WS/src.tgz -C $WS/seekdb --strip-components=1
fi

# 2) 预编译依赖
if [ ! -d $WS/seekdb/deps/3rd/usr/local/oceanbase ]; then
  rm -rf $WS/seekdb/deps/3rd
  cp -a ${CACHE}3rd $WS/seekdb/deps/3rd
fi

# 3) cargo 走公网源
: > /root/.cargo/config.toml

cd $WS/seekdb || exit 1
nohup ./build.sh release --make -j64 > $WS/build.log 2>&1 &
echo "BUILD_PID=$!"
