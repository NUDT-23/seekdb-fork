#!/bin/bash
# 在 AMD 实例上准备 seekdb 构建环境（源码下载 -> 依赖复制 -> 编译）
set -x

BR="${1:-feat/ai-enhancement-2026}"
WS=/workspace
CACHE=$(ls -d /opt/seekdb-deps-cache/*/ 2>/dev/null | head -1)

mkdir -p $WS

# 1) 源码
if [ ! -s $WS/seekdb/CMakeLists.txt ]; then
  for i in 1 2 3; do
    curl -sL --retry 2 --max-time 1800 -o $WS/src.tgz \
      "https://codeload.github.com/NUDT-23/seekdb-fork/tar.gz/refs/heads/${BR}"
    if tar -tzf $WS/src.tgz >/dev/null 2>&1; then break; fi
    echo "download retry $i"; sleep 3
  done
  rm -rf $WS/seekdb.raw && mkdir -p $WS/seekdb.raw
  tar -xzf $WS/src.tgz -C $WS/seekdb.raw --strip-components=1
  rm -rf $WS/seekdb && mv $WS/seekdb.raw $WS/seekdb
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
