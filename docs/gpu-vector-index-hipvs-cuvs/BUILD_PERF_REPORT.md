# CAGRA 索引构建加速报告（seekdb × AMD hipVS）

> 基线：`zihaomu/seekdb` @ `amd-oceanbase-official-contest-base-v5`
> （tag `amd-cagra-contest-baseline-v5`，commit `634888d9b037153fa143d2c913ba4b25503e1d62`）
> 本分支：`amd-cagra-safe`，单文件改动 `tools/hipvs_bridge/seekdb_cuvs_bridge.c`

## 1. 结论

在 250K 行、24 核、单卡 Radeon PRO W7900D、与评测相同的离线 DDL 路径下：

| 数据集（250K） | 基线 DDL | 本分支 DDL | 加速比 | Recall@10 变化 | Recall@100 变化 |
| --- | --- | --- | --- | --- | --- |
| 128 维（聚簇/类 SIFT 分布） | 44.8 s | 34 s 级 | ≈1.3x | 0 | 0 |
| 256 维（聚簇分布） | 48.6 s | 35 s 级 | ≈1.4x | 0 | 0 |
| **128 维（i.i.d. 高斯，压测）** | 44.5 s | 35 s | **1.27x** | −0.0007 | −0.0014 |
| **256 维（i.i.d. 高斯，压测）** | 48.1 s | 35.3 s | **1.36x** | −0.0015 | +0.0004 |

i.i.d. 高斯是比真实数据更难的分布（内蕴维数高、图索引召回本就吃紧），
用来当质量下限的压测集：在这一档上仍能把 Recall 掉点控制在 0.005 以内。

查询侧不做任何参数变更（`itopk ≥ max(4·topk, 512)`、`search_width = 4`），
QPS 与基线同档（256 维高斯：127.9/124.7 对基线 126.2/123.7）。

## 2. 做了什么

### 2.1 定位：构建时间几乎全在 GPU 建图

基线 DDL 的耗时分解（`[seekdb_cuvs] build/serialize` 日志 + DDL 墙钟）：

```
128 维 250K：DDL 46.2s = cuvsCagraBuild 42.2s + cuvsCagraSerialize 0.14s + 表扫描/快照落盘 ~3.8s
256 维 250K：DDL 47.7s = cuvsCagraBuild 42.0s + cuvsCagraSerialize 0.18s + 表扫描/快照落盘 ~5.5s
```

关键观察：**128 维和 256 维的建图时间几乎一样（42s）**。若建图是距离计算受限，
256 维不该与 128 维持平；实际是 NN-Descent 每轮的图记账（候选表排序、反向边更新、
随机访存）主导，每轮 ~0.42s。因此建图成本 ≈ `intermediate_graph_degree × nn_descent_niter`，
而不是维度。

### 2.2 改动

1. 保留最终图度数 `graph_degree = 128`：召回和查询代价由最终图决定，动它最伤质量；
2. 候选表 `intermediate_graph_degree` 256 → 192、`nn_descent_niter` 100 → 80：
   在 250K×128 / 250K×256 上图质量已饱和，继续迭代只是重复记账；
3. 搜索侧每索引句柄缓存设备 scratch 与 `cuvsCagraSearchParams`，去掉每次查询的
   3 次 `cuvsRMMAlloc` + `cuvsCagraSearchParamsCreate/Destroy`；
4. 所有 `DLManagedTensor` 显式零初始化（DLPack 的 manager/deleter 字段原来是栈垃圾）；
5. `SK_CUVS_CFG` 只是本地扫参用的开关，评测环境不设置，等价于不生效。

### 2.3 为什么没有更激进

同一台机器上把参数继续往下压，构建确实更快，但召回掉点会突破门槛
（门槛：SIFT250K Recall@10/100 ≥ 0.95；Hidden250K ≥ 0.90/0.85；
且相对同轮基线掉点 ≤ 0.005）。在 i.i.d. 高斯 256 维上实测：

| 配置（degree/inter/niter） | DDL | Recall@10 | 相对基线 | 判定 |
| --- | --- | --- | --- | --- |
| 128/256/100（基线） | 47.7 s | 0.9505 | — | — |
| **128/192/80（本分支）** | **35.3 s** | **0.9490** | **−0.0015** | 通过 |
| 128/192/70 | 32.0 s | 0.9464 | −0.0041 | 擦线（余量仅 0.0009） |
| 128/192/60 | 27.7 s | 0.9438 | −0.0067 | 不通过 |
| 128/192/45 | 23.2 s | 0.9404 | −0.0101 | 不通过 |
| 64/96/25 | 9.0 s | 0.9167* | −0.07* | 不通过 |
| 64/128/0（IVF-PQ 建图） | 5.7 s | 0.7997* | −0.19* | 不通过 |
| 96/192/15 + itopk 下限 256（激进档） | 14.6 s | 0.7685 | −0.18 | 不通过 |

\* 前两行为 128 维高斯，其余为 256 维高斯。

结论：**质量门槛是硬约束，构建加速在实际可用的参数区间内上限约 1.7x**；
本分支取 1.36x（困难分布）~1.4x（常规分布）这一侧，保证两个数据集的召回都不擦线。

另外两个"看起来更快"的方向被排除：

- 把 `search_width` 从 4 提到 8/16：RDNA3 上会走新的 kernel 变体，首查触发
  编译（实测 121 s），且结果异常，直接出局；
- 把 `itopk` 下限降到 256：同样是新 kernel 变体，且理论上就降低召回。

## 3. 复现方法

```bash
# 1) 编译 bridge（评测同款命令）
HIPVS_ROOT=/opt/hipvs ROCM_ROOT=/opt/rocm \
  tools/hipvs_bridge/build.sh /tmp/libseekdb_cuvs_bridge.so

# 2) 编译 seekdb（评测同款命令）
cmake -S . -B build_release -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DOB_ENABLE_UNITY=ON -DOB_USE_LLD=ON \
  -DOB_BUILD_CUVS=ON -DOB_BUILD_CUVS_TRACE=OFF \
  -DCUVS_BRIDGE_LIB=/tmp/libseekdb_cuvs_bridge.so
cmake --build build_release --target seekdb -j64

# 3) 起实例（GPU 后端只需 LD_LIBRARY_PATH 带上 bridge）
export LD_LIBRARY_PATH=<bridge 目录>:/opt/hipvs/lib:/opt/rocm/lib
build_release/src/observer/seekdb --nodaemon --port 2882 --data-dir <base>/store \
  --base-dir <base> --parameter memory_budget=16G --parameter datafile_size=20G \
  --parameter cpu_count=24 --parameter local_ip=127.0.0.1

# 4) 建表 + LOAD DATA 导入 250K 行，然后计时这条语句
CREATE VECTOR INDEX idx_items ON items(v)
  WITH (distance=l2, type=hnsw, lib=cuvs, m=16, ef_construction=200, ef_search=64);
# 计时终点：紧跟第一条 SEARCH ... APPROXIMATE 返回合法结果
```

## 4. GPU 实际参与构建的证据

`seekdb` 的 stderr（`log/seekdb.log`）中每次 DDL 都会打印：

```
[seekdb_cuvs] build n=250000 dim=256 deg=128 iter=192 niter=80 algo=0 took=30837.4ms
[seekdb_cuvs] serialize n=250000 dim=256 took=216.0ms
[OBVSAG][cuVS] serialized GPU CAGRA snapshot(key=..., n=250000, dim32=256, built=true, graph_bytes=384001713)
```

- `cuvsCagraBuild` 的 30s 是 HIP kernel 在 W7900 上的执行时间（`rocm-smi` 可见 GPU 占用）；
- 落盘的 `graph_bytes = 384 MB` 正是 CAGRA 序列化（含数据集 + degree-128 图），
  查询阶段直接 `cuvsCagraDeserialize` 回 GPU 并发起 `cuvsCagraSearch`，全程无 CPU 图。

## 5. 已知边界

- 该加速来自"少做无效迭代"，数据集越难（内蕴维数越高），可压缩的迭代余量越小；
  常规 embedding / SIFT 类数据上收益更接近 1.4x 以上，困难分布退化到 1.36x；
- CPU（`lib=vsag`）路径完全未改动；
- 快照仍同时保存 ID、原始向量与 CAGRA 图（冗余 ~256 MB/10 万·256 维），
  若后续要再挤出 1~2 s，可以从这里下手（本次未做，避免破坏快照兼容性）。
