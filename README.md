# MiniVectorDB

A single-process, single-client vector database implemented from scratch in Rust for the OceanBase 2026 competition `2026-competition-round2` branch. Storage, WAL recovery, SQL parsing/execution, transaction visibility, TopK and vector distance code use only the Rust standard library; no database or vector-index library is linked.

## Build and run

### Docker (Linux x86_64)

```bash
docker build -t minivectordb .
mkdir -p /tmp/mvdb-data
docker run --rm -i -v /tmp/mvdb-data:/data minivectordb
```

The container stays in the foreground, reads one statement per stdin line and writes results to stdout. Diagnostics go to stderr. No port is opened and no `EXPOSE`, host or port option is needed.

### Native development build

```bash
cargo build --release
./target/release/minivectordb --data-dir /tmp/mvdb-data
```

The data directory defaults to `/data` or `MINIVEC_DATA_DIR`. `--data-dir` takes precedence.

## Wire protocol

**Official transport (published 2026-11-10, official commit `f4d4e2665`).** The judge speaks JSON Lines in both directions per `STDIO_PROTOCOL.md`; `protocol_check/` is the official local self-check tool. The default (`MINIVEC_PROTOCOL=official`) implements exactly that:

Startup handshake (after recovery, before the first request, on every start):

```json
{"type":"ready"}
```

The judge sends one JSON request per stdin line:

```json
{"id":1,"op":"execute","sql":"SELECT id FROM docs ORDER BY id;","search_mode":"exact"}
{"id":2,"op":"shutdown"}
```

Responses echo the request `id` verbatim (type-preserving: a numeric id stays numeric, a string id stays a string) and wrap results with a boolean `ok`:

```json
{"id":1,"ok":true}
{"id":2,"ok":true,"affected_rows":3}
{"id":3,"ok":true,"columns":["id","distance"],"rows":[[2,1.0]]}
{"id":4,"ok":false,"error":{"code":"duplicate_key","message":"duplicate_key"}}
```

`{"op":"shutdown"}` rolls back uncommitted transactions, checkpoints, responds `{"id":n,"ok":true}`, flushes, and exits 0 with nothing further on stdout — the judge restarts the process itself between `--restart` steps. `search_mode` is honored per request (`ann` switches to the IVF path, anything else scans exactly). Malformed input lines are logged to stderr and skipped (stdout stays silent: an unsolicited response would fail the judge transport). BIGINT values are emitted as exact integer tokens; floats are finite; every response is one UTF-8 line flushed immediately.

Request parsing is hand-rolled over the standard library (`src/jsonreq.rs`, no dependencies added): full JSON string unescaping including `\uXXXX` surrogate pairs, and the raw id token is spliced into responses so its JSON type round-trips exactly. Request parsing, statement execution, response rendering and the shutdown checkpoint run under `catch_unwind`: a residual panic is downgraded to an `internal_error` response instead of killing the process (the judge transport fails the whole case list with "database exited before response" when the process dies), so later statements and cases keep running. The release profile deliberately keeps Rust's default `panic=unwind`; setting `panic=abort` would make this protection dead code.

Historical variants (our pre-publication hypotheses) remain selectable with `MINIVEC_PROTOCOL`: `jsonl` (bare `.result`-vocabulary lines), `json`, `json-noready`, `miniob`, `tsv`. `tools/check_protocols.py` smoke-tests all six, and the judge-side `--restart`/`--mode` directives of those line protocols keep working.

### Verification against the official tool

```bash
export VECTORDB_IMAGE='你的镜像名:标签'
python3 protocol_check/run.py --image "$VECTORDB_IMAGE" --handshake-only --output <new-dir>
python3 protocol_check/run.py --image "$VECTORDB_IMAGE" --output <new-dir>
python3 protocol_check/run.py --command '["./target/release/minivectordb","--data-dir","{data_dir}"]' --output <new-dir>
```

Both the `--command` (local binary) and `--image` (Docker) runs pass 20/20 public cases plus the handshake check at the current HEAD.

## Supported SQL

- `CREATE TABLE` with exactly one `BIGINT PRIMARY KEY`, any number of `VARCHAR(n)` columns, and at most one `VECTOR(n)` column. Both inline and table-level primary-key declarations are accepted.
- `CREATE VECTOR INDEX name ON table(vector_column)`; index metadata is durable and survives restart.
- Multi-row `INSERT`, including an optional complete column list; duplicate keys reject the entire statement.
- `SELECT *` or column projection, `L2_DISTANCE(vector_col, '[...]') AS alias`.
- `WHERE` comparisons `=`, `!=`, `<>`, `<`, `<=`, `>`, `>=` with `AND` / `OR`.
- `ORDER BY` one or more columns or the distance alias, `ASC` / `DESC`, and `LIMIT`.
- `UPDATE` (including vector values) and `DELETE`; primary-key updates are rejected.
- `BEGIN`, `COMMIT`, `ROLLBACK`; autocommit outside an explicit transaction.

Distance TopK filters first, then selects the K smallest distances. Ties always use ascending primary key. Exact mode returns distances accumulated in float64; ANN mode uses a float32 accumulator over stored float32 vectors. On the serial 128-dimensional flat-cache path, TopM=32 candidates are rescored from component zero with the original f64 kernel and accepted only when every excluded row is certified strictly worse than the exact Kth candidate. Candidate scanning uses a table-scaled int8 or signed-14-bit i16 mirror with exact integer SIMD, an RNE-f16 mirror, and the original full-f32 TopM pass. Int8 reads 128 bytes per row and f16 reads 256; both use quantization-aware lower bounds. Int8 uses one table-wide scale, so an all-zero table, an out-of-range query or an inadequate distance gap rejects that level. Each serial snapshot keeps one compressed mirror, calibrated from up to eight table rows: prefer int8 when at most one probe fails, then test i14, and compare f16 if i14 remains inadequate. Each actual query still proves its own result. Its failure falls back to f32 and then the original full f64 scan, always discarding intermediate heaps. Thus exact distances, row selection and primary-key ties are unchanged. Other dimensions, parallel scans, transactions and K>=32 use the original path. Small-table ANN may probe IVF lists; larger tables fall back to bounded exact scanning.

Unsupported SQL fails explicitly rather than being reinterpreted. The implementation rejects NULL, JOIN, aggregation, subqueries, ALTER/DROP, primary-key updates, unsupported scalar types, and nested transactions.

Important semantic errors include `duplicate_key`, `dimension_mismatch`, `invalid_vector`, `value_overflow`, `string_too_long`, `unsupported_sql`, `table_exists`, `table_not_found`, `column_not_found`, `index_exists`, `invalid_transaction_state`, `type_mismatch`, and `syntax_error`.

## Storage, transactions and recovery

All durable files live under the selected data directory:

- `wal.log`: length + CRC32 framed redo records. Each committed statement or explicit transaction is one frame with a monotonically increasing LSN.
- `meta`: checkpoint generation, last checkpointed LSN, table schemas and vector-index definitions.
- `d<generation>/t<table>.rows`: length + CRC32 framed live rows, written in primary-key order.

A commit appends its complete WAL frame and calls `sync_data` **before** a success response is emitted. Statement execution builds and validates all row changes before publishing them, so a failed multi-row statement leaves no partial changes. An explicit transaction stores changes in a private overlay; reads and vector queries see that overlay, while rollback simply discards it. COMMIT writes the transaction's operations as one durable WAL frame.

Checkpoints write a new generation, fsync files and directories, and atomically rename `meta.new` to `meta`. Only then is the WAL truncated. Recovery loads the committed generation, replays newer valid WAL records, and truncates a torn WAL tail. Orphan generations from an interrupted checkpoint are removed. Readiness is printed only after recovery finishes.

Scalars and primary keys remain in memory. Cached vector bytes are bounded by `MINIVEC_VECTOR_ARENA_MB` (default 128); evicted vectors are re-read with positioned reads from their WAL/checkpoint row location. Checkpoint compaction reads at most one evicted row at a time. Query-side acceleration is also bounded: flat exact-scan snapshots are skipped above `MINIVEC_SCAN_CACHE_MB` (default 128), and full in-memory IVF vector copies are skipped above `MINIVEC_INDEX_VECTOR_MB` (default 256), both falling back to the same row-at-a-time durable-read scan. This keeps exact and ANN correct on vector datasets larger than the 2 GiB container instead of OOM-ing on the first query. WAL replay and checkpoint loading stream one frame at a time, so multi-GiB recovery never reads a whole durable file into memory. Checkpoint policy defaults to WAL above `max(256 MiB, 4 * live row bytes)` and can be tuned for testing with `MINIVEC_CHECKPOINT_MIN_BYTES` and `MINIVEC_CHECKPOINT_LIVE_FACTOR`.

## Example

```text
CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(8), embedding VECTOR(2));
INSERT INTO docs VALUES (1,'keep','[3,4]'), (2,'skip','[0,0]');
SELECT id, L2_DISTANCE(embedding,'[0,0]') AS distance FROM docs WHERE category='keep' ORDER BY distance LIMIT 1;
BEGIN;
UPDATE docs SET embedding='[1,0]' WHERE id=1;
ROLLBACK;
--restart
```

## Verification

Run from the repository root after `cargo build`:

```bash
python3 -I tools/run_public_tests.py \
  --binary target/debug/minivectordb --manifest mysqltest/manifest.json
python3 -I tools/run_extended_tests.py --binary target/debug/minivectordb
cargo build --release
python3 -I tools/benchmark.py --binary target/release/minivectordb --mode exact
python3 -I tools/benchmark.py --binary target/release/minivectordb --mode ann
```

The public runner checks all assertions, restart reuse of the data directory, exact columns/rows, unordered multisets when requested, and floating-point tolerance. The extended suite independently checks illegal vectors and integer boundaries, transactional DDL and rollback, SIGKILL recovery, repeated recovery, a torn WAL tail, forced checkpoints with a zero-sized vector arena, and random filtered TopK against Python brute-force truth. `tools/check_protocols.py` smoke-tests every wire-protocol variant (DDL/DML/query/error/directives/semicolon handling and `--restart` rollback semantics) so a fallback switch is a verified change, not a hope.

Local release result on this workspace through the official protocol (10,000 rows, 128 dimensions, 1,000 queries, K=10; not an official platform score):

- exact: write 1.39 s, query 0.370 s (≈2700 qps), Recall@10 = 1.0 (full truth for all 1000 queries, filtered and unfiltered). The bare-transport numbers before JSON wrapping were 1.19 s / 0.325 s, so request parsing and response wrapping cost ~14% on the query path;
- ann: IVF-Flat recall on 128-dim uniform data measures ~0.84 (weak cluster separation), so **exact is the submission mode**; the benchmark's ann mode fails its own ≥0.95 recall assert here by design;
- query path (measured trajectory 1.19 s → 0.60 s → 0.36 s → 0.325 s): AVX2 f64-accumulated distance kernels (a first transmute-based version was 2.5× SLOWER than scalar - caught by benchmarking against the baseline, rewritten with `#[target_feature]` + `extractf128`), filter pre-binding, no pre-sort for TopK, then a bounded flat contiguous scan cache for tables whose vectors fit the cache budget (per-table snapshot with vectors concatenated into one buffer, per-row vector caches evicted after the build) with squared-distance comparison and early abandon against the current k-th best, and a four-accumulator kernel that breaks the FMA latency chain; larger tables use the bounded row-at-a-time scan instead. The current serial 128-dimensional path additionally uses a TopM=32 full-f32 candidate pass with a strict lower-bound certificate and whole-query f64 fallback, plus a TopM-only four-accumulator f32 kernel. In persistent alternating A/B, TopM improved median query time by 26.93% on WSL and 12.90% under the official Docker flags; the four-accumulator change then improved it by another 5.52% and 6.75% respectively. An optional f16 mirror, with original-f32 rescoring and a quantization-aware certificate, then improved the accepted TopM+x4 baseline by another 10.91% on WSL and 6.15% under official Docker flags. A table-scaled int8 mirror improved the f16 baseline by another 12.67% and 8.06%. W1 span-based JSON/SQL text parsing and allocation-free vector-number parsing then improved query time by another 12.17% on WSL and 6.45% in Docker. Exact responses were byte-identical in dimension/tie/filter/update differential tests, and official 20/20 plus extended 6/6 passed for all retained stages. Horizontal reduction (-0.03% WSL), whole-scan inlining (+3.17%), f16 norm/dot expansion (+0.18%), seven-bit maddubs norm/dot (-29.14%), direct int8 dispatch (+3.87%), lazy f16 construction (only 0.92% lower peak memory), actual-error int8 certificates (-0.32%), signed-norm/dot int8 arithmetic (-2.86%) and halfway int8 pruning (-5.74%) were rejected and rolled back rather than accumulated below the 5% gate;
- parallel scan: implemented and correctness-verified (recall 1.0 at 1/2/3/4/8 threads) but measured SLOWER on this machine (per-query thread spawn outweighs the win at ~0.3 ms/query), so the default is serial; `MINIVEC_SCAN_THREADS=N` enables it;
- write path: slice-by-8 CRC32 (known-vector tested), one WAL frame per statement with fdatasync before ack. W1 batch-copies unescaped JSON and ASCII SQL string spans, avoids materializing `Vec<char>` for ASCII SQL, and removes per-component lowercase allocations. Final alternating 12-pair A/B reduced end-to-end write time (including official-like SQL/JSON generation) by 14.11% on WSL and 13.11% under official Docker flags. Every pair had identical WAL SHA256; a separate mixed Unicode/transaction/update/delete/index/forced-checkpoint run produced byte-identical responses and all durable files. No WAL format, sync, acknowledgement or checkpoint semantics changed;
- database-process VmHWM during the exact benchmark: 12-17 MiB (the scan-cache build adds a ~5 MiB transient peak; steady state evicts per-row vector copies). This RSS observation is not the platform's cgroup memory metric;
- cumulative controlled A/B from the formally evaluated `7979f8d3c` binary to the accepted W1 candidate: median query time improved 47.94% on WSL and 26.21% under official Docker flags; end-to-end write time improved 11.44% and 11.14%. These are local timing ratios, not predicted platform E values;
- official-flags resource probe: final `memory.peak` was 32,411,648 bytes for the old image and 36,179,968 bytes for W1 (an 11.63% increase); final physical `rbytes` changed from 3,424,256 to 4,358,144 bytes while `wbytes` was 5,931,008 for both. The lazy-f16 experiment reached 35,848,192 bytes, only 0.92% below W1, and was rejected. cgroup memory includes page cache, so these figures must not be conflated with process RSS.

Larger-than-memory verification uses `tools/judge_sim.py` with the official reference container flags (2 CPU, 2 GiB, no swap, read-only rootfs, `cap-drop=ALL`, `no-new-privileges`, bind-mounted `/data`). Before the bounded-scan fix, 1,100,000 rows × 512 dimensions (≈2.25 GB of vector payload) died with `rc=137` on the first TopK query. After the fix the identical workload completed load, exact and ANN TopK, shutdown, restart and post-recovery queries in 265.1 s with exit 0 (restart-to-ready 4.7 s). The final TopM+x4 repository-built image also passed the full chain with exit 0 in 236.5 s (load 128.6 s, initial queries 87.5 s, shutdown 12.6 s, restart-to-ready 3.8 s, recovery queries 3.3 s). The final TopM+x4+f16 image passed the same chain with exit 0 in 280.1 s (load 151.1 s, initial queries 103.2 s, shutdown 12.8 s, restart-to-ready 4.5 s, recovery queries 7.6 s). The int8 image passed in 247.1 s and the final int8+W1 image in 186.6 s (load 78.7 s, initial queries 89.6 s, shutdown 11.1 s, restart-to-ready 3.8 s, recovery queries 2.7 s); these separate runs are regression results, not controlled performance comparisons or official platform results.

## Known gaps

The latest formal evaluation is commit `48a23c603ad3faec50e7b842011faa4427c91447`
(2026-11-12): functional admission passed, performance E=3516.901115 and valid
Agent Prompt raw score=3.5. The current candidate additionally
memoizes one filter's row positions per scan snapshot (at most 131072 rows and
32 filter instructions; otherwise the original streaming filter applies), and
keeps int8 TopM candidates in exact integer distance order before applying the
shared scale for the certificate. Original-f64 rescoring and strict fallback
remain in place. Data-version rebuilds discard the filter memo.

Two local seeds with result-cache hits explicitly cleared before each timed
block measured native query improvements of 20.64%/18.34% and official-flags
Docker improvements of 9.02%/6.55% against the latest formal version. Full truth
and baseline results matched, with both recall groups at 1.0. Tests passed:
18 release units, official 20/20, extended 6/6, deep input, panic/ANN checks and
2.25 GB streaming recovery (208.7 s, exit 0). Repeated result-cache hits showed
essentially no additional gain, and median cgroup peak stayed about 31.9 MB.
These local results do not establish E>=5000. On 2026-11-12 the user explicitly
authorized using the remaining daily formal attempt with this validated candidate;
its new platform scores are pending.

The current candidate also advises Linux to reclaim complete, clean
pages in each newly appended WAL range after `sync_data`, while the WAL fits
the vector arena budget. This reduces duplicate file-cache and arena storage;
larger WALs retain their cache for disk-backed reads. The advice is optional
and does not change file bytes, durable row locations or acknowledgement order.
Three alternating official-flags resource pairs measured median cgroup peak
32,538,624 -> 26,472,448 bytes (-18.64%). Repeated-query median time changed
0.42948 -> 0.41591 s (+3.16%, below the performance adoption gate); prior
12-pair write checks were flat (+0.14% native, -0.26% Docker), with identical
WAL hashes in every pair. This change is retained for memory savings.
Release units 18/18, public protocol cases 20/20, extended 6/6, deep input and
panic/ANN checks pass. The 2.25 GB streaming/recovery case passed in 193.8 s;
a targeted nonzero-arena test also verified durable vector projections after
scan-cache eviction, two SIGKILL recoveries and repeated forced checkpoints.
Timing figures remain local evidence and do not predict the full platform E.

The candidate also builds only one compressed mirror: when int8 construction
succeeds it omits f16, retaining the original f32 candidate pass and f64 scan
for certificate failures. Three alternating resource pairs measured median
peak 26,529,792 -> 24,006,656 bytes (-9.51%); miss-query time was +3.66% native
and -0.76% Docker, so this is a memory change. A fresh cumulative comparison
against the formal binary measured 7.65% faster Docker query misses with full
independent truth, and peak 32,145,408 -> 23,568,384 bytes (-26.68%); repeated
query hits were flat. Current checks pass 19 release units, public 20/20,
extended 6/6, deep inputs, advised-WAL readback and forced checkpoint recovery.
The 1,100,000x512 streaming chain passed with exit 0 in 205.0 s. Source-level
regression results and local ratios still do not establish E>=5000.

New vector records can use a lossless 16-entry dictionary for their most
significant byte, preserving the other 24 bits verbatim and using a four-bit
dictionary index. Vectors with too many distinct high bytes or fewer than
32 components retain the original encoding. New readers accept both tags;
WAL recovery records the actual encoded row length instead of recomputing it.
The older executable cannot read the new tag. Three local resource pairs
measured physical writes 5,931,008 -> 5,386,240 bytes (-9.19%), with no time
or memory gain claimed (peak +0.88%; Docker writes -0.29%, queries +0.44%).
Bit-exact encoding units, old/mixed-WAL and packed-checkpoint zero-arena
readback, public 20/20, extended 6/6, deep input, 20 random crash cycles and
the 2.25 GB recovery chain all pass (191.4 s, exit 0). This candidate changes
the durable encoding; new readers accept legacy and packed records.

The 2026-11-12 candidate adds a process-local result cache for repeated identical
SELECT statements with LIMIT 1..128 outside explicit transactions. It defaults to
8 MiB (`MINIVEC_RESULT_CACHE_MB=0` disables it), has a 2048-entry ceiling and
evicts the least recently used entry. Every parsed non-SELECT statement clears
it before execution, as do exact/ANN mode changes; restart starts empty. Official
JSON responses reuse a cached suffix while rebuilding each request ID. Oversized
result estimates are rejected before cloning and rendering for the cache.

Against W1, repeated-query median time improved 60.13% on WSL and 24.38% in
official-flags Docker (12 alternating pairs, independent full truth, both filtered
and unfiltered recall 1.0). First-pass timing showed no gain: Docker 0.6364 s vs
0.6628 s, measured sequentially. The cache benefit depends on query repetition
and adds memory; it does not establish a platform E prediction. Final checks:
17 release unit tests, official Docker 20/20, extended 6/6, zero panic/ANN issues,
and the 1,100,000x512 streaming recovery workload passed under 2 GiB in 236.5 s.
The large-workload run and Docker timing overlapped, so absolute times should
not be compared to earlier standalone runs. The learning material includes these
limits and exact internal evidence references in both synchronized JSON copies.

1. The official protocol is now published (`STDIO_PROTOCOL.md`, commit `f4d4e2665`) and fully implemented; the frozen platform-side error-code mapping (`config.json "error_codes"`) is still empty locally, so error codes beyond the public cases' vocabulary are our semantic names (the judge currently only requires a non-empty `error.code` string).
2. IVF-Flat ANN is implemented (deterministic k-means, inverted lists, incremental maintenance, restart rebuild) but recall on 128-dim uniform benchmark data is ~0.84, below the 0.95 bar, so submission mode is exact scanning. IVF remains enabled for `search_mode:"ann"` on smaller/clusterable datasets.
3. Single-client and single-transaction limits are intentional. No network protocol, NULL, joins, aggregates, distributed execution or concurrent transaction support is provided.
4. The certified TopM acceleration is restricted to serial, 128-dimensional flat-cache TopK with K<32 and a default IEEE arithmetic environment; f16 additionally requires F16C and int8 requires AVX2 plus enough cache budget for the mirrors. Int8's table-wide scale can be ineffective on skewed data, and duplicate-heavy or otherwise certificate-failing inputs pay for one or more candidate passes before the full f64 fallback. WSL and local Docker gains are controlled local measurements only; the latest formal result is E=3841.678288 and Prompt=4.5 from commit `ac85bab5` (MySQL admission passed). The historical `015651fb8` result was E=3769.381119 with no valid numeric Prompt result; its specific review failure was not exposed in the result-detail API. The earlier commit `48a23c603` had valid Prompt raw3.5.

Submission preparation on 2026-11-12: final source matches the retained validated
candidate. Fresh checks pass 20 release units, 20 public official cases, six
extended groups, deep inputs, durable readback/forced checkpoints and 20 random
crash cycles. Commit uses actual serialized row lengths without a second
dictionary pass. Learning materials are identical UTF-8 JSON copies below
128 KiB with exact internal evidence references; no new score is predicted.


The local 2026-11-13 learning-material revision preserves every student prompt,
Agent conversation and experiment output from the submitted material, removes
repeated summaries and rule text, and includes twelve measured candidate experiment chains,
phase diagnostics and an attributed accepted technical plan. Both
conversation.json copies are synchronized. This
material revision is not formally submitted and has no predicted Prompt score.
Eleven experiment chains were rejected, including lossless exponent compression
whose disk-read cost outweighed its modest resource benefit. The user accepted the working tree combination as the next local optimization
baseline: resident original-vector snapshots combined with serialized-only
official result caching. In three host-sampled container comparisons its peak
median fell from 18,124,800 to 16,035,840 bytes (-11.5254%); Docker query time
regressed 1.0136% and 1.3208% on two independent seeds, so no query speedup is
claimed. The candidate passes 21 release units, 20 public cases, six extended
groups and deep-input/restart checks. It is not a formal submission, and these
local measurements do not predict platform E.


The local sprint adds semantic vector cache keys with full collision checks,
a generic static runtime, and snapshot-local adaptive candidate mirrors.
After four int8 certificate failures the engine replaces that mirror with
f16 when supported and within budget; four f16 failures release that mirror.
Original f32 values and the exact f32-certificate/f64 fallback stay available.
Independent distribution tests expose a workload dependency: Gaussian and
unit-sphere data benefit, while uniform data is approximately flat.
Direct Docker query gains against accepted residentpayload (12 pairs, seed20261225): uniform +1.9636%, gaussian +20.9314%, sphere +19.6526%.
Final checks pass26 units,20 public cases,6 extended groups, deep-input restart,
durable readback and forced checkpoints. The preceding adaptive+static build
also passes a2.25GB logical-vector crash/restart chain under2GiB in303.90s;
the final change affects128-dimensional candidate mirrors only. All results
remain local, with no new formal submission or E/Prompt score prediction.

A one-time formal submission is authorized for2026-11-13 20:00Asia/Shanghai,
regardless of the local score estimate. This version is prepared for that run;
its official admission, E and Prompt results are not yet known.


Post-submission update (2026-11-13): d8d9b057 passed MySQL; official E2686.990441,
Prompt4.5. Those results do not include the following local policy fix. Only
completed compressed candidate passes now train mirror-failure counters;
queries outside the representation range use exact fallback without retiring
a usable table mirror. Two changing-range workload seeds improve Docker
query time14.49%/12.99%, with ordinary uniform/Gaussian workloads flat.
27units,20public cases,6extended groups and deep-input/restart checks pass.
The platform performance regression remains unexplained; no E projection.


Evening candidate (2026-11-13): use the original query/cache paths with
resident original vectors and serialized-only official responses. Compressed
SQL cache keys, adaptive mirror retirement and static runtime are withdrawn
from this candidate after the formal performance regression. Relative to
015651fb8, Docker queries change−0.4406%, writes+1.8575%, and container
peak falls14.3621% in matched local tests.21units,20public cases,6extended
groups and deep-input/restart checks pass. Earlier sprint descriptions above
are historical experiments. Prompt source records from the valid4.5 version
remain intact. A new official score is pending.


Latest formal result (ac85bab5, submitted2026-11-13 23:57 Asia/Shanghai):
MySQL passed,E3841.678288,Prompt4.5. A subsequent local candidate selects
one certified mirror using eight deterministic table-row probes at snapshot
construction. It keeps int8 if at most one probe fails; otherwise it tests
f16 and retains the mirror with more successful certificates, dropping both
if neither succeeds. Later query order cannot change this choice. Each
query still needs its own strict certificate or the original f32/f64 fallback.
Two local seeds pass independent truth and exact output comparisons across
six distributions;22units,20public,6extended and deep/restart checks pass.
See minivec/nextday-20261114 in the parent workspace for measurements.
This candidate is not formally submitted;E5000/Prompt15 remain unmet.


Local i14 follow-up (not formally submitted): a finer [-8191,8191] grid stored
in i16 avoids f16 decode on snapshots where int8 is inadequate. AVX2 lane
sums are reinterpreted as u32 and reduced to u64; full-range tests cover
the signed boundary. Original-f32 rescoring and strict certificates remain.
Against calibrated-final, two Gaussian Docker seeds improve5.82%/8.31%;
held-out sphere7.02%,clustered12.74%,uniform-0.61%. Whole-cgroup peak
changes uniform-1.53%/Gaussian+1.63%;write-1.10%,all WAL hashes equal.
26units,20public Docker cases,6extended and deep/restart checks pass.
See parent workspace minivec/widegrid-20261114/review.json for raw runs.
Formal E remains3841.678288,Prompt4.5;local gains do not forecast E5000.


Final local proof refinement:the i14 error allowance explicitly covers f64
division at grid magnitude8191,independently of the existing f32 slack.
All correctness/performance/resource gates were rerun on the final binary;
Gaussian Docker gains vs calibrated-final:+6.54%/+7.81% on two seeds.
Held-out U/G/S/T/O/C query gains:+0.52/+7.81/+7.29/+0.41/+0.46/+12.77%;whole-cgroup U/G peaks:-1.59/+0.07%.
26units,20public,6extended,deep/restart pass;all write WAL hashes match.
Shared immutable row prototype was rejected:queries-3.59%/-3.71%,with
only1.97%/6.27%peak reduction. It was never copied into this repo.
Current artifacts and full measurements:parent workspace
minivec/widegrid-proof-20261114/README.md and review.json. No formal submission.


Latest local follow-up: snapshot f32-candidate dispatch and exact zero-distance
early stop with an out-of-line full-f64 fallback. Every accepted candidate
still needs its per-query certificate. Unique ascending PKs justify the
zero-threshold stop. Docker ties gains on three seeds:+27.12/+25.84/+25.00%;
held-out U/G/S/T/O/C:-0.72/+1.24/+0.19/+25.84/+0.40/+2.78% vs widegrid-proof. Other ordinary
distribution results are small;no official E extrapolation.28units,20public,
6extended,parallel/deep/restart pass;all12write WAL hashes equal.
Shared column-value arrays were rejected after outlier/sphere regression.
Full evidence:parent workspace minivec/dispatch-split-20261114/review.json.
Formal commit and scores unchanged;no new formal submission.

Latest structural follow-up (workspace artifacts dated2026-11-14):the retained
implementation is still dispatch-split. An isolated lossless cache-key prototype
failed its initial ties-time gate. A combined compact-key/borrowed single-buffer
JSON prototype passed32units and independent query truths, with roughly5–8%
lower cgroup peak across two resource seeds. It failed the predeclared write-time
gate (-5.18% ratio of independent medians), so neither prototype was copied here.
Paired write estimates are less negative but inconclusive;all12WAL hashes match.
Full downstream regression was not run after that failure. Complete evidence is
in the parent workspace minivec/compact-response-20261114/review.json.
The two Prompt material files now record the student's actual structural question
and attributed Agent experiments:130730bytes,110valid references;the original
valid4.5core remains intact. No new student postinterpretation or acceptance is
claimed. Formal ac85bab5 remains MySQL passed,E3841.678288,Prompt4.5;no new
submission or E/Prompt forecast was made.


Submission preparation (2026-11-14, before 16:00 Asia/Shanghai): the user
authorized one attempt with the retained dispatch-split artifact, followed by
continued optimization. Its executable SHA256 is
52f13723cc3fe65ff551b7acc5f0b4695a0dfcc8f0bb29e223005fc40e52a459.
The latest direct comparison uses the actual submitted ac85bab executable,
156 sequential fresh containers, two seeds, and six distributions. Query time
gains are uniform +0.116%, Gaussian +32.072%, sphere +31.882%, ties +26.904%,
outlier +10.708%, clustered +22.330%. Physical I/O is unchanged; Gaussian,
sphere, and clustered query peaks increase about 7–11%. Sphere write wall
time regresses 2.354%; same-binary controls show write-time variability, so
the regression cause and nonregression remain unresolved. This comparison
includes initialization, load, preparation, disjoint warmup and primary-query
resource costs, with a shared local device and warm runtime pages. It is not
an official E forecast. Evidence is in parent minivec/direct-costs-20261114.
The frozen source previously passed 28 release units, 20 public Docker cases,
extended, parallel, and deep/restart checks. No fresh multi-GiB recovery is
claimed for this artifact. Both learning-material copies contain 130957 bytes
and 111 valid references, preserving the original valid-4.5 core and student
words. Submission authorization adds no student technical explanation.
