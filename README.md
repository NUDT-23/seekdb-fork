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

Request parsing is hand-rolled over the standard library (`src/jsonreq.rs`, no dependencies added): full JSON string unescaping including `\uXXXX` surrogate pairs, and the raw id token is spliced into responses so its JSON type round-trips exactly. Statement execution and the shutdown checkpoint run under `catch_unwind`: a residual panic is downgraded to an `internal_error` response instead of killing the process (the judge transport fails the whole case list with "database exited before response" when the process dies), so later statements and cases keep running.

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

Distance TopK filters first, then selects the K smallest distances. Ties always use ascending primary key. Exact mode accumulates distance in float64; ANN mode uses a float32 accumulator over stored float32 vectors. Both currently scan visible candidates (ANN recall therefore matches the exact candidate ordering up to float accumulation); the durable vector-index definition is not yet used for pruning.

Unsupported SQL fails explicitly rather than being reinterpreted. The implementation rejects NULL, JOIN, aggregation, subqueries, ALTER/DROP, primary-key updates, unsupported scalar types, and nested transactions.

Important semantic errors include `duplicate_key`, `dimension_mismatch`, `invalid_vector`, `value_overflow`, `string_too_long`, `unsupported_sql`, `table_exists`, `table_not_found`, `column_not_found`, `index_exists`, `invalid_transaction_state`, `type_mismatch`, and `syntax_error`.

## Storage, transactions and recovery

All durable files live under the selected data directory:

- `wal.log`: length + CRC32 framed redo records. Each committed statement or explicit transaction is one frame with a monotonically increasing LSN.
- `meta`: checkpoint generation, last checkpointed LSN, table schemas and vector-index definitions.
- `d<generation>/t<table>.rows`: length + CRC32 framed live rows, written in primary-key order.

A commit appends its complete WAL frame and calls `sync_data` **before** a success response is emitted. Statement execution builds and validates all row changes before publishing them, so a failed multi-row statement leaves no partial changes. An explicit transaction stores changes in a private overlay; reads and vector queries see that overlay, while rollback simply discards it. COMMIT writes the transaction's operations as one durable WAL frame.

Checkpoints write a new generation, fsync files and directories, and atomically rename `meta.new` to `meta`. Only then is the WAL truncated. Recovery loads the committed generation, replays newer valid WAL records, and truncates a torn WAL tail. Orphan generations from an interrupted checkpoint are removed. Readiness is printed only after recovery finishes.

Scalars and primary keys remain in memory. Cached vector bytes are bounded by `MINIVEC_VECTOR_ARENA_MB` (default 512); evicted vectors are re-read with positioned reads from their WAL/checkpoint row location. Checkpoint compaction reads at most one evicted row at a time. Checkpoint policy defaults to WAL above `max(256 MiB, 4 * live row bytes)` and can be tuned for testing with `MINIVEC_CHECKPOINT_MIN_BYTES` and `MINIVEC_CHECKPOINT_LIVE_FACTOR`.

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
- query path (measured trajectory 1.19 s → 0.60 s → 0.36 s → 0.325 s): AVX2 f64-accumulated distance kernels (a first transmute-based version was 2.5× SLOWER than scalar - caught by benchmarking against the baseline, rewritten with `#[target_feature]` + `extractf128`), filter pre-binding, no pre-sort for TopK, then a flat contiguous scan cache (per-table snapshot with vectors concatenated into one buffer, per-row vector caches evicted after the build) with squared-distance comparison and early abandon against the current k-th best, and a four-accumulator kernel that breaks the FMA latency chain;
- parallel scan: implemented and correctness-verified (recall 1.0 at 1/2/3/4/8 threads) but measured SLOWER on this machine (per-query thread spawn outweighs the win at ~0.3 ms/query), so the default is serial; `MINIVEC_SCAN_THREADS=N` enables it;
- write path: slice-by-8 CRC32 (known-vector tested), one WAL frame per statement with fdatasync before ack;
- database-process VmHWM during the exact benchmark: 12-17 MiB (the scan-cache build adds a ~5 MiB transient peak; steady state evicts per-row vector copies).

## Known gaps

1. The official protocol is now published (`STDIO_PROTOCOL.md`, commit `f4d4e2665`) and fully implemented; the frozen platform-side error-code mapping (`config.json "error_codes"`) is still empty locally, so error codes beyond the public cases' vocabulary are our semantic names (the judge currently only requires a non-empty `error.code` string).
2. IVF-Flat ANN is implemented (deterministic k-means, inverted lists, incremental maintenance, restart rebuild) but recall on 128-dim uniform benchmark data is ~0.84, below the 0.95 bar, so submission mode is exact scanning. IVF remains enabled for `search_mode:"ann"` on smaller/clusterable datasets.
3. Single-client and single-transaction limits are intentional. No network protocol, NULL, joins, aggregates, distributed execution or concurrent transaction support is provided.
