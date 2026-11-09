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

## Current local wire protocol

**Protocol status (2026-11-09):** the public package contains cases and assertions but not the official stdin/stdout driver. Its README explicitly says process communication is defined by the separate driver. The transport below is therefore an isolated interim adapter for local validation; it must be matched to the official driver when that specification is published. SQL execution and result comparison are independent of this adapter.

On startup:

```json
{"ready":true}
```

Successful DDL / transaction command:

```json
{"ok":true}
```

Successful INSERT / UPDATE / DELETE:

```json
{"ok":true,"affected_rows":3}
```

Query:

```json
{"ok":true,"columns":["id","distance"],"rows":[[1,5.0]]}
```

Error:

```json
{"ok":false,"error":"duplicate_key","code":"duplicate_key"}
```

Local driver directives are `--restart` (implemented by the test process), `--mode exact`, `--mode ann`, and `--shutdown`. Every response is one UTF-8 JSON line and stdout is flushed immediately. BIGINT values are emitted as integer tokens, not rounded through floating point.

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

The public runner checks all assertions, restart reuse of the data directory, exact columns/rows, unordered multisets when requested, and floating-point tolerance. The extended suite independently checks illegal vectors and integer boundaries, transactional DDL and rollback, SIGKILL recovery, repeated recovery, a torn WAL tail, forced checkpoints with a zero-sized vector arena, and random filtered TopK against Python brute-force truth.

Local release result on this workspace (10,000 rows, 128 dimensions, 1,000 queries, K=10; not an official platform score):

- exact: write 1.12 s, query 1.07 s, sampled Recall@10 = 1.0;
- ann: write 1.12 s, query 0.93 s, sampled Recall@10 = 1.0;
- database-process VmHWM during the exact benchmark: 11 MiB (the Python data generator/driver is excluded).

## Known gaps

1. The official evaluation wire format, resource limits and frozen error mapping have not been published in the branch's public package; the interim transport must be adapted before formal submission.
2. ANN currently uses exact candidate scanning rather than IVF/HNSW pruning. This favors recall and update correctness; further query-time indexing is an optimization, not a claim of platform performance.
3. Single-client and single-transaction limits are intentional. No network protocol, NULL, joins, aggregates, distributed execution or concurrent transaction support is provided.
