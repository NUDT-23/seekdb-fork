//! Durable storage: write-ahead log, checkpoint compaction and crash recovery.
//!
//! One committed statement is one WAL frame. The frame is fsync'd before the
//! caller may acknowledge success. Checkpoints atomically replace the `meta`
//! file after writing a complete new `d<gen>` directory; WAL replay starts
//! after the LSN recorded in `meta`. A torn WAL tail is discarded.

use std::collections::{BTreeMap, HashMap};
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;

// Positioned reads: unix and windows name them differently.
trait ReadAt {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize>;
}

#[cfg(unix)]
impl ReadAt for File {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        std::os::unix::fs::FileExt::read_at(self, buf, offset)
    }
}

#[cfg(windows)]
impl ReadAt for File {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        std::os::windows::fs::FileExt::seek_read(self, buf, offset)
    }
}

use crate::error::{DbError, DbResult};
use crate::value::{ColKind, Column, Schema};

pub const CHECKPOINT_MIN_BYTES: u64 = 256 * 1024 * 1024;
pub const CHECKPOINT_LIVE_FACTOR: u64 = 4;
pub const SHUTDOWN_CHECKPOINT_MIN: u64 = 16 * 1024 * 1024;

/// CRC-32 (IEEE, reflected) with a 8x256 slice-by-8 table: ~10x the bitwise
/// loop at the same exact output, used for every WAL frame.
const CRC_POLY: u32 = 0xEDB8_8320;

#[cfg(test)]
mod crc_tests {
    #[test]
    fn known_vectors() {
        assert_eq!(super::crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(super::crc32(b""), 0x0000_0000);
        let long: Vec<u8> = (0..=255u8).cycle().take(10_000).collect();
        // Cross-check slice-by-8 against a straightforward bitwise reference.
        let mut crc = 0xFFFF_FFFFu32;
        for &b in &long {
            crc ^= b as u32;
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (super::CRC_POLY & mask);
            }
        }
        assert_eq!(super::crc32(&long), !crc);
    }
}

fn crc_table(slot: usize) -> [u32; 256] {
    let mut base = [0u32; 256];
    for (i, entry) in base.iter_mut().enumerate() {
        let mut crc = i as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (CRC_POLY & mask);
        }
        *entry = crc;
    }
    // Standard slice-by-8 recurrence: fold always against the base table.
    let mut table = base;
    for _ in 0..slot {
        let mut next = [0u32; 256];
        for i in 0..256 {
            next[i] = (table[i] >> 8) ^ base[(table[i] & 0xFF) as usize];
        }
        table = next;
    }
    table
}

fn crc32(data: &[u8]) -> u32 {
    // Lazy static tables without external crates: built once per process
    // (the engine is single-threaded).
    struct Tables([[u32; 256]; 8]);
    static TABLES: std::sync::OnceLock<Tables> = std::sync::OnceLock::new();
    let tables = TABLES.get_or_init(|| Tables(std::array::from_fn(crc_table)));
    let t = &tables.0;
    let mut crc = 0xFFFF_FFFFu32;
    let mut chunks = data.chunks_exact(8);
    for chunk in &mut chunks {
        let v = crc
            ^ u32::from(chunk[0])
            ^ (u32::from(chunk[1]) << 8)
            ^ (u32::from(chunk[2]) << 16)
            ^ (u32::from(chunk[3]) << 24);
        crc = t[7][(v & 0xFF) as usize]
            ^ t[6][(((v >> 8) & 0xFF)) as usize]
            ^ t[5][(((v >> 16) & 0xFF)) as usize]
            ^ t[4][(((v >> 24) & 0xFF)) as usize]
            ^ t[3][chunk[4] as usize]
            ^ t[2][chunk[5] as usize]
            ^ t[1][chunk[6] as usize]
            ^ t[0][chunk[7] as usize];
    }
    for &b in chunks.remainder() {
        let idx = ((crc ^ b as u32) & 0xFF) as usize;
        crc = t[0][idx] ^ (crc >> 8);
    }
    !crc
}

#[derive(Debug, Clone)]
pub enum VVal {
    Int(i64),
    Str(String),
    /// `None` means this vector was evicted from the memory arena and can be
    /// reloaded from the row's durable location.
    Vec(Option<Rc<Vec<f32>>>),
}

#[derive(Debug, Clone)]
pub struct Row {
    pub pk: i64,
    /// Values in schema column order, including the primary-key column.
    pub vals: Vec<VVal>,
}

impl Row {
    pub fn cached_vec(&self, idx: usize) -> Option<Rc<Vec<f32>>> {
        match &self.vals[idx] {
            VVal::Vec(Some(v)) => Some(Rc::clone(v)),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowLoc {
    /// 0 = wal.log; otherwise checkpoint generation.
    pub gen: u64,
    /// Absolute file offset of the row payload (after its frame header).
    pub off: u64,
    pub len: u32,
}

#[derive(Debug, Clone)]
pub struct RowEntry {
    pub row: Row,
    pub loc: RowLoc,
}

#[derive(Debug, Clone)]
pub struct IndexDef {
    pub name: String,
    pub table_id: u32,
    pub column: String,
}

#[derive(Debug, Clone)]
pub enum WalOp {
    CreateTable(Schema),
    UpsertRow { table_id: u32, row: Row },
    DeleteRow { table_id: u32, pk: i64 },
    CreateVectorIndex(IndexDef),
}

const TAG_CREATE_TABLE: u8 = 1;
const TAG_UPSERT: u8 = 2;
const TAG_DELETE: u8 = 3;
const TAG_CREATE_VINDEX: u8 = 4;
const VT_INT: u8 = 1;
const VT_STR: u8 = 2;
const VT_VEC: u8 = 3;

struct Buf {
    b: Vec<u8>,
}

impl Buf {
    fn new() -> Self {
        Self { b: Vec::new() }
    }
    fn u8(&mut self, v: u8) {
        self.b.push(v);
    }
    fn u32(&mut self, v: u32) {
        self.b.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.b.extend_from_slice(&v.to_le_bytes());
    }
    fn i64(&mut self, v: i64) {
        self.b.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.b.extend_from_slice(&v.to_le_bytes());
    }
    fn str(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.b.extend_from_slice(s.as_bytes());
    }
}

struct Cur<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cur<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, p: 0 }
    }
    fn take(&mut self, n: usize) -> DbResult<&'a [u8]> {
        if self.p.checked_add(n).map_or(true, |end| end > self.b.len()) {
            return Err(DbError::Internal);
        }
        let out = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(out)
    }
    fn u8(&mut self) -> DbResult<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> DbResult<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> DbResult<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn i64(&mut self) -> DbResult<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> DbResult<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn str(&mut self) -> DbResult<String> {
        let n = self.u32()? as usize;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| DbError::Internal)
    }
}

fn put_row(buf: &mut Buf, row: &Row) -> DbResult<()> {
    buf.i64(row.pk);
    buf.u32(row.vals.len() as u32);
    for value in &row.vals {
        match value {
            VVal::Int(v) => {
                buf.u8(VT_INT);
                buf.i64(*v);
            }
            VVal::Str(v) => {
                buf.u8(VT_STR);
                buf.str(v);
            }
            VVal::Vec(Some(v)) => {
                buf.u8(VT_VEC);
                buf.u32(v.len() as u32);
                for f in v.iter() {
                    buf.f32(*f);
                }
            }
            // A durable write must never silently replace an evicted vector
            // with an empty one. Callers materialize rows before committing.
            VVal::Vec(None) => return Err(DbError::Internal),
        }
    }
    Ok(())
}

fn get_row(cur: &mut Cur) -> DbResult<Row> {
    let pk = cur.i64()?;
    let n = cur.u32()? as usize;
    let mut vals = Vec::with_capacity(n);
    for _ in 0..n {
        let value = match cur.u8()? {
            VT_INT => VVal::Int(cur.i64()?),
            VT_STR => VVal::Str(cur.str()?),
            VT_VEC => {
                let dim = cur.u32()? as usize;
                let mut values = Vec::with_capacity(dim);
                for _ in 0..dim {
                    values.push(cur.f32()?);
                }
                VVal::Vec(Some(Rc::new(values)))
            }
            _ => return Err(DbError::Internal),
        };
        vals.push(value);
    }
    Ok(Row { pk, vals })
}

fn row_len(row: &Row) -> DbResult<u32> {
    let mut n = 8usize + 4;
    for value in &row.vals {
        n += 1;
        match value {
            VVal::Int(_) => n += 8,
            VVal::Str(s) => n += 4 + s.len(),
            VVal::Vec(Some(v)) => n += 4 + v.len() * 4,
            VVal::Vec(None) => return Err(DbError::Internal),
        }
    }
    u32::try_from(n).map_err(|_| DbError::Internal)
}

fn put_schema(buf: &mut Buf, schema: &Schema) {
    buf.str(&schema.name);
    buf.u32(schema.id);
    buf.u32(schema.columns.len() as u32);
    for col in &schema.columns {
        buf.str(&col.name);
        match col.kind {
            ColKind::BigInt => {
                buf.u8(1);
                buf.u32(0);
            }
            ColKind::Varchar(n) => {
                buf.u8(2);
                buf.u32(n);
            }
            ColKind::Vector(n) => {
                buf.u8(3);
                buf.u32(n);
            }
        }
        buf.u8(col.primary_key as u8);
    }
}

fn get_schema(cur: &mut Cur) -> DbResult<Schema> {
    let name = cur.str()?;
    let id = cur.u32()?;
    let ncols = cur.u32()? as usize;
    let mut columns = Vec::with_capacity(ncols);
    for _ in 0..ncols {
        let name = cur.str()?;
        let kind = match cur.u8()? {
            1 => ColKind::BigInt,
            2 => ColKind::Varchar(cur.u32()?),
            3 => ColKind::Vector(cur.u32()?),
            _ => return Err(DbError::Internal),
        };
        // BIGINT still has its placeholder argument in the encoding.
        if kind == ColKind::BigInt {
            let _ = cur.u32()?;
        }
        let primary_key = cur.u8()? != 0;
        columns.push(Column { name, kind, primary_key });
    }
    Ok(Schema { name, id, columns })
}

fn put_index(buf: &mut Buf, index: &IndexDef) {
    buf.str(&index.name);
    buf.u32(index.table_id);
    buf.str(&index.column);
}

fn get_index(cur: &mut Cur) -> DbResult<IndexDef> {
    Ok(IndexDef {
        name: cur.str()?,
        table_id: cur.u32()?,
        column: cur.str()?,
    })
}

fn put_op(op: &WalOp) -> DbResult<Buf> {
    let mut buf = Buf::new();
    match op {
        WalOp::CreateTable(schema) => {
            buf.u8(TAG_CREATE_TABLE);
            put_schema(&mut buf, schema);
        }
        WalOp::UpsertRow { table_id, row } => {
            buf.u8(TAG_UPSERT);
            buf.u32(*table_id);
            put_row(&mut buf, row)?;
        }
        WalOp::DeleteRow { table_id, pk } => {
            buf.u8(TAG_DELETE);
            buf.u32(*table_id);
            buf.i64(*pk);
        }
        WalOp::CreateVectorIndex(index) => {
            buf.u8(TAG_CREATE_VINDEX);
            put_index(&mut buf, index);
        }
    }
    Ok(buf)
}

fn get_op(cur: &mut Cur) -> DbResult<WalOp> {
    match cur.u8()? {
        TAG_CREATE_TABLE => Ok(WalOp::CreateTable(get_schema(cur)?)),
        TAG_UPSERT => {
            let table_id = cur.u32()?;
            let row = get_row(cur)?;
            Ok(WalOp::UpsertRow { table_id, row })
        }
        TAG_DELETE => {
            let table_id = cur.u32()?;
            let pk = cur.i64()?;
            Ok(WalOp::DeleteRow { table_id, pk })
        }
        TAG_CREATE_VINDEX => Ok(WalOp::CreateVectorIndex(get_index(cur)?)),
        _ => Err(DbError::Internal),
    }
}

pub struct TableData {
    pub schema: Schema,
    pub rows: HashMap<i64, RowEntry>,
}

pub struct Store {
    dir: PathBuf,
    gen: u64,
    lsn: u64,
    pub tables: BTreeMap<u32, TableData>,
    pub indexes: Vec<IndexDef>,
    next_table_id: u32,
    wal_write: File,
    wal_read: File,
    wal_len: u64,
    wal_since_checkpoint: u64,
    live_bytes: u64,
    gen_readers: HashMap<(u64, u32), File>,
    arena_budget: usize,
    arena_bytes: usize,
    checkpoint_min_bytes: u64,
    checkpoint_live_factor: u64,
}

impl Store {
    pub fn open(dir: &Path, arena_budget: usize) -> io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let meta_path = dir.join("meta");
        let (gen, lsn, schemas, indexes) = if meta_path.exists() {
            read_meta(&meta_path)?
        } else {
            (0, 0, Vec::new(), Vec::new())
        };
        let mut tables = BTreeMap::new();
        let mut next_table_id = 1u32;
        for schema in schemas {
            next_table_id = next_table_id.max(schema.id + 1);
            tables.insert(schema.id, TableData { schema, rows: HashMap::new() });
        }
        let mut gen_readers = HashMap::new();
        if gen > 0 {
            let checkpoint_bytes: u64 = tables
                .keys()
                .filter_map(|table_id| {
                    std::fs::metadata(dir.join(format!("d{}/t{}.rows", gen, table_id)))
                        .ok()
                        .map(|meta| meta.len())
                })
                .sum();
            let cache_vectors = checkpoint_bytes <= arena_budget as u64;
            for (table_id, table) in tables.iter_mut() {
                let path = dir.join(format!("d{}/t{}.rows", gen, table_id));
                if path.exists() {
                    let mut file = File::open(&path)?;
                    load_checkpoint_rows(&mut file, gen, table, cache_vectors)?;
                    gen_readers.insert((gen, *table_id), file);
                }
            }
        }
        let wal_path = dir.join("wal.log");
        let wal_write = OpenOptions::new().create(true).append(true).read(true).open(&wal_path)?;
        let wal_read = File::open(&wal_path)?;
        let wal_len = wal_read.metadata()?.len();
        let mut store = Self {
            dir: dir.to_path_buf(),
            gen,
            lsn,
            tables,
            indexes,
            next_table_id,
            wal_write,
            wal_read,
            wal_len,
            wal_since_checkpoint: wal_len,
            live_bytes: 0,
            gen_readers,
            arena_budget,
            arena_bytes: 0,
            checkpoint_min_bytes: std::env::var("MINIVEC_CHECKPOINT_MIN_BYTES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(CHECKPOINT_MIN_BYTES),
            checkpoint_live_factor: std::env::var("MINIVEC_CHECKPOINT_LIVE_FACTOR")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(CHECKPOINT_LIVE_FACTOR),
        };
        store.remove_orphan_generations();
        let (valid_end, max_lsn) = store.replay_wal()?;
        if valid_end < store.wal_len {
            store.wal_write.set_len(valid_end)?;
            store.wal_write.sync_all()?;
            store.wal_len = valid_end;
        }
        store.wal_since_checkpoint = valid_end;
        store.lsn = max_lsn;
        store.recompute_usage();
        Ok(store)
    }

    fn recompute_usage(&mut self) {
        self.live_bytes = 0;
        self.arena_bytes = 0;
        for table in self.tables.values() {
            for entry in table.rows.values() {
                self.live_bytes += 16 + entry.loc.len as u64;
                if let Some(vector_col) = table.schema.vector_col() {
                    if let VVal::Vec(Some(vector)) = &entry.row.vals[vector_col] {
                        self.arena_bytes += vector.len() * 4;
                    }
                }
            }
        }
        self.enforce_arena_budget();
    }

    fn remove_orphan_generations(&self) {
        if let Ok(entries) = std::fs::read_dir(&self.dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                if let Some(number) = name.strip_prefix('d').and_then(|s| s.parse::<u64>().ok()) {
                    if number != self.gen {
                        let _ = std::fs::remove_dir_all(entry.path());
                    }
                }
            }
        }
    }

    pub fn next_table_id(&mut self) -> u32 {
        let id = self.next_table_id;
        self.next_table_id += 1;
        id
    }

    /// Serialize, append and fsync one commit record, then publish it to the
    /// in-memory tables. Returned locations correspond to upserts in op order.
    pub fn commit(&mut self, ops: &[WalOp]) -> DbResult<Vec<RowLoc>> {
        if ops.is_empty() {
            return Ok(Vec::new());
        }
        let mut payload = Buf::new();
        payload.u64(self.lsn + 1);
        let mut locs = Vec::new();
        for op in ops {
            let encoded = put_op(op)?;
            if let WalOp::UpsertRow { row, .. } = op {
                // frame header (8) + LSN (8) + previous payload + op tag (1)
                // + table id (4) = absolute row-payload offset.
                let off = self.wal_len + 8 + payload.b.len() as u64 + 5;
                locs.push(RowLoc { gen: 0, off, len: row_len(row)? });
            }
            payload.b.extend_from_slice(&encoded.b);
        }
        let mut frame = Buf::new();
        frame.u32(payload.b.len() as u32);
        frame.u32(crc32(&payload.b));
        frame.b.extend_from_slice(&payload.b);
        self.wal_write.write_all(&frame.b).map_err(|_| DbError::Internal)?;
        self.wal_write.flush().map_err(|_| DbError::Internal)?;
        self.wal_write.sync_data().map_err(|_| DbError::Internal)?;
        self.wal_len += frame.b.len() as u64;
        self.wal_since_checkpoint += frame.b.len() as u64;
        self.lsn += 1;
        self.apply_ops(ops, &locs);
        Ok(locs)
    }

    fn apply_ops(&mut self, ops: &[WalOp], locs: &[RowLoc]) {
        let mut loc_iter = locs.iter().copied();
        for op in ops {
            match op {
                WalOp::CreateTable(schema) => {
                    self.next_table_id = self.next_table_id.max(schema.id + 1);
                    self.tables.insert(
                        schema.id,
                        TableData { schema: schema.clone(), rows: HashMap::new() },
                    );
                }
                WalOp::UpsertRow { table_id, row } => {
                    let loc = loc_iter.next().expect("upsert loc");
                    if let Some(table) = self.tables.get_mut(table_id) {
                        if let Some(old) = table.rows.remove(&row.pk) {
                            self.live_bytes = self.live_bytes.saturating_sub(16 + old.loc.len as u64);
                            self.arena_bytes = self
                                .arena_bytes
                                .saturating_sub(row_vector_bytes(&old.row, &table.schema));
                        }
                        self.live_bytes += 16 + loc.len as u64;
                        self.arena_bytes += row_vector_bytes(row, &table.schema);
                        table.rows.insert(row.pk, RowEntry { row: row.clone(), loc });
                    }
                }
                WalOp::DeleteRow { table_id, pk } => {
                    if let Some(table) = self.tables.get_mut(table_id) {
                        if let Some(old) = table.rows.remove(pk) {
                            self.live_bytes = self.live_bytes.saturating_sub(16 + old.loc.len as u64);
                            self.arena_bytes = self
                                .arena_bytes
                                .saturating_sub(row_vector_bytes(&old.row, &table.schema));
                        }
                    }
                }
                WalOp::CreateVectorIndex(index) => self.indexes.push(index.clone()),
            }
        }
        self.enforce_arena_budget();
    }

    /// Drop the in-memory vector caches of every row in one table once a flat
    /// scan cache owns the query-side copies. Rows keep their durable
    /// locations, so `read_vec` still serves full projections.
    pub fn evict_table_vectors(&mut self, table_id: u32) {
        let Some(vector_col) = self.tables.get(&table_id).and_then(|t| t.schema.vector_col())
        else {
            return;
        };
        if let Some(table) = self.tables.get_mut(&table_id) {
            for entry in table.rows.values_mut() {
                if matches!(entry.row.vals[vector_col], VVal::Vec(Some(_))) {
                    entry.row.vals[vector_col] = VVal::Vec(None);
                }
            }
        }
        self.arena_bytes = 0;
        for table in self.tables.values() {
            self.arena_bytes += table
                .rows
                .values()
                .map(|entry| row_vector_bytes(&entry.row, &table.schema))
                .sum::<usize>();
        }
    }

    fn enforce_arena_budget(&mut self) {
        if self.arena_bytes <= self.arena_budget {
            return;
        }
        let target = self.arena_budget * 9 / 10;
        let table_ids: Vec<u32> = self.tables.keys().copied().collect();
        for table_id in table_ids {
            let Some(vector_col) = self.tables.get(&table_id).and_then(|t| t.schema.vector_col()) else {
                continue;
            };
            let table = self.tables.get_mut(&table_id).unwrap();
            for entry in table.rows.values_mut() {
                if self.arena_bytes <= target {
                    return;
                }
                if let VVal::Vec(Some(v)) = &entry.row.vals[vector_col] {
                    self.arena_bytes = self.arena_bytes.saturating_sub(v.len() * 4);
                    entry.row.vals[vector_col] = VVal::Vec(None);
                }
            }
        }
    }

    pub fn read_vec(&self, table_id: u32, pk: i64) -> DbResult<Rc<Vec<f32>>> {
        let (vector_col, loc, cached) = {
            let table = self.tables.get(&table_id).ok_or(DbError::TableNotFound)?;
            let vector_col = table.schema.vector_col().ok_or(DbError::Internal)?;
            let entry = table.rows.get(&pk).ok_or(DbError::Internal)?;
            (vector_col, entry.loc, entry.row.cached_vec(vector_col))
        };
        if let Some(v) = cached {
            return Ok(v);
        }
        let mut bytes = vec![0u8; loc.len as usize];
        if loc.gen == 0 {
            self.wal_read.read_at(&mut bytes, loc.off).map_err(|_| DbError::Internal)?;
        } else {
            if let Some(file) = self.gen_readers.get(&(loc.gen, table_id)) {
                file.read_at(&mut bytes, loc.off).map_err(|_| DbError::Internal)?;
            } else {
                let path = self.dir.join(format!("d{}/t{}.rows", loc.gen, table_id));
                let file = File::open(path).map_err(|_| DbError::Internal)?;
                file.read_at(&mut bytes, loc.off).map_err(|_| DbError::Internal)?;
            }
        }
        let row = get_row(&mut Cur::new(&bytes))?;
        row.cached_vec(vector_col).ok_or(DbError::Internal)
    }

    pub fn should_checkpoint(&self) -> bool {
        self.wal_since_checkpoint
            > self
                .checkpoint_min_bytes
                .max(self.live_bytes.saturating_mul(self.checkpoint_live_factor))
    }

    pub fn maybe_checkpoint(&mut self) -> DbResult<()> {
        if self.should_checkpoint() {
            self.checkpoint().map_err(|_| DbError::Internal)?;
        }
        Ok(())
    }

    pub fn shutdown(&mut self) -> DbResult<()> {
        if self.wal_since_checkpoint > SHUTDOWN_CHECKPOINT_MIN {
            self.checkpoint().map_err(|_| DbError::Internal)?;
        } else {
            self.wal_write.sync_all().map_err(|_| DbError::Internal)?;
        }
        Ok(())
    }

    /// Compact all live rows into a new generation. The atomic `meta` rename
    /// is the checkpoint commit point; only after it do we truncate the WAL.
    pub fn checkpoint(&mut self) -> io::Result<()> {
        let new_gen = self.gen + 1;
        let new_dir = self.dir.join(format!("d{}", new_gen));
        if new_dir.exists() {
            std::fs::remove_dir_all(&new_dir)?;
        }
        std::fs::create_dir_all(&new_dir)?;
        let mut new_locs: HashMap<(u32, i64), RowLoc> = HashMap::new();
        for (table_id, table) in self.tables.iter() {
            let temp_path = new_dir.join(format!("t{}.rows.tmp", table_id));
            let mut file = File::create(&temp_path)?;
            let mut offset = 0u64;
            let mut keys: Vec<i64> = table.rows.keys().copied().collect();
            keys.sort_unstable();
            for pk in keys {
                let stored = &table.rows.get(&pk).unwrap().row;
                // Materialize at most one evicted row at a time; never refill
                // the whole vector arena just to compact the data files.
                let materialized;
                let row = if stored.vals.iter().any(|value| matches!(value, VVal::Vec(None))) {
                    let mut clone = stored.clone();
                    for value in clone.vals.iter_mut() {
                        if matches!(value, VVal::Vec(None)) {
                            let vector = self.read_vec(*table_id, pk).map_err(|_| {
                                io::Error::new(io::ErrorKind::InvalidData, "missing vector")
                            })?;
                            *value = VVal::Vec(Some(vector));
                        }
                    }
                    materialized = clone;
                    &materialized
                } else {
                    stored
                };
                let mut payload = Buf::new();
                put_row(&mut payload, row).map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "evicted row"))?;
                let mut frame = Buf::new();
                frame.u32(payload.b.len() as u32);
                frame.u32(crc32(&payload.b));
                frame.b.extend_from_slice(&payload.b);
                file.write_all(&frame.b)?;
                new_locs.insert(
                    (*table_id, pk),
                    RowLoc { gen: new_gen, off: offset + 8, len: payload.b.len() as u32 },
                );
                offset += frame.b.len() as u64;
            }
            file.sync_all()?;
            std::fs::rename(&temp_path, new_dir.join(format!("t{}.rows", table_id)))?;
        }
        File::open(&new_dir)?.sync_all()?;

        let meta_temp = self.dir.join("meta.new");
        let mut text = String::from("MVDB1\n");
        text.push_str(&format!("gen {}\nlsn {}\n", new_gen, self.lsn));
        for table in self.tables.values() {
            text.push_str(&format!("table {} {}\n", table.schema.id, table.schema.name));
            for col in &table.schema.columns {
                let (kind, dim) = match col.kind {
                    ColKind::BigInt => ("BIGINT", 0),
                    ColKind::Varchar(n) => ("VARCHAR", n),
                    ColKind::Vector(n) => ("VECTOR", n),
                };
                text.push_str(&format!(
                    "col {} {} {} {} {}\n",
                    table.schema.id, col.name, kind, dim, col.primary_key as u8
                ));
            }
        }
        for index in &self.indexes {
            text.push_str(&format!("vindex {} {} {}\n", index.name, index.table_id, index.column));
        }
        let mut meta = File::create(&meta_temp)?;
        meta.write_all(text.as_bytes())?;
        meta.sync_all()?;
        std::fs::rename(&meta_temp, self.dir.join("meta"))?;
        File::open(&self.dir)?.sync_all()?;

        // The new generation is committed. Swap readers and row locations.
        self.gen_readers.clear();
        for (table_id, table) in self.tables.iter_mut() {
            let path = new_dir.join(format!("t{}.rows", table_id));
            if path.exists() {
                self.gen_readers.insert((new_gen, *table_id), File::open(path)?);
            }
            for entry in table.rows.values_mut() {
                if let Some(loc) = new_locs.get(&(*table_id, entry.row.pk)) {
                    entry.loc = *loc;
                }
            }
        }
        self.wal_write.set_len(0)?;
        self.wal_write.sync_all()?;
        self.wal_len = 0;
        self.wal_since_checkpoint = 0;
        self.gen = new_gen;
        self.recompute_usage();
        self.remove_orphan_generations();
        File::open(&self.dir)?.sync_all()?;
        Ok(())
    }

    fn replay_wal(&mut self) -> io::Result<(u64, u64)> {
        let mut bytes = Vec::new();
        self.wal_read.read_to_end(&mut bytes)?;
        let mut pos = 0usize;
        let mut valid_end = 0u64;
        let mut max_lsn = self.lsn;
        'frames: while pos + 8 <= bytes.len() {
            let len = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
            let expected_crc = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap());
            if pos + 8 + len > bytes.len() {
                break;
            }
            let payload = &bytes[pos + 8..pos + 8 + len];
            if len < 8 || crc32(payload) != expected_crc {
                break;
            }
            let mut cur = Cur::new(payload);
            let record_lsn = match cur.u64() {
                Ok(v) => v,
                Err(_) => break,
            };
            let mut ops = Vec::new();
            let mut locs = Vec::new();
            while cur.p < payload.len() {
                let op_start = cur.p;
                let op = match get_op(&mut cur) {
                    Ok(op) => op,
                    Err(_) => break 'frames,
                };
                if let WalOp::UpsertRow { row, .. } = &op {
                    locs.push(RowLoc {
                        gen: 0,
                        off: (pos + 8 + op_start + 5) as u64,
                        len: match row_len(row) {
                            Ok(v) => v,
                            Err(_) => break 'frames,
                        },
                    });
                }
                ops.push(op);
            }
            max_lsn = max_lsn.max(record_lsn);
            if record_lsn > self.lsn {
                self.apply_ops(&ops, &locs);
            }
            pos += 8 + len;
            valid_end = pos as u64;
        }
        Ok((valid_end, max_lsn))
    }
}

fn row_vector_bytes(row: &Row, schema: &Schema) -> usize {
    match schema.vector_col() {
        Some(index) => match &row.vals[index] {
            VVal::Vec(Some(vector)) => vector.len() * 4,
            _ => 0,
        },
        None => 0,
    }
}

fn read_meta(path: &Path) -> io::Result<(u64, u64, Vec<Schema>, Vec<IndexDef>)> {
    let text = std::fs::read_to_string(path)?;
    let mut gen = 0;
    let mut lsn = 0;
    let mut schemas = Vec::new();
    let mut indexes = Vec::new();
    for line in text.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        match parts.as_slice() {
            ["gen", value] => gen = value.parse().unwrap_or(0),
            ["lsn", value] => lsn = value.parse().unwrap_or(0),
            ["table", id, name] => schemas.push(Schema {
                id: id.parse().unwrap_or(0),
                name: (*name).to_string(),
                columns: Vec::new(),
            }),
            ["col", table_id, name, kind, dim, pk] => {
                let kind = match *kind {
                    "BIGINT" => ColKind::BigInt,
                    "VARCHAR" => ColKind::Varchar(dim.parse().unwrap_or(0)),
                    "VECTOR" => ColKind::Vector(dim.parse().unwrap_or(0)),
                    _ => continue,
                };
                if let Some(schema) = schemas.iter_mut().find(|s| s.id == table_id.parse().unwrap_or(0)) {
                    schema.columns.push(Column {
                        name: (*name).to_string(),
                        kind,
                        primary_key: *pk == "1",
                    });
                }
            }
            ["vindex", name, table_id, column] => indexes.push(IndexDef {
                name: (*name).to_string(),
                table_id: table_id.parse().unwrap_or(0),
                column: (*column).to_string(),
            }),
            _ => {}
        }
    }
    Ok((gen, lsn, schemas, indexes))
}

fn load_checkpoint_rows(
    file: &mut File,
    gen: u64,
    table: &mut TableData,
    cache_vectors: bool,
) -> io::Result<()> {
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    let mut pos = 0usize;
    while pos + 8 <= bytes.len() {
        let len = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        let expected_crc = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap());
        if pos + 8 + len > bytes.len() {
            break;
        }
        let payload = &bytes[pos + 8..pos + 8 + len];
        if crc32(payload) != expected_crc {
            break;
        }
        let mut row = get_row(&mut Cur::new(payload))
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad row"))?;
        if !cache_vectors {
            if let Some(vector_col) = table.schema.vector_col() {
                row.vals[vector_col] = VVal::Vec(None);
            }
        }
        table.rows.insert(
            row.pk,
            RowEntry { row, loc: RowLoc { gen, off: (pos + 8) as u64, len: len as u32 } },
        );
        pos += 8 + len;
    }
    Ok(())
}

impl From<io::Error> for DbError {
    fn from(_: io::Error) -> Self {
        DbError::Internal
    }
}
