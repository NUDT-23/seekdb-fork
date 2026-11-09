//! SQL execution, visibility overlays and transaction semantics.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::rc::Rc;

use crate::error::{DbError, DbResult};
use crate::parser::{CmpOp, Expr, Literal, OrderDir, OrderKey, ProjItem, SelectStmt, Stmt};
use crate::storage::{IndexDef, Row, Store, VVal, WalOp};
use crate::value::{
    check_varchar, l2_distance, l2_distance_ann, parse_vector, ColKind, Schema,
};

#[derive(Debug, Clone, PartialEq)]
pub enum OutValue {
    Int(i64),
    Str(String),
    Float(f64),
    Vector(Vec<f32>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExecResult {
    Ok { affected_rows: Option<u64> },
    Query { columns: Vec<String>, rows: Vec<Vec<OutValue>> },
}

impl ExecResult {
    pub fn ok() -> Self {
        Self::Ok { affected_rows: None }
    }
    pub fn affected(n: u64) -> Self {
        Self::Ok { affected_rows: Some(n) }
    }
}

#[derive(Default)]
struct TableOverlay {
    upserts: HashMap<i64, Row>,
    deletes: HashSet<i64>,
}

#[derive(Default)]
struct Transaction {
    ops: Vec<WalOp>,
    tables: HashMap<u32, TableOverlay>,
    created_tables: HashMap<u32, Schema>,
    created_indexes: Vec<IndexDef>,
}

pub struct Engine {
    pub store: Store,
    txn: Option<Transaction>,
    pub ann_mode: bool,
}

impl Engine {
    pub fn new(store: Store) -> Self {
        Self { store, txn: None, ann_mode: false }
    }

    pub fn set_ann_mode(&mut self, ann: bool) {
        self.ann_mode = ann;
    }

    pub fn execute(&mut self, sql: &str) -> DbResult<ExecResult> {
        let stmt = crate::parser::parse(sql)?;
        self.execute_stmt(stmt)
    }

    fn execute_stmt(&mut self, stmt: Stmt) -> DbResult<ExecResult> {
        match stmt {
            Stmt::Begin => {
                if self.txn.is_some() {
                    return Err(DbError::InvalidTransactionState);
                }
                self.txn = Some(Transaction::default());
                Ok(ExecResult::ok())
            }
            Stmt::Commit => {
                let txn = self.txn.take().ok_or(DbError::InvalidTransactionState)?;
                self.store.commit(&txn.ops)?;
                if let Err(error) = self.store.maybe_checkpoint() {
                    eprintln!("minivectordb: optional checkpoint failed: {}", error);
                }
                Ok(ExecResult::ok())
            }
            Stmt::Rollback => {
                if self.txn.take().is_none() {
                    return Err(DbError::InvalidTransactionState);
                }
                Ok(ExecResult::ok())
            }
            Stmt::CreateTable { name, columns } => {
                self.create_table(name, columns)?;
                Ok(ExecResult::ok())
            }
            Stmt::CreateVectorIndex { index, table, column } => {
                self.create_vector_index(index, table, column)?;
                Ok(ExecResult::ok())
            }
            Stmt::Insert { table, cols, rows } => {
                let count = self.insert(table, cols, rows)?;
                Ok(ExecResult::affected(count))
            }
            Stmt::Update { table, sets, filter } => {
                let count = self.update(table, sets, filter)?;
                Ok(ExecResult::affected(count))
            }
            Stmt::Delete { table, filter } => {
                let count = self.delete(table, filter)?;
                Ok(ExecResult::affected(count))
            }
            Stmt::Select(stmt) => self.select(stmt),
        }
    }

    /// Finish one durable statement or transaction and compact if useful.
    fn publish(&mut self, ops: Vec<WalOp>) -> DbResult<()> {
        if let Some(txn) = self.txn.as_mut() {
            for op in ops {
                apply_txn_op(txn, &op);
                txn.ops.push(op);
            }
            Ok(())
        } else {
            self.store.commit(&ops)?;
            if let Err(error) = self.store.maybe_checkpoint() {
                eprintln!("minivectordb: optional checkpoint failed: {}", error);
            }
            Ok(())
        }
    }

    fn resolve_table(&self, name: &str) -> DbResult<(u32, Schema)> {
        if let Some(txn) = &self.txn {
            if let Some(schema) = txn
                .created_tables
                .values()
                .find(|schema| schema.name.eq_ignore_ascii_case(name))
            {
                return Ok((schema.id, schema.clone()));
            }
        }
        self.store
            .tables
            .values()
            .find(|table| table.schema.name.eq_ignore_ascii_case(name))
            .map(|table| (table.schema.id, table.schema.clone()))
            .ok_or(DbError::TableNotFound)
    }

    fn create_table(
        &mut self,
        name: String,
        columns: Vec<(String, ColKind, bool)>,
    ) -> DbResult<()> {
        if self.resolve_table(&name).is_ok() {
            return Err(DbError::TableExists);
        }
        if columns.is_empty() {
            return Err(DbError::SyntaxError);
        }
        let mut seen = HashSet::new();
        let mut pk_count = 0;
        let mut vector_count = 0;
        for (col_name, kind, primary_key) in &columns {
            if !seen.insert(col_name.to_ascii_lowercase()) {
                return Err(DbError::SyntaxError);
            }
            if *primary_key {
                pk_count += 1;
                if *kind != ColKind::BigInt {
                    return Err(DbError::UnsupportedSql);
                }
            }
            match kind {
                ColKind::Vector(dim) => {
                    vector_count += 1;
                    if *dim == 0 {
                        return Err(DbError::DimensionMismatch);
                    }
                }
                ColKind::Varchar(n) if *n == 0 => return Err(DbError::StringTooLong),
                _ => {}
            }
        }
        if pk_count != 1 || vector_count > 1 {
            return Err(DbError::UnsupportedSql);
        }
        let schema = Schema {
            id: self.store.next_table_id(),
            name,
            columns: columns
                .into_iter()
                .map(|(name, kind, primary_key)| crate::value::Column { name, kind, primary_key })
                .collect(),
        };
        self.publish(vec![WalOp::CreateTable(schema)])
    }

    fn create_vector_index(&mut self, name: String, table: String, column: String) -> DbResult<()> {
        let (table_id, schema) = self.resolve_table(&table)?;
        let col_index = schema.col_index(&column).ok_or(DbError::ColumnNotFound)?;
        if !matches!(schema.columns[col_index].kind, ColKind::Vector(_)) {
            return Err(DbError::UnsupportedSql);
        }
        let duplicate = self.store.indexes.iter().any(|idx| idx.name.eq_ignore_ascii_case(&name))
            || self
                .txn
                .as_ref()
                .map_or(false, |txn| txn.created_indexes.iter().any(|idx| idx.name.eq_ignore_ascii_case(&name)));
        if duplicate {
            return Err(DbError::IndexExists);
        }
        self.publish(vec![WalOp::CreateVectorIndex(IndexDef { name, table_id, column })])
    }

    fn insert(
        &mut self,
        table_name: String,
        columns: Option<Vec<String>>,
        input_rows: Vec<Vec<Literal>>,
    ) -> DbResult<u64> {
        let (table_id, schema) = self.resolve_table(&table_name)?;
        let column_order = match columns {
            Some(names) => {
                if names.len() != schema.columns.len() {
                    return Err(DbError::UnsupportedSql);
                }
                let mut order = Vec::with_capacity(names.len());
                let mut seen = HashSet::new();
                for name in names {
                    let index = schema.col_index(&name).ok_or(DbError::ColumnNotFound)?;
                    if !seen.insert(index) {
                        return Err(DbError::SyntaxError);
                    }
                    order.push(index);
                }
                order
            }
            None => (0..schema.columns.len()).collect(),
        };
        let pk_col = schema.pk_index();
        let mut pending_pks = HashSet::new();
        let mut ops = Vec::new();
        for input in input_rows {
            if input.len() != column_order.len() {
                return Err(DbError::SyntaxError);
            }
            let mut values: Vec<Option<VVal>> = (0..schema.columns.len()).map(|_| None).collect();
            for (literal, col_index) in input.into_iter().zip(column_order.iter().copied()) {
                values[col_index] = Some(bind_literal(&schema.columns[col_index].kind, literal)?);
            }
            let vals: Vec<VVal> = values
                .into_iter()
                .map(|value| value.ok_or(DbError::UnsupportedSql))
                .collect::<DbResult<_>>()?;
            let pk = match vals[pk_col] {
                VVal::Int(pk) => pk,
                _ => return Err(DbError::TypeMismatch),
            };
            if !pending_pks.insert(pk) || self.visible_row(table_id, pk)?.is_some() {
                return Err(DbError::DuplicateKey);
            }
            ops.push(WalOp::UpsertRow { table_id, row: Row { pk, vals } });
        }
        let count = ops.len() as u64;
        self.publish(ops)?;
        Ok(count)
    }

    fn update(
        &mut self,
        table_name: String,
        sets: Vec<(String, Literal)>,
        filter: Option<Expr>,
    ) -> DbResult<u64> {
        let (table_id, schema) = self.resolve_table(&table_name)?;
        validate_filter(&schema, filter.as_ref())?;
        let mut bound_sets = Vec::new();
        for (name, literal) in sets {
            let index = schema.col_index(&name).ok_or(DbError::ColumnNotFound)?;
            if schema.columns[index].primary_key {
                return Err(DbError::UnsupportedSql);
            }
            bound_sets.push((index, bind_literal(&schema.columns[index].kind, literal)?));
        }
        let rows = self.visible_rows(table_id)?;
        let mut ops = Vec::new();
        for row in rows {
            if !eval_filter(&schema, &row, filter.as_ref())? {
                continue;
            }
            let mut updated = row.clone();
            for (index, value) in &bound_sets {
                updated.vals[*index] = value.clone();
            }
            updated = self.materialize_row(table_id, updated)?;
            ops.push(WalOp::UpsertRow { table_id, row: updated });
        }
        let count = ops.len() as u64;
        self.publish(ops)?;
        Ok(count)
    }

    fn delete(&mut self, table_name: String, filter: Option<Expr>) -> DbResult<u64> {
        let (table_id, schema) = self.resolve_table(&table_name)?;
        validate_filter(&schema, filter.as_ref())?;
        let rows = self.visible_rows(table_id)?;
        let mut ops = Vec::new();
        for row in rows {
            if eval_filter(&schema, &row, filter.as_ref())? {
                ops.push(WalOp::DeleteRow { table_id, pk: row.pk });
            }
        }
        let count = ops.len() as u64;
        self.publish(ops)?;
        Ok(count)
    }

    fn materialize_row(&mut self, table_id: u32, mut row: Row) -> DbResult<Row> {
        let missing = row.vals.iter().any(|value| matches!(value, VVal::Vec(None)));
        if missing {
            let vector = self.store.read_vec(table_id, row.pk)?;
            for value in row.vals.iter_mut() {
                if matches!(value, VVal::Vec(None)) {
                    *value = VVal::Vec(Some(Rc::clone(&vector)));
                }
            }
        }
        Ok(row)
    }

    fn visible_row(&self, table_id: u32, pk: i64) -> DbResult<Option<Row>> {
        if let Some(txn) = &self.txn {
            if let Some(overlay) = txn.tables.get(&table_id) {
                if let Some(row) = overlay.upserts.get(&pk) {
                    return Ok(Some(row.clone()));
                }
                if overlay.deletes.contains(&pk) {
                    return Ok(None);
                }
            }
            if txn.created_tables.contains_key(&table_id) {
                return Ok(None);
            }
        }
        Ok(self
            .store
            .tables
            .get(&table_id)
            .and_then(|table| table.rows.get(&pk))
            .map(|entry| entry.row.clone()))
    }

    fn visible_rows(&self, table_id: u32) -> DbResult<Vec<Row>> {
        let overlay = self.txn.as_ref().and_then(|txn| txn.tables.get(&table_id));
        let mut rows = Vec::new();
        if let Some(table) = self.store.tables.get(&table_id) {
            for (pk, entry) in &table.rows {
                if overlay.map_or(false, |o| o.deletes.contains(pk) || o.upserts.contains_key(pk)) {
                    continue;
                }
                rows.push(entry.row.clone());
            }
        }
        if let Some(overlay) = overlay {
            rows.extend(overlay.upserts.values().cloned());
        }
        rows.sort_by_key(|row| row.pk);
        Ok(rows)
    }

    fn visible_row_refs(&self, table_id: u32) -> DbResult<Vec<&Row>> {
        let overlay = self.txn.as_ref().and_then(|txn| txn.tables.get(&table_id));
        let mut rows = Vec::new();
        if let Some(table) = self.store.tables.get(&table_id) {
            for (pk, entry) in &table.rows {
                if overlay.map_or(false, |o| o.deletes.contains(pk) || o.upserts.contains_key(pk)) {
                    continue;
                }
                rows.push(&entry.row);
            }
        }
        if let Some(overlay) = overlay {
            rows.extend(overlay.upserts.values());
        }
        rows.sort_by_key(|row| row.pk);
        Ok(rows)
    }

    fn select(&mut self, stmt: SelectStmt) -> DbResult<ExecResult> {
        let (table_id, schema) = self.resolve_table(&stmt.table)?;
        validate_filter(&schema, stmt.filter.as_ref())?;
        let items = bind_projection(&schema, &stmt.items)?;
        let l2_index = items.iter().position(|item| matches!(item, BoundProj::L2 { .. }));
        if items.iter().filter(|item| matches!(item, BoundProj::L2 { .. })).count() > 1 {
            return Err(DbError::UnsupportedSql);
        }
        if let Some(BoundProj::L2 { vector_col, qvec, .. }) = l2_index.map(|i| &items[i]) {
            let dim = match schema.columns[*vector_col].kind {
                ColKind::Vector(dim) => dim,
                _ => return Err(DbError::TypeMismatch),
            };
            if qvec.len() as u32 != dim {
                return Err(DbError::DimensionMismatch);
            }
        }
        let order = bind_order(&schema, &items, &stmt.order)?;
        let topk_limit = match (l2_index, order.as_slice(), stmt.limit) {
            (Some(_), [BoundOrder::L2 { desc: false }], Some(limit)) => Some(limit as usize),
            _ => None,
        };
        let visible = self.visible_row_refs(table_id)?;
        let mut records = Vec::new();
        let mut topk = BinaryHeap::new();
        for row in visible {
            if !eval_filter(&schema, row, stmt.filter.as_ref())? {
                continue;
            }
            let distance = match l2_index.map(|i| &items[i]) {
                Some(BoundProj::L2 { vector_col, qvec, .. }) => {
                    let vector = match row.cached_vec(*vector_col) {
                        Some(vector) => vector,
                        None => self.store.read_vec(table_id, row.pk)?,
                    };
                    Some(if self.ann_mode {
                        l2_distance_ann(&vector, qvec)
                    } else {
                        l2_distance(&vector, qvec)
                    })
                }
                _ => None,
            };
            if let Some(limit) = topk_limit {
                if limit == 0 {
                    continue;
                }
                let candidate = HeapItem { row, distance: distance.unwrap() };
                if topk.len() < limit {
                    topk.push(candidate);
                } else if topk.peek().map_or(false, |worst| candidate < *worst) {
                    topk.pop();
                    topk.push(candidate);
                }
            } else {
                records.push(SelectedRow { row, distance });
            }
        }
        if topk_limit.is_some() {
            records.extend(topk.into_iter().map(|item| SelectedRow {
                row: item.row,
                distance: Some(item.distance),
            }));
        }

        if !order.is_empty() {
            records.sort_by(|a, b| {
                for key in &order {
                    let mut cmp = match key {
                        BoundOrder::Column { index, .. } => {
                            compare_values(&a.row.vals[*index], &b.row.vals[*index])
                        }
                        BoundOrder::L2 { .. } => a
                            .distance
                            .partial_cmp(&b.distance)
                            .unwrap_or(Ordering::Equal),
                    };
                    if key.desc() {
                        cmp = cmp.reverse();
                    }
                    if cmp != Ordering::Equal {
                        return cmp;
                    }
                }
                a.row.pk.cmp(&b.row.pk)
            });
        }
        if let Some(limit) = stmt.limit {
            records.truncate(limit as usize);
        }

        let mut columns = Vec::new();
        for item in &items {
            match item {
                BoundProj::Column { index } => columns.push(schema.columns[*index].name.clone()),
                BoundProj::L2 { alias, .. } => columns.push(alias.clone()),
            }
        }
        let mut output_rows = Vec::with_capacity(records.len());
        for selected in records {
            let mut output = Vec::with_capacity(items.len());
            for item in &items {
                match item {
                    BoundProj::Column { index } => {
                        if matches!(selected.row.vals[*index], VVal::Vec(None)) {
                            let vector = self.store.read_vec(table_id, selected.row.pk)?;
                            output.push(OutValue::Vector(vector.as_ref().clone()));
                        } else {
                            output.push(out_value(&selected.row.vals[*index])?);
                        }
                    }
                    BoundProj::L2 { .. } => output.push(OutValue::Float(selected.distance.unwrap())),
                }
            }
            output_rows.push(output);
        }
        Ok(ExecResult::Query { columns, rows: output_rows })
    }
}

struct SelectedRow<'a> {
    row: &'a Row,
    distance: Option<f64>,
}

/// Max-heap ordered so its root is the worst current TopK candidate:
/// greater distance first, then greater primary key for exact ties.
struct HeapItem<'a> {
    row: &'a Row,
    distance: f64,
}

impl PartialEq for HeapItem<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.row.pk == other.row.pk
    }
}
impl Eq for HeapItem<'_> {}
impl PartialOrd for HeapItem<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapItem<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.total_cmp(&other.distance).then(self.row.pk.cmp(&other.row.pk))
    }
}

#[derive(Clone)]
enum BoundProj {
    Column { index: usize },
    L2 { vector_col: usize, qvec: Vec<f32>, alias: String },
}

enum BoundOrder {
    Column { index: usize, desc: bool },
    L2 { desc: bool },
}

impl BoundOrder {
    fn desc(&self) -> bool {
        match self {
            BoundOrder::Column { desc, .. } | BoundOrder::L2 { desc } => *desc,
        }
    }
}

fn bind_projection(schema: &Schema, input: &Option<Vec<ProjItem>>) -> DbResult<Vec<BoundProj>> {
    match input {
        None => Ok((0..schema.columns.len()).map(|index| BoundProj::Column { index }).collect()),
        Some(items) => items
            .iter()
            .map(|item| match item {
                ProjItem::Col(name) => schema
                    .col_index(name)
                    .map(|index| BoundProj::Column { index })
                    .ok_or(DbError::ColumnNotFound),
                ProjItem::L2 { col, qvec, alias } => {
                    let vector_col = schema.col_index(col).ok_or(DbError::ColumnNotFound)?;
                    if !matches!(schema.columns[vector_col].kind, ColKind::Vector(_)) {
                        return Err(DbError::TypeMismatch);
                    }
                    Ok(BoundProj::L2 {
                        vector_col,
                        qvec: qvec.clone(),
                        alias: alias.clone().unwrap_or_else(|| "distance".to_string()),
                    })
                }
            })
            .collect(),
    }
}

fn bind_order(schema: &Schema, items: &[BoundProj], order: &[OrderKey]) -> DbResult<Vec<BoundOrder>> {
    order
        .iter()
        .map(|key| match key {
            OrderKey::Col { name, dir } => {
                let desc = *dir == OrderDir::Desc;
                if items.iter().any(|item| matches!(item, BoundProj::L2 { alias, .. } if alias.eq_ignore_ascii_case(name))) {
                    return Ok(BoundOrder::L2 { desc });
                }
                let index = schema.col_index(name).ok_or(DbError::ColumnNotFound)?;
                if matches!(schema.columns[index].kind, ColKind::Vector(_)) {
                    return Err(DbError::UnsupportedSql);
                }
                Ok(BoundOrder::Column { index, desc })
            }
        })
        .collect()
}

fn bind_literal(kind: &ColKind, literal: Literal) -> DbResult<VVal> {
    match (kind, literal) {
        (ColKind::BigInt, Literal::Int(value)) => Ok(VVal::Int(value)),
        (ColKind::BigInt, _) => Err(DbError::TypeMismatch),
        (ColKind::Varchar(max), Literal::Str(value)) => {
            check_varchar(&value, *max)?;
            Ok(VVal::Str(value))
        }
        (ColKind::Varchar(_), _) => Err(DbError::TypeMismatch),
        (ColKind::Vector(dim), Literal::Str(text)) => {
            let inner = text.trim();
            let inner = inner
                .strip_prefix('[')
                .and_then(|s| s.strip_suffix(']'))
                .ok_or(DbError::InvalidVector)?;
            Ok(VVal::Vec(Some(Rc::new(parse_vector(inner, Some(*dim))?))))
        }
        (ColKind::Vector(_), _) => Err(DbError::TypeMismatch),
    }
}

fn validate_filter(schema: &Schema, expr: Option<&Expr>) -> DbResult<()> {
    match expr {
        None => Ok(()),
        Some(Expr::And(left, right)) | Some(Expr::Or(left, right)) => {
            validate_filter(schema, Some(left))?;
            validate_filter(schema, Some(right))
        }
        Some(Expr::Cmp { col, lit, .. }) => {
            let index = schema.col_index(col).ok_or(DbError::ColumnNotFound)?;
            match (&schema.columns[index].kind, lit) {
                (ColKind::BigInt, Literal::Int(_)) | (ColKind::Varchar(_), Literal::Str(_)) => Ok(()),
                (ColKind::Vector(_), _) => Err(DbError::UnsupportedSql),
                _ => Err(DbError::TypeMismatch),
            }
        }
    }
}

fn eval_filter(schema: &Schema, row: &Row, expr: Option<&Expr>) -> DbResult<bool> {
    match expr {
        None => Ok(true),
        Some(Expr::And(left, right)) => Ok(eval_filter(schema, row, Some(left))?
            && eval_filter(schema, row, Some(right))?),
        Some(Expr::Or(left, right)) => Ok(eval_filter(schema, row, Some(left))?
            || eval_filter(schema, row, Some(right))?),
        Some(Expr::Cmp { col, op, lit }) => {
            let index = schema.col_index(col).ok_or(DbError::ColumnNotFound)?;
            let ordering = compare_literal(&row.vals[index], lit)?;
            Ok(match op {
                CmpOp::Eq => ordering == Ordering::Equal,
                CmpOp::Ne => ordering != Ordering::Equal,
                CmpOp::Lt => ordering == Ordering::Less,
                CmpOp::Le => ordering != Ordering::Greater,
                CmpOp::Gt => ordering == Ordering::Greater,
                CmpOp::Ge => ordering != Ordering::Less,
            })
        }
    }
}

fn compare_literal(value: &VVal, literal: &Literal) -> DbResult<Ordering> {
    match (value, literal) {
        (VVal::Int(left), Literal::Int(right)) => Ok(left.cmp(right)),
        (VVal::Str(left), Literal::Str(right)) => Ok(left.cmp(right)),
        _ => Err(DbError::TypeMismatch),
    }
}

fn compare_values(left: &VVal, right: &VVal) -> Ordering {
    match (left, right) {
        (VVal::Int(a), VVal::Int(b)) => a.cmp(b),
        (VVal::Str(a), VVal::Str(b)) => a.cmp(b),
        _ => Ordering::Equal,
    }
}

fn out_value(value: &VVal) -> DbResult<OutValue> {
    match value {
        VVal::Int(v) => Ok(OutValue::Int(*v)),
        VVal::Str(v) => Ok(OutValue::Str(v.clone())),
        VVal::Vec(Some(v)) => Ok(OutValue::Vector(v.as_ref().clone())),
        VVal::Vec(None) => Err(DbError::Internal),
    }
}

fn apply_txn_op(txn: &mut Transaction, op: &WalOp) {
    match op {
        WalOp::CreateTable(schema) => {
            txn.created_tables.insert(schema.id, schema.clone());
        }
        WalOp::CreateVectorIndex(index) => txn.created_indexes.push(index.clone()),
        WalOp::UpsertRow { table_id, row } => {
            let overlay = txn.tables.entry(*table_id).or_default();
            overlay.deletes.remove(&row.pk);
            overlay.upserts.insert(row.pk, row.clone());
        }
        WalOp::DeleteRow { table_id, pk } => {
            let overlay = txn.tables.entry(*table_id).or_default();
            overlay.upserts.remove(pk);
            overlay.deletes.insert(*pk);
        }
    }
}
