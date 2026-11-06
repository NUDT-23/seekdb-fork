/**
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define USING_LOG_PREFIX SQL_ENG

#include "sql/engine/dml/ob_ai_column_dml_helper.h"

#include "lib/allocator/page_arena.h"
#include "lib/hash_func/murmur_hash.h"
#include "lib/string/ob_sql_string.h"
#include "lib/utility/ob_mod_define.h"
#include "share/inner_table/ob_inner_table_schema.h"
#include "share/ob_dml_sql_splicer.h"
#include "share/ob_server_struct.h"
#include "share/schema/ob_schema_struct.h"
#include "share/schema/ob_schema_getter_guard.h"
#include "share/schema/ob_table_schema.h"
#include "sql/engine/dml/ob_table_modify_op.h"
#include "sql/engine/ob_exec_context.h"
#include "sql/ob_sql_utils.h"

namespace oceanbase
{
namespace sql
{
using namespace common;
using namespace share;

namespace
{

// The system database that holds __all_ai_column_task.  The nested SQL below
// runs on the user session (whose current database is the user database), so
// the task table must be fully qualified.
const char *OB_AI_SYS_DATABASE_NAME = "oceanbase";

// per statement table context, resolved once per process_after_rows call
struct ObAITableCtx
{
  ObAITableCtx() : table_id_(OB_INVALID_ID), db_name_(), table_name_(),
                   rowkey_cids_(), rowkey_names_(), items_(), found_(false) {}
  uint64_t table_id_;
  ObString db_name_;
  ObString table_name_;
  ObArray<uint64_t> rowkey_cids_;
  ObArray<ObString> rowkey_names_;
  ObArray<ObAIColumnItem> items_;
  bool found_;
};

int get_exec_schema_guard(ObTableModifyOp &op, schema::ObSchemaGetterGuard *&guard)
{
  int ret = OB_SUCCESS;
  guard = nullptr;
  ObExecContext &exec_ctx = op.get_exec_ctx();
  if (OB_NOT_NULL(exec_ctx.get_sql_ctx()) && OB_NOT_NULL(exec_ctx.get_sql_ctx()->schema_guard_)) {
    guard = exec_ctx.get_sql_ctx()->schema_guard_;
  } else if (OB_NOT_NULL(exec_ctx.get_my_session())) {
    guard = &exec_ctx.get_my_session()->get_schema_guard();
  }
  if (OB_ISNULL(guard)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("no schema guard available for ai column dml helper", K(ret));
  }
  return ret;
}

// Resolve the target table schema info needed to build the maintenance SQL.
int load_ai_table_ctx(schema::ObSchemaGetterGuard &guard,
                      const uint64_t table_id,
                      const int64_t schema_version,
                      ObIAllocator &alloc,
                      ObAITableCtx &table_ctx)
{
  int ret = OB_SUCCESS;
  const schema::ObTableSchema *table_schema = nullptr;
  table_ctx.found_ = false;
  table_ctx.table_id_ = table_id;
  if (OB_ISNULL(GCTX.schema_service_)) {
    // observer not ready (bootstrap), treat as no AI columns
  } else {
    // the process wide cache is initialized lazily and idempotently; when
    // another thread initializes it concurrently the error is benign
    int init_ret = ObAiColumnCache::get_instance().init();
    if (OB_SUCCESS != init_ret) {
      LOG_WARN("init ai column cache failed", K(init_ret));
    }
    if (OB_FAIL(ObAiColumnCache::get_instance().get_table_items(
                 table_id, schema_version, alloc, table_ctx.items_, table_ctx.found_))) {
      LOG_WARN("load ai column items failed", K(ret), K(table_id), K(schema_version));
    }
  }
  if (OB_SUCC(ret) && table_ctx.found_) {
    if (OB_FAIL(guard.get_table_schema(table_id, table_schema))) {
      LOG_WARN("get table schema failed", K(ret), K(table_id));
    } else if (OB_ISNULL(table_schema)) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("table schema is null", K(ret), K(table_id));
    } else {
      uint64_t db_id = table_schema->get_database_id();
      const schema::ObDatabaseSchema *db_schema = nullptr;
      if (OB_FAIL(guard.get_database_schema(db_id, db_schema))) {
        LOG_WARN("get database schema failed", K(ret), K(db_id));
      } else if (OB_ISNULL(db_schema)) {
        // fall back to the unqualified table name when db resolution fails
      } else if (OB_FAIL(ob_write_string(alloc, db_schema->get_database_name_str(),
                                         table_ctx.db_name_))) {
        LOG_WARN("write db name failed", K(ret));
      }
      if (OB_SUCC(ret)) {
        if (OB_FAIL(ob_write_string(alloc, table_schema->get_table_name_str(),
                                    table_ctx.table_name_))) {
          LOG_WARN("write table name failed", K(ret));
        }
      }
      for (int64_t i = 0; OB_SUCC(ret) && i < table_schema->get_rowkey_info().get_size(); ++i) {
        uint64_t cid = OB_INVALID_ID;
        const schema::ObColumnSchemaV2 *col_schema = nullptr;
        ObString name_copy;
        if (OB_FAIL(table_schema->get_rowkey_info().get_column_id(i, cid))) {
          LOG_WARN("get rowkey column id failed", K(ret), K(i));
        } else if (OB_ISNULL(col_schema = table_schema->get_column_schema(cid))) {
          ret = OB_ERR_UNEXPECTED;
          LOG_WARN("rowkey column schema not found", K(ret), K(cid));
        } else if (OB_FAIL(ob_write_string(alloc, col_schema->get_column_name_str(), name_copy))) {
          LOG_WARN("write rowkey name failed", K(ret));
        } else if (OB_FAIL(table_ctx.rowkey_cids_.push_back(cid))) {
          LOG_WARN("push back rowkey cid failed", K(ret));
        } else if (OB_FAIL(table_ctx.rowkey_names_.push_back(name_copy))) {
          LOG_WARN("push back rowkey name failed", K(ret));
        }
      }
    }
  }
  return ret;
}

// Build the rowkey predicate of one row image, e.g. `id` = 2.
// The format must stay in sync with the AI worker: each rowkey column is
// printed as "`col_name` = literal" joined by " and ".
int gen_rowkey_predicate(ObEvalCtx &eval_ctx,
                         const UIntFixedArray &column_ids,
                         const ObIArray<ObExpr *> &row_exprs,
                         const ObIArray<uint64_t> &rowkey_cids,
                         const ObIArray<ObString> &rowkey_names,
                         ObIAllocator &alloc,
                         const ObObjPrintParams &print_params,
                         char *&buf,
                         int64_t &len,
                         int64_t &pos)
{
  int ret = OB_SUCCESS;
  const char *column_fmt = "`%.*s` = ";
  const char *delimiter = " and ";
  ObDatum *col_datum = nullptr;
  if (OB_ISNULL(print_params.tz_info_)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (rowkey_cids.count() <= 0 || rowkey_cids.count() != rowkey_names.count()) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("rowkey info of ai table is invalid", K(ret), K(rowkey_cids.count()),
             K(rowkey_names.count()));
  }
  for (int64_t i = 0; OB_SUCC(ret) && i < rowkey_cids.count(); ++i) {
    const uint64_t cid = rowkey_cids.at(i);
    int64_t expr_idx = OB_INVALID_INDEX;
    for (int64_t j = 0; OB_SUCC(ret) && OB_INVALID_INDEX == expr_idx && j < column_ids.count(); ++j) {
      if (column_ids.at(j) == cid) {
        expr_idx = j;
      }
    }
    if (OB_FAIL(ret)) {
    } else if (OB_UNLIKELY(OB_INVALID_INDEX == expr_idx)
               || OB_UNLIKELY(expr_idx >= row_exprs.count())
               || OB_ISNULL(row_exprs.at(expr_idx))) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("rowkey column not found in dml row exprs", K(ret), K(cid), K(expr_idx));
    } else if (OB_FAIL(row_exprs.at(expr_idx)->eval(eval_ctx, col_datum))) {
      LOG_WARN("eval rowkey expr failed", K(ret), K(cid));
    } else if (OB_FAIL(databuff_printf(buf, len, pos, alloc, column_fmt,
                                       rowkey_names.at(i).length(), rowkey_names.at(i).ptr()))) {
      LOG_WARN("print rowkey column name failed", K(ret));
    } else {
      ObObj col_obj;
      if (OB_FAIL(col_datum->to_obj(col_obj, row_exprs.at(expr_idx)->obj_meta_))) {
        LOG_WARN("convert rowkey datum to obj failed", K(ret), K(cid));
      } else if (OB_FAIL(col_obj.print_sql_literal(buf, len, pos, alloc, print_params))) {
        LOG_WARN("print rowkey sql literal failed", K(ret), K(cid));
      } else if (OB_FAIL(databuff_printf(buf, len, pos, alloc, "%s", delimiter))) {
        LOG_WARN("print rowkey delimiter failed", K(ret));
      }
    }
  }
  if (OB_SUCC(ret)) {
    pos -= STRLEN(delimiter);
    buf[pos++] = 0;
  }
  return ret;
}

// "`db`.`table`" of the user table, falls back to "`table`" when the db name
// cannot be resolved (tests use a single database).
int gen_user_table_name(const ObAITableCtx &table_ctx,
                        ObIAllocator &alloc,
                        char *&buf,
                        int64_t &len,
                        int64_t &pos)
{
  int ret = OB_SUCCESS;
  if (table_ctx.db_name_.empty()) {
    if (OB_FAIL(databuff_printf(buf, len, pos, alloc, "`%.*s`",
                                table_ctx.table_name_.length(), table_ctx.table_name_.ptr()))) {
      LOG_WARN("print table name failed", K(ret));
    }
  } else if (OB_FAIL(databuff_printf(buf, len, pos, alloc, "`%.*s`.`%.*s`",
                                     table_ctx.db_name_.length(), table_ctx.db_name_.ptr(),
                                     table_ctx.table_name_.length(), table_ctx.table_name_.ptr()))) {
    LOG_WARN("print qualified table name failed", K(ret));
  }
  return ret;
}

// "`oceanbase`.`__all_ai_column_task`"
int gen_task_table_name(ObIAllocator &alloc, char *&buf, int64_t &len, int64_t &pos)
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(databuff_printf(buf, len, pos, alloc, "`%s`.`%s`",
                              OB_AI_SYS_DATABASE_NAME, OB_ALL_AI_COLUMN_TASK_TNAME))) {
    LOG_WARN("print task table name failed", K(ret));
  }
  return ret;
}

// update `<db>`.`<table>` set the hidden AI columns of one AI column of one row
int emit_invalidate_stmt(ObTableModifyOp &op,
                         const ObAITableCtx &table_ctx,
                         const ObAIColumnItem &item,
                         const char *request_id,
                         const ObString &rowkey_pred,
                         ObIAllocator &alloc)
{
  int ret = OB_SUCCESS;
  char *name_buf = nullptr;
  int64_t name_len = 0;
  int64_t name_pos = 0;
  char *pred_buf = nullptr;
  int64_t pred_len = 0;
  int64_t pred_pos = 0;
  ObSqlString sql;
  if (OB_FAIL(gen_user_table_name(table_ctx, alloc, name_buf, name_len, name_pos))) {
    LOG_WARN("gen user table name failed", K(ret));
  } else if (OB_FAIL(databuff_printf(pred_buf, pred_len, pred_pos, alloc, "%.*s",
                                     rowkey_pred.length(), rowkey_pred.ptr()))) {
    LOG_WARN("copy rowkey predicate failed", K(ret));
  } else if (OB_FAIL(sql.assign_fmt(
                 "update %s set `%.*s` = '%s', `%.*s` = '%s', `%.*s` = NULL, `%.*s` = '' where %s",
                 name_buf,
                 item.state_column_name_.length(), item.state_column_name_.ptr(),
                 OB_AI_STATE_PENDING,
                 item.request_column_name_.length(), item.request_column_name_.ptr(),
                 request_id,
                 item.result_column_name_.length(), item.result_column_name_.ptr(),
                 item.result_spec_column_name_.length(), item.result_spec_column_name_.ptr(),
                 pred_buf))) {
    LOG_WARN("build ai invalidate sql failed", K(ret));
  } else if (OB_FAIL(op.execute_write(sql.ptr()))) {
    LOG_WARN("execute ai invalidate sql failed", K(ret), K(sql.ptr()));
  } else {
    LOG_TRACE("ai column invalidate row", K(sql.ptr()));
  }
  return ret;
}

// delete the task rows of one (rowkey_predicate hash, column)
int emit_delete_task_stmt(ObTableModifyOp &op,
                          const uint64_t table_id,
                          const uint64_t rowkey_hash,
                          const bool with_column,
                          const uint64_t column_id,
                          ObIAllocator &alloc)
{
  int ret = OB_SUCCESS;
  char *name_buf = nullptr;
  int64_t name_len = 0;
  int64_t name_pos = 0;
  ObSqlString sql;
  if (OB_FAIL(gen_task_table_name(alloc, name_buf, name_len, name_pos))) {
    LOG_WARN("gen task table name failed", K(ret));
  } else if (with_column) {
    if (OB_FAIL(sql.assign_fmt("delete from %s where table_id = %lu and rowkey_hash = %lu"
                               " and column_id = %lu",
                               name_buf, table_id, rowkey_hash, column_id))) {
      LOG_WARN("build ai task delete sql failed", K(ret));
    }
  } else if (OB_FAIL(sql.assign_fmt("delete from %s where table_id = %lu and rowkey_hash = %lu",
                                    name_buf, table_id, rowkey_hash))) {
    LOG_WARN("build ai task delete sql failed", K(ret));
  }
  if (OB_SUCC(ret)) {
    if (OB_FAIL(op.execute_write(sql.ptr()))) {
      LOG_WARN("execute ai task delete sql failed", K(ret), K(sql.ptr()));
    } else {
      LOG_TRACE("ai column task deleted", K(sql.ptr()));
    }
  }
  return ret;
}

// insert one pending task row of one (row, AI column)
int emit_insert_task_stmt(ObTableModifyOp &op,
                          const ObAIColumnItem &item,
                          const uint64_t rowkey_hash,
                          const char *request_id,
                          const ObString &rowkey_pred,
                          ObIAllocator &alloc)
{
  int ret = OB_SUCCESS;
  char *name_buf = nullptr;
  int64_t name_len = 0;
  int64_t name_pos = 0;
  char *pred_buf = nullptr;
  int64_t pred_len = 0;
  int64_t pred_pos = 0;
  ObDMLSqlSplicer splicer;
  ObSqlString sql;
  if (OB_FAIL(gen_task_table_name(alloc, name_buf, name_len, name_pos))) {
    LOG_WARN("gen task table name failed", K(ret));
  } else if (OB_FAIL(databuff_printf(pred_buf, pred_len, pred_pos, alloc, "%.*s",
                                     rowkey_pred.length(), rowkey_pred.ptr()))) {
    LOG_WARN("copy rowkey predicate failed", K(ret));
  } else if (OB_FAIL(splicer.add_pk_column("table_id", item.table_id_))) {
    LOG_WARN("add table_id failed", K(ret));
  } else if (OB_FAIL(splicer.add_pk_column("rowkey_hash", rowkey_hash))) {
    LOG_WARN("add rowkey_hash failed", K(ret));
  } else if (OB_FAIL(splicer.add_pk_column("column_id", item.column_id_))) {
    LOG_WARN("add column_id failed", K(ret));
  } else if (OB_FAIL(splicer.add_column("rowkey_predicate",
                                        ObHexEscapeSqlStr(ObString(pred_pos, pred_buf),
                                                          false/*skip_escape*/,
                                                          true/*do_single_quote_escape*/)))) {
    LOG_WARN("add rowkey_predicate failed", K(ret));
  } else if (OB_FAIL(splicer.add_column("request_id", request_id))) {
    LOG_WARN("add request_id failed", K(ret));
  } else if (OB_FAIL(splicer.add_column("state",
                                        static_cast<int64_t>(OB_AI_TASK_STATE_PENDING)))) {
    LOG_WARN("add state failed", K(ret));
  } else if (OB_FAIL(splicer.add_column("retry_count", static_cast<int64_t>(0)))) {
    LOG_WARN("add retry_count failed", K(ret));
  } else if (OB_FAIL(splicer.add_column("max_retries", item.max_retries_))) {
    LOG_WARN("add max_retries failed", K(ret));
  } else if (OB_FAIL(splicer.add_column("def_version", item.def_version_))) {
    LOG_WARN("add def_version failed", K(ret));
  } else if (OB_FAIL(splicer.add_function_call("eligible_ts", "now(6)"))) {
    LOG_WARN("add eligible_ts failed", K(ret));
  } else if (OB_FAIL(splicer.splice_insert_sql(name_buf, sql))) {
    LOG_WARN("splice insert task sql failed", K(ret));
  } else if (OB_FAIL(op.execute_write(sql.ptr()))) {
    LOG_WARN("execute ai task insert sql failed", K(ret), K(sql.ptr()));
  } else {
    LOG_TRACE("ai column task inserted", K(sql.ptr()));
  }
  return ret;
}

uint64_t hash_rowkey_predicate(const ObString &pred)
{
  return murmurhash64A(pred.ptr(), static_cast<int32_t>(pred.length()), 0);
}

// process one inserted row
int process_insert_row(ObTableModifyOp &op,
                       const ObAITableCtx &table_ctx,
                       ObDMLModifyRowNode &row,
                       ObIAllocator &alloc,
                       const ObObjPrintParams &print_params)
{
  int ret = OB_SUCCESS;
  const ObDMLBaseCtDef *dml_ctdef = row.dml_ctdef_;
  ObEvalCtx &eval_ctx = op.get_eval_ctx();
  char *pred_buf = nullptr;
  int64_t pred_len = 0;
  int64_t pred_pos = 0;
  char rid_buf[64];
  if (OB_ISNULL(row.new_row_)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("insert modify row without new row", K(ret));
  } else if (OB_FAIL(row.new_row_->to_expr(dml_ctdef->new_row_, eval_ctx))) {
    LOG_WARN("restore insert new row failed", K(ret));
  } else if (OB_FAIL(gen_rowkey_predicate(eval_ctx, dml_ctdef->column_ids_, dml_ctdef->new_row_,
                                          table_ctx.rowkey_cids_, table_ctx.rowkey_names_,
                                          alloc, print_params, pred_buf, pred_len, pred_pos))) {
    LOG_WARN("gen insert rowkey predicate failed", K(ret));
  } else if (OB_FAIL(gen_ai_request_id(rid_buf, sizeof(rid_buf)))) {
    LOG_WARN("gen ai request id failed", K(ret));
  } else {
    const ObString rowkey_pred(pred_pos, pred_buf);
    const uint64_t rowkey_hash = hash_rowkey_predicate(rowkey_pred);
    for (int64_t i = 0; OB_SUCC(ret) && i < table_ctx.items_.count(); ++i) {
      const ObAIColumnItem &item = table_ctx.items_.at(i);
      // the insert already wrote the row with hidden defaults, mark the identity
      if (OB_FAIL(emit_invalidate_stmt(op, table_ctx, item, rid_buf, rowkey_pred,
                                       alloc))) {
        LOG_WARN("invalidate inserted row failed", K(ret), K(i));
      } else if (OB_FAIL(emit_delete_task_stmt(op, table_ctx.table_id_, rowkey_hash,
                                               true/*with_column*/, item.column_id_,
                                               alloc))) {
        LOG_WARN("supersede stale insert task failed", K(ret), K(i));
      } else if (OB_FAIL(emit_insert_task_stmt(op, item, rowkey_hash, rid_buf, rowkey_pred,
                                               alloc))) {
        LOG_WARN("insert task failed", K(ret), K(i));
      }
    }
  }
  return ret;
}

// process one updated row
int process_update_row(ObTableModifyOp &op,
                       const ObAITableCtx &table_ctx,
                       ObDMLModifyRowNode &row,
                       ObIAllocator &alloc,
                       const ObObjPrintParams &print_params)
{
  int ret = OB_SUCCESS;
  const ObDMLBaseCtDef *dml_ctdef = row.dml_ctdef_;
  ObEvalCtx &eval_ctx = op.get_eval_ctx();
  const ObUpdCtDef *upd_ctdef = nullptr;
  char *old_pred_buf = nullptr;
  int64_t old_pred_len = 0;
  int64_t old_pred_pos = 0;
  char *new_pred_buf = nullptr;
  int64_t new_pred_len = 0;
  int64_t new_pred_pos = 0;
  bool row_changed = false;
  if (OB_ISNULL(upd_ctdef = static_cast<const ObUpdCtDef *>(dml_ctdef))) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("update modify row without upd ctdef", K(ret));
  } else if (OB_NOT_NULL(row.full_row_)) {
    // in-place update (also the default path of a primary key update): the
    // stored full row holds the old values followed by the assigned values,
    // so both the old and the new images are available after the restore.
    if (OB_FAIL(row.full_row_->to_expr(dml_ctdef->full_row_, eval_ctx))) {
      LOG_WARN("restore update full row failed", K(ret));
    } else {
      row_changed = true;
    }
  } else if (OB_NOT_NULL(row.old_row_) && OB_NOT_NULL(row.new_row_)
             && row.old_row_ != row.new_row_) {
    // primary key update split into delete + insert
    if (OB_FAIL(row.old_row_->to_expr(dml_ctdef->old_row_, eval_ctx))) {
      LOG_WARN("restore update old row failed", K(ret));
    } else if (OB_FAIL(row.new_row_->to_expr(dml_ctdef->new_row_, eval_ctx))) {
      LOG_WARN("restore update new row failed", K(ret));
    } else {
      row_changed = true;
    }
  } else if (OB_NOT_NULL(row.old_row_) && OB_NOT_NULL(row.new_row_)
             && row.old_row_ == row.new_row_) {
    // old row equals new row, only the row lock was taken: keep the result
  } else {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("update modify row without row image", K(ret));
  }
  if (OB_SUCC(ret) && row_changed) {
    if (OB_FAIL(gen_rowkey_predicate(eval_ctx, dml_ctdef->column_ids_, dml_ctdef->old_row_,
                                     table_ctx.rowkey_cids_, table_ctx.rowkey_names_,
                                     alloc, print_params, old_pred_buf,
                                     old_pred_len, old_pred_pos))) {
      LOG_WARN("gen update old rowkey predicate failed", K(ret));
    } else if (OB_FAIL(gen_rowkey_predicate(eval_ctx, dml_ctdef->column_ids_, dml_ctdef->new_row_,
                                            table_ctx.rowkey_cids_, table_ctx.rowkey_names_,
                                            alloc, print_params, new_pred_buf,
                                            new_pred_len, new_pred_pos))) {
      LOG_WARN("gen update new rowkey predicate failed", K(ret));
    } else {
      const ObString old_pred(old_pred_pos, old_pred_buf);
      const ObString new_pred(new_pred_pos, new_pred_buf);
      const bool pk_changed = (old_pred != new_pred);
      const UIntFixedArray &updated_cids = upd_ctdef->dupd_ctdef_.updated_column_ids_;
      // one request id per (row, statement change), shared by all AI columns
      char rid_buf[64];
      if (!pk_changed) {
        bool has_dep_updated = false;
        for (int64_t i = 0; OB_SUCC(ret) && !has_dep_updated && i < table_ctx.items_.count(); ++i) {
          const ObAIColumnItem &item = table_ctx.items_.at(i);
          for (int64_t j = 0; !has_dep_updated && j < item.dep_column_ids_.count(); ++j) {
            const uint64_t dep_cid = item.dep_column_ids_.at(j);
            for (int64_t k = 0; !has_dep_updated && k < updated_cids.count(); ++k) {
              if (updated_cids.at(k) == dep_cid) {
                has_dep_updated = true;
              }
            }
          }
        }
        if (OB_FAIL(ret)) {
        } else if (!has_dep_updated) {
          // unrelated update, keep the AI result and create no task
        } else if (OB_FAIL(gen_ai_request_id(rid_buf, sizeof(rid_buf)))) {
          LOG_WARN("gen ai request id failed", K(ret));
        } else {
          for (int64_t i = 0; OB_SUCC(ret) && i < table_ctx.items_.count(); ++i) {
            const ObAIColumnItem &item = table_ctx.items_.at(i);
            bool item_dep_updated = false;
            for (int64_t j = 0; !item_dep_updated && j < item.dep_column_ids_.count(); ++j) {
              for (int64_t k = 0; !item_dep_updated && k < updated_cids.count(); ++k) {
                if (updated_cids.at(k) == item.dep_column_ids_.at(j)) {
                  item_dep_updated = true;
                }
              }
            }
            if (OB_FAIL(ret)) {
            } else if (!item_dep_updated) {
            } else if (OB_FAIL(emit_invalidate_stmt(op, table_ctx, item, rid_buf, new_pred,
                                                     alloc))) {
              LOG_WARN("invalidate updated row failed", K(ret), K(i));
            } else if (OB_FAIL(emit_delete_task_stmt(op, table_ctx.table_id_,
                                                     hash_rowkey_predicate(new_pred),
                                                     true/*with_column*/, item.column_id_,
                                                     alloc))) {
              LOG_WARN("supersede stale updated task failed", K(ret), K(i));
            } else if (OB_FAIL(emit_insert_task_stmt(op, item, hash_rowkey_predicate(new_pred),
                                                     rid_buf, new_pred, alloc))) {
              LOG_WARN("insert updated task failed", K(ret), K(i));
            }
          }
        }
      } else {
        // primary key changed: all AI columns of the row must regenerate and
        // the tasks of the old rowkey must be removed
        const uint64_t old_hash = hash_rowkey_predicate(old_pred);
        const uint64_t new_hash = hash_rowkey_predicate(new_pred);
        if (OB_FAIL(gen_ai_request_id(rid_buf, sizeof(rid_buf)))) {
          LOG_WARN("gen ai request id failed", K(ret));
        } else if (OB_FAIL(emit_delete_task_stmt(op, table_ctx.table_id_, old_hash,
                                                 false/*with_column*/, 0, alloc))) {
          LOG_WARN("remove old primary key tasks failed", K(ret));
        } else {
          for (int64_t i = 0; OB_SUCC(ret) && i < table_ctx.items_.count(); ++i) {
            const ObAIColumnItem &item = table_ctx.items_.at(i);
            if (OB_FAIL(emit_invalidate_stmt(op, table_ctx, item, rid_buf, new_pred,
                                             alloc))) {
              LOG_WARN("invalidate pk updated row failed", K(ret), K(i));
            } else if (OB_FAIL(emit_delete_task_stmt(op, table_ctx.table_id_, new_hash,
                                                     true/*with_column*/, item.column_id_,
                                                     alloc))) {
              LOG_WARN("supersede stale pk updated task failed", K(ret), K(i));
            } else if (OB_FAIL(emit_insert_task_stmt(op, item, new_hash, rid_buf, new_pred,
                                                     alloc))) {
              LOG_WARN("insert pk updated task failed", K(ret), K(i));
            }
          }
        }
      }
    }
  }
  return ret;
}

// process one deleted row
int process_delete_row(ObTableModifyOp &op,
                       const ObAITableCtx &table_ctx,
                       ObDMLModifyRowNode &row,
                       ObIAllocator &alloc,
                       const ObObjPrintParams &print_params)
{
  int ret = OB_SUCCESS;
  const ObDMLBaseCtDef *dml_ctdef = row.dml_ctdef_;
  ObEvalCtx &eval_ctx = op.get_eval_ctx();
  char *pred_buf = nullptr;
  int64_t pred_len = 0;
  int64_t pred_pos = 0;
  if (OB_ISNULL(row.old_row_)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("delete modify row without old row", K(ret));
  } else if (OB_FAIL(row.old_row_->to_expr(dml_ctdef->old_row_, eval_ctx))) {
    LOG_WARN("restore delete old row failed", K(ret));
  } else if (OB_FAIL(gen_rowkey_predicate(eval_ctx, dml_ctdef->column_ids_, dml_ctdef->old_row_,
                                          table_ctx.rowkey_cids_, table_ctx.rowkey_names_,
                                          alloc, print_params, pred_buf, pred_len, pred_pos))) {
    LOG_WARN("gen delete rowkey predicate failed", K(ret));
  } else {
    const ObString rowkey_pred(pred_pos, pred_buf);
    const uint64_t rowkey_hash = hash_rowkey_predicate(rowkey_pred);
    // the row is already gone, only the pending tasks must be removed
    if (OB_FAIL(emit_delete_task_stmt(op, table_ctx.table_id_, rowkey_hash,
                                      false/*with_column*/, 0, alloc))) {
      LOG_WARN("remove deleted row tasks failed", K(ret));
    }
  }
  return ret;
}

}  // anonymous namespace

int ObAIColumnDMLHelper::check_table_has_ai_columns(ObTableModifyOp &op, bool &has_ai_columns)
{
  int ret = OB_SUCCESS;
  has_ai_columns = false;
  const ObDMLBaseCtDef *dml_ctdef = nullptr;
  int tmp_ret = op.get_spec().get_single_dml_ctdef(dml_ctdef);
  if (OB_SUCCESS != tmp_ret || OB_ISNULL(dml_ctdef)) {
    // multi table DML or the spec does not support the interface: no AI
    // linkage here, and this is not an error for ordinary DML.
  } else if (!dml_ctdef->is_primary_index_ || dml_ctdef->is_table_without_pk_) {
    // only the primary table ctdef of a table with primary key carries the
    // AI result columns
  } else {
    const uint64_t table_id = dml_ctdef->das_base_ctdef_.table_id_;
    schema::ObSchemaGetterGuard *guard = nullptr;
    int64_t schema_version = 0;
    ObArenaAllocator alloc(ObModIds::OB_MODULE_PAGE_ALLOCATOR,
                           OB_MALLOC_NORMAL_BLOCK_SIZE);
    ObAITableCtx table_ctx;
    if (OB_FAIL(get_exec_schema_guard(op, guard))) {
      // keep DML working even without a schema guard, just no AI linkage
      ret = OB_SUCCESS;
      LOG_WARN("no schema guard for ai column check", K(table_id));
    } else if (OB_FAIL(guard->get_schema_version(schema_version))) {
      ret = OB_SUCCESS;
      LOG_WARN("get schema version for ai column check failed",
               K(ret), K(table_id));
    } else if (OB_FAIL(load_ai_table_ctx(*guard, table_id, schema_version, alloc, table_ctx))) {
      LOG_WARN("load ai table ctx failed", K(ret), K(table_id), K(schema_version));
    } else {
      has_ai_columns = table_ctx.found_;
    }
  }
  return ret;
}

int ObAIColumnDMLHelper::process_after_rows(ObTableModifyOp &op, ObDMLModifyRowsList &rows)
{
  int ret = OB_SUCCESS;
  if (rows.empty() || !op.has_ai_columns_) {
    // hot path: the table has no AI column (or nothing was modified)
  } else if (ObSQLUtils::is_nested_sql(&op.get_exec_ctx())) {
    // nested SQL (foreign key cascade, trigger, PL, or the maintenance SQL of
    // this helper itself) must not re-enter the AI linkage
  } else {
    ObArenaAllocator alloc(ObModIds::OB_MODULE_PAGE_ALLOCATOR,
                           OB_MALLOC_NORMAL_BLOCK_SIZE);
    const ObObjPrintParams &print_params = op.get_obj_print_params();
    ObAITableCtx table_ctx;
    ObDMLModifyRowsList::iterator row_iter = rows.begin();
    // resolve the AI metadata once from the first primary table row
    for (; OB_SUCC(ret) && !table_ctx.found_ && row_iter != rows.end(); row_iter++) {
      ObDMLModifyRowNode &row = *row_iter;
      if (OB_ISNULL(row.dml_ctdef_)) {
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("dml ctdef of modify row is null", K(ret));
      } else if (!row.dml_ctdef_->is_primary_index_) {
        // index maintain rows do not carry AI result columns
      } else {
        schema::ObSchemaGetterGuard *guard = nullptr;
        int64_t schema_version = 0;
        if (OB_FAIL(get_exec_schema_guard(op, guard))) {
          LOG_WARN("get schema guard failed", K(ret));
        } else if (OB_FAIL(guard->get_schema_version(schema_version))) {
          LOG_WARN("get schema version failed", K(ret));
        } else if (OB_FAIL(load_ai_table_ctx(*guard, row.dml_ctdef_->das_base_ctdef_.table_id_,
                                             schema_version, alloc, table_ctx))) {
          LOG_WARN("load ai table ctx failed", K(ret));
        }
      }
    }
    if (OB_SUCC(ret) && table_ctx.found_) {
      // same transaction, same pattern as the foreign key cascade
      const bool save_is_fk_cascading = op.get_exec_ctx().get_das_ctx().is_fk_cascading_;
      const bool save_in_ignore_cascading = op.get_exec_ctx().get_das_ctx().in_ignore_cascading_;
      op.get_exec_ctx().get_das_ctx().is_fk_cascading_ = true;
      op.get_exec_ctx().get_das_ctx().in_ignore_cascading_ = false;
      if (OB_FAIL(op.check_stack())) {
        LOG_WARN("check stack for ai column process failed", K(ret));
      } else if (OB_FAIL(op.open_inner_conn())) {
        LOG_WARN("open inner conn for ai column process failed", K(ret));
      } else if (OB_FAIL(op.begin_nested_session(true/*skip_cur_stmt_tables*/))) {
        LOG_WARN("begin nested session for ai column process failed", K(ret));
      } else {
        row_iter = rows.begin();
        for (; OB_SUCC(ret) && row_iter != rows.end(); row_iter++) {
          ObDMLModifyRowNode &row = *row_iter;
          if (OB_ISNULL(row.dml_ctdef_) || OB_ISNULL(row.dml_op_)) {
            ret = OB_ERR_UNEXPECTED;
            LOG_WARN("modify row is invalid", K(ret));
          } else if (!row.dml_ctdef_->is_primary_index_) {
            // skip index maintain rows
          } else if (ObDmlEventType::DE_INSERTING == row.dml_event_) {
            if (OB_FAIL(process_insert_row(op, table_ctx, row, alloc, print_params))) {
              LOG_WARN("process ai insert row failed", K(ret));
            }
          } else if (ObDmlEventType::DE_UPDATING == row.dml_event_) {
            if (OB_FAIL(process_update_row(op, table_ctx, row, alloc, print_params))) {
              LOG_WARN("process ai update row failed", K(ret));
            }
          } else if (ObDmlEventType::DE_DELETING == row.dml_event_) {
            if (OB_FAIL(process_delete_row(op, table_ctx, row, alloc, print_params))) {
              LOG_WARN("process ai delete row failed", K(ret));
            }
          } else {
            // other events (e.g. lock) are not AI relevant
          }
        }
        int end_ret = op.end_nested_session();
        if (OB_SUCCESS != end_ret) {
          LOG_WARN("failed to end nested session of ai column process", K(end_ret));
          if (OB_SUCCESS == ret) {
            ret = end_ret;
          }
        }
      }
      op.get_exec_ctx().get_das_ctx().is_fk_cascading_ = save_is_fk_cascading;
      op.get_exec_ctx().get_das_ctx().in_ignore_cascading_ = save_in_ignore_cascading;
    }
  }
  return ret;
}

}  // namespace sql
}  // namespace oceanbase
