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

#define USING_LOG_PREFIX SHARE

#include "share/ai_column/ob_ai_column_util.h"

#include "lib/allocator/page_arena.h"
#include "lib/mysqlclient/ob_mysql_proxy.h"
#include "lib/atomic/ob_atomic.h"
#include "lib/mysqlclient/ob_mysql_result.h"
#include "lib/profile/ob_trace_id.h"
#include "lib/random/ob_random.h"
#include "lib/time/ob_time_utility.h"
#include "lib/utility/ob_utility.h"
#include "share/inner_table/ob_inner_table_schema.h"
#include "share/ob_server_struct.h"
#include "share/schema/ob_multi_version_schema_service.h"
#include "share/schema/ob_schema_getter_guard.h"
#include "share/schema/ob_table_schema.h"

namespace oceanbase
{
namespace share
{

static const char *OB_AI_STATE_COLUMN_PREFIX = "__ai_state_";
static const char *OB_AI_REQUEST_COLUMN_PREFIX = "__ai_request_";
static const char *OB_AI_RESULT_SPEC_COLUMN_PREFIX = "__ai_result_spec_";

ObAiColumnCache &ObAiColumnCache::get_instance()
{
  static ObAiColumnCache instance_;
  return instance_;
}

int ObAiColumnCache::init()
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(inited_)) {
  } else if (OB_ISNULL(cache_alloc_ = static_cast<common::ObArenaAllocator *>(
      common::ob_malloc(sizeof(common::ObArenaAllocator), ObModIds::OB_SCHEMA_GETTER_GUARD)))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
    LOG_WARN("allocate cache allocator failed", K(ret));
  } else {
    new (cache_alloc_) common::ObArenaAllocator(ObModIds::OB_SCHEMA_GETTER_GUARD);
    if (OB_FAIL(table_items_.create(1024, ObModIds::OB_SCHEMA_GETTER_GUARD))) {
      LOG_WARN("create table item map failed", K(ret));
    } else {
      inited_ = true;
    }
  }
  return ret;
}

void ObAiColumnCache::destroy()
{
  if (inited_) {
    common::SpinWLockGuard guard(lock_);
    TableItemMap::iterator iter = table_items_.begin();
    for (; iter != table_items_.end(); ++iter) {
      ItemList *list = iter->second;
      if (nullptr != list) {
        list->~ObArray<ObAIColumnItem>();
        common::ob_free(list);
      }
      iter->second = nullptr;
    }
    table_items_.destroy();
    if (nullptr != cache_alloc_) {
      cache_alloc_->~ObArenaAllocator();
      common::ob_free(cache_alloc_);
      cache_alloc_ = nullptr;
    }
    inited_ = false;
  }
}

int ObAiColumnCache::get_table_items(const uint64_t table_id, const int64_t schema_version,
                                     common::ObIAllocator &alloc,
                                     common::ObArray<ObAIColumnItem> &items,
                                     bool &found)
{
  int ret = OB_SUCCESS;
  found = false;
  bool need_load = true;
  {
    common::SpinRLockGuard guard(lock_);
    TableItemMap::const_iterator iter = table_items_.get(table_id);
    if (table_items_.end() != iter) {
      need_load = false;
      const ItemList &list = *iter->second;
      if (OB_UNLIKELY(list.empty()) || OB_UNLIKELY(list.at(0).schema_version_ != schema_version)) {
        need_load = true;
      }
    }
  }
  if (need_load) {
    if (OB_FAIL(load_table_items(table_id, schema_version))) {
      if (OB_ENTRY_NOT_EXIST == ret) {
        // no AI columns for this table; cache the empty entry of this version
        common::SpinWLockGuard guard(lock_);
        ItemList *empty_list = static_cast<ItemList *>(
            cache_alloc_->alloc(sizeof(ItemList)));
        if (OB_ISNULL(empty_list)) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
        } else {
          empty_list = new (empty_list) ItemList();
          empty_list->set_allocator(cache_alloc_);
          TableItemMap::iterator iter = table_items_.get(table_id);
          if (table_items_.end() != iter) {
            iter->second->~ObArray<ObAIColumnItem>();
            cache_alloc_->free(iter->second);
            table_items_.erase(iter);
          }
          if (OB_FAIL(table_items_.put(table_id, empty_list))) {
            LOG_WARN("put empty list failed", K(ret));
          } else {
            ret = OB_SUCCESS;
          }
        }
      } else {
        LOG_WARN("load table ai column items failed", K(ret), K(table_id), K(schema_version));
      }
    }
  }
  if (OB_SUCC(ret)) {
    common::SpinRLockGuard guard(lock_);
    TableItemMap::const_iterator iter = table_items_.get(table_id);
    if (table_items_.end() != iter) {
      const ItemList &list = *iter->second;
      if (OB_LIKELY(!list.empty()) && OB_LIKELY(list.at(0).schema_version_ == schema_version)) {
        found = true;
        for (int64_t i = 0; OB_SUCC(ret) && i < list.count(); ++i) {
          ObAIColumnItem dst;
          if (OB_FAIL(deep_copy_item(alloc, list.at(i), dst))) {
            LOG_WARN("deep copy item failed", K(ret));
          } else if (OB_FAIL(items.push_back(dst))) {
            LOG_WARN("push back item failed", K(ret));
          }
        }
      }
    }
  }
  return ret;
}

void ObAiColumnCache::drop_table_items(const uint64_t table_id)
{
  common::SpinWLockGuard guard(lock_);
  TableItemMap::iterator iter = table_items_.get(table_id);
  if (table_items_.end() != iter) {
    iter->second->~ObArray<ObAIColumnItem>();
    cache_alloc_->free(iter->second);
    iter->second = nullptr;
    table_items_.erase(iter);
  }
}

int ObAiColumnCache::load_table_items(const uint64_t table_id, const int64_t schema_version)
{
  int ret = OB_SUCCESS;
  const ObTableSchema *table_schema = nullptr;
  ObSchemaGetterGuard schema_guard;
  int64_t cur_version = 0;
  bool table_exists = true;
  if (OB_ISNULL(GCTX.schema_service_) || OB_ISNULL(GCTX.sql_proxy_)) {
    ret = OB_ENTRY_NOT_EXIST;  // observer not ready yet (e.g. during bootstrap)
  } else if (OB_FAIL(GCTX.schema_service_->get_runtime_schema_guard(schema_guard))) {
    LOG_WARN("get runtime schema guard failed", K(ret));
  } else if (OB_FAIL(schema_guard.get_schema_version(cur_version))) {
    LOG_WARN("get schema version failed", K(ret));
  } else if (OB_FAIL(schema_guard.get_table_schema(table_id, table_schema))) {
    if (OB_TABLE_NOT_EXIST == ret) {
      ret = OB_SUCCESS;
      table_exists = false;
    } else {
      LOG_WARN("get table schema failed", K(ret), K(table_id));
    }
  } else if (OB_ISNULL(table_schema)) {
    table_exists = false;
  }
  if (OB_SUCC(ret) && !table_exists) {
    // the table is gone; refresh the cache entry to empty
    common::SpinWLockGuard guard(lock_);
    TableItemMap::iterator iter = table_items_.get(table_id);
    if (table_items_.end() != iter) {
      iter->second->~ObArray<ObAIColumnItem>();
      cache_alloc_->free(iter->second);
      iter->second = nullptr;
      table_items_.erase(iter);
    }
    ItemList *empty_list = static_cast<ItemList *>(cache_alloc_->alloc(sizeof(ItemList)));
    if (OB_ISNULL(empty_list)) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else {
      empty_list = new (empty_list) ItemList();
      empty_list->set_allocator(cache_alloc_);
      if (OB_FAIL(table_items_.put(table_id, empty_list))) {
        LOG_WARN("put empty list failed", K(ret));
      }
    }
  } else if (OB_SUCC(ret) && table_exists) {
    ItemList *new_list = static_cast<ItemList *>(cache_alloc_->alloc(sizeof(ItemList)));
    if (OB_ISNULL(new_list)) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else {
      new_list = new (new_list) ItemList();
      new_list->set_allocator(cache_alloc_);
      // read the AI columns of this table from the system table
      common::ObSqlString sql;
      SMART_VAR(common::ObMySQLProxy::MySQLResult, res) {
        common::ObMySQLResult *result = nullptr;
        if (OB_FAIL(sql.assign_fmt("SELECT column_id, column_name, func_type, model_name, sync_mode, "
                                   "max_retries, def_version, user_expr, normalized_expr, dep_column_ids, "
                                   "result_type, vector_dim, state_column_id, request_column_id, "
                                   "result_spec_column_id FROM %s WHERE table_id = %lu ORDER BY column_id",
                                   OB_ALL_AI_COLUMN_TNAME, table_id))) {
          LOG_WARN("build sql failed", K(ret));
        } else if (OB_FAIL(GCTX.sql_proxy_->read(res, sql.ptr()))) {
          LOG_WARN("read ai columns failed", K(ret), K(sql.ptr()));
        } else if (OB_ISNULL(result = res.get_result())) {
          ret = OB_ERR_UNEXPECTED;
          LOG_WARN("no result", K(ret));
        } else {
          while (OB_SUCC(ret)) {
            const int tmp_ret = result->next();
            if (OB_ITER_END == tmp_ret) {
              break;
            } else if (OB_SUCCESS != tmp_ret) {
              ret = tmp_ret;
              LOG_WARN("iterate ai column rows failed", K(ret));
            } else {
              ObAIColumnItem item;
              item.table_id_ = table_id;
              item.schema_version_ = cur_version;
              int64_t v = 0;
              EXTRACT_INT_FIELD_MYSQL(*result, "column_id", v, int64_t);
              item.column_id_ = v;
              EXTRACT_VARCHAR_FIELD_MYSQL(*result, "column_name", item.column_name_);
              if (OB_SUCC(ret)) {
                EXTRACT_INT_FIELD_MYSQL(*result, "func_type", v, int64_t);
                item.func_type_ = v;
              }
              if (OB_SUCC(ret)) {
                EXTRACT_VARCHAR_FIELD_MYSQL(*result, "model_name", item.model_name_);
              }
              if (OB_SUCC(ret)) {
                EXTRACT_VARCHAR_FIELD_MYSQL(*result, "sync_mode", item.sync_mode_);
              }
              if (OB_SUCC(ret)) {
                EXTRACT_INT_FIELD_MYSQL(*result, "max_retries", v, int64_t);
                item.max_retries_ = v;
              }
              if (OB_SUCC(ret)) {
                EXTRACT_INT_FIELD_MYSQL(*result, "def_version", v, int64_t);
                item.def_version_ = v;
              }
              if (OB_SUCC(ret)) {
                EXTRACT_VARCHAR_FIELD_MYSQL(*result, "user_expr", item.user_expr_);
              }
              if (OB_SUCC(ret)) {
                EXTRACT_VARCHAR_FIELD_MYSQL(*result, "normalized_expr", item.normalized_expr_);
              }
              if (OB_SUCC(ret)) {
                common::ObString dep_str;
                EXTRACT_VARCHAR_FIELD_MYSQL(*result, "dep_column_ids", dep_str);
                const char *p = dep_str.ptr();
                const int64_t len = dep_str.length();
                int64_t i = 0;
                while (i < len) {
                  uint64_t dep = 0;
                  bool has_digit = false;
                  while (i < len && p[i] >= '0' && p[i] <= '9') {
                    dep = dep * 10 + (p[i] - '0');
                    ++i;
                    has_digit = true;
                  }
                  if (has_digit && OB_FAIL(item.dep_column_ids_.push_back(dep))) {
                    LOG_WARN("push back dep failed", K(ret));
                  }
                  ++i;  // skip the separator
                }
              }
              if (OB_SUCC(ret)) {
                common::ObString res_type;
                EXTRACT_VARCHAR_FIELD_MYSQL(*result, "result_type", res_type);
                if (OB_FAIL(ob_write_string(*cache_alloc_, res_type, item.result_column_type_str_))) {
                  LOG_WARN("write result type failed", K(ret));
                }
              }
              if (OB_SUCC(ret)) {
                EXTRACT_INT_FIELD_MYSQL_WITH_DEFAULT_VALUE(*result, "vector_dim", v, int64_t, true, true, 0);
                item.vector_dim_ = v;
                item.is_vector_ = (v > 0);
              }
              if (OB_SUCC(ret)) {
                EXTRACT_INT_FIELD_MYSQL(*result, "state_column_id", v, int64_t);
                item.state_column_id_ = v;
              }
              if (OB_SUCC(ret)) {
                EXTRACT_INT_FIELD_MYSQL(*result, "request_column_id", v, int64_t);
                item.request_column_id_ = v;
              }
              if (OB_SUCC(ret)) {
                EXTRACT_INT_FIELD_MYSQL(*result, "result_spec_column_id", v, int64_t);
                item.result_spec_column_id_ = v;
              }
              // derive names and the result obj type from the table schema
              if (OB_SUCC(ret)) {
                const ObColumnSchemaV2 *col = table_schema->get_column_schema(item.column_id_);
                const ObColumnSchemaV2 *state_col = table_schema->get_column_schema(item.state_column_id_);
                const ObColumnSchemaV2 *req_col = table_schema->get_column_schema(item.request_column_id_);
                const ObColumnSchemaV2 *spec_col = table_schema->get_column_schema(item.result_spec_column_id_);
                if (OB_ISNULL(col) || OB_ISNULL(state_col) || OB_ISNULL(req_col) || OB_ISNULL(spec_col)) {
                  ret = OB_ERR_UNEXPECTED;
                  LOG_WARN("ai column not found in schema", K(ret), K(item.column_id_));
                } else {
                  item.result_column_name_ = col->get_column_name_str();
                  item.state_column_name_ = state_col->get_column_name_str();
                  item.request_column_name_ = req_col->get_column_name_str();
                  item.result_spec_column_name_ = spec_col->get_column_name_str();
                  item.result_obj_type_ = col->get_meta_type().get_type();
                  for (int64_t i = 0; OB_SUCC(ret) && i < table_schema->get_rowkey_info().get_size(); ++i) {
                    uint64_t col_id = 0;
                    if (OB_FAIL(table_schema->get_rowkey_info().get_column_id(i, col_id))) {
                      LOG_WARN("get rowkey column id failed", K(ret));
                    } else {
                      const ObColumnSchemaV2 *pk_col = table_schema->get_column_schema(col_id);
                      if (OB_ISNULL(pk_col)) {
                        ret = OB_ERR_UNEXPECTED;
                        LOG_WARN("rowkey column not found", K(ret), K(col_id));
                      } else if (OB_FAIL(item.rowkey_column_names_.push_back(pk_col->get_column_name_str()))) {
                        LOG_WARN("push back rowkey name failed", K(ret));
                      }
                    }
                  }
                }
              }
              if (OB_SUCC(ret) && OB_FAIL(new_list->push_back(item))) {
                LOG_WARN("push back item failed", K(ret));
              }
            }
          }
        }
      }
      if (OB_SUCC(ret)) {
        common::SpinWLockGuard guard(lock_);
        TableItemMap::iterator iter = table_items_.get(table_id);
        if (table_items_.end() != iter) {
          iter->second->~ObArray<ObAIColumnItem>();
          cache_alloc_->free(iter->second);
          table_items_.erase(iter);
        }
        if (OB_FAIL(table_items_.put(table_id, new_list))) {
          LOG_WARN("put into cache failed", K(ret));
          new_list->~ObArray<ObAIColumnItem>();
          cache_alloc_->free(new_list);
        }
      } else {
        new_list->~ObArray<ObAIColumnItem>();
        cache_alloc_->free(new_list);
      }
    }
  }
  return ret;
}

int ObAiColumnCache::deep_copy_item(common::ObIAllocator &alloc,
                                    const ObAIColumnItem &src, ObAIColumnItem &dst)
{
  int ret = OB_SUCCESS;
  dst = src;
  dst.dep_column_ids_.reset();
  dst.rowkey_column_names_.reset();
  dst.dep_column_ids_.set_allocator(&alloc);
  dst.rowkey_column_names_.set_allocator(&alloc);
  if (OB_FAIL(common::ob_write_string(alloc, src.column_name_, dst.column_name_))) {
    LOG_WARN("copy column name failed", K(ret));
  } else if (OB_FAIL(common::ob_write_string(alloc, src.model_name_, dst.model_name_))) {
    LOG_WARN("copy model name failed", K(ret));
  } else if (OB_FAIL(common::ob_write_string(alloc, src.sync_mode_, dst.sync_mode_))) {
    LOG_WARN("copy sync mode failed", K(ret));
  } else if (OB_FAIL(common::ob_write_string(alloc, src.user_expr_, dst.user_expr_))) {
    LOG_WARN("copy user expr failed", K(ret));
  } else if (OB_FAIL(common::ob_write_string(alloc, src.normalized_expr_, dst.normalized_expr_))) {
    LOG_WARN("copy normalized expr failed", K(ret));
  } else if (OB_FAIL(common::ob_write_string(alloc, src.state_column_name_, dst.state_column_name_))) {
    LOG_WARN("copy state name failed", K(ret));
  } else if (OB_FAIL(common::ob_write_string(alloc, src.request_column_name_, dst.request_column_name_))) {
    LOG_WARN("copy request name failed", K(ret));
  } else if (OB_FAIL(common::ob_write_string(alloc, src.result_spec_column_name_, dst.result_spec_column_name_))) {
    LOG_WARN("copy spec name failed", K(ret));
  } else if (OB_FAIL(common::ob_write_string(alloc, src.result_column_name_, dst.result_column_name_))) {
    LOG_WARN("copy result column name failed", K(ret));
  } else if (OB_FAIL(common::ob_write_string(alloc, src.result_column_type_str_, dst.result_column_type_str_))) {
    LOG_WARN("copy result type failed", K(ret));
  } else {
    for (int64_t i = 0; OB_SUCC(ret) && i < src.dep_column_ids_.count(); ++i) {
      if (OB_FAIL(dst.dep_column_ids_.push_back(src.dep_column_ids_.at(i)))) {
        LOG_WARN("copy dep failed", K(ret));
      }
    }
    for (int64_t i = 0; OB_SUCC(ret) && i < src.rowkey_column_names_.count(); ++i) {
      common::ObString copied;
      if (OB_FAIL(common::ob_write_string(alloc, src.rowkey_column_names_.at(i), copied))) {
        LOG_WARN("copy rowkey name failed", K(ret));
      } else if (OB_FAIL(dst.rowkey_column_names_.push_back(copied))) {
        LOG_WARN("push back rowkey name failed", K(ret));
      }
    }
  }
  return ret;
}

int gen_ai_request_id(char *buf, const int64_t buf_len)
{
  int ret = OB_SUCCESS;
  static volatile int64_t seq = 0;
  const int64_t cur_seq = common::atomic_add(&seq, 1);
  const int64_t now_us = common::ObTimeUtility::current_time();
  const int64_t rand_val = common::ObRandom::rand(0, INT64_MAX);
  if (OB_FAIL(common::databuff_printf(buf, buf_len,
      "%016lx%016lx%016lx", now_us, cur_seq, rand_val))) {
    LOG_WARN("format request id failed", K(ret));
  }
  return ret;
}

int build_ai_hidden_column_name(const char *prefix, const int64_t prefix_len,
                                const common::ObString &column_name,
                                char *buf, const int64_t buf_len)
{
  int ret = OB_SUCCESS;
  const int64_t max_name_len = buf_len - prefix_len - 1;
  const int64_t name_len = MIN(column_name.length(), max_name_len);
  if (OB_FAIL(common::databuff_printf(buf, buf_len, "%.*s%.*s",
      static_cast<int32_t>(prefix_len), prefix,
      static_cast<int32_t>(name_len), column_name.ptr()))) {
    LOG_WARN("build hidden column name failed", K(ret));
  }
  return ret;
}

}  // end namespace share
}  // end namespace oceanbase
