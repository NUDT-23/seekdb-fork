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

#ifndef OCEANBASE_SHARE_AI_COLUMN_OB_AI_COLUMN_UTIL_H_
#define OCEANBASE_SHARE_AI_COLUMN_OB_AI_COLUMN_UTIL_H_

#include "lib/hash/ob_hashmap.h"
#include "lib/lock/ob_spin_rwlock.h"
#include "lib/string/ob_sql_string.h"
#include "lib/string/ob_string.h"

namespace oceanbase
{
namespace common
{
class ObIAllocator;
class ObArenaAllocator;
}
namespace share
{

static const int64_t OB_AI_COL_FUNC_COMPLETE = 1;
static const int64_t OB_AI_COL_FUNC_EMBED = 2;

// states of __all_ai_column_task.state
static const int64_t OB_AI_TASK_STATE_PENDING = 1;
static const int64_t OB_AI_TASK_STATE_EXECUTING = 2;
static const int64_t OB_AI_TASK_STATE_RETRY_WAIT = 3;

// per-row state column values
static const char *OB_AI_STATE_PENDING = "PENDING";
static const char *OB_AI_STATE_READY = "READY";
static const char *OB_AI_STATE_FAILED = "FAILED";

// JSON keys of the AI column definition persisted in the comment of the
// hidden state column of each AI column (see the DDL resolver)
static const char *OB_AI_DEF_FUNC = "func";
static const char *OB_AI_DEF_MODEL = "model";
static const char *OB_AI_DEF_SYNC = "sync";
static const char *OB_AI_DEF_MAX_RETRIES = "max_retries";
static const char *OB_AI_DEF_VERSION = "ver";
static const char *OB_AI_DEF_DEPS = "deps";
static const char *OB_AI_DEF_USER_EXPR = "expr";
static const char *OB_AI_DEF_NORM_EXPR = "norm";
static const char *OB_AI_DEF_RESULT_TYPE = "res_type";
static const char *OB_AI_DEF_VECTOR_DIM = "vec_dim";

// The runtime metadata of one AI column of a user table.  Everything is
// loaded from the persistent __all_ai_column system table (written by DDL)
// so it survives restarts.
struct ObAIColumnItem
{
  ObAIColumnItem()
    : table_id_(0), column_id_(0), func_type_(OB_AI_COL_FUNC_COMPLETE),
      max_retries_(0), def_version_(0),
      state_column_id_(0), request_column_id_(0), result_spec_column_id_(0),
      result_obj_type_(0), vector_dim_(0), is_vector_(false)
  {}
  uint64_t table_id_;
  uint64_t column_id_;             // the AI result column
  common::ObString column_name_;
  int64_t func_type_;              // OB_AI_COL_FUNC_*
  common::ObString model_name_;
  common::ObString sync_mode_;
  int64_t max_retries_;
  int64_t def_version_;
  common::ObString user_expr_;     // the inner prompt expression text
  common::ObString normalized_expr_;
  common::ObArray<uint64_t> dep_column_ids_;
  uint64_t state_column_id_;
  uint64_t request_column_id_;
  uint64_t result_spec_column_id_;
  // derived at load time from the table schema
  common::ObString state_column_name_;
  common::ObString request_column_name_;
  common::ObString result_spec_column_name_;
  common::ObString result_column_name_;
  common::ObArray<common::ObString> rowkey_column_names_;
  common::ObString result_column_type_str_;  // e.g. "JSON"/"BIGINT"/"VECTOR"
  int64_t result_obj_type_;        // ObObjType of the result column
  int64_t vector_dim_;             // valid when is_vector_
  bool is_vector_;
  int64_t schema_version_;         // schema version at load time

  bool is_valid() const { return table_id_ != 0 && column_id_ != 0; }
  TO_STRING_KV(K_(table_id), K_(column_id), K_(column_name), K_(func_type),
               K_(model_name), K_(max_retries), K_(def_version),
               K_(state_column_id), K_(request_column_id),
               K_(result_spec_column_id), K_(schema_version));
};

// A process wide cache of the AI columns of each table, loaded from the
// persistent __all_ai_column system table.  Readers pass the schema version
// they resolved with so that stale entries reload after DDL.
class ObAiColumnCache
{
public:
  static ObAiColumnCache &get_instance();
  int init();
  void destroy();
  // Deep copies every AI column of the table into items (allocated on
  // alloc).  found=false when the table has no AI column.
  int get_table_items(const uint64_t table_id, const int64_t schema_version,
                      common::ObIAllocator &alloc,
                      common::ObArray<ObAIColumnItem> &items,
                      bool &found);
  void drop_table_items(const uint64_t table_id);
private:
  ObAiColumnCache() : inited_(false), cache_alloc_(nullptr) {}
  ~ObAiColumnCache() {}
  int load_table_items(const uint64_t table_id, const int64_t schema_version);
  static int deep_copy_item(common::ObIAllocator &alloc,
                            const ObAIColumnItem &src, ObAIColumnItem &dst);
  typedef common::ObArray<ObAIColumnItem> ItemList;
  bool inited_;
  common::ObArenaAllocator *cache_alloc_;
  typedef common::ObHashMap<uint64_t, ItemList *> TableItemMap;
  TableItemMap table_items_;
  common::SpinRWLock lock_;
  typedef common::SpinRLockGuard RLockGuard;
  typedef common::SpinWLockGuard WLockGuard;
};

// Generates a fresh request id (48-char hex string) for one (row, statement)
// change group.
int gen_ai_request_id(char *buf, const int64_t buf_len);

// Builds "__ai_state_<column name>" style hidden column names.
int build_ai_hidden_column_name(const char *prefix, const int64_t prefix_len,
                                const common::ObString &column_name,
                                char *buf, const int64_t buf_len);

}  // end namespace share
}  // end namespace oceanbase

#endif  // OCEANBASE_SHARE_AI_COLUMN_OB_AI_COLUMN_UTIL_H_
