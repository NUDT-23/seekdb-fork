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

#define USING_LOG_PREFIX OBSERVER

#include "observer/ai_column/ob_ai_column_worker.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <netdb.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "common/json_type/ob_json_parse.h"
#include "common/mysqlclient/ob_mysql_result.h"
#include "common/mysqlclient/ob_mysql_transaction.h"
#include "common/ob_timeout_ctx.h"
#include "lib/allocator/ob_malloc.h"
#include "lib/container/ob_array.h"
#include "lib/container/ob_se_array.h"
#include "lib/mysqlclient/ob_mysql_proxy.h"
#include "lib/time/ob_time_utility.h"
#include "lib/utility/ob_mod_define.h"
#include "lib/utility/ob_utility.h"
#include "query/ai/ob_ai_endpoint_resolver.h"
#include "query/engine/expr/ob_ai_model_resolver.h"
#include "share/ai_service/ob_ai_service_struct.h"
#include "share/inner_table/ob_inner_table_schema.h"
#include "share/ob_server_struct.h"
#include "share/rc/ob_server_runtime.h"
#include "share/schema/ob_schema_getter_guard.h"
#include "share/schema/ob_table_schema.h"
#include "sql/engine/expr/ob_expr_ai/ob_ai_func_client.h"
#include "sql/engine/expr/ob_expr_ai/ob_ai_func_utils.h"

namespace oceanbase
{
namespace share
{

// worker cadence and limits
static const int64_t OB_AI_WORKER_SCHEDULE_INTERVAL_US = 200 * 1000;  // 200ms
static const int64_t OB_AI_WORKER_TIMEOUT_US = 60L * 1000L * 1000L;   // 60s: model calls take seconds
static const int64_t OB_AI_WORKER_MAX_POLL_TASKS = 64;
static const int64_t OB_AI_WORKER_MAX_GROUPS_PER_TICK = 2;
static const int64_t OB_AI_WORKER_RETRY_INTERVAL_US = 500L * 1000L;   // 0.5s retry wait
static const int64_t OB_AI_WORKER_PROBE_TIMEOUT_MS = 500;
static const int64_t OB_AI_WORKER_TIMEOUT_SEC = 60;

// typed system prompts: identical to the texts the DDL resolver embeds into
// the normalized expression of typed AI columns (ob_ai_column_resolver.cpp)
static const char *OB_AI_WORKER_PROMPT_JSON =
    "Return exactly one valid JSON value matching the result column type. ";
static const char *OB_AI_WORKER_PROMPT_INTEGER =
    "Return exactly one base-10 integer matching the result column type. ";
static const char *OB_AI_WORKER_PROMPT_NUMBER =
    "Return exactly one base-10 number matching the result column type. ";
static const char *OB_AI_WORKER_PROMPT_DATE =
    "Return exactly one date in YYYY-MM-DD format. ";
static const char *OB_AI_WORKER_PROMPT_DATETIME =
    "Return exactly one datetime in YYYY-MM-DD HH:MM:SS format. ";

static const char *get_worker_system_prompt(const common::ObString &type_str)
{
  const char *prompt = nullptr;
  if (0 == type_str.case_compare("JSON")) {
    prompt = OB_AI_WORKER_PROMPT_JSON;
  } else if (0 == type_str.case_compare("TINYINT")
             || 0 == type_str.case_compare("SMALLINT")
             || 0 == type_str.case_compare("MEDIUMINT")
             || 0 == type_str.case_compare("INT")
             || 0 == type_str.case_compare("BIGINT")
             || 0 == type_str.case_compare("BOOLEAN")) {
    prompt = OB_AI_WORKER_PROMPT_INTEGER;
  } else if (0 == type_str.case_compare("FLOAT")
             || 0 == type_str.case_compare("DOUBLE")) {
    prompt = OB_AI_WORKER_PROMPT_NUMBER;
  } else if (0 == type_str.case_compare("DATE")) {
    prompt = OB_AI_WORKER_PROMPT_DATE;
  } else if (0 == type_str.case_compare("DATETIME")
             || 0 == type_str.case_compare("TIMESTAMP")) {
    prompt = OB_AI_WORKER_PROMPT_DATETIME;
  } else {
    // VARCHAR/CHAR/TEXT and everything else: no system message
  }
  return prompt;
}

static bool is_integer_result_type(const common::ObString &type_str)
{
  return 0 == type_str.case_compare("TINYINT")
      || 0 == type_str.case_compare("SMALLINT")
      || 0 == type_str.case_compare("MEDIUMINT")
      || 0 == type_str.case_compare("INT")
      || 0 == type_str.case_compare("BIGINT")
      || 0 == type_str.case_compare("BOOLEAN");
}

static bool is_number_result_type(const common::ObString &type_str)
{
  return 0 == type_str.case_compare("FLOAT") || 0 == type_str.case_compare("DOUBLE");
}

static void trim_sql_whitespace(const common::ObString &in, common::ObString &out)
{
  int64_t start = 0;
  int64_t end = in.length();
  while (start < end && (' ' == in.ptr()[start] || '\t' == in.ptr()[start]
                         || '\r' == in.ptr()[start] || '\n' == in.ptr()[start])) {
    ++start;
  }
  while (end > start && (' ' == in.ptr()[end - 1] || '\t' == in.ptr()[end - 1]
                         || '\r' == in.ptr()[end - 1] || '\n' == in.ptr()[end - 1])) {
    --end;
  }
  out.assign_ptr(in.ptr() + start, static_cast<int32_t>(end - start));
}

// ============================ ObAiColumnWorkerTask ============================

ObAiColumnWorkerTask::ObAiColumnWorkerTask()
  : inited_(false), recovered_(false), mem_groups_(), mem_lock_()
{}

ObAiColumnWorkerTask::~ObAiColumnWorkerTask()
{
  destroy();
}

int ObAiColumnWorkerTask::init()
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(inited_)) {
    ret = OB_INIT_TWICE;
  } else if (OB_FAIL(ObAiColumnCache::get_instance().init())) {
    LOG_WARN("init ai column cache failed", K(ret));
  } else {
    recovered_ = false;
    inited_ = true;
  }
  return ret;
}

int ObAiColumnWorkerTask::start(common::ObTimer &timer)
{
  int ret = OB_SUCCESS;
  const bool did_repeat = true;
  if (OB_UNLIKELY(!inited_)) {
    ret = OB_NOT_INIT;
  } else if (timer.task_exist(*this)) {
    // ignore duplicate schedule
  } else if (OB_FAIL(timer.schedule(*this, OB_AI_WORKER_SCHEDULE_INTERVAL_US, did_repeat))) {
    LOG_WARN("schedule ai column worker task failed", K(ret));
  } else {
    LOG_INFO("ai column worker task scheduled", K(OB_AI_WORKER_SCHEDULE_INTERVAL_US));
  }
  return ret;
}

void ObAiColumnWorkerTask::stop(common::ObTimer &timer)
{
  if (timer.inited()) {
    int ret = OB_SUCCESS;
    if (OB_FAIL(timer.cancel_task(*this))) {
    } else if (OB_FAIL(timer.wait_task(*this))) {
    }
  }
}

void ObAiColumnWorkerTask::destroy()
{
  if (inited_) {
    {
      lib::ObMutexGuard guard(mem_lock_);
      while (!mem_groups_.is_empty()) {
        MemGroup *group = mem_groups_.remove_first();
        if (nullptr != group && group != mem_groups_.get_header()) {
          group->~MemGroup();
          common::ob_free(group);
        }
      }
    }
    inited_ = false;
    recovered_ = false;
  }
}

void ObAiColumnWorkerTask::runTimerTask()
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(!inited_)) {
    ret = OB_NOT_INIT;
  } else if (!share::server_is_write_enabled()) {
    // read-only server: nothing to do
  } else {
    // model calls may take seconds, use a large worker timeout
    ObTimeoutCtx ctx;
    if (OB_FAIL(ctx.set_timeout(OB_AI_WORKER_TIMEOUT_US))) {
      LOG_WARN("set ai column worker timeout failed", K(ret));
    } else if (OB_FAIL(run_once_())) {
      LOG_WARN("ai column worker tick failed", K(ret));
    }
  }
}

int ObAiColumnWorkerTask::run_once_()
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(!inited_)) {
    ret = OB_NOT_INIT;
  } else if (!recovered_) {
    // first tick after start: requeue tasks stuck in EXECUTING by a crash
    if (OB_FAIL(recover_executing_tasks_())) {
      LOG_WARN("recover executing ai column tasks failed", K(ret));
    } else {
      recovered_ = true;
    }
  }
  if (OB_SUCC(ret)) {
    // the tick arena backs all strings of the polled tasks and of the loaded
    // ai column items of this tick
    common::ObArenaAllocator tick_alloc(ObModIds::OMT);
    ObSEArray<PolledTask, 16> tasks;
    if (OB_FAIL(poll_tasks_(tick_alloc, tasks))) {
      LOG_WARN("poll ai column tasks failed", K(ret));
    } else if (tasks.empty()) {
      // no eligible task
    } else {
      // the poll is ordered by (table_id, request_id): group the contiguous
      // rows of the same request and process at most 2 groups per tick
      const int64_t count = tasks.count();
      int64_t idx = 0;
      int64_t processed = 0;
      while (OB_SUCC(ret) && idx < count && processed < OB_AI_WORKER_MAX_GROUPS_PER_TICK) {
        PolledTask &first = tasks.at(idx);
        int64_t end = idx + 1;
        for (; end < count; ++end) {
          PolledTask &cur = tasks.at(end);
          if (cur.table_id_ != first.table_id_ || !(cur.request_id_ == first.request_id_)) {
            break;
          }
        }
        bool skip = false;
        {
          lib::ObMutexGuard guard(mem_lock_);
          MemGroup *group = find_mem_group_(first.table_id_, first.request_id_);
          // a group that is currently owned by another tick is left alone
          // (and costs no budget of this tick)
          skip = (nullptr != group && group->processing_);
        }
        if (skip) {
          idx = end;
          continue;
        }
        ObSEArray<PolledTask *, 8> group_tasks;
        for (int64_t i = idx; OB_SUCC(ret) && i < end; ++i) {
          if (OB_FAIL(group_tasks.push_back(&tasks.at(i)))) {
            LOG_WARN("push back polled task failed", K(ret));
          }
        }
        if (OB_SUCC(ret)) {
          ++processed;
          if (OB_FAIL(process_group_(tick_alloc, group_tasks))) {
            // one broken group must not stop the worker
            LOG_WARN("process ai column group failed", K(ret), K(first.table_id_),
                     K(first.request_id_));
            ret = OB_SUCCESS;
          }
        }
        idx = end;
      }
    }
  }
  return ret;
}

int ObAiColumnWorkerTask::recover_executing_tasks_()
{
  int ret = OB_SUCCESS;
  ObSqlString sql;
  int64_t affected = 0;
  ObMySQLProxy *proxy = GCTX.sql_proxy_;
  if (OB_ISNULL(proxy)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("sql proxy is null", K(ret));
  } else if (OB_FAIL(sql.assign_fmt("UPDATE %s SET state = %ld WHERE state = %ld",
                                    OB_ALL_AI_COLUMN_TASK_TNAME,
                                    static_cast<int64_t>(OB_AI_TASK_STATE_PENDING),
                                    static_cast<int64_t>(OB_AI_TASK_STATE_EXECUTING)))) {
    LOG_WARN("build recovery sql failed", K(ret));
  } else if (OB_FAIL(proxy->write(sql.ptr(), affected))) {
    LOG_WARN("recover executing ai column tasks failed", K(ret), K(sql.ptr()));
  } else if (affected > 0) {
    LOG_INFO("recovered executing ai column tasks", K(affected));
  }
  return ret;
}

int ObAiColumnWorkerTask::poll_tasks_(common::ObIAllocator &alloc,
                                      common::ObIArray<PolledTask> &tasks)
{
  int ret = OB_SUCCESS;
  ObSqlString sql;
  ObMySQLProxy *proxy = GCTX.sql_proxy_;
  if (OB_ISNULL(proxy)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("sql proxy is null", K(ret));
  } else if (OB_FAIL(sql.assign_fmt(
      "SELECT table_id, rowkey_hash, column_id, rowkey_predicate, request_id, state, "
      "retry_count, max_retries, def_version FROM %s "
      "WHERE state IN (%ld, %ld) AND eligible_ts <= now(6) "
      "ORDER BY table_id, request_id LIMIT %ld",
      OB_ALL_AI_COLUMN_TASK_TNAME,
      static_cast<int64_t>(OB_AI_TASK_STATE_PENDING),
      static_cast<int64_t>(OB_AI_TASK_STATE_RETRY_WAIT),
      OB_AI_WORKER_MAX_POLL_TASKS))) {
    LOG_WARN("build poll sql failed", K(ret));
  } else {
    SMART_VAR(ObMySQLProxy::MySQLResult, res) {
      ObMySQLResult *result = nullptr;
      if (OB_FAIL(proxy->read(res, sql.ptr()))) {
        LOG_WARN("poll ai column tasks failed", K(ret), K(sql.ptr()));
      } else if (OB_ISNULL(result = res.get_result())) {
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("poll result is null", K(ret));
      } else {
        while (OB_SUCC(ret)) {
          const int tmp_ret = result->next();
          if (OB_ITER_END == tmp_ret) {
            break;
          } else if (OB_SUCCESS != tmp_ret) {
            ret = tmp_ret;
            LOG_WARN("iterate ai column task rows failed", K(ret));
          } else {
            PolledTask task;
            int64_t v = 0;
            EXTRACT_INT_FIELD_MYSQL(*result, "table_id", v, int64_t);
            task.table_id_ = static_cast<uint64_t>(v);
            if (OB_SUCC(ret)) {
              EXTRACT_INT_FIELD_MYSQL(*result, "rowkey_hash", v, int64_t);
              task.rowkey_hash_ = static_cast<uint64_t>(v);
            }
            if (OB_SUCC(ret)) {
              EXTRACT_INT_FIELD_MYSQL(*result, "column_id", v, int64_t);
              task.column_id_ = static_cast<uint64_t>(v);
            }
            if (OB_SUCC(ret)) {
              EXTRACT_INT_FIELD_MYSQL(*result, "state", v, int64_t);
              task.state_ = v;
            }
            if (OB_SUCC(ret)) {
              EXTRACT_INT_FIELD_MYSQL(*result, "retry_count", v, int64_t);
              task.retry_count_ = v;
            }
            if (OB_SUCC(ret)) {
              EXTRACT_INT_FIELD_MYSQL(*result, "max_retries", v, int64_t);
              task.max_retries_ = v;
            }
            if (OB_SUCC(ret)) {
              EXTRACT_INT_FIELD_MYSQL(*result, "def_version", v, int64_t);
              task.def_version_ = v;
            }
            common::ObString pred;
            common::ObString rid;
            if (OB_SUCC(ret)) {
              EXTRACT_VARCHAR_FIELD_MYSQL(*result, "rowkey_predicate", pred);
            }
            if (OB_SUCC(ret)) {
              EXTRACT_VARCHAR_FIELD_MYSQL(*result, "request_id", rid);
            }
            // deep copy into the tick allocator, the result set goes away
            if (OB_SUCC(ret)) {
              if (OB_FAIL(ob_write_string(alloc, pred, task.rowkey_predicate_, true))) {
                LOG_WARN("copy rowkey predicate failed", K(ret));
              } else if (OB_FAIL(ob_write_string(alloc, rid, task.request_id_, true))) {
                LOG_WARN("copy request id failed", K(ret));
              }
            }
            if (OB_SUCC(ret) && OB_FAIL(tasks.push_back(task))) {
              LOG_WARN("push back polled task failed", K(ret));
            }
          }
        }
      }
    }
  }
  return ret;
}

// reads the task rows the group currently owns (state EXECUTING after the
// claim); this may cover more members than the poll batch, e.g. members
// claimed by an earlier tick while another member waited for its retry
int ObAiColumnWorkerTask::load_group_tasks_(common::ObIAllocator &alloc,
                                            const uint64_t table_id,
                                            const common::ObString &request_id,
                                            common::ObIArray<PolledTask> &tasks)
{
  int ret = OB_SUCCESS;
  ObSqlString sql;
  ObMySQLProxy *proxy = GCTX.sql_proxy_;
  if (OB_ISNULL(proxy)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("sql proxy is null", K(ret));
  } else if (OB_FAIL(sql.assign_fmt(
      "SELECT rowkey_hash, column_id, rowkey_predicate, request_id, state, "
      "retry_count, max_retries, def_version FROM %s "
      "WHERE table_id = %lu AND request_id = '%.*s' AND state = %ld",
      OB_ALL_AI_COLUMN_TASK_TNAME, table_id, request_id.length(), request_id.ptr(),
      static_cast<int64_t>(OB_AI_TASK_STATE_EXECUTING)))) {
    LOG_WARN("build group task sql failed", K(ret));
  } else {
    SMART_VAR(ObMySQLProxy::MySQLResult, res) {
      ObMySQLResult *result = nullptr;
      if (OB_FAIL(proxy->read(res, sql.ptr()))) {
        LOG_WARN("load ai column group tasks failed", K(ret), K(sql.ptr()));
      } else if (OB_ISNULL(result = res.get_result())) {
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("group task result is null", K(ret));
      } else {
        while (OB_SUCC(ret)) {
          const int tmp_ret = result->next();
          if (OB_ITER_END == tmp_ret) {
            break;
          } else if (OB_SUCCESS != tmp_ret) {
            ret = tmp_ret;
            LOG_WARN("iterate ai column group tasks failed", K(ret));
          } else {
            PolledTask task;
            task.table_id_ = table_id;
            int64_t v = 0;
            if (OB_SUCC(ret)) {
              EXTRACT_INT_FIELD_MYSQL(*result, "rowkey_hash", v, int64_t);
              task.rowkey_hash_ = static_cast<uint64_t>(v);
            }
            if (OB_SUCC(ret)) {
              EXTRACT_INT_FIELD_MYSQL(*result, "column_id", v, int64_t);
              task.column_id_ = static_cast<uint64_t>(v);
            }
            if (OB_SUCC(ret)) {
              EXTRACT_INT_FIELD_MYSQL(*result, "state", v, int64_t);
              task.state_ = v;
            }
            if (OB_SUCC(ret)) {
              EXTRACT_INT_FIELD_MYSQL(*result, "retry_count", v, int64_t);
              task.retry_count_ = v;
            }
            if (OB_SUCC(ret)) {
              EXTRACT_INT_FIELD_MYSQL(*result, "max_retries", v, int64_t);
              task.max_retries_ = v;
            }
            if (OB_SUCC(ret)) {
              EXTRACT_INT_FIELD_MYSQL(*result, "def_version", v, int64_t);
              task.def_version_ = v;
            }
            common::ObString pred;
            if (OB_SUCC(ret)) {
              EXTRACT_VARCHAR_FIELD_MYSQL(*result, "rowkey_predicate", pred);
            }
            if (OB_SUCC(ret)) {
              if (OB_FAIL(ob_write_string(alloc, pred, task.rowkey_predicate_, true))) {
                LOG_WARN("copy rowkey predicate failed", K(ret));
              } else if (OB_FAIL(ob_write_string(alloc, request_id, task.request_id_, true))) {
                LOG_WARN("copy request id failed", K(ret));
              }
            }
            if (OB_SUCC(ret) && OB_FAIL(tasks.push_back(task))) {
              LOG_WARN("push back group task failed", K(ret));
            }
          }
        }
      }
    }
  }
  return ret;
}

int ObAiColumnWorkerTask::process_group_(common::ObIAllocator &tick_alloc,
                                         common::ObIArray<PolledTask *> &tasks)
{
  int ret = OB_SUCCESS;
  const uint64_t table_id = tasks.at(0)->table_id_;
  const common::ObString request_id = tasks.at(0)->request_id_;

  // 1. find or create the in-memory group and mark it as processing
  MemGroup *group = nullptr;
  bool created = false;
  {
    lib::ObMutexGuard guard(mem_lock_);
    group = find_mem_group_(table_id, request_id);
    if (nullptr == group) {
      void *buf = common::ob_malloc(sizeof(MemGroup), ObModIds::OMT);
      if (OB_ISNULL(buf)) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
        LOG_WARN("allocate ai column mem group failed", K(ret));
      } else if (FALSE_IT(group = new (buf) MemGroup())) {
      } else if (OB_FAIL(ob_write_string(group->alloc_, request_id, group->request_id_, true))) {
        LOG_WARN("copy request id failed", K(ret));
      } else if (OB_FAIL(ob_write_string(group->alloc_, tasks.at(0)->rowkey_predicate_,
                                         group->rowkey_predicate_, true))) {
        LOG_WARN("copy rowkey predicate failed", K(ret));
      } else if (!mem_groups_.add_last(group)) {
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("add ai column mem group failed", K(ret));
      } else {
        group->table_id_ = table_id;
        created = true;
      }
      if (OB_FAIL(ret) && nullptr != group) {
        group->~MemGroup();
        common::ob_free(group);
        group = nullptr;
      }
    }
    if (OB_SUCC(ret) && nullptr != group) {
      group->processing_ = true;
    }
  }

  // 2. claim the group's task rows with a CAS in its own transaction
  bool claimed = false;
  int64_t affected = 0;
  if (OB_FAIL(ret)) {
    // fall through to the cleanup below
  } else if (OB_FAIL(claim_group_(table_id, request_id, affected))) {
    LOG_WARN("claim ai column group failed", K(ret), K(table_id), K(request_id));
  } else if (0 == affected) {
    // another worker owns the group, or the request was replaced while we
    // polled; a group without remaining task rows is finished elsewhere
    int64_t remaining = 0;
    const int tmp_ret = count_group_task_rows_(table_id, request_id, remaining);
    if (OB_SUCCESS != tmp_ret) {
      LOG_WARN_RET(tmp_ret, "count remaining ai column task rows failed", K(tmp_ret), K(table_id),
                   K(request_id));
    }
    lib::ObMutexGuard guard(mem_lock_);
    if (created || 0 == remaining) {
      free_mem_group_(group);
      group = nullptr;
    } else {
      group->processing_ = false;
    }
  } else {
    claimed = true;
  }
  if (OB_SUCC(ret) && !claimed) {
    return ret;  // the group is released (or freed) above
  }

  // 3. load the ai metadata of the table
  bool table_exists = false;
  int64_t schema_version = 0;
  TableName names;
  common::ObArray<ObAIColumnItem> items;
  if (OB_FAIL(ret)) {
    // fall through to the cleanup below
  } else if (OB_FAIL(resolve_table_meta_(tick_alloc, table_id, table_exists, schema_version,
                                         names))) {
    LOG_WARN("resolve ai column table meta failed", K(ret), K(table_id));
  } else if (!table_exists) {
    // the table is gone: drop the group's task rows and the in-memory group
    int64_t tmp_affected = 0;
    const int tmp_ret = delete_group_tasks_(table_id, request_id, tmp_affected);
    if (OB_SUCCESS != tmp_ret) {
      LOG_WARN_RET(tmp_ret, "delete ai column task rows of dropped table failed", K(tmp_ret),
                   K(table_id), K(request_id));
    }
    lib::ObMutexGuard guard(mem_lock_);
    free_mem_group_(group);
    group = nullptr;
  } else {
    bool found = false;
    if (OB_FAIL(ObAiColumnCache::get_instance().get_table_items(
            table_id, schema_version, tick_alloc, items, found))) {
      LOG_WARN("load ai column items failed", K(ret), K(table_id), K(schema_version));
    } else if (!found || items.empty()) {
      // the table has no ai column any more: drop the group's task rows
      int64_t tmp_affected = 0;
      const int tmp_ret = delete_group_tasks_(table_id, request_id, tmp_affected);
      if (OB_SUCCESS != tmp_ret) {
        LOG_WARN_RET(tmp_ret, "delete ai column task rows of dropped ai columns failed",
                     K(tmp_ret), K(table_id), K(request_id));
      }
      lib::ObMutexGuard guard(mem_lock_);
      free_mem_group_(group);
      group = nullptr;
    }
  }

  // 4. load every task row the group owns and execute it
  bool row_deleted = false;
  ObSEArray<PolledTask, 8> exec_tasks;
  if (OB_SUCC(ret) && nullptr != group) {
    if (OB_FAIL(load_group_tasks_(tick_alloc, table_id, request_id, exec_tasks))) {
      LOG_WARN("load ai column group tasks failed", K(ret), K(table_id), K(request_id));
    } else if (exec_tasks.empty()) {
      // the task rows vanished (replaced request): nothing left to do
      lib::ObMutexGuard guard(mem_lock_);
      free_mem_group_(group);
      group = nullptr;
    } else {
      for (int64_t i = 0; OB_SUCC(ret) && !row_deleted && i < exec_tasks.count(); ++i) {
        PolledTask &task = exec_tasks.at(i);
        if (OB_FAIL(execute_task_(tick_alloc, *group, task, names, items, row_deleted))) {
          // a broken task must not stop the group
          LOG_WARN("execute ai column task failed", K(ret), K(table_id), K(request_id),
                   K(task.column_id_));
          ret = OB_SUCCESS;
        }
      }
    }
  }

  // 5. group lifecycle
  if (OB_FAIL(ret)) {
    // a transient failure after the claim: give the claimed rows back to a
    // later tick; cached results of succeeded members stay in memory so that
    // they are not recomputed
    int tmp_ret = requeue_group_(table_id, request_id);
    if (OB_SUCCESS != tmp_ret) {
      LOG_WARN_RET(tmp_ret, "requeue ai column group failed", K(tmp_ret), K(table_id),
                   K(request_id));
    }
    lib::ObMutexGuard guard(mem_lock_);
    if (nullptr != group) {
      group->processing_ = false;
    }
  } else if (row_deleted) {
    // the user row is gone: the group and its task rows are dropped already
    lib::ObMutexGuard guard(mem_lock_);
    free_mem_group_(group);
    group = nullptr;
  } else if (nullptr == group) {
    // the group was dropped in one of the steps above
  } else {
    // the group can be published when every member it owns has a terminal
    // (cached) result
    bool all_terminal = false;
    {
      lib::ObMutexGuard guard(mem_lock_);
      all_terminal = !exec_tasks.empty();
      for (int64_t i = 0; all_terminal && i < exec_tasks.count(); ++i) {
        const MemTask *mem = group->find_task(exec_tasks.at(i).column_id_);
        all_terminal = (nullptr != mem && MEM_TASK_RUNNING != mem->status_);
      }
    }
    if (!all_terminal) {
      // some member waits for its retry: keep the cached results of the
      // already succeeded members for the next tick
      lib::ObMutexGuard guard(mem_lock_);
      group->processing_ = false;
    } else if (OB_FAIL(publish_group_(*group, names, items, affected))) {
      LOG_WARN("publish ai column group failed", K(ret), K(table_id), K(request_id));
      // give the whole group another chance in a later tick; the cached
      // results are dropped so that the members are recomputed
      int tmp_ret = requeue_group_(table_id, request_id);
      if (OB_SUCCESS != tmp_ret) {
        LOG_WARN_RET(tmp_ret, "requeue ai column group failed", K(tmp_ret), K(table_id),
                     K(request_id));
      }
      lib::ObMutexGuard guard(mem_lock_);
      free_mem_group_(group);
      group = nullptr;
    } else {
      // published (affected may be 0 when a newer request replaced this one,
      // the task rows are removed either way): drop the in-memory group
      lib::ObMutexGuard guard(mem_lock_);
      free_mem_group_(group);
      group = nullptr;
    }
  }
  return ret;
}

int ObAiColumnWorkerTask::claim_group_(const uint64_t table_id,
                                       const common::ObString &request_id, int64_t &affected)
{
  int ret = OB_SUCCESS;
  affected = 0;
  ObSqlString sql;
  ObMySQLTransaction trans;
  if (OB_ISNULL(GCTX.sql_proxy_)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("sql proxy is null", K(ret));
  } else if (OB_FAIL(trans.start(GCTX.sql_proxy_))) {
    LOG_WARN("start trans failed", K(ret));
  } else if (OB_FAIL(sql.assign_fmt(
      "UPDATE %s SET state = %ld WHERE table_id = %lu AND request_id = '%.*s' AND state IN (%ld, %ld)",
      OB_ALL_AI_COLUMN_TASK_TNAME,
      static_cast<int64_t>(OB_AI_TASK_STATE_EXECUTING),
      table_id, request_id.length(), request_id.ptr(),
      static_cast<int64_t>(OB_AI_TASK_STATE_PENDING),
      static_cast<int64_t>(OB_AI_TASK_STATE_RETRY_WAIT)))) {
    LOG_WARN("build claim sql failed", K(ret));
  } else if (OB_FAIL(trans.write(sql.ptr(), affected))) {
    LOG_WARN("claim ai column tasks failed", K(ret), K(sql.ptr()));
  }
  if (trans.is_started()) {
    int tmp_ret = OB_SUCCESS;
    if (OB_SUCCESS != (tmp_ret = trans.end(OB_SUCC(ret)))) {
      LOG_WARN("end trans failed", K(tmp_ret));
      ret = OB_SUCC(ret) ? tmp_ret : ret;
    }
  }
  return ret;
}

int ObAiColumnWorkerTask::count_group_task_rows_(const uint64_t table_id,
                                                 const common::ObString &request_id,
                                                 int64_t &count)
{
  int ret = OB_SUCCESS;
  count = 0;
  ObSqlString sql;
  ObMySQLProxy *proxy = GCTX.sql_proxy_;
  if (OB_ISNULL(proxy)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(sql.assign_fmt(
      "SELECT count(*) AS cnt FROM %s WHERE table_id = %lu AND request_id = '%.*s'",
      OB_ALL_AI_COLUMN_TASK_TNAME, table_id, request_id.length(), request_id.ptr()))) {
    LOG_WARN("build count sql failed", K(ret));
  } else {
    SMART_VAR(ObMySQLProxy::MySQLResult, res) {
      ObMySQLResult *result = nullptr;
      if (OB_FAIL(proxy->read(res, sql.ptr()))) {
        LOG_WARN("count ai column task rows failed", K(ret), K(sql.ptr()));
      } else if (OB_ISNULL(result = res.get_result())) {
        ret = OB_ERR_UNEXPECTED;
      } else {
        const int tmp_ret = result->next();
        if (OB_ITER_END == tmp_ret) {
          // no row: count stays 0
        } else if (OB_SUCCESS != tmp_ret) {
          ret = tmp_ret;
          LOG_WARN("iterate count row failed", K(ret));
        } else {
          int64_t v = 0;
          EXTRACT_INT_FIELD_MYSQL(*result, "cnt", v, int64_t);
          count = v;
        }
      }
    }
  }
  return ret;
}

int ObAiColumnWorkerTask::resolve_table_meta_(common::ObIAllocator &alloc,
                                              const uint64_t table_id, bool &table_exists,
                                              int64_t &schema_version, TableName &names)
{
  int ret = OB_SUCCESS;
  table_exists = false;
  schema_version = 0;
  names.db_name_.reset();
  names.table_name_.reset();
  schema::ObSchemaGetterGuard guard;
  const schema::ObTableSchema *table_schema = nullptr;
  const schema::ObDatabaseSchema *db_schema = nullptr;
  if (OB_ISNULL(GCTX.schema_service_)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("schema service is null", K(ret));
  } else if (OB_FAIL(GCTX.schema_service_->get_runtime_schema_guard(guard))) {
    LOG_WARN("get runtime schema guard failed", K(ret));
  } else if (OB_FAIL(guard.get_schema_version(schema_version))) {
    LOG_WARN("get schema version failed", K(ret));
  } else if (OB_FAIL(guard.get_table_schema(table_id, table_schema))) {
    if (OB_TABLE_NOT_EXIST == ret) {
      ret = OB_SUCCESS;  // the table is gone
    } else {
      LOG_WARN("get table schema failed", K(ret), K(table_id));
    }
  } else if (OB_ISNULL(table_schema)) {
    // the table is gone
  } else if (OB_FAIL(guard.get_database_schema(table_schema->get_database_id(), db_schema))) {
    if (OB_TABLE_NOT_EXIST == ret) {
      ret = OB_SUCCESS;  // the database is gone
    } else {
      LOG_WARN("get database schema failed", K(ret), K(table_id));
    }
  } else if (OB_ISNULL(db_schema)) {
    // the database is gone
  } else if (OB_FAIL(ob_write_string(alloc, db_schema->get_database_name_str(),
                                     names.db_name_, true))) {
    LOG_WARN("copy database name failed", K(ret));
  } else if (OB_FAIL(ob_write_string(alloc, table_schema->get_table_name_str(),
                                     names.table_name_, true))) {
    LOG_WARN("copy table name failed", K(ret));
  } else {
    table_exists = true;
  }
  return ret;
}

int ObAiColumnWorkerTask::execute_task_(common::ObIAllocator &tick_alloc, MemGroup &group,
                                        PolledTask &task, const TableName &names,
                                        common::ObIArray<ObAIColumnItem> &items,
                                        bool &row_deleted)
{
  int ret = OB_SUCCESS;
  row_deleted = false;
  const ObAIColumnItem *item = nullptr;
  for (int64_t i = 0; OB_ISNULL(item) && i < items.count(); ++i) {
    if (items.at(i).column_id_ == task.column_id_) {
      item = &items.at(i);
    }
  }
  MemTask *mem = nullptr;
  {
    lib::ObMutexGuard guard(mem_lock_);
    mem = group.find_task(task.column_id_);
  }
  if (nullptr == item) {
    // the ai column was dropped while the task was in flight: record a
    // terminal failure so that the group can be published without it
    LOG_WARN("ai column item not found, fail the task", K(task.column_id_), K(task.table_id_));
    lib::ObMutexGuard guard(mem_lock_);
    if (nullptr == mem) {
      MemTask fresh;
      fresh.column_id_ = task.column_id_;
      fresh.status_ = MEM_TASK_FAILED;
      fresh.def_version_ = task.def_version_;
      if (OB_FAIL(group.tasks_.push_back(fresh))) {
        LOG_WARN("push back mem task failed", K(ret));
      }
    } else {
      mem->status_ = MEM_TASK_FAILED;
      mem->is_null_ = true;
      mem->result_.reset();
    }
  } else if (nullptr != mem && MEM_TASK_RUNNING != mem->status_) {
    // already computed by an earlier tick while another member of the group
    // waited for its retry: reuse the cached result, never call again
    LOG_TRACE("reuse cached ai column result", K(task.column_id_));
  } else {
    // evaluate the current prompt of the row
    int64_t row_count = 0;
    common::ObString user_prompt;
    bool prompt_ok = false;
    if (OB_FAIL(read_user_prompt_(tick_alloc, names, *item, task.rowkey_predicate_,
                                  row_count, user_prompt))) {
      // infra failure: a failed attempt is recorded below
      LOG_WARN("read ai prompt failed", K(ret), K(task.table_id_), K(task.column_id_));
      ret = OB_SUCCESS;
    } else if (0 == row_count) {
      // the user row is gone: the whole group's task rows are dropped
      row_deleted = true;
      int64_t tmp_affected = 0;
      const int tmp_ret = delete_group_tasks_(group.table_id_, group.request_id_, tmp_affected);
      if (OB_SUCCESS != tmp_ret) {
        LOG_WARN_RET(tmp_ret, "delete ai column task rows of deleted row failed", K(tmp_ret),
                     K(group.table_id_), K(group.request_id_));
      }
    } else if (user_prompt.empty()) {
      LOG_WARN("ai prompt is empty, fail the attempt", K(task.column_id_));
    } else {
      prompt_ok = true;
    }

    common::ObString output;
    bool valid = false;
    common::ObString literal;
    if (row_deleted) {
      // nothing to do
    } else if (!prompt_ok) {
      // failed attempt without a model call
    } else if (OB_FAIL(call_model_(tick_alloc, *item, user_prompt, output))) {
      LOG_WARN("ai model call failed", K(ret), K(task.column_id_), K(item->model_name_));
      ret = OB_SUCCESS;
    } else if (output.empty()) {
      LOG_WARN("ai model output is empty", K(task.column_id_));
    } else {
      const int tmp_ret = validate_and_build_literal_(group.alloc_, *item, output, valid, literal);
      if (OB_SUCCESS != tmp_ret) {
        LOG_WARN_RET(tmp_ret, "build ai output literal failed", K(tmp_ret), K(task.column_id_));
        valid = false;
      }
    }

    if (!row_deleted && OB_SUCC(ret)) {
      // decide the outcome of the attempt
      bool terminal_fail = true;
      if (valid) {
        terminal_fail = false;
      } else if (task.retry_count_ < task.max_retries_) {
        terminal_fail = false;
        if (OB_FAIL(defer_retry_(task))) {
          LOG_WARN("defer ai task retry failed", K(ret), K(task.column_id_));
          terminal_fail = true;
          ret = OB_SUCCESS;
        }
      }
      // record the result of the attempt in the in-memory group
      lib::ObMutexGuard guard(mem_lock_);
      if (nullptr == mem) {
        MemTask fresh;
        fresh.column_id_ = task.column_id_;
        if (OB_FAIL(group.tasks_.push_back(fresh))) {
          LOG_WARN("push back mem task failed", K(ret));
        } else {
          mem = group.find_task(task.column_id_);
        }
      }
      if (OB_SUCC(ret) && nullptr != mem) {
        mem->def_version_ = task.def_version_;
        if (valid) {
          mem->status_ = MEM_TASK_READY;
          mem->is_null_ = false;
          mem->result_ = literal;
        } else if (terminal_fail) {
          mem->status_ = MEM_TASK_FAILED;
          mem->is_null_ = true;
          mem->result_.reset();
        } else {
          mem->status_ = MEM_TASK_RUNNING;
          mem->is_null_ = true;
          mem->result_.reset();
        }
      }
    }
  }
  return ret;
}

int ObAiColumnWorkerTask::read_user_prompt_(common::ObIAllocator &alloc, const TableName &names,
                                            const ObAIColumnItem &item,
                                            const common::ObString &rowkey_predicate,
                                            int64_t &row_count, common::ObString &prompt)
{
  int ret = OB_SUCCESS;
  row_count = 0;
  prompt.reset();
  ObSqlString sql;
  ObMySQLProxy *proxy = GCTX.sql_proxy_;
  if (OB_ISNULL(proxy)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("sql proxy is null", K(ret));
  } else if (OB_FAIL(sql.assign_fmt("SELECT %.*s AS p FROM `%.*s`.`%.*s` WHERE %.*s",
                                    item.user_expr_.length(), item.user_expr_.ptr(),
                                    names.db_name_.length(), names.db_name_.ptr(),
                                    names.table_name_.length(), names.table_name_.ptr(),
                                    rowkey_predicate.length(), rowkey_predicate.ptr()))) {
    LOG_WARN("build ai prompt sql failed", K(ret));
  } else {
    SMART_VAR(ObMySQLProxy::MySQLResult, res) {
      ObMySQLResult *result = nullptr;
      if (OB_FAIL(proxy->read(res, sql.ptr()))) {
        LOG_WARN("read ai prompt failed", K(ret), K(sql.ptr()));
      } else if (OB_ISNULL(result = res.get_result())) {
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("prompt result is null", K(ret));
      } else {
        const int tmp_ret = result->next();
        if (OB_ITER_END == tmp_ret) {
          // the user row is gone
        } else if (OB_SUCCESS != tmp_ret) {
          ret = tmp_ret;
          LOG_WARN("iterate ai prompt row failed", K(ret));
        } else {
          row_count = 1;
          common::ObString val;
          if (OB_FAIL(result->get_varchar("p", val))) {
            if (OB_ERR_NULL_VALUE == ret) {
              // NULL prompt: a failed attempt without a model call
              ret = OB_SUCCESS;
            } else {
              LOG_WARN("get ai prompt value failed", K(ret));
            }
          } else if (OB_FAIL(ob_write_string(alloc, val, prompt, true))) {
            LOG_WARN("copy ai prompt failed", K(ret));
          }
        }
      }
    }
  }
  return ret;
}

bool ObAiColumnWorkerTask::probe_endpoint_reachable_(const common::ObString &url)
{
  bool reachable = false;
  // parse "[scheme://][user[:pass]@]host[:port]/..." and probe the endpoint
  // with a cheap TCP connect so that an offline endpoint fails an attempt
  // without waiting for the http client retries
  int64_t pos = 0;
  const int64_t len = url.length();
  const char *p = url.ptr();
  int64_t scheme_end = -1;
  while (pos + 2 < len) {
    if (':' == p[pos] && '/' == p[pos + 1] && '/' == p[pos + 2]) {
      scheme_end = pos;
      break;
    }
    ++pos;
  }
  bool is_https = false;
  int64_t host_start = 0;
  if (scheme_end >= 0) {
    is_https = (5 == scheme_end && 0 == strncasecmp(p, "https", 5));
    host_start = scheme_end + 3;
  }
  // skip an optional userinfo part
  for (int64_t i = host_start; i < len; ++i) {
    if ('@' == p[i]) {
      host_start = i + 1;
    } else if ('/' == p[i] || ':' == p[i] || '?' == p[i] || '#' == p[i]) {
      break;
    }
  }
  int64_t host_end = host_start;
  while (host_end < len && '/' != p[host_end] && ':' != p[host_end]
         && '?' != p[host_end] && '#' != p[host_end]) {
    ++host_end;
  }
  const int64_t host_len = host_end - host_start;
  int64_t port = is_https ? 443 : 80;
  bool port_ok = true;
  if (host_end < len && ':' == p[host_end]) {
    int64_t q = host_end + 1;
    port = 0;
    while (q < len && p[q] >= '0' && p[q] <= '9') {
      port = port * 10 + (p[q] - '0');
      ++q;
    }
    if (q == host_end + 1 || port <= 0 || port > 65535
        || (q < len && '/' != p[q] && '?' != p[q] && '#' != p[q])) {
      port_ok = false;
    }
  }
  char host_buf[256];
  if (host_len <= 0 || host_len >= static_cast<int64_t>(sizeof(host_buf)) || !port_ok
      || OB_ISNULL(p)) {
    LOG_WARN("cannot parse ai endpoint url for the reachability probe", K(url));
  } else {
    MEMCPY(host_buf, p + host_start, host_len);
    host_buf[host_len] = '\0';
    char port_buf[16];
    snprintf(port_buf, sizeof(port_buf), "%ld", port);
    struct addrinfo hints;
    struct addrinfo *addr = nullptr;
    MEMSET(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (0 != getaddrinfo(host_buf, port_buf, &hints, &addr) || OB_ISNULL(addr)) {
      LOG_WARN("resolve ai endpoint host failed", K(host_buf));
    } else {
      const int fd = socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol);
      if (fd < 0) {
        LOG_WARN("create probe socket failed", K(errno));
      } else {
        const int flags = fcntl(fd, F_GETFL, 0);
        if (flags >= 0) {
          IGNORE_RETURN fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        }
        if (0 == connect(fd, addr->ai_addr, addr->ai_addrlen)) {
          reachable = true;
        } else if (EINPROGRESS == errno) {
          struct pollfd pfd;
          pfd.fd = fd;
          pfd.events = POLLOUT;
          pfd.revents = 0;
          if (poll(&pfd, 1, OB_AI_WORKER_PROBE_TIMEOUT_MS) > 0) {
            int so_err = 0;
            socklen_t so_len = sizeof(so_err);
            if (0 == getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_err, &so_len) && 0 == so_err) {
              reachable = true;
            }
          }
        }
        close(fd);
      }
      freeaddrinfo(addr);
    }
  }
  return reachable;
}

int ObAiColumnWorkerTask::call_model_(common::ObIAllocator &alloc, const ObAIColumnItem &item,
                                      const common::ObString &user_prompt,
                                      common::ObString &output)
{
  int ret = OB_SUCCESS;
  output.reset();
  query::ObIAiEndpointResolver *resolver =
      ::oceanbase::share::server_service<::oceanbase::query::ObIAiEndpointResolver>();
  ObAiModelEndpointInfo endpoint;
  if (OB_ISNULL(resolver)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("ai endpoint resolver not bound", K(ret));
  } else if (OB_FAIL(resolver->resolve_by_model_name(item.model_name_, alloc, endpoint))) {
    LOG_WARN("resolve ai endpoint failed", K(ret), K(item.model_name_));
  } else if (!probe_endpoint_reachable_(endpoint.get_url())) {
    ret = OB_CURL_ERROR;
    LOG_WARN("ai endpoint is not reachable", K(ret), K(endpoint.get_url()));
  } else {
    // the request model name: endpoint override first, then the model schema
    common::ObString request_model_name = endpoint.get_request_model_name();
    if (request_model_name.empty()) {
      common::ObString resolved;
      if (OB_FAIL(query::ObAIModelResolver::resolve_model_name(alloc, item.model_name_,
                                                               resolved))) {
        LOG_WARN("resolve ai model name failed", K(ret), K(item.model_name_));
      } else {
        request_model_name = resolved;
      }
    }
    common::ObString unencrypted_key;
    common::ObArray<common::ObString> headers;
    if (OB_FAIL(ret)) {
    } else if (OB_FAIL(endpoint.get_unencrypted_access_key(alloc, unencrypted_key))) {
      LOG_WARN("get unencrypted access key failed", K(ret), K(item.model_name_));
    } else if (OB_AI_COL_FUNC_EMBED == item.func_type_) {
      ObAIFuncIEmbed *provider = nullptr;
      common::ObArray<common::ObString> contents;
      common::ObJsonObject *body = nullptr;
      common::ObJsonObject *response = nullptr;
      common::ObIJsonBase *result_base = nullptr;
      ObAIFuncClient client;
      client.set_timeout_sec(OB_AI_WORKER_TIMEOUT_SEC);
      if (OB_FAIL(ObAIFuncUtils::get_embed_provider(alloc, endpoint.get_provider(), provider))) {
        LOG_WARN("get embed provider failed", K(ret), K(endpoint.get_provider()));
      } else if (OB_FAIL(contents.push_back(user_prompt))) {
        LOG_WARN("push back embed content failed", K(ret));
      } else if (OB_FAIL(provider->get_header(alloc, unencrypted_key, headers))) {
        LOG_WARN("get embed header failed", K(ret));
      } else if (OB_FAIL(provider->get_body(alloc, request_model_name, contents, nullptr, body))) {
        LOG_WARN("get embed body failed", K(ret));
      } else if (OB_FAIL(client.send_post(alloc, endpoint.get_url(), headers, body, response))) {
        LOG_WARN("ai embed request failed", K(ret), K(endpoint.get_url()));
      } else if (OB_FAIL(provider->parse_output(alloc, response, result_base))) {
        LOG_WARN("parse ai embed output failed", K(ret));
      } else if (item.is_vector_ && item.vector_dim_ > 0
                 && OB_FAIL(ObAIFuncUtils::validate_embedding_vector(result_base,
                                                                     item.vector_dim_))) {
        LOG_WARN("embedding vector does not match the column dimension", K(ret),
                 K(item.vector_dim_));
      } else if (OB_FAIL(ObAIFuncJsonUtils::print_json_to_str(alloc, result_base, output))) {
        LOG_WARN("print ai embed output failed", K(ret));
      }
    } else {
      ObAIFuncIComplete *provider = nullptr;
      const char *system_prompt = get_worker_system_prompt(item.result_column_type_str_);
      common::ObString system_prompt_str;
      if (nullptr != system_prompt) {
        system_prompt_str.assign_ptr(system_prompt, static_cast<int32_t>(STRLEN(system_prompt)));
      }
      common::ObString user_prompt_str = user_prompt;
      common::ObJsonObject *body = nullptr;
      common::ObJsonObject *response = nullptr;
      common::ObIJsonBase *result_base = nullptr;
      ObAIFuncClient client;
      client.set_timeout_sec(OB_AI_WORKER_TIMEOUT_SEC);
      if (OB_FAIL(ObAIFuncUtils::get_complete_provider(alloc, endpoint.get_provider(),
                                                       provider))) {
        LOG_WARN("get complete provider failed", K(ret), K(endpoint.get_provider()));
      } else if (OB_FAIL(provider->get_header(alloc, unencrypted_key, headers))) {
        LOG_WARN("get complete header failed", K(ret));
      } else if (OB_FAIL(provider->get_body(alloc, request_model_name, system_prompt_str,
                                            user_prompt_str, nullptr, body))) {
        LOG_WARN("get complete body failed", K(ret));
      } else if (OB_FAIL(client.send_post(alloc, endpoint.get_url(), headers, body, response))) {
        LOG_WARN("ai complete request failed", K(ret), K(endpoint.get_url()));
      } else if (OB_FAIL(provider->parse_output(alloc, response, result_base))) {
        LOG_WARN("parse ai complete output failed", K(ret));
      } else if (OB_FAIL(ObAIFuncJsonUtils::print_json_to_str(alloc, result_base, output))) {
        LOG_WARN("print ai complete output failed", K(ret));
      }
    }
  }
  return ret;
}

int ObAiColumnWorkerTask::validate_and_build_literal_(common::ObIAllocator &alloc,
                                                      const ObAIColumnItem &item,
                                                      const common::ObString &output,
                                                      bool &valid, common::ObString &literal)
{
  int ret = OB_SUCCESS;
  valid = false;
  literal.reset();
  common::ObSqlString lit_buf;
  const common::ObString &type_str = item.result_column_type_str_;
  if (OB_AI_COL_FUNC_EMBED == item.func_type_) {
    // the printed embedding array is the bare bracket vector literal
    if (OB_FAIL(lit_buf.assign(output.ptr(), output.length()))) {
      LOG_WARN("assign vector literal failed", K(ret));
    } else {
      valid = true;
    }
  } else if (0 == type_str.case_compare("JSON")) {
    ObJsonNode *j_tree = nullptr;
    if (OB_ISNULL(output.ptr()) || output.empty()) {
      LOG_WARN("empty ai output for a json result column", K(item.column_id_));
    } else if (OB_FAIL(ObJsonParser::get_tree(&alloc, output, j_tree))) {
      LOG_WARN("ai output is not a valid json value", K(ret), K(output));
      ret = OB_SUCCESS;  // validation failure, not an infra failure
    } else {
      valid = true;
    }
    if (valid && OB_FAIL(sql_append_hex_escape_str(output, lit_buf))) {
      LOG_WARN("escape json literal failed", K(ret));
    }
  } else if (is_integer_result_type(type_str)) {
    common::ObString trimmed;
    trim_sql_whitespace(output, trimmed);
    char buf[64];
    if (trimmed.empty() || trimmed.length() >= static_cast<int64_t>(sizeof(buf))) {
      LOG_WARN("ai output is not a valid integer", K(item.column_id_), K(output));
    } else {
      MEMCPY(buf, trimmed.ptr(), trimmed.length());
      buf[trimmed.length()] = '\0';
      errno = 0;
      char *end = nullptr;
      const long long value = strtoll(buf, &end, 10);
      if (buf == end || nullptr == end || '\0' != *end || ERANGE == errno) {
        LOG_WARN("ai output is not a valid integer", K(item.column_id_), K(output));
      } else if (OB_FAIL(lit_buf.append_fmt("%lld", value))) {
        LOG_WARN("append integer literal failed", K(ret));
      } else {
        valid = true;
      }
    }
  } else if (is_number_result_type(type_str)) {
    common::ObString trimmed;
    trim_sql_whitespace(output, trimmed);
    char buf[128];
    bool numeric_chars = !trimmed.empty()
        && trimmed.length() < static_cast<int64_t>(sizeof(buf));
    for (int64_t i = 0; numeric_chars && i < trimmed.length(); ++i) {
      const char c = trimmed.ptr()[i];
      numeric_chars = (c >= '0' && c <= '9') || '.' == c || '+' == c || '-' == c
          || 'e' == c || 'E' == c;
    }
    if (!numeric_chars) {
      LOG_WARN("ai output is not a valid number", K(item.column_id_), K(output));
    } else {
      MEMCPY(buf, trimmed.ptr(), trimmed.length());
      buf[trimmed.length()] = '\0';
      errno = 0;
      char *end = nullptr;
      const double value = strtod(buf, &end);
      if (buf == end || nullptr == end || '\0' != *end || ERANGE == errno || !isfinite(value)) {
        LOG_WARN("ai output is not a valid number", K(item.column_id_), K(output));
      } else if (OB_FAIL(lit_buf.assign(trimmed.ptr(), trimmed.length()))) {
        LOG_WARN("assign number literal failed", K(ret));
      } else {
        valid = true;
      }
    }
  } else {
    // VARCHAR and the date/time types: publish as-is, the UPDATE cast of the
    // result column validates the value
    if (OB_FAIL(sql_append_hex_escape_str(output, lit_buf))) {
      LOG_WARN("escape string literal failed", K(ret));
    } else {
      valid = true;
    }
  }
  if (OB_SUCC(ret) && valid) {
    if (OB_FAIL(ob_write_string(alloc, lit_buf.string(), literal))) {
      LOG_WARN("copy result literal failed", K(ret));
    }
  }
  return ret;
}

int ObAiColumnWorkerTask::defer_retry_(PolledTask &task)
{
  int ret = OB_SUCCESS;
  ObSqlString sql;
  ObMySQLTransaction trans;
  if (OB_ISNULL(GCTX.sql_proxy_)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("sql proxy is null", K(ret));
  } else if (OB_FAIL(trans.start(GCTX.sql_proxy_))) {
    LOG_WARN("start trans failed", K(ret));
  } else if (OB_FAIL(sql.assign_fmt(
      "UPDATE %s SET state = %ld, retry_count = retry_count + 1, "
      "eligible_ts = DATE_ADD(now(6), INTERVAL %ld MICROSECOND) "
      "WHERE table_id = %lu AND rowkey_hash = %lu AND column_id = %lu AND request_id = '%.*s'",
      OB_ALL_AI_COLUMN_TASK_TNAME,
      static_cast<int64_t>(OB_AI_TASK_STATE_RETRY_WAIT),
      OB_AI_WORKER_RETRY_INTERVAL_US,
      task.table_id_, task.rowkey_hash_, task.column_id_,
      task.request_id_.length(), task.request_id_.ptr()))) {
    LOG_WARN("build retry sql failed", K(ret));
  } else {
    int64_t affected = 0;
    if (OB_FAIL(trans.write(sql.ptr(), affected))) {
      LOG_WARN("defer ai task retry failed", K(ret), K(sql.ptr()));
    } else if (0 == affected) {
      // the task row was replaced while we were working on it
      LOG_INFO("ai task row vanished before the retry was recorded",
               K(task.table_id_), K(task.column_id_));
    }
  }
  if (trans.is_started()) {
    int tmp_ret = OB_SUCCESS;
    if (OB_SUCCESS != (tmp_ret = trans.end(OB_SUCC(ret)))) {
      LOG_WARN("end trans failed", K(tmp_ret));
      ret = OB_SUCC(ret) ? tmp_ret : ret;
    }
  }
  return ret;
}

int ObAiColumnWorkerTask::publish_group_(MemGroup &group, const TableName &names,
                                         common::ObIArray<ObAIColumnItem> &items,
                                         int64_t &affected)
{
  int ret = OB_SUCCESS;
  affected = 0;
  ObSqlString sql;
  ObMySQLTransaction trans;
  bool published = false;  // whether at least one member contributes a SET/WHERE part
  if (OB_ISNULL(GCTX.sql_proxy_)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("sql proxy is null", K(ret));
  } else if (OB_FAIL(sql.assign_fmt("UPDATE `%.*s`.`%.*s` SET ",
                                    names.db_name_.length(), names.db_name_.ptr(),
                                    names.table_name_.length(), names.table_name_.ptr()))) {
    LOG_WARN("build publish sql head failed", K(ret));
  } else {
    // SET clauses of every member with a live column definition
    bool first = true;
    for (int64_t i = 0; OB_SUCC(ret) && i < group.tasks_.count(); ++i) {
      MemTask &mem = group.tasks_.at(i);
      const ObAIColumnItem *item = find_item_(items, mem.column_id_);
      if (OB_ISNULL(item)) {
        // the column was dropped: nothing to publish for it
        continue;
      }
      published = true;
      if (!first && OB_FAIL(sql.append(", "))) {
        LOG_WARN("append set separator failed", K(ret));
      }
      first = false;
      if (OB_SUCC(ret)) {
        if (mem.is_null_) {
          if (OB_FAIL(sql.append_fmt("`%.*s` = NULL",
                                     item->result_column_name_.length(),
                                     item->result_column_name_.ptr()))) {
            LOG_WARN("append null result failed", K(ret));
          }
        } else if (OB_FAIL(sql.append_fmt("`%.*s` = ",
                                          item->result_column_name_.length(),
                                          item->result_column_name_.ptr()))
                   || OB_FAIL(sql.append(mem.result_))) {
          LOG_WARN("append result literal failed", K(ret));
        }
      }
      if (OB_SUCC(ret)) {
        const char *state = (MEM_TASK_READY == mem.status_)
            ? OB_AI_STATE_READY : OB_AI_STATE_FAILED;
        if (OB_FAIL(sql.append_fmt(", `%.*s` = '%s', `%.*s` = 'v%ld'",
                                   item->state_column_name_.length(),
                                   item->state_column_name_.ptr(),
                                   state,
                                   item->result_spec_column_name_.length(),
                                   item->result_spec_column_name_.ptr(),
                                   mem.def_version_))) {
          LOG_WARN("append state and spec failed", K(ret));
        }
      }
    }
    // the request columns guard the publish against stale results of a
    // replaced request
    for (int64_t i = 0; OB_SUCC(ret) && published && i < group.tasks_.count(); ++i) {
      MemTask &mem = group.tasks_.at(i);
      const ObAIColumnItem *item = find_item_(items, mem.column_id_);
      if (OB_ISNULL(item)) {
        continue;
      }
      if (OB_FAIL(sql.append_fmt(" AND `%.*s` = '%.*s'",
                                 item->request_column_name_.length(),
                                 item->request_column_name_.ptr(),
                                 group.request_id_.length(),
                                 group.request_id_.ptr()))) {
        LOG_WARN("append request guard failed", K(ret));
      }
    }
    if (OB_SUCC(ret) && published) {
      if (OB_FAIL(sql.append_fmt(" WHERE %.*s",
                                 group.rowkey_predicate_.length(),
                                 group.rowkey_predicate_.ptr()))) {
        LOG_WARN("append rowkey predicate failed", K(ret));
      }
    }
  }
  if (OB_FAIL(ret)) {
  } else if (OB_FAIL(trans.start(GCTX.sql_proxy_))) {
    LOG_WARN("start trans failed", K(ret));
  } else {
    if (published) {
      if (OB_FAIL(trans.write(sql.ptr(), affected))) {
        LOG_WARN("publish ai column results failed", K(ret), K(sql.ptr()));
      }
    }
    if (OB_SUCC(ret)) {
      // remove the group's task rows in the same transaction; when affected
      // is 0 a newer request replaced this one and only the stale task rows
      // (if any) are removed here
      ObSqlString del_sql;
      int64_t del_affected = 0;
      if (OB_FAIL(del_sql.assign_fmt(
          "DELETE FROM %s WHERE table_id = %lu AND request_id = '%.*s'",
          OB_ALL_AI_COLUMN_TASK_TNAME, group.table_id_,
          group.request_id_.length(), group.request_id_.ptr()))) {
        LOG_WARN("build task delete sql failed", K(ret));
      } else if (OB_FAIL(trans.write(del_sql.ptr(), del_affected))) {
        LOG_WARN("delete ai column task rows failed", K(ret), K(del_sql.ptr()));
      } else {
        LOG_TRACE("ai column group published", K(group.table_id_), K(group.request_id_),
                  K(affected));
      }
    }
  }
  if (trans.is_started()) {
    int tmp_ret = OB_SUCCESS;
    if (OB_SUCCESS != (tmp_ret = trans.end(OB_SUCC(ret)))) {
      LOG_WARN("end trans failed", K(tmp_ret));
      ret = OB_SUCC(ret) ? tmp_ret : ret;
    }
  }
  return ret;
}

int ObAiColumnWorkerTask::requeue_group_(const uint64_t table_id,
                                         const common::ObString &request_id)
{
  int ret = OB_SUCCESS;
  ObSqlString sql;
  ObMySQLTransaction trans;
  if (OB_ISNULL(GCTX.sql_proxy_)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(trans.start(GCTX.sql_proxy_))) {
    LOG_WARN("start trans failed", K(ret));
  } else if (OB_FAIL(sql.assign_fmt(
      "UPDATE %s SET state = %ld WHERE table_id = %lu AND request_id = '%.*s' AND state = %ld",
      OB_ALL_AI_COLUMN_TASK_TNAME,
      static_cast<int64_t>(OB_AI_TASK_STATE_PENDING),
      table_id, request_id.length(), request_id.ptr(),
      static_cast<int64_t>(OB_AI_TASK_STATE_EXECUTING)))) {
    LOG_WARN("build requeue sql failed", K(ret));
  } else {
    int64_t affected = 0;
    if (OB_FAIL(trans.write(sql.ptr(), affected))) {
      LOG_WARN("requeue ai column group failed", K(ret), K(sql.ptr()));
    }
  }
  if (trans.is_started()) {
    int tmp_ret = OB_SUCCESS;
    if (OB_SUCCESS != (tmp_ret = trans.end(OB_SUCC(ret)))) {
      LOG_WARN("end trans failed", K(tmp_ret));
      ret = OB_SUCC(ret) ? tmp_ret : ret;
    }
  }
  return ret;
}

int ObAiColumnWorkerTask::delete_group_tasks_(const uint64_t table_id,
                                              const common::ObString &request_id,
                                              int64_t &affected)
{
  int ret = OB_SUCCESS;
  affected = 0;
  ObSqlString sql;
  ObMySQLTransaction trans;
  if (OB_ISNULL(GCTX.sql_proxy_)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("sql proxy is null", K(ret));
  } else if (OB_FAIL(trans.start(GCTX.sql_proxy_))) {
    LOG_WARN("start trans failed", K(ret));
  } else if (OB_FAIL(sql.assign_fmt(
      "DELETE FROM %s WHERE table_id = %lu AND request_id = '%.*s'",
      OB_ALL_AI_COLUMN_TASK_TNAME, table_id, request_id.length(), request_id.ptr()))) {
    LOG_WARN("build task delete sql failed", K(ret));
  } else if (OB_FAIL(trans.write(sql.ptr(), affected))) {
    LOG_WARN("delete ai column task rows failed", K(ret), K(sql.ptr()));
  }
  if (trans.is_started()) {
    int tmp_ret = OB_SUCCESS;
    if (OB_SUCCESS != (tmp_ret = trans.end(OB_SUCC(ret)))) {
      LOG_WARN("end trans failed", K(tmp_ret));
      ret = OB_SUCC(ret) ? tmp_ret : ret;
    }
  }
  return ret;
}

const ObAIColumnItem *ObAiColumnWorkerTask::find_item_(common::ObIArray<ObAIColumnItem> &items,
                                                       const uint64_t column_id)
{
  const ObAIColumnItem *found = nullptr;
  for (int64_t i = 0; OB_ISNULL(found) && i < items.count(); ++i) {
    if (items.at(i).column_id_ == column_id) {
      found = &items.at(i);
    }
  }
  return found;
}

ObAiColumnWorkerTask::MemTask *ObAiColumnWorkerTask::MemGroup::find_task(const uint64_t column_id)
{
  MemTask *found = nullptr;
  for (int64_t i = 0; nullptr == found && i < tasks_.count(); ++i) {
    if (tasks_.at(i).column_id_ == column_id) {
      found = &tasks_.at(i);
    }
  }
  return found;
}

int64_t ObAiColumnWorkerTask::MemGroup::terminal_count() const
{
  int64_t count = 0;
  for (int64_t i = 0; i < tasks_.count(); ++i) {
    if (MEM_TASK_RUNNING != tasks_.at(i).status_) {
      ++count;
    }
  }
  return count;
}

ObAiColumnWorkerTask::MemGroup *ObAiColumnWorkerTask::find_mem_group_(
    const uint64_t table_id, const common::ObString &request_id)
{
  MemGroup *found = nullptr;
  for (MemGroup *cur = mem_groups_.get_first();
       nullptr == found && cur != mem_groups_.get_header();
       cur = cur->get_next()) {
    if (cur->table_id_ == table_id && cur->request_id_ == request_id) {
      found = cur;
    }
  }
  return found;
}

void ObAiColumnWorkerTask::free_mem_group_(MemGroup *group)
{
  if (nullptr != group) {
    mem_groups_.remove(group);
    group->~MemGroup();
    common::ob_free(group);
  }
}

// =========================== ObAiColumnWorkerService ==========================

ObAiColumnWorkerService::ObAiColumnWorkerService()
  : inited_(false), timer_(), task_()
{}

ObAiColumnWorkerService::~ObAiColumnWorkerService()
{
  destroy();
}

int ObAiColumnWorkerService::server_module_init(ObAiColumnWorkerService *&service)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(service)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(service->init())) {
  }
  return ret;
}

int ObAiColumnWorkerService::init()
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(inited_)) {
    ret = OB_INIT_TWICE;
  } else if (OB_FAIL(task_.init())) {
    LOG_WARN("init ai column worker task failed", K(ret));
  } else {
    inited_ = true;
  }
  return ret;
}

int ObAiColumnWorkerService::start()
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(!inited_)) {
    ret = OB_NOT_INIT;
  } else if (!timer_.inited()
             && OB_FAIL(timer_.init("AiColumnWorker", common::ObMemAttr("AiColumnWorker")))) {
    LOG_WARN("init ai column worker timer failed", K(ret));
  } else if (OB_FAIL(timer_.start())) {
    LOG_WARN("start ai column worker timer failed", K(ret));
  } else if (OB_FAIL(task_.start(timer_))) {
    LOG_WARN("start ai column worker task failed", K(ret));
  }
  return ret;
}

void ObAiColumnWorkerService::stop()
{
  const int64_t start_time = ObTimeUtility::fast_current_time();
  FLOG_INFO("start to stop ai column worker service");
  if (timer_.inited()) {
    task_.stop(timer_);
    timer_.stop();
  }
  const int64_t cost = ObTimeUtility::fast_current_time() - start_time;
  FLOG_INFO("finish to stop ai column worker service", K(cost));
}

void ObAiColumnWorkerService::wait()
{
  const int64_t start_time = ObTimeUtility::fast_current_time();
  FLOG_INFO("start to wait ai column worker service");
  if (timer_.inited()) {
    timer_.wait();
  }
  const int64_t cost = ObTimeUtility::fast_current_time() - start_time;
  FLOG_INFO("finish to wait ai column worker service", K(cost));
}

void ObAiColumnWorkerService::destroy()
{
  FLOG_INFO("start to destroy ai column worker service");
  timer_.destroy();
  if (inited_) {
    task_.destroy();
    inited_ = false;
  }
  FLOG_INFO("finish to destroy ai column worker service");
}

}  // end namespace share
}  // end namespace oceanbase
