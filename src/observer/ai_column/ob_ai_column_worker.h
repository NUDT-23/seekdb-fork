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

#ifndef OCEANBASE_OBSERVER_AI_COLUMN_OB_AI_COLUMN_WORKER_H_
#define OCEANBASE_OBSERVER_AI_COLUMN_OB_AI_COLUMN_WORKER_H_

#include "lib/allocator/page_arena.h"
#include "lib/list/ob_dlist.h"
#include "lib/lock/ob_mutex.h"
#include "lib/string/ob_string.h"
#include "lib/task/ob_timer.h"
#include "share/ai_column/ob_ai_column_util.h"

namespace oceanbase
{
namespace share
{

// Background AI column worker.  It periodically scans __all_ai_column_task
// for eligible tasks (state PENDING or RETRY_WAIT), claims the task rows of
// one (table, request) group, evaluates the prompt expression of every AI
// column of the group against the configured model endpoint, validates the
// model output against the result column type and finally publishes all
// results of the group to the hidden AI columns of the user row in one
// transaction (PENDING -> READY/FAILED).
class ObAiColumnWorkerTask : public common::ObTimerTask
{
public:
  ObAiColumnWorkerTask();
  virtual ~ObAiColumnWorkerTask();

  int init();
  int start(common::ObTimer &timer);
  void stop(common::ObTimer &timer);
  void destroy();

  virtual void runTimerTask() override;

private:
  // lifecycle of one cached per-column result of an in-flight group
  enum MemTaskStatus
  {
    MEM_TASK_RUNNING = 0,  // a failed attempt waits for its next retry
    MEM_TASK_READY = 1,    // validated model result waits for the group publish
    MEM_TASK_FAILED = 2,   // terminal failure (retry budget exhausted)
  };

  // The in-memory result of one AI column of one in-flight group.  A
  // succeeded member is kept here so that a later retry of another member of
  // the same request does not recompute it (a model call must happen at most
  // once per attempt).
  struct MemTask
  {
    MemTask() : column_id_(0), status_(MEM_TASK_RUNNING), def_version_(0),
                is_null_(true), result_()
    {}
    uint64_t column_id_;
    MemTaskStatus status_;
    int64_t def_version_;
    bool is_null_;             // publish the result column as NULL
    common::ObString result_;  // escaped SQL literal, allocated on the group allocator
  };

  // One in-flight (table, request) group.  Lives in the in-flight list until
  // the group is published or dropped.  All strings are deep copied into the
  // per-group arena so that they survive across ticks.
  struct MemGroup : public common::ObDLinkBase<MemGroup>
  {
    MemGroup()
      : table_id_(0), request_id_(), rowkey_predicate_(), tasks_(),
        processing_(false), alloc_(ObModIds::OMT)
    {
    }
    ~MemGroup() {}

    MemTask *find_task(const uint64_t column_id);
    int64_t terminal_count() const;

    uint64_t table_id_;
    common::ObString request_id_;
    common::ObString rowkey_predicate_;
    common::ObArray<MemTask> tasks_;
    bool processing_;  // owned by a running runTimerTask invocation
    common::ObArenaAllocator alloc_;  // keep last: strings of this group live here
  };

  // One row of the poll on __all_ai_column_task.
  struct PolledTask
  {
    PolledTask() : table_id_(0), rowkey_hash_(0), column_id_(0), state_(0),
                   retry_count_(0), max_retries_(0), def_version_(0),
                   rowkey_predicate_(), request_id_()
    {}
    uint64_t table_id_;
    uint64_t rowkey_hash_;
    uint64_t column_id_;
    int64_t state_;
    int64_t retry_count_;
    int64_t max_retries_;
    int64_t def_version_;
    common::ObString rowkey_predicate_;  // deep copied into the tick allocator
    common::ObString request_id_;
  };

  // db/table names of one user table, resolved with a runtime schema guard
  struct TableName
  {
    TableName() : db_name_(), table_name_() {}
    common::ObString db_name_;
    common::ObString table_name_;
  };

private:
  int run_once_();
  // first-tick crash recovery: EXECUTING -> PENDING
  int recover_executing_tasks_();
  int poll_tasks_(common::ObIAllocator &alloc, common::ObIArray<PolledTask> &tasks);
  int process_group_(common::ObIAllocator &tick_alloc, common::ObIArray<PolledTask *> &tasks);
  // reads the task rows a group currently owns (state EXECUTING)
  int load_group_tasks_(common::ObIAllocator &alloc, const uint64_t table_id,
                        const common::ObString &request_id,
                        common::ObIArray<PolledTask> &tasks);

  // CAS of the group's task rows into EXECUTING in one transaction
  int claim_group_(const uint64_t table_id, const common::ObString &request_id,
                   int64_t &affected);
  // number of task rows still present for a (table, request) group
  int count_group_task_rows_(const uint64_t table_id, const common::ObString &request_id,
                             int64_t &count);
  // db/table names and schema version of a user table
  int resolve_table_meta_(common::ObIAllocator &alloc, const uint64_t table_id,
                          bool &table_exists, int64_t &schema_version, TableName &names);
  // evaluates one task of the group; a failed model attempt is recorded in
  // the in-memory group (retry or terminal failure), row_deleted tells the
  // caller that the user row (and thus the whole group) is gone
  int execute_task_(common::ObIAllocator &tick_alloc, MemGroup &group, PolledTask &task,
                    const TableName &names, common::ObIArray<ObAIColumnItem> &items,
                    bool &row_deleted);
  // reads the current prompt of one row: SELECT <user_expr> FROM <db>.<table> WHERE <pred>
  int read_user_prompt_(common::ObIAllocator &alloc, const TableName &names,
                        const ObAIColumnItem &item, const common::ObString &rowkey_predicate,
                        int64_t &row_count, common::ObString &prompt);
  // calls the completion / embedding endpoint of one AI column
  int call_model_(common::ObIAllocator &alloc, const ObAIColumnItem &item,
                  const common::ObString &user_prompt, common::ObString &output);
  // cheap TCP reachability probe of the endpoint URL so that an offline
  // endpoint fails an attempt without waiting for the http client retries
  static bool probe_endpoint_reachable_(const common::ObString &url);
  // validates the model output and builds the escaped SET literal
  int validate_and_build_literal_(common::ObIAllocator &alloc, const ObAIColumnItem &item,
                                  const common::ObString &output, bool &valid,
                                  common::ObString &literal);
  // failed attempt of a task that still has retry budget: state -> RETRY_WAIT
  int defer_retry_(PolledTask &task);
  // publishes all terminal results of a group to the user row and removes the
  // task rows; affected == 0 means the request was replaced (stale guard)
  int publish_group_(MemGroup &group, const TableName &names,
                     common::ObIArray<ObAIColumnItem> &items, int64_t &affected);
  // puts claimed rows of a group back into PENDING for a later tick
  int requeue_group_(const uint64_t table_id, const common::ObString &request_id);
  int delete_group_tasks_(const uint64_t table_id, const common::ObString &request_id,
                          int64_t &affected);

  const ObAIColumnItem *find_item_(common::ObIArray<ObAIColumnItem> &items,
                                   const uint64_t column_id);

  MemGroup *find_mem_group_(const uint64_t table_id, const common::ObString &request_id);
  void free_mem_group_(MemGroup *group);  // caller holds mem_lock_

private:
  bool inited_;
  bool recovered_;  // first-tick crash recovery done
  common::ObDList<MemGroup> mem_groups_;
  lib::ObMutex mem_lock_;  // protects mem_groups_ and its nodes

private:
  DISALLOW_COPY_AND_ASSIGN(ObAiColumnWorkerTask);
};

// Server module wrapper of the AI column worker timer.
class ObAiColumnWorkerService
{
public:
  ObAiColumnWorkerService();
  virtual ~ObAiColumnWorkerService();

  static int server_module_init(ObAiColumnWorkerService *&service);

  int init();
  int start();
  void stop();
  void wait();
  void destroy();

private:
  bool inited_;
  common::ObTimer timer_;
  ObAiColumnWorkerTask task_;

private:
  DISALLOW_COPY_AND_ASSIGN(ObAiColumnWorkerService);
};

}  // end namespace share
}  // end namespace oceanbase

#endif  // OCEANBASE_OBSERVER_AI_COLUMN_OB_AI_COLUMN_WORKER_H_
