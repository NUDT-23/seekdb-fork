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

#ifndef OCEANBASE_SQL_ENGINE_DML_OB_AI_COLUMN_DML_HELPER_H_
#define OCEANBASE_SQL_ENGINE_DML_OB_AI_COLUMN_DML_HELPER_H_

#include "share/ai_column/ob_ai_column_util.h"
#include "sql/engine/dml/ob_dml_ctx_define.h"

namespace oceanbase
{
namespace sql
{

class ObTableModifyOp;

// DML linkage hook of the AI COLUMN feature (赛题三).
//
// After the user rows of a DML statement on a table with AI columns have been
// flushed, this helper runs (in the same transaction, through nested SQL on
// the inner connection, exactly like the foreign key cascade does):
//   INSERT : invalidate the AI result columns of each inserted row and create
//            one pending task per AI column in __all_ai_column_task.
//   UPDATE : invalidate and re-create tasks only when the primary key changed
//            or one of the dependency columns of an AI column was assigned.
//   DELETE : remove the pending tasks of each deleted row.
//
// All work is transactional: when the user statement rolls back the state
// column writes and the task rows roll back with it.
class ObAIColumnDMLHelper
{
public:
  // Process the after-row list of one DML operator.  Returns immediately when
  // the target table has no AI column (hot path for ordinary tables).
  static int process_after_rows(ObTableModifyOp &op, ObDMLModifyRowsList &rows);

  // Decide at operator open time whether the target table of this DML has AI
  // columns; the result drives whether rows are pushed to the after-row list.
  static int check_table_has_ai_columns(ObTableModifyOp &op, bool &has_ai_columns);
};

}  // namespace sql
}  // namespace oceanbase

#endif  // OCEANBASE_SQL_ENGINE_DML_OB_AI_COLUMN_DML_HELPER_H_
