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

#ifndef OCEANBASE_SHARE_AI_COLUMN_OB_AI_COLUMN_DDL_UTIL_H_
#define OCEANBASE_SHARE_AI_COLUMN_OB_AI_COLUMN_DDL_UTIL_H_

#include "lib/string/ob_string.h"

namespace oceanbase
{
namespace common
{
class ObISQLClient;
}
namespace share
{
namespace schema
{
class ObTableSchema;
}

// DDL side helper that keeps the persistent __all_ai_column /
// __all_ai_column_task system tables in sync with the AI columns of a user
// table.  All the writes run inside the caller's DDL transaction (the
// ObISQLClient is expected to be the DDL ObMySQLTransaction), so every
// failure simply aborts the surrounding DDL.
class ObAIColumnDDLUtil
{
public:
  // Cheap pre-check: whether the table schema contains any AI result column.
  static bool has_ai_column(const schema::ObTableSchema &table_schema);

  // Writes one __all_ai_column row for every AI column of the table schema.
  // The final column ids are always read from %table_schema (they may have
  // been reassigned during DDL execution), never from the definition JSON.
  static int insert_ai_column_rows(common::ObISQLClient &sql_client,
                                   const uint64_t table_id,
                                   const schema::ObTableSchema &table_schema);

  // Removes all AI column rows and pending tasks of the dropped table.
  static int delete_ai_column_rows(common::ObISQLClient &sql_client,
                                   const uint64_t table_id);

  // Same as insert_ai_column_rows but only writes the row of the single AI
  // column named %ai_column_name (used by ALTER TABLE ADD COLUMN where the
  // other columns of the table were already handled).
  static int insert_ai_column_row_for_add_column(common::ObISQLClient &sql_client,
                                                 const uint64_t table_id,
                                                 const schema::ObTableSchema &table_schema,
                                                 const common::ObString &ai_column_name);
};

}  // end namespace share
}  // end namespace oceanbase

#endif  // OCEANBASE_SHARE_AI_COLUMN_OB_AI_COLUMN_DDL_UTIL_H_
