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

#ifndef OCEANBASE_SQL_RESOLVER_DDL_OB_AI_COLUMN_RESOLVER_H_
#define OCEANBASE_SQL_RESOLVER_DDL_OB_AI_COLUMN_RESOLVER_H_

#include "sql/resolver/ddl/ob_ddl_resolver.h"

namespace oceanbase
{
namespace sql
{

// Resolves the "AI COLUMN (func(args)) [WITH (...)]" column attribute:
// validates the definition (function, model, vector dimension, dependencies)
// and creates the three hidden per-row columns (state / request / result
// spec) whose state column comment carries the definition JSON.
class ObAIColumnResolver
{
public:
  // @param column       the AI result column (name and data type resolved)
  // @param ai_attr_node the T_CONSTR_AI_COLUMN parse node
  // @param type_node    the column data type parse node (never NULL here)
  // @param stmt         the CREATE TABLE / ALTER TABLE stmt being resolved
  // @param table_schema the table schema so far (mutable on the create path;
  //                     on the alter add-column path it is the original table
  //                     and hidden columns are appended to the stmt instead)
  static int resolve_ai_column_attribute(
      common::ObIAllocator &alloc,
      ObStmt &stmt,
      ObSchemaChecker *schema_checker,
      ObColumnSchemaV2 &column,
      ParseNode &ai_attr_node,
      const ParseNode *type_node,
      ObTableSchema &table_schema,
      common::ObIArray<ObColumnSchemaV2> *pending_create_cols);

private:
  static int parse_options(common::ObIAllocator &alloc, ParseNode *options_node,
                           common::ObString &sync_mode, int64_t &max_retries);
  static int check_model_exists(ObSchemaChecker &schema_checker,
                                const common::ObString &model_name);
  static int collect_column_refs(const ParseNode *node,
                                 common::ObIArray<common::ObString> &names);
  static int check_dependencies(ObTableSchema &table_schema,
                                const ObColumnSchemaV2 &column,
                                const ParseNode &expr_node);
  static int build_system_prompt(int64_t obj_type, char *buf, const int64_t buf_len,
                                 bool &has_prompt);
  static int build_result_type_str(int64_t obj_type, char *buf, const int64_t buf_len);
  static int build_normalized_expr(common::ObIAllocator &alloc, int64_t func_type,
                                   const common::ObString &model_name,
                                   const common::ObString &user_expr,
                                   int64_t embed_dim,
                                   const char *system_prompt,
                                   common::ObString &normalized_expr);
  static int build_def_json(common::ObIAllocator &alloc, int64_t func_type,
                            const common::ObString &model_name,
                            const common::ObString &sync_mode, int64_t max_retries,
                            int64_t def_version,
                            const common::ObIArray<uint64_t> &dep_ids,
                            const common::ObString &user_expr,
                            const common::ObString &normalized_expr,
                            const char *result_type, int64_t vec_dim,
                            common::ObString &def_json);
  static int create_hidden_columns(common::ObIAllocator &alloc,
                                   ObStmt &stmt,
                                   ObColumnSchemaV2 &column,
                                   ObTableSchema &table_schema,
                                   const common::ObString &def_json,
                                   bool on_create_path,
                                   common::ObIArray<ObColumnSchemaV2> *pending_create_cols);
public:
  // CREATE TABLE post pass: assigns column ids to the hidden columns resolved
  // earlier and adds them to the table schema.  Must run before
  // ObTableSchema::set_max_used_column_id() fixes the id watermark.
  static int add_pending_hidden_columns(uint64_t &column_id_seed,
                                        ObTableSchema &table_schema,
                                        common::ObIArray<ObColumnSchemaV2> &pending);
};

}  // end namespace sql
}  // end namespace oceanbase

#endif  // OCEANBASE_SQL_RESOLVER_DDL_OB_AI_COLUMN_RESOLVER_H_
