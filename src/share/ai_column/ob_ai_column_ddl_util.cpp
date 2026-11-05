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

#include "share/ai_column/ob_ai_column_ddl_util.h"

#include "common/json_type/ob_json_parse.h"
#include "common/json_type/ob_json_tree.h"
#include "common/mysqlclient/ob_isql_client.h"
#include "common/object/ob_object.h"
#include "lib/allocator/page_arena.h"
#include "lib/ob_define.h"
#include "share/ai_column/ob_ai_column_util.h"
#include "share/inner_table/ob_inner_table_schema.h"
#include "share/ob_dml_sql_splicer.h"
#include "share/schema/ob_column_schema.h"
#include "share/schema/ob_table_schema.h"

namespace oceanbase
{
namespace share
{

static const char *OB_AI_DDL_STATE_PREFIX = "__ai_state_";
static const char *OB_AI_DDL_REQUEST_PREFIX = "__ai_request_";
static const char *OB_AI_DDL_RESULT_SPEC_PREFIX = "__ai_result_spec_";

// the parsed fields of one AI column definition (persisted as the JSON
// comment of the hidden state column)
struct AIColumnDefParts
{
  AIColumnDefParts()
    : func_type_(OB_AI_COL_FUNC_COMPLETE), model_name_(), sync_mode_(),
      max_retries_(0), def_version_(0), dep_column_ids_(),
      user_expr_(), normalized_expr_(), result_type_(), vector_dim_(0)
  {}
  void reset()
  {
    func_type_ = OB_AI_COL_FUNC_COMPLETE;
    model_name_.reset();
    sync_mode_.reset();
    max_retries_ = 0;
    def_version_ = 0;
    dep_column_ids_.reset();
    user_expr_.reset();
    normalized_expr_.reset();
    result_type_.reset();
    vector_dim_ = 0;
  }
  int64_t func_type_;
  common::ObString model_name_;
  common::ObString sync_mode_;
  int64_t max_retries_;
  int64_t def_version_;
  common::ObString dep_column_ids_;
  common::ObString user_expr_;
  common::ObString normalized_expr_;
  common::ObString result_type_;
  int64_t vector_dim_;
};

static int get_def_json_str_field_(common::ObJsonObject &root,
                                   const char *key,
                                   common::ObString &value)
{
  int ret = OB_SUCCESS;
  common::ObJsonNode *node = root.get_value(common::ObString(key));
  if (nullptr == node) {
    // missing key: keep the default (empty) value
  } else if (common::ObJsonNodeType::J_STRING != node->json_type()) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("ai column def json field is not a string", K(ret), K(key));
  } else {
    value.assign_ptr(node->get_data(),
                     static_cast<common::ObString::obstr_size_t>(node->get_data_length()));
  }
  return ret;
}

static int get_def_json_int_field_(common::ObJsonObject &root,
                                   const char *key,
                                   int64_t &value)
{
  int ret = OB_SUCCESS;
  common::ObJsonNode *node = root.get_value(common::ObString(key));
  if (nullptr == node) {
    // missing key: keep the default value
  } else if (common::ObJsonNodeType::J_INT == node->json_type()) {
    value = node->get_int();
  } else if (common::ObJsonNodeType::J_UINT == node->json_type()) {
    value = static_cast<int64_t>(node->get_uint());
  } else {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("ai column def json field is not an integer", K(ret), K(key));
  }
  return ret;
}

static int parse_ai_column_def_json_(common::ObIAllocator &allocator,
                                     const common::ObString &def_json,
                                     AIColumnDefParts &parts)
{
  int ret = OB_SUCCESS;
  common::ObJsonNode *node = nullptr;
  parts.reset();
  if (OB_FAIL(common::ObJsonParser::get_tree(&allocator, def_json.ptr(), def_json.length(), node))) {
    LOG_WARN("parse ai column def json failed", K(ret), K(def_json));
  } else if (OB_ISNULL(node) || common::ObJsonNodeType::J_OBJECT != node->json_type()) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("ai column def json is not a json object", K(ret), K(def_json));
  } else {
    common::ObJsonObject &root = *static_cast<common::ObJsonObject *>(node);
    if (OB_FAIL(get_def_json_int_field_(root, OB_AI_DEF_FUNC, parts.func_type_))) {
      LOG_WARN("parse func failed", K(ret));
    } else if (OB_FAIL(get_def_json_str_field_(root, OB_AI_DEF_MODEL, parts.model_name_))) {
      LOG_WARN("parse model failed", K(ret));
    } else if (OB_FAIL(get_def_json_str_field_(root, OB_AI_DEF_SYNC, parts.sync_mode_))) {
      LOG_WARN("parse sync failed", K(ret));
    } else if (OB_FAIL(get_def_json_int_field_(root, OB_AI_DEF_MAX_RETRIES, parts.max_retries_))) {
      LOG_WARN("parse max_retries failed", K(ret));
    } else if (OB_FAIL(get_def_json_int_field_(root, OB_AI_DEF_VERSION, parts.def_version_))) {
      LOG_WARN("parse ver failed", K(ret));
    } else if (OB_FAIL(get_def_json_str_field_(root, OB_AI_DEF_DEPS, parts.dep_column_ids_))) {
      LOG_WARN("parse deps failed", K(ret));
    } else if (OB_FAIL(get_def_json_str_field_(root, OB_AI_DEF_USER_EXPR, parts.user_expr_))) {
      LOG_WARN("parse expr failed", K(ret));
    } else if (OB_FAIL(get_def_json_str_field_(root, OB_AI_DEF_NORM_EXPR, parts.normalized_expr_))) {
      LOG_WARN("parse norm failed", K(ret));
    } else if (OB_FAIL(get_def_json_str_field_(root, OB_AI_DEF_RESULT_TYPE, parts.result_type_))) {
      LOG_WARN("parse res_type failed", K(ret));
    } else if (OB_FAIL(get_def_json_int_field_(root, OB_AI_DEF_VECTOR_DIM, parts.vector_dim_))) {
      LOG_WARN("parse vec_dim failed", K(ret));
    }
  }
  return ret;
}

static const schema::ObColumnSchemaV2 *find_column_by_name_(
    const schema::ObTableSchema &table_schema,
    const common::ObString &column_name)
{
  const schema::ObColumnSchemaV2 *found = nullptr;
  const int64_t column_count = table_schema.get_column_count();
  for (int64_t i = 0; nullptr == found && i < column_count; ++i) {
    const schema::ObColumnSchemaV2 *column = table_schema.get_column_schema_by_idx(i);
    if (nullptr != column && column->get_column_name_str() == column_name) {
      found = column;
    }
  }
  return found;
}

static int push_unique_name_(common::ObSEArray<common::ObString, 4> &names,
                             const common::ObString &name)
{
  int ret = OB_SUCCESS;
  bool found = false;
  for (int64_t i = 0; !found && i < names.count(); ++i) {
    found = (names.at(i) == name);
  }
  if (!found && OB_FAIL(names.push_back(name))) {
    LOG_WARN("push back ai column name failed", K(ret), K(name));
  }
  return ret;
}

// Collects the names of the AI result columns of a table.  An AI column is
// discoverable from its hidden state column "__ai_state_<result name>" (the
// resolver always creates it); the AI_COLUMN_FLAG on the result column itself
// is accepted as an alternative marker for robustness.
static int collect_ai_result_column_names_(
    const schema::ObTableSchema &table_schema,
    common::ObSEArray<common::ObString, 4> &names)
{
  int ret = OB_SUCCESS;
  const int64_t state_prefix_len = strlen(OB_AI_DDL_STATE_PREFIX);
  const int64_t column_count = table_schema.get_column_count();
  for (int64_t i = 0; OB_SUCC(ret) && i < column_count; ++i) {
    const schema::ObColumnSchemaV2 *column = table_schema.get_column_schema_by_idx(i);
    if (OB_ISNULL(column)) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("column schema is null", K(ret), K(i));
    } else if (column->is_ai_state_column()) {
      const common::ObString &state_name = column->get_column_name_str();
      if (state_name.length() > state_prefix_len
          && OB_FAIL(push_unique_name_(names,
                 common::ObString(static_cast<common::ObString::obstr_size_t>(
                                      state_name.length() - state_prefix_len),
                                  state_name.ptr() + state_prefix_len)))) {
        LOG_WARN("collect ai column name failed", K(ret), K(state_name));
      }
    } else if (column->is_ai_column()) {
      if (OB_FAIL(push_unique_name_(names, column->get_column_name_str()))) {
        LOG_WARN("collect ai column name failed", K(ret),
                 K(column->get_column_name_str()));
      }
    }
  }
  return ret;
}

static int build_hidden_column_name_(const char *prefix,
                                     const common::ObString &column_name,
                                     char *buf,
                                     const int64_t buf_len)
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(build_ai_hidden_column_name(prefix, strlen(prefix), column_name, buf, buf_len))) {
    LOG_WARN("build ai hidden column name failed", K(ret), K(prefix), K(column_name));
  }
  return ret;
}

// Writes the __all_ai_column row of one AI result column.  The three hidden
// columns and the final column ids are always resolved from %table_schema.
static int write_ai_column_row_(common::ObISQLClient &sql_client,
                                const uint64_t table_id,
                                const schema::ObTableSchema &table_schema,
                                const schema::ObColumnSchemaV2 &result_column)
{
  int ret = OB_SUCCESS;
  const common::ObString result_name = result_column.get_column_name_str();
  // must match the resolver's hidden-name buffer size (OB_MAX_COLUMN_NAME_LENGTH)
  // so both sides truncate long column names identically
  char buf[OB_MAX_COLUMN_NAME_LENGTH];
  const schema::ObColumnSchemaV2 *state_column = nullptr;
  const schema::ObColumnSchemaV2 *request_column = nullptr;
  const schema::ObColumnSchemaV2 *result_spec_column = nullptr;
  if (OB_FAIL(build_hidden_column_name_(OB_AI_DDL_STATE_PREFIX, result_name, buf, sizeof(buf)))) {
  } else if (OB_ISNULL(state_column = find_column_by_name_(table_schema, common::ObString(buf)))) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("ai state column not found in table schema", K(ret), K(table_id), K(result_name), K(buf));
  } else if (OB_FAIL(build_hidden_column_name_(OB_AI_DDL_REQUEST_PREFIX, result_name, buf, sizeof(buf)))) {
  } else if (OB_ISNULL(request_column = find_column_by_name_(table_schema, common::ObString(buf)))) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("ai request column not found in table schema", K(ret), K(table_id), K(result_name), K(buf));
  } else if (OB_FAIL(build_hidden_column_name_(OB_AI_DDL_RESULT_SPEC_PREFIX, result_name, buf, sizeof(buf)))) {
  } else if (OB_ISNULL(result_spec_column = find_column_by_name_(table_schema, common::ObString(buf)))) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("ai result spec column not found in table schema", K(ret), K(table_id), K(result_name), K(buf));
  }

  const char *comment = (nullptr != state_column) ? state_column->get_comment() : nullptr;
  // the parsed strings point into this allocator's memory, so it must stay
  // alive until the row has been spliced below
  common::ObArenaAllocator allocator(ObModIds::OB_SCHEMA_GETTER_GUARD);
  AIColumnDefParts parts;
  if (OB_FAIL(ret)) {
  } else if (nullptr == comment || '\0' == comment[0]) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("ai state column comment does not contain the def json",
             K(ret), K(table_id), K(result_name));
  } else if (OB_FAIL(parse_ai_column_def_json_(allocator, common::ObString(comment), parts))) {
    LOG_WARN("parse ai column def json failed", K(ret), K(table_id), K(result_name));
  }
  if (OB_SUCC(ret)) {
    ObDMLSqlSplicer splicer;
    ObSqlString sql;
    int64_t affected_rows = 0;
    if (OB_FAIL(splicer.add_uint64_pk_column("table_id", table_id))) {
      LOG_WARN("add table_id failed", K(ret));
    } else if (OB_FAIL(splicer.add_uint64_pk_column("column_id", result_column.get_column_id()))) {
      LOG_WARN("add column_id failed", K(ret));
    } else if (OB_FAIL(splicer.add_column("column_name", common::ObHexEscapeSqlStr(result_name)))) {
      LOG_WARN("add column_name failed", K(ret));
    } else if (OB_FAIL(splicer.add_column("func_type", parts.func_type_))) {
      LOG_WARN("add func_type failed", K(ret));
    } else if (OB_FAIL(splicer.add_column("model_name", common::ObHexEscapeSqlStr(parts.model_name_)))) {
      LOG_WARN("add model_name failed", K(ret));
    } else if (OB_FAIL(splicer.add_column("sync_mode", common::ObHexEscapeSqlStr(parts.sync_mode_)))) {
      LOG_WARN("add sync_mode failed", K(ret));
    } else if (OB_FAIL(splicer.add_column("max_retries", parts.max_retries_))) {
      LOG_WARN("add max_retries failed", K(ret));
    } else if (OB_FAIL(splicer.add_column("def_version", parts.def_version_))) {
      LOG_WARN("add def_version failed", K(ret));
    } else if (OB_FAIL(splicer.add_column("user_expr", common::ObHexEscapeSqlStr(parts.user_expr_)))) {
      LOG_WARN("add user_expr failed", K(ret));
    } else if (OB_FAIL(splicer.add_column("normalized_expr", common::ObHexEscapeSqlStr(parts.normalized_expr_)))) {
      LOG_WARN("add normalized_expr failed", K(ret));
    } else if (OB_FAIL(splicer.add_column("dep_column_ids", common::ObHexEscapeSqlStr(parts.dep_column_ids_)))) {
      LOG_WARN("add dep_column_ids failed", K(ret));
    } else if (OB_FAIL(splicer.add_column("result_type", common::ObHexEscapeSqlStr(parts.result_type_)))) {
      LOG_WARN("add result_type failed", K(ret));
    } else if (OB_FAIL(splicer.add_column("vector_dim", parts.vector_dim_))) {
      LOG_WARN("add vector_dim failed", K(ret));
    } else if (OB_FAIL(splicer.add_uint64_column("state_column_id", state_column->get_column_id()))) {
      LOG_WARN("add state_column_id failed", K(ret));
    } else if (OB_FAIL(splicer.add_uint64_column("request_column_id", request_column->get_column_id()))) {
      LOG_WARN("add request_column_id failed", K(ret));
    } else if (OB_FAIL(splicer.add_uint64_column("result_spec_column_id", result_spec_column->get_column_id()))) {
      LOG_WARN("add result_spec_column_id failed", K(ret));
    } else if (OB_FAIL(splicer.splice_insert_sql(OB_ALL_AI_COLUMN_TNAME, sql))) {
      LOG_WARN("splice insert sql failed", K(ret));
    } else if (OB_FAIL(sql_client.write(sql.ptr(), affected_rows))) {
      LOG_WARN("insert ai column row failed", K(ret), K(sql.ptr()), K(table_id), K(result_name));
    } else {
      LOG_TRACE("insert ai column row", K(table_id), K(result_name), K(affected_rows));
    }
  }
  return ret;
}

bool ObAIColumnDDLUtil::has_ai_column(const schema::ObTableSchema &table_schema)
{
  bool has_ai = false;
  const int64_t column_count = table_schema.get_column_count();
  for (int64_t i = 0; !has_ai && i < column_count; ++i) {
    const schema::ObColumnSchemaV2 *column = table_schema.get_column_schema_by_idx(i);
    has_ai = (nullptr != column
              && (column->is_ai_column() || column->is_ai_state_column()));
  }
  return has_ai;
}

int ObAIColumnDDLUtil::insert_ai_column_rows(common::ObISQLClient &sql_client,
                                             const uint64_t table_id,
                                             const schema::ObTableSchema &table_schema)
{
  int ret = OB_SUCCESS;
  common::ObSEArray<common::ObString, 4> result_names;
  if (OB_FAIL(collect_ai_result_column_names_(table_schema, result_names))) {
    LOG_WARN("collect ai result column names failed", K(ret), K(table_id));
  }
  for (int64_t i = 0; OB_SUCC(ret) && i < result_names.count(); ++i) {
    const schema::ObColumnSchemaV2 *column = find_column_by_name_(table_schema,
                                                                  result_names.at(i));
    if (OB_ISNULL(column)) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("ai result column not found in table schema", K(ret), K(table_id),
               K(result_names.at(i)));
    } else if (OB_FAIL(write_ai_column_row_(sql_client, table_id, table_schema, *column))) {
      LOG_WARN("write ai column row failed", K(ret), K(table_id), K(result_names.at(i)));
    }
  }
  return ret;
}

int ObAIColumnDDLUtil::delete_ai_column_rows(common::ObISQLClient &sql_client,
                                             const uint64_t table_id)
{
  int ret = OB_SUCCESS;
  ObSqlString sql;
  int64_t affected_rows = 0;
  if (OB_FAIL(sql.assign_fmt("DELETE FROM %s WHERE table_id = %lu",
                             OB_ALL_AI_COLUMN_TNAME, table_id))) {
    LOG_WARN("build delete ai column sql failed", K(ret));
  } else if (OB_FAIL(sql_client.write(sql.ptr(), affected_rows))) {
    LOG_WARN("delete ai column rows failed", K(ret), K(sql.ptr()), K(table_id));
  } else if (OB_FAIL(sql.assign_fmt("DELETE FROM %s WHERE table_id = %lu",
                                    OB_ALL_AI_COLUMN_TASK_TNAME, table_id))) {
    LOG_WARN("build delete ai column task sql failed", K(ret));
  } else if (OB_FAIL(sql_client.write(sql.ptr(), affected_rows))) {
    LOG_WARN("delete ai column task rows failed", K(ret), K(sql.ptr()), K(table_id));
  }
  return ret;
}

int ObAIColumnDDLUtil::insert_ai_column_row_for_add_column(common::ObISQLClient &sql_client,
                                                           const uint64_t table_id,
                                                           const schema::ObTableSchema &table_schema,
                                                           const common::ObString &ai_column_name)
{
  int ret = OB_SUCCESS;
  const schema::ObColumnSchemaV2 *column = find_column_by_name_(table_schema, ai_column_name);
  if (nullptr == column) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("the added ai column is not in the table schema",
             K(ret), K(table_id), K(ai_column_name));
  } else if (OB_FAIL(write_ai_column_row_(sql_client, table_id, table_schema, *column))) {
    LOG_WARN("write ai column row failed", K(ret), K(table_id), K(ai_column_name));
  }
  return ret;
}

}  // end namespace share
}  // end namespace oceanbase
