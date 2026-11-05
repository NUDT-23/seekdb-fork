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

#define USING_LOG_PREFIX SQL_RESV

#include "sql/resolver/ddl/ob_ai_column_resolver.h"

#include "common/json_type/ob_json_base.h"
#include "common/json_type/ob_json_tree.h"
#include "share/ai_column/ob_ai_column_util.h"
#include "share/schema/ob_ai_model_schema.h"
#include "share/schema/ob_schema_struct.h"
#include "share/schema/ob_table_schema.h"
#include "sql/parser/parse_node.h"
#include "sql/resolver/ddl/ob_alter_table_stmt.h"
#include "sql/resolver/ddl/ob_create_table_stmt.h"
#include "sql/resolver/ob_schema_checker.h"

namespace oceanbase
{
namespace sql
{

using namespace common;
using namespace share;

static const char *OB_AI_STATE_COLUMN_PREFIX = "__ai_state_";
static const char *OB_AI_REQUEST_COLUMN_PREFIX = "__ai_request_";
static const char *OB_AI_RESULT_SPEC_COLUMN_PREFIX = "__ai_result_spec_";

static const char *OB_AI_PROMPT_JSON = "Return exactly one valid JSON value matching the result column type. ";
static const char *OB_AI_PROMPT_INTEGER = "Return exactly one base-10 integer matching the result column type. ";
static const char *OB_AI_PROMPT_NUMBER = "Return exactly one base-10 number matching the result column type. ";
static const char *OB_AI_PROMPT_DATE = "Return exactly one date in YYYY-MM-DD format. ";
static const char *OB_AI_PROMPT_DATETIME = "Return exactly one datetime in YYYY-MM-DD HH:MM:SS format. ";

static int json_escape_string(ObIAllocator &alloc, const ObString &src, ObString &dst)
{
  int ret = OB_SUCCESS;
  char *buf = nullptr;
  const int64_t buf_len = src.length() * 2 + 1;
  if (OB_ISNULL(buf = static_cast<char *>(alloc.alloc(buf_len)))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
    LOG_WARN("allocate escape buffer failed", K(ret));
  } else {
    int64_t pos = 0;
    for (int64_t i = 0; OB_SUCC(ret) && i < src.length(); ++i) {
      const char c = src.ptr()[i];
      if ('\\' == c || '"' == c) {
        buf[pos++] = '\\';
      }
      buf[pos++] = c;
    }
    if (OB_SUCC(ret)) {
      dst.assign_ptr(buf, static_cast<int32_t>(pos));
    }
  }
  return ret;
}

int ObAIColumnResolver::resolve_ai_column_attribute(
    ObIAllocator &alloc,
    ObSQLStmt &stmt,
    ObSchemaChecker *schema_checker,
    ObColumnSchemaV2 &column,
    ParseNode &ai_attr_node,
    const ParseNode *type_node,
    ObTableSchema &table_schema)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(schema_checker) || OB_ISNULL(schema_checker->get_schema_guard())) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("schema checker is null", K(ret));
  } else if (OB_ISNULL(ai_attr_node.children_) || ai_attr_node.num_child_ != 2
             || OB_ISNULL(ai_attr_node.children_[0])
             || T_CONSTR_AI_COLUMN_EXPR != ai_attr_node.children_[0]->type_) {
    ret = OB_ERR_PARSER_SYNTAX;
    LOG_WARN("invalid AI COLUMN attribute node", K(ret));
  } else {
    ParseNode *expr_node = ai_attr_node.children_[0];
    ParseNode *options_node = ai_attr_node.children_[1];
    const int64_t func_type = expr_node->value_;
    const ObString model_name(expr_node->children_[0]->str_len_,
                              expr_node->children_[0]->str_value_);
    const ObString user_expr(expr_node->children_[1]->str_len_,
                             expr_node->children_[1]->str_value_);
    const int64_t embed_dim = (nullptr != expr_node->children_[2]) ? expr_node->children_[2]->value_ : 0;
    ObString sync_mode("async");
    int64_t max_retries = 0;

    if (OB_FAIL(parse_options(alloc, options_node, sync_mode, max_retries))) {
      LOG_WARN("parse ai column options failed", K(ret));
    } else if (0 != sync_mode.case_compare("async")) {
      ret = OB_NOT_SUPPORTED;
      LOG_USER_ERROR(OB_NOT_SUPPORTED, "sync_mode other than async is");
    } else if (OB_UNLIKELY(max_retries < 0 || max_retries > 100)) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "max_retries");
    } else if (OB_FAIL(check_model_exists(*schema_checker, model_name))) {
      LOG_WARN("ai model does not exist", K(ret), K(model_name));
    }

    // data type checks
    const int64_t result_obj_type = column.get_meta_type().get_type();
    bool is_vector_col = false;
    int64_t vector_col_dim = 0;
    if (OB_SUCC(ret)) {
      if (nullptr != type_node && T_COLLECTION == type_node->type_ && 1 == type_node->int32_values_[0]) {
        is_vector_col = true;
        vector_col_dim = (nullptr != type_node->children_[0]) ? type_node->children_[0]->value_ : 0;
      }
      if (OB_AI_COL_FUNC_EMBED == func_type) {
        if (!is_vector_col) {
          ret = OB_ERR_INVALID_TYPE_FOR_ARGUMENT;
          LOG_USER_ERROR(OB_ERR_INVALID_TYPE_FOR_ARGUMENT, "AI_EMBED result column", "vector column");
        } else if (embed_dim > 0 && embed_dim != vector_col_dim) {
          ret = OB_ERR_AI_COLUMN_DIMENSION_MISMATCH;
          LOG_USER_ERROR(OB_ERR_AI_COLUMN_DIMENSION_MISMATCH, vector_col_dim, embed_dim);
        }
      } else if (OB_AI_COL_FUNC_COMPLETE == func_type) {
        if (is_vector_col) {
          ret = OB_ERR_INVALID_TYPE_FOR_ARGUMENT;
          LOG_USER_ERROR(OB_ERR_INVALID_TYPE_FOR_ARGUMENT, "AI_COMPLETE result column", "non-vector column");
        }
      } else {
        ret = OB_ERR_PARSER_SYNTAX;
        LOG_WARN("unsupported ai column function", K(ret), K(func_type));
      }
    }

    // dependency checks: the prompt expression may only reference existing
    // non-AI columns
    if (OB_SUCC(ret)) {
      if (OB_FAIL(check_dependencies(table_schema, column, *expr_node->children_[1]))) {
        LOG_WARN("check ai column dependencies failed", K(ret));
      }
    }

    char system_prompt[512] = {0};
    bool has_prompt = false;
    char result_type_str[64] = {0};
    ObString normalized_expr;
    ObString def_json;
    if (OB_SUCC(ret)) {
      if (OB_FAIL(build_system_prompt(result_obj_type, system_prompt, sizeof(system_prompt), has_prompt))) {
        LOG_WARN("build system prompt failed", K(ret));
      } else if (OB_FAIL(build_result_type_str(result_obj_type, result_type_str, sizeof(result_type_str)))) {
        LOG_WARN("build result type string failed", K(ret));
      } else if (OB_FAIL(build_normalized_expr(alloc, func_type, model_name, user_expr,
                                               embed_dim, has_prompt ? system_prompt : nullptr,
                                               normalized_expr))) {
        LOG_WARN("build normalized expr failed", K(ret));
      }
    }
    ObSEArray<uint64_t, 8> dep_ids;
    if (OB_SUCC(ret)) {
      ObSEArray<ObString, 8> col_names;
      if (OB_FAIL(collect_column_refs(expr_node->children_[1], col_names))) {
        LOG_WARN("collect column refs failed", K(ret));
      } else {
        for (int64_t i = 0; OB_SUCC(ret) && i < col_names.count(); ++i) {
          const ObColumnSchemaV2 *dep_col = table_schema.get_column_schema(col_names.at(i));
          if (OB_NOT_NULL(dep_col)) {
            if (OB_FAIL(dep_ids.push_back(dep_col->get_column_id()))) {
              LOG_WARN("push back dep id failed", K(ret));
            }
          }
          // references to unknown columns are reported by the worker when the
          // prompt expression is evaluated
        }
      }
    }
    if (OB_SUCC(ret)) {
      if (OB_FAIL(build_def_json(alloc, func_type, model_name, sync_mode, max_retries,
                                 1, dep_ids, user_expr, normalized_expr,
                                 result_type_str, is_vector_col ? vector_col_dim : 0,
                                 def_json))) {
        LOG_WARN("build def json failed", K(ret));
      } else {
        const bool on_create_path = (stmt::T_CREATE_TABLE == stmt.get_stmt_type());
        if (OB_FAIL(create_hidden_columns(alloc, stmt, column, table_schema, def_json,
                                          on_create_path))) {
          LOG_WARN("create ai hidden columns failed", K(ret));
        }
      }
    }
  }
  return ret;
}

int ObAIColumnResolver::parse_options(ObIAllocator &alloc, ParseNode *options_node,
                                      ObString &sync_mode, int64_t &max_retries)
{
  UNUSED(alloc);
  int ret = OB_SUCCESS;
  if (OB_ISNULL(options_node)) {
    // defaults
  } else if (T_CONSTR_AI_COLUMN_OPTS != options_node->type_) {
    ret = OB_ERR_PARSER_SYNTAX;
    LOG_WARN("invalid ai column options node", K(ret));
  } else {
    // the grammar merges the option chain into a flat T_OPTION_LIST whose
    // children alternate name, value, name, value ... (same layout as
    // T_VEC_INDEX_PARAMS)
    if (0 != options_node->num_child_ % 2) {
      ret = OB_ERR_PARSER_SYNTAX;
      LOG_WARN("invalid ai column option list", K(ret));
    }
    for (int64_t i = 0; OB_SUCC(ret) && i + 1 < options_node->num_child_; i += 2) {
      {
        ParseNode *name_node = options_node->children_[i];
        ParseNode *value_node = options_node->children_[i + 1];
        if (OB_ISNULL(name_node) || OB_ISNULL(value_node)) {
          ret = OB_ERR_PARSER_SYNTAX;
          LOG_WARN("invalid ai column option pair", K(ret));
          break;
        }
        const ObString opt_name(name_node->str_len_, name_node->str_value_);
        if (0 == opt_name.case_compare("sync_mode")) {
          if (OB_ISNULL(value_node) || 0 == value_node->str_len_) {
            ret = OB_ERR_PARSER_SYNTAX;
          } else {
            sync_mode.assign_ptr(value_node->str_value_, value_node->str_len_);
          }
        } else if (0 == opt_name.case_compare("max_retries")) {
          if (OB_ISNULL(value_node)) {
            ret = OB_ERR_PARSER_SYNTAX;
          } else {
            max_retries = value_node->value_;
          }
        } else {
          ret = OB_NOT_SUPPORTED;
          LOG_USER_ERROR(OB_NOT_SUPPORTED, "this AI COLUMN option is");
        }
      }
    }
  }
  return ret;
}

int ObAIColumnResolver::check_model_exists(ObSchemaChecker &schema_checker,
                                           const ObString &model_name)
{
  int ret = OB_SUCCESS;
  const ObAiModelSchema *ai_model_schema = nullptr;
  if (OB_FAIL(schema_checker.get_schema_guard()->get_ai_model_schema(model_name, ai_model_schema))) {
    LOG_WARN("get ai model schema failed", K(ret), K(model_name));
  } else if (OB_ISNULL(ai_model_schema)) {
    ret = OB_AI_FUNC_MODEL_NOT_FOUND;
    LOG_USER_ERROR(OB_AI_FUNC_MODEL_NOT_FOUND, model_name.length(), model_name.ptr());
  }
  return ret;
}

int ObAIColumnResolver::collect_column_refs(const ParseNode *node,
                                            ObIArray<ObString> &names)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(node)) {
  } else if (T_COLUMN_REF == node->type_ && 3 == node->num_child_) {
    // children: db, table, column
    ParseNode *col_node = node->children_[2];
    if (OB_NOT_NULL(col_node) && col_node->str_len_ > 0) {
      if (OB_FAIL(names.push_back(ObString(col_node->str_len_, col_node->str_value_)))) {
        LOG_WARN("push back column name failed", K(ret));
      }
    }
  } else if (node->children_ != nullptr) {
    for (int64_t i = 0; OB_SUCC(ret) && i < node->num_child_; ++i) {
      if (OB_FAIL(collect_column_refs(node->children_[i], names))) {
        LOG_WARN("collect column refs failed", K(ret));
      }
    }
  }
  return ret;
}

int ObAIColumnResolver::check_dependencies(ObTableSchema &table_schema,
                                           const ObColumnSchemaV2 &column,
                                           const ParseNode &expr_node)
{
  int ret = OB_SUCCESS;
  ObSEArray<ObString, 8> names;
  if (OB_FAIL(collect_column_refs(&expr_node, names))) {
    LOG_WARN("collect column refs failed", K(ret));
  } else {
    for (int64_t i = 0; OB_SUCC(ret) && i < names.count(); ++i) {
      const ObString &name = names.at(i);
      if (0 == name.case_compare(column.get_column_name_str())) {
        ret = OB_NOT_SUPPORTED;
        SQL_RESV_LOG(WARN, "ai column depends on itself", K(ret), K(name));
      } else {
        const ObColumnSchemaV2 *dep = table_schema.get_column_schema(name);
        if (OB_NOT_NULL(dep) && dep->is_ai_column()) {
          ret = OB_NOT_SUPPORTED;
          SQL_RESV_LOG(WARN, "ai column depends on another ai column", K(ret), K(name));
        }
      }
    }
  }
  return ret;
}

int ObAIColumnResolver::build_system_prompt(int64_t obj_type, char *buf,
                                            const int64_t buf_len, bool &has_prompt)
{
  int ret = OB_SUCCESS;
  has_prompt = true;
  const char *prompt = nullptr;
  switch (obj_type) {
    case ObJsonType:
      prompt = OB_AI_PROMPT_JSON;
      break;
    case ObTinyIntType:
    case ObSmallIntType:
    case ObMediumIntType:
    case ObInt32Type:
    case ObIntType:
    case ObBooleanType:
    case ObBitType:
      prompt = OB_AI_PROMPT_INTEGER;
      break;
    case ObFloatType:
    case ObDoubleType:
      prompt = OB_AI_PROMPT_NUMBER;
      break;
    case ObDateType:
      prompt = OB_AI_PROMPT_DATE;
      break;
    case ObDateTimeType:
    case ObTimestampType:
      prompt = OB_AI_PROMPT_DATETIME;
      break;
    default:
      // VARCHAR/CHAR/TEXT and everything else: no extra system prompt
      has_prompt = false;
      break;
  }
  if (OB_SUCC(ret) && has_prompt) {
    if (OB_FAIL(databuff_printf(buf, buf_len, "%s", prompt))) {
      LOG_WARN("format system prompt failed", K(ret));
    }
  }
  return ret;
}

int ObAIColumnResolver::build_result_type_str(int64_t obj_type, char *buf, const int64_t buf_len)
{
  int ret = OB_SUCCESS;
  const char *type_str = "VARCHAR";
  switch (obj_type) {
    case ObJsonType:
      type_str = "JSON";
      break;
    case ObTinyIntType:
      type_str = "TINYINT";
      break;
    case ObSmallIntType:
      type_str = "SMALLINT";
      break;
    case ObMediumIntType:
      type_str = "MEDIUMINT";
      break;
    case ObInt32Type:
      type_str = "INT";
      break;
    case ObIntType:
      type_str = "BIGINT";
      break;
    case ObBooleanType:
    case ObBitType:
      type_str = "BOOLEAN";
      break;
    case ObFloatType:
      type_str = "FLOAT";
      break;
    case ObDoubleType:
      type_str = "DOUBLE";
      break;
    case ObDateType:
      type_str = "DATE";
      break;
    case ObDateTimeType:
      type_str = "DATETIME";
      break;
    case ObTimestampType:
      type_str = "TIMESTAMP";
      break;
    default:
      type_str = "VARCHAR";
      break;
  }
  if (OB_FAIL(databuff_printf(buf, buf_len, "%s", type_str))) {
    LOG_WARN("format result type failed", K(ret));
  }
  return ret;
}

int ObAIColumnResolver::build_normalized_expr(ObIAllocator &alloc, int64_t func_type,
                                              const ObString &model_name,
                                              const ObString &user_expr,
                                              int64_t embed_dim,
                                              const char *system_prompt,
                                              ObString &normalized_expr)
{
  int ret = OB_SUCCESS;
  ObSqlString buf;
  if (OB_AI_COL_FUNC_EMBED == func_type) {
    if (embed_dim > 0) {
      if (OB_FAIL(buf.assign_fmt("AI_EMBED('%.*s', %.*s, %ld)",
                                 model_name.length(), model_name.ptr(),
                                 user_expr.length(), user_expr.ptr(), embed_dim))) {
        LOG_WARN("build embed normalized expr failed", K(ret));
      }
    } else if (OB_FAIL(buf.assign_fmt("AI_EMBED('%.*s', %.*s)",
                                      model_name.length(), model_name.ptr(),
                                      user_expr.length(), user_expr.ptr()))) {
      LOG_WARN("build embed normalized expr failed", K(ret));
    }
  } else if (nullptr != system_prompt) {
    if (OB_FAIL(buf.assign_fmt("AI_COMPLETE('%.*s', concat('%s', %.*s))",
                               model_name.length(), model_name.ptr(),
                               system_prompt,
                               user_expr.length(), user_expr.ptr()))) {
      LOG_WARN("build complete normalized expr failed", K(ret));
    }
  } else if (OB_FAIL(buf.assign_fmt("AI_COMPLETE('%.*s', %.*s)",
                                    model_name.length(), model_name.ptr(),
                                    user_expr.length(), user_expr.ptr()))) {
    LOG_WARN("build complete normalized expr failed", K(ret));
  }
  if (OB_SUCC(ret)) {
    const ObString &str = buf.string();
    if (OB_FAIL(ob_write_string(alloc, str, normalized_expr))) {
      LOG_WARN("write normalized expr failed", K(ret));
    }
  }
  return ret;
}

int ObAIColumnResolver::build_def_json(ObIAllocator &alloc, int64_t func_type,
                                       const ObString &model_name,
                                       const ObString &sync_mode, int64_t max_retries,
                                       int64_t def_version,
                                       const ObIArray<uint64_t> &dep_ids,
                                       const ObString &user_expr,
                                       const ObString &normalized_expr,
                                       const char *result_type, int64_t vec_dim,
                                       ObString &def_json)
{
  int ret = OB_SUCCESS;
  ObSqlString buf;
  ObString esc_model;
  ObString esc_sync;
  ObString esc_expr;
  ObString esc_norm;
  if (OB_FAIL(json_escape_string(alloc, model_name, esc_model))) {
  } else if (OB_FAIL(json_escape_string(alloc, sync_mode, esc_sync))) {
  } else if (OB_FAIL(json_escape_string(alloc, user_expr, esc_expr))) {
  } else if (OB_FAIL(json_escape_string(alloc, normalized_expr, esc_norm))) {
  } else if (OB_FAIL(buf.assign_fmt("{\"%s\":%ld,\"%s\":\"%.*s\",\"%s\":\"%.*s\",\"%s\":%ld,"
                                    "\"%s\":%ld,\"%s\":\"",
                                    OB_AI_DEF_FUNC, func_type,
                                    OB_AI_DEF_MODEL, esc_model.length(), esc_model.ptr(),
                                    OB_AI_DEF_SYNC, esc_sync.length(), esc_sync.ptr(),
                                    OB_AI_DEF_MAX_RETRIES, max_retries,
                                    OB_AI_DEF_VERSION, def_version,
                                    OB_AI_DEF_DEPS))) {
    LOG_WARN("build def json head failed", K(ret));
  } else {
    for (int64_t i = 0; OB_SUCC(ret) && i < dep_ids.count(); ++i) {
      if (OB_FAIL(buf.append_fmt("%s%lu", (0 == i ? "" : ","), dep_ids.at(i)))) {
        LOG_WARN("append dep id failed", K(ret));
      }
    }
    if (OB_SUCC(ret)) {
      if (OB_FAIL(buf.append_fmt("\",\"%s\":\"%.*s\",\"%s\":\"%.*s\",\"%s\":\"%s\",\"%s\":%ld}",
                                 OB_AI_DEF_USER_EXPR, esc_expr.length(), esc_expr.ptr(),
                                 OB_AI_DEF_NORM_EXPR, esc_norm.length(), esc_norm.ptr(),
                                 OB_AI_DEF_RESULT_TYPE, result_type,
                                 OB_AI_DEF_VECTOR_DIM, vec_dim))) {
        LOG_WARN("build def json tail failed", K(ret));
      }
    }
  }
  if (OB_SUCC(ret)) {
    const ObString &str = buf.string();
    if (OB_FAIL(ob_write_string(alloc, str, def_json))) {
      LOG_WARN("write def json failed", K(ret));
    }
  }
  return ret;
}

int ObAIColumnResolver::create_hidden_columns(ObIAllocator &alloc,
                                              ObSQLStmt &stmt,
                                              ObColumnSchemaV2 &column,
                                              ObTableSchema &table_schema,
                                              const ObString &def_json,
                                              bool on_create_path)
{
  UNUSED(alloc);
  int ret = OB_SUCCESS;
  char name_buf[OB_MAX_COLUMN_NAME_LENGTH] = {0};
  AlterColumnSchema hidden_cols[3];
  const char *prefixes[3] = {OB_AI_STATE_COLUMN_PREFIX, OB_AI_REQUEST_COLUMN_PREFIX,
                             OB_AI_RESULT_SPEC_COLUMN_PREFIX};
  const int64_t flag_bits[3] = {AI_STATE_COLUMN_FLAG, AI_REQUEST_COLUMN_FLAG,
                                AI_RESULT_SPEC_COLUMN_FLAG};
  const int64_t type_lens[3] = {16, 64, 256};
  ObString def_values[3];
  ObString state_pending(OB_AI_STATE_PENDING);
  ObString empty_str("");
  def_values[0] = state_pending;
  def_values[1] = empty_str;
  def_values[2] = empty_str;

  for (int64_t i = 0; OB_SUCC(ret) && i < 3; ++i) {
    AlterColumnSchema &hidden = hidden_cols[i];
    hidden.reset();
    if (OB_FAIL(build_ai_hidden_column_name(prefixes[i], strlen(prefixes[i]),
                                            column.get_column_name_str(),
                                            name_buf, sizeof(name_buf)))) {
      LOG_WARN("build hidden column name failed", K(ret));
    } else if (OB_FAIL(hidden.set_column_name(name_buf))) {
      LOG_WARN("set hidden column name failed", K(ret));
    } else {
      ObObjParam default_value;
      default_value.set_varchar(def_values[i].ptr(), static_cast<int32_t>(def_values[i].length()));
      default_value.set_collation_type(CS_TYPE_UTF8MB4_GENERAL_CI);
      hidden.set_rowkey_position(0);
      hidden.set_index_position(0);
      hidden.set_tbl_part_key_pos(0);
      hidden.set_table_id(table_schema.get_table_id());
      hidden.add_column_flag(flag_bits[i]);
      hidden.set_is_hidden(true);
      hidden.set_data_type(ObVarcharType);
      hidden.set_data_length(type_lens[i]);
      hidden.set_collation_type(CS_TYPE_UTF8MB4_GENERAL_CI);
      hidden.set_prev_column_id(UINT64_MAX);
      hidden.set_next_column_id(UINT64_MAX);
      hidden.set_nullable(false);
      if (OB_FAIL(hidden.set_orig_default_value(default_value))) {
        LOG_WARN("set orig default failed", K(ret));
      } else if (OB_FAIL(hidden.set_cur_default_value(default_value, false))) {
        LOG_WARN("set cur default failed", K(ret));
      } else if (0 == i && OB_FAIL(hidden.set_comment(def_json))) {
        LOG_WARN("set state column comment failed", K(ret));
      }
    }
    if (OB_SUCC(ret)) {
      if (on_create_path) {
        if (OB_FAIL(table_schema.add_column(hidden))) {
          LOG_WARN("add hidden column failed", K(ret), K(name_buf));
        }
      } else {
        AlterColumnSchema &alter_hidden = hidden;
        alter_hidden.alter_type_ = OB_DDL_ADD_COLUMN;
        ObAlterTableStmt *alter_stmt = static_cast<ObAlterTableStmt *>(&stmt);
        if (stmt::T_ALTER_TABLE != stmt.get_stmt_type() || OB_ISNULL(alter_stmt)) {
          ret = OB_ERR_UNEXPECTED;
          LOG_WARN("alter stmt expected", K(ret));
        } else if (OB_FAIL(alter_stmt->add_column(alter_hidden))) {
          LOG_WARN("add hidden column to alter stmt failed", K(ret), K(name_buf));
        }
      }
    }
  }
  return ret;
}

}  // end namespace sql
}  // end namespace oceanbase
