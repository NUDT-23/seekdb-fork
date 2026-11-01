/*
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
#include "ob_expr_ai_embed.h"
#include "query/ai/ob_ai_endpoint_resolver.h"
#include "share/rc/ob_server_runtime.h"

using namespace oceanbase::common;
using namespace oceanbase::sql;

namespace oceanbase 
{
namespace sql 
{
ObExprAIEmbed::ObExprAIEmbed(common::ObIAllocator &alloc)
    : ObFuncExprOperator(alloc, 
                        T_FUN_SYS_AI_EMBED, 
                        N_AI_EMBED, 
                        MORE_THAN_ZERO,
                        NOT_VALID_FOR_GENERATED_COL, 
                        NOT_ROW_DIMENSION) 
{
}

ObExprAIEmbed::~ObExprAIEmbed() 
{
}

int ObExprAIEmbed::calc_result_typeN(ObExprResType &type,
                                     ObExprResType *types_stack,
                                     int64_t param_num,
                                     common::ObExprTypeCtx &type_ctx) const 
{
  UNUSED(type_ctx);
  UNUSED(types_stack);
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(param_num > 3 || param_num < 2)) {
    ObString func_name_(get_name());
    ret = OB_ERR_PARAM_SIZE;
    LOG_USER_ERROR(OB_ERR_PARAM_SIZE, func_name_.length(), func_name_.ptr());
  } else {
    types_stack[MODEL_IDX].set_calc_type(ObVarcharType);
    types_stack[MODEL_IDX].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
    types_stack[CONTENT_IDX].set_calc_type(ObVarcharType);
    types_stack[CONTENT_IDX].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
    if (param_num == 3) {
      if (ob_is_integer_type(types_stack[DIM_IDX].get_type())) {
        types_stack[DIM_IDX].set_calc_type(ObIntType);
        types_stack[DIM_IDX].set_precision(10);
        types_stack[DIM_IDX].set_scale(0);
      } else {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, dimension parameter must be an integer, not a decimal or float");
      }
    }
    type.set_varchar();
    type.set_collation_type(CS_TYPE_UTF8MB4_BIN);
    type.set_collation_level(CS_LEVEL_COERCIBLE);
  }
  return ret;
}

int ObExprAIEmbed::eval_ai_embed(const ObExpr &expr, ObEvalCtx &ctx, ObDatum &res) 
{
  INIT_SUCC(ret);
  ObDatum *arg_model_id = nullptr;
  ObDatum *arg_content = nullptr;
  ObDatum *arg_dim = nullptr;
  if (expr.arg_cnt_ == 3 ? OB_FAIL(expr.eval_param_value(ctx, arg_model_id, arg_content, arg_dim))
                         : OB_FAIL(expr.eval_param_value(ctx, arg_model_id, arg_content))) {
  } else if (arg_model_id->is_null()) {
    ret = OB_INVALID_ARGUMENT;
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, model id is null");
    res.set_null();
  } else if (arg_content->is_null()) {
    // a null input produces a null embedding, the row is kept in the result and
    // is not sent to the model service
    res.set_null();
  } else {
    ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
    
    MultimodeAlloctor temp_allocator(tmp_alloc_g.get_allocator());
    lib::ObMallocHookAttrGuard malloc_guard(lib::ObMemAttr(N_AI_EMBED));
    ObAIFuncExprInfo *info = nullptr;
    share::ObAiModelEndpointInfo resolved_endpoint;
    const share::ObAiModelEndpointInfo *endpoint_info = &resolved_endpoint;
    query::ObIAiEndpointResolver *endpoint_resolver =
        ::oceanbase::share::server_service<::oceanbase::query::ObIAiEndpointResolver>();
    ObString model_id = arg_model_id->get_string();
    ObString content = arg_content->get_string();
    if (model_id.empty() || content.empty()) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, model id or input is empty");
      res.set_null();
    }
    int64_t dim = 0;
    ObJsonInt *dim_json = nullptr;
    ObJsonObject *config = nullptr;
    if (OB_FAIL(ret)) {
    } else if (OB_NOT_NULL(arg_dim) && !arg_dim->is_null()) {
      dim = arg_dim->get_int();
      if (dim <= 0) {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, dimension parameter must be a positive integer");
        res.set_null();
      } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(temp_allocator, config))) {
      } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_int(temp_allocator, dim, dim_json))) {
      } else if (OB_FAIL(config->add("dimensions", dim_json))) {
      }
    }
    if (OB_FAIL(ret)) {
    } else if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(temp_allocator, model_id, info))) {
    } else if (OB_ISNULL(endpoint_resolver)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(endpoint_resolver->resolve_by_model_name(
                   model_id, temp_allocator, resolved_endpoint))) {
    } else {
      ObAIFuncModel model(temp_allocator, *info, *endpoint_info);
      ObString result;
      if (OB_FAIL(model.call_dense_embedding(content, config, result))) {
      } else if (OB_FAIL(ObAIFuncUtils::set_string_result(expr, ctx, res, result))) {
      }
    }
  }
  return ret;
}

int ObExprAIEmbed::eval_embed_chunk(ObIAllocator &allocator,
                                    const ObString &model_id,
                                    const int64_t dimension,
                                    ObArray<ObString> &contents,
                                    const ObIArray<int64_t> &rows,
                                    const ObExpr &expr,
                                    ObEvalCtx &ctx)
{
  INIT_SUCC(ret);
  ObDatum *res_datums = expr.locate_batch_datums(ctx);
  ObAIFuncExprInfo *info = nullptr;
  share::ObAiModelEndpointInfo resolved_endpoint;
  query::ObIAiEndpointResolver *endpoint_resolver =
      ::oceanbase::share::server_service<::oceanbase::query::ObIAiEndpointResolver>();
  ObJsonObject *config = nullptr;
  ObJsonInt *dim_json = nullptr;
  ObArray<ObString> results;
  if (OB_ISNULL(res_datums) || rows.count() != contents.count()) {
    ret = OB_ERR_UNEXPECTED;
  } else if (contents.empty()) {
    // nothing to do
  } else if (dimension > 0) {
    // the requested dimension is a parameter of the request, not of a single input
    if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(allocator, config))) {
    } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_int(allocator, dimension, dim_json))) {
    } else if (OB_FAIL(config->add("dimensions", dim_json))) {
    }
  }
  if (OB_FAIL(ret)) {
  } else if (contents.empty()) {
  } else if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(allocator, model_id, info))) {
  } else if (OB_ISNULL(info)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_ISNULL(endpoint_resolver)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(endpoint_resolver->resolve_by_model_name(model_id, allocator, resolved_endpoint))) {
  } else {
    ObAIFuncModel model(allocator, *info, resolved_endpoint);
    if (OB_FAIL(model.call_dense_embedding_vector_v2(contents, config, results))) {
    } else if (results.count() != rows.count()) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("unexpected number of embeddings returned by the service", K(ret),
               K(results.count()), K(rows.count()));
    } else {
      for (int64_t i = 0; OB_SUCC(ret) && i < rows.count(); i++) {
        const int64_t row = rows.at(i);
        const ObString &value = results.at(i);
        char *buf = nullptr;
        if (OB_ISNULL(buf = expr.get_str_res_mem(ctx, value.length(), row))) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
        } else {
          if (value.length() > 0) {
            MEMCPY(buf, value.ptr(), value.length());
          }
          res_datums[row].set_string(buf, static_cast<uint32_t>(value.length()));
        }
      }
    }
  }
  return ret;
}

int ObExprAIEmbed::eval_ai_embed_batch(const ObExpr &expr, ObEvalCtx &ctx,
                                       const ObBitVector &skip, const int64_t size)
{
  INIT_SUCC(ret);
  ObDatumVector model_vector;
  ObDatumVector content_vector;
  ObDatumVector dim_vector;
  ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
  MultimodeAlloctor tmp_allocator(tmp_alloc_g.get_allocator());
  lib::ObMallocHookAttrGuard malloc_guard(lib::ObMemAttr(N_AI_EMBED));
  ObDatum *res_datums = expr.locate_batch_datums(ctx);
  // rows waiting for the answer of the request being built
  ObSEArray<int64_t, 32> pending_rows;
  ObArray<ObString> pending_contents;
  ObString pending_model_id;
  int64_t pending_dimension = 0;
  const bool has_dimension = static_cast<int64_t>(expr.arg_cnt_) > DIM_IDX;
  if (OB_ISNULL(res_datums)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(expr.eval_batch_param_value(ctx, skip, size, model_vector, content_vector, dim_vector))) {
  } else {
    for (int64_t row = 0; OB_SUCC(ret) && row < size; row++) {
      ObDatum *model_datum = model_vector.at(row);
      ObDatum *content_datum = content_vector.at(row);
      ObDatum *dim_datum = has_dimension ? dim_vector.at(row) : nullptr;
      if (skip.at(row)) {
        // the row is not evaluated in this batch
        continue;
      } else if (OB_ISNULL(model_datum) || OB_ISNULL(content_datum)) {
        ret = OB_ERR_UNEXPECTED;
      } else if (model_datum->is_null()) {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, model id is null");
        res_datums[row].set_null();
      } else if (content_datum->is_null()) {
        // a null input is never sent to the service and produces a null result
        res_datums[row].set_null();
        continue;
      } else {
        const ObString model_id = model_datum->get_string();
        int64_t dimension = 0;
        if (OB_NOT_NULL(dim_datum) && !dim_datum->is_null()) {
          dimension = dim_datum->get_int();
        }
        if (dimension <= 0 && OB_NOT_NULL(dim_datum) && !dim_datum->is_null()) {
          ret = OB_INVALID_ARGUMENT;
          LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, dimension parameter must be a positive integer");
          res_datums[row].set_null();
        } else if (!pending_rows.empty()
                   && (model_id != pending_model_id || dimension != pending_dimension)) {
          // a request carries the inputs of a single model with a single dimension
          if (OB_FAIL(eval_embed_chunk(*tmp_allocator, pending_model_id, pending_dimension,
                                       pending_contents, pending_rows, expr, ctx))) {
          } else {
            pending_rows.reuse();
            pending_contents.reuse();
            pending_model_id.reset();
            pending_dimension = 0;
          }
        }
        if (OB_SUCC(ret) && pending_rows.empty()) {
          pending_model_id = model_id;
          pending_dimension = dimension;
        }
        if (OB_SUCC(ret)) {
          if (OB_FAIL(pending_contents.push_back(content_datum->get_string()))) {
          } else if (OB_FAIL(pending_rows.push_back(row))) {
          }
        }
        if (OB_SUCC(ret) && pending_rows.count() >= MAX_BATCH_INPUTS) {
          // one request carries at most MAX_BATCH_INPUTS inputs, the following rows
          // are sent by the next request
          if (OB_FAIL(eval_embed_chunk(*tmp_allocator, pending_model_id, pending_dimension,
                                       pending_contents, pending_rows, expr, ctx))) {
          } else {
            pending_rows.reuse();
            pending_contents.reuse();
            pending_model_id.reset();
            pending_dimension = 0;
          }
        }
      }
    }
    if (OB_SUCC(ret) && !pending_rows.empty()) {
      if (OB_FAIL(eval_embed_chunk(*tmp_allocator, pending_model_id, pending_dimension,
                                   pending_contents, pending_rows, expr, ctx))) {
      }
    }
  }
  return ret;
}

int ObExprAIEmbed::cg_expr(ObExprCGCtx &expr_cg_ctx, 
                           const ObRawExpr &raw_expr,
                           ObExpr &rt_expr) const 
{
  int ret = OB_SUCCESS;
  // TODO: support schema version match in plan cache for ai func
  // const ObRawExpr *model_key = raw_expr.get_param_expr(0);
  // if (OB_NOT_NULL(model_key)
  //     && (model_key->is_static_scalar_const_expr() || model_key->is_const_expr())
  //     && model_key->get_expr_type() != T_OP_GET_USER_VAR &&
  //     OB_NOT_NULL(expr_cg_ctx.schema_guard_)) {
  //   ObIAllocator *allocator = expr_cg_ctx.allocator_;
  //   ObExecContext *exec_ctx = expr_cg_ctx.session_->get_cur_exec_ctx();
  //   ObObj const_data;
  //   bool got_data = false;
  //   ObAIFuncExprInfo *info = nullptr;
  //   if (OB_ISNULL(allocator)) {
  //     ret = OB_ERR_UNEXPECTED;
  //     LOG_WARN("allocator is null", K(ret));
  //   } else if (OB_FAIL(ObSQLUtils::calc_const_or_calculable_expr(exec_ctx,
  //                                                         model_key,
  //                                                         const_data,
  //                                                         got_data,
  //                                                         *allocator))) {
  //     LOG_WARN("failed to calc offset expr", K(ret));
  //   } else if (!got_data || const_data.is_null()) {
  //   } else if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(*allocator, const_data.get_string(), *expr_cg_ctx.schema_guard_, info))) {
  //     LOG_WARN("failed to get ai func info", K(ret), K(const_data.get_string()));
  //   } else {
  //     rt_expr.extra_info_ = info;
  //   }
  // }

  if (OB_SUCC(ret)) {
    rt_expr.eval_func_ = ObExprAIEmbed::eval_ai_embed;
    // when the operator evaluates a whole batch of rows the inputs are sent
    // together instead of one request per row
    rt_expr.eval_batch_func_ = ObExprAIEmbed::eval_ai_embed_batch;
  }
  return ret;
}

} // namespace sql
} // namespace oceanbase
