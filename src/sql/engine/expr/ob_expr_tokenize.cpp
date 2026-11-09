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

#include "sql/engine/expr/ob_expr_tokenize.h"

#include "lib/alloc/alloc_struct.h"
#include "lib/allocator/page_arena.h"
#include "lib/allocator/ob_malloc.h"
#include "lib/lock/ob_mutex.h"
#include "lib/charset/ob_charset.h"
#include "common/json_type/ob_json_base.h"
#include "common/json_type/ob_json_tree.h"
#include "lib/ob_errno.h"
#include "lib/oblog/ob_log_module.h"
#include "lib/string/ob_string.h"
#include "lib/utility/ob_macro_utils.h"
#include "object/ob_object.h"
#include "sql/resolver/ddl/ob_fts_index_builder_util.h"
#include "share/ob_json_access_utils.h"
#include "data_plane/fts/ob_fts_parser_helper.h"
#include "data_plane/fts/dict/ob_gen_dic_loader.h"
#include "data_plane/fts/ob_fts_parser_property.h"

#define USING_LOG_PREFIX SQL_ENG
#include "sql/engine/expr/ob_expr_json_func_helper.h" // file not self-contained, there're logs inside.

namespace oceanbase
{
namespace sql
{
namespace
{
// tokenize() is a pure function: the same (text, parser, properties, collation)
// always produces the same packed JSON binary. Workloads such as index builds
// and monitoring dashboards call it repeatedly on the same values, so keep a
// small process-wide LRU of the final packed binary and serve repeats from it.
// The cache is strictly best effort: any failure falls back to the uncached
// path, and entries are bounded in count and size.
class ObTokenizeResultCache
{
public:
  static ObTokenizeResultCache &instance()
  {
    static ObTokenizeResultCache cache;
    return cache;
  }

  // Copies the payload into the caller-provided buffer while holding the
  // mutex, so the returned ObString never aliases slot memory that a
  // concurrent eviction could free.
  int lookup(const ObString &key, uint64_t key_hash,
             char *value_buf, int64_t buf_len, ObString &value)
  {
    int ret = OB_SUCCESS;
    lib::ObMutexGuard guard(mutex_);
    Slot *slot = find(key, key_hash);
    if (OB_ISNULL(slot)) {
      ret = OB_ENTRY_NOT_EXIST;
    } else if (slot->value_len_ > buf_len) {
      // cannot happen when insert() enforces the same bound, stay safe anyway
      ret = OB_ENTRY_NOT_EXIST;
    } else {
      MEMCPY(value_buf, slot->value_, slot->value_len_);
      value.assign_ptr(value_buf, slot->value_len_);
      slot->last_used_ = ++clock_;
    }
    return ret;
  }

  int insert(const ObString &key, uint64_t key_hash, const ObString &value)
  {
    int ret = OB_SUCCESS;
    lib::ObMutexGuard guard(mutex_);
    Slot *slot = find(key, key_hash);
    if (OB_NOT_NULL(slot)) {
      // same key computed twice concurrently: keep the newer payload
      if (OB_FAIL(slot->set_value(value))) {
        slot->reset();
      } else {
        slot->last_used_ = ++clock_;
      }
    } else {
      slot = pick_victim();
      if (OB_ISNULL(slot)) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else if (OB_FAIL(slot->set_key(key, key_hash))) {
        slot->reset();
      } else if (OB_FAIL(slot->set_value(value))) {
        slot->reset();
      } else {
        slot->last_used_ = ++clock_;
      }
    }
    return ret;
  }

private:
  static const int64_t SLOT_CNT = 16;
  ObTokenizeResultCache() : mutex_(), clock_(0), slots_() {}
  struct Slot
  {
    Slot() : used_(false), last_used_(0), key_hash_(0), key_(nullptr),
             key_len_(0), value_(nullptr), value_len_(0) {}
    ~Slot() { reset(); }
    void reset()
    {
      if (OB_NOT_NULL(key_)) { ob_free(key_); }
      if (OB_NOT_NULL(value_)) { ob_free(value_); }
      used_ = false;
      last_used_ = 0;
      key_hash_ = 0;
      key_ = nullptr;
      key_len_ = 0;
      value_ = nullptr;
      value_len_ = 0;
    }
    static int set_buf(const ObString &src, char *&buf, int64_t &len)
    {
      int ret = OB_SUCCESS;
      const int64_t alloc_len = src.length() > 0 ? src.length() : 1;
      char *new_buf = static_cast<char *>(ob_malloc(alloc_len, ObMemAttr("TokCache")));
      if (OB_ISNULL(new_buf)) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else {
        MEMCPY(new_buf, src.ptr(), src.length());
        if (OB_NOT_NULL(buf)) { ob_free(buf); }
        buf = new_buf;
        len = src.length();
      }
      return ret;
    }
    int set_key(const ObString &key, uint64_t key_hash)
    {
      int ret = OB_SUCCESS;
      if (OB_FAIL(set_buf(key, key_, key_len_))) {
      } else {
        key_hash_ = key_hash;
        used_ = true;
      }
      return ret;
    }
    int set_value(const ObString &value) { return set_buf(value, value_, value_len_); }
    bool match(const ObString &key, uint64_t key_hash) const
    {
      return used_ && key_hash_ == key_hash && key_len_ == key.length()
             && 0 == MEMCMP(key_, key.ptr(), key_len_);
    }
    bool used_;
    int64_t last_used_;
    uint64_t key_hash_;
    char *key_;
    int64_t key_len_;
    char *value_;
    int64_t value_len_;
  };

  Slot *find(const ObString &key, uint64_t key_hash)
  {
    Slot *found = nullptr;
    for (int64_t i = 0; OB_ISNULL(found) && i < SLOT_CNT; ++i) {
      if (slots_[i].match(key, key_hash)) { found = &slots_[i]; }
    }
    return found;
  }

  Slot *pick_victim()
  {
    Slot *victim = &slots_[0];
    for (int64_t i = 1; i < SLOT_CNT; ++i) {
      if (slots_[i].last_used_ < victim->last_used_) { victim = &slots_[i]; }
    }
    victim->reset();
    return victim;
  }

  lib::ObMutex mutex_;
  int64_t clock_;
  Slot slots_[SLOT_CNT];
};

const int64_t TOKENIZE_CACHE_MAX_TEXT_LEN = 8192;
const int64_t TOKENIZE_CACHE_MAX_VALUE_LEN = 8192;

// Build the cache key from the raw argument datums. Only the two-argument and
// one-argument forms are cached: the optional third argument is a JSON value
// whose datum may be an out-of-row locator, and skipping it keeps the key
// always comparable. The key layout is text \0 parser \0 collation.
int tokenize_cache_build_key(const ObExpr &expr, ObEvalCtx &ctx,
                              char *buf, int64_t buf_len, ObString &key,
                              bool &cacheable)
{
  int ret = OB_SUCCESS;
  cacheable = false;
  ObDatum *text_datum = nullptr;
  ObDatum *parser_datum = nullptr;
  if (OB_UNLIKELY(expr.arg_cnt_ < 1 || expr.arg_cnt_ > 2)) {
    // three-argument form: skip cache
  } else if (OB_FAIL(expr.args_[0]->eval(ctx, text_datum))) {
    LOG_WARN("failed to eval fulltext datum", K(ret));
  } else if (OB_UNLIKELY(text_datum->is_null())) {
    // null fulltext is handled by the uncached path
  } else if (text_datum->get_string().length() > TOKENIZE_CACHE_MAX_TEXT_LEN) {
    // oversize input: skip cache
  } else {
    ObString parser_name;
    if (expr.arg_cnt_ == 2) {
      if (OB_FAIL(expr.args_[1]->eval(ctx, parser_datum))) {
        LOG_WARN("failed to eval parser datum", K(ret));
      } else if (OB_LIKELY(!parser_datum->is_null())) {
        parser_name = parser_datum->get_string();
      }
    }
    if (OB_FAIL(ret)) {
    } else if (parser_name.length() > 64) {
      // unreasonable parser name: skip cache
    } else {
      const int64_t coll = static_cast<int64_t>(expr.args_[0]->obj_meta_.get_collation_type());
      const ObString text = text_datum->get_string();
      int64_t pos = 0;
      if (OB_UNLIKELY(text.length() + parser_name.length() + 16 > buf_len)) {
        ret = OB_SIZE_OVERFLOW;
      } else {
        MEMCPY(buf + pos, text.ptr(), text.length()); pos += text.length();
        buf[pos++] = '\0';
        if (parser_name.length() > 0) {
          MEMCPY(buf + pos, parser_name.ptr(), parser_name.length());
          pos += parser_name.length();
        }
        buf[pos++] = '\0';
        MEMCPY(buf + pos, &coll, sizeof(coll)); pos += sizeof(coll);
        key.assign_ptr(buf, pos);
        cacheable = true;
      }
    }
  }
  return ret;
}
} // namespace

ObExprTokenize::ObExprTokenize(common::ObIAllocator &alloc)
    : ObStringExprOperator(alloc,
                           T_FUN_TOKENIZE,
                           N_TOKENIZE,
                           MORE_THAN_ZERO,
                           VALID_FOR_GENERATED_COL)
{
}

ObExprTokenize::~ObExprTokenize() {}

int ObExprTokenize::eval_tokenize(const ObExpr &expr, ObEvalCtx &ctx, ObDatum &expr_datum)
{
  int ret = OB_SUCCESS;

  ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
  common::ObArenaAllocator &temp_allocator = tmp_alloc_g.get_allocator();

  ObIJsonBase *json_result = nullptr;
  TokenizeParam param;

  // check param num, which is checked in ObExprOperator::calc_result_typeN.
  if (OB_UNLIKELY(expr.arg_cnt_ < 1 || expr.arg_cnt_ > 3)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    // tokenize() is a pure function of (text, parser, properties, collation),
    // so a repeated call can be served from the packed-binary cache instead of
    // re-tokenizing and re-serializing. Failures fall back to the uncached path.
    bool cache_hit = false;
    ObString packed_bin;
    char key_buf[TOKENIZE_CACHE_MAX_TEXT_LEN + 96];
    char value_buf[TOKENIZE_CACHE_MAX_VALUE_LEN];
    ObString key;
    bool cacheable = false;
    if (OB_FAIL(tokenize_cache_build_key(expr, ctx, key_buf, sizeof(key_buf), key, cacheable))) {
      // failed to evaluate arguments here; let the uncached path surface the error
      ret = OB_SUCCESS;
      cacheable = false;
    } else if (cacheable
               && OB_SUCC(ObTokenizeResultCache::instance().lookup(
                      key, key.hash(), value_buf, sizeof(value_buf), packed_bin))) {
      cache_hit = true;
    } else if (cacheable) {
      // cache miss is not an error; fill the cache on the uncached path below
      ret = OB_SUCCESS;
    }

    if (OB_FAIL(ret)) {
    } else if (cache_hit) {
      if (OB_FAIL(ObJsonExprHelper::pack_json_str_res(expr, ctx, expr_datum, packed_bin))) {
        LOG_WARN("failed to pack cached tokenize result", K(ret));
      }
    } else if (OB_FAIL(parse_param(expr, ctx, temp_allocator, param))) {
    } else if (OB_FAIL(tokenize_fulltext(param, param.output_mode_, temp_allocator, json_result))) {
    } else if (OB_ISNULL(json_result)) {
      expr_datum.set_null();
    } else if (OB_FAIL(json_result->get_raw_binary(packed_bin, &temp_allocator))) {
      LOG_WARN("failed to serialize tokenize result", K(ret));
    } else {
      if (cacheable && packed_bin.length() > 0
          && packed_bin.length() <= TOKENIZE_CACHE_MAX_VALUE_LEN) {
        int cache_ret = ObTokenizeResultCache::instance().insert(key, key.hash(), packed_bin);
        if (OB_FAIL(cache_ret)) {
          // best effort only
          ret = OB_SUCCESS;
        }
      }
      if (OB_FAIL(ObJsonExprHelper::pack_json_str_res(expr, ctx, expr_datum, packed_bin))) {
        LOG_WARN("failed to pack tokenize result", K(ret));
      }
    }
  }

  return ret;
}

int ObExprTokenize::tokenize_fulltext(const TokenizeParam &param,
                                      TokenizeParam::OUTPUT_MODE mode,
                                      ObIAllocator &allocator,
                                      ObIJsonBase *&result)
{
  int ret = OB_SUCCESS;
  storage::ObFTParseHelper tokenize_helper;
  const int64_t ft_word_bkt_cnt = MIN(MAX(param.fulltext_.length() / 2, 2), 997);
  int64_t doc_len = 0;
  storage::ObFTWordMap token_map;

  ObArenaAllocator tmp_parse_alloc(ObMemAttr("Tmp buffer"));

  if (TokenizeParam::OUTPUT_MODE::DEFAULT != mode && TokenizeParam::OUTPUT_MODE::ALL != mode) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(tokenize_helper.init(&allocator, param.parser_name_, param.properties_))) {
  } else if (OB_FAIL(token_map.create(ft_word_bkt_cnt, common::ObMemAttr("FTWordMap")))) {
  } else if (
      (0 != param.fulltext_.length())
      && OB_FAIL(tokenize_helper.segment(
                     param.meta_,
                     param.fulltext_.ptr(),
                     param.fulltext_.length(),
                     doc_len,
                     token_map))) {
  } else {
    switch (param.output_mode_) {
    case TokenizeParam::OUTPUT_MODE::DEFAULT: {
      if (OB_FAIL(tokenize_helper.make_token_array_json(token_map, result))) {
      } else {
        // pass
      }
      break;
    }
    case TokenizeParam::OUTPUT_MODE::ALL: {
      if (OB_FAIL(tokenize_helper.make_detail_json(token_map, doc_len, result))) {
      } else {
        // pass
      }
      break;
    }
    default:
      ret = OB_INVALID_ARGUMENT;
    }
  }
  return ret;
}

ObExprTokenize::TokenizeParam ::TokenizeParam()
  : allocator_(ObMemAttr("TokenizeParam")),
    parser_name_(ObString(OB_DEFAULT_FULLTEXT_PARSER_NAME)),
    meta_(),
    fulltext_(),
    output_mode_(OUTPUT_MODE::DEFAULT)
{
}

int ObExprTokenize::TokenizeParam::parse_json_param(const ObIJsonBase *obj)
{
  int ret = OB_SUCCESS;
  ObString str;
  ObIJsonBase *val;

  if (OB_UNLIKELY(nullptr == obj)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (ObJsonNodeType::J_OBJECT != obj->json_type()) {
    ret = OB_INVALID_ARGUMENT;
  } else if (obj->element_count() == 0) {
    // no data
  } else if (OB_FAIL(obj->get_object_value(0, str, val))) {
  } else if (0 == str.case_compare(CASE_INDICATOR_STR)) {
    if (ObJsonNodeType::J_STRING != val->json_type()) {
      ret = OB_INVALID_ARGUMENT;
    } else if (0 == ObString(val->get_data_length(), val->get_data()).case_compare("UPPER")) {
    } else if (0 == ObString(val->get_data_length(), val->get_data()).case_compare("LOWER")) {
    } else {
      ret = OB_INVALID_ARGUMENT;
    }
    ret = OB_NOT_SUPPORTED;
    LOG_USER_ERROR(OB_NOT_SUPPORTED, "case indentifier");
  } else if (0 == str.case_compare(OUTPUT_MODE_STR)) {
    if (ObJsonNodeType::J_STRING != val->json_type()) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "output mode should be string default or all");
    } else if (0 == ObString(val->get_data_length(), val->get_data()).case_compare("DEFAULT")) {
      output_mode_ = DEFAULT;
    } else if (0 == ObString(val->get_data_length(), val->get_data()).case_compare("ALL")) {
      output_mode_ = ALL;
    } else {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "output mode should be string default or all");
    }
  } else if (0 == str.case_compare(STOPWORDS_LIST_STR)) {
    ret = OB_NOT_SUPPORTED;
    LOG_USER_ERROR(OB_NOT_SUPPORTED, "stopwords");
  } else if (0 == str.case_compare(ADDITIONAL_ARGS_STR)) {
    if (ObJsonNodeType::J_ARRAY != val->json_type()) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "parser arguments");
    } else {
      ObString json_str;
      if (OB_FAIL(storage::ObFTParserJsonProps::tokenize_array_to_props_json(allocator_, val, json_str))) {
        ObSqlString message;
        message.append_fmt("format in %s form", ADDITIONAL_ARGS_STR);
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, message.ptr());
      } else {
        properties_ = json_str;
      }
    }
  } else {
    ret = OB_INVALID_ARGUMENT;
    ObSqlString message;
    message.append_fmt("config: %s", str.ptr());
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, message.ptr());
  }
  return ret;
}

int ObExprTokenize::parse_param(const ObExpr &expr,
                                ObEvalCtx &ctx,
                                common::ObArenaAllocator &allocator,
                                TokenizeParam &param)
{
  int ret = OB_SUCCESS;

  ObDatum *parser_params_datum;
  ObString raw_parser_name = ObString::make_string(OB_DEFAULT_FULLTEXT_PARSER_NAME);

  ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
  
  MultimodeAlloctor temp_allocator(tmp_alloc_g.get_allocator());

  if (OB_UNLIKELY(expr.arg_cnt_ < 1 || expr.arg_cnt_ > 3)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(parse_fulltext(expr, ctx, param))) {
  } else if (OB_FAIL(parse_parser_name(expr, ctx, param))) {
  } else if (OB_FAIL(parse_parser_properties(expr, ctx, temp_allocator, param))) {
  } else if (OB_FAIL(param.reform_parser_properties(param.properties_))) {
  } else if (OB_FAIL(param.try_load_dictionary_for_ik())) {
  }
  return ret;
}

int ObExprTokenize::construct_ft_parser_inner_name(const ObString &input_str, TokenizeParam &param)
{
  int ret = OB_SUCCESS;
  storage::ObFTParser parser;

  char *parser_name_buf = nullptr;
  if (OB_ISNULL(parser_name_buf
                = static_cast<char *>(param.allocator_.alloc(storage::OB_FT_PARSER_NAME_LENGTH)))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else if (OB_FAIL(parser.init(input_str))) {
  } else if (OB_FAIL(parser.serialize_to_str(parser_name_buf, storage::OB_FT_PARSER_NAME_LENGTH))) {
  } else {
    param.parser_name_ = ObString::make_string(parser_name_buf);
  }
  return ret;
}

int ObExprTokenize::calc_result_typeN(ObExprResType &type,
                                      ObExprResType *types,
                                      int64_t param_num,
                                      common::ObExprTypeCtx &type_ctx) const
{
  int ret = OB_SUCCESS;

  if (OB_UNLIKELY(param_num < 1 || param_num > 3)) {
    ret = OB_ERR_PARAM_SIZE;
    ObString expr_name(N_TOKENIZE);
    LOG_USER_ERROR(OB_ERR_PARAM_SIZE, expr_name.length(), expr_name.ptr());
  } else {
    // just okay
  }

  ObLength length = ObAccuracy::DDL_DEFAULT_ACCURACY[ObJsonType].get_length();

  if (OB_SUCC(ret)) {
    // set res type
    type.set_json();
    type.set_length(length); // keep consistent with other json expr, maybe calc it later.

    // param type set, skip charset after first param
    for (int64_t i = 1; OB_SUCC(ret) && i < param_num; ++i) {
      if (ob_is_string_type(types[i].get_type())) {
        if (types[i].get_charset_type() != CHARSET_UTF8MB4) {
          types[i].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
        }
      }
    }

    // handle param
    if (param_num >= 2) {
      types[1].set_varchar();
    }

    if (param_num >= 3) {
      if (OB_FAIL(ObJsonExprHelper::is_valid_for_json(types, 2, N_TOKENIZE))) {
      }
    }
  }

  return ret;
}

int ObExprTokenize::cg_expr(ObExprCGCtx &op_cg_ctx,
                            const ObRawExpr &raw_expr,
                            ObExpr &rt_expr) const
{
  int ret = OB_SUCCESS;
  CK((rt_expr.arg_cnt_ >= 1 && rt_expr.arg_cnt_ <= 3));
  if (OB_SUCC(ret)) {
    // do register
    rt_expr.eval_func_ = eval_tokenize;
  }
  return ret;
}

int ObExprTokenize::parse_fulltext(const ObExpr &expr, ObEvalCtx &ctx, TokenizeParam &param)
{
  int ret = OB_SUCCESS;

  ObDatum *fulltext_datum;

  if (OB_FAIL(expr.args_[0]->eval(ctx, fulltext_datum))) {
  } else {
    if (fulltext_datum->is_null()) {
      // do nothing, return empty result
      param.fulltext_ = ObString::make_empty_string();
    } else {
      param.fulltext_ = fulltext_datum->get_string();
    }
    param.meta_.set_varchar(); // as we hardcoded in fts_index
    param.meta_.set_collation_type(expr.args_[0]->obj_meta_.get_collation_type());
  }
  return ret;
}

int ObExprTokenize::parse_parser_name(const ObExpr &expr, ObEvalCtx &ctx, TokenizeParam &param)
{
  int ret = OB_SUCCESS;
  ObDatum *parser_datum = nullptr;
  ObString raw_parser_name;

  if (expr.arg_cnt_ < 2) {
    raw_parser_name = ObString::make_string(OB_DEFAULT_FULLTEXT_PARSER_NAME);
  } else if (OB_FAIL(expr.args_[1]->eval(ctx, parser_datum))) {
  } else {
    if (parser_datum->is_null()) {
      raw_parser_name = ObString::make_string(OB_DEFAULT_FULLTEXT_PARSER_NAME);
    } else {
      ObString name = parser_datum->get_string();
      raw_parser_name = name.trim();
    }
  }

  if (OB_FAIL(ret)) {
    // already logged
  } else if (OB_FAIL(construct_ft_parser_inner_name(raw_parser_name, param))) {
  }

  return ret;
}

int ObExprTokenize::parse_parser_properties(const ObExpr &expr,
                                            ObEvalCtx &ctx,
                                            MultimodeAlloctor &mm_alloc,
                                            TokenizeParam &param)
{
  int ret = OB_SUCCESS;
  ObIJsonBase *base = nullptr;

  if (expr.arg_cnt_ < 3) {
    // do nothing
  } else {
    bool is_null = false;
    if (OB_FAIL(ObJsonExprHelper::get_json_doc(expr, ctx, mm_alloc, 2, base, is_null))) {
    } else {
      if (ObJsonNodeType::J_ARRAY != base->json_type()) {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "parser args should be in an array.");
      } else {
        for (uint64_t i = 0; OB_SUCC(ret) && i < base->element_count(); ++i) {
          ObIJsonBase *node = nullptr;
          if (OB_FAIL(base->get_array_element(i, node))) {
          } else if (ObJsonNodeType::J_OBJECT != (node->json_type())) {
            ret = OB_INVALID_ARGUMENT;
          } else if (OB_FAIL(param.parse_json_param(node))) {
          }
        } // for
      }
    }
  }

  return ret;
}

int ObExprTokenize::TokenizeParam::reform_parser_properties(const ObString &properties)
{
  int ret = OB_SUCCESS;
  storage::ObFTParserJsonProps parser_properties;

  if (OB_FAIL(parser_properties.init())) {
  } else if (OB_FAIL(parser_properties.parse_from_valid_str(properties))) {
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "parser properties invalid.");
  } else if (OB_FAIL(parser_properties.rebuild_props_for_ddl(parser_name_,
                                                             ObCollationType::CS_TYPE_UTF8MB4_BIN,
                                                             true))) {
  } else if (OB_FAIL(parser_properties.to_format_json(allocator_, properties_))) {
  }

  return ret;
}

int ObExprTokenize::TokenizeParam::try_load_dictionary_for_ik()
{
  int ret = OB_SUCCESS;
  bool need_to_load_dic = false;
  storage::ObDicLoaderHandle dic_loader_handle;
  if (OB_FAIL(share::ObFtsIndexBuilderUtil::check_need_to_load_dic(
          parser_name_, need_to_load_dic))) {
  } else if (need_to_load_dic) {
    if (OB_FAIL(storage::ObGenDicLoader::get_instance().get_dic_loader(
                    ObString::make_string(storage::ObFTSLiteral::PARSER_NAME_IK), // currently only ik, use parser_name_ without version suffix
                    ObCharset::charset_type_by_coll(meta_.get_collation_type()),
                    dic_loader_handle))) {
    } else if (OB_UNLIKELY(!dic_loader_handle.is_valid())) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(dic_loader_handle.get_loader()->try_load_dictionary_in_trans())) {
    }
  }
  return ret;
}

} // namespace sql
} // namespace oceanbase
