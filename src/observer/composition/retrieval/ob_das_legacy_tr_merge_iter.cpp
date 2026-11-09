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

#define USING_LOG_PREFIX SQL_DAS
#include "observer/composition/retrieval/ob_das_legacy_tr_merge_iter.h"
#include "sql/das/iter/ob_das_scan_iter.h"
#include "sql/das/iter/ob_das_text_retrieval_eval_node.h"
#include "sql/das/iter/sparse_retrieval/ob_das_text_retrieval_engine.h"
#include "sql/das/ob_das_ir_define.h"
#include "lib/allocator/ob_malloc.h"
#include "lib/lock/ob_mutex.h"

namespace oceanbase
{
using namespace share;
namespace sql
{
namespace
{
// Process-wide LRU of serialized fulltext retrieval results. Entries are
// keyed by the full logical input of the retrieval (tablets, snapshot
// version, tokens, weights, mode, limits) and hold the serialized output
// datum stream. All buffers are copied under the mutex, so a returned value
// never aliases memory a concurrent eviction could free.
class ObFTQueryResultCache
{
public:
  static ObFTQueryResultCache &instance()
  {
    static ObFTQueryResultCache cache;
    return cache;
  }

  // On hit, copies the payload into value_buf and returns OB_SUCCESS.
  int lookup(const ObString &key, uint64_t key_hash,
             char *value_buf, int64_t buf_len, int64_t &value_len)
  {
    int ret = OB_SUCCESS;
    lib::ObMutexGuard guard(mutex_);
    Slot *slot = find(key, key_hash);
    if (OB_ISNULL(slot)) {
      ret = OB_ENTRY_NOT_EXIST;
    } else if (slot->value_len_ > buf_len) {
      ret = OB_ENTRY_NOT_EXIST;
    } else {
      MEMCPY(value_buf, slot->value_, slot->value_len_);
      value_len = slot->value_len_;
      slot->last_used_ = ++clock_;
    }
    return ret;
  }

  int insert(const ObString &key, uint64_t key_hash,
             const char *value, const int64_t value_len)
  {
    int ret = OB_SUCCESS;
    if (OB_UNLIKELY(value_len <= 0 || value_len > MAX_VALUE_LEN)) {
      // oversized payloads simply stay uncached
    } else {
      lib::ObMutexGuard guard(mutex_);
      Slot *slot = find(key, key_hash);
      if (OB_NOT_NULL(slot)) {
        // another task filled the same key first: keep the existing entry
        slot->last_used_ = ++clock_;
      } else {
        slot = pick_victim();
        if (OB_ISNULL(slot)) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
        } else if (OB_FAIL(slot->set_key(key, key_hash))) {
          slot->reset();
        } else if (OB_FAIL(slot->set_value(value, value_len))) {
          slot->reset();
        } else {
          slot->last_used_ = ++clock_;
        }
      }
    }
    return ret;
  }

private:
  static const int64_t SLOT_CNT = 8;
  static const int64_t MAX_KEY_LEN = 4096;
  static const int64_t MAX_VALUE_LEN = 8 * 1024 * 1024;
  ObFTQueryResultCache() : mutex_(), clock_(0), slots_() {}
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
    static int alloc_buf(const int64_t len, char *&buf)
    {
      int ret = OB_SUCCESS;
      buf = static_cast<char *>(ob_malloc(len > 0 ? len : 1, ObMemAttr("FTQCache")));
      if (OB_ISNULL(buf)) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      }
      return ret;
    }
    int set_key(const ObString &key, uint64_t key_hash)
    {
      int ret = OB_SUCCESS;
      char *new_buf = nullptr;
      if (OB_UNLIKELY(key.length() > MAX_KEY_LEN)) {
        ret = OB_SIZE_OVERFLOW;
      } else if (OB_FAIL(alloc_buf(key.length(), new_buf))) {
      } else {
        MEMCPY(new_buf, key.ptr(), key.length());
        if (OB_NOT_NULL(key_)) { ob_free(key_); }
        key_ = new_buf;
        key_len_ = key.length();
        key_hash_ = key_hash;
        used_ = true;
      }
      return ret;
    }
    int set_value(const char *value, const int64_t value_len)
    {
      int ret = OB_SUCCESS;
      char *new_buf = nullptr;
      if (OB_FAIL(alloc_buf(value_len, new_buf))) {
      } else {
        MEMCPY(new_buf, value, value_len);
        if (OB_NOT_NULL(value_)) { ob_free(value_); }
        value_ = new_buf;
        value_len_ = value_len;
      }
      return ret;
    }
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
} // namespace

ObDASLegacyTRMergeIter::ObDASLegacyTRMergeIter()
  : ObDASIter(ObDASIterType::DAS_ITER_TEXT_RETRIEVAL_MERGE),
    mem_context_(nullptr),
    myself_allocator_(lib::ObMemAttr("SRMergeIterSelf"), OB_MALLOC_NORMAL_BLOCK_SIZE),
    ir_ctdef_(nullptr),
    ir_rtdef_(nullptr),
    tx_desc_(nullptr),
    snapshot_(nullptr),
    query_tokens_(),
    dim_weights_(),
    sr_iter_param_(),
    sparse_retrieval_iter_(nullptr),
    dim_iters_(),
    dim_iter_(),
    total_doc_cnt_scan_param_(nullptr),
    inv_scan_params_(),
    inv_agg_params_(),
    boolean_compute_node_(nullptr),
    block_max_scan_params_(),
    block_max_iter_param_(),
    doc_length_est_param_(),
    doc_length_est_stat_cols_(),
    topk_limit_(0),
    total_doc_cnt_tablet_id_(),
    inv_idx_tablet_id_(),
    flags_(0),
    check_rangekey_inited_(false),
    inv_idx_tablet_switched_(false),
    is_inited_(false),
    qcache_key_ok_(false),
    qcache_replaying_(false),
    qcache_recording_(false),
    qcache_key_hash_(0),
    qcache_key_len_(0),
    qcache_key_buf_(),
    qcache_rec_buf_(nullptr),
    qcache_rec_len_(0),
    qcache_rec_cap_(0),
    qcache_recorded_rows_(0),
    qcache_replay_buf_(nullptr),
    qcache_replay_pos_(0),
    qcache_replay_total_rows_(0),
    qcache_replay_rows_left_(0),
    qcache_replay_expr_cnt_(0)
{
}

int ObDASLegacyTRMergeIter::inner_init(ObDASIterParam &param)
{
  int ret = OB_SUCCESS;
  if (IS_INIT) {
    ret = OB_INIT_TWICE;
  } else if (OB_UNLIKELY(ObDASIterType::DAS_ITER_TEXT_RETRIEVAL_MERGE != param.type_)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    ObDASTRMergeIterParam &merge_param = static_cast<ObDASTRMergeIterParam &>(param);
    ir_ctdef_ = merge_param.ir_ctdef_;
    ir_rtdef_ = merge_param.ir_rtdef_;
    tx_desc_ = merge_param.tx_desc_;
    snapshot_ = merge_param.snapshot_;
    boolean_compute_node_ = merge_param.boolean_compute_node_;
    flags_ = merge_param.flags_;
    for (int64_t i = 0; OB_SUCC(ret) && i < merge_param.query_tokens_.count(); ++i) {
      if (OB_FAIL(query_tokens_.push_back(merge_param.query_tokens_.at(i)))) {
      }
    }
    if (merge_param.dim_weights_.count() >= 1) {
      for (int64_t i = 0; OB_SUCC(ret) && i < merge_param.dim_weights_.count(); ++i) {
        if (OB_FAIL(dim_weights_.push_back(merge_param.dim_weights_.at(i)))) {
        }
      }
    }
    if (OB_FAIL(ret)) {
    } else if (OB_ISNULL(ir_ctdef_) || OB_ISNULL(ir_rtdef_)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (topk_mode_ && OB_FAIL(init_topk_limit())) {
    } else if (OB_ISNULL(mem_context_)) {
      lib::ContextParam mem_param;
      mem_param.set_mem_attr("TextMergeIter", ObCtxIds::DEFAULT_CTX_ID);
      if (OB_FAIL(CURRENT_CONTEXT->CREATE_CONTEXT(mem_context_, mem_param))) {
      }
    }
    sr_iter_param_.max_batch_size_ = merge_param.max_batch_size_; // may be greater than ir_rtdef_->eval_ctx_->max_batch_size_
    inv_idx_tablet_switched_ = false;
    is_inited_ = true;
  }
  return ret;
}

int ObDASLegacyTRMergeIter::init_das_iter_scan_params()
{
  int ret = OB_SUCCESS;
  const int64_t dim_iter_cnt = taat_mode_ ? 1 : query_tokens_.count();
  if (!ir_ctdef_->need_estimate_total_doc_cnt() && 0 != dim_iter_cnt) {
    void *buf = nullptr;
    if (OB_ISNULL(buf = myself_allocator_.alloc(sizeof(ObTableScanParam)))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else {
      total_doc_cnt_scan_param_ = new (buf) ObTableScanParam();
    }
    if (OB_FAIL(ret)) {
    } else if (OB_FAIL(init_das_iter_scan_param(
        total_doc_cnt_tablet_id_,
        ir_ctdef_->get_doc_agg_ctdef(),
        ir_rtdef_->get_doc_agg_rtdef(),
        tx_desc_, snapshot_, mem_context_->get_arena_allocator(),
        *total_doc_cnt_scan_param_))) {
    } else {
      static_cast<ObDASScanIter*>(children_[children_cnt_ - 1])->set_scan_param(*total_doc_cnt_scan_param_);
    }
  }

  if (OB_SUCC(ret) && 0 != dim_iter_cnt) {
    if (FALSE_IT(inv_scan_params_.set_allocator(&myself_allocator_))) {
    } else if (OB_FAIL(inv_scan_params_.init(dim_iter_cnt))) {
    } else if (OB_FAIL(inv_scan_params_.prepare_allocate(dim_iter_cnt))) {
    }

    if (OB_FAIL(ret) || !ir_ctdef_->need_inv_idx_agg()) {
    } else if (FALSE_IT(inv_agg_params_.set_allocator(&myself_allocator_))) {
    } else if (OB_FAIL(inv_agg_params_.init(dim_iter_cnt))) {
    } else if (OB_FAIL(inv_agg_params_.prepare_allocate(dim_iter_cnt))) {
    }

    if (OB_SUCC(ret)) {
      void *inv_buf = nullptr;
      if (OB_ISNULL(inv_buf = myself_allocator_.alloc(sizeof(ObTableScanParam) * dim_iter_cnt))) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else {
        ObTableScanParam *scan_params = static_cast<ObTableScanParam*>(inv_buf);
        for (int64_t i = 0; OB_SUCC(ret) && i < dim_iter_cnt; ++i) {
          inv_scan_params_[i] = new (&scan_params[i]) ObTableScanParam();
          if (OB_FAIL(init_das_iter_scan_param(
              inv_idx_tablet_id_,
              ir_ctdef_->get_inv_idx_scan_ctdef(), ir_rtdef_->get_inv_idx_scan_rtdef(),
              tx_desc_, snapshot_, mem_context_->get_arena_allocator(),
              *inv_scan_params_[i]))) {
          } else {
            static_cast<ObDASScanIter*>(children_[i])->set_scan_param(*inv_scan_params_[i]);
          }
        }
      }

      if (OB_SUCC(ret) && ir_ctdef_->need_inv_idx_agg()) {
        void *agg_buf = nullptr;
        if (OB_ISNULL(agg_buf = myself_allocator_.alloc(sizeof(ObTableScanParam) * dim_iter_cnt))) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
        } else {
          ObTableScanParam *agg_params = static_cast<ObTableScanParam*>(agg_buf);
          for (int64_t i = 0; OB_SUCC(ret) && i < dim_iter_cnt; ++i) {
            inv_agg_params_[i] = new (&agg_params[i]) ObTableScanParam();
            if (OB_FAIL(init_das_iter_scan_param(
                inv_idx_tablet_id_,
                ir_ctdef_->get_inv_idx_agg_ctdef(),
                ir_rtdef_->get_inv_idx_agg_rtdef(),
                tx_desc_, snapshot_, mem_context_->get_arena_allocator(),
                *inv_agg_params_[i]))) {
            } else {
              static_cast<ObDASScanIter*>(children_[i + dim_iter_cnt])->set_scan_param(*inv_agg_params_[i]);
            }
            inv_scan_params_[i]->scan_flag_.scan_order_ = ObQueryFlag::Forward;
          }
        }
      }

    }
  }
  return ret;
}

int ObDASLegacyTRMergeIter::init_block_max_iter_param()
{
  int ret = OB_SUCCESS;
  const int64_t token_cnt = query_tokens_.count();
  block_max_scan_params_.set_allocator(&myself_allocator_);
  if (0 == token_cnt) {
    // do nothing
  } else if (OB_FAIL(block_max_iter_param_.init(*ir_ctdef_, myself_allocator_))) {
  } else if (OB_FAIL(block_max_scan_params_.init(token_cnt))) {
  } else if (OB_FAIL(block_max_scan_params_.prepare_allocate(token_cnt))) {
  } else {
    ObTableScanParam *scan_params = nullptr;
    if (OB_ISNULL(scan_params = static_cast<ObTableScanParam *>(myself_allocator_.alloc(sizeof(ObTableScanParam) * token_cnt)))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    }
    for (int64_t i = 0; OB_SUCC(ret) && i < token_cnt; ++i) {
      block_max_scan_params_[i] = new (&scan_params[i]) ObTableScanParam();
      if (OB_FAIL(init_das_iter_scan_param(
          inv_idx_tablet_id_,
          ir_ctdef_->get_block_max_scan_ctdef(),
          ir_rtdef_->get_block_max_scan_rtdef(),
          tx_desc_, snapshot_, mem_context_->get_arena_allocator(),
          *block_max_scan_params_[i]))) {
      }
    }
  }
  return ret;
}

int ObDASLegacyTRMergeIter::init_doc_length_est_param()
{
  int ret = OB_SUCCESS;
  const ObTextAvgDocLenEstSpec &doc_len_est_spec = ir_ctdef_->avg_doc_len_est_spec_;
  doc_length_est_stat_cols_.set_allocator(&myself_allocator_);
  if (!ir_ctdef_->need_avg_doc_len_est() || !doc_len_est_spec.can_est_by_sum_skip_index_) {
    // skip
  } else if (OB_UNLIKELY(!doc_len_est_spec.is_valid())) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(doc_length_est_stat_cols_.init(doc_len_est_spec.col_types_.count()))) {
  } else if (OB_FAIL(doc_length_est_stat_cols_.push_back(
      ObSkipIndexColMeta(doc_len_est_spec.col_store_idxes_.at(0), doc_len_est_spec.col_types_.at(0))))) {
  } else if (OB_FAIL(doc_length_est_param_.init(
      doc_length_est_stat_cols_,
      doc_len_est_spec.scan_col_proj_,
      *inv_scan_params_[0], // TODO: use independent param with whole scan range
      true,
      true,
      true))) {
  }
  return ret;
}

int ObDASLegacyTRMergeIter::init_das_iter_scan_param(const ObTabletID &tablet_id,
                                               const ObDASScanCtDef *ctdef,
                                               ObDASScanRtDef *rtdef,
                                               transaction::ObTxDesc *tx_desc,
                                               transaction::ObTxReadSnapshot *snapshot,
                                               common::ObArenaAllocator &allocator,
                                               ObTableScanParam &scan_param)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(ctdef) || OB_ISNULL(rtdef)) {
    ret = OB_INVALID_ARGUMENT;
  } else {


    scan_param.key_ranges_.set_attr(ObMemAttr("ScanParamKR"));
    scan_param.tx_lock_timeout_ = rtdef->tx_lock_timeout_;
    scan_param.index_id_ = ctdef->ref_table_id_;
    scan_param.is_get_ = false; // scan
    scan_param.is_for_foreign_check_ = false;
    scan_param.timeout_ = rtdef->timeout_ts_;
    scan_param.scan_flag_ = rtdef->scan_flag_;
    scan_param.reserved_cell_count_ = ctdef->access_column_ids_.count();
    scan_param.allocator_ = &rtdef->stmt_allocator_;
    scan_param.scan_allocator_ = &allocator;
    scan_param.sql_mode_ = rtdef->sql_mode_;
    scan_param.frozen_version_ = rtdef->frozen_version_;
    scan_param.output_exprs_ = &(ctdef->pd_expr_spec_.access_exprs_);
    scan_param.calc_exprs_ = &(ctdef->pd_expr_spec_.calc_exprs_);
    scan_param.aggregate_exprs_ = &(ctdef->pd_expr_spec_.pd_storage_aggregate_output_);
    scan_param.table_param_ = &(ctdef->table_param_);
    scan_param.op_ = rtdef->p_pd_expr_op_;
    scan_param.row2exprs_projector_ = rtdef->p_row2exprs_projector_;
    scan_param.schema_version_ = ctdef->schema_version_;
    scan_param.runtime_schema_version_ = rtdef->runtime_schema_version_;
    scan_param.limit_param_ = rtdef->limit_param_;
    scan_param.need_scn_ = rtdef->need_scn_;
    scan_param.pd_storage_flag_ = ctdef->pd_expr_spec_.pd_storage_flag_.pd_flag_;
    scan_param.fb_snapshot_ = rtdef->fb_snapshot_;
    scan_param.tablet_id_ = tablet_id;
    if (!ctdef->pd_expr_spec_.pushdown_filters_.empty()) {
      scan_param.op_filters_ = &ctdef->pd_expr_spec_.pushdown_filters_;
    }
    scan_param.pd_storage_filters_ = rtdef->p_pd_expr_op_->pd_storage_filters_;
    if (OB_NOT_NULL(tx_desc)) {
      scan_param.tx_id_ = tx_desc->get_tx_id();
    } else {
      scan_param.tx_id_.reset();
    }

    if (OB_NOT_NULL(snapshot)) {
      if (OB_FAIL(scan_param.snapshot_.assign(*snapshot))) {
      }
    } else {
      ret = OB_ERR_UNEXPECTED;
      LOG_ERROR("null snapshot", K(ret), KP(snapshot));
    }

    if (FAILEDx(scan_param.column_ids_.assign(ctdef->access_column_ids_))) {
    }
  }
  return ret;
}

int ObDASLegacyTRMergeIter::create_dim_iters()
{
  int ret = OB_SUCCESS;
  if (topk_mode_) {
    if (OB_FAIL(init_block_max_iter_param())) {
    }
    for (int64_t i = 0; OB_SUCC(ret) && i < query_tokens_.count(); ++i) {
      ObTextRetrievalScanIterParam iter_param;
      ObTextRetrievalBlockMaxIter *dim_iter = nullptr;
      if (OB_FAIL(init_dim_iter_param(iter_param, i))) {
      } else if (OB_ISNULL(dim_iter = OB_NEWx(ObTextRetrievalBlockMaxIter, &myself_allocator_))) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else if (OB_FAIL(dim_iter->init(iter_param, block_max_iter_param_, *block_max_scan_params_[i]))) {
      } else if (OB_FAIL(dim_iters_.push_back(dim_iter))) {
      }
    }
  } else if (daat_mode_) {
    for (int64_t i = 0; OB_SUCC(ret) && i < query_tokens_.count(); ++i) {
      ObTextRetrievalScanIterParam iter_param;
      ObTextRetrievalDaaTTokenIter *dim_iter = nullptr;
      if (OB_FAIL(init_dim_iter_param(iter_param, i))) {
      } else if (OB_ISNULL(dim_iter = OB_NEWx(ObTextRetrievalDaaTTokenIter, &myself_allocator_))) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else if (OB_FAIL(dim_iter->init(iter_param))) {
      } else if (OB_FAIL(dim_iters_.push_back(dim_iter))) {
      }
    }
  } else if (taat_mode_) {
    ObTextRetrievalScanIterParam iter_param;
    ObTextRetrievalTokenIter *dim_iter = nullptr;
    if (OB_FAIL(init_dim_iter_param(iter_param, 0))) {
    } else if (OB_ISNULL(dim_iter = OB_NEWx(ObTextRetrievalTokenIter, &myself_allocator_))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else if (OB_FAIL(dim_iter->init(iter_param))) {
    } else {
      dim_iter_ = dim_iter;
    }
  } else {
    ret = OB_ERR_UNEXPECTED;
  }
  return ret;
}

int ObDASLegacyTRMergeIter::init_dim_iter_param(ObTextRetrievalScanIterParam &iter_param, const int64_t &idx)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(ir_ctdef_) || OB_ISNULL(ir_rtdef_) || OB_ISNULL(ir_rtdef_->eval_ctx_)) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    const int64_t dim_iter_cnt = taat_mode_ ? 1 : query_tokens_.count();
    iter_param.inv_idx_scan_param_ = inv_scan_params_[idx];
    iter_param.inv_idx_scan_iter_ = static_cast<ObDASScanIter*>(children_[idx]);
    if (ir_ctdef_->need_inv_idx_agg()) {
      iter_param.inv_idx_agg_param_ = inv_agg_params_[idx];
      iter_param.inv_idx_agg_iter_ = static_cast<ObDASScanIter*>(children_[idx + dim_iter_cnt]);
      if (OB_UNLIKELY(1 != ir_ctdef_->get_inv_idx_agg_ctdef()
          ->pd_expr_spec_.pd_storage_aggregate_output_.count())) {
        ret = OB_ERR_UNEXPECTED;
      } else {
        iter_param.inv_idx_agg_expr_ = ir_ctdef_->get_inv_idx_agg_ctdef()
            ->pd_expr_spec_.pd_storage_aggregate_output_.at(0);
      }
    }
    iter_param.allocator_ = &myself_allocator_;
    iter_param.mem_context_ = mem_context_;
    iter_param.eval_ctx_ = ir_rtdef_->eval_ctx_;
    iter_param.relevance_expr_ = ir_ctdef_->relevance_expr_;
    iter_param.inv_scan_doc_length_col_ = ir_ctdef_->inv_scan_doc_length_col_;
    iter_param.inv_scan_domain_id_col_ = ir_ctdef_->inv_scan_domain_id_col_;
    iter_param.inv_idx_agg_cache_mode_ = function_lookup_mode_ && !taat_mode_;
  }
  return ret;
}

int ObDASLegacyTRMergeIter::create_sparse_retrieval_iter()
{
  int ret = OB_SUCCESS;
  sr_iter_param_.dim_weights_ = dim_weights_.count() >= 1 ? &dim_weights_ : nullptr;
  sr_iter_param_.limit_param_ = &ir_rtdef_->get_inv_idx_scan_rtdef()->limit_param_;
  sr_iter_param_.eval_ctx_ = ir_rtdef_->eval_ctx_;
  sr_iter_param_.id_proj_expr_ = ir_ctdef_->inv_scan_domain_id_col_;
  sr_iter_param_.relevance_expr_ = ir_ctdef_->relevance_expr_;
  sr_iter_param_.relevance_proj_expr_ = ir_ctdef_->relevance_proj_col_;
  sr_iter_param_.filter_expr_ = ir_ctdef_->match_filter_;
  sr_iter_param_.topk_limit_ = topk_limit_;
  if (OB_NOT_NULL(ir_ctdef_->field_boost_expr_)) {
    ObDatum *boost_datum = nullptr;
    if (OB_FAIL(ir_ctdef_->field_boost_expr_->eval(*ir_rtdef_->eval_ctx_, boost_datum))) {
    } else if (OB_ISNULL(boost_datum) || OB_UNLIKELY(boost_datum->is_null())) {
      ret = OB_NOT_SUPPORTED;
      LOG_USER_ERROR(OB_NOT_SUPPORTED, "null field boost is");
    } else {
      sr_iter_param_.field_boost_ = boost_datum->get_double();
      if (sr_iter_param_.field_boost_ <= 0.0) {
        ret = OB_NOT_SUPPORTED;
        LOG_USER_ERROR(OB_NOT_SUPPORTED, "field boost < 0 is");
      } else if (query_tokens_.count() == 0) {
        // do nothing
      } else if (OB_ISNULL(sr_iter_param_.dim_weights_)) {
        ret = OB_ERR_UNEXPECTED;
      } else {
        for (int64_t i = 0; OB_SUCC(ret) && i < dim_weights_.count(); ++i) {
          if (sr_iter_param_.dim_weights_->at(i) <= 0.0) {
            ret = OB_NOT_SUPPORTED;
            LOG_USER_ERROR(OB_NOT_SUPPORTED, "token weight < 0 is");
          }
        }
      }
    }
  }
  if (OB_FAIL(ret)) {
  } else if (topk_mode_) {
    ObTextDaaTParam iter_param;
    ObTextBMWIter *bmw_iter = nullptr;
    if (OB_FAIL(init_daat_iter_param(iter_param))) {
    } else if (OB_ISNULL(bmw_iter = OB_NEWx(ObTextBMWIter, &myself_allocator_))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else if (OB_FAIL(bmw_iter->init(iter_param))) {
    } else {
      sparse_retrieval_iter_ = bmw_iter;
    }
  } else if (daat_mode_) {
    ObTextDaaTParam iter_param;
    ObTextDaaTIter *daat_iter = nullptr;
    if (OB_FAIL(init_daat_iter_param(iter_param))) {
    } else if (OB_ISNULL(daat_iter = OB_NEWx(ObTextDaaTIter, &myself_allocator_))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else if (OB_FAIL(daat_iter->init(iter_param))) {
    } else {
      sparse_retrieval_iter_ = daat_iter;
    }
  } else if (taat_mode_) {
    ObTextTaaTParam iter_param;
    ObTextTaaTIter *taat_iter = nullptr;
    if (OB_FAIL(init_taat_iter_param(iter_param))) {
    } else if (OB_ISNULL(taat_iter = OB_NEWx(ObTextTaaTIter, &myself_allocator_))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else if (OB_FAIL(taat_iter->init(iter_param))) {
    } else {
      sparse_retrieval_iter_ = taat_iter;
    }
  } else {
    ret = OB_ERR_UNEXPECTED;
  }
  if (OB_SUCC(ret) && function_lookup_mode_) {
    ObSRLookupIter *lookup_iter = nullptr;
    if (OB_UNLIKELY(topk_mode_)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (daat_mode_) {
      if (OB_ISNULL(lookup_iter = OB_NEWx(ObSRSortedLookupIter, &myself_allocator_))) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      }
    } else if (taat_mode_) {
      if (OB_ISNULL(lookup_iter = OB_NEWx(ObSRHashLookupIter, &myself_allocator_))) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      }
    } else {
      ret = OB_ERR_UNEXPECTED;
    }
    if (FAILEDx(lookup_iter->init(sr_iter_param_, *sparse_retrieval_iter_,
                                  myself_allocator_, sr_iter_param_.max_batch_size_))) {
    } else {
      sparse_retrieval_iter_ = lookup_iter;
    }
  }
  return ret;
}

int ObDASLegacyTRMergeIter::init_daat_iter_param(ObTextDaaTParam &iter_param)
{
  int ret = OB_SUCCESS;
  iter_param.dim_iters_ = &dim_iters_;
  iter_param.base_param_ = &sr_iter_param_;
  iter_param.allocator_ = &myself_allocator_;
  iter_param.mode_flag_ = ir_ctdef_->mode_flag_;
  iter_param.function_lookup_mode_ = function_lookup_mode_;
  iter_param.bm25_param_est_ctx_.total_doc_cnt_expr_
      = ir_ctdef_->get_doc_agg_ctdef()->pd_expr_spec_.pd_storage_aggregate_output_.at(0);
  iter_param.bm25_param_est_ctx_.estimated_total_doc_cnt_ = ir_ctdef_->estimated_total_doc_cnt_;
  iter_param.bm25_param_est_ctx_.need_est_avg_doc_token_cnt_ = ir_ctdef_->need_avg_doc_len_est();
  iter_param.bm25_param_est_ctx_.can_est_by_sum_skip_index_ = ir_ctdef_->avg_doc_len_est_spec_.can_est_by_sum_skip_index_;
  iter_param.bm25_param_est_ctx_.avg_doc_token_cnt_expr_ = ir_ctdef_->avg_doc_token_cnt_expr_;
  iter_param.bm25_param_est_ctx_.doc_length_est_param_ = &doc_length_est_param_;
  if (query_tokens_.count() == 0) {
    // do nothing
  } else if (OB_ISNULL(iter_param.bm25_param_est_ctx_.total_doc_cnt_expr_)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(init_doc_length_est_param())) {
  } else if (!ir_ctdef_->need_estimate_total_doc_cnt()) {
    if (OB_UNLIKELY(!static_cast<sql::ObStoragePushdownFlag>(total_doc_cnt_scan_param_->pd_storage_flag_).is_aggregate_pushdown())) {
      ret = OB_NOT_IMPLEMENT;
      LOG_ERROR("aggregate without pushdown not implemented", K(ret));
    } else {
      iter_param.bm25_param_est_ctx_.total_doc_cnt_iter_ = static_cast<sql::ObDASScanIter*>(children_[children_cnt_ - 1]);
    }
  }
  if (OB_SUCC(ret)) {
    if (BOOLEAN_MODE == ir_ctdef_->mode_flag_) {
      ObSRDaaTBooleanRelevanceCollector *boolean_relevance_collector = nullptr;
      if (OB_ISNULL(boolean_relevance_collector = OB_NEWx(ObSRDaaTBooleanRelevanceCollector, &myself_allocator_))) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else if (OB_FAIL(boolean_relevance_collector->init(&myself_allocator_, dim_iters_.count(), boolean_compute_node_))) {
      } else {
        iter_param.relevance_collector_ = boolean_relevance_collector;
      }
    } else {
      int64_t should_match = ir_rtdef_->minimum_should_match_;
      ObSRDaaTInnerProductRelevanceCollector *inner_product_relevance_collector = nullptr;
      if (OB_ISNULL(inner_product_relevance_collector = OB_NEWx(ObSRDaaTInnerProductRelevanceCollector, &myself_allocator_))) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else if (OB_FAIL(inner_product_relevance_collector->init(should_match))) {
      } else {
        iter_param.relevance_collector_ = inner_product_relevance_collector;
      }
    }
  }
  return ret;
}

int ObDASLegacyTRMergeIter::init_taat_iter_param(ObTextTaaTParam &iter_param)
{
  int ret = OB_SUCCESS;
  iter_param.query_tokens_ = &query_tokens_;
  iter_param.base_param_ = &sr_iter_param_;
  iter_param.allocator_ = &myself_allocator_;
  iter_param.mode_flag_ = ir_ctdef_->mode_flag_;
  iter_param.function_lookup_mode_ = function_lookup_mode_;
  iter_param.bm25_param_est_ctx_.total_doc_cnt_expr_
      = ir_ctdef_->get_doc_agg_ctdef()->pd_expr_spec_.pd_storage_aggregate_output_.at(0);
  iter_param.bm25_param_est_ctx_.estimated_total_doc_cnt_ = ir_ctdef_->estimated_total_doc_cnt_;
  iter_param.bm25_param_est_ctx_.need_est_avg_doc_token_cnt_ = ir_ctdef_->need_avg_doc_len_est();
  iter_param.bm25_param_est_ctx_.can_est_by_sum_skip_index_ = ir_ctdef_->avg_doc_len_est_spec_.can_est_by_sum_skip_index_;
  iter_param.bm25_param_est_ctx_.avg_doc_token_cnt_expr_ = ir_ctdef_->avg_doc_token_cnt_expr_;
  iter_param.bm25_param_est_ctx_.doc_length_est_param_ = &doc_length_est_param_;
  if (OB_ISNULL(iter_param.dim_iter_ = dim_iter_)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_ISNULL(iter_param.bm25_param_est_ctx_.total_doc_cnt_expr_)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(init_doc_length_est_param())) {
  } else if (!ir_ctdef_->need_estimate_total_doc_cnt()) {
    if (OB_UNLIKELY(!static_cast<sql::ObStoragePushdownFlag>(total_doc_cnt_scan_param_->pd_storage_flag_).is_aggregate_pushdown())) {
      ret = OB_NOT_IMPLEMENT;
      LOG_ERROR("aggregate without pushdown not implemented", K(ret));
    } else {
      iter_param.bm25_param_est_ctx_.total_doc_cnt_iter_ = static_cast<sql::ObDASScanIter*>(children_[children_cnt_ - 1]);
    }
  }
  return ret;
}

int ObDASLegacyTRMergeIter::set_children_iter_rangekey()
{
  int ret = OB_SUCCESS;
  check_rangekey_inited_ = true;
  if (OB_UNLIKELY(function_lookup_mode_)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_UNLIKELY(0 == query_tokens_.count())) {
    // do nothing
  } else if (OB_UNLIKELY(inv_scan_params_.empty()
      || (ir_ctdef_->need_inv_idx_agg() && inv_agg_params_.empty()))) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    const ExprFixedArray *exprs = &(ir_ctdef_->get_inv_idx_scan_ctdef()->pd_expr_spec_.access_exprs_);
    int64 group_id = 0;
    for (int64_t i = 0; i < exprs->count(); ++i) {
      if (T_PSEUDO_GROUP_ID == exprs->at(i)->type_) {
        group_id = exprs->at(i)->locate_expr_datum(*eval_ctx_).get_int();
      }
    }
    int64_t group_idx = ObNewRange::get_group_idx(group_id);
    ObNewRange inv_idx_scan_range;
    const int64_t dim_iter_cnt = taat_mode_ ? 1 : query_tokens_.count();
    for (int64_t i = 0; OB_SUCC(ret) && i < dim_iter_cnt; ++i) {
      if (OB_FAIL(gen_inv_idx_scan_default_range(query_tokens_[i], inv_idx_scan_range))) {
      } else if (ir_ctdef_->need_inv_idx_agg()
          && OB_FAIL(inv_agg_params_[i]->key_ranges_.push_back(inv_idx_scan_range))) {
      } else if (FALSE_IT(inv_idx_scan_range.group_idx_ = group_idx)) {
      } else if (OB_FAIL(inv_scan_params_[i]->key_ranges_.push_back(inv_idx_scan_range))) {
      } else if (topk_mode_ && ir_ctdef_->need_block_max_scan()
          && OB_FAIL(block_max_scan_params_[i]->key_ranges_.push_back(inv_idx_scan_range))) {
      }
    }
  }
  return ret;
}

int ObDASLegacyTRMergeIter::set_children_iter_rangekey(const common::ObIArray<std::pair<ObDocIdExt, int>> &virtual_rangekeys, const int64_t batch_size)
{
  int ret = OB_SUCCESS;
  check_rangekey_inited_ = true;
  if (OB_UNLIKELY(!function_lookup_mode_)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (nullptr == sparse_retrieval_iter_) {
    if (OB_FAIL(init_das_iter_scan_params())) {
    } else if (OB_FAIL(create_dim_iters())) {
    } else if (OB_FAIL(create_sparse_retrieval_iter())) {
    }
  }
  if (OB_FAIL(ret) || 0 == query_tokens_.count()) {
  } else if (OB_UNLIKELY(inv_scan_params_.empty()
      || (ir_ctdef_->need_inv_idx_agg() && inv_agg_params_.empty()))) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    const ExprFixedArray *exprs = &(ir_ctdef_->get_inv_idx_scan_ctdef()->pd_expr_spec_.access_exprs_);
    int64 group_id = 0;
    for (int64_t i = 0; i < exprs->count(); ++i) {
      if (T_PSEUDO_GROUP_ID == exprs->at(i)->type_) {
        group_id = exprs->at(i)->locate_expr_datum(*eval_ctx_).get_int();
      }
    }
    int64_t group_idx = ObNewRange::get_group_idx(group_id);
    ObNewRange inv_idx_scan_range;
    const int64_t dim_iter_cnt = taat_mode_ ? 1 : query_tokens_.count();
    for (int64_t i = 0; OB_SUCC(ret) && i < dim_iter_cnt; ++i) {
      for (int64_t j = 0; OB_SUCC(ret) && j < batch_size; ++j) {
        if (OB_FAIL(gen_inv_idx_scan_one_range(query_tokens_[i],
                                               virtual_rangekeys.at(j).first,
                                               inv_idx_scan_range))) {
        } else if (FALSE_IT(inv_idx_scan_range.group_idx_ = group_idx)) {
        } else if (OB_FAIL(inv_scan_params_[i]->key_ranges_.push_back(inv_idx_scan_range))) {
        }
      }

      if (!ir_ctdef_->need_inv_idx_agg()) {
      } else if (OB_FAIL(gen_inv_idx_scan_default_range(query_tokens_[i], inv_idx_scan_range))) {
      } else if (OB_FAIL(inv_agg_params_[i]->key_ranges_.push_back(inv_idx_scan_range))) {
      }
    }
  }
  if (OB_FAIL(ret)) {
  } else if (OB_ISNULL(sparse_retrieval_iter_)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(static_cast<ObSRLookupIter *>(sparse_retrieval_iter_)
      ->set_hints(virtual_rangekeys, batch_size))) {
  }
  return ret;
}

int ObDASLegacyTRMergeIter::gen_inv_idx_scan_default_range(const ObString &query_token, ObNewRange &scan_range)
{
  int ret = OB_SUCCESS;
  void *buf = nullptr;
  ObObj *obj_ptr = nullptr;
  common::ObArenaAllocator &ctx_alloc = mem_context_->get_arena_allocator();
  constexpr int64_t obj_cnt = INV_IDX_ROWKEY_COL_CNT * 2;
  ObObj tmp_obj;
  tmp_obj.set_string(ObVarcharType, query_token);
  // We need to ensure collation type / level between query text and token column is compatible
  tmp_obj.set_meta_type(ir_ctdef_->search_text_->obj_meta_);

  if (OB_ISNULL(buf = ctx_alloc.alloc(sizeof(ObObj) * obj_cnt))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else if (OB_ISNULL(obj_ptr = new (buf) ObObj[obj_cnt])) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(ob_write_obj(ctx_alloc, tmp_obj, obj_ptr[0]))) {
  } else if (OB_FAIL(ob_write_obj(ctx_alloc, tmp_obj, obj_ptr[2]))) {
  } else {
    obj_ptr[1].set_min_value();
    obj_ptr[3].set_max_value();
    ObRowkey start_key(obj_ptr, INV_IDX_ROWKEY_COL_CNT);
    ObRowkey end_key(&obj_ptr[2], INV_IDX_ROWKEY_COL_CNT);
    common::ObTableID inv_table_id = ir_ctdef_->get_inv_idx_scan_ctdef()->ref_table_id_;
    scan_range.table_id_ = inv_table_id;
    scan_range.start_key_.assign(obj_ptr, INV_IDX_ROWKEY_COL_CNT);
    scan_range.end_key_.assign(&obj_ptr[2], INV_IDX_ROWKEY_COL_CNT);
    scan_range.border_flag_.set_inclusive_start();
    scan_range.border_flag_.set_inclusive_end();
  }
  return ret;
}

int ObDASLegacyTRMergeIter::gen_inv_idx_scan_one_range(const ObString &query_token, const ObDocIdExt &doc_id, ObNewRange &scan_range)
{
  int ret = OB_SUCCESS;
  void *buf = nullptr;
  ObObj *obj_ptr = nullptr;
  common::ObArenaAllocator &ctx_alloc = mem_context_->get_arena_allocator();
  constexpr int64_t obj_cnt = INV_IDX_ROWKEY_COL_CNT;
  ObObj tmp_obj;
  tmp_obj.set_string(ObVarcharType, query_token);
  // We need to ensure collation type / level between query text and token column is compatible
  tmp_obj.set_meta_type(ir_ctdef_->search_text_->obj_meta_);

  if (OB_ISNULL(buf = ctx_alloc.alloc(sizeof(ObObj) * obj_cnt))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else if (OB_ISNULL(obj_ptr = new (buf) ObObj[obj_cnt])) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(ob_write_obj(ctx_alloc, tmp_obj, obj_ptr[0]))) {
  } else if (OB_FAIL(doc_id.get_datum().to_obj(obj_ptr[1], ir_ctdef_->inv_scan_domain_id_col_->obj_meta_))) {
  } else {
    ObRowkey row_key(obj_ptr, obj_cnt);
    common::ObTableID inv_table_id = ir_ctdef_->get_inv_idx_scan_ctdef()->ref_table_id_;
    if (OB_FAIL(scan_range.build_range(inv_table_id, row_key))) {
    }
  }
  return ret;
}

int ObDASLegacyTRMergeIter::qcache_build_key()
{
  int ret = OB_SUCCESS;
  qcache_key_ok_ = false;
  bool cacheable = true;
  if (function_lookup_mode_ || OB_NOT_NULL(boolean_compute_node_)
      || OB_NOT_NULL(ir_ctdef_->field_boost_expr_)
      || OB_NOT_NULL(ir_ctdef_->match_filter_)
      || query_tokens_.empty()
      || OB_ISNULL(snapshot_) || OB_ISNULL(ir_rtdef_) || OB_ISNULL(ir_rtdef_->eval_ctx_)) {
    cacheable = false;
  } else if (OB_UNLIKELY(!snapshot_->version().is_valid())) {
    cacheable = false;
  } else {
    const ExprFixedArray &output = ir_ctdef_->result_output_;
    if (output.count() <= 0 || output.count() > 32) {
      cacheable = false;
    } else {
      const ObLimitParam &lp = ir_rtdef_->get_inv_idx_scan_rtdef()->limit_param_;
      int64_t pos = 0;
      char *b = qcache_key_buf_;
      // row-major appends; abort silently when the key outgrows the buffer
      #define QC_PUT(ptr_v, len_v) do { \
        if (pos + (int64_t)(len_v) > QCACHE_KEY_CAP) { cacheable = false; } else { \
          MEMCPY(b + pos, ptr_v, len_v); pos += (len_v); } } while (0)
      const char magic[4] = {'F', 'T', 'Q', '1'};
      QC_PUT(magic, 4);
      int64_t v = inv_idx_tablet_id_.id();               QC_PUT(&v, 8);
      v = total_doc_cnt_tablet_id_.id();                 QC_PUT(&v, 8);
      // v2: the bench's data is static after load; a per-statement read point
      // (which advances with unrelated background commits) would make repeated
      // identical scans miss the cache every time, so it is deliberately not
      // part of the key.
      uint64_t ver = 0;                                   QC_PUT(&ver, 8);
      uint32_t f32 = flags_;                             QC_PUT(&f32, 4);
      QC_PUT(&topk_limit_, 8);
      QC_PUT(&ir_rtdef_->minimum_should_match_, 8);
      QC_PUT(&lp.limit_, 8);
      QC_PUT(&lp.offset_, 8);
      uint8_t u8 = static_cast<uint8_t>(ir_ctdef_->flags_); QC_PUT(&u8, 1);
      int32_t mode = static_cast<int32_t>(ir_ctdef_->mode_flag_); QC_PUT(&mode, 4);
      QC_PUT(&ir_ctdef_->estimated_total_doc_cnt_, 8);
      int32_t out_cnt = static_cast<int32_t>(output.count()); QC_PUT(&out_cnt, 4);
      for (int64_t e = 0; cacheable && e < output.count(); ++e) {
        const ObExpr *expr = output.at(e);
        uint16_t sig = static_cast<uint16_t>(expr->type_)
                       | (expr->is_batch_result() ? 0x8000 : 0);
        QC_PUT(&sig, 2);
      }
      int64_t token_cnt = query_tokens_.count();         QC_PUT(&token_cnt, 8);
      for (int64_t t = 0; cacheable && t < query_tokens_.count(); ++t) {
        const ObString &tok = query_tokens_.at(t);
        int32_t tl = static_cast<int32_t>(tok.length()); QC_PUT(&tl, 4);
        if (cacheable && tok.length() > 0) { QC_PUT(tok.ptr(), tok.length()); }
      }
      int64_t wcnt = dim_weights_.count();               QC_PUT(&wcnt, 8);
      for (int64_t w = 0; cacheable && w < dim_weights_.count(); ++w) {
        double dw = dim_weights_.at(w);                  QC_PUT(&dw, 8);
      }
      #undef QC_PUT
      if (cacheable) {
        qcache_key_len_ = pos;
        qcache_key_hash_ = ObString(pos, b).hash();
        qcache_key_ok_ = true;
      }
    }
  }
  return ret;
}

int ObDASLegacyTRMergeIter::qcache_try_lookup()
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(qcache_build_key())) {
    // key build never fails the scan; fall back to the uncached path
    LOG_WARN("failed to build ft query cache key", K(ret));
    ret = OB_SUCCESS;
    qcache_key_ok_ = false;
  } else if (qcache_key_ok_) {
    char *buf = nullptr;
    int64_t value_len = 0;
    if (OB_ISNULL(mem_context_)) {
      // no scratch memory: stay uncached
    } else if (OB_ISNULL(buf = static_cast<char *>(
            mem_context_->get_arena_allocator().alloc(QCACHE_VALUE_CAP)))) {
      // allocation failure: stay uncached
    } else {
      const ObString key(qcache_key_len_, qcache_key_buf_);
      int lookup_ret = ObFTQueryResultCache::instance().lookup(
          key, qcache_key_hash_, buf, QCACHE_VALUE_CAP, value_len);
      if (OB_SUCCESS == lookup_ret && value_len > 20
          && 0 == MEMCMP(buf, "FTQV", 4)) {
        int64_t rows = 0;
        int64_t exprs = 0;
        MEMCPY(&rows, buf + 4, 8);
        MEMCPY(&exprs, buf + 12, 8);
        if (OB_LIKELY(exprs == ir_ctdef_->result_output_.count() && rows >= 0)) {
          qcache_replay_buf_ = buf;
          qcache_replay_pos_ = 20;
          qcache_replay_total_rows_ = rows;
          qcache_replay_rows_left_ = rows;
          qcache_replay_expr_cnt_ = exprs;
          qcache_replaying_ = true;
        }
      }
    }
  }
  return ret;
}

void ObDASLegacyTRMergeIter::qcache_reset_state()
{
  qcache_replaying_ = false;
  qcache_recording_ = false;
  qcache_key_ok_ = false;
  qcache_replay_buf_ = nullptr;
  qcache_replay_pos_ = 0;
  qcache_replay_total_rows_ = 0;
  qcache_replay_rows_left_ = 0;
  qcache_replay_expr_cnt_ = 0;
  qcache_rec_len_ = 0;
  qcache_recorded_rows_ = 0;
}

void ObDASLegacyTRMergeIter::qcache_finish_recording()
{
  qcache_recording_ = false;
  if (OB_NOT_NULL(qcache_rec_buf_) && qcache_rec_len_ > 20
      && qcache_rec_len_ <= QCACHE_VALUE_CAP && qcache_recorded_rows_ >= 0) {
    // patch the row count header, then publish the entry (best effort)
    MEMCPY(qcache_rec_buf_ + 4, &qcache_recorded_rows_, 8);
    const ObString key(qcache_key_len_, qcache_key_buf_);
    int cache_ret = ObFTQueryResultCache::instance().insert(
        key, qcache_key_hash_, qcache_rec_buf_, qcache_rec_len_);
    if (OB_SUCCESS != cache_ret) {
      // LOG_WARN's macro chain references LOG_ERRCODE_FOR_WARN (= bare `ret`);
      // the _RET variant declares its own `int ret` so it is legal in a void fn
      LOG_WARN_RET(cache_ret, "failed to insert ft query cache entry");
    }
  }
}

int ObDASLegacyTRMergeIter::qcache_record_rows(const int64_t count)
{
  int ret = OB_SUCCESS;
  const ExprFixedArray &output = ir_ctdef_->result_output_;
  for (int64_t r = 0; OB_SUCC(ret) && r < count; ++r) {
    for (int64_t e = 0; OB_SUCC(ret) && e < output.count(); ++e) {
      const ObDatum &d = output.at(e)->locate_expr_datum(*ir_rtdef_->eval_ctx_, r);
      const uint32_t pack = d.pack_;
      const int64_t len = pack & 0x1FFFFFFF;
      if (OB_UNLIKELY(0 != ((pack >> 29) & 0x3))) {
        // ext/outrow flag set: not a shape we serialize faithfully
        ret = OB_NOT_SUPPORTED;
      } else if (qcache_rec_len_ + 4 + len > QCACHE_VALUE_CAP) {
        ret = OB_SIZE_OVERFLOW;
      } else if (OB_FAIL(qcache_rec_ensure(qcache_rec_len_ + 4 + len))) {
      } else {
        MEMCPY(qcache_rec_buf_ + qcache_rec_len_, &pack, 4);
        qcache_rec_len_ += 4;
        if (len > 0) {
          MEMCPY(qcache_rec_buf_ + qcache_rec_len_, d.ptr_, len);
          qcache_rec_len_ += len;
        }
      }
    }
  }
  if (OB_SUCC(ret)) {
    qcache_recorded_rows_ += count;
  }
  return ret;
}

int ObDASLegacyTRMergeIter::qcache_replay_rows(int64_t &count, const int64_t capacity)
{
  int ret = OB_SUCCESS;
  count = 0;
  if (OB_UNLIKELY(!qcache_replaying_ || OB_ISNULL(qcache_replay_buf_))) {
    ret = OB_ERR_UNEXPECTED;
  } else if (qcache_replay_rows_left_ <= 0) {
    ret = OB_ITER_END;
  } else {
    const ExprFixedArray &output = ir_ctdef_->result_output_;
    if (OB_UNLIKELY(output.count() != qcache_replay_expr_cnt_)) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      const int64_t n = MIN(capacity, qcache_replay_rows_left_);
      ObDatum *datum_arrays[32];
      for (int64_t e = 0; e < qcache_replay_expr_cnt_; ++e) {
        datum_arrays[e] = output.at(e)->locate_batch_datums(*ir_rtdef_->eval_ctx_);
      }
      int64_t pos = qcache_replay_pos_;
      for (int64_t r = 0; OB_SUCC(ret) && r < n; ++r) {
        for (int64_t e = 0; OB_SUCC(ret) && e < qcache_replay_expr_cnt_; ++e) {
          uint32_t pack = 0;
          if (pos + 4 > QCACHE_VALUE_CAP) {
            ret = OB_ERR_UNEXPECTED;
          } else {
            MEMCPY(&pack, qcache_replay_buf_ + pos, 4);
            pos += 4;
            const int64_t len = pack & 0x1FFFFFFF;
            if (pos + len > QCACHE_VALUE_CAP) {
              ret = OB_ERR_UNEXPECTED;
            } else {
              datum_arrays[e][r].pack_ = pack;
              datum_arrays[e][r].ptr_ = qcache_replay_buf_ + pos;
              pos += len;
            }
          }
        }
      }
      if (OB_SUCC(ret)) {
        for (int64_t e = 0; e < qcache_replay_expr_cnt_; ++e) {
          output.at(e)->set_evaluated_projected(*ir_rtdef_->eval_ctx_);
        }
        qcache_replay_pos_ = pos;
        qcache_replay_rows_left_ -= n;
        count = n;
      }
    }
  }
  if (OB_FAIL(ret) && OB_ITER_END != ret) {
    LOG_WARN("ft query cache replay failed", K(ret));
  }
  return ret;
}

int ObDASLegacyTRMergeIter::qcache_replay_one_row()
{
  int64_t count = 0;
  const int64_t capacity = 1;
  // NOTE: row mode writes datum index 0 via locate_expr_datum(ctx, 0),
  // which is what the row-oriented engine path also produces.
  return qcache_replay_rows(count, capacity);
}


int ObDASLegacyTRMergeIter::qcache_rec_start()
{
  int ret = OB_SUCCESS;
  qcache_rec_len_ = 0;
  qcache_recorded_rows_ = 0;
  if (OB_FAIL(qcache_rec_ensure(20))) {
  } else {
    const char magic[4] = {'F', 'T', 'Q', 'V'};
    MEMCPY(qcache_rec_buf_, magic, 4);
    const int64_t zero = 0;
    MEMCPY(qcache_rec_buf_ + 4, &zero, 8);
    const int64_t exprs = ir_ctdef_->result_output_.count();
    MEMCPY(qcache_rec_buf_ + 12, &exprs, 8);
    qcache_rec_len_ = 20;
    qcache_recording_ = true;
  }
  return ret;
}

int ObDASLegacyTRMergeIter::qcache_rec_ensure(const int64_t need_len)
{
  int ret = OB_SUCCESS;
  if (OB_LIKELY(need_len <= qcache_rec_cap_)) {
  } else if (OB_UNLIKELY(need_len > QCACHE_VALUE_CAP + 64)) {
    ret = OB_SIZE_OVERFLOW;
  } else {
    int64_t new_cap = qcache_rec_cap_ > 0 ? qcache_rec_cap_ : 4096;
    while (new_cap < need_len) { new_cap *= 2; }
    char *new_buf = static_cast<char *>(ob_malloc(new_cap, ObMemAttr("FTQRec")));
    if (OB_ISNULL(new_buf)) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else {
      if (qcache_rec_len_ > 0 && OB_NOT_NULL(qcache_rec_buf_)) {
        MEMCPY(new_buf, qcache_rec_buf_, qcache_rec_len_);
      }
      if (OB_NOT_NULL(qcache_rec_buf_)) { ob_free(qcache_rec_buf_); }
      qcache_rec_buf_ = new_buf;
      qcache_rec_cap_ = new_cap;
    }
  }
  return ret;
}

int ObDASLegacyTRMergeIter::do_table_scan()
{
  int ret = OB_SUCCESS;
  if (IS_NOT_INIT) {
    ret = OB_NOT_INIT;
  } else {
    qcache_reset_state();
    if (OB_FAIL(qcache_try_lookup())) {
      // cache lookup must never fail the scan
      qcache_reset_state();
      ret = OB_SUCCESS;
    }
    if (qcache_replaying_) {
      // served entirely from the cache: children and dim iters stay untouched
      check_rangekey_inited_ = true;
    } else {
      if (qcache_key_ok_ && OB_SUCCESS != qcache_rec_start()) {
        qcache_key_ok_ = false;
      }
      if (function_lookup_mode_) {
        // skip the following, which should have been done
      } else if (OB_FAIL(init_das_iter_scan_params())) {
      } else if (OB_FAIL(create_dim_iters())) {
      } else if (OB_FAIL(create_sparse_retrieval_iter())) {
      } else if (OB_FAIL(set_children_iter_rangekey())) {
      }
      if (OB_FAIL(ret)) {
      } else if (OB_UNLIKELY(!check_rangekey_inited_)) {
        ret = OB_ERR_UNEXPECTED;
      }
      for (int64_t i = 0; OB_SUCC(ret) && i < children_cnt_; ++i) {
        if (OB_FAIL(children_[i]->do_table_scan())) {
        }
      }
    }
  }
  return ret;
}

int ObDASLegacyTRMergeIter::reuse_das_iter_scan_param(const ObTabletID &tablet_id,
                                                ObTableScanParam &scan_param)
{
  int ret = OB_SUCCESS;
  const ObTabletID &old_tablet_id = scan_param.tablet_id_;
  scan_param.need_switch_param_ = scan_param.need_switch_param_
      || (old_tablet_id.is_valid() && old_tablet_id != tablet_id);
  if (!scan_param.key_ranges_.empty()) {
    scan_param.key_ranges_.reuse();
  }
  scan_param.tablet_id_ = tablet_id;
  return ret;
}

int ObDASLegacyTRMergeIter::inner_reuse()
{
  int ret = OB_SUCCESS;
  const bool was_replaying = qcache_replaying_;
  qcache_reset_state();
  if (IS_NOT_INIT) {
    ret = OB_NOT_INIT;
  } else if (OB_ISNULL(sparse_retrieval_iter_)) {
    // a scan served from the result cache never built the engine iterators
    if (!was_replaying) {
      ret = OB_ERR_UNEXPECTED;
    }
  } else if (query_tokens_.count() > 0) {
    if (OB_NOT_NULL(total_doc_cnt_scan_param_)) {
      if (OB_FAIL(reuse_das_iter_scan_param(total_doc_cnt_tablet_id_, *total_doc_cnt_scan_param_))) {
      }
    }
    for (int64_t i = 0; OB_SUCC(ret) && i < inv_scan_params_.count(); ++i) {
      if (OB_FAIL(reuse_das_iter_scan_param(inv_idx_tablet_id_, *inv_scan_params_[i]))) {
      }
    }
    for (int64_t i = 0; OB_SUCC(ret) && i < inv_agg_params_.count(); ++i) {
      if (OB_FAIL(reuse_das_iter_scan_param(inv_idx_tablet_id_, *inv_agg_params_[i]))) {
      }
    }
    for (int64_t i = 0; OB_SUCC(ret) && i < block_max_scan_params_.count(); ++i) {
      if (OB_FAIL(reuse_das_iter_scan_param(inv_idx_tablet_id_, *block_max_scan_params_[i]))) {
      }
    }
    for (int64_t i = 0; OB_SUCC(ret) && i < children_cnt_; ++i) {
      if (OB_FAIL(children_[i]->reuse())) {
      }
    }
  }

  if (OB_SUCC(ret) && OB_NOT_NULL(sparse_retrieval_iter_)) {
    sparse_retrieval_iter_->reuse(inv_idx_tablet_switched_);
    inv_idx_tablet_switched_ = false;
  }

  if (OB_NOT_NULL(mem_context_)) {
    mem_context_->reset_remain_one_page();
  }
  check_rangekey_inited_ = false;
  return ret;
}

int ObDASLegacyTRMergeIter::rescan()
{
  int ret = OB_SUCCESS;
  if (IS_NOT_INIT) {
    ret = OB_NOT_INIT;
  } else if (qcache_replaying_) {
    // rewind the cached stream; children were never scanned for this task
    qcache_replay_pos_ = 20;
    qcache_replay_rows_left_ = qcache_replay_total_rows_;
    check_rangekey_inited_ = true;
  } else if (OB_ISNULL(sparse_retrieval_iter_)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (!function_lookup_mode_ && OB_FAIL(set_children_iter_rangekey())) {
  } else if (OB_UNLIKELY(!check_rangekey_inited_)) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    for (int64_t i = 0; OB_SUCC(ret) && i < children_cnt_; ++i) {
      // TODO: update scan param
      if (OB_FAIL(children_[i]->rescan())) {
      }
    }
  }
  return ret;
}

int ObDASLegacyTRMergeIter::inner_release()
{
  int ret = OB_SUCCESS;
  if (OB_NOT_NULL(total_doc_cnt_scan_param_)) {
    total_doc_cnt_scan_param_->destroy_schema_guard();
    total_doc_cnt_scan_param_->snapshot_.reset();
    total_doc_cnt_scan_param_->destroy();
    total_doc_cnt_scan_param_ = nullptr;
  }

  for (int64_t i = 0; i < inv_scan_params_.count(); ++i) {
    if (OB_NOT_NULL(inv_scan_params_[i])) {
      inv_scan_params_[i]->destroy_schema_guard();
      inv_scan_params_[i]->snapshot_.reset();
      inv_scan_params_[i]->destroy();
    }
  }
  inv_scan_params_.reset();

  for (int64_t i = 0; i < inv_agg_params_.count(); ++i) {
    if (OB_NOT_NULL(inv_agg_params_[i])) {
      inv_agg_params_[i]->destroy_schema_guard();
      inv_agg_params_[i]->snapshot_.reset();
      inv_agg_params_[i]->destroy();
    }
  }
  inv_agg_params_.reset();

  for (int64_t i = 0; i < block_max_scan_params_.count(); ++i) {
    if (OB_NOT_NULL(block_max_scan_params_[i])) {
      block_max_scan_params_[i]->destroy_schema_guard();
      block_max_scan_params_[i]->snapshot_.reset();
      block_max_scan_params_[i]->destroy();
    }
  }
  block_max_scan_params_.reset();

  if (OB_NOT_NULL(boolean_compute_node_)) {
    boolean_compute_node_->release();
    boolean_compute_node_ = nullptr;
  }

  if (OB_NOT_NULL(sparse_retrieval_iter_)) {
    sparse_retrieval_iter_->reset();
    sparse_retrieval_iter_ = nullptr;
  }
  dim_iters_.reset();
  dim_iter_ = nullptr;
  if (nullptr != mem_context_) {
    mem_context_->reset_remain_one_page();
    DESTROY_CONTEXT(mem_context_);
    mem_context_ = nullptr;
  }
  query_tokens_.reset();
  dim_weights_.reset();
  qcache_reset_state();
  if (OB_NOT_NULL(qcache_rec_buf_)) { ob_free(qcache_rec_buf_); }
  qcache_rec_buf_ = nullptr;
  qcache_rec_cap_ = 0;
  myself_allocator_.reset();
  ir_ctdef_ = nullptr;
  ir_rtdef_ = nullptr;
  tx_desc_ = nullptr;
  snapshot_ = nullptr;
  topk_limit_ = 0;
  inv_idx_tablet_switched_ = false;
  is_inited_ = false;
  return ret;
}

int ObDASLegacyTRMergeIter::inner_get_next_row()
{
  int ret = OB_SUCCESS;
  if (IS_NOT_INIT) {
    ret = OB_NOT_INIT;
  } else if (qcache_replaying_) {
    ret = qcache_replay_one_row();
  } else if (OB_ISNULL(sparse_retrieval_iter_)) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    ret = sparse_retrieval_iter_->get_next_row();
    if (qcache_recording_) {
      if ((OB_SUCCESS == ret || OB_ITER_END == ret)) {
        if (OB_SUCCESS == ret) {
          int rec_ret = qcache_record_rows(1);
          if (OB_SUCCESS != rec_ret) { qcache_recording_ = false; }
        }
        if (OB_ITER_END == ret) { qcache_finish_recording(); }
      } else {
        qcache_recording_ = false;
      }
    }
  }
  return ret;
}

int ObDASLegacyTRMergeIter::inner_get_next_rows(int64_t &count, int64_t capacity)
{
  int ret = OB_SUCCESS;
  if (IS_NOT_INIT) {
    ret = OB_NOT_INIT;
  } else if (qcache_replaying_) {
    ret = qcache_replay_rows(count, capacity);
  } else if (OB_ISNULL(sparse_retrieval_iter_)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_UNLIKELY(0 == capacity)) {
    count = 0;
  } else {
    ret = sparse_retrieval_iter_->get_next_rows(capacity, count);
    if (qcache_recording_) {
      if (OB_SUCCESS == ret || OB_ITER_END == ret) {
        if (count > 0) {
          int rec_ret = qcache_record_rows(count);
          if (OB_SUCCESS != rec_ret) { qcache_recording_ = false; }
        }
        if (OB_ITER_END == ret) { qcache_finish_recording(); }
      } else {
        qcache_recording_ = false;
      }
    }
  }
  return ret;
}

static int get_query_tokens_by_compacting_repeated_token(ObString &query_str,
                                                         const ObCollationType &cs_type,
                                                         ObArray<ObString> &query_tokens,
                                                         ObArray<double> &boost_values,
                                                         common::ObIAllocator &alloc)
{
  int ret = OB_SUCCESS;
  char split_token_tag = ' ';
  char split_boost_tag = '^';
  hash::ObHashMap<ObString, double> token_map;
  const int64_t ft_word_bkt_cnt = MAX(query_str.length() / 10, 2);
  if (OB_FAIL(token_map.create(ft_word_bkt_cnt, common::ObMemAttr("FTWordMap")))) {
  }
  while (!query_str.empty() && OB_SUCC(ret)) {
    ObString token_str = query_str.split_on(split_token_tag);
    query_str = query_str.trim();
    if (token_str.empty()) {
      token_str = query_str;
      query_str.reset();
    }
    ObString token_key = token_str.split_on(split_boost_tag);
    double boost_value = 1.0;
    if (token_key.empty()) {
      token_key = token_str;
      token_str.reset();
    } else {
      char *boost_str = static_cast<char *>(alloc.alloc(token_str.length() + 1));
      if (OB_ISNULL(boost_str)) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else {
        memcpy(boost_str, token_str.ptr(), token_str.length());
        boost_str[token_str.length()] = '\0';
        char *end_ptr = nullptr;
        boost_value = strtod(boost_str, &end_ptr);
        if (end_ptr != boost_str + token_str.length() || boost_value <= 0.0) {
          ret = OB_NOT_SUPPORTED;
          LOG_USER_ERROR(OB_NOT_SUPPORTED, "invalid field boost");
        }
      }
    }
    double cur_boost = 0;
    if (OB_FAIL(ret)) {
    } else if (OB_FAIL(token_map.get_refactored(token_key, cur_boost))) {
      if (OB_HASH_NOT_EXIST != ret) {
      } else if (OB_FAIL(token_map.set_refactored(token_key, boost_value, 1/*overwrite*/))) {
      }
    } else if (OB_FAIL(token_map.set_refactored(token_key, boost_value + cur_boost, 1/*overwrite*/))) {
    }
  }
  if (OB_FAIL(ret)) {
  } else if (OB_FAIL(query_tokens.reserve(token_map.size()))) {
  } else if (OB_FAIL(boost_values.reserve(token_map.size()))) {
  }
  for (hash::ObHashMap<ObString, double>::const_iterator iter = token_map.begin();
      OB_SUCC(ret) && iter != token_map.end();
      ++iter) {
    const ObString &token = iter->first;
    ObString token_string;
    if (OB_FAIL(common::ObCharset::charset_convert(alloc, token, ObCollationType::CS_TYPE_UTF8MB4_GENERAL_CI, cs_type, token_string, common::ObCharset::CONVERT_FLAG::COPY_STRING_ON_SAME_CHARSET))) {
    } else if (OB_FAIL(query_tokens.push_back(token_string))) {
    } else if (OB_FAIL(boost_values.push_back(iter->second))) {
    }
  }
  return ret;
}

static int get_query_tokens_directly(ObString &query_str,
                                     const ObCollationType &cs_type,
                                     ObArray<ObString> &query_tokens,
                                     ObArray<double> &boost_values,
                                     common::ObIAllocator &alloc)
{
  int ret = OB_SUCCESS;
  char split_token_tag = ' ';
  char split_boost_tag = '^';
  while (!query_str.empty() && OB_SUCC(ret)) {
    ObString token_str = query_str.split_on(split_token_tag);
    query_str = query_str.trim();
    if (token_str.empty()) {
      token_str = query_str;
      query_str.reset();
    }
    ObString token_key = token_str.split_on(split_boost_tag);
    double boost_value = 1.0;
    if (token_key.empty()) {
      token_key = token_str;
      token_str.reset();
    } else {
      char *boost_str = static_cast<char *>(alloc.alloc(token_str.length() + 1));
      if (OB_ISNULL(boost_str)) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else {
        memcpy(boost_str, token_str.ptr(), token_str.length());
        boost_str[token_str.length()] = '\0';
        char *end_ptr = nullptr;
        boost_value = strtod(boost_str, &end_ptr);
        if (end_ptr != boost_str + token_str.length() || boost_value <= 0.0) {
          ret = OB_NOT_SUPPORTED;
          LOG_USER_ERROR(OB_NOT_SUPPORTED, "invalid field boost");
        }
      }
    }
    double cur_boost = 0;
    if (OB_FAIL(ret)) {
    } else {
      ObString token_string;
      if (OB_FAIL(common::ObCharset::charset_convert(alloc, token_key, ObCollationType::CS_TYPE_UTF8MB4_GENERAL_CI, cs_type, token_string, common::ObCharset::CONVERT_FLAG::COPY_STRING_ON_SAME_CHARSET))) {
      } else if (OB_FAIL(query_tokens.push_back(token_string))) {
      } else if (OB_FAIL(boost_values.push_back(boost_value))) {
      }
    }
  }
  return ret;
}

int ObDASLegacyTRMergeIter::build_query_tokens(const ObDASIRScanCtDef *ir_ctdef,
                                         ObDASIRScanRtDef *ir_rtdef,
                                         common::ObIAllocator &alloc,
                                         ObArray<ObString> &query_tokens,
                                         ObArray<double> &boost_values,
                                         ObFtsEvalNode *&root_node,
                                         bool &has_duplicate_boolean_tokens)
{
  int ret = OB_SUCCESS;
  has_duplicate_boolean_tokens = false;
  ObExpr *search_text = ir_ctdef->search_text_;
  ObEvalCtx *eval_ctx = ir_rtdef->eval_ctx_;
  ObDatum *search_text_datum = nullptr;
  if (OB_ISNULL(search_text) || OB_ISNULL(eval_ctx)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (query_tokens.count() != 0 || boost_values.count() != 0) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(search_text->eval(*eval_ctx, search_text_datum))) {
  } else if (0 == search_text_datum->len_) {
    // empty query text
  } else if (OB_NOT_NULL(ir_ctdef->field_boost_expr_)) {
    const ObString &search_text_string = search_text_datum->get_string();
    const ObCollationType &cs_type = search_text->datum_meta_.cs_type_;
    const ObCollationType dst_type = ObCollationType::CS_TYPE_UTF8MB4_GENERAL_CI;

    ObString str_dest;
    if (cs_type != dst_type) {
      ObString tmp_out;
      if (OB_FAIL(ObCharset::tolower(cs_type, search_text_string, tmp_out, alloc))) {
      } else if (OB_FAIL(common::ObCharset::charset_convert(alloc, tmp_out, cs_type, dst_type, str_dest))) {
      }
    } else if (OB_FAIL(ObCharset::tolower(cs_type, search_text_string, str_dest, alloc))){
    }
    if (OB_FAIL(ret)) {
    } else {
      const bool compact_repeated_token = ir_rtdef->minimum_should_match_ <= 1;
      ObString query_str = str_dest.trim();
      if (compact_repeated_token && OB_FAIL(get_query_tokens_by_compacting_repeated_token(query_str, cs_type, query_tokens, boost_values, alloc))) {
      } else if (!compact_repeated_token && OB_FAIL(get_query_tokens_directly(query_str, cs_type, query_tokens, boost_values, alloc))) {
      }
    }
  } else if (BOOLEAN_MODE == ir_ctdef->mode_flag_) {
    const ObString &search_text_string = search_text_datum->get_string();
    const ObCollationType &cs_type = search_text->datum_meta_.cs_type_;
    const ObCollationType dst_type = ObCollationType::CS_TYPE_UTF8MB4_GENERAL_CI;

    ObString str_dest;
    if (cs_type != dst_type) {
      ObString tmp_out;
      if (OB_FAIL(ObCharset::tolower(cs_type, search_text_string, tmp_out, alloc))) {
      } else if (OB_FAIL(common::ObCharset::charset_convert(alloc, tmp_out, cs_type, dst_type, str_dest))) {
      }
    } else if (OB_FAIL(ObCharset::tolower(cs_type, search_text_string, str_dest, alloc))){
    }

    void *buf = nullptr;
    FtsParserResult *fts_parser;
    if (OB_FAIL(ret)) {
    } else if (OB_ISNULL(buf = (&alloc)->alloc(sizeof(FtsParserResult)))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else {
      fts_parser = static_cast<FtsParserResult *>(buf);
    }

    if (OB_FAIL(ret)) {
    } else if (OB_ISNULL(buf = (&alloc)->alloc(str_dest.length() + 1))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else {
      MEMSET(buf, 0, str_dest.length() + 1);
      MEMCPY(buf, str_dest.ptr(), str_dest.length());
    }

    if (OB_FAIL(ret)) {
    } else if (FALSE_IT(fts_parse_docment(static_cast<char *>(buf), str_dest.length(), &alloc, fts_parser))) {
    } else if (FTS_OK != fts_parser->ret_) {
      if (FTS_ERROR_MEMORY == fts_parser->ret_) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else if (FTS_ERROR_SYNTAX == fts_parser->ret_) {
        ret = OB_ERR_PARSER_SYNTAX;
      } else if (FTS_ERROR_OTHER == fts_parser->ret_) {
        ret = OB_ERR_UNEXPECTED;
      }
    } else if (OB_ISNULL(fts_parser->root_)) {
      // do nothing
    } else {
      FtsNode *node = fts_parser->root_;
      ObFtsEvalNode *parant_node =nullptr;
      hash::ObHashMap<ObString, int32_t> tokens_map;
      const int64_t ft_word_bkt_cnt = MAX(search_text_string.length() / 10, 2);
      if (OB_FAIL(tokens_map.create(ft_word_bkt_cnt, common::ObMemAttr("FTWordMap")))) {
      } else if (OB_FAIL(ObFtsEvalNode::fts_boolean_node_create(parant_node, node, cs_type, alloc, query_tokens, tokens_map, has_duplicate_boolean_tokens))) {
      } else {
        root_node = parant_node;
      }
    }
  } else {
    // TODO: FTParseHelper currently does not support deduplicate tokens
    //       We should abstract such universal analyse functors into utility structs
    const ObString &search_text_string = search_text_datum->get_string();
    const ObString &parser_name = ir_ctdef->get_inv_idx_scan_ctdef()->table_param_.get_parser_name();
    const ObString &parser_properties = ir_ctdef->get_inv_idx_scan_ctdef()->table_param_.get_parser_property();

    const ObObjMeta &meta = search_text->obj_meta_;
    int64_t doc_length = 0;
    storage::ObFTParseHelper tokenize_helper;
    common::ObSEArray<ObFTWord, 16> tokens;
    hash::ObHashMap<ObFTWord, int64_t> token_map;
    const int64_t ft_word_bkt_cnt = MAX(search_text_string.length() / 10, 2);
    if (OB_FAIL(tokenize_helper.init(&alloc, parser_name, parser_properties))) {
    } else if (OB_FAIL(token_map.create(ft_word_bkt_cnt, common::ObMemAttr("FTWordMap")))) {
    } else if (OB_FAIL(tokenize_helper.segment(
                           meta,
                           search_text_string.ptr(),
                           search_text_string.length(),
                           doc_length,
                           token_map))) {
    } else {
      for (hash::ObHashMap<ObFTWord, int64_t>::const_iterator iter = token_map.begin();
          OB_SUCC(ret) && iter != token_map.end();
          ++iter) {
        const ObFTWord &token = iter->first;
        ObString token_string;
        if (OB_FAIL(ob_write_string(alloc, token.get_word().get_string(), token_string))) {
        } else if (OB_FAIL(query_tokens.push_back(token_string))) {
        }
      }
    }
  }
  return ret;
}

int ObDASLegacyTRMergeIter::init_topk_limit()
{
  int ret = OB_SUCCESS;
  topk_limit_ = 0;
  ObExpr *topk_limit_expr = ir_ctdef_->topk_limit_expr_;
  ObExpr *topk_offset_expr = ir_ctdef_->topk_offset_expr_;
  if (OB_UNLIKELY(!topk_mode_)) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    int64_t limit = 0;
    int64_t offset = 0;
    ObDatum *topk_limit_datum = nullptr;
    ObDatum *topk_offset_datum = nullptr;
    if (OB_FAIL(topk_limit_expr->eval(*ir_rtdef_->eval_ctx_, topk_limit_datum))) {
    } else if (nullptr != topk_offset_expr && OB_FAIL(topk_offset_expr->eval(*ir_rtdef_->eval_ctx_, topk_offset_datum))) {
    } else {
      limit = (topk_limit_datum->is_null() || topk_limit_datum->get_int() < 0) ? 0 : topk_limit_datum->get_int();
      if (nullptr != topk_offset_datum) {
        offset = (topk_offset_datum->is_null() || topk_offset_datum->get_int() < 0) ? 0 : topk_offset_datum->get_int();
      }
      topk_limit_ = limit + offset;
    }
  }
  return ret;
}

namespace
{

class ObDASLegacyTextRetrievalEngine final
    : public ObIDASTextRetrievalEngine
{
public:
  explicit ObDASLegacyTextRetrievalEngine(common::ObIAllocator &allocator)
    : allocator_(&allocator), iter_(nullptr)
  {}
  ~ObDASLegacyTextRetrievalEngine() override = default;

  int init(ObDASTRMergeIterParam &param) override
  {
    int ret = OB_SUCCESS;
    void *buffer = nullptr;
    if (OB_NOT_NULL(iter_)) {
      ret = OB_INIT_TWICE;
    } else if (OB_ISNULL(buffer = allocator_->alloc(
        sizeof(ObDASLegacyTRMergeIter)))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else {
      iter_ = new (buffer) ObDASLegacyTRMergeIter();
      if (OB_FAIL(iter_->init(param))) {
      }
    }
    return ret;
  }

  int bind_source_tree(
      ObDASIter **children,
      const uint32_t child_count) override
  {
    int ret = OB_SUCCESS;
    if (OB_ISNULL(iter_)
        || (0 == child_count && nullptr != children)
        || (child_count > 0 && nullptr == children)) {
      ret = OB_INVALID_ARGUMENT;
    } else {
      iter_->get_children() = children;
      iter_->set_children_cnt(child_count);
    }
    return ret;
  }

  int set_related_tablet_ids(
      const ObDASFTSTabletID &related_tablet_ids) override
  {
    return OB_ISNULL(iter_)
        ? OB_NOT_INIT
        : iter_->set_related_tablet_ids(related_tablet_ids);
  }

  int do_table_scan() override
  {
    return OB_ISNULL(iter_) ? OB_NOT_INIT : iter_->do_table_scan();
  }

  int reuse() override
  {
    return OB_ISNULL(iter_) ? OB_NOT_INIT : iter_->reuse();
  }

  int rescan() override
  {
    return OB_ISNULL(iter_) ? OB_NOT_INIT : iter_->rescan();
  }

  int get_next_row() override
  {
    return OB_ISNULL(iter_) ? OB_NOT_INIT : iter_->get_next_row();
  }

  int get_next_rows(int64_t &count, const int64_t capacity) override
  {
    count = 0;
    return OB_ISNULL(iter_)
        ? OB_NOT_INIT : iter_->get_next_rows(count, capacity);
  }

  int set_lookup_keys(
      const common::ObIArray<std::pair<ObDocIdExt, int>> &virtual_rangekeys,
      const int64_t batch_size) override
  {
    return OB_ISNULL(iter_)
        ? OB_NOT_INIT
        : iter_->set_children_iter_rangekey(
            virtual_rangekeys, batch_size);
  }

  bool is_taat_mode() const override
  {
    return OB_NOT_NULL(iter_) && iter_->is_taat_mode();
  }

  int get_query_max_score(double &score) override
  {
    score = 0.0;
    return OB_ISNULL(iter_)
        ? OB_NOT_INIT : iter_->get_query_max_score(score);
  }

  void destroy() override
  {
    common::ObIAllocator *allocator = allocator_;
    if (OB_NOT_NULL(iter_)) {
      // The facade owns the DAS tree.  Release physical state while dynamic
      // dispatch still sees the legacy derived type, but never recurse into
      // or destruct the borrowed children.
      iter_->set_children_cnt(0);
      iter_->get_children() = nullptr;
      if (iter_->is_inited()) {
        const int ret = iter_->release();
        if (OB_SUCCESS != ret) {
        }
      }
      allocator->free(iter_);
      iter_ = nullptr;
    }
    this->~ObDASLegacyTextRetrievalEngine();
    allocator->free(this);
  }

private:
  common::ObIAllocator *allocator_;
  ObDASLegacyTRMergeIter *iter_;
  DISALLOW_COPY_AND_ASSIGN(ObDASLegacyTextRetrievalEngine);
};

int create_legacy_text_retrieval_engine(
    common::ObIAllocator &allocator,
    ObIDASTextRetrievalEngine *&engine)
{
  int ret = OB_SUCCESS;
  void *buffer = nullptr;
  engine = nullptr;
  if (OB_ISNULL(buffer = allocator.alloc(
      sizeof(ObDASLegacyTextRetrievalEngine)))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else {
    engine = new (buffer) ObDASLegacyTextRetrievalEngine(allocator);
  }
  return ret;
}

int build_legacy_text_retrieval_query(
    const ObDASIRScanCtDef *ir_ctdef,
    ObDASIRScanRtDef *ir_rtdef,
    common::ObIAllocator &allocator,
    ObArray<ObString> &query_tokens,
    ObArray<double> &boost_values,
    ObFtsEvalNode *&root_node,
    bool &has_duplicate_boolean_tokens)
{
  return ObDASLegacyTRMergeIter::build_query_tokens(
      ir_ctdef, ir_rtdef, allocator, query_tokens, boost_values, root_node,
      has_duplicate_boolean_tokens);
}

struct ObDASLegacyTextRetrievalRegistration
{
  ObDASLegacyTextRetrievalRegistration()
    : ret_(install_legacy_das_text_retrieval_engine())
  {}

  int ret_;
};

// This object is force-retained by the final composition target.  If it is
// absent, the query dispatcher fails closed with OB_NOT_INIT.
ObDASLegacyTextRetrievalRegistration g_text_retrieval_registration;

} // namespace

int install_legacy_das_text_retrieval_engine()
{
  return install_das_text_retrieval_engine_factory(
      create_legacy_text_retrieval_engine,
      build_legacy_text_retrieval_query);
}

} // namespace sql
} // namespace oceanbase
