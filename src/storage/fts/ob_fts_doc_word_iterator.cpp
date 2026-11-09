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

#define USING_LOG_PREFIX STORAGE_FTS

#include "storage/fts/ob_fts_doc_word_iterator.h"
#include "data_plane/fts/ob_fts_doc_word_scan.h"
#include "share/rc/ob_server_runtime.h"

#include "storage/access/ob_table_scan_iterator.h"
#include "storage/tx_storage/ob_access_service.h"

namespace oceanbase
{
namespace storage
{

ObFTDocWordScanIterator::ObFTDocWordScanIterator()
  : allocator_(lib::ObMemAttr("FTDWAlloc")),
    scan_allocator_(lib::ObMemAttr("FTDWScanA")),
    table_param_(allocator_),
    scan_param_(),
    doc_word_iter_(nullptr),
    docid_type_(ObDocIDType::INVALID),
    is_inited_(false),
    seq_opened_(false),
    seq_broken_(false),
    seq_stream_end_(false),
    pending_valid_(false),
    pending_row_(),
    doc_alloc_(lib::ObMemAttr("FTDWSeqDoc")),
    pending_alloc_(lib::ObMemAttr("FTDWSeqPend")),
    seq_rows_(),
    seq_idx_(0),
    seq_serving_(0),
    last_doc_len_(0),
    last_doc_valid_(false),
    seq_out_row_(),
    seq_out_row_inited_(false)
{
}

ObFTDocWordScanIterator::~ObFTDocWordScanIterator()
{
  reset();
}

int ObFTDocWordScanIterator::init(
    const uint64_t table_id,
    const common::ObTabletID &tablet_id,
    const transaction::ObTxReadSnapshot *snapshot,
    const int64_t schema_version)
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(is_inited_)) {
    ret = OB_INIT_TWICE;
  } else if (OB_FAIL(init_scan_param(table_id, tablet_id, snapshot, schema_version))) {
  } else {
    is_inited_ = true;
  }
  if (OB_UNLIKELY(!is_inited_)) {
    reset();
  }
  return ret;
}

void ObFTDocWordScanIterator::reset()
{
  if (OB_NOT_NULL(doc_word_iter_)) {
    ::oceanbase::share::server_service<::oceanbase::storage::ObAccessService>()->revert_scan_iter(doc_word_iter_);
    doc_word_iter_ = nullptr;
  }
  table_param_.reset();
  // table_param_ owns main_read_info_ allocated from allocator_.  The pointer
  // remains part of table_param_ after reset and is reused by the next scan,
  // so allocator_ must live until table_param_ is destroyed.
  scan_allocator_.reset();
  seq_opened_ = false;
  seq_broken_ = false;
  seq_stream_end_ = false;
  pending_valid_ = false;
  pending_alloc_.reset();
  doc_alloc_.reset();
  seq_rows_.reuse();
  seq_idx_ = 0;
  seq_serving_ = 0;
  last_doc_len_ = 0;
  last_doc_valid_ = false;
  is_inited_ = false;
}

int ObFTDocWordScanIterator::reuse()
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(doc_word_iter_)) {
    ret = OB_ERR_UNDEFINED;
  } else if (OB_FAIL(::oceanbase::share::server_service<::oceanbase::storage::ObAccessService>()->reuse_scan_iter(false/*switch param*/, doc_word_iter_))) {
  } else {
    scan_param_.key_ranges_.reuse();
    scan_param_.mbr_filters_.reuse();
    scan_allocator_.reset_remain_one_page();
  }
  return ret;
}

int ObFTDocWordScanIterator::do_table_rescan()
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(!is_inited_)) {
    ret = OB_NOT_INIT;
  } else if (OB_ISNULL(doc_word_iter_)) {
    ret = OB_ERR_UNDEFINED;
  } else if (OB_FAIL(::oceanbase::share::server_service<::oceanbase::storage::ObAccessService>()->table_rescan(scan_param_, doc_word_iter_))) {
  }
#ifdef OB_BUILD_PACKAGE
  LOG_TRACE("doc word rescan", K(ret), K(scan_param_));
#else
  LOG_INFO("doc word rescan", K(ret), K(scan_param_));
#endif
  return ret;
}

int ObFTDocWordScanIterator::do_scan(const uint64_t table_id, const ObDatum &doc_id_datum)
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(!is_inited_)) {
    ret = OB_NOT_INIT;
  } else if (OB_UNLIKELY(OB_INVALID_ID == table_id)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    // v3: prefer one whole-range sequential scan consumed document by
    // document; regress to per-document rescan once the input order breaks.
    if (!seq_opened_ && !seq_broken_) {
      if (OB_FAIL(open_seq_scan(table_id))) {
        LOG_WARN("failed to open sequential doc word scan, fallback to rescan", K(ret));
        seq_broken_ = true;
        ret = OB_SUCCESS;
      } else {
        seq_opened_ = true;
      }
    }
    if (seq_opened_ && !seq_broken_) {
      bool order_ok = true;
      if (last_doc_valid_) {
        ObDatum last_datum;
        last_datum.set_string(last_doc_buf_, static_cast<int64_t>(last_doc_len_));
        order_ok = cmp_doc_id(doc_id_datum, last_datum) >= 0;
      }
      if (order_ok) {
        if (OB_FAIL(drain_to_doc(doc_id_datum))) {
          if (OB_ITER_END != ret) {
            LOG_WARN("sequential doc word stream broken, fallback to rescan", K(ret));
            seq_broken_ = true;
            ret = OB_SUCCESS;
          }
        }
        if (OB_SUCC(ret) || OB_ITER_END == ret) {
          // OB_ITER_END here only means "no rows for this document", the
          // stream itself stays usable; normalize for the caller.
          ret = OB_SUCCESS;
          if (last_doc_len_ != doc_id_datum.len_
              || 0 != MEMCMP(last_doc_buf_, doc_id_datum.ptr_, doc_id_datum.len_)) {
            if (OB_UNLIKELY(doc_id_datum.len_ > 64)) {
              // oversized doc id, stay safe on the legacy path
              seq_broken_ = true;
            } else {
              last_doc_len_ = doc_id_datum.len_;
              MEMCPY(last_doc_buf_, doc_id_datum.ptr_, doc_id_datum.len_);
              last_doc_valid_ = true;
            }
          }
          if (!seq_broken_) {
            seq_serving_ = 1;
            return OB_SUCCESS;
          }
        }
      } else {
        seq_broken_ = true;
      }
    }
    if (OB_FAIL(serve_doc_by_rescan(table_id, doc_id_datum))) {
      if (OB_ITER_END == ret) {
        // an empty document is not an error for the caller
        ret = OB_SUCCESS;
      }
    }
  }
  return ret;
}

int ObFTDocWordScanIterator::get_next_row(blocksstable::ObDatumRow *&datum_row)
{
  int ret = OB_SUCCESS;
  if (1 == seq_serving_) {
    // serve from the per-document sequential buffer
    if (OB_UNLIKELY(!seq_out_row_inited_)) {
      if (OB_FAIL(seq_out_row_.init(4))) {
        STORAGE_FTS_LOG(WARN, "failed to init seq out row", K(ret));
      } else {
        seq_out_row_inited_ = true;
      }
    }
    if (OB_FAIL(ret)) {
    } else if (seq_idx_ >= seq_rows_.count()) {
      ret = OB_ITER_END;
    } else {
      const SeqWordRow &r = seq_rows_.at(seq_idx_);
      seq_out_row_.storage_datums_[0].shallow_copy_from_datum(r.doc_id_);
      seq_out_row_.storage_datums_[1].shallow_copy_from_datum(r.word_);
      seq_out_row_.storage_datums_[2].set_uint(r.cnt_);
      seq_out_row_.storage_datums_[3].set_uint(r.doc_len_);
      datum_row = &seq_out_row_;
      ++seq_idx_;
    }
  } else {
    ObTableScanIterator *tsc_iter = nullptr;
    if (OB_ISNULL(doc_word_iter_)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (FALSE_IT(tsc_iter = static_cast<ObTableScanIterator *>(doc_word_iter_))) {
    } else if (OB_FAIL(tsc_iter->get_next_row(datum_row))) {
      if (OB_ITER_END != ret) {
      }
    }
  }
  return ret;
}

int ObFTDocWordScanIterator::init_scan_param(
    const uint64_t table_id,
    const common::ObTabletID &tablet_id,
    const transaction::ObTxReadSnapshot *snapshot,
    const int64_t schema_version)
{
  int ret = OB_SUCCESS;
  ObQueryFlag query_flag(ObQueryFlag::Forward, // scan_order
                         false, // daily_merge
                         false, // optimize
                         false, // sys scan
                         false, // full_row
                         false, // index_back
                         false, // query_stat
                         false // read_latest
                        );
  if (OB_UNLIKELY(OB_INVALID_ID == table_id
              || !tablet_id.is_valid()
              || nullptr == snapshot
              || schema_version < 0)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(build_table_param(table_id, table_param_, scan_param_.column_ids_))) {
  } else {
    scan_param_.tablet_id_ = tablet_id;
    scan_param_.schema_version_ = schema_version;
    scan_param_.is_get_ = false;
    scan_param_.scan_flag_.flag_ = query_flag.flag_;
    scan_param_.key_ranges_.set_attr(ObMemAttr("ScanParamKR"));
    scan_param_.index_id_ = 0;
    scan_param_.timeout_ = THIS_WORKER.get_timeout_ts();
    scan_param_.reserved_cell_count_ = scan_param_.column_ids_.count();
    scan_param_.limit_param_.limit_ = -1;
    scan_param_.limit_param_.offset_ = 0;
    scan_param_.sql_mode_ = SMO_DEFAULT;
    scan_param_.allocator_ = &allocator_;
    scan_param_.for_update_ = false;
    scan_param_.for_update_wait_timeout_ = scan_param_.timeout_;
    scan_param_.scan_allocator_ = &scan_allocator_;
    scan_param_.frozen_version_ = -1;
    scan_param_.output_exprs_ = nullptr;
    scan_param_.aggregate_exprs_ = nullptr;
    scan_param_.op_ = nullptr;
    scan_param_.row2exprs_projector_ = nullptr;
    scan_param_.need_scn_ = false;
    scan_param_.pd_storage_flag_ = false;
    scan_param_.table_param_ = &table_param_;
    scan_param_.key_ranges_.reset();
    if (OB_FAIL(scan_param_.snapshot_.assign(*snapshot))) {
    } else {
      scan_param_.tx_id_ = scan_param_.snapshot_.core_.tx_id_;
    }
  }
  return ret;
}

int ObFTDocWordScanIterator::build_table_param(
    const uint64_t table_id,
    share::schema::ObTableParam &table_param,
    common::ObIArray<uint64_t> &column_ids)
{
  int ret = OB_SUCCESS;
  
  share::schema::ObSchemaGetterGuard schema_guard;
  const share::schema::ObTableSchema *table_schema = nullptr;
  uint64_t generated_doc_id_col = OB_INVALID_ID;
  column_ids.reset();
  if (OB_UNLIKELY(OB_INVALID_ID == table_id)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(ObMultiVersionSchemaService::get_instance().get_runtime_schema_guard(schema_guard))) {
  } else if (OB_FAIL(schema_guard.get_table_schema( table_id, table_schema))) {
  } else if (OB_ISNULL(table_schema)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(table_schema->get_column_ids(column_ids))) {
  } else if (OB_UNLIKELY(4 != column_ids.count())) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(table_param.convert(*table_schema, column_ids, sql::ObStoragePushdownFlag()))) {
  } else {
    if (OB_FAIL(table_schema->get_docid_col_id(generated_doc_id_col))) {
      if (ret == OB_ERR_INDEX_KEY_NOT_FOUND) {
        ret = OB_SUCCESS;
        docid_type_ = ObDocIDType::HIDDEN_INC_PK;
      } else {
      }
    } else {
      docid_type_ = ObDocIDType::TABLET_SEQUENCE;
    }
  }
  return ret;
}

int ObFTDocWordScanIterator::build_key_range(const uint64_t table_id,
                                             const ObDatum &doc_id_datum,
                                             common::ObIArray<ObNewRange> &rowkey_range)
{
  int ret = OB_SUCCESS;
  static int64_t ROWKEY_COLUMN_COUNT = 2;
  common::ObNewRange range;
  common::ObObj *objs = nullptr;
  if (OB_UNLIKELY(OB_INVALID_ID == table_id)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_ISNULL(objs = OB_NEW_ARRAY(common::ObObj, &allocator_, ROWKEY_COLUMN_COUNT * 2))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else {
    if (docid_type_ == ObDocIDType::HIDDEN_INC_PK) {
      objs[0].set_uint64(doc_id_datum.get_uint64());
      objs[2].set_uint64(doc_id_datum.get_uint64());
    } else if (docid_type_ == ObDocIDType::TABLET_SEQUENCE) {
      objs[0].set_varbinary(doc_id_datum.get_string());
      objs[2].set_varbinary(doc_id_datum.get_string());
    }
    objs[1] = common::ObObj::make_min_obj();
    objs[3] = common::ObObj::make_max_obj();

    range.table_id_ = table_id;
    range.start_key_ = ObRowkey(objs, ROWKEY_COLUMN_COUNT);
    range.end_key_ = ObRowkey(objs + ROWKEY_COLUMN_COUNT, ROWKEY_COLUMN_COUNT);
    range.border_flag_.set_inclusive_start();
    range.border_flag_.set_inclusive_end();
    if (OB_FAIL(rowkey_range.push_back(range))) {
    }
  }
  return ret;
}

int ObFTDocWordScanIterator::build_whole_range(const uint64_t table_id,
                                               common::ObIArray<common::ObNewRange> &rowkey_range)
{
  int ret = OB_SUCCESS;
  static int64_t ROWKEY_COLUMN_COUNT = 2;
  common::ObNewRange range;
  common::ObObj *objs = nullptr;
  if (OB_UNLIKELY(OB_INVALID_ID == table_id)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_ISNULL(objs = OB_NEW_ARRAY(common::ObObj, &allocator_, ROWKEY_COLUMN_COUNT * 2))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else {
    objs[0] = common::ObObj::make_min_obj();
    objs[1] = common::ObObj::make_min_obj();
    objs[2] = common::ObObj::make_max_obj();
    objs[3] = common::ObObj::make_max_obj();
    range.table_id_ = table_id;
    range.start_key_ = ObRowkey(objs, ROWKEY_COLUMN_COUNT);
    range.end_key_ = ObRowkey(objs + ROWKEY_COLUMN_COUNT, ROWKEY_COLUMN_COUNT);
    range.border_flag_.set_inclusive_start();
    range.border_flag_.set_inclusive_end();
    if (OB_FAIL(rowkey_range.push_back(range))) {
    }
  }
  return ret;
}

int ObFTDocWordScanIterator::open_seq_scan(const uint64_t table_id)
{
  int ret = OB_SUCCESS;
  scan_param_.key_ranges_.reuse();
  if (OB_FAIL(build_whole_range(table_id, scan_param_.key_ranges_))) {
    STORAGE_FTS_LOG(WARN, "failed to build whole range", K(ret));
  } else if (OB_FAIL(do_table_scan())) {
    STORAGE_FTS_LOG(WARN, "failed to open whole range scan", K(ret));
  }
  return ret;
}

int ObFTDocWordScanIterator::cmp_doc_id(const ObDatum &l, const ObDatum &r)
{
  int cmp = 0;
  const int64_t min_len = MIN(l.len_, r.len_);
  if (0 != min_len) {
    cmp = MEMCMP(l.ptr_, r.ptr_, min_len);
  }
  if (0 == cmp) {
    if (l.len_ < r.len_) {
      cmp = -1;
    } else if (l.len_ > r.len_) {
      cmp = 1;
    }
  }
  return cmp;
}

int ObFTDocWordScanIterator::drain_to_doc(const ObDatum &doc_id_datum)
{
  int ret = OB_SUCCESS;
  ObTableScanIterator *tsc_iter = nullptr;
  if (OB_UNLIKELY(seq_stream_end_)) {
    // nothing left in the stream: this document simply has no rows
  } else if (OB_ISNULL(tsc_iter = static_cast<ObTableScanIterator *>(doc_word_iter_))) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    doc_alloc_.reuse();
    seq_rows_.reuse();
    seq_idx_ = 0;
    // resolve the pending row left by the previous document first
    if (pending_valid_) {
      if (0 == cmp_doc_id(pending_row_.doc_id_, doc_id_datum)) {
        // belongs to this document: copy it into the per-document buffer so
        // the pending arena can be recycled later
        char *doc_buf = static_cast<char *>(doc_alloc_.alloc(pending_row_.doc_id_.len_));
        char *word_buf = static_cast<char *>(doc_alloc_.alloc(pending_row_.word_.len_));
        if (OB_ISNULL(doc_buf) || OB_ISNULL(word_buf)) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
        } else {
          SeqWordRow dst;
          MEMCPY(doc_buf, pending_row_.doc_id_.ptr_, pending_row_.doc_id_.len_);
          MEMCPY(word_buf, pending_row_.word_.ptr_, pending_row_.word_.len_);
          dst.doc_id_.set_string(doc_buf, pending_row_.doc_id_.len_);
          dst.word_.set_string(word_buf, pending_row_.word_.len_);
          dst.cnt_ = pending_row_.cnt_;
          dst.doc_len_ = pending_row_.doc_len_;
          if (OB_FAIL(seq_rows_.push_back(dst))) {
          } else {
            pending_valid_ = false;
          }
        }
      } else if (cmp_doc_id(pending_row_.doc_id_, doc_id_datum) > 0) {
        // rows of a later document, keep pending and stop
      } else {
        // stale row of an already served document, drop it
        pending_valid_ = false;
      }
    }
    blocksstable::ObDatumRow *row = nullptr;
    while (OB_SUCC(ret) && !pending_valid_ && !seq_stream_end_) {
      if (OB_FAIL(tsc_iter->get_next_row(row))) {
        if (OB_ITER_END == ret) {
          seq_stream_end_ = true;
          ret = OB_SUCCESS;
        } else {
          STORAGE_FTS_LOG(WARN, "sequential doc word scan failed", K(ret));
        }
      } else if (OB_ISNULL(row) || OB_UNLIKELY(4 != row->count_)) {
        ret = OB_ERR_UNEXPECTED;
      } else {
        const int c = cmp_doc_id(row->storage_datums_[0], doc_id_datum);
        if (c < 0) {
          // row of an already served document, skip
        } else if (c > 0) {
          // row of a later document: park it in pending and stop. The stale
          // pending bytes are no longer referenced (consumed or dropped above)
          // so the arena can be recycled before the new copies are made.
          pending_alloc_.reuse();
          char *doc_buf = static_cast<char *>(pending_alloc_.alloc(row->storage_datums_[0].len_));
          char *word_buf = static_cast<char *>(pending_alloc_.alloc(row->storage_datums_[1].len_));
          if (OB_ISNULL(doc_buf) || OB_ISNULL(word_buf)) {
            ret = OB_ALLOCATE_MEMORY_FAILED;
          } else {
            MEMCPY(doc_buf, row->storage_datums_[0].ptr_, row->storage_datums_[0].len_);
            MEMCPY(word_buf, row->storage_datums_[1].ptr_, row->storage_datums_[1].len_);
            pending_row_.doc_id_.set_string(doc_buf, row->storage_datums_[0].len_);
            pending_row_.word_.set_string(word_buf, row->storage_datums_[1].len_);
            pending_row_.cnt_ = row->storage_datums_[2].get_uint();
            pending_row_.doc_len_ = row->storage_datums_[3].get_uint();
            pending_valid_ = true;
          }
        } else if (OB_FAIL(seq_rows_.push_back(SeqWordRow()))) {
        } else {
          SeqWordRow &dst = seq_rows_.at(seq_rows_.count() - 1);
          char *doc_buf = static_cast<char *>(doc_alloc_.alloc(row->storage_datums_[0].len_));
          char *word_buf = static_cast<char *>(doc_alloc_.alloc(row->storage_datums_[1].len_));
          if (OB_ISNULL(doc_buf) || OB_ISNULL(word_buf)) {
            ret = OB_ALLOCATE_MEMORY_FAILED;
          } else {
            MEMCPY(doc_buf, row->storage_datums_[0].ptr_, row->storage_datums_[0].len_);
            MEMCPY(word_buf, row->storage_datums_[1].ptr_, row->storage_datums_[1].len_);
            dst.doc_id_.set_string(doc_buf, row->storage_datums_[0].len_);
            dst.word_.set_string(word_buf, row->storage_datums_[1].len_);
            dst.cnt_ = row->storage_datums_[2].get_uint();
            dst.doc_len_ = row->storage_datums_[3].get_uint();
          }
        }
      }
    }
  }
  return ret;
}

int ObFTDocWordScanIterator::serve_doc_by_rescan(const uint64_t table_id, const ObDatum &doc_id_datum)
{
  int ret = OB_SUCCESS;
  const bool need_rescan = nullptr != doc_word_iter_;
  if (need_rescan) {
    reuse();
  }
  seq_serving_ = 2;
  if (FAILEDx(build_key_range(table_id, doc_id_datum, scan_param_.key_ranges_))) {
  } else if (need_rescan && OB_FAIL(do_table_rescan())) {
  } else if (!need_rescan && OB_FAIL(do_table_scan())) {
  }
  return ret;
}

int ObFTDocWordScanIterator::do_table_scan()
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(!is_inited_)) {
    ret = OB_NOT_INIT;
  } else if (OB_FAIL(::oceanbase::share::server_service<::oceanbase::storage::ObAccessService>()->table_scan(scan_param_, doc_word_iter_))) {
  }
#ifdef OB_BUILD_PACKAGE
  LOG_TRACE("doc word scan", K(ret), K(scan_param_));
#else
  LOG_INFO("doc word scan", K(ret), K(scan_param_));
#endif
  return ret;
}

} // end namespace storage

namespace data_plane
{
namespace
{
storage::ObFTDocWordScanIterator *as_storage_iterator(
    ObFTDocWordIterator *iterator)
{
  return reinterpret_cast<storage::ObFTDocWordScanIterator *>(iterator);
}
}

int create_ft_doc_word_iterator(common::ObIAllocator &allocator,
                                ObFTDocWordIterator *&iterator)
{
  int ret = OB_SUCCESS;
  void *buffer = allocator.alloc(sizeof(storage::ObFTDocWordScanIterator));
  if (OB_ISNULL(buffer)) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else {
    iterator = reinterpret_cast<ObFTDocWordIterator *>(
        new (buffer) storage::ObFTDocWordScanIterator());
  }
  return ret;
}

void destroy_ft_doc_word_iterator(ObFTDocWordIterator *&iterator)
{
  if (OB_NOT_NULL(iterator)) {
    as_storage_iterator(iterator)->~ObFTDocWordScanIterator();
    iterator = nullptr;
  }
}

void reset_ft_doc_word_iterator(ObFTDocWordIterator *iterator)
{
  if (OB_NOT_NULL(iterator)) {
    as_storage_iterator(iterator)->reset();
  }
}

int init_ft_doc_word_iterator(ObFTDocWordIterator *iterator,
                              uint64_t table_id,
                              const common::ObTabletID &tablet_id,
                              const transaction::ObTxReadSnapshot *snapshot,
                              int64_t schema_version)
{
  return OB_ISNULL(iterator)
      ? OB_INVALID_ARGUMENT
      : as_storage_iterator(iterator)->init(
            table_id, tablet_id, snapshot, schema_version);
}

int scan_ft_doc_words(ObFTDocWordIterator *iterator,
                      uint64_t table_id,
                      const common::ObDatum &row_mapping_id)
{
  return OB_ISNULL(iterator)
      ? OB_INVALID_ARGUMENT
      : as_storage_iterator(iterator)->do_scan(table_id, row_mapping_id);
}

int next_ft_doc_word(ObFTDocWordIterator *iterator,
                     blocksstable::ObDatumRow *&row)
{
  return OB_ISNULL(iterator)
      ? OB_INVALID_ARGUMENT
      : as_storage_iterator(iterator)->get_next_row(row);
}

} // namespace data_plane
} // end namespace oceanbase
