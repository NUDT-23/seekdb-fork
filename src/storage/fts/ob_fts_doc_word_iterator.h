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

#ifndef OCEANBASE_STORAGE_FTS_DOC_WORD_ITERATOR_H
#define OCEANBASE_STORAGE_FTS_DOC_WORD_ITERATOR_H

#include "lib/allocator/page_arena.h"
#include "common/datum/ob_datum.h"
#include "common/ob_searray.h"
#include "storage/blocksstable/ob_datum_row.h"
#include "data_plane/fts/ob_doc_id.h"
#include "storage/access/ob_table_param.h"
#include "storage/access/ob_dml_param.h"
#include "storage/tx/ob_trans_define.h"

namespace oceanbase
{
namespace storage
{

class ObFTDocWordScanIterator final
{
public:
  ObFTDocWordScanIterator();
  ~ObFTDocWordScanIterator();

  int init(
      const uint64_t table_id,
      const common::ObTabletID &tablet_id,
      const transaction::ObTxReadSnapshot *snapshot,
      const int64_t schema_version);
  int do_scan(const uint64_t table_id, const ObDatum &row_mapping_id);
  int get_next_row(blocksstable::ObDatumRow *&datum_row);

  void reset();

  TO_STRING_KV(KP(doc_word_iter_), K(scan_param_), K(table_param_));
private:
  int init_scan_param(
      const uint64_t table_id,
      const common::ObTabletID &tablet_id,
      const transaction::ObTxReadSnapshot *snapshot,
      const int64_t schema_version);
  int build_table_param(
      const uint64_t table_id,
      share::schema::ObTableParam &table_param,
      common::ObIArray<uint64_t> &column_ids);
  int build_key_range(const uint64_t table_id,
                      const ObDatum &row_mapping_id,
                      common::ObIArray<ObNewRange> &rowkey_range);
  int do_table_scan();
  int do_table_rescan();
  int reuse();
  // v3: sequential streaming mode. The doc word table is scanned once over the
  // whole range and consumed in primary key order, one document at a time, so
  // the per-document range rescan (iterator reuse + access service round trip
  // for every document of the data table) disappears from the FTS index build.
  // A document id that regresses below the last served one (data table scan
  // order is not guaranteed) breaks the stream and falls back to the legacy
  // per-document rescan path for correctness.
  int open_seq_scan(const uint64_t table_id);
  int drain_to_doc(const ObDatum &doc_id_datum);
  int serve_doc_by_rescan(const uint64_t table_id, const ObDatum &doc_id_datum);
  static int cmp_doc_id(const ObDatum &l, const ObDatum &r);

private:
  struct SeqWordRow
  {
    SeqWordRow() : cnt_(0), doc_len_(0) {}
    ObDatum doc_id_;
    ObDatum word_;
    uint64_t cnt_;
    uint64_t doc_len_;
  };

  common::ObArenaAllocator allocator_;
  common::ObArenaAllocator scan_allocator_;
  share::schema::ObTableParam table_param_;
  storage::ObTableScanParam scan_param_;
  common::ObNewRowIterator *doc_word_iter_;
  ObDocIDType docid_type_;
  bool is_inited_;
  // sequential streaming state
  bool seq_opened_;
  bool seq_broken_;
  bool seq_stream_end_;
  bool pending_valid_;
  SeqWordRow pending_row_;
  common::ObArenaAllocator doc_alloc_;
  common::ObArenaAllocator pending_alloc_;
  common::ObSEArray<SeqWordRow, 64> seq_rows_;
  int64_t seq_idx_;
  int64_t seq_serving_; // 0: none, 1: sequential buffer, 2: rescan iterator
  char last_doc_buf_[64];
  int64_t last_doc_len_;
  bool last_doc_valid_;
  blocksstable::ObDatumRow seq_out_row_;
  bool seq_out_row_inited_;

  DISALLOW_COPY_AND_ASSIGN(ObFTDocWordScanIterator);
};
} // end namespace storage
} // end namespace oceanbase

#endif // OCEANBASE_STORAGE_FTS_DOC_WORD_ITERATOR_H
