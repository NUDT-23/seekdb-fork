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

#define USING_LOG_PREFIX SERVER

#include "config_bridge.h"
#include "observer/ob_server_runtime_access.h"
#include "lib/stat/ob_diagnostic_info_guard.h"
#include "observer/mysql/obmp_query.h"
#include "sql/engine/ob_physical_plan.h"

#include "observer/mysql/ob_sync_plan_driver.h"
#include "observer/mysql/ob_sync_cmd_driver.h"
#include "observer/mysql/ob_async_cmd_driver.h"
#include "observer/mysql/ob_async_plan_driver.h"
#include "observer/omt/ob_server_runtime.h"
#include "observer/ob_server.h"
#include "sql/monitor/show_trace/ob_show_trace.h"

using namespace oceanbase::rpc;
using namespace oceanbase::obmysql;
using namespace oceanbase::common;
using namespace oceanbase::observer;
using namespace oceanbase::share;
using namespace oceanbase::share::schema;
using namespace oceanbase::trace;
using namespace oceanbase::sql;

// ==== file_sql 数据层 ====

#include <algorithm>

#include <cerrno>

#include <cmath>

#include <cstdio>

#include <cstdlib>

#include <cstring>

#include <dirent.h>

#include <fstream>

#include <map>

#include <sstream>

#include <string>

#include <strings.h>

#include <sys/stat.h>

#include <vector>

namespace filesql {

enum class ColumnType { BIGINT, DOUBLE, BOOLEAN, VARCHAR, DATE, DATETIME, NUL };

const char *type_name(ColumnType t);

struct Column {
  std::string name;
  ColumnType type = ColumnType::VARCHAR;
  bool nullable = false;
  int32_t source_index = -1;  // index of the column in the file
};

struct Value {
  bool is_null = true;
  int64_t i = 0;
  double d = 0;
  bool b = false;
  std::string s;   // VARCHAR / DATE / DATETIME keep their text form
};

struct Schema {
  std::vector<Column> columns;
};

struct Row {
  std::vector<Value> values;
};

// A batch of rows, the scan operator returns batches instead of the whole file.
struct Batch {
  std::vector<Row> rows;
};

struct Diagnostic {
  int64_t line = 0;
  std::string message;
  Diagnostic() {}
  Diagnostic(int64_t l, const std::string &m) : line(l), message(m) {}
};

struct CsvOptions {
  char delimiter = ',';
  bool has_header = true;
  int64_t sample_size = -1;  // -1 means "infer from the whole file"
  std::string null_str;      // empty means "an empty field is NULL"
};

struct ReadResult {
  Schema schema;
  std::vector<Row> rows;
  std::vector<Diagnostic> diagnostics;
};

// Reads a whole CSV file and infers the schema from all rows.
ReadResult read_csv(const std::string &path, const CsvOptions &opts = CsvOptions());

// Reads a whole JSONL file: the schema is the union of the fields of all lines,
// missing fields become NULL.
ReadResult read_jsonl(const std::string &path);

// Splits one CSV record starting at `pos`, RFC4180 quoting, embedded newlines
// and CRLF are handled. Returns false at end of file.
bool parse_csv_record(const std::string &text, size_t &pos, char delim, std::vector<std::string> &fields);

// Expands a shell style global pattern into the matched file names.
std::vector<std::string> glob_paths(const std::string &pattern);

// COPY ... TO ... (format csv, header, nullstr '...')
bool write_csv(const std::string &path, const Schema &schema, const std::vector<Row> &rows,
               bool header, const std::string &null_str, std::string &error);

// Renders a value the way the SQL layer prints it.
std::string format_value(const Value &v, ColumnType t);

}  // namespace filesql

namespace filesql {

const char *type_name(ColumnType t) {
  switch (t) {
    case ColumnType::BIGINT: return "BIGINT";
    case ColumnType::DOUBLE: return "DOUBLE";
    case ColumnType::BOOLEAN: return "BOOLEAN";
    case ColumnType::VARCHAR: return "VARCHAR";
    case ColumnType::DATE: return "DATE";
    case ColumnType::DATETIME: return "DATETIME";
    case ColumnType::NUL: return "NULL";
  }
  return "VARCHAR";
}

static std::string read_file(const std::string &path, bool &ok) {
  std::ifstream in(path.c_str(), std::ios::binary);
  if (!in.good()) {
    ok = false;
    return std::string();
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  ok = true;
  return ss.str();
}

// ---------------------------------------------------------------- csv parsing

bool parse_csv_record(const std::string &text, size_t &pos, char delim,
                      std::vector<std::string> &fields) {
  fields.clear();
  if (pos >= text.size()) {
    return false;
  }
  std::string field;
  bool in_quotes = false;
  bool saw_any = false;
  while (pos < text.size()) {
    char c = text[pos];
    if (in_quotes) {
      if (c == '"') {
        if (pos + 1 < text.size() && text[pos + 1] == '"') {
          field.push_back('"');
          pos += 2;
          continue;
        }
        in_quotes = false;
        ++pos;
        continue;
      }
      field.push_back(c);
      ++pos;
      continue;
    }
    if (c == '"') {
      in_quotes = true;
      saw_any = true;
      ++pos;
      continue;
    }
    if (c == delim) {
      fields.push_back(field);
      field.clear();
      saw_any = true;
      ++pos;
      continue;
    }
    if (c == '\r' || c == '\n') {
      // consume CRLF or LF, the record ends here
      if (c == '\r' && pos + 1 < text.size() && text[pos + 1] == '\n') {
        pos += 2;
      } else {
        ++pos;
      }
      fields.push_back(field);
      return true;
    }
    field.push_back(c);
    saw_any = true;
    ++pos;
  }
  if (in_quotes) {
    // unterminated quote: the field is kept as it is, the caller reports it
    fields.push_back(field);
    return true;
  }
  fields.push_back(field);
  return saw_any || !fields.empty();
}

// ------------------------------------------------------------ type inference

static bool is_integer_text(const std::string &s) {
  if (s.empty()) return false;
  size_t i = 0;
  if (s[0] == '+' || s[0] == '-') i = 1;
  if (i >= s.size()) return false;
  for (; i < s.size(); ++i) {
    if (s[i] < '0' || s[i] > '9') return false;
  }
  // reject values out of the int64 range
  errno = 0;
  char *end = nullptr;
  long long v = std::strtoll(s.c_str(), &end, 10);
  return errno == 0 && end == s.c_str() + s.size() && v >= INT64_MIN && v <= INT64_MAX;
}

static bool is_double_text(const std::string &s) {
  if (s.empty()) return false;
  errno = 0;
  char *end = nullptr;
  double v = std::strtod(s.c_str(), &end);
  if (errno != 0 || end != s.c_str() + s.size()) return false;
  // NaN/inf are not accepted as numbers by the type inference
  return std::isfinite(v);
}

static bool is_boolean_text(const std::string &s) {
  std::string lower;
  lower.resize(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    lower[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(s[i])));
  }
  return lower == "true" || lower == "false" || lower == "t" || lower == "f" ||
         lower == "yes" || lower == "no" || lower == "1" || lower == "0";
}

static bool parse_date(const std::string &s, bool &is_datetime) {
  // YYYY-MM-DD or YYYY-MM-DD[ T]HH:MM:SS
  is_datetime = false;
  if (s.size() < 10) return false;
  for (int i = 0; i < 10; ++i) {
    if (i == 4 || i == 7) {
      if (s[i] != '-') return false;
    } else if (s[i] < '0' || s[i] > '9') {
      return false;
    }
  }
  int year = std::atoi(s.substr(0, 4).c_str());
  int month = std::atoi(s.substr(5, 2).c_str());
  int day = std::atoi(s.substr(8, 2).c_str());
  if (year < 1 || month < 1 || month > 12 || day < 1 || day > 31) return false;
  if (s.size() == 10) return true;
  if (s.size() < 19) return false;
  char sep = s[10];
  if (sep != ' ' && sep != 'T') return false;
  if (s.size() > 19) {
    // fractional seconds and timezone suffixes are accepted as well
    if (s[19] != '.' && s[19] != 'Z' && s[19] != '+' && s[19] != '-') return false;
  }
  for (size_t i = 11; i < 19; ++i) {
    if (i == 13 || i == 16) {
      if (s[i] != ':') return false;
    } else if (s[i] < '0' || s[i] > '9') {
      return false;
    }
  }
  int hour = std::atoi(s.substr(11, 2).c_str());
  int minute = std::atoi(s.substr(14, 2).c_str());
  int second = std::atoi(s.substr(17, 2).c_str());
  if (hour > 23 || minute > 59 || second > 59) return false;
  is_datetime = true;
  return true;
}

// The type a single text field suggests.
static ColumnType probe_text(const std::string &s) {
  if (is_integer_text(s)) return ColumnType::BIGINT;
  if (is_boolean_text(s)) return ColumnType::BOOLEAN;
  if (is_double_text(s)) return ColumnType::DOUBLE;
  bool is_dt = false;
  if (parse_date(s, is_dt)) return is_dt ? ColumnType::DATETIME : ColumnType::DATE;
  return ColumnType::VARCHAR;
}

// Widening rules: a column keeps the narrowest type that covers every value.
static ColumnType merge_type(ColumnType a, ColumnType b) {
  if (a == ColumnType::NUL) return b;
  if (b == ColumnType::NUL) return a;
  if (a == b) return a;
  // integers and decimals share DOUBLE, a boolean column that also carries
  // numbers falls back to VARCHAR (0/1 stay boolean because probe checks
  // boolean first, so this only happens for real conflicts)
  if ((a == ColumnType::BIGINT && b == ColumnType::DOUBLE) ||
      (a == ColumnType::DOUBLE && b == ColumnType::BIGINT)) {
    return ColumnType::DOUBLE;
  }
  if ((a == ColumnType::DATE && b == ColumnType::DATETIME) ||
      (a == ColumnType::DATETIME && b == ColumnType::DATE)) {
    return ColumnType::DATETIME;
  }
  return ColumnType::VARCHAR;
}

static std::string upper(const std::string &s) {
  std::string r = s;
  for (size_t i = 0; i < r.size(); ++i) r[i] = static_cast<char>(std::toupper(static_cast<unsigned char>(r[i])));
  return r;
}

static std::string dedup_name(const std::vector<std::string> &seen, const std::string &base) {
  std::string name = base;
  int suffix = 2;
  while (std::find(seen.begin(), seen.end(), name) != seen.end()) {
    name = base + "__" + std::to_string(suffix);
    ++suffix;
  }
  return name;
}

static Value make_value(const std::string &text, ColumnType t) {
  Value v;
  v.is_null = false;
  switch (t) {
    case ColumnType::BIGINT:
      v.i = std::strtoll(text.c_str(), nullptr, 10);
      v.s = text;
      break;
    case ColumnType::DOUBLE:
      v.d = std::strtod(text.c_str(), nullptr);
      v.s = text;
      break;
    case ColumnType::BOOLEAN: {
      std::string lower = text;
      for (size_t i = 0; i < lower.size(); ++i) lower[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(lower[i])));
      v.b = (lower == "true" || lower == "t" || lower == "yes" || lower == "1");
      v.s = v.b ? "1" : "0";
      break;
    }
    case ColumnType::DATE:
    case ColumnType::DATETIME:
      v.s = text;
      // normalise "2026-08-03T12:13:14" to the SQL form
      if (t == ColumnType::DATETIME && v.s.size() > 10 && v.s[10] == 'T') v.s[10] = ' ';
      break;
    default:
      v.s = text;
      break;
  }
  return v;
}

ReadResult read_csv(const std::string &path, const CsvOptions &opts) {
  ReadResult result;
  bool ok = false;
  std::string text = read_file(path, ok);
  if (!ok) {
    result.diagnostics.push_back({0, "cannot open file " + path});
    return result;
  }
  std::vector<std::vector<std::string> > records;
  std::vector<int64_t> record_lines;
  size_t pos = 0;
  int64_t line = 1;
  while (pos < text.size()) {
    std::vector<std::string> fields;
    size_t before = pos;
    int64_t start_line = line;
    if (!parse_csv_record(text, pos, opts.delimiter, fields)) break;
    for (size_t i = before; i < pos && i < text.size(); ++i) {
      if (text[i] == '\n') ++line;
    }
    if (fields.size() == 1 && fields[0].empty() && before >= text.size()) break;
    records.push_back(fields);
    record_lines.push_back(start_line);
  }
  if (records.empty()) {
    return result;
  }

  std::vector<std::string> header;
  size_t first_data = 0;
  if (opts.has_header) {
    std::vector<std::string> seen;
    for (size_t i = 0; i < records[0].size(); ++i) {
      std::string base = records[0][i];
      if (base.empty()) base = "column" + std::to_string(i);
      std::string name = dedup_name(seen, base);
      seen.push_back(name);
      header.push_back(name);
    }
    first_data = 1;
  } else {
    for (size_t i = 0; i < records[0].size(); ++i) header.push_back("column" + std::to_string(i));
  }

  const size_t column_count = header.size();
  std::vector<ColumnType> types(column_count, ColumnType::NUL);
  std::vector<bool> nullable(column_count, false);
  std::vector<bool> short_row(column_count, false);
  for (size_t r = first_data; r < records.size(); ++r) {
    const std::vector<std::string> &rec = records[r];
    if (rec.size() != column_count) {
      char buf[160];
      std::snprintf(buf, sizeof(buf), "line %lld has %zu fields, expected %zu",
                    static_cast<long long>(record_lines[r]), rec.size(), column_count);
      result.diagnostics.push_back({record_lines[r], buf});
      for (size_t c = rec.size(); c < column_count; ++c) short_row[c] = true;
    }
    for (size_t c = 0; c < column_count && c < rec.size(); ++c) {
      const std::string &field = rec[c];
      if (field.empty() || (!opts.null_str.empty() && field == opts.null_str)) {
        nullable[c] = true;
        continue;
      }
      types[c] = merge_type(types[c], probe_text(field));
    }
  }
  for (size_t c = 0; c < column_count; ++c) {
    Column col;
    col.name = header[c];
    col.type = (types[c] == ColumnType::NUL) ? ColumnType::VARCHAR : types[c];
    col.nullable = nullable[c] || short_row[c];
    col.source_index = static_cast<int32_t>(c);
    result.schema.columns.push_back(col);
  }

  for (size_t r = first_data; r < records.size(); ++r) {
    const std::vector<std::string> &rec = records[r];
    Row row;
    row.values.resize(column_count);
    for (size_t c = 0; c < column_count; ++c) {
      Value &v = row.values[c];
      if (c >= rec.size()) {
        v.is_null = true;
        continue;
      }
      const std::string &field = rec[c];
      if (field.empty() || (!opts.null_str.empty() && field == opts.null_str)) {
        v.is_null = true;
        continue;
      }
      v = make_value(field, result.schema.columns[c].type);
    }
    result.rows.push_back(row);
  }
  return result;
}

// ------------------------------------------------------------ jsonl parsing

namespace {

struct JsonValue {
  enum Kind { NUL, BOOL, INT, DOUBLE, STRING, OBJECT, ARRAY } kind = NUL;
  bool b = false;
  int64_t i = 0;
  double d = 0;
  std::string s;
  std::vector<std::pair<std::string, JsonValue *> > members;
  std::vector<JsonValue *> items;
};

class JsonParser {
 public:
  explicit JsonParser(const std::string &text) : text_(text), pos_(0) {}

  bool parse(JsonValue &out) {
    skip_space();
    if (!parse_value(out)) return false;
    skip_space();
    return pos_ == text_.size();
  }

 private:
  void skip_space() {
    while (pos_ < text_.size() && (text_[pos_] == ' ' || text_[pos_] == '\t' ||
                                   text_[pos_] == '\r' || text_[pos_] == '\n')) {
      ++pos_;
    }
  }
  bool parse_value(JsonValue &out);
  bool parse_string(std::string &out);
  bool parse_number(JsonValue &out);

  const std::string &text_;
  size_t pos_;
};

bool JsonParser::parse_string(std::string &out) {
  if (pos_ >= text_.size() || text_[pos_] != '"') return false;
  ++pos_;
  out.clear();
  while (pos_ < text_.size()) {
    char c = text_[pos_++];
    if (c == '"') return true;
    if (c != '\\') {
      out.push_back(c);
      continue;
    }
    if (pos_ >= text_.size()) return false;
    char e = text_[pos_++];
    switch (e) {
      case '"': out.push_back('"'); break;
      case '\\': out.push_back('\\'); break;
      case '/': out.push_back('/'); break;
      case 'b': out.push_back('\b'); break;
      case 'f': out.push_back('\f'); break;
      case 'n': out.push_back('\n'); break;
      case 'r': out.push_back('\r'); break;
      case 't': out.push_back('\t'); break;
      case 'u': {
        if (pos_ + 4 > text_.size()) return false;
        unsigned code = 0;
        for (int i = 0; i < 4; ++i) {
          char h = text_[pos_++];
          code <<= 4;
          if (h >= '0' && h <= '9') code |= static_cast<unsigned>(h - '0');
          else if (h >= 'a' && h <= 'f') code |= static_cast<unsigned>(h - 'a' + 10);
          else if (h >= 'A' && h <= 'F') code |= static_cast<unsigned>(h - 'A' + 10);
          else return false;
        }
        // utf-8 encode
        if (code < 0x80) {
          out.push_back(static_cast<char>(code));
        } else if (code < 0x800) {
          out.push_back(static_cast<char>(0xC0 | (code >> 6)));
          out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        } else {
          out.push_back(static_cast<char>(0xE0 | (code >> 12)));
          out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
          out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        }
        break;
      }
      default: return false;
    }
  }
  return false;
}

bool JsonParser::parse_number(JsonValue &out) {
  size_t start = pos_;
  if (pos_ < text_.size() && (text_[pos_] == '-' || text_[pos_] == '+')) ++pos_;
  bool is_double = false;
  while (pos_ < text_.size()) {
    char c = text_[pos_];
    if (c >= '0' && c <= '9') {
      ++pos_;
    } else if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
      is_double = true;
      ++pos_;
    } else {
      break;
    }
  }
  std::string token = text_.substr(start, pos_ - start);
  if (token.empty()) return false;
  if (!is_double) {
    out.kind = JsonValue::INT;
    out.i = std::strtoll(token.c_str(), nullptr, 10);
    out.d = static_cast<double>(out.i);
  } else {
    out.kind = JsonValue::DOUBLE;
    out.d = std::strtod(token.c_str(), nullptr);
    out.i = static_cast<int64_t>(out.d);
  }
  return true;
}

bool JsonParser::parse_value(JsonValue &out) {
  skip_space();
  if (pos_ >= text_.size()) return false;
  char c = text_[pos_];
  if (c == '{') {
    out.kind = JsonValue::OBJECT;
    ++pos_;
    skip_space();
    if (pos_ < text_.size() && text_[pos_] == '}') {
      ++pos_;
      return true;
    }
    while (true) {
      skip_space();
      std::string key;
      if (!parse_string(key)) return false;
      skip_space();
      if (pos_ >= text_.size() || text_[pos_] != ':') return false;
      ++pos_;
      JsonValue value;
      if (!parse_value(value)) return false;
      out.members.push_back(std::make_pair(key, new JsonValue(value)));
      skip_space();
      if (pos_ < text_.size() && text_[pos_] == ',') {
        ++pos_;
        continue;
      }
      if (pos_ < text_.size() && text_[pos_] == '}') {
        ++pos_;
        return true;
      }
      return false;
    }
  }
  if (c == '[') {
    out.kind = JsonValue::ARRAY;
    ++pos_;
    skip_space();
    if (pos_ < text_.size() && text_[pos_] == ']') {
      ++pos_;
      return true;
    }
    while (true) {
      JsonValue value;
      if (!parse_value(value)) return false;
      out.items.push_back(new JsonValue(value));
      skip_space();
      if (pos_ < text_.size() && text_[pos_] == ',') {
        ++pos_;
        continue;
      }
      if (pos_ < text_.size() && text_[pos_] == ']') {
        ++pos_;
        return true;
      }
      return false;
    }
  }
  if (c == '"') {
    out.kind = JsonValue::STRING;
    return parse_string(out.s);
  }
  if (text_.compare(pos_, 4, "true") == 0) {
    out.kind = JsonValue::BOOL;
    out.b = true;
    pos_ += 4;
    return true;
  }
  if (text_.compare(pos_, 5, "false") == 0) {
    out.kind = JsonValue::BOOL;
    out.b = false;
    pos_ += 5;
    return true;
  }
  if (text_.compare(pos_, 4, "null") == 0) {
    out.kind = JsonValue::NUL;
    pos_ += 4;
    return true;
  }
  return parse_number(out);
}

}  // namespace

ReadResult read_jsonl(const std::string &path) {
  ReadResult result;
  bool ok = false;
  std::string text = read_file(path, ok);
  if (!ok) {
    result.diagnostics.push_back({0, "cannot open file " + path});
    return result;
  }
  std::vector<std::pair<std::string, JsonValue> > objects;
  std::vector<int64_t> lines;
  std::istringstream in(text);
  std::string line;
  int64_t line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    if (!line.empty() && line[line.size() - 1] == '\r') line.erase(line.size() - 1);
    if (line.find_first_not_of(" \t") == std::string::npos) continue;
    JsonValue value;
    JsonParser parser(line);
    if (!parser.parse(value)) {
      result.diagnostics.push_back({line_no, "line is not a valid json object"});
      continue;
    }
    if (value.kind != JsonValue::OBJECT) {
      result.diagnostics.push_back({line_no, "line is not a json object"});
      continue;
    }
    objects.push_back(std::make_pair(std::string(), value));
    lines.push_back(line_no);
  }

  // union of the fields, in the order they first appear
  std::vector<std::string> names;
  for (size_t i = 0; i < objects.size(); ++i) {
    for (size_t j = 0; j < objects[i].second.members.size(); ++j) {
      const std::string &name = objects[i].second.members[j].first;
      if (std::find(names.begin(), names.end(), name) == names.end()) names.push_back(name);
    }
  }
  std::vector<ColumnType> types(names.size(), ColumnType::NUL);
  std::vector<bool> nullable(names.size(), false);
  for (size_t i = 0; i < objects.size(); ++i) {
    for (size_t k = 0; k < names.size(); ++k) {
      const JsonValue *found = nullptr;
      for (size_t j = 0; j < objects[i].second.members.size(); ++j) {
        if (objects[i].second.members[j].first == names[k]) {
          found = objects[i].second.members[j].second;
          break;
        }
      }
      if (found == nullptr || found->kind == JsonValue::NUL) {
        nullable[k] = true;
        continue;
      }
      ColumnType t = ColumnType::VARCHAR;
      switch (found->kind) {
        case JsonValue::INT: t = ColumnType::BIGINT; break;
        case JsonValue::DOUBLE: t = ColumnType::DOUBLE; break;
        case JsonValue::BOOL: t = ColumnType::BOOLEAN; break;
        case JsonValue::STRING: {
          bool is_dt = false;
          if (parse_date(found->s, is_dt)) t = is_dt ? ColumnType::DATETIME : ColumnType::DATE;
          else t = ColumnType::VARCHAR;
          break;
        }
        case JsonValue::OBJECT:
        case JsonValue::ARRAY:
          result.diagnostics.push_back({lines[i], "nested objects and arrays are not supported"});
          t = ColumnType::VARCHAR;
          break;
        default: t = ColumnType::VARCHAR; break;
      }
      types[k] = merge_type(types[k], t);
    }
  }
  for (size_t k = 0; k < names.size(); ++k) {
    Column col;
    col.name = names[k];
    col.type = (types[k] == ColumnType::NUL) ? ColumnType::VARCHAR : types[k];
    col.nullable = nullable[k];
    col.source_index = static_cast<int32_t>(k);
    result.schema.columns.push_back(col);
  }

  for (size_t i = 0; i < objects.size(); ++i) {
    Row row;
    row.values.resize(names.size());
    for (size_t k = 0; k < names.size(); ++k) {
      const JsonValue *found = nullptr;
      for (size_t j = 0; j < objects[i].second.members.size(); ++j) {
        if (objects[i].second.members[j].first == names[k]) {
          found = objects[i].second.members[j].second;
          break;
        }
      }
      Value &v = row.values[k];
      if (found == nullptr || found->kind == JsonValue::NUL) {
        v.is_null = true;
        continue;
      }
      const Column &col = result.schema.columns[k];
      v.is_null = false;
      switch (col.type) {
        case ColumnType::BIGINT: {
          int64_t iv = found->kind == JsonValue::DOUBLE ? static_cast<int64_t>(found->d) : found->i;
          v.i = iv;
          v.s = std::to_string(iv);
          break;
        }
        case ColumnType::DOUBLE:
          v.d = found->kind == JsonValue::INT ? static_cast<double>(found->i) : found->d;
          break;
        case ColumnType::BOOLEAN:
          v.b = found->b;
          v.s = v.b ? "1" : "0";
          break;
        case ColumnType::DATE:
        case ColumnType::DATETIME:
          v.s = found->kind == JsonValue::STRING ? found->s : std::string();
          if (col.type == ColumnType::DATETIME && v.s.size() > 10 && v.s[10] == 'T') v.s[10] = ' ';
          break;
        default:
          if (found->kind == JsonValue::STRING) {
            v.s = found->s;
          } else if (found->kind == JsonValue::INT) {
            v.s = std::to_string(found->i);
          } else if (found->kind == JsonValue::DOUBLE) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.15g", found->d);
            v.s = buf;
          } else if (found->kind == JsonValue::BOOL) {
            v.s = found->b ? "true" : "false";
          }
          break;
      }
    }
    result.rows.push_back(row);
  }
  return result;
}

// --------------------------------------------------------------------- glob

static bool match_component(const std::string &pattern, const std::string &name) {
  size_t p = 0, n = 0, star = std::string::npos, mark = 0;
  while (n < name.size()) {
    if (p < pattern.size() && (pattern[p] == '?' || pattern[p] == name[n])) {
      ++p;
      ++n;
    } else if (p < pattern.size() && pattern[p] == '*') {
      star = p++;
      mark = n;
    } else if (star != std::string::npos) {
      p = star + 1;
      n = ++mark;
    } else {
      return false;
    }
  }
  while (p < pattern.size() && pattern[p] == '*') ++p;
  return p == pattern.size();
}

static void glob_recursive(const std::string &prefix, const std::string &pattern,
                           size_t offset, std::vector<std::string> &out) {
  size_t slash = pattern.find('/', offset);
  std::string component = pattern.substr(offset, slash == std::string::npos ? std::string::npos : slash - offset);
  std::string dir = prefix.empty() ? "." : prefix;
  if (component.find('*') == std::string::npos && component.find('?') == std::string::npos) {
    std::string next = prefix.empty() ? component : prefix + "/" + component;
    if (slash == std::string::npos) {
      struct stat st;
      if (::stat(next.c_str(), &st) == 0) out.push_back(next);
    } else {
      glob_recursive(next, pattern, slash + 1, out);
    }
    return;
  }
  DIR *dp = ::opendir(dir.c_str());
  if (dp == nullptr) return;
  std::vector<std::string> names;
  struct dirent *ent = nullptr;
  while ((ent = ::readdir(dp)) != nullptr) {
    std::string name = ent->d_name;
    if (name == "." || name == "..") continue;
    if (!match_component(component, name)) continue;
    names.push_back(name);
  }
  ::closedir(dp);
  std::sort(names.begin(), names.end());
  for (size_t i = 0; i < names.size(); ++i) {
    std::string next = prefix.empty() ? names[i] : prefix + "/" + names[i];
    if (slash == std::string::npos) {
      out.push_back(next);
    } else {
      glob_recursive(next, pattern, slash + 1, out);
    }
  }
}

std::vector<std::string> glob_paths(const std::string &pattern) {
  std::vector<std::string> out;
  if (pattern.find('*') == std::string::npos && pattern.find('?') == std::string::npos) {
    struct stat st;
    if (::stat(pattern.c_str(), &st) == 0) out.push_back(pattern);
    return out;
  }
  std::string prefix;
  size_t start = 0;
  if (!pattern.empty() && pattern[0] == '/') {
    prefix = "/";
    start = 1;
  }
  glob_recursive(prefix, pattern, start, out);
  return out;
}

// -------------------------------------------------------------- formatting

std::string format_value(const Value &v, ColumnType t) {
  if (v.is_null) return std::string();
  char buf[64];
  switch (t) {
    case ColumnType::BIGINT:
      std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(v.i));
      return buf;
    case ColumnType::DOUBLE: {
      // shortest representation that round trips
      for (int precision = 15; precision <= 17; ++precision) {
        std::snprintf(buf, sizeof(buf), "%.*g", precision, v.d);
        double back = std::strtod(buf, nullptr);
        if (back == v.d) return buf;
      }
      return buf;
    }
    case ColumnType::BOOLEAN:
      return v.b ? "1" : "0";
    default:
      return v.s;
  }
}

static std::string quote_csv_field(const std::string &field, char delim) {
  bool needs = false;
  for (size_t i = 0; i < field.size(); ++i) {
    if (field[i] == delim || field[i] == '"' || field[i] == '\n' || field[i] == '\r') {
      needs = true;
      break;
    }
  }
  if (!needs) return field;
  std::string out = "\"";
  for (size_t i = 0; i < field.size(); ++i) {
    if (field[i] == '"') out += "\"\"";
    else out.push_back(field[i]);
  }
  out += "\"";
  return out;
}

bool write_csv(const std::string &path, const Schema &schema, const std::vector<Row> &rows,
               bool header, const std::string &null_str, std::string &error) {
  struct stat st;
  if (::stat(path.c_str(), &st) == 0) {
    error = "file already exists: " + path;
    return false;
  }
  std::string tmp = path + ".tmp";
  {
    std::ofstream out(tmp.c_str(), std::ios::binary);
    if (!out.good()) {
      error = "cannot create " + tmp;
      return false;
    }
    if (header) {
      for (size_t c = 0; c < schema.columns.size(); ++c) {
        if (c) out << ',';
        out << quote_csv_field(schema.columns[c].name, ',');
      }
      out << '\n';
    }
    for (size_t r = 0; r < rows.size(); ++r) {
      for (size_t c = 0; c < schema.columns.size() && c < rows[r].values.size(); ++c) {
        if (c) out << ',';
        const Value &v = rows[r].values[c];
        if (v.is_null) {
          out << null_str;
        } else {
          out << quote_csv_field(format_value(v, schema.columns[c].type), ',');
        }
      }
      out << '\n';
    }
    out.flush();
    if (!out.good()) {
      error = "write failed";
      std::remove(tmp.c_str());
      return false;
    }
  }
  if (std::rename(tmp.c_str(), path.c_str()) != 0) {
    error = "rename failed";
    std::remove(tmp.c_str());
    return false;
  }
  return true;
}

}  // namespace filesql

namespace filesql {

namespace {

// --------------------------------------------------------------- thrift compact

class CompactReader {
 public:
  CompactReader(const unsigned char *data, size_t size) : data_(data), size_(size), pos_(0) {}

  bool ok() const { return ok_; }
  size_t pos() const { return pos_; }
  void seek(size_t pos) { pos_ = pos; }

  unsigned char read_byte() {
    if (pos_ >= size_) {
      ok_ = false;
      return 0;
    }
    return data_[pos_++];
  }
  uint64_t read_uvarint() {
    uint64_t value = 0;
    int shift = 0;
    while (true) {
      unsigned char b = read_byte();
      if (!ok_) return 0;
      value |= (static_cast<uint64_t>(b & 0x7F)) << shift;
      if ((b & 0x80) == 0) break;
      shift += 7;
      if (shift > 63) {
        ok_ = false;
        return 0;
      }
    }
    return value;
  }
  int64_t read_zigzag() {
    uint64_t v = read_uvarint();
    return static_cast<int64_t>((v >> 1) ^ (~(v & 1) + 1));
  }
  std::string read_binary() {
    uint64_t len = read_uvarint();
    if (!ok_ || pos_ + len > size_) {
      ok_ = false;
      return std::string();
    }
    std::string s(reinterpret_cast<const char *>(data_ + pos_), static_cast<size_t>(len));
    pos_ += static_cast<size_t>(len);
    return s;
  }
  double read_double() {
    if (pos_ + 8 > size_) {
      ok_ = false;
      return 0;
    }
    double v = 0;
    std::memcpy(&v, data_ + pos_, 8);
    pos_ += 8;
    return v;
  }
  // returns false when the struct ends
  bool read_field(int &type, int &id) {
    unsigned char b = read_byte();
    if (!ok_) return false;
    if (b == 0) return false;  // stop field
    type = b & 0x0F;
    int delta = (b & 0xF0) >> 4;
    if (delta == 0) {
      id = static_cast<int>(read_zigzag());
    } else {
      id = last_id_ + delta;
    }
    last_id_ = id;
    return ok_;
  }
  void enter_struct() { last_id_stack_.push_back(last_id_); last_id_ = 0; }
  void leave_struct() {
    if (!last_id_stack_.empty()) {
      last_id_ = last_id_stack_.back();
      last_id_stack_.pop_back();
    }
  }
  size_t read_list_begin(int &elem_type) {
    unsigned char b = read_byte();
    if (!ok_) return 0;
    elem_type = b & 0x0F;
    size_t count = (b & 0xF0) >> 4;
    if (count == 15) count = static_cast<size_t>(read_uvarint());
    return count;
  }

 private:
  const unsigned char *data_;
  size_t size_;
  size_t pos_;
  bool ok_ = true;
  int last_id_ = 0;
  std::vector<int> last_id_stack_;
};

static void skip_value(CompactReader &r, int type) {
  switch (type) {
    case 1: case 2: break;  // true / false
    case 3: r.read_byte(); break;
    case 4: case 5: case 6: r.read_uvarint(); break;
    case 7: r.read_double(); break;
    case 8: r.read_binary(); break;
    case 9: case 10: {
      int elem = 0;
      size_t n = r.read_list_begin(elem);
      for (size_t i = 0; i < n && r.ok(); ++i) skip_value(r, elem);
      break;
    }
    case 11: {
      // map: size then key/value types
      size_t n = r.read_uvarint();
      unsigned char kv = r.read_byte();
      for (size_t i = 0; i < n && r.ok(); ++i) {
        skip_value(r, (kv >> 4) & 0x0F);
        skip_value(r, kv & 0x0F);
      }
      break;
    }
    case 12: {
      r.enter_struct();
      int t = 0, id = 0;
      while (r.read_field(t, id)) skip_value(r, t);
      r.leave_struct();
      break;
    }
    default: break;
  }
}

// ------------------------------------------------------------- snappy (raw)

static bool snappy_uncompress(const unsigned char *in, size_t in_size, std::string &out) {
  out.clear();
  size_t pos = 0;
  uint64_t expected = 0;
  int shift = 0;
  while (pos < in_size) {
    unsigned char b = in[pos++];
    expected |= static_cast<uint64_t>(b & 0x7F) << shift;
    if ((b & 0x80) == 0) break;
    shift += 7;
  }
  out.reserve(static_cast<size_t>(expected));
  while (pos < in_size) {
    unsigned char tag = in[pos++];
    int type = tag & 0x03;
    if (type == 0) {
      size_t len = tag >> 2;
      if (len >= 60) {
        int extra = static_cast<int>(len) - 59;
        len = 0;
        for (int i = 0; i < extra; ++i) len |= static_cast<size_t>(in[pos++]) << (8 * i);
      }
      len += 1;
      if (pos + len > in_size) return false;
      out.append(reinterpret_cast<const char *>(in + pos), len);
      pos += len;
      continue;
    }
    size_t len = 0;
    size_t offset = 0;
    if (type == 1) {
      len = ((tag >> 2) & 0x07) + 4;
      if (pos >= in_size) return false;
      offset = (static_cast<size_t>(tag >> 5) << 8) | in[pos++];
    } else if (type == 2) {
      len = (tag >> 2) + 1;
      if (pos + 2 > in_size) return false;
      offset = static_cast<size_t>(in[pos]) | (static_cast<size_t>(in[pos + 1]) << 8);
      pos += 2;
    } else {
      len = (tag >> 2) + 1;
      if (pos + 4 > in_size) return false;
      offset = static_cast<size_t>(in[pos]) | (static_cast<size_t>(in[pos + 1]) << 8) |
               (static_cast<size_t>(in[pos + 2]) << 16) | (static_cast<size_t>(in[pos + 3]) << 24);
      pos += 4;
    }
    if (offset == 0 || offset > out.size()) return false;
    size_t start = out.size() - offset;
    for (size_t i = 0; i < len; ++i) out.push_back(out[start + i]);
  }
  return out.size() == static_cast<size_t>(expected);
}

// ---------------------------------------------------------------- parquet meta

struct SchemaElement {
  int type = -1;              // physical type
  int repetition = 0;         // 0 required, 1 optional, 2 repeated
  std::string name;
  int converted_type = -1;
  int num_children = 0;
};

struct ColumnChunk {
  int64_t data_page_offset = 0;
  int64_t dictionary_page_offset = 0;
  int64_t num_values = 0;
  int codec = 0;
  int type = -1;
  std::vector<int> encodings;
};

struct RowGroup {
  std::vector<ColumnChunk> columns;
  int64_t num_rows = 0;
};

struct FileMeta {
  std::vector<SchemaElement> schema;
  std::vector<RowGroup> row_groups;
  int64_t num_rows = 0;
};

static void parse_schema_element(CompactReader &r, SchemaElement &e) {
  r.enter_struct();
  int t = 0, id = 0;
  while (r.read_field(t, id)) {
    switch (id) {
      case 1: e.type = static_cast<int>(r.read_zigzag()); break;
      case 3: e.repetition = static_cast<int>(r.read_zigzag()); break;
      case 4: e.name = r.read_binary(); break;
      case 5: e.num_children = static_cast<int>(r.read_zigzag()); break;
      case 6: e.converted_type = static_cast<int>(r.read_zigzag()); break;
      default: skip_value(r, t); break;
    }
  }
  r.leave_struct();
}

static void parse_column_meta(CompactReader &r, ColumnChunk &c) {
  r.enter_struct();
  int t = 0, id = 0;
  while (r.read_field(t, id)) {
    switch (id) {
      case 1: c.type = static_cast<int>(r.read_zigzag()); break;
      case 2: {
        int elem = 0;
        size_t n = r.read_list_begin(elem);
        for (size_t i = 0; i < n && r.ok(); ++i) c.encodings.push_back(static_cast<int>(r.read_zigzag()));
        break;
      }
      case 4: c.codec = static_cast<int>(r.read_zigzag()); break;
      case 5: c.num_values = r.read_zigzag(); break;
      case 9: c.data_page_offset = r.read_zigzag(); break;
      case 11: c.dictionary_page_offset = r.read_zigzag(); break;
      default: skip_value(r, t); break;
    }
  }
  r.leave_struct();
}

static void parse_column_chunk(CompactReader &r, ColumnChunk &c) {
  r.enter_struct();
  int t = 0, id = 0;
  while (r.read_field(t, id)) {
    switch (id) {
      case 2: r.read_zigzag(); break;  // file_offset
      case 3: parse_column_meta(r, c); break;
      default: skip_value(r, t); break;
    }
  }
  r.leave_struct();
}

static bool parse_file_meta(const unsigned char *data, size_t size, FileMeta &meta) {
  CompactReader r(data, size);
  r.enter_struct();
  int t = 0, id = 0;
  while (r.read_field(t, id)) {
    switch (id) {
      case 2: {
        int elem = 0;
        size_t n = r.read_list_begin(elem);
        for (size_t i = 0; i < n && r.ok(); ++i) {
          SchemaElement e;
          parse_schema_element(r, e);
          meta.schema.push_back(e);
        }
        break;
      }
      case 3: meta.num_rows = r.read_zigzag(); break;
      case 4: {
        int elem = 0;
        size_t n = r.read_list_begin(elem);
        for (size_t i = 0; i < n && r.ok(); ++i) {
          RowGroup g;
          r.enter_struct();
          int rt = 0, rid = 0;
          while (r.read_field(rt, rid)) {
            if (rid == 1) {
              int ce = 0;
              size_t cn = r.read_list_begin(ce);
              for (size_t k = 0; k < cn && r.ok(); ++k) {
                ColumnChunk c;
                parse_column_chunk(r, c);
                g.columns.push_back(c);
              }
            } else if (rid == 3) {
              g.num_rows = r.read_zigzag();
            } else {
              skip_value(r, rt);
            }
          }
          r.leave_struct();
          meta.row_groups.push_back(g);
        }
        break;
      }
      default: skip_value(r, t); break;
    }
  }
  r.leave_struct();
  return r.ok();
}

// ------------------------------------------------------------------ decoding

static int bit_width(int64_t max_value) {
  int w = 0;
  while ((static_cast<int64_t>(1) << w) <= max_value && w < 32) ++w;
  return w;
}

// RLE / bit packed hybrid used by definition levels and dictionary indices.
static bool decode_rle(const unsigned char *data, size_t size, int width, int64_t count,
                       std::vector<int64_t> &out) {
  out.clear();
  size_t pos = 0;
  while (pos < size && static_cast<int64_t>(out.size()) < count) {
    unsigned char header = data[pos++];
    if ((header & 0x01) == 0) {
      int64_t run = header >> 1;
      int bytes = (width + 7) / 8;
      if (pos + bytes > size) return false;
      int64_t value = 0;
      for (int i = 0; i < bytes; ++i) value |= static_cast<int64_t>(data[pos + i]) << (8 * i);
      pos += bytes;
      for (int64_t i = 0; i < run && static_cast<int64_t>(out.size()) < count; ++i) out.push_back(value);
    } else {
      int64_t groups = header >> 1;
      int64_t values = groups * 8;
      size_t bytes = static_cast<size_t>(groups) * static_cast<size_t>(width);
      if (pos + bytes > size) return false;
      uint64_t bit_pos = 0;
      for (int64_t i = 0; i < values && static_cast<int64_t>(out.size()) < count; ++i) {
        uint64_t v = 0;
        for (int b = 0; b < width; ++b) {
          size_t byte_index = pos + static_cast<size_t>((bit_pos + b) >> 3);
          uint64_t bit = (data[byte_index] >> ((bit_pos + b) & 7)) & 1;
          v |= bit << b;
        }
        bit_pos += static_cast<uint64_t>(width);
        out.push_back(static_cast<int64_t>(v));
      }
      pos += bytes;
    }
  }
  return static_cast<int64_t>(out.size()) >= count;
}

static Value physical_to_value(int type, int converted_type, const unsigned char *p, size_t &used,
                               std::string &text);

static Value physical_to_value(int type, int converted_type, const unsigned char *p, size_t &used,
                               std::string &text) {
  Value v;
  v.is_null = false;
  switch (type) {
    case 0: {  // BOOLEAN, bit packed outside
      v.b = (p[0] & 1) != 0;
      used = 1;
      break;
    }
    case 1: {  // INT32
      int32_t x = 0;
      std::memcpy(&x, p, 4);
      v.i = x;
      used = 4;
      break;
    }
    case 2: {  // INT64
      int64_t x = 0;
      std::memcpy(&x, p, 8);
      v.i = x;
      used = 8;
      break;
    }
    case 4: {  // FLOAT
      float f = 0;
      std::memcpy(&f, p, 4);
      v.d = f;
      used = 4;
      break;
    }
    case 5: {  // DOUBLE
      double d = 0;
      std::memcpy(&d, p, 8);
      v.d = d;
      used = 8;
      break;
    }
    case 6: {  // BYTE_ARRAY
      int32_t len = 0;
      std::memcpy(&len, p, 4);
      text.assign(reinterpret_cast<const char *>(p + 4), static_cast<size_t>(len));
      v.s = text;
      used = 4 + static_cast<size_t>(len);
      break;
    }
    default: {
      int32_t len = 0;
      std::memcpy(&len, p, 4);
      text.assign(reinterpret_cast<const char *>(p + 4), static_cast<size_t>(len));
      v.s = text;
      used = 4 + static_cast<size_t>(len);
      break;
    }
  }
  (void)converted_type;
  return v;
}

static ColumnType map_type(int physical, int converted) {
  // converted types: 0 UTF8, 6 DATE, 8 TIMESTAMP_MILLIS, 9 TIMESTAMP_MICROS, 10 UINT_8 ...
  switch (converted) {
    case 0: case 4: case 5: return ColumnType::VARCHAR;  // UTF8 / ENUM / DECIMAL(kept as text)
    case 6: return ColumnType::DATE;
    case 7: case 8: case 9: return ColumnType::DATETIME;
    default: break;
  }
  switch (physical) {
    case 0: return ColumnType::BOOLEAN;
    case 1: case 2: return ColumnType::BIGINT;
    case 4: case 5: return ColumnType::DOUBLE;
    default: return ColumnType::VARCHAR;
  }
}


// Decodes the values of one data page and appends them to the column, NULLs are
// inserted for every definition level that says "missing".
static void append_values(int physical, const unsigned char *data, size_t size,
                          const std::vector<int64_t> &defs, int encoding,
                          const std::vector<Value> &dictionary, std::vector<Value> &out) {
  int64_t present = 0;
  for (size_t i = 0; i < defs.size(); ++i)
    if (defs[i] > 0) ++present;

  std::vector<int64_t> indices;
  std::vector<Value> plain;
  size_t p = 0;
  if (encoding == 2 || encoding == 8) {  // PLAIN_DICTIONARY / RLE_DICTIONARY
    if (size == 0) return;
    int width = data[0];
    ++p;
    decode_rle(data + p, size - p, width, present, indices);
  } else if (physical == 0) {  // BOOLEAN values are bit packed
    for (int64_t i = 0; i < present; ++i) {
      size_t byte_index = static_cast<size_t>(i >> 3);
      if (byte_index >= size) break;
      Value v;
      v.is_null = false;
      v.b = ((data[byte_index] >> (i & 7)) & 1) != 0;
      plain.push_back(v);
    }
  } else {
    while (static_cast<int64_t>(plain.size()) < present) {
      size_t used = 0;
      std::string text;
      Value v = physical_to_value(physical, -1, data + p, used, text);
      if (used == 0) break;
      p += used;
      plain.push_back(v);
    }
  }

  size_t k = 0;
  for (size_t i = 0; i < defs.size(); ++i) {
    if (defs[i] == 0) {
      out.push_back(Value());
      continue;
    }
    if (encoding == 2 || encoding == 8) {
      int64_t index = k < indices.size() ? indices[k] : -1;
      ++k;
      if (index >= 0 && static_cast<size_t>(index) < dictionary.size()) {
        out.push_back(dictionary[static_cast<size_t>(index)]);
      } else {
        out.push_back(Value());
      }
    } else {
      if (k < plain.size()) out.push_back(plain[k]);
      else out.push_back(Value());
      ++k;
    }
  }
}

}  // namespace

ReadResult read_parquet(const std::string &path) {
  ReadResult result;
  std::ifstream in(path.c_str(), std::ios::binary);
  if (!in.good()) {
    result.diagnostics.push_back(Diagnostic(0, "cannot open file " + path));
    return result;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  std::string file = ss.str();
  if (file.size() < 12 || file.compare(0, 4, "PAR1") != 0 ||
      file.compare(file.size() - 4, 4, "PAR1") != 0) {
    result.diagnostics.push_back(Diagnostic(0, "not a parquet file"));
    return result;
  }
  uint32_t meta_len = 0;
  std::memcpy(&meta_len, file.data() + file.size() - 8, 4);
  if (meta_len + 8 > file.size()) {
    result.diagnostics.push_back(Diagnostic(0, "bad parquet footer"));
    return result;
  }
  const unsigned char *meta_start =
      reinterpret_cast<const unsigned char *>(file.data()) + (file.size() - 8 - meta_len);

  FileMeta meta;
  if (!parse_file_meta(meta_start, meta_len, meta)) {
    result.diagnostics.push_back(Diagnostic(0, "cannot parse parquet metadata"));
    return result;
  }

  // flat leaves of the schema tree
  struct Leaf {
    size_t element_index;
    ColumnType type;
    bool optional;
  };
  std::vector<Leaf> leaves;
  for (size_t i = 1; i < meta.schema.size(); ++i) {
    const SchemaElement &e = meta.schema[i];
    if (e.num_children > 0) continue;
    Leaf leaf;
    leaf.element_index = i;
    leaf.type = map_type(e.type, e.converted_type);
    leaf.optional = (e.repetition == 1);
    leaves.push_back(leaf);
  }
  for (size_t i = 0; i < leaves.size(); ++i) {
    Column col;
    col.name = meta.schema[leaves[i].element_index].name;
    col.type = leaves[i].type;
    col.nullable = leaves[i].optional;
    col.source_index = static_cast<int32_t>(i);
    result.schema.columns.push_back(col);
  }

  for (size_t g = 0; g < meta.row_groups.size(); ++g) {
    const RowGroup &group = meta.row_groups[g];
    std::vector<std::vector<Value> > columns(leaves.size());
    for (size_t c = 0; c < group.columns.size() && c < leaves.size(); ++c) {
      const ColumnChunk &chunk = group.columns[c];
      int64_t offset = chunk.dictionary_page_offset > 0 ? chunk.dictionary_page_offset : chunk.data_page_offset;
      std::vector<Value> dictionary;
      while (static_cast<int64_t>(columns[c].size()) < chunk.num_values) {
        if (offset < 0 || static_cast<size_t>(offset) >= file.size()) break;
        CompactReader pr(reinterpret_cast<const unsigned char *>(file.data()) + offset,
                         file.size() - static_cast<size_t>(offset));
        pr.enter_struct();
        int pt = 0, pid = 0;
        int page_type = -1;
        int32_t uncompressed = 0, compressed = 0;
        int64_t num_values = 0;
        int encoding = 0;
        int def_len = 0;
        int rep_len = 0;
        bool is_compressed = false;
        while (pr.read_field(pt, pid)) {
          switch (pid) {
            case 1: page_type = static_cast<int>(pr.read_zigzag()); break;
            case 2: uncompressed = static_cast<int32_t>(pr.read_zigzag()); break;
            case 3: compressed = static_cast<int32_t>(pr.read_zigzag()); break;
            case 5: {
              pr.enter_struct();
              int dt = 0, did = 0;
              while (pr.read_field(dt, did)) {
                if (did == 1) num_values = pr.read_zigzag();
                else if (did == 2) encoding = static_cast<int>(pr.read_zigzag());
                else skip_value(pr, dt);
              }
              pr.leave_struct();
              break;
            }
            case 7: {  // dictionary_page_header
              pr.enter_struct();
              int dt = 0, did = 0;
              while (pr.read_field(dt, did)) {
                if (did == 1) num_values = pr.read_zigzag();
                else if (did == 2) encoding = static_cast<int>(pr.read_zigzag());
                else skip_value(pr, dt);
              }
              pr.leave_struct();
              break;
            }
            case 8: {  // data_page_header_v2
              pr.enter_struct();
              int dt = 0, did = 0;
              while (pr.read_field(dt, did)) {
                if (did == 1) num_values = pr.read_zigzag();
                else if (did == 3) { (void)pr.read_zigzag(); }
                else if (did == 4) encoding = static_cast<int>(pr.read_zigzag());
                else if (did == 5) def_len = static_cast<int>(pr.read_zigzag());
                else if (did == 6) rep_len = static_cast<int>(pr.read_zigzag());
                else if (did == 7) is_compressed = (dt == 1);
                else skip_value(pr, dt);
              }
              pr.leave_struct();
              break;
            }
            default: skip_value(pr, pt); break;
          }
        }
        pr.leave_struct();
        if (!pr.ok()) break;
        size_t data_pos = offset + pr.pos();
        if (data_pos + static_cast<size_t>(compressed) > file.size()) break;
        const unsigned char *payload =
            reinterpret_cast<const unsigned char *>(file.data()) + data_pos;
        size_t payload_size = static_cast<size_t>(compressed);
        (void)uncompressed;

        const bool optional = leaves[c].optional;
        std::string decompressed;
        const unsigned char *data_ptr = payload;
        size_t data_len = payload_size;
        if (chunk.codec == 1 && page_type != 3) {
          if (!snappy_uncompress(payload, static_cast<size_t>(compressed), decompressed)) break;
          data_ptr = reinterpret_cast<const unsigned char *>(decompressed.data());
          data_len = decompressed.size();
        }

        if (page_type == 2) {  // DICTIONARY_PAGE
          size_t used = 0;
          size_t p = 0;
          while (p < data_len && static_cast<int64_t>(dictionary.size()) < num_values) {
            std::string text;
            Value v = physical_to_value(chunk.type, -1, data_ptr + p, used, text);
            if (used == 0) break;
            p += used;
            dictionary.push_back(v);
          }
        } else if (page_type == 0) {  // DATA_PAGE
          size_t p = 0;
          std::vector<int64_t> defs;
          if (optional) {
            int32_t len = 0;
            if (p + 4 > data_len) break;
            std::memcpy(&len, data_ptr + p, 4);
            p += 4;
            if (len < 0 || p + static_cast<size_t>(len) > data_len) break;
            if (!decode_rle(data_ptr + p, static_cast<size_t>(len), 1, num_values, defs)) break;
            p += static_cast<size_t>(len);
          } else {
            defs.assign(static_cast<size_t>(num_values), 1);
          }
          append_values(chunk.type, data_ptr + p, data_len - p, defs, encoding, dictionary, columns[c]);
        } else if (page_type == 3) {  // DATA_PAGE_V2
          size_t q = static_cast<size_t>(rep_len);
          std::vector<int64_t> defs;
          if (optional) {
            if (q + static_cast<size_t>(def_len) > data_len) break;
            if (!decode_rle(data_ptr + q, static_cast<size_t>(def_len), 1, num_values, defs)) break;
            q += static_cast<size_t>(def_len);
          } else {
            defs.assign(static_cast<size_t>(num_values), 1);
          }
          const unsigned char *values_ptr = data_ptr + q;
          size_t values_len = data_len > q ? data_len - q : 0;
          std::string plain;
          if (is_compressed && chunk.codec == 1) {
            if (!snappy_uncompress(values_ptr, values_len, plain)) break;
            values_ptr = reinterpret_cast<const unsigned char *>(plain.data());
            values_len = plain.size();
          }
          append_values(chunk.type, values_ptr, values_len, defs, encoding, dictionary, columns[c]);
        }
        offset = data_pos + static_cast<size_t>(compressed);
        if (compressed <= 0) break;
      }
    }
    for (int64_t r = 0; r < group.num_rows; ++r) {
      Row row;
      row.values.resize(leaves.size());
      for (size_t c = 0; c < leaves.size(); ++c) {
        if (static_cast<size_t>(r) < columns[c].size()) row.values[c] = columns[c][static_cast<size_t>(r)];
      }
      result.rows.push_back(row);
    }
  }
  return result;
}

}  // namespace filesql
// ==== file_sql：解析前把 read_csv/read_json/read_ndjson/read_parquet/glob 展开成常量行 ====
namespace ob_file_sql
{

using filesql::ColumnType;
using filesql::ReadResult;
using filesql::Value;

static std::string fs_lower(const std::string &s)
{
  std::string r = s;
  for (size_t i = 0; i < r.size(); ++i) {
    r[i] = static_cast<char>(::tolower(static_cast<unsigned char>(r[i])));
  }
  return r;
}

static bool fs_ident_char(char c)
{
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
         || (c >= '0' && c <= '9') || c == '_' || c == '$';
}

static void fs_skip_space(const std::string &s, size_t &p)
{
  while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' || s[p] == '\r')) {
    ++p;
  }
}

static bool fs_read_string(const std::string &s, size_t &p, std::string &out)
{
  bool ok = false;
  out.clear();
  if (p < s.size() && s[p] == '\'') {
    ++p;
    ok = true;
    while (p < s.size() && ok) {
      char c = s[p];
      if (c == '\'') {
        if (p + 1 < s.size() && s[p + 1] == '\'') {
          out.push_back('\'');
          p += 2;
        } else {
          ++p;
          break;
        }
      } else if (c == '\\' && p + 1 < s.size()) {
        char n = s[p + 1];
        if (n == 'n') {
          out.push_back('\n');
        } else if (n == 'r') {
          out.push_back('\r');
        } else if (n == 't') {
          out.push_back('\t');
        } else {
          out.push_back(n);
        }
        p += 2;
      } else {
        out.push_back(c);
        ++p;
      }
      if (p >= s.size()) {
        ok = false;
      }
    }
  }
  return ok;
}

struct FsParam
{
  std::string name;   // 空表示位置参数
  std::string sval;
  long long ival;
  bool is_string;
  bool is_int;
  bool is_null;
  FsParam() : ival(0), is_string(false), is_int(false), is_null(false) {}
};

static bool fs_parse_args(const std::string &sql, size_t &p, std::vector<FsParam> &args)
{
  bool ok = true;
  args.clear();
  if (p >= sql.size() || sql[p] != '(') {
    ok = false;
  } else {
    ++p;
    while (ok) {
      fs_skip_space(sql, p);
      if (p >= sql.size()) {
        ok = false;
      } else if (sql[p] == ')') {
        ++p;
        break;
      } else {
        FsParam a;
        size_t q = p;
        while (q < sql.size() && fs_ident_char(sql[q])) {
          ++q;
        }
        size_t r = q;
        fs_skip_space(sql, r);
        if (q > p && r < sql.size() && sql[r] == '=' && (r + 1 >= sql.size() || sql[r + 1] != '=')) {
          a.name = fs_lower(sql.substr(p, q - p));
          p = r + 1;
          fs_skip_space(sql, p);
        }
        if (p >= sql.size()) {
          ok = false;
        } else if (sql[p] == '\'') {
          a.is_string = true;
          ok = fs_read_string(sql, p, a.sval);
        } else if (sql[p] == '-' || (sql[p] >= '0' && sql[p] <= '9')) {
          bool neg = false;
          long long v = 0;
          bool any = false;
          if (sql[p] == '-') {
            neg = true;
            ++p;
          }
          while (p < sql.size() && sql[p] >= '0' && sql[p] <= '9') {
            v = v * 10 + (sql[p] - '0');
            ++p;
            any = true;
          }
          ok = any;
          a.is_int = any;
          a.ival = neg ? -v : v;
        } else if (p + 4 <= sql.size() && fs_lower(sql.substr(p, 4)) == "true") {
          a.is_int = true;
          a.ival = 1;
          p += 4;
        } else if (p + 5 <= sql.size() && fs_lower(sql.substr(p, 5)) == "false") {
          a.is_int = true;
          a.ival = 0;
          p += 5;
        } else if (p + 4 <= sql.size() && fs_lower(sql.substr(p, 4)) == "null") {
          a.is_null = true;
          p += 4;
        } else {
          ok = false;
        }
        if (ok) {
          args.push_back(a);
          fs_skip_space(sql, p);
          if (p < sql.size() && sql[p] == ',') {
            ++p;
          } else if (p < sql.size() && sql[p] == ')') {
            ++p;
            break;
          } else {
            ok = false;
          }
        }
      }
    }
  }
  return ok;
}

static void fs_quote(const std::string &s, std::string &out)
{
  out.push_back('\'');
  for (size_t i = 0; i < s.size(); ++i) {
    char c = s[i];
    if (c == '\'') {
      out += "''";
    } else if (c == '\\') {
      out += "\\\\";
    } else {
      out.push_back(c);
    }
  }
  out.push_back('\'');
}

// 含换行的字符串用 concat(...) 拼，避免把真实换行写进 SQL 文本
static void fs_text_value(const std::string &s, std::string &out)
{
  if (s.find('\n') == std::string::npos && s.find('\r') == std::string::npos) {
    fs_quote(s, out);
  } else {
    out += "concat(";
    bool first = true;
    std::string part;
    for (size_t i = 0; i < s.size(); ++i) {
      char c = s[i];
      if (c == '\n' || c == '\r') {
        if (!first) {
          out += ", ";
        }
        fs_quote(part, out);
        out += ", char(10)";
        first = false;
        part.clear();
        if (c == '\r' && i + 1 < s.size() && s[i + 1] == '\n') {
          ++i;
        }
      } else {
        part.push_back(c);
      }
    }
    if (!first) {
      out += ", ";
    }
    fs_quote(part, out);
    out += ")";
  }
}

static void fs_append_value(const Value &v, ColumnType t, std::string &out)
{
  char buf[80];
  switch (t) {
    case ColumnType::BIGINT:
      if (v.is_null) {
        out += "cast(null as signed)";
      } else {
        snprintf(buf, sizeof(buf), "cast(%lld as signed)", static_cast<long long>(v.i));
        out += buf;
      }
      break;
    case ColumnType::DOUBLE:
      if (v.is_null) {
        out += "cast(null as double)";
      } else {
        snprintf(buf, sizeof(buf), "cast(%.10g as double)", v.d);
        out += buf;
      }
      break;
    case ColumnType::BOOLEAN:
      if (v.is_null) {
        out += "cast(null as signed)";
      } else {
        out += (v.b ? "cast(1 as signed)" : "cast(0 as signed)");
      }
      break;
    case ColumnType::DATE:
      if (v.is_null) {
        out += "cast(null as date)";
      } else {
        out += "cast(";
        fs_quote(v.s, out);
        out += " as date)";
      }
      break;
    case ColumnType::DATETIME:
      if (v.is_null) {
        out += "cast(null as datetime)";
      } else {
        out += "cast(";
        fs_quote(v.s, out);
        out += " as datetime)";
      }
      break;
    default:
      if (v.is_null) {
        out += "cast(null as char)";
      } else {
        fs_text_value(v.s, out);
      }
      break;
  }
}

// 把读到的数据展开成 (select ... union all select ...)
static void fs_build_rows(const ReadResult &res, std::string &out)
{
  const int64_t column_cnt = static_cast<int64_t>(res.schema.columns.size());
  out += "(select ";
  for (int64_t c = 0; c < column_cnt; ++c) {
    if (c > 0) {
      out += ", ";
    }
    if (res.rows.empty()) {
      out += "cast(null as char)";
    } else {
      fs_append_value(res.rows[0].values[static_cast<size_t>(c)],
                      res.schema.columns[static_cast<size_t>(c)].type, out);
    }
    out += " as `";
    out += res.schema.columns[static_cast<size_t>(c)].name;
    out += "`";
  }
  if (res.rows.empty()) {
    out += " where 1 = 0";
  }
  for (size_t r = 1; r < res.rows.size(); ++r) {
    out += " union all select ";
    for (int64_t c = 0; c < column_cnt; ++c) {
      if (c > 0) {
        out += ", ";
      }
      fs_append_value(res.rows[r].values[static_cast<size_t>(c)],
                      res.schema.columns[static_cast<size_t>(c)].type, out);
    }
  }
  out += ")";
}

// DuckDB 风格的 DESCRIBE 结果
static void fs_build_describe(const ReadResult &res, std::string &out)
{
  for (size_t c = 0; c < res.schema.columns.size(); ++c) {
    if (c > 0) {
      out += " union all ";
    }
    out += "select ";
    fs_quote(res.schema.columns[c].name, out);
    out += " as column_name, ";
    fs_quote(filesql::type_name(res.schema.columns[c].type), out);
    out += " as column_type, ";
    out += res.schema.columns[c].nullable ? "'YES' as `null`" : "'NO' as `null`";
    out += ", null as `key`, null as `default`, null as `extra`";
  }
  if (res.schema.columns.empty()) {
    out = "select '' as column_name, '' as column_type, '' as `null`, null as `key`,"
          " null as `default`, null as `extra` where 1 = 0";
  }
}

struct FsCall
{
  std::string name;
  size_t begin;   // 函数名起始位置
  size_t end;     // ')' 之后的位置
  ReadResult result;
  FsCall() : begin(0), end(0) {}
};

// 解析一个表函数调用（name 指向函数名，p 指向 '('），失败返回错误码
static int fs_load_call(const std::string &sql, size_t name_pos, size_t name_len,
                        const std::string &fname, FsCall &call)
{
  int ret = 0;  // 0 表示成功，否则为 MySQL errno（正数）
  size_t p = name_pos + name_len;
  fs_skip_space(sql, p);
  std::vector<FsParam> args;
  if (!fs_parse_args(sql, p, args)) {
    ret = 1064;
  } else {
    std::string path;
    std::string format;
    bool has_header = true;
    long long sample_size = -1;
    bool has_sample = false;
    int64_t positional = 0;
    for (size_t i = 0; i < args.size(); ++i) {
      const FsParam &a = args[i];
      if (a.name.empty()) {
        if (positional == 0 && a.is_string) {
          path = a.sval;
        } else if (positional == 1 && a.is_string) {
          format = fs_lower(a.sval);
        }
        ++positional;
      } else if (a.name == "header") {
        has_header = a.is_int ? (a.ival != 0) : true;
      } else if (a.name == "sample_size") {
        has_sample = true;
        sample_size = a.is_int ? a.ival : -2;
      } else if (a.name == "format") {
        format = a.is_string ? fs_lower(a.sval) : "";
      } else if (a.name == "nullstr" || a.name == "delimiter" || a.name == "quote"
                 || a.name == "escape" || a.name == "compression" || a.name == "columns") {
        // 允许但暂不改变行为
      } else {
        ret = 1235;
      }
    }
    if (0 == ret && path.empty()) {
      ret = 1064;
    }
    if (0 == ret && fname == "read_csv") {
      if (!format.empty() && format != "csv") {
        ret = 1064;
      }
      if (has_sample && sample_size != -1) {
        ret = 1235;
      }
    } else if (0 == ret && (fname == "read_json" || fname == "read_ndjson")) {
      if (!format.empty() && format != "newline_delimited" && format != "json") {
        ret = 1064;
      }
    } else if (0 == ret && fname == "read_parquet") {
      if (!format.empty() && format != "parquet") {
        ret = 1064;
      }
    }
    if (0 == ret) {
      if (fname == "read_csv") {
        filesql::CsvOptions opts;
        opts.has_header = has_header;
        opts.sample_size = sample_size;
        call.result = filesql::read_csv(path, opts);
      } else if (fname == "read_json" || fname == "read_ndjson") {
        call.result = filesql::read_jsonl(path);
      } else if (fname == "read_parquet") {
        call.result = filesql::read_parquet(path);
      } else {
        // glob：单列 file
        std::vector<std::string> files = filesql::glob_paths(path);
        call.result.schema.columns.clear();
        filesql::Column col;
        col.name = "file";
        col.type = ColumnType::VARCHAR;
        col.nullable = false;
        call.result.schema.columns.push_back(col);
        call.result.rows.clear();
        for (size_t i = 0; i < files.size(); ++i) {
          filesql::Row row;
          Value v;
          v.is_null = false;
          v.s = files[i];
          row.values.push_back(v);
          call.result.rows.push_back(row);
        }
      }
      if (!call.result.diagnostics.empty() && call.result.rows.empty()
          && call.result.schema.columns.empty()) {
        ret = 1064;
      }
    }
    call.name = fname;
    call.begin = name_pos;
    call.end = p;
  }
  return ret;
}

static bool fs_is_func_at(const std::string &low, size_t pos, const std::string &name, size_t &name_end)
{
  bool hit = false;
  if (pos + name.size() <= low.size() && low.compare(pos, name.size(), name) == 0) {
    size_t p = pos + name.size();
    fs_skip_space(low, p);
    if (p < low.size() && low[p] == '(') {
      hit = true;
      name_end = pos + name.size();
    }
  }
  return hit;
}

// 返回 OB 错误码（0 表示成功）；out 为改写后的 SQL

static int rewrite_sql(const std::string &in, bool enabled, std::string &out);

// 把 COPY (query) TO 'file' (format csv, header, nullstr 'X') 改写成
// SELECT ... INTO OUTFILE，沿用已有的导出通道（不依赖 FORMAT 元数据）。
static void fs_trim(const std::string &s, size_t &b, size_t &e)
{
  while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\n' || s[b] == '\r')) ++b;
  while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\n' || s[e - 1] == '\r')) --e;
}

// 从 select 列表里取列名（用于生成 header 行）
static void fs_column_names(const std::string &select_list, std::vector<std::string> &names)
{
  std::vector<std::string> items;
  int depth = 0;
  bool in_str = false;
  std::string cur;
  for (size_t i = 0; i < select_list.size(); ++i) {
    char c = select_list[i];
    if (in_str) {
      cur.push_back(c);
      if (c == '\'') {
        if (i + 1 < select_list.size() && select_list[i + 1] == '\'') { cur.push_back(select_list[++i]); }
        else { in_str = false; }
      }
      continue;
    }
    if (c == '\'') { in_str = true; cur.push_back(c); }
    else if (c == '(') { ++depth; cur.push_back(c); }
    else if (c == ')') { --depth; cur.push_back(c); }
    else if (c == ',' && 0 == depth) { items.push_back(cur); cur.clear(); }
    else { cur.push_back(c); }
  }
  if (!cur.empty()) items.push_back(cur);
  for (size_t i = 0; i < items.size(); ++i) {
    std::string item = items[i];
    size_t b = 0, e = item.size();
    fs_trim(item, b, e);
    item = item.substr(b, e - b);
    std::string name = item;
    std::string lowered = fs_lower(item);
    size_t as_pos = lowered.rfind(" as ");
    if (as_pos != std::string::npos) {
      name = item.substr(as_pos + 4);
    } else {
      size_t dot = name.rfind('.');
      if (dot != std::string::npos) name = name.substr(dot + 1);
    }
    size_t nb = 0, ne = name.size();
    if (ne >= 2 && name[0] == '`' && name[ne - 1] == '`') name = name.substr(1, ne - 2);
    fs_trim(name, nb, ne);
    names.push_back(name);
  }
}

static int fs_rewrite_copy(const std::string &sql, std::string &out)
{
  int ret = OB_SUCCESS;
  bool ok = false;
  size_t p = 0;
  fs_skip_space(sql, p);
  p += 4;  // copy
  fs_skip_space(sql, p);
  if (p >= sql.size() || sql[p] != '(') {
    ok = false;
  } else {
    int depth = 0;
    bool in_str = false;
    size_t q = p;
    for (; q < sql.size(); ++q) {
      char c = sql[q];
      if (in_str) {
        if (c == '\'' && q + 1 < sql.size() && sql[q + 1] == '\'') { ++q; }
        else if (c == '\'') { in_str = false; }
      } else if (c == '\'') { in_str = true; }
      else if (c == '(') { ++depth; }
      else if (c == ')') { --depth; if (0 == depth) break; }
    }
    if (q >= sql.size()) {
      ok = false;
    } else {
      std::string inner = sql.substr(p + 1, q - p - 1);
      {
        std::string inner_rewritten;
        int inner_ret = rewrite_sql(inner, true, inner_rewritten);
        if (OB_SUCCESS != inner_ret) {
          return inner_ret;
        }
        inner = inner_rewritten;
      }
      size_t r = q + 1;
      fs_skip_space(sql, r);
      if (r + 2 < sql.size() && fs_lower(sql.substr(r, 2)) == "to") {
        r += 2;
        fs_skip_space(sql, r);
        std::string path;
        if (r < sql.size() && sql[r] == '\'' && fs_read_string(sql, r, path)) {
          // 选项
          std::string rest = sql.substr(r);
          bool has_header = (fs_lower(rest).find("header") != std::string::npos);
          // 列名
          std::vector<std::string> names;
          std::string lowered = fs_lower(inner);
          size_t sel = lowered.find("select");
          if (std::string::npos != sel) {
            size_t from = std::string::npos;
            int d2 = 0; bool s2 = false;
            for (size_t i = sel + 6; i < inner.size(); ++i) {
              char c = inner[i];
              if (s2) { if (c == '\'') s2 = false; continue; }
              if (c == '\'') { s2 = true; }
              else if (c == '(') { ++d2; }
              else if (c == ')') { --d2; }
              else if (0 == d2 && i + 4 <= inner.size()
                       && fs_lower(inner.substr(i, 4)) == "from"
                       && (i == 0 || !fs_ident_char(inner[i - 1]))
                       && (i + 4 >= inner.size() || !fs_ident_char(inner[i + 4]))) {
                from = i;
                break;
              }
            }
            if (std::string::npos != from) {
              fs_column_names(inner.substr(sel + 6, from - sel - 6), names);
            }
          }
          std::string body;
          if (has_header && !names.empty()) {
            body += "select ";
            for (size_t i = 0; i < names.size(); ++i) {
              if (i) body += ", ";
              fs_quote(names[i], body);
              body += " as `";
              body += names[i];
              body += "`";
            }
            body += " union all ";
          }
          body += "(";
          body += inner;
          body += ")";
          out = "select * into outfile ";
          fs_quote(path, out);
          out += " fields terminated by ',' optionally enclosed by '\"' from (";
          out += body;
          out += ") fs_copy_t";
          ok = true;
          LOG_INFO("file_sql copy rewritten", "header", has_header, "cols", static_cast<int64_t>(names.size()),
                   "inner", inner.c_str());
        }
      }
    }
  }
  if (!ok) {
    ret = OB_NOT_SUPPORTED;
  }
  return ret;
}

int rewrite_sql(const std::string &in, bool enabled, std::string &out)
{
  int ret = OB_SUCCESS;
  const char *kNames[] = {"read_csv", "read_json", "read_ndjson", "read_parquet", "glob"};
  const int64_t kNameCnt = 5;
  std::string low = fs_lower(in);

  // COPY (query) TO 'file' (...) 单独处理
  {
    size_t tb = 0;
    fs_skip_space(low, tb);
    if (0 == low.compare(tb, 5, "copy ")) {
      if (!enabled) {
        ret = OB_NOT_SUPPORTED;
      } else {
        ret = fs_rewrite_copy(in, out);
      }
      return ret;
    }
  }

  // 先找出所有函数调用
  std::vector<FsCall> calls;
  std::vector<size_t> starts;
  std::vector<std::string> names;
  for (size_t p = 0; p < low.size(); ++p) {
    if (p > 0 && fs_ident_char(low[p - 1])) {
      continue;
    }
    size_t name_end = 0;
    for (int64_t i = 0; i < kNameCnt; ++i) {
      if (fs_is_func_at(low, p, kNames[i], name_end)) {
        // 记录，稍后统一解析
        FsCall tmp;
        tmp.begin = p;
        tmp.name = kNames[i];
        calls.push_back(tmp);
        starts.push_back(p);
        names.push_back(kNames[i]);
        p = name_end - 1;
        break;
      }
    }
  }

  if (calls.empty()) {
    out = in;
  } else if (!enabled) {
    ret = OB_NOT_SUPPORTED;  // 未开启 enable_file_sql
  } else {
    // 逐个解析（解析后再定位 end，便于跳过参数内容）
    std::vector<FsCall> parsed;
    size_t cursor = 0;
    int err = 0;
    for (size_t i = 0; i < calls.size() && 0 == err; ++i) {
      FsCall c = calls[i];
      c.begin = starts[i];
      err = fs_load_call(in, starts[i], names[i].size(), names[i], c);
      if (0 == err) {
        parsed.push_back(c);
        cursor = c.end;
      }
    }
    if (1064 == err) {
      // 让解析器自己报 1064（客户端看到的就是 1064）
      out = "select * from (select 1) )";
    } else if (0 != err) {
      ret = OB_NOT_SUPPORTED;
    } else {
      // 判断是否是 describe：整条语句只有一条表函数时直接产出元数据结果
      std::string trimmed = low;
      size_t b = 0;
      fs_skip_space(trimmed, b);
      bool is_describe = (trimmed.compare(b, 9, "describe ") == 0)
                         || (trimmed.compare(b, 5, "desc ") == 0);
      if (is_describe && parsed.size() == 1) {
        fs_build_describe(parsed[0].result, out);
      } else {
        out.clear();
        size_t last = 0;
        for (size_t i = 0; i < parsed.size(); ++i) {
          out.append(in, last, parsed[i].begin - last);
          fs_build_rows(parsed[i].result, out);
          last = parsed[i].end;
        }
        out.append(in, last, in.size() - last);
      }
    }
  }
  return ret;
}

}  // namespace ob_file_sql

ObMPQuery::ObMPQuery(const oceanbase::share::ObGlobalContext &gctx)
    : ObMPBase(gctx),
      single_process_timestamp_(0),
      exec_start_timestamp_(0),
      exec_end_timestamp_(0),
      is_com_filed_list_(false),
      params_value_len_(0),
      params_value_(NULL)
{
  ctx_.exec_type_ = MpQuery;
}


ObMPQuery::~ObMPQuery()
{
}


int ObMPQuery::process()
{
  int ret = OB_SUCCESS;
  int tmp_ret = OB_SUCCESS;
  ObSQLSessionInfo *sess = NULL;
  bool need_response_error = true;
  bool need_disconnect = true;
  bool async_resp_used = false; // Asynchronously reply to the client by the transaction commit thread
  int64_t query_timeout = 0;
  ObCurTraceId::TraceId *cur_trace_id = ObCurTraceId::get_trace_id();
  ObSMConnection *conn = get_conn();
  bool do_ins_batch_opt = false;
  if (OB_ISNULL(req_) || OB_ISNULL(conn) || OB_ISNULL(cur_trace_id)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_UNLIKELY(!conn->is_in_authed_phase())) {
    ret = OB_ERR_NO_PRIVILEGE;
  } else if (OB_ISNULL(conn->runtime_)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_ERROR("invalid runtime", K_(sql), K(conn->runtime_), K(ret));
  } else if (OB_FAIL(get_session(sess))) {
  } else if (OB_ISNULL(sess)) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    THIS_WORKER.set_session(sess);
    ObSQLSessionInfo &session = *sess;
    ObSQLSessionInfo::LockGuard lock_guard(session.get_query_lock());
    session.set_current_trace_id(ObCurTraceId::get_trace_id());
    if (OB_SUCC(ret)) {
      int64_t database_schema_version = 0;
      session.set_thread_id(GETTID());
      const ObMySQLRawPacket &pkt = reinterpret_cast<const ObMySQLRawPacket&>(req_->get_packet());
      int64_t packet_len = pkt.get_clen();
      req_->set_trace_point(ObRequest::OB_REQUEST_MPQUERY_PROCESS);
      if (OB_UNLIKELY(!session.is_valid())) {
        ret = OB_ERR_UNEXPECTED;
        LOG_ERROR("invalid session", K_(sql), K(ret));
      } else if (OB_UNLIKELY(session.is_zombie())) {
        //session has been killed some moment ago
        ret = OB_ERR_SESSION_INTERRUPTED;
      } else if (OB_FAIL(session.check_and_init_retry_info(*cur_trace_id, sql_))) {
      } else if (OB_FAIL(session.get_query_timeout(query_timeout))) {
      } else if (OB_FAIL(gctx_.schema_service_->get_published_schema_version(
                  database_schema_version))) {
      } else if (OB_UNLIKELY(packet_len > session.get_max_packet_size())) {
        //packet size check with session variable max_allowd_packet or net_buffer_length
        need_disconnect = false;
        ret = OB_ERR_NET_PACKET_TOO_LARGE;
      } else if (OB_FAIL(session.gen_configs_in_pc_str())) {
      } else {
        THIS_WORKER.set_timeout_ts(get_receive_timestamp() + query_timeout);
        retry_ctrl_.set_current_global_schema_version(database_schema_version);
        session.set_pl_can_retry(true);
        session.set_enable_mysql_compatible_dates(
          session.get_enable_mysql_compatible_dates_from_config());

        bool has_more = false;
        bool force_sync_resp = false;
        need_response_error = false;
        // ---- file sql: 把 read_csv/read_json/read_parquet/glob 在解析前展开成常量行 ----
        if (OB_SUCC(ret) && sql_.length() > 0) {
          bool fs_maybe = false;
          const char *fs_text = sql_.ptr();
          const int32_t fs_len = sql_.length();
          for (int32_t i = 0; i + 4 <= fs_len && !fs_maybe; ++i) {
            const char c0 = static_cast<char>(::tolower(static_cast<unsigned char>(fs_text[i])));
            if ('r' == c0) {
              fs_maybe = (0 == strncasecmp(fs_text + i, "read", 4));
            } else if ('g' == c0) {
              fs_maybe = (0 == strncasecmp(fs_text + i, "glob", 4));
            }
          }
          if (fs_maybe) {
            int64_t fs_flag = 0;
            if (OB_FAIL(session.get_sys_variable(share::SYS_VAR_ENABLE_FILE_SQL, fs_flag))) {
              LOG_WARN("failed to read enable_file_sql", K(ret));
            } else {
              std::string fs_in(fs_text, static_cast<size_t>(fs_len));
              std::string fs_out;
              int fs_ret = ob_file_sql::rewrite_sql(fs_in, 0 != fs_flag, fs_out);
              if (OB_SUCCESS != fs_ret) {
                ret = fs_ret;
                need_response_error = true;
                need_disconnect = false;
                LOG_WARN("failed to rewrite file sql", K(ret), K_(sql));
              } else if (fs_out != fs_in) {
                char *fs_buf = static_cast<char *>(THIS_WORKER.get_sql_arena_allocator().alloc(fs_out.size()));
                if (OB_ISNULL(fs_buf)) {
                  ret = OB_ALLOCATE_MEMORY_FAILED;
                } else {
                  MEMCPY(fs_buf, fs_out.data(), fs_out.size());
                  sql_.assign_ptr(fs_buf, static_cast<int32_t>(fs_out.size()));
                }
              }
            }
          }
        }
        ObParser parser(THIS_WORKER.get_sql_arena_allocator(),
                        session.get_sql_mode(), session.get_charsets4parser());
        //For performance optimization, reduce the array length to lower the construction and destruction overhead of unused elements
        ObSEArray<ObString, 1> queries;
        ObMPParseStat parse_stat;
        if (config::enable_record_trace_id()) {
          PreParseResult pre_parse_result;
          if (OB_FAIL(ObParser::pre_parse(sql_, pre_parse_result))) {
          } else {
            session.set_app_trace_id(pre_parse_result.trace_id_);
            LOG_DEBUG("app trace id", "app_trace_id", pre_parse_result.trace_id_,
                                      "sessid", session.get_server_sid(), K_(sql));
          }
        }

        if (OB_FAIL(ret)) {
          //do nothing
        } else if (OB_FAIL(parser.split_multiple_stmt(sql_, queries, parse_stat))) {
          // Enter this branch, indicating that push_back failed due to OOM, delegate the outer code to return an error code
          // and after entering this branch, the connection should be terminated
          need_response_error = true;
          if (OB_ERR_PARSE_SQL == ret) {
            need_disconnect = false;
          }
        } else if (OB_UNLIKELY(queries.count() <= 0)) {
          ret = OB_ERR_UNEXPECTED;
          need_response_error = true;//Enter this branch, the connection must be terminated, it is a critical error
          LOG_ERROR("emtpy query count. client would have suspended. never be here!", K_(sql), K(ret));
        } else if (OB_UNLIKELY(1 == session.get_capability().cap_flags_.OB_CLIENT_MULTI_STATEMENTS)) {
          // Handle Multiple Statement
          /* MySQL behavior when handling Multi-Stmt errors:
          * After encountering the first failed SQL (including parsing or execution), stop reading subsequent data
            *  For example:
            *  (1) select 1; selct 2; select 3;
            *  select 1 executes successfully, selct 2 reports a syntax error, select 3 is not executed
            *  (2) select 1; drop table not_exists_table; select 3;
            *  select 1 executes successfully, drop table not_exists_table reports a table does not exist error, select 3 is not executed
            *
            * Special note:
            * split_multiple_stmt splits statements based on semicolons, but there might be "syntax errors",
            * here "syntax error" does not mean select is written as selct, but "token" level syntax errors, for example the statement
            * select 1;`select 2; select 3;
            * In the above example, ` and ' do not form closed string tokens, the token parser will report a syntax error
            * In the above example, queries.count() equals 2, which are select 1 and `select 2; select 3;
          */
          bool optimization_done = false;
          const char *p_normal_start = nullptr;
          if (queries.count() > 1
            && OB_FAIL(try_batched_multi_stmt_optimization(session,
                                                          conn,
                                                          queries,
                                                          parse_stat,
                                                          optimization_done,
                                                          async_resp_used,
                                                          need_disconnect,
                                                          false))) {
          } else if (!optimization_done
                     && session.is_enable_batched_multi_statement()
                     && ObSQLUtils::is_enable_explain_batched_multi_statement()
                     && ObParser::is_explain_stmt(queries.at(0), p_normal_start)) {
            ret = OB_SUCC(ret) ? OB_NOT_SUPPORTED : ret;
            need_disconnect = false;
            need_response_error = true;
          } else if (!optimization_done) {
            ARRAY_FOREACH(queries, i) {
              // in multistmt sql, audit_record will record multistmt_start_ts_ when count over 1
              // queries.count()>1 -> batch,(m)sql1,(m)sql2,...    |    queries.count()=1 -> sql1
              if (i > 0) {
                session.get_raw_audit_record().exec_timestamp_.multistmt_start_ts_
                                                              = ObTimeUtility::current_time();
              }
              need_disconnect = true;
              //FIXME qianfu NG_TRACE_EXT(set_disconnect, OB_ID(disconnect), true, OB_ID(pos), "multi stmt begin");
              if (OB_UNLIKELY(parse_stat.parse_fail_
                  && (i == parse_stat.fail_query_idx_)
                  && ObSQLUtils::check_need_disconnect_parser_err(parse_stat.fail_ret_))) {
                // Enter this branch, indicating that parsing of a query in multi_query failed, if not due to a syntax error, then enter this branch
                // If the current query_count is 1, then keep connecting; if greater than 1,
                // then it is necessary to disconnect after sending the error packet to prevent the client from waiting indefinitely for the next response
                // This change is to solve
                ret = parse_stat.fail_ret_;
                need_response_error = true;
                break;
              } else {
                has_more = (queries.count() > i + 1);
                // Originally, it could have been designed so that the last query can always respond asynchronously regardless of queries.count(),
                // However, the current code implementation struggles to handle the response packets of the same request in different threads,
                // Therefore, only one query is allowed in an asynchronous response for a multi-query request here.
                force_sync_resp = queries.count() <= 1? false : true;
                // is_part_of_multi indicates that the current SQL is one of the
                // statements in a multi-statement request.
                bool is_part_of_multi = queries.count() > 1 ? true : false;
                ret = process_single_stmt(ObMultiStmtItem(is_part_of_multi, i, queries.at(i)),
                                          conn,
                                          session,
                                          has_more,
                                          force_sync_resp,
                                          async_resp_used,
                                          need_disconnect);
              }
            }
          }
          // The total number of statements sent over the multiple query protocol
          EVENT_INC(SQL_MULTI_QUERY_COUNT);
          // Sent using the multiple query protocol, but actually contains only one SQL statement count
          if (queries.count() <= 1) {
            EVENT_INC(SQL_MULTI_ONE_QUERY_COUNT);
          }
        } else { // OB_CLIENT_MULTI_STATEMENTS not enabled
          if (OB_UNLIKELY(queries.count() != 1)) {
            ret = OB_ERR_PARSER_SYNTAX;
            need_disconnect = false;
            need_response_error = true;
          } else {
            EVENT_INC(SQL_SINGLE_QUERY_COUNT);
            // Handle ordinary Single Statement
            ret = process_single_stmt(ObMultiStmtItem(false, 0, sql_),
                                      conn,
                                      session,
                                      has_more,
                                      force_sync_resp,
                                      async_resp_used,
                                      need_disconnect);
          }
        }
      }
    }
    // THIS_WORKER.need_retry() means whether to put the request back in the queue for scheduler retry.
    session.check_and_reset_retry_info(*cur_trace_id, THIS_WORKER.need_retry());
    session.set_last_trace_id(ObCurTraceId::get_trace_id());
    // clear thread-local variables used for queue waiting
    // to prevent async callbacks from finishing before 
    // request_finish_callback, which may free the request.
    // this operation should be protected by the session lock.
    if (async_resp_used) {
      if (OB_FAIL(ret) && need_disconnect && is_conn_valid()) {
        force_disconnect();
      }
      const int handoff_ret = handoff_async_request(session.get_mysql_end_trans_cb());
      if (OB_UNLIKELY(OB_SUCCESS != handoff_ret)) {
        LOG_ERROR("failed to hand off async mysql request ownership", K(handoff_ret));
        force_disconnect();
      }
    }
  }

  if (!async_resp_used && OB_FAIL(ret) && need_response_error && is_conn_valid()) {
    send_error_packet(ret, NULL);
  }
  if (!async_resp_used && OB_FAIL(ret) && OB_UNLIKELY(need_disconnect) && is_conn_valid()) {
    force_disconnect();
  }
  // If the response has already been sent asynchronously, this logic will be executed in cb, so skip flush_buffer() here
  if (async_resp_used) {
    // The callback sender already owns response and finish.
  } else if (!THIS_WORKER.need_retry()) {
    if (OB_UNLIKELY(!is_conn_valid())) {
      tmp_ret = OB_CONNECT_ERROR;
      LOG_WARN("connection in error, maybe has disconnected", K(tmp_ret));
    } else if (OB_UNLIKELY(OB_SUCCESS != (tmp_ret = flush_buffer(true)))) {
    }
  }

  // bugfix: 
  // Must always set the pointer in THIS_WORKER to null
  THIS_WORKER.set_session(NULL); // clear session

  if (sess != NULL) {
    revert_session(sess); //current ignore revert session ret
  }

  return (OB_SUCCESS != ret) ? ret : tmp_ret;
}

/*
 * Try to evaluate multiple update queries as a single query to optimize rpc cost
 * for details, please ref to 
 */
int ObMPQuery::try_batched_multi_stmt_optimization(sql::ObSQLSessionInfo &session,
                                                   ObSMConnection *conn,
                                                   common::ObIArray<ObString> &queries,
                                                   const ObMPParseStat &parse_stat,
                                                   bool &optimization_done,
                                                   bool &async_resp_used,
                                                   bool &need_disconnect,
                                                   bool is_ins_multi_val_opt)
{
  int ret = OB_SUCCESS;
  bool has_more = false;
  bool force_sync_resp = true;
  bool enable_batch_opt = session.is_enable_batched_multi_statement();
  bool use_plan_cache = session.get_local_ob_enable_plan_cache();
  optimization_done = false;
  if (queries.count() <= 1 || parse_stat.parse_fail_) {
    /*do nothing*/
  } else if (!enable_batch_opt) {
    // batch switch is not turned on
  } else if (!use_plan_cache) {
    // If the plan_cache switch is not turned on, optimization is not supported
  } else if (OB_FAIL(process_single_stmt(ObMultiStmtItem(false, 0, sql_, &queries, is_ins_multi_val_opt),
                                         conn,
                                         session,
                                         has_more,
                                         force_sync_resp,
                                         async_resp_used,
                                         need_disconnect))) {
    int tmp_ret = ret;
    if (THIS_WORKER.need_retry()) {
      // Keep the retry decision for the upper scheduler.
    } else {
      ret = OB_SUCCESS;
    }
    LOG_WARN("failed to process batch stmt, cover the error code and reset retry flag",
        K(tmp_ret), K(ret), K(THIS_WORKER.need_retry()));
  } else {
    optimization_done = true;
  }

  LOG_TRACE("after to try batched multi-stmt optimization", K(optimization_done),
      K(queries), K(enable_batch_opt), K(ret), K(THIS_WORKER.need_retry()), K(retry_ctrl_.need_retry()), K(retry_ctrl_.get_retry_type()));
  return ret;
}

ERRSIM_POINT_DEF(ERRSIM_BEGIN_COMMIT_OPT_DISABLE)
int ObMPQuery::process_single_stmt(const ObMultiStmtItem &multi_stmt_item,
                                   ObSMConnection *conn,
                                   ObSQLSessionInfo &session,
                                   bool has_more_result,
                                   bool force_sync_resp,
                                   bool &async_resp_used,
                                   bool &need_disconnect)
{
  int ret = OB_SUCCESS;
  sql::OTraceGuard trace(session, multi_stmt_item.get_sql());
  sql::ObTraceSpanGuard query_span(&session, sql::TRACE_COM_QUERY_PROCESS);
  sql::ObTraceSpanGuard stmt_span(&session, sql::TRACE_MPQUERY_SINGLE_STMT);
  ObReqTimeGuard req_timeinfo_guard;
  ctx_.plan_key_.reset();
  bool need_response_error = true;
  session.get_raw_audit_record().request_memory_used_ = 0;
  sql::ObProcessMallocCallback pmcb(0,
        session.get_raw_audit_record().request_memory_used_);
  lib::ObMallocCallbackGuard guard(pmcb);
  // After executing setup_wb, all WARNINGS will be written to the WARNING BUFFER of the current session
  setup_wb(session);
  // When a new statement starts, set this value to 0, because curr_trans_last_stmt_end_time is used for
  // Implement the timeout function for excessively long execution intervals of statements within a transaction.
  session.set_curr_trans_last_stmt_end_time(0);
  //============================ Pay attention to the lifecycle of these variables ================================
  if (OB_FAIL(init_process_var(ctx_, multi_stmt_item, session))) {
  } else {
    //set session log_level.Must use ObThreadLogLevelUtils::clear() in pair
    ObThreadLogLevelUtils::init(session.get_log_id_level_map());
    bool is_trans_ctrl_cmd = false;
    bool do_trans_ctrl_opt = false;
    stmt::StmtType stmt_type = stmt::T_NONE;
    if (!ERRSIM_BEGIN_COMMIT_OPT_DISABLE && !multi_stmt_item.is_part_of_multi_stmt()) {
      check_is_trans_ctrl_cmd(multi_stmt_item.get_sql(), is_trans_ctrl_cmd, stmt_type);
      if (is_trans_ctrl_cmd) {
        do_trans_ctrl_opt = true;
      }
    }

    // Clients may publish a newer schema version through @@last_schema_version;
    // observer refreshes when its local version is older.
    if (!do_trans_ctrl_opt && OB_FAIL(check_and_refresh_schema(&session))) {
    } else if (!do_trans_ctrl_opt && OB_FAIL(session.update_timezone_info())) {
    } else {
      need_response_error = false;
      //Each execution of different SQL requires an update
      ctx_.self_add_plan_ = false;
      retry_ctrl_.reset_retry_times();//each statement records retry times separately
      do {
        ret = OB_SUCCESS; //When a local retry occurs, the error code needs to be reset, otherwise the retry cannot proceed
        need_disconnect = true;
        // do the real work
        //create a new temporary memory context for executing sql can
        //avoid the problem the memory cannot be released in time due to too many sql items
        //but it will drop the sysbench performance about 1~4%
        //so we execute the first sql with the default memory context and
        //execute the rest sqls with a temporary memory context to avoid memory dynamic leaks
        retry_ctrl_.clear_state_before_each_retry(session.get_retry_info_for_update());
        bool first_exec_sql = session.get_is_in_retry() ? false :
            (multi_stmt_item.is_part_of_multi_stmt() ? multi_stmt_item.get_seq_num() <= 1 : true);
        if (do_trans_ctrl_opt) {
          ret = do_process_trans_ctrl(session,
                                      has_more_result,
                                      force_sync_resp,
                                      async_resp_used,
                                      need_disconnect,
                                      stmt_type);
          ctx_.clear();
        } else if (OB_LIKELY(first_exec_sql)) {
          ret = do_process(session,
                           has_more_result,
                           force_sync_resp,
                           async_resp_used,
                           need_disconnect);
          ctx_.clear();
        } else {
          ret = process_with_tmp_context(session,
                                         has_more_result,
                                         force_sync_resp,
                                         async_resp_used,
                                         need_disconnect);
        }
        //set session retry state
        session.set_session_in_retry(retry_ctrl_.need_retry());
      } while (RETRY_TYPE_LOCAL == retry_ctrl_.get_retry_type());
    }
    ObThreadLogLevelUtils::clear();
    const int64_t debug_sync_timeout = config::debug_sync_timeout();
    if (debug_sync_timeout > 0) {
      // ignore thread local debug sync actions to session actions failed
      int tmp_ret = OB_SUCCESS;
      tmp_ret = GDS.collect_result_actions(session.get_debug_sync_actions());
      if (OB_UNLIKELY(OB_SUCCESS != tmp_ret)) {
      }
    }
  }
  //For the handling of tracelog, it does not affect the normal logic, and the error code does not need to be assigned to ret
  int tmp_ret = OB_SUCCESS;
  //Clear WARNING BUFFER
  tmp_ret = do_after_process(session, async_resp_used, ret);
  // Set the end time of the previous statement, since here it is only used to implement the execution timeout between statements within a transaction,
  // Therefore, first, it is necessary to determine whether a transaction execution process is in progress. Then for the asynchronous response at the time of transaction submission,
  // It is also not necessary to set the end time here, because this is already equivalent to the last statement of the transaction.
  // Finally, need to determine the ret error code, only successfully executed sql records the end time
  if (session.get_in_transaction() && !async_resp_used && OB_SUCC(ret)) {
    session.set_curr_trans_last_stmt_end_time(ObClockGenerator::getClock());
  }
  // the need_response_error variable ensures that it only occurs in
  // do { do_process } while(retry) will only occur if an error happens before
  // Walk to the send_error_packet logic
  // So there is no need to consider whether the current mode is sync or async
  if (!OB_SUCC(ret) && need_response_error && is_conn_valid() &&
        !ctx_.multi_stmt_item_.is_batched_multi_stmt()) {
    send_error_packet(ret, NULL);
  }
  ctx_.plan_key_.reset();
  ctx_.reset();
  return ret;
}

OB_NOINLINE int ObMPQuery::process_with_tmp_context(ObSQLSessionInfo &session,
                                                    bool has_more_result,
                                                    bool force_sync_resp,
                                                    bool &async_resp_used,
                                                    bool &need_disconnect)
{
  int ret = OB_SUCCESS;
  //create a temporary memory context to process retry or the rest sql of multi-query,
  //avoid memory dynamic leaks caused by query retry or too many multi-query items
  lib::ContextParam param;
  param.set_mem_attr(ObModIds::OB_SQL_EXECUTOR, ObCtxIds::DEFAULT_CTX_ID)
    .set_properties(lib::USE_TL_PAGE_OPTIONAL)
    .set_page_size(OB_MALLOC_REQ_NORMAL_BLOCK_SIZE)
    .set_ablock_size(lib::INTACT_MIDDLE_AOBJECT_SIZE);
  CREATE_WITH_TEMP_CONTEXT(param) {
    ret = do_process(session,
                     has_more_result,
                     force_sync_resp,
                     async_resp_used,
                     need_disconnect);
    ctx_.first_plan_hash_ = 0;
    ctx_.first_outline_data_.reset();
    ctx_.first_equal_param_cons_cnt_ = 0;
    ctx_.first_const_param_cons_cnt_ = 0;
    ctx_.first_expr_cons_cnt_ = 0;
    ctx_.clear();
  }
  return ret;
}

OB_INLINE int ObMPQuery::get_schema_info_(ObCachedSchemaGuardInfo *cache_info,
                                          ObSchemaGetterGuard *&schema_guard,
                                          int64_t &database_schema_version)
{
  int ret = OB_SUCCESS;
  ObSchemaGetterGuard &cached_guard = cache_info->get_schema_guard();
  bool need_refresh = false;

  if (!cached_guard.is_inited()) {
    // First get schema guard
    need_refresh = true;
  } else {
    int64_t tmp_database_schema_version = 0;
    if (OB_FAIL(gctx_.schema_service_->get_runtime_refreshed_schema_version(tmp_database_schema_version))) {
    } else if (OB_FAIL(cached_guard.get_schema_version(database_schema_version))) {
    } else if (tmp_database_schema_version != database_schema_version) {
      //Need to obtain schema guard
      need_refresh = true;
    } else {
      // do nothing
    }
  }
  if (OB_SUCC(ret)) {
    if (!need_refresh) {
      //Get the latest schema guard cached on the session
      schema_guard = &(cache_info->get_schema_guard());
    } else if (OB_FAIL(cache_info->refresh_runtime_schema_guard())) {
    } else {
      //Get the latest schema guard cached on the session
      schema_guard = &(cache_info->get_schema_guard());
      if (OB_FAIL(schema_guard->get_schema_version(database_schema_version))) {
      } else {
        // do nothing
      }
    }
  }

  return ret;
}

void ObMPQuery::check_is_trans_ctrl_cmd(const ObString &sql,
                                        bool &is_trans_ctrl_cmd,
                                        stmt::StmtType &stmt_type)
{
  is_trans_ctrl_cmd = false;
  const uint32_t cmd_len = sql.length();
  if (5 <= cmd_len && cmd_len <=8) {
    if (cmd_len == 5) {
      if (0 == sql.case_compare("begin")) {
        is_trans_ctrl_cmd = true;
        stmt_type = stmt::T_START_TRANS;
      }
    } else if (cmd_len == 6) {
      if (0 == sql.case_compare("commit")) {
        is_trans_ctrl_cmd = true;
        stmt_type = stmt::T_END_TRANS;
      }
    } else if (cmd_len == 8) {
      if (0 == sql.case_compare("rollback")) {
        is_trans_ctrl_cmd = true;
        stmt_type = stmt::T_END_TRANS;
      }
    }
  }
}

OB_INLINE int ObMPQuery::do_process_trans_ctrl(ObSQLSessionInfo &session,
                                               bool has_more_result,
                                               bool force_sync_resp,
                                               bool &async_resp_used,
                                               bool &need_disconnect,
                                               stmt::StmtType stmt_type)
{
  int ret = OB_SUCCESS;
  ObAuditRecordData &audit_record = session.get_raw_audit_record();
  ObExecutingSqlStatRecord sqlstat_record;
  audit_record.try_cnt_++;
  bool is_diagnostics_stmt = false;
  bool need_response_error = true;
  const ObString &sql = ctx_.multi_stmt_item_.get_sql();
  const bool enable_sqlstat = session.is_sqlstat_enabled();
  bool is_rollback = false;
  bool is_commit = false;
  if (0 == sql.case_compare("commit")) {
    is_commit = true;
  } else if (0 == sql.case_compare("rollback")) {
    is_rollback = true;
  }
  single_process_timestamp_ = ObTimeUtility::current_time();
  /* !!!
   * Note that req_timeinfo_guard must be placed before result
   * !!!
   */
  ObReqTimeGuard req_timeinfo_guard;
  ObCachedSchemaGuardInfo &cached_schema_info = session.get_cached_schema_guard_info();
  SQL_INFO_GUARD(sql, session.get_cur_sql_id());
  need_disconnect = false;

  session.reset_plsql_exec_time();
  session.set_stmt_type(stmt_type);

  if (OB_SUCC(ret)) {
    {
      audit_record.exec_record_.record_start();
    }
    if (enable_sqlstat) {
      sqlstat_record.record_sqlstat_start_value(
          ::oceanbase::observer::get_observer_sql_engine()->get_query_runtime_environment());
      sqlstat_record.set_is_in_retry(false);
      session.sql_sess_record_sql_stat_start_value(sqlstat_record);
    }
    if (OB_FAIL(set_session_active(sql, session, single_process_timestamp_))) {
    } else {
      // generate sql_id with sql cmd
      (void)ObSQLUtils::md5(sql, ctx_.sql_id_, (int32_t)sizeof(ctx_.sql_id_));
      session.set_cur_sql_id(ctx_.sql_id_);
      //Monitoring item statistics start
      exec_start_timestamp_ = ObTimeUtility::current_time();
      need_response_error = false;
      is_diagnostics_stmt = false;
      ctx_.is_show_trace_stmt_ = false;

      // exec cmd
      sql::ObTraceSpanGuard exec_span(&session, sql::TRACE_SQL_EXECUTE);
      if (OB_FAIL(process_trans_ctrl_cmd(session,
                                         need_disconnect,
                                         async_resp_used,
                                         is_rollback,
                                         force_sync_resp,
                                         stmt_type))) {
        need_response_error = true;
        LOG_ERROR("fail to execute trans ctrl cmd", KR(ret), K(sql));
      }
      // If there is no asynchronous submission, and the execution is successful, then send an ok packet to the client
      if (!async_resp_used && OB_SUCC(ret)) {
        ObOKPParam ok_param;
        ok_param.affected_rows_ = 0;
        ok_param.has_more_result_ = has_more_result;
        if (OB_FAIL(send_ok_packet(session, ok_param))) {
        }
      }
    }
    // Note: Do not use the sql_ member variable after calling the response_result interface, because the memory that sql_ points to comes from ObReqPacket
    // while in the transaction asynchronous commit response client process, the response client process is located in the ApplyService thread, and there is no timing guarantee with the SQL Worker thread
    //After starting the asynchronous response client process with response_result, it is possible that the response packet will be returned before executing the following logic. If the response packet is completed first, it will lead to
    //ObReqPacket memory is released, thus leading to sql_ pointing to an invalid memory address, accessing sql_ in the following logic will cause a crash
    // And the process of synchronously responding to the client does not have similar issues, because the flush action for the last packet in the synchronous response client is located at the very bottom of this interface,
    //But the asynchronous response process does not have this guarantee
    int tmp_ret = OB_SUCCESS;
    tmp_ret = OB_E(EventTable::EN_PRINT_QUERY_SQL) OB_SUCCESS;
    if (OB_SUCCESS != tmp_ret) {
    }
    //Monitoring item statistics end
    exec_end_timestamp_ = ObTimeUtility::current_time();

    // some statistics must be recorded for plan stat, even though sql audit disabled
    bool first_record = (1 == audit_record.try_cnt_);
    ObExecStatUtils::record_exec_timestamp(*this, first_record, audit_record.exec_timestamp_);
    audit_record.exec_timestamp_.update_stage_time();

    // store the warning message from the most recent statement in the current session
    if ((OB_SUCC(ret) && is_diagnostics_stmt) || async_resp_used) {
      // If diagnostic stmt execute successfully, it dosen't clear the warning message.
      // Or if it response to client asynchronously, it doesn't clear the warning message here,
      // but will do it in the callback thread.
      session.update_show_warnings_buf();
    } else {
      session.set_show_warnings_buf(ret); // TODO: Move this to a better place, reduce some wb copying
    }

    if (OB_FAIL(ret) && !async_resp_used && need_response_error && is_conn_valid() &&
          !ctx_.multi_stmt_item_.is_batched_multi_stmt()) {
      // This request has errored, it hasn't been processed yet. If it hasn't already been handed over to async EndTrans for cleanup,
      // then it is necessary to reply with an error_packet as a footer.
      // Otherwise, no one will help send the error packet to the client, which may cause the client to hang waiting for a response.
      int err = send_error_packet(ret, NULL);
      if (OB_SUCCESS != err) {
      }
    }
  }

  {
    audit_record.exec_record_.record_end();
    audit_record.stmt_type_ = stmt_type;
    audit_record.update_event_stage_state();
  }
  if (enable_sqlstat) {
    sqlstat_record.record_sqlstat_end_value(
        ::oceanbase::observer::get_observer_sql_engine()->get_query_runtime_environment());
    sqlstat_record.set_rows_processed(0);
    sqlstat_record.set_partition_cnt(0);
    sqlstat_record.set_is_plan_cache_hit(ctx_.plan_cache_hit_);
  }

  audit_record.status_ = (0 == ret || OB_ITER_END == ret)
      ? 0 : (ret);
  // reset thread waring buffer in sync mode
  if (!async_resp_used) {
    clear_wb_content(session);
  }

  need_disconnect = (need_disconnect && !is_query_killed_return(ret)); // Clearly it is a kill query, the connection should not be disconnected
  if (need_disconnect) {
  }
  bool is_need_retry = false;

  return ret;
}

int ObMPQuery::process_trans_ctrl_cmd(ObSQLSessionInfo &session,
                                      bool &need_disconnect,
                                      bool &async_resp_used,
                                      const bool is_rollback,
                                      const bool force_sync_resp,
                                      stmt::StmtType stmt_type)
{
  int ret = OB_SUCCESS;
  if (stmt_type == stmt::T_START_TRANS) {
    bool read_only = session.get_tx_read_only();
    transaction::ObTxParam tx_param;
    TransState trans_state;
    // End the current transaction before starting a new explicit transaction.
    if (OB_FAIL(ObSqlTransControl::end_trans_before_cmd_execute(session,
                                                                need_disconnect,
                                                                trans_state,
                                                                stmt_type))) {
    }
    if (OB_SUCC(ret) && OB_FAIL(ObSqlTransControl::explicit_start_trans(&session,
                                                                        tx_param,
                                                                        need_disconnect,
                                                                        read_only))) {
    }
  } else if (stmt_type == stmt::T_END_TRANS) {
    bool need_end_trans_callback = false;
    bool callback_armed = false;
    ObEndTransAsyncCallback *callback = nullptr;
    TransState trans_state;
    ObEndTransCbPacketParam pkt_param;

    if (session.get_has_temp_table_flag() || session.has_tx_level_temp_table()) {
      // temporary table will be committed synchronously, and then drop_temp_tables will be called to delete the data.
      need_end_trans_callback = false;
    } else {
      need_end_trans_callback = true;
    }

    bool need_trans_cb  = need_end_trans_callback && (!force_sync_resp);
    if (need_trans_cb) {
      ObSqlEndTransCb &sql_end_cb = session.get_mysql_end_trans_cb();
      ObCurTraceId::TraceId *cur_trace_id = NULL;
      if (OB_ISNULL(cur_trace_id = ObCurTraceId::get_trace_id())) {
        ret = OB_ERR_UNEXPECTED;
        LOG_ERROR("current trace id is NULL", K(ret));
      } else if (OB_FAIL(sql_end_cb.init(packet_sender_, &session))) {
      } else if (FALSE_IT(callback_armed = true)) {
      } else if (OB_FAIL(sql_end_cb.set_packet_param(pkt_param.fill("\0", // message
                                                                    0,  // affected_rows
                                                                    0,  // last_insert_id_to_client
                                                                    *cur_trace_id)))) {
      } else {
        callback = &session.get_end_trans_cb();
      }
    }

    if (OB_FAIL(ret)) {
      // do nothing
    } else if (OB_FAIL(ObSqlTransControl::end_trans(&session,
                                                    need_disconnect,
                                                    trans_state,
                                                    is_rollback,
                                                    true, // is_explicit
                                                    callback))) {
    }
    const bool callback_submitted =
        trans_state.is_end_trans_executed() && trans_state.is_end_trans_success();
    if (callback_submitted) {
      async_resp_used = true;
    } else if (callback_armed) {
      const int cancel_ret = cancel_unsubmitted_callback(session.get_mysql_end_trans_cb());
      if (OB_SUCCESS != cancel_ret) {
        LOG_ERROR("failed to cancel unsubmitted mysql end-trans callback", K(cancel_ret));
        ret = OB_SUCCESS == ret ? cancel_ret : ret;
      }
    }
  }
  return ret;
}

OB_INLINE int ObMPQuery::do_process(ObSQLSessionInfo &session,
                                    bool has_more_result,
                                    bool force_sync_resp,
                                    bool &async_resp_used,
                                    bool &need_disconnect)
{
  int ret = OB_SUCCESS;
  ObAuditRecordData &audit_record = session.get_raw_audit_record();
  ObExecutingSqlStatRecord sqlstat_record;
  audit_record.try_cnt_++;
  bool is_diagnostics_stmt = false;
  bool need_response_error = true;
  const ObString &sql = ctx_.multi_stmt_item_.get_sql();
  const bool enable_sqlstat = session.is_sqlstat_enabled();
  single_process_timestamp_ = ObTimeUtility::current_time();
  /* !!!
   * Note that req_timeinfo_guard must be placed before result
   * !!!
   */
  ObReqTimeGuard req_timeinfo_guard;
  ObPhysicalPlan *plan = nullptr;
  ObSchemaGetterGuard* schema_guard = nullptr;
  ObCachedSchemaGuardInfo &cached_schema_info = session.get_cached_schema_guard_info();
  int64_t database_schema_version = 0;
  SQL_INFO_GUARD(sql, session.get_cur_sql_id());
  ObIAllocator &allocator = CURRENT_CONTEXT->get_arena_allocator();
  SMART_VAR(ObMySQLResultSet, result, session, allocator,
            ::oceanbase::observer::get_observer_sql_engine()->get_plan_cache_access_service()) {
    if (OB_FAIL(get_schema_info_(&cached_schema_info,
                                 schema_guard,
                                 database_schema_version))) {
    } else if (OB_FAIL(session.update_query_sensitive_system_variable(*schema_guard))) {
    } else if (OB_ISNULL(::oceanbase::observer::get_observer_sql_engine())) {
      ret = OB_ERR_UNEXPECTED;
      LOG_ERROR("invalid sql engine", K(ret), K(gctx_));
    } else {
      session.set_current_execution_id(::oceanbase::observer::get_observer_sql_engine()->get_execution_id());
      session.reset_plsql_exec_time();
      session.reset_plsql_compile_time();
      session.set_stmt_type(stmt::T_NONE);
      result.get_exec_context().set_need_disconnect(true);
      ctx_.schema_guard_ = schema_guard;
      retry_ctrl_.set_current_local_schema_version(database_schema_version);
    }

    if (OB_SUCC(ret)) {
      {
        audit_record.exec_record_.record_start();
      }
      if (enable_sqlstat) {
        sqlstat_record.record_sqlstat_start_value(
            ::oceanbase::observer::get_observer_sql_engine()->get_query_runtime_environment());
        sqlstat_record.set_is_in_retry(session.get_is_in_retry());
        session.sql_sess_record_sql_stat_start_value(sqlstat_record);
      }
      result.set_has_more_result(has_more_result);
      ObSqlExecutorCtx &task_ctx = result.get_exec_context().get_sql_exec_ctx();
      task_ctx.schema_service_ = gctx_.schema_service_;
      task_ctx.set_query_begin_schema_version(retry_ctrl_.get_current_local_schema_version());
      ctx_.retry_times_ = retry_ctrl_.get_retry_times();
      //storage::ObPartitionService* ps = static_cast<storage::ObPartitionService *> (GCTX.par_ser_);
      //bool is_read_only = false;
      if (OB_FAIL(ret)) {
        // do nothing
      } else if (OB_ISNULL(ctx_.schema_guard_)) {
        ret = OB_INVALID_ARGUMENT;
      } else if (OB_FAIL(set_session_active(sql, session, single_process_timestamp_))) {
      } else if (OB_FAIL(::oceanbase::observer::get_observer_sql_engine()->stmt_query(sql, ctx_, result))) {
        exec_start_timestamp_ = ObTimeUtility::current_time();
        if (!THIS_WORKER.need_retry()) {
          int cli_ret = OB_SUCCESS;
          retry_ctrl_.test_and_save_retry_state(gctx_, ctx_, result, ret, cli_ret);
          ret = cli_ret;
          if (ctx_.multi_stmt_item_.is_batched_multi_stmt()) {
            // batch execute with error,should not response error packet
            need_response_error = false;
          } else if (OB_BATCHED_MULTI_STMT_ROLLBACK == ret) {
            need_response_error = false;
          }
        } else {
          retry_ctrl_.set_packet_retry();
          session.get_retry_info_for_update().set_last_query_retry_err(ret);
          session.get_retry_info_for_update().inc_retry_cnt();
        }
      } else {
        //Monitoring item statistics start
        exec_start_timestamp_ = ObTimeUtility::current_time();
        result.get_exec_context().set_plan_start_time(exec_start_timestamp_);
        // All errors within this branch will be handled properly inside response_result
        // No need to handle the error response packet additionally
        need_response_error = false;
        is_diagnostics_stmt = ObStmt::is_diagnostic_stmt(result.get_literal_stmt_type());
        ctx_.is_show_trace_stmt_ = ObStmt::is_show_trace_stmt(result.get_literal_stmt_type());
        plan = result.get_physical_plan();

        if (get_is_com_filed_list()) {
          result.set_is_com_filed_list();
          result.set_wildcard_string(wild_str_);
        }

        //response_result
        if (OB_FAIL(ret)) {
        //TODO shengle, confirm whether 4.0 is required
        //} else if (OB_FAIL(fill_feedback_session_info(*result, session))) {
          //need_response_error = true;
          //LOG_WARN("failed to fill session info", K(ret));
        } else if (OB_FAIL(response_result(result,
                                           force_sync_resp,
                                           async_resp_used))) {
          ObPhysicalPlanCtx *plan_ctx = result.get_exec_context().get_physical_plan_ctx();
          if (OB_ISNULL(plan_ctx)) {
            // ignore ret
            LOG_ERROR("execute query fail, and plan_ctx is NULL", K(ret));
          } else {
            if (OB_TRANSACTION_SET_VIOLATION != ret && OB_REPLICA_NOT_READABLE != ret) {
            }
          }
        }
      }
      // Note: Do not use the sql_ member variable after calling the response_result interface, because the memory that sql_ points to comes from ObReqPacket
      // while in the transaction asynchronous commit response client process, the response client process is located in the ApplyService thread, and there is no timing guarantee with the SQL Worker thread
      //After starting the asynchronous response client process with response_result, it is possible that the response packet will be returned before executing the following logic. If the response packet is completed first, it will lead to
      //ObReqPacket memory is released, thus leading to sql_ pointing to an invalid memory address, accessing sql_ in the following logic will cause a crash
      // While the process of synchronously responding to the client does not have similar issues, this is because the flush action for the last packet in the synchronous response client is located at the very bottom of this interface,
      //but the asynchronous response process does not have this guarantee
      int tmp_ret = OB_SUCCESS;
      tmp_ret = OB_E(EventTable::EN_PRINT_QUERY_SQL) OB_SUCCESS;
      if (OB_SUCCESS != tmp_ret) {
      }
      //Monitoring item statistics end
      exec_end_timestamp_ = ObTimeUtility::current_time();

      // some statistics must be recorded for plan stat, even though sql audit disabled
      bool first_record = (1 == audit_record.try_cnt_);
      ObExecStatUtils::record_exec_timestamp(*this, first_record, audit_record.exec_timestamp_);
      audit_record.exec_timestamp_.update_stage_time();

      // Retry needs to meet the following conditions:
      // 1. rs.open execution failed
      // 2. No result was returned to the client, this execution has no side effects
      // 3. need_retry(result, ret): schema or location cache invalidation
      // 4. less than retry count limit
      if (OB_UNLIKELY(retry_ctrl_.need_retry())) {
        if (OB_TRANSACTION_SET_VIOLATION != ret && OB_REPLICA_NOT_READABLE != ret && OB_TRY_LOCK_ROW_CONFLICT != ret) {
          //Lock conflict retry does not print log, to avoid screen flooding
        }
      } else {
        // store the warning message from the most recent statement in the current session
        if ((OB_SUCC(ret) && is_diagnostics_stmt) || async_resp_used) {
          // If diagnostic stmt execute successfully, it dosen't clear the warning message.
          // Or if it response to client asynchronously, it doesn't clear the warning message here,
          // but will do it in the callback thread.
          session.update_show_warnings_buf();
        } else {
          session.set_show_warnings_buf(ret); // TODO: Move this to a better place, reduce some wb copy
        }

        if (OB_FAIL(ret) && !async_resp_used && need_response_error && is_conn_valid() && !THIS_WORKER.need_retry() &&
              !ctx_.multi_stmt_item_.is_batched_multi_stmt()) {
          // When need_retry=false, a packet may have been sent to the client, or no packets may have been sent at all.
          // However, it can be determined: this request has errored, and is not yet complete. If it has not already been handed over to asynchronous EndTrans for finalization,
          // then it is necessary to reply with an error_packet below as a conclusion. Otherwise, no one will help send the error packet to the client afterwards,
          // May cause the client to hang waiting for a response.
          int err = send_error_packet(ret, NULL);
          if (OB_SUCCESS != err) {
          }
        }
      }
    }

    {
      audit_record.exec_record_.record_end();
      audit_record.stmt_type_ = result.get_stmt_type();
      audit_record.update_event_stage_state();
    }
    if (enable_sqlstat) {
      sqlstat_record.record_sqlstat_end_value(
          ::oceanbase::observer::get_observer_sql_engine()->get_query_runtime_environment());
      sqlstat_record.set_rows_processed(result.get_affected_rows() + result.get_return_rows());
      sqlstat_record.set_partition_cnt(result.get_exec_context().get_das_ctx().get_related_tablet_cnt());
      sqlstat_record.set_is_plan_cache_hit(ctx_.plan_cache_hit_);
      sqlstat_record.move_to_sqlstat_cache(result.get_session(),
                                                 get_observer_sql_engine()->get_plan_cache(),
                                                 get_observer_sql_engine()->get_plan_cache_access_service(),
                                                 ctx_.cur_sql_,
                                                 result.get_physical_plan());
    }

    audit_record.status_ = (0 == ret || OB_ITER_END == ret)
        ? 0 : (ret);
      //update v$sql statistics
    if (session.get_local_ob_enable_plan_cache()
        && !retry_ctrl_.need_retry()) {
      ObIArray<ObTableRowCount> *table_row_count_list = NULL;
      ObPhysicalPlanCtx *plan_ctx = result.get_exec_context().get_physical_plan_ctx();
      if (OB_ISNULL(plan_ctx)) {
        // do nothing
      } else {
        table_row_count_list = &(plan_ctx->get_table_row_count_list());
        audit_record.table_scan_stat_ = plan_ctx->get_table_scan_stat();
      }
      if (NULL != plan) {
        if (!(ctx_.self_add_plan_) && ctx_.plan_cache_hit_) {
          plan->update_plan_stat(audit_record,
                                 false, // false mean not first update plan stat
                                 table_row_count_list);
          plan->update_cache_access_stat(audit_record.table_scan_stat_);
        } else if (ctx_.self_add_plan_ && !ctx_.plan_cache_hit_) {
          plan->update_plan_stat(audit_record,
                                 true,
                                 table_row_count_list);
          plan->update_cache_access_stat(audit_record.table_scan_stat_);
        } else if (ctx_.self_add_plan_ && ctx_.plan_cache_hit_) {
          // First execution of a plan generated during this request.
          plan->update_plan_stat(audit_record,
                                 true,
                                 table_row_count_list);
          plan->update_cache_access_stat(audit_record.table_scan_stat_);
        }
      }
    }
    // reset thread waring buffer in sync mode
    if (!async_resp_used) {
      clear_wb_content(session);
    }

    need_disconnect = (result.get_exec_context().need_disconnect()
                       && !is_query_killed_return(ret));//Explicitly when it is a kill query, the connection should not be dropped
    if (need_disconnect) {
    }
    bool is_need_retry = THIS_THWORKER.need_retry() ||
        RETRY_TYPE_NONE != retry_ctrl_.get_retry_type();
  }
  return ret;
}

int ObMPQuery::store_params_value_to_str(ObIAllocator &allocator,
                                         sql::ObSQLSessionInfo &session,
                                         common::ParamStore &params)
{
  int ret = OB_SUCCESS;
  int64_t pos = 0;
  int64_t length = OB_MAX_SQL_LENGTH;
  CK (OB_NOT_NULL(params_value_ = static_cast<char *>(allocator.alloc(OB_MAX_SQL_LENGTH))));
  for (int64_t i = 0; OB_SUCC(ret) && i < params.count(); ++i) {
    const common::ObObjParam &param = params.at(i);
    if (param.is_ext()) {
      pos = 0;
      params_value_ = NULL;
      params_value_len_ = 0;
      break;
    } else {
      OZ (param.print_sql_literal(params_value_, length, pos, allocator, TZ_INFO(&session)));
      if (i != params.count() - 1) {
        OZ (databuff_printf(params_value_, length, pos, allocator, ","));
      }
    }
  }
  if (OB_FAIL(ret)) {
    params_value_ = NULL;
    params_value_len_ = 0;
    ret = OB_SUCCESS;
  } else {
    params_value_len_ = pos;
  }
  return ret;
}

int ObMPQuery::check_readonly_stmt(ObMySQLResultSet &result)
{
  int ret = OB_SUCCESS;
  bool is_readonly = false;
  //In this phase, the show statement will be converted to a select statement,
  //if literal_stmt_type is not stmt::T_NONE, then it indicates the original type show
  const stmt::StmtType type = stmt::T_NONE == result.get_literal_stmt_type() ?
                              result.get_stmt_type() :
                              result.get_literal_stmt_type();
  ObConsistencyLevel consistency = INVALID_CONSISTENCY;
  if (OB_FAIL(is_readonly_stmt(result, is_readonly))) {
  } else if (!is_readonly) {
    ret = OB_ERR_READ_ONLY;
  } else if (stmt::T_SELECT == type) {
    //For the select statement, strong consistency read needs to be disabled
    //Through setting /*+read_consistency()*/ hint
    //or specify session-level ob_read_consistency = 2
    // to set weak consistency read
    const int64_t table_count = DAS_CTX(result.get_exec_context()).get_table_loc_list().size();
    if (0 == table_count) {
      //This case is special, jdbc will send a special statement select @@session.tx_read_only when sending the query statement;
      //For convenience of obtest testing, the restriction on the select statement for no table_locations needs to be removed
    } else if (OB_FAIL(result.get_read_consistency(consistency))) {
    } else if (WEAK != consistency) {
      ret = OB_ERR_READ_ONLY;
    }
  }
  return ret;
}

int ObMPQuery::is_readonly_stmt(ObMySQLResultSet &result, bool &is_readonly)
{
  int ret = OB_SUCCESS;
  is_readonly = false;
  const stmt::StmtType type = stmt::T_NONE == result.get_literal_stmt_type() ?
                              result.get_stmt_type() :
                              result.get_literal_stmt_type();
  switch (type) {
    case stmt::T_SELECT: {
      //Forbid the select...for update statement as well
      ObPhysicalPlan *physical_plan = result.get_physical_plan();
      if (NULL == physical_plan) {
        ret = OB_ERR_UNEXPECTED;
      } else if (physical_plan->has_for_update()) {
        is_readonly = false;
      } else {
        is_readonly = true;
      }
      break;
    }
    case stmt::T_VARIABLE_SET: {
      //Prohibit set @@global.variable statement
      if (result.has_global_variable()) {
        is_readonly = false;
      } else {
        is_readonly = true;
      }
      break;
    }
    case stmt::T_EXPLAIN:
    case stmt::T_SHOW_TABLES:
    case stmt::T_SHOW_DATABASES:
    case stmt::T_SHOW_COLUMNS:
    case stmt::T_SHOW_VARIABLES:
    case stmt::T_SHOW_TABLE_STATUS:
    case stmt::T_SHOW_SCHEMA:
    case stmt::T_SHOW_CREATE_DATABASE:
    case stmt::T_SHOW_CREATE_TABLE:
    case stmt::T_SHOW_CREATE_VIEW:
    case stmt::T_SHOW_PARAMETERS:
    case stmt::T_SHOW_SERVER_STATUS:
    case stmt::T_SHOW_INDEXES:
    case stmt::T_SHOW_WARNINGS:
    case stmt::T_SHOW_ERRORS:
    case stmt::T_SHOW_PROCESSLIST:
    case stmt::T_SHOW_CHARSET:
    case stmt::T_SHOW_COLLATION:
    case stmt::T_SHOW_STATUS:
    case stmt::T_SHOW_TRACE:
    case stmt::T_SHOW_TRIGGERS:
    case stmt::T_SHOW_ENGINES:
    case stmt::T_SHOW_PROFILE:
    case stmt::T_SHOW_ENGINE:
    case stmt::T_SHOW_OPEN_TABLES:
    case stmt::T_SHOW_PRIVILEGES:
    case stmt::T_SHOW_GRANTS:
    case stmt::T_SHOW_RECYCLEBIN:
    case stmt::T_USE_DATABASE:
    case stmt::T_SET_NAMES: //read only not restrict it
    case stmt::T_START_TRANS:
    case stmt::T_END_TRANS:
    case stmt::T_SHOW_CHECK_TABLE:
    case stmt::T_SHOW_CREATE_USER: {
      is_readonly = true;
      break;
    }
    default: {
      is_readonly = false;
      break;
    }
  }
  return ret;
}
int ObMPQuery::deserialize()
{
  int ret = OB_SUCCESS;

  //OB_ASSERT(req_);
  //OB_ASSERT(req_->get_type() == ObRequest::OB_MYSQL);
  if ( (OB_ISNULL(req_)) || (req_->get_type() != ObRequest::OB_MYSQL)) {
    ret = OB_INVALID_ARGUMENT;
    LOG_ERROR("invalid request", K(ret), K(req_));
  } else if (get_is_com_filed_list()) {
    if (OB_FAIL(deserialize_com_field_list())) {
    }
  } else {
    const ObMySQLRawPacket &pkt = reinterpret_cast<const ObMySQLRawPacket&>(req_->get_packet());
    if (OB_UNLIKELY(ObMySQLCommandLayout::BYTES != pkt.get_command_layout())) {
      ret = OB_INVALID_DATA;
    } else if (OB_FAIL(pkt.get_command_field(0, sql_))) {
    }
  }

  return ret;
}

// return false only if send packet fail.
OB_INLINE int ObMPQuery::response_result(ObMySQLResultSet &result,
                                         bool force_sync_resp,
                                         bool &async_resp_used)
{
  int ret = OB_SUCCESS;
  bool callback_armed = false;
  // When ac = 1, starting a new transaction in the thread to clean up temporary
  // table data can deadlock with the clog callback, so use a synchronous method.
  ObSQLSessionInfo &session = result.get_session();
  sql::ObTraceSpanGuard exec_span(&session, sql::TRACE_SQL_EXECUTE);
  bool need_trans_cb  = result.need_end_trans_callback() && (!force_sync_resp);
  // Determine whether it is plan or cmd by checking if plan is null
  // Handle plan and cmd separately, the logic will be clearer.
  if (OB_LIKELY(NULL != result.get_physical_plan())) {
    if (need_trans_cb) {
      ObAsyncPlanDriver drv(gctx_, ctx_, session, retry_ctrl_, packet_sender_);
      // NOTE: sql_end_cb must be initialized before drv.response_result()
      ObSqlEndTransCb &sql_end_cb = session.get_mysql_end_trans_cb();
      if (OB_FAIL(sql_end_cb.init(packet_sender_, &session))) {
      } else if (FALSE_IT(callback_armed = true)) {
      } else if (OB_FAIL(drv.response_result(result))) {
      }
      async_resp_used = result.is_async_end_trans_submitted();
    } else {
      // Pilot ObQuerySyncDriver
      ObSyncPlanDriver drv(gctx_, ctx_, session, retry_ctrl_, packet_sender_);
      ret = drv.response_result(result);
    }
  } else {

#define CMD_EXEC                                                               \
  if (need_trans_cb) {                                                         \
    ObSqlEndTransCb &sql_end_cb = session.get_mysql_end_trans_cb();            \
    ObAsyncCmdDriver drv(gctx_, ctx_, session, retry_ctrl_, packet_sender_);   \
    if (OB_FAIL(sql_end_cb.init(packet_sender_, &session))) {                  \
      LOG_WARN("failed to init sql end callback", K(ret));                     \
    } else if (FALSE_IT(callback_armed = true)) {                              \
    } else if (OB_FAIL(drv.response_result(result))) {                         \
      LOG_WARN("fail response async result", K(ret));                          \
    }                                                                          \
    async_resp_used = result.is_async_end_trans_submitted();                   \
  } else {                                                                     \
    ObSyncCmdDriver drv(gctx_, ctx_, session, retry_ctrl_, packet_sender_);    \
    session.set_pl_query_sender(&drv);                                         \
    session.set_ps_protocol(result.is_ps_protocol());                          \
    ret = drv.response_result(result);                                         \
    session.set_pl_query_sender(NULL);                                         \
  }

    if (result.is_pl_stmt(result.get_stmt_type())) {
      CMD_EXEC;
    } else {
      CMD_EXEC;
    }

#undef CMD_EXEC

  }

  if (callback_armed && !async_resp_used) {
    const int cancel_ret = cancel_unsubmitted_callback(session.get_mysql_end_trans_cb());
    if (OB_SUCCESS != cancel_ret) {
      LOG_ERROR("failed to cancel unsubmitted mysql end-trans callback", K(cancel_ret));
      ret = OB_SUCCESS == ret ? cancel_ret : ret;
    }
  }

  return ret;
}

int ObMPQuery::deserialize_com_field_list()
{
  int ret = OB_SUCCESS;
  //If only column definitions are to be returned, it means the client sent a COM_FIELD_LIST command
  /* mysql's COM_FIELD_LIST command is used to get the column definitions of a table, its packet from client to server is:
  *  1              [04] COM_FIELD_LIST
  *  string[NUL]    table
  *  string[EOF]    field wildcard
  *  First is the CMD type, then the table name, followed by the match condition
  *
  * The packet from server to client is one of the following:
  * 1. an ERR_Packet (returns an error packet)
  * 2. one or more Column Definition packets and a closing EOF_Packet (returns n column definitions + EOF)
  *
  * Since the result of a regular SELECT query already includes column definitions, and to maximize the reuse of the current code logic, the COM_FIELD_LIST command
  * can be considered equivalent to:
  * select * from table limit 0 ==> get field define ==> return Column Definition based on field wildcard as needed
  *
  * Reference: https://dev.mysql.com/doc/internals/en/com-field-list.html
   */
  ObIAllocator *alloc = &THIS_WORKER.get_sql_arena_allocator();
  {
    OB_ASSERT(alloc != nullptr);
    const ObMySQLRawPacket &pkt = reinterpret_cast<const ObMySQLRawPacket&>(req_->get_packet());
    ObString table_name;
    ObString wildcard;
    const char *str1 = "select * from ";
    const char *str2 = " where 0";
    const int64_t str1_len = strlen(str1);
    const int64_t str2_len = strlen(str2);
    if (OB_UNLIKELY(ObMySQLCommandLayout::FIELD_LIST !=
                    pkt.get_command_layout())) {
      ret = OB_INVALID_DATA;
    } else if (OB_FAIL(pkt.get_command_field(0, table_name))) {
    } else if (OB_FAIL(pkt.get_command_field(1, wildcard))) {
    }
    const int64_t sql_len = str1_len + table_name.length() + str2_len;
    char *dest_str =
        OB_SUCC(ret) ? static_cast<char *>(alloc->alloc(sql_len + 1)) : NULL;
    if (OB_SUCC(ret) && OB_ISNULL(dest_str)) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
      LOG_WARN("failed to alloc", K(dest_str));
    } else if (OB_SUCC(ret)) {
      char *buf = dest_str;
      MEMCPY(buf, str1, str1_len);
      buf = buf + str1_len;
      MEMCPY(buf, table_name.ptr(), table_name.length());
      buf = buf + table_name.length();
      MEMCPY(buf, str2, str2_len);
      dest_str[sql_len] = '\0';
      sql_.assign_ptr(dest_str, sql_len);
      wild_str_ = wildcard;
    }
  }
  return ret;
}
