//! Wire-protocol adapters.
//!
//! The official driver has not been published. The evidence hierarchy
//! (strongest first):
//!
//! 1. `mysqltest/cases/*.result` in the official round2 branch: a JSON array
//!    with exactly one entry per SQL statement - `{}` for plain success,
//!    `{"affected_rows":n}` for DML, `{"columns":[..],"rows":[[..]]}` for
//!    queries, `{"error":"name"}` for errors. Driver directives occupy no
//!    assertion slot, so they must produce no response. The README states
//!    the old example-style JSON *wrapping* is gone; these bare shapes are
//!    exactly what a positionally comparing driver would read line by line.
//!    => default `jsonl`.
//! 2. Another team's platform evidence: a TSV + `OK <n>` framing with no
//!    stdout ready marker ran to completion while one-line-wrapped JSON
//!    failed. => `tsv` kept as the fallback hypothesis with that framing.
//!
//! All transports share: one response per statement, flushed before reading
//! on; diagnostics on stderr only; BIGINT printed as exact integer tokens.
//! Select with `MINIVEC_PROTOCOL=jsonl|json|json-noready|miniob|tsv`.

use crate::engine::{ExecResult, OutValue};
use crate::error::DbError;
use crate::value::{fmt_f32, fmt_f64};

pub enum ProtoKind {
    /// Bare one-line JSON mirroring the official `.result` vocabulary.
    Jsonl,
    /// Earlier interim transport: `{"ready":true}` then ok-wrapped JSON.
    Json,
    /// Same responses as `Json` but no startup line.
    JsonNoReady,
    /// miniob-style plain text with SUCCESS/FAILURE lines.
    Miniob,
    /// Consensus TSV framing: header+rows then `OK <n>`; `ERROR <name>: <msg>`.
    Tsv,
}

impl ProtoKind {
    pub fn from_env() -> ProtoKind {
        match std::env::var("MINIVEC_PROTOCOL").as_deref() {
            Ok("json") => ProtoKind::Json,
            Ok("json-noready") => ProtoKind::JsonNoReady,
            Ok("miniob") => ProtoKind::Miniob,
            Ok("tsv") => ProtoKind::Tsv,
            _ => ProtoKind::Jsonl,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            ProtoKind::Jsonl => "jsonl",
            ProtoKind::Json | ProtoKind::JsonNoReady => "json",
            ProtoKind::Miniob => "miniob",
            ProtoKind::Tsv => "tsv",
        }
    }
}

pub struct Protocol {
    pub kind: ProtoKind,
}

impl Protocol {
    pub fn new(kind: ProtoKind) -> Protocol {
        Protocol { kind }
    }

    /// Line printed once after recovery finished, or None when the driver
    /// does not expect a ready marker.
    pub fn ready_line(&self) -> Option<String> {
        match self.kind {
            // The official assertion data is positionally aligned per SQL
            // statement; any extra line would desync every comparison.
            ProtoKind::Jsonl => None,
            ProtoKind::Json => Some("{\"ready\":true}".to_string()),
            ProtoKind::JsonNoReady | ProtoKind::Miniob => None,
            ProtoKind::Tsv => Some("READY".to_string()),
        }
    }

    pub fn render(&self, result: &Result<ExecResult, DbError>) -> String {
        match self.kind {
            ProtoKind::Jsonl => render_jsonl(result),
            ProtoKind::Json | ProtoKind::JsonNoReady => render_json(result),
            ProtoKind::Miniob => render_miniob(result),
            ProtoKind::Tsv => render_tsv(result),
        }
    }
}

// ---------------------------------------------------------------------------
// Shared JSON cell rendering
// ---------------------------------------------------------------------------

pub fn json_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if c < '\u{20}' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn render_json_value(value: &OutValue) -> String {
    match value {
        OutValue::Int(v) => v.to_string(),
        OutValue::Str(v) => json_string(v),
        OutValue::Float(v) => fmt_f64(*v),
        OutValue::Vector(v) => format!(
            "[{}]",
            v.iter().map(|f| fmt_f32(*f)).collect::<Vec<_>>().join(",")
        ),
    }
}

// ---------------------------------------------------------------------------
// jsonl: bare objects mirroring mysqltest .result entries
// ---------------------------------------------------------------------------

fn render_jsonl(result: &Result<ExecResult, DbError>) -> String {
    match result {
        Err(error) => format!("{{\"error\":{}}}", json_string(error.code())),
        Ok(ExecResult::Ok { affected_rows }) => match affected_rows {
            Some(n) => format!("{{\"affected_rows\":{}}}", n),
            None => "{}".to_string(),
        },
        Ok(ExecResult::Query { columns, rows }) => {
            let columns = columns.iter().map(|name| json_string(name)).collect::<Vec<_>>().join(",");
            let rows = rows
                .iter()
                .map(|row| {
                    format!(
                        "[{}]",
                        row.iter().map(render_json_value).collect::<Vec<_>>().join(",")
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{\"columns\":[{}],\"rows\":[{}]}}", columns, rows)
        }
    }
}

// ---------------------------------------------------------------------------
// JSON transport with ok-wrapper (interim; kept for compatibility)
// ---------------------------------------------------------------------------

fn render_json(result: &Result<ExecResult, DbError>) -> String {
    match result {
        Err(error) => format!(
            "{{\"ok\":false,\"error\":{},\"code\":{}}}",
            json_string(error.code()),
            json_string(error.code())
        ),
        Ok(ExecResult::Ok { affected_rows }) => match affected_rows {
            Some(n) => format!("{{\"ok\":true,\"affected_rows\":{}}}", n),
            None => "{\"ok\":true}".to_string(),
        },
        Ok(ExecResult::Query { columns, rows }) => {
            let columns = columns.iter().map(|name| json_string(name)).collect::<Vec<_>>().join(",");
            let rows = rows
                .iter()
                .map(|row| {
                    format!(
                        "[{}]",
                        row.iter().map(render_json_value).collect::<Vec<_>>().join(",")
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{\"ok\":true,\"columns\":[{}],\"rows\":[{}]}}", columns, rows)
        }
    }
}

// ---------------------------------------------------------------------------
// miniob-style transport (candidate; exact tokens pending the official spec)
// ---------------------------------------------------------------------------

fn render_miniob_value(value: &OutValue) -> String {
    match value {
        OutValue::Int(v) => v.to_string(),
        OutValue::Str(v) => v.clone(),
        OutValue::Float(v) => fmt_f64(*v),
        OutValue::Vector(v) => format!(
            "[{}]",
            v.iter().map(|f| fmt_f32(*f)).collect::<Vec<_>>().join(",")
        ),
    }
}

fn render_miniob(result: &Result<ExecResult, DbError>) -> String {
    match result {
        Err(error) => format!("FAILURE\n{}", error.code()),
        Ok(ExecResult::Ok { affected_rows }) => match affected_rows {
            Some(n) => format!("SUCCESS\naffected rows: {}", n),
            None => "SUCCESS".to_string(),
        },
        Ok(ExecResult::Query { columns, rows }) => {
            let mut out = Vec::with_capacity(rows.len() + 2);
            if !columns.is_empty() {
                out.push(columns.join(" | "));
            }
            for row in rows {
                out.push(row.iter().map(render_miniob_value).collect::<Vec<_>>().join(" | "));
            }
            out.push("SUCCESS".to_string());
            out.join("\n")
        }
    }
}

// ---------------------------------------------------------------------------
// TSV consensus transport (fallback hypothesis)
// ---------------------------------------------------------------------------

/// Escape a cell so one TSV row stays one physical line.
fn tsv_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c => out.push(c),
        }
    }
    out
}

fn render_tsv_value(value: &OutValue) -> String {
    match value {
        OutValue::Int(v) => v.to_string(),
        OutValue::Str(v) => tsv_escape(v),
        OutValue::Float(v) => fmt_f64(*v),
        OutValue::Vector(v) => format!(
            "[{}]",
            v.iter().map(|f| fmt_f32(*f)).collect::<Vec<_>>().join(",")
        ),
    }
}

fn render_tsv(result: &Result<ExecResult, DbError>) -> String {
    match result {
        Err(error) => format!("ERROR {}: {}", error.code(), error.code()),
        Ok(ExecResult::Ok { affected_rows }) => match affected_rows {
            Some(n) => format!("OK {}", n),
            None => "OK 0".to_string(),
        },
        Ok(ExecResult::Query { columns, rows }) => {
            let mut out = Vec::with_capacity(rows.len() + 2);
            out.push(columns.iter().map(|c| tsv_escape(c)).collect::<Vec<_>>().join("\t"));
            for row in rows {
                out.push(row.iter().map(render_tsv_value).collect::<Vec<_>>().join("\t"));
            }
            out.push(format!("OK {}", rows.len()));
            out.join("\n")
        }
    }
}
