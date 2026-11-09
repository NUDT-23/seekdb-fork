mod engine;
mod error;
mod lexer;
mod parser;
mod storage;
mod value;

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use engine::{Engine, ExecResult, OutValue};
use error::DbError;
use storage::Store;
use value::{fmt_f32, fmt_f64};

fn json_string(text: &str) -> String {
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

fn render_value(value: &OutValue) -> String {
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

fn render_result(result: Result<ExecResult, DbError>) -> String {
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
            let columns = columns
                .iter()
                .map(|name| json_string(name))
                .collect::<Vec<_>>()
                .join(",");
            let rows = rows
                .iter()
                .map(|row| {
                    format!(
                        "[{}]",
                        row.iter().map(render_value).collect::<Vec<_>>().join(",")
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{\"ok\":true,\"columns\":[{}],\"rows\":[{}]}}", columns, rows)
        }
    }
}

fn data_dir() -> PathBuf {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--data-dir" {
            if let Some(path) = args.next() {
                return PathBuf::from(path);
            }
        } else if !arg.starts_with('-') {
            return PathBuf::from(arg);
        }
    }
    std::env::var_os("MINIVEC_DATA_DIR").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/data"))
}

fn main() {
    let dir = data_dir();
    let arena_mb = std::env::var("MINIVEC_VECTOR_ARENA_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(512);
    let store = match Store::open(&dir, arena_mb * 1024 * 1024) {
        Ok(store) => store,
        Err(error) => {
            eprintln!("minivectordb: failed to open {}: {}", dir.display(), error);
            std::process::exit(1);
        }
    };
    let mut engine = Engine::new(store);
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let _ = writeln!(out, "{{\"ready\":true}}");
    let _ = out.flush();

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(error) => {
                eprintln!("minivectordb: stdin error: {}", error);
                break;
            }
        };
        let sql = line.trim();
        if sql.is_empty() {
            continue;
        }
        if sql.eq_ignore_ascii_case("--mode ann") {
            engine.set_ann_mode(true);
            let _ = writeln!(out, "{{\"ok\":true}}");
            let _ = out.flush();
            continue;
        }
        if sql.eq_ignore_ascii_case("--mode exact") {
            engine.set_ann_mode(false);
            let _ = writeln!(out, "{{\"ok\":true}}");
            let _ = out.flush();
            continue;
        }
        if sql.eq_ignore_ascii_case("--shutdown") || sql.eq_ignore_ascii_case("SHUTDOWN") {
            let _ = writeln!(out, "{{\"ok\":true}}");
            let _ = out.flush();
            break;
        }
        let response = render_result(engine.execute(sql));
        if writeln!(out, "{}", response).and_then(|_| out.flush()).is_err() {
            break;
        }
    }
    if let Err(error) = engine.store.shutdown() {
        eprintln!("minivectordb: shutdown checkpoint failed: {}", error);
    }
    let _ = out.flush();
}
