mod engine;
mod error;
mod ivf;
mod lexer;
mod parser;
mod protocol;
mod storage;
mod value;

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use engine::Engine;
use protocol::{ProtoKind, Protocol};
use storage::Store;

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
    std::env::var_os("MINIVEC_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/data"))
}

/// Split one input line into complete SQL statements on quote-unaware-safe
/// semicolons: single-quoted strings (with '' escapes) are never split.
fn split_statements(line: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let bytes = line.as_bytes();
    let mut start = 0usize;
    let mut in_string = false;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'\'' => {
                if in_string && i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                    i += 2; // escaped quote inside the literal
                    continue;
                }
                in_string = !in_string;
            }
            b';' if !in_string => {
                parts.push(&line[start..i]);
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    parts.push(&line[start..]);
    parts.into_iter().map(str::trim).filter(|p| !p.is_empty()).collect()
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
    let proto = Protocol::new(ProtoKind::from_env());
    eprintln!(
        "minivectordb: data={} protocol={}",
        dir.display(),
        proto.kind.name()
    );

    let stdout = io::stdout();
    let mut out = stdout.lock();
    if let Some(ready) = proto.ready_line() {
        let _ = writeln!(out, "{}", ready);
        let _ = out.flush();
    }

    let stdin = io::stdin();
    let mut first_line = true;
    for line in stdin.lock().lines() {
        let mut line = match line {
            Ok(line) => line,
            Err(error) => {
                eprintln!("minivectordb: stdin error: {}", error);
                break;
            }
        };
        if first_line {
            first_line = false;
            if line.starts_with('\u{feff}') {
                line.remove(0);
            }
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Driver directives and exit spellings occupy no assertion slot:
        // they are consumed silently (official README: "驱动指令不占断言位置").
        let directive = trimmed.to_ascii_lowercase();
        let directive = directive.trim_end_matches(';').trim();
        match directive {
            "--mode ann" => {
                engine.set_ann_mode(true);
                continue;
            }
            "--mode exact" => {
                engine.set_ann_mode(false);
                continue;
            }
            // The driver closes and restarts us with the same data dir; our
            // side of the contract is a clean shutdown (rollback uncommitted
            // transactions, durable checkpoint) and exit 0.
            "--restart" | "--shutdown" | "shutdown" | "exit" | "quit" | "bye" | "\\q" => {
                break;
            }
            _ => {}
        }
        // SQL line comments: swallow silently.
        if trimmed.starts_with("--") || trimmed.starts_with('#') {
            continue;
        }
        for sql in split_statements(trimmed) {
            let response = proto.render(&engine.execute(sql));
            if writeln!(out, "{}", response).and_then(|_| out.flush()).is_err() {
                break;
            }
        }
    }
    let _ = engine.store.shutdown();
    let _ = out.flush();
}
