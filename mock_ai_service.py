#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal Mock AI Service for seekdb AI function tests.

Serves OpenAI-compatible (/v1/chat/completions, /v1/embeddings, /v1/rerank)
and Ollama-compatible (/api/generate, /api/embeddings) endpoints on 0.0.0.0:8000.
Request type is detected from path first, then from the JSON body, so an
endpoint whose url is the bare service root also works.
"""
import json
import hashlib
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_DIM = int(os.environ.get("MOCK_EMBED_DIM", "4"))
LOG = os.environ.get("MOCK_LOG", "1") == "1"


def vec_for(text, dim):
    h = hashlib.sha256(text.encode("utf-8")).digest()
    out = []
    for i in range(dim):
        b = h[i % len(h)]
        out.append(round((b / 255.0) * 2.0 - 1.0, 6))
    return out


def detect_kind(path, body):
    p = (path or "").lower()
    if "/api/generate" in p:
        return "ollama_generate"
    if "/api/embeddings" in p or "/api/embed" in p:
        return "ollama_embed"
    if "rerank" in p:
        return "rerank"
    if "embed" in p:
        return "embed"
    if "chat" in p or "completion" in p:
        return "chat"
    if isinstance(body, dict):
        if "messages" in body:
            return "chat"
        if "documents" in body:
            return "rerank"
        if "input" in body:
            return "embed"
        if "prompt" in body:
            return "generate"
        if "texts" in body:
            return "rerank"
        if "query" in body:
            return "rerank"
    return "chat"


def texts_of(body):
    """Normalise the various input shapes to a list of strings."""
    def to_str(v):
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            for k in ("text", "content", "document", "input", "value"):
                if k in v and isinstance(v[k], str):
                    return v[k]
            return json.dumps(v, ensure_ascii=False, sort_keys=True)
        if isinstance(v, (int, float)):
            return str(v)
        return json.dumps(v, ensure_ascii=False, sort_keys=True)

    if not isinstance(body, dict):
        return [to_str(body)]
    for key in ("input", "documents", "texts", "prompt", "query"):
        v = body.get(key)
        if isinstance(v, list):
            return [to_str(x) for x in v]
        if isinstance(v, str):
            return [v]
    for m in body.get("messages", []) or []:
        if isinstance(m, dict) and "content" in m:
            return [to_str(m["content"])]
    return [json.dumps(body, ensure_ascii=False, sort_keys=True)]


def dim_of(body):
    if isinstance(body, dict):
        for k in ("dimensions", "dim", "dimension"):
            v = body.get(k)
            if isinstance(v, int) and v > 0:
                return v
            if isinstance(v, str) and v.isdigit() and int(v) > 0:
                return int(v)
        opts = body.get("options")
        if isinstance(opts, dict) and isinstance(opts.get("dimensions"), int):
            if opts["dimensions"] > 0:
                return opts["dimensions"]
    return DEFAULT_DIM


def handle(kind, body):
    texts = texts_of(body)
    dim = dim_of(body)
    model = body.get("model", "mock-model") if isinstance(body, dict) else "mock-model"

    if kind in ("embed", "ollama_embed"):
        vecs = [vec_for(t, dim) for t in texts]
        if kind == "ollama_embed":
            if len(vecs) == 1:
                return {"embedding": vecs[0]}
            return {"embeddings": vecs}
        return {
            "object": "list",
            "model": model,
            "data": [
                {"object": "embedding", "index": i, "embedding": v}
                for i, v in enumerate(vecs)
            ],
            "usage": {"prompt_tokens": sum(len(t) for t in texts), "total_tokens": sum(len(t) for t in texts)},
        }

    if kind == "rerank":
        # score by text length so ordering is deterministic but non-trivial
        scores = []
        for i, t in enumerate(texts):
            scores.append({"index": i, "relevance_score": round(min(1.0, 0.10 + len(t) / 200.0), 6),
                           "document": t})
        results = sorted(scores, key=lambda x: -x["relevance_score"])
        if isinstance(body, dict) and body.get("raw_scores") is True:
            return {"results": results}
        return {"results": results, "id": "mock-rerank-1", "model": model,
                "usage": {"total_tokens": sum(len(t) for t in texts)}}

    if kind == "ollama_generate":
        return {"model": model, "created_at": "2026-01-01T00:00:00Z",
                "response": "mock completion", "done": True,
                "prompt_eval_count": 1, "eval_count": 1}

    # chat completion
    return {
        "id": "mock-chat-1",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "mock completion"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            return json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            return {}

    def _send(self, code, obj):
        payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path.rstrip("/") in ("", "/health", "/api/tags", "/api/version"):
            if "/api/tags" in self.path:
                self._send(200, {"models": [{"name": "mock-model"}]})
            elif "/api/version" in self.path:
                self._send(200, {"version": "0.1.0"})
            else:
                self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        body = self._read_body()
        kind = detect_kind(self.path, body)
        if LOG:
            sys.stderr.write("[mock] %s %s -> %s\n" % (self.path, json.dumps(body, ensure_ascii=False)[:200], kind))
            sys.stderr.flush()
        try:
            self._send(200, handle(kind, body))
        except Exception as e:  # pragma: no cover
            self._send(500, {"error": str(e)})

    def log_message(self, fmt, *args):
        pass


def main():
    port = int(os.environ.get("MOCK_PORT", "8000"))
    host = os.environ.get("MOCK_HOST", "0.0.0.0")
    srv = ThreadingHTTPServer((host, port), Handler)
    sys.stderr.write("mock ai service listening on %s:%d\n" % (host, port))
    sys.stderr.flush()
    srv.serve_forever()


if __name__ == "__main__":
    main()
