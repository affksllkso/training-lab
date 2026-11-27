"""Run the local TRAINING LAB dashboard with Python 3.11 or later."""

import argparse
import hashlib
import json
import math
import re
import secrets
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from generate import request_document
from lab import document_ids, evaluate, export_ai_jsonl, import_jsonl, model_file, summarize

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD = 20 * 1024 * 1024


def rule_baseline(analyzer):
    runner = """
        import {readFileSync} from 'node:fs';
        import {pathToFileURL} from 'node:url';
        const {analyze} = await import(pathToFileURL(process.argv[1]).href);
        const texts = JSON.parse(readFileSync(0, 'utf8'));
        console.log(JSON.stringify(texts.map(text => analyze(text).score)));
    """

    def score(rows):
        try:
            result = subprocess.run(
                ["node", "--input-type=module", "--eval", runner, str(analyzer)],
                input=json.dumps([row["text"] for row in rows], ensure_ascii=False),
                encoding="utf-8", capture_output=True, check=True, timeout=120,
            )
            scores = json.loads(result.stdout)
            if len(scores) != len(rows) or any(type(value) not in (float, int)
                                               or not math.isfinite(value)
                                               or not 0 <= value <= 100 for value in scores):
                raise ValueError("Invalid baseline score")
        except (OSError, subprocess.SubprocessError, ValueError, TypeError) as error:
            raise ValueError("比較には Node.js 22.13以上と、analyze(text).score を返す解析モジュールが必要です。") from error
        return {row["id"]: value / 100 for row, value in zip(rows, scores, strict=True)}

    return score


def serve(path, port, analyzer=None):
    token = secrets.token_urlsafe(32)
    training = threading.Lock()

    class Dashboard(BaseHTTPRequestHandler):
        def allowed(self):
            return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

        def respond(self, status, data, mime="application/json; charset=utf-8", filename=None):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                             "script-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if not self.allowed():
                self.respond(403, {"error": "ローカルURLを使用してください。"})
                return
            route = unquote(urlparse(self.path).path)
            if route == "/":
                page = (ROOT / "index.html").read_text(encoding="utf-8").replace("__LAB_TOKEN__", token, 1)
                self.respond(200, page.encode("utf-8"), "text/html; charset=utf-8")
            elif route == "/generation.mjs":
                self.respond(200, (ROOT / "generation.mjs").read_bytes(), "text/javascript; charset=utf-8")
            elif route == "/api/status":
                self.respond(200, summarize(path, analyzer is not None))
            elif route == "/api/document-ids":
                self.respond(200, document_ids(path))
            elif route == "/api/ai-data":
                self.respond(200, {"jsonl": export_ai_jsonl(path)})
            elif route.startswith("/models/"):
                version = route.removeprefix("/models/").removesuffix(".json")
                model = model_file(path, version)
                if model is None:
                    self.respond(404, {"error": "モデルが見つかりません。"})
                    return
                self.respond(200, model.encode("utf-8"), filename=f"{version}.json")
            else:
                self.respond(404, {"error": "ページが見つかりません。"})

        def do_POST(self):
            if not self.allowed() or not secrets.compare_digest(self.headers.get("X-Lab-Token", ""), token):
                self.respond(403, {"error": "ページを再読み込みしてください。"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= MAX_UPLOAD:
                    raise ValueError("ファイルの上限は20 MBです。空のデータは取り込めません。")
                route = urlparse(self.path).path
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("ファイルの受信が途中で終了しました。")
                if route == "/api/import":
                    counts, errors = import_jsonl(path, raw)
                    self.respond(200, {"counts": counts, "errors": errors})
                elif route == "/api/generate":
                    settings = json.loads(raw)
                    if not isinstance(settings, dict):
                        raise ValueError("生成設定が無効です。")
                    engine, model, prompt = (settings.get(key) for key in ("engine", "model", "prompt"))
                    if (engine not in ("gemini", "ollama") or not isinstance(model, str)
                            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,159}", model)
                            or not isinstance(prompt, str) or not 1 <= len(prompt) <= 2000):
                        raise ValueError("モデルIDと題材を確認してください。")
                    self.respond(200, {"text": request_document(engine, model, prompt)})
                elif route == "/api/train":
                    settings = json.loads(raw)
                    if not isinstance(settings, dict):
                        raise ValueError("学習設定が無効です。")
                    values = [settings.get(key, "") for key in ("hold_model", "hold_genre")]
                    if any(not isinstance(value, str) or len(value) > 500 for value in values):
                        raise ValueError("モデル名とジャンル名は500字以内で入力してください。")
                    if not training.acquire(blocking=False):
                        self.respond(409, {"error": "別の学習が実行中です。"})
                        return
                    try:
                        baseline = rule_baseline(analyzer) if analyzer and settings.get("compare_rules") is True else None
                        baseline_id = hashlib.sha256(analyzer.read_bytes()).hexdigest() if baseline else None
                        self.respond(200, evaluate(path, *values, baseline=baseline, baseline_id=baseline_id))
                    finally:
                        training.release()
                else:
                    self.respond(404, {"error": "操作が見つかりません。"})
            except UnicodeDecodeError:
                self.respond(400, {"error": "UTF-8のJSONLファイルを使用してください。"})
            except ValueError as error:
                self.respond(400, {"error": str(error)})

    with ThreadingHTTPServer(("127.0.0.1", port), Dashboard) as httpd:
        print(f"TRAINING LAB  http://127.0.0.1:{httpd.server_port}", flush=True)
        print(f"データベース  {path}", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "training.sqlite3")
    parser.add_argument("--port", type=int, default=4181)
    parser.add_argument("--baseline", type=Path, help="scoreを返す既存の解析モジュール")
    args = parser.parse_args()
    if args.baseline and not args.baseline.is_file():
        parser.error("--baseline のファイルが見つかりません。")
    args.db.parent.mkdir(parents=True, exist_ok=True)
    serve(args.db.resolve(), args.port, args.baseline.resolve() if args.baseline else None)
