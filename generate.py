"""Generate Japanese training texts from a local JSONL prompt plan."""

import argparse
import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from lab import database, get_field, import_jsonl, parse_record

ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS = {
    "gemini": "gemini-4-hydrogen",
    "ollama": "gemma4:e2b",
}
MODEL_ALIASES = {
    "gemini4hydrogen": "gemini-4-hydrogen",
    "gemini4neon": "gemini-4-neon",
    "gemini3.1flashlite": "gemini-3.1-flash-lite",
}


def normalize_model_id(engine, model):
    normalized = unicodedata.normalize("NFKC", model).strip()
    if engine == "gemini":
        compact = re.sub(r"[\s_-]+", "", normalized.lower())
        return MODEL_ALIASES.get(compact, normalized)
    return normalized


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request_document(engine, model, prompt):
    model = normalize_model_id(engine, model)
    instruction = "次の依頼に沿って、800〜1200字の日本語本文だけを書いてください。架空の出典は書かないでください。\n\n" + prompt
    headers = {"Content-Type": "application/json"}
    if engine == "gemini":
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", model):
            raise ValueError("Gemini のモデルIDを確認してください。")
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY を環境変数に設定してください。")
        headers["x-goog-api-key"] = key
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        body = {"contents": [{"parts": [{"text": instruction}]}]}
    else:
        url = "http://127.0.0.1:11434/api/generate"
        body = {"model": model, "prompt": instruction, "stream": False}

    request = Request(url, json.dumps(body, ensure_ascii=False).encode("utf-8"), headers)
    try:
        with build_opener(NoRedirect()).open(request, timeout=180) as response:
            result = json.loads(response.read(1_000_000))
    except HTTPError as error:
        if error.code == 429:
            raise ValueError("利用上限またはレート制限に達したため停止しました。") from error
        raise ValueError(f"生成サービスが HTTP {error.code} を返したため停止しました。") from error
    except (URLError, TimeoutError) as error:
        raise ValueError("生成サービスに接続できなかったため停止しました。") from error
    try:
        if engine == "ollama":
            return result["response"].strip()
        parts = result["candidates"][0]["content"]["parts"]
        return "".join(part.get("text", "") for part in parts if not part.get("thought")).strip()
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise ValueError("生成サービスから本文を取得できませんでした。") from error


def generate(plan, path, engine, model, limit, provider=request_document, notify=print):
    source = {"gemini": "Gemini", "ollama": "Ollama"}[engine]
    model = normalize_model_id(engine, model)
    with database(path) as db:
        known = {row["id"] for row in db.execute("SELECT id FROM documents")}
    completed = 0
    for number, line in enumerate(plan.read_text(encoding="utf-8-sig").split("\n"), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError("JSONオブジェクトを指定してください。")
            identifier = get_field(item, "id")
            if identifier in known:
                continue
            prompt = item.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 20_000:
                raise ValueError("prompt を1〜20,000字で指定してください。")
            record = {**item, "id": identifier, "label": 1, "source": source,
                      "model": model, "text": "仮" * 100}
            parse_record(record)
            document = provider(engine, model, prompt)
            if not 800 <= len(document) <= 1200:
                raise ValueError(f"生成結果が{len(document)}字でした。800〜1200字で手動調整するか、依頼文を直してください。")
            record.update(text=document, generated_at=datetime.now(timezone.utc).isoformat())
            raw = json.dumps(record, ensure_ascii=False).encode("utf-8")
            counts, errors = import_jsonl(path, raw)
            if not counts["added"]:
                raise ValueError(errors[0] if errors else "既存の本文と重複しました。")
        except ValueError as error:
            raise ValueError(f"{number}行目で停止しました。{error}") from error
        notify(f"保存: {identifier} ({len(document)}字)")
        known.add(identifier)
        completed += 1
        if completed >= limit:
            break
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--provider", required=True, choices=("gemini", "ollama"))
    parser.add_argument("--model", help="モデルID（省略時は gemini-4-hydrogen または gemma4:e2b）")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "training.sqlite3")
    parser.add_argument("--limit", type=int, default=10, help="今回の生成件数の上限")
    args = parser.parse_args()
    if not 1 <= args.limit <= 100:
        parser.error("--limit は1〜100で指定してください。")
    selected_model = args.model or DEFAULT_MODELS[args.provider]
    args.db.parent.mkdir(parents=True, exist_ok=True)
    try:
        count = generate(args.plan, args.db, args.provider, selected_model, args.limit)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"保存件数: {count}")


if __name__ == "__main__":
    main()
