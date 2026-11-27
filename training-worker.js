"use strict";

const runtimeUrl = "https://cdn.jsdelivr.net/pyodide/v0.29.3/full/";
const sqlitePath = "/training.sqlite3";
const importPath = "/import.jsonl";
const storeName = "workspace";
const limit = 20 * 1024 * 1024;
let runtime;
let storage;
let queue = Promise.resolve();

function openStorage() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("training-lab-sqlite-v1", 1);
    request.onupgradeneeded = () => request.result.createObjectStore(storeName);
    request.onsuccess = () => resolve(request.result);
    request.onerror = request.onblocked = () => reject(new Error("ブラウザーの保存領域を開けませんでした。"));
  });
}

function readStorage() {
  return new Promise((resolve, reject) => {
    const request = storage.transaction(storeName).objectStore(storeName).get("database");
    request.onsuccess = () => resolve(request.result ?? {revision: 0});
    request.onerror = () => reject(new Error("学習データを読み込めませんでした。"));
  });
}

function saveStorage(previous, bytes) {
  return new Promise((resolve, reject) => {
    const transaction = storage.transaction(storeName, "readwrite");
    const store = transaction.objectStore(storeName);
    const request = store.get("database");
    const failure = new Error("ブラウザーに保存できませんでした。空き容量や設定を確認してください。");
    transaction.oncomplete = () => resolve();
    transaction.onabort = () => reject(failure);
    request.onsuccess = () => {
      if ((request.result?.revision ?? 0) !== previous) {
        failure.message = "別のタブで学習データが更新されました。もう一度操作してください。";
        transaction.abort();
        return;
      }
      store.put({revision: previous + 1, bytes}, "database");
    };
  });
}

async function initialize() {
  try {
    importScripts(runtimeUrl + "pyodide.js");
    const python = await loadPyodide({indexURL: runtimeUrl});
    await python.loadPackage("sqlite3");
    const code = await fetch("lab.py");
    if (!code.ok) throw new Error("lab.py");
    python.FS.writeFile("/lab.py", await code.text());
    await python.runPythonAsync(`
import json
import sys
from pathlib import Path
sys.path.insert(0, "/")
from lab import document_ids, evaluate, export_ai_jsonl, import_jsonl, model_file, summarize

def dispatch(route, options):
    path = "/training.sqlite3"
    try:
        if route == "/api/status":
            result = summarize(path)
        elif route == "/api/document-ids":
            result = document_ids(path)
        elif route == "/api/ai-data":
            result = {"jsonl": export_ai_jsonl(path)}
        elif route == "/api/import":
            counts, errors = import_jsonl(path, Path("/import.jsonl").read_bytes())
            result = {"counts": counts, "errors": errors}
        elif route == "/api/train":
            settings = json.loads(options)
            result = evaluate(path, settings.get("hold_model", ""), settings.get("hold_genre", ""))
        else:
            result = model_file(path, route.removeprefix("/models/"))
            if result is None:
                raise ValueError("モデルが見つかりません。")
        return json.dumps({"value": result}, ensure_ascii=False)
    except UnicodeDecodeError:
        return json.dumps({"error": "UTF-8のJSONLファイルを使用してください。"}, ensure_ascii=False)
    except ValueError as error:
        return json.dumps({"error": str(error)}, ensure_ascii=False)
`);
    storage = await openStorage();
    runtime = python;
  } catch (error) {
    console.error(error);
    throw new Error("学習環境を読み込めませんでした。ネット接続とブラウザーの保存設定を確認してください。");
  }
}

async function handle({route, body}) {
  if (!runtime) await initialize();
  const snapshot = await readStorage();
  const fs = runtime.FS;
  if (snapshot.bytes) fs.writeFile(sqlitePath, snapshot.bytes);
  else if (fs.analyzePath(sqlitePath).exists) fs.unlink(sqlitePath);
  const importing = route === "/api/import";
  if (importing) {
    if (!(body instanceof ArrayBuffer) || !body.byteLength || body.byteLength > limit) {
      throw new Error("ファイルは1バイト以上、20 MB以下にしてください。");
    }
    fs.writeFile(importPath, new Uint8Array(body));
  }
  runtime.globals.set("browser_route", route);
  runtime.globals.set("browser_options", typeof body === "string" ? body : "");
  try {
    const reply = JSON.parse(runtime.runPython("dispatch(browser_route, browser_options)"));
    if (reply.error) throw new Error(reply.error);
    if (importing || route === "/api/train") await saveStorage(snapshot.revision, fs.readFile(sqlitePath));
    return reply.value;
  } finally {
    if (importing) fs.unlink(importPath);
  }
}

self.onmessage = ({data}) => {
  queue = queue.then(async () => {
    try {
      self.postMessage({id: data.id, value: await handle(data)});
    } catch (error) {
      self.postMessage({id: data.id, error: error.message || "処理に失敗しました。"});
    }
  });
};
