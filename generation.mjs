export const defaultModels = {
  gemini: "gemini-4-hydrogen",
  gemma: "gemma-4-26b-a4b-it",
  ollama: "gemma4:e2b",
};

export const modelPresets = {
  gemini: [
    {label: "Gemini 4 Hydrogen", id: "gemini-4-hydrogen"},
    {label: "Gemini 4 Neon", id: "gemini-4-neon"},
    {label: "Gemini 3.1 Flash-Lite", id: "gemini-3.1-flash-lite"},
  ],
  gemma: [
    {label: "Gemma 4", id: "gemma-4-26b-a4b-it"},
  ],
  ollama: [
    {label: "Gemma 4 E2B", id: "gemma4:e2b"},
  ],
};

const modelAliases = {
  gemini4hydrogen: "gemini-4-hydrogen",
  gemini4neon: "gemini-4-neon",
  "gemini3.1flashlite": "gemini-3.1-flash-lite",
};

export function normalizeModelId(model) {
  const normalized = String(model ?? "").normalize("NFKC").trim();
  const compact = normalized.toLowerCase().replace(/[\s_-]+/gu, "");
  return modelAliases[compact] ?? normalized;
}

const variants = ["背景と現状", "具体的な事例", "複数の立場", "課題と対応策", "日常への影響"];
const encoder = new TextEncoder();

export async function makePlan({topics, genre, style, model, provider, perTopic, limit}, known) {
  const unique = [...new Set(topics.split(/\r?\n/u).map((topic) => topic.normalize("NFKC").trim()).filter(Boolean))];
  if (!unique.length || unique.length > 200 || unique.some((topic) => Array.from(topic).length > 160)) {
    throw new Error("題材を1行に1つ、最大200件・1件160字まで入力してください。");
  }
  if (!genre || !style || !Number.isInteger(perTopic) || perTopic < 1 || perTopic > variants.length
    || !Number.isInteger(limit) || limit < 1 || limit > 20) {
    throw new Error("ジャンル・文体と、題材あたり1〜5本、今回1〜20本を指定してください。");
  }
  const ollama = provider === "ollama";
  const resolvedModel = ollama ? String(model ?? "").normalize("NFKC").trim() : normalizeModelId(model);
  if (!(ollama ? /^[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,159}$/ : /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,159}$/).test(resolvedModel)) {
    throw new Error("モデルIDを確認してください。");
  }
  const source = ollama ? "Ollama" : "Gemini API";
  const jobs = [];
  for (const topic of unique) {
    for (const angle of variants.slice(0, perTopic)) {
      const bytes = encoder.encode(JSON.stringify([source, resolvedModel, genre, style, topic, angle]));
      const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
      const id = "auto-" + Array.from(digest.slice(0, 16), (byte) => byte.toString(16).padStart(2, "0")).join("");
      if (known.has(id)) continue;
      const prompt = "題材「" + topic + "」について、「" + angle + "」に焦点を当てて書いてください。"
        + " ジャンルは「" + genre + "」、文体は「" + style + "」です。";
      jobs.push({id, label: 1, model: resolvedModel, source, topic, genre, style, prompt});
      if (jobs.length === limit) return jobs;
    }
  }
  return jobs;
}

export async function requestGoogle(model, key, prompt, fetcher = fetch) {
  if (!key) throw new Error("Google AI Studioで取得したAPIキーを入力してください。");
  const resolvedModel = normalizeModelId(model);
  const url = "https://generativelanguage.googleapis.com/v1beta/models/" + encodeURIComponent(resolvedModel) + ":generateContent";
  const instruction = "次の依頼に沿って、800〜1200字（目安1000字）の日本語本文だけを書いてください。"
    + " 題名、前置き、文字数の注記、架空の出典は書かないでください。\n\n" + prompt;
  let response;
  try {
    response = await fetcher(url, {
      method: "POST", credentials: "omit", redirect: "error", cache: "no-store",
      headers: {"Content-Type": "application/json", "x-goog-api-key": key},
      body: JSON.stringify({contents: [{parts: [{text: instruction}]}]}),
      signal: AbortSignal.timeout(180000),
    });
  } catch {
    throw new Error("Gemini APIに接続できませんでした。通信状況やAPIキーの利用制限を確認してください。");
  }
  if (response.status === 429) {
    throw new Error("Gemini APIの利用上限またはレート制限に達したため停止しました（有料枠専用モデルの場合は課金設定も確認してください）。");
  }
  if (response.status === 400 || response.status === 401 || response.status === 403) {
    throw new Error("Gemini APIでリクエストが拒否されました。APIキー・モデルID・プロジェクトの権限や課金設定を確認してください。");
  }
  if (response.status === 404) throw new Error("選んだモデルが見つかりません。モデルIDを確認してください。");
  if (!response.ok) throw new Error("Gemini APIが HTTP " + response.status + " を返したため停止しました。");
  const result = await response.json();
  const candidate = result.candidates?.[0];
  if (candidate?.finishReason && candidate.finishReason !== "STOP") {
    throw new Error("Gemini APIが本文生成を完了せず終了しました。題材や設定を確認してください。");
  }
  const text = candidate?.content?.parts?.filter((part) => !part.thought && typeof part.text === "string")
    .map((part) => part.text).join("").trim();
  if (!text) throw new Error("Gemini APIから本文を取得できませんでした。");
  return text;
}

export async function runBatch(jobs, {generate, save, stopping, progress}) {
  const counts = {saved: 0, skipped: 0};
  for (const [index, job] of jobs.entries()) {
    if (stopping()) break;
    progress({index, job, ...counts});
    const text = (await generate(job)).trim();
    const size = Array.from(text).length;
    if (size < 800 || size > 1200) {
      counts.skipped += 1;
      progress({index, job, ...counts, note: "文字数 " + size + " 字のため保留"});
      continue;
    }
    const result = await save({...job, text, character_count: size, generated_at: new Date().toISOString()});
    if (result.counts.invalid) throw new Error(result.errors?.[0] || "生成文を保存できませんでした。");
    if (result.counts.added) counts.saved += 1;
    else counts.skipped += 1;
    progress({index, job, ...counts, note: result.counts.added ? size + " 字を保存" : "本文またはIDが重複"});
  }
  return counts;
}
