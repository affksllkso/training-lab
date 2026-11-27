import assert from "node:assert/strict";
import test from "node:test";
import {defaultModels, makePlan, modelPresets, normalizeModelId, requestGoogle, runBatch} from "./generation.mjs";

const settings = {topics:"鉄道\n図書館\n鉄道", genre:"解説", style:"常体", provider:"gemma",
  model:"gemma-4-26b-a4b-it", perTopic:2, limit:3};

test("Gemini 4 Hydrogen and Neon are exposed as presets and normalized", async () => {
  assert.equal(defaultModels.gemini, "gemini-4-hydrogen");
  assert.deepEqual(modelPresets.gemini.map(({id}) => id), [
    "gemini-4-hydrogen",
    "gemini-4-neon",
    "gemini-3.1-flash-lite",
  ]);
  assert.equal(normalizeModelId("Gemini 4 Hydrogen"), "gemini-4-hydrogen");
  assert.equal(normalizeModelId("gemini4neon"), "gemini-4-neon");
  const plan = await makePlan({...settings, provider:"gemini", model:"Gemini 4 Neon", limit:1}, new Set());
  assert.equal(plan[0].model, "gemini-4-neon");
});

test("plan respects the limit and resumes after already-saved variants", async () => {
  const first = await makePlan(settings, new Set());
  assert.deepEqual(first.map(({topic}) => topic), ["鉄道", "鉄道", "図書館"]);
  const resumed = await makePlan(settings, new Set(first.map(({id}) => id)));
  assert.equal(resumed.length, 1);
  assert.equal(resumed[0].topic, "図書館");
  assert.equal(resumed[0].label, 1);
  assert.equal(resumed[0].source, "Gemini API");
  await assert.rejects(makePlan({...settings, model:"../../bad"}, new Set()), /モデルID/);
});

test("Google request supports Gemini 4 models, keeps the key in its header, and omits thought parts", async () => {
  for (const [inputModel, expectedId] of [
    ["gemini-4-hydrogen", "gemini-4-hydrogen"],
    ["Gemini 4 Neon", "gemini-4-neon"],
    [settings.model, "gemma-4-26b-a4b-it"],
  ]) {
    const text = await requestGoogle(inputModel, "test-key", "テスト題材", async (url, options) => {
      assert.equal(url, `https://generativelanguage.googleapis.com/v1beta/models/${expectedId}:generateContent`);
      assert.equal(options.headers["x-goog-api-key"], "test-key");
      assert.equal(options.redirect, "error");
      assert.match(options.body, /テスト題材/);
      assert.doesNotMatch(options.body, /test-key/);
      return new Response(JSON.stringify({candidates:[{finishReason:"STOP",content:{parts:[
        {text:"推論",thought:true},{text:"本文"},
      ]}}]}));
    });
    assert.equal(text, "本文");
  }
  await assert.rejects(requestGoogle("gemini-4-neon", "test-key", "題材",
    async () => new Response("", {status:429})), /レート制限/);
});

test("batch skips out-of-range text and keeps progress if the next call fails", async () => {
  const jobs = await makePlan(settings, new Set());
  const outcomes = ["短文", "文".repeat(850), new Error("429")];
  const saved = [];
  await assert.rejects(runBatch(jobs, {
    generate: () => {
      const outcome = outcomes.shift();
      if (outcome instanceof Error) throw outcome;
      return outcome;
    },
    save: (record) => {
      saved.push(record);
      return {counts:{added:1}};
    },
    stopping: () => false,
    progress: () => {},
  }), /429/);
  assert.equal(saved.length, 1);
  assert.equal(saved[0].id, jobs[1].id);
  assert.equal(saved[0].character_count, 850);
});
