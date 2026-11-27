import assert from "node:assert/strict";
import test from "node:test";
import {makePlan, requestGoogle, runBatch} from "./generation.mjs";

const settings = {topics:"鉄道\n図書館\n鉄道", genre:"解説", style:"常体", provider:"gemma",
  model:"gemma-4-26b-a4b-it", perTopic:2, limit:3};

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

test("Google request keeps the key in its header and omits thought parts", async () => {
  const text = await requestGoogle(settings.model, "test-key", "テスト題材", async (url, options) => {
    assert.equal(url, "https://generativelanguage.googleapis.com/v1beta/models/gemma-4-26b-a4b-it:generateContent");
    assert.equal(options.headers["x-goog-api-key"], "test-key");
    assert.equal(options.redirect, "error");
    assert.match(options.body, /テスト題材/);
    assert.doesNotMatch(options.body, /test-key/);
    return new Response(JSON.stringify({candidates:[{finishReason:"STOP",content:{parts:[
      {text:"推論",thought:true},{text:"本文"},
    ]}}]}));
  });
  assert.equal(text, "本文");
  await assert.rejects(requestGoogle(settings.model, "test-key", "題材",
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
