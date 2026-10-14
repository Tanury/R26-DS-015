import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import ts from "typescript";

const source = fs.readFileSync(new URL("../lib/voice-result-guidance.ts", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2021 } });
const guidanceModule = { exports: {} };
new Function("module", "exports", compiled.outputText)(guidanceModule, guidanceModule.exports);
const { buildVoiceResultGuidance } = guidanceModule.exports;

const prediction = {
  predicted_class: "PD", confidence_score: 0.8, risk_score: 0.8, risk_level: "high",
  probabilities: { Healthy: 0.05, AD: 0.1, PD: 0.8, MS: 0.05 },
  observed_issues: ["Model selected PD."], recommendations: [], disclaimer: "Research only.",
};
const details = (filename) => ({ filename, extraction_quality: "usable", quality_notes: [], patient_age: 58, recording_task: "reading", extraction_disclaimer: "Research only." });

test("WebM varies Healthy from 70 to 80 and the four simulated scores total exactly 100", () => {
  const observedScores = new Set();
  for (let index = 0; index < 1000; index += 1) {
    const result = buildVoiceResultGuidance(`assessment-${index}`, prediction, details("microphone.WEBM"));
    assert.equal(result.top.condition, "Healthy");
    assert.ok(result.top.points >= 70 && result.top.points <= 80);
    observedScores.add(result.top.points);
    assert.ok(result.selectionNote.includes(`is ${result.top.points}%`));
    assert.ok(result.selectionNote.includes(`remaining ${100 - result.top.points}%`));
    assert.equal(result.conditions.length, 4);
    assert.equal(result.conditions.reduce((sum, condition) => sum + condition.points, 0), 100);
    assert.equal(result.conditions.filter((condition) => condition.isPrimary).length, 1);
    assert.ok(result.conditions.every((condition) => Number.isInteger(condition.points) && condition.points >= 0));
  }
  assert.deepEqual([...observedScores].sort((a, b) => a - b), [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]);
});

test("educational explanation follows the displayed demo category", () => {
  const beginnings = {
    Healthy: "A healthy classification means that the analyzed voice sample",
    PD: "Parkinson’s disease can affect the muscles used for speaking.",
    AD: "In Alzheimer’s disease, speech may change because of difficulties with memory",
    MS: "Multiple sclerosis can affect the nervous system",
  };
  for (const predicted_class of ["Healthy", "PD", "AD", "MS"]) {
    const result = buildVoiceResultGuidance("explanation", { ...prediction, predicted_class }, details("sample.ogg"));
    assert.ok(result.classificationExplanation.startsWith(beginnings[predicted_class]));
  }
  const webm = buildVoiceResultGuidance("webm", prediction, details("sample.webm"));
  assert.ok(webm.classificationExplanation.startsWith(beginnings.Healthy));
  assert.doesNotMatch(webm.selectionNote, /classifier selected/);
});

test("rounding preserves a total of 100 for all existing formats and model selections", () => {
  for (const extension of ["wav", "mp3", "mp4", "m4a", "ogg", "webm", "unknown"]) {
    for (const predicted_class of ["Healthy", "AD", "PD", "MS"]) {
      for (let index = 0; index < 100; index += 1) {
        const result = buildVoiceResultGuidance(String(index), { ...prediction, predicted_class }, details(`sample.${extension}`));
        assert.equal(result.conditions.reduce((sum, condition) => sum + condition.points, 0), 100);
        assert.ok(result.conditions.every((condition) => condition.points >= 0 && condition.points <= 100));
      }
    }
  }
});

test("demo scores are stable across reopening and explicitly identified as simulated", () => {
  const result = buildVoiceResultGuidance("saved-id", prediction, details("voice.webm"));
  assert.deepEqual(result, buildVoiceResultGuidance("saved-id", prediction, details("voice.webm")));
  assert.match(result.headline, /^Demo:/);
  assert.match(result.summary, /do not assess your health/);
  assert.doesNotMatch(result.summary + result.selectionNote, /actual classifier|classifier selected/i);
});

test("WebM demo preserves stored predictions and does not reveal them through its guidance", () => {
  const input = structuredClone(prediction);
  const original = structuredClone(input);
  const result = buildVoiceResultGuidance("id", input, details("voice.webm"));
  assert.deepEqual(input, original);
  assert.equal(result.top.condition, "Healthy");
  for (const predicted_class of ["Healthy", "AD", "PD", "MS"]) {
    assert.deepEqual(result, buildVoiceResultGuidance("id", { ...input, predicted_class }, details("voice.webm")));
  }
  assert.ok(result.recommendations.some((recommendation) => recommendation.includes("simulated scores")));
  assert.ok(!result.recommendations.some((recommendation) => recommendation.startsWith("Continue routine health monitoring")));
});

test("legacy assessments without filename still render a complete 100% demo distribution", () => {
  const result = buildVoiceResultGuidance("old-id", prediction);
  assert.equal(result.top.condition, "PD");
  assert.equal(result.conditions.reduce((sum, condition) => sum + condition.points, 0), 100);
});
