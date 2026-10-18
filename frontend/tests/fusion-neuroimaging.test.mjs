import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import ts from "typescript";

const source = fs.readFileSync(new URL("../lib/fusion.ts", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2021 } });
const fusionModule = { exports: {} };
new Function("module", "exports", compiled.outputText)(fusionModule, fusionModule.exports);
const { buildFusionResult, imagingCandidates, fusionCandidates } = fusionModule.exports;

const prediction = (probabilities) => ({
  predicted_class: "AD", confidence_score: 0.6, risk_score: 0.6, risk_level: "medium",
  probabilities, observed_issues: [], recommendations: [], disclaimer: "Research only.",
});
const eegCondition = (risk_score) => ({ risk_score, confound_severity: "None" });
const core = {
  Voice: { id: "V1", createdAt: "2026-10-18T00:00:00Z", type: "Voice", prediction: prediction({ Healthy: 0.4, AD: 0.4, PD: 0.1, MS: 0.1 }), voiceDetails: { extraction_quality: "usable" } },
  General: { id: "G1", createdAt: "2026-10-18T00:00:00Z", type: "General", prediction: prediction({ Healthy: 0.4, AD: 0.4, PD: 0.1, MS: 0.1 }), biomarkers: { a: 1, b: 2 } },
  EEG: {
    id: "E1", createdAt: "2026-10-18T00:00:00Z", type: "EEG",
    eegReport: {
      signal_quality: { grade: "good" },
      risk_scores: {},
      risk_assessment: { conditions: { AD: eegCondition(0.5), PD: eegCondition(0.2), MS: eegCondition(0.2) } },
      confound_disclosure: { severity_by_condition: {} },
    },
  },
};
const scan = (id, condition, probabilities) => ({
  id, createdAt: "2026-10-18T00:00:00Z", type: "Neuroimaging",
  neuroimaging: { condition, scanType: "Test scan", modality: "Test", prediction: condition, confidence: 90, riskLevel: "HIGH", probabilities },
});

test("fusion without neuroimaging keeps the three-model result", () => {
  const result = buildFusionResult(core);
  // AD: voice 0.5, general 0.5, EEG 0.5 at equal weight
  assert.equal(Math.round(result.conditions.AD.score * 100), 50);
  assert.equal(result.conditions.AD.contributions.length, 3);
  assert.deepEqual(result.imagingConditions, []);
  assert.ok(result.cautions.some((caution) => caution.startsWith("Not included: Neuroimaging")));
});

test("an AD MRI scan only contributes to AD, counting MCI as half", () => {
  const result = buildFusionResult(core, { AD: scan("I1", "AD", { AD: 0.6, MCI: 0.4, CN: 0 }) });
  const imaging = result.conditions.AD.contributions.find((c) => c.modality === "Neuroimaging");
  assert.ok(Math.abs(imaging.evidence - 0.8) < 1e-9);
  // (0.5*1 + 0.5*1 + 0.5*1 + 0.8*1) / 4
  assert.ok(Math.abs(result.conditions.AD.score - 0.575) < 1e-9);
  assert.equal(result.conditions.PD.contributions.length, 3);
  assert.equal(result.conditions.MS.contributions.length, 3);
  assert.deepEqual(result.imagingSourceIds, { AD: "I1" });
  assert.ok(result.explanation.includes("neuroimaging scan supports"));
});

test("binary PD and MS scans use disease vs Healthy Control and flag discordance", () => {
  const result = buildFusionResult(core, {
    PD: scan("I2", "PD", { PD: 0.9, HC: 0.1 }),
    MS: scan("I3", "MS", { MS: 0.2, HC: 0.8 }),
  });
  const pd = result.conditions.PD.contributions.find((c) => c.modality === "Neuroimaging");
  const ms = result.conditions.MS.contributions.find((c) => c.modality === "Neuroimaging");
  assert.ok(Math.abs(pd.evidence - 0.9) < 1e-9);
  assert.ok(Math.abs(ms.evidence - 0.2) < 1e-9);
  assert.equal(result.highestCondition, "AD");
  assert.ok(result.cautions.some((caution) => caution.includes("PD neuroimaging scan is elevated")));
});

test("a scan stored for another condition is ignored", () => {
  const result = buildFusionResult(core, { AD: scan("I4", "PD", { PD: 1, HC: 0 }) });
  assert.equal(result.conditions.AD.contributions.length, 3);
  assert.deepEqual(result.imagingConditions, []);
});

test("imaging candidates are filtered by condition", () => {
  const items = [scan("I1", "AD", { AD: 1 }), scan("I2", "MS", { MS: 1 }), core.Voice];
  assert.deepEqual(imagingCandidates(items, "MS").map((item) => item.id), ["I2"]);
  assert.deepEqual(fusionCandidates(items, "Neuroimaging").map((item) => item.id), ["I1", "I2"]);
});

test("demo imaging output is down-weighted and flagged", () => {
  const demo = scan("I5", "AD", { AD: 1, MCI: 0, CN: 0 });
  demo.neuroimaging.demo = true;
  const result = buildFusionResult(core, { AD: demo });
  const imaging = result.conditions.AD.contributions.find((c) => c.modality === "Neuroimaging");
  assert.equal(imaging.effectiveWeight, 0.25);
  assert.ok(result.cautions.some((caution) => caution.includes("demo")));
});

test("final class is the top condition at or above 50, otherwise HC", () => {
  // Core-only AD fused score is exactly 0.5.
  assert.equal(buildFusionResult(core).finalClass, "AD");
  const strongHealthyScan = scan("I6", "AD", { AD: 0, MCI: 0, CN: 1 });
  const result = buildFusionResult(core, { AD: strongHealthyScan });
  assert.ok(result.highestScore < 0.5);
  assert.equal(result.finalClass, "HC");
  assert.ok(result.headline.includes("Healthy control"));
});

test("voice and EEG only: excluded components are left out of the calculation", () => {
  const result = buildFusionResult({ Voice: core.Voice, EEG: core.EEG });
  assert.deepEqual(result.includedModalities, ["Voice", "EEG"]);
  assert.deepEqual(result.excludedModalities, ["General", "Neuroimaging"]);
  assert.equal(result.conditions.AD.contributions.length, 2);
  assert.ok(!result.conditions.AD.contributions.some((c) => c.modality === "General"));
  assert.equal(result.rankingModalities, 2);
  assert.deepEqual(Object.keys(result.sourceIds), ["Voice", "EEG"]);
  assert.ok(result.cautions.some((caution) => caution.startsWith("Not included: Biomedical and Neuroimaging")));
});

test("a single core component is reported as a single-model result", () => {
  const result = buildFusionResult({ EEG: core.EEG });
  assert.equal(result.agreement, "Single model");
  assert.ok(result.cautions.some((caution) => caution.includes("single-modality")));
});

test("imaging only scores just the scanned condition", () => {
  const result = buildFusionResult({}, { PD: scan("I7", "PD", { PD: 0.8, HC: 0.2 }) });
  assert.equal(result.conditions.PD.informed, true);
  assert.equal(result.conditions.AD.informed, false);
  assert.equal(result.conditions.MS.informed, false);
  assert.equal(result.highestCondition, "PD");
  assert.equal(result.finalClass, "PD");
  assert.equal(result.agreement, "Not applicable");
});

test("nothing included is rejected", () => {
  assert.throws(() => buildFusionResult({}), /at least one/);
});
