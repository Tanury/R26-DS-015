const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const ts = require("typescript");

const source = fs.readFileSync(path.join(__dirname, "../lib/eeg-prediction-guidance.ts"), "utf8");
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2021 } });
const guidanceModule = { exports: {} };
new Function("module", "exports", compiled.outputText)(guidanceModule, guidanceModule.exports);
const { buildEegUserGuidance } = guidanceModule.exports;
const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, "../../backend/app/models/eeg/cohort/AD-AR-sub-30001.json"), "utf8"));
const report = () => structuredClone(fixture);

test("shows actual model scores with a numerical reason for every alternative", () => {
  const value = buildEegUserGuidance(report());
  assert.equal(value.top.condition, "AD");
  assert.equal(value.top.points, 94.5);
  assert.equal(value.top.riskScore, .9746);
  assert.match(value.conditions.find(c => c.condition === "MS").explanation, /92.5 percentage points below AD/);
  assert.match(value.evidence.join(" "), /does not identify a compatible output head/);
  assert.equal(value.conditions.find(c => c.condition === "Healthy").points, 1.8);
});

test("new occlusion evidence identifies its target even when the heads disagree", () => {
  const input = report();
  input.explainability.target_condition = "MS";
  input.explainability.target_output = "independent_risk";
  const value = buildEegUserGuidance(input);
  assert.equal(value.top.condition, "AD");
  assert.match(value.evidence.join(" "), /MS independent risk head/);
  assert.match(value.evidence.join(" "), /not proof of disease or an attribution of the four-class selection/);
});

test("ranking follows measured probabilities, not a stale predicted label or subject id", () => {
  const input = report();
  input.optional_four_class_prediction.class_probabilities = { HC: .7, AD: .1, PD: .1, MS: .1 };
  const first = buildEegUserGuidance(input);
  assert.equal(first.top.condition, "Healthy");
  assert.equal(first.top.points, 70);
  assert.ok(first.cautions.some(c => c.includes("elevated for AD")));
  input.subject_id = "a different recording";
  input.generated_at = "tomorrow";
  assert.deepEqual(buildEegUserGuidance(input).conditions, first.conditions);
});

test("missing auxiliary head uses real snake-case risk scores and never invents healthy", () => {
  const input = report();
  input.optional_four_class_prediction = null;
  input.risk_assessment.conditions = {};
  input.risk_scores = { ad_risk_score: .05, pd_risk_score: .2, ms_risk_score: .1 };
  const value = buildEegUserGuidance(input);
  assert.equal(value.top.condition, "PD");
  assert.equal(value.top.points, 20);
  assert.equal(value.conditions[0].score, null);
  assert.ok(value.cautions.some(c => c.includes("low band")));
});

test("ties do not single out a disease", () => {
  const input = report();
  input.optional_four_class_prediction.class_probabilities = { HC: .1, AD: .4, PD: .4, MS: .1 };
  const value = buildEegUserGuidance(input);
  assert.ok(value.conditions.every(c => !c.isPrimary));
  assert.equal(value.headline, "No single preferred EEG pattern");
  assert.match(value.recommendations[0], /inconclusive/);
});

test("invalid or absent measurements produce an unavailable result", () => {
  const input = report();
  input.optional_four_class_prediction.class_probabilities = { HC: NaN, AD: 2, PD: -1, MS: .2 };
  input.risk_assessment.conditions = {};
  input.risk_scores = {};
  const value = buildEegUserGuidance(input);
  assert.equal(value.headline, "EEG result unavailable");
  assert.ok(value.conditions.every(c => c.points === null && !c.isPrimary));
});

test("four displayed percentages total exactly 100% for rounding and API precision edge cases", () => {
  for (const probabilities of [
    { HC: .3333, AD: .3333, PD: .3334, MS: 0 }, // Individually rounds to 99.9%.
    { HC: .018, AD: .9447, PD: .0176, MS: .0197 }, // Individually rounds to 100.1%.
    { HC: .2499, AD: .2499, PD: .2499, MS: .2499 }, // Rounded API total.
    { HC: 1, AD: 0, PD: 0, MS: 0 },
    { HC: .9999, AD: .0001, PD: 0, MS: 0 },
  ]) {
    const input = report();
    input.optional_four_class_prediction.class_probabilities = probabilities;
    const original = structuredClone(input);
    const value = buildEegUserGuidance(input);
    assert.equal(value.scoreLabel, "Four-class model score");
    // Sum the displayed decimal values in integer tenths, without binary float noise.
    assert.equal(value.conditions.reduce((sum, c) => sum + Math.round(c.points * 10), 0), 1000);
    for (const c of value.conditions) {
      assert.ok(c.points >= 0 && c.points <= 100);
      assert.ok(c.explanation.includes(`${c.points.toFixed(1)}%`));
    }
    assert.deepEqual(input, original, "display balancing must not change saved model outputs");
  }
});

test("balanced rounding preserves measured ties and independent risk scores", () => {
  const input = report();
  input.optional_four_class_prediction.class_probabilities = { HC: 0, AD: 1 / 3, PD: 1 / 3, MS: 1 / 3 };
  const value = buildEegUserGuidance(input);
  assert.deepEqual(value.conditions.map(c => c.points), [0, 33.4, 33.3, 33.3]);
  assert.ok(value.conditions.every(c => !c.isPrimary));
  assert.equal(value.headline, "No single preferred EEG pattern");
  input.optional_four_class_prediction = null;
  input.risk_assessment.conditions = {};
  input.risk_scores = { AD: .8, PD: .7, MS: .6 };
  assert.deepEqual(buildEegUserGuidance(input).conditions.map(c => c.points), [null, 80, 70, 60]);
});

test("every stored cohort four-class result displays a total of 100%", () => {
  const directory = path.join(__dirname, "../../backend/app/models/eeg/cohort");
  let checked = 0;
  for (const file of fs.readdirSync(directory).filter(name => name.endsWith(".json"))) {
    const input = JSON.parse(fs.readFileSync(path.join(directory, file), "utf8"));
    if (!input.optional_four_class_prediction) continue;
    const value = buildEegUserGuidance(input);
    assert.equal(value.scoreLabel, "Four-class model score", file);
    assert.equal(value.conditions.reduce((sum, c) => sum + Math.round(c.points * 10), 0), 1000, file);
    checked++;
  }
  assert.ok(checked > 0);
});
