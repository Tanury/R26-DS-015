import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import ts from "typescript";

const source = fs.readFileSync(new URL("../lib/general-assessment.ts", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2021 } });
const assessmentModule = { exports: {} };
new Function("module", "exports", compiled.outputText)(assessmentModule, assessmentModule.exports);
const { buildGeneralInterpretation, buildTopFactorRows } = assessmentModule.exports;

const prediction = (overrides = {}) => ({
  predicted_class: "AD", confidence_score: 0.6814, risk_score: 0.7, risk_level: "high",
  probabilities: { AD: 0.6814, Healthy: 0.1878, MS: 0.0997, PD: 0.0311 },
  observed_issues: [], recommendations: [], disclaimer: "Research only.",
  ...overrides,
});

test("rows keep backend order, readable labels, formatted values and percentage points", () => {
  const rows = buildTopFactorRows(prediction({
    top_factors: [
      { feature: "amyloid_beta_42_40_ratio", value: 0.05, contribution: 0.171 },
      { feature: "cognitive_screen_score_0_30", value: 19, contribution: 0.118 },
      { feature: "p_tau181_pg_ml", value: 95, contribution: 0.107 },
      { feature: "apoe_e4_count", value: 2, contribution: 0.04 },
      { feature: "sex", value: "Female", contribution: 0.005 },
    ],
  }));
  assert.deepEqual(rows.map((row) => row.label), ["Amyloid beta 42/40 ratio", "Cognitive screen", "Phosphorylated tau 181", "APOE ε4 allele count", "Sex"]);
  assert.deepEqual(rows.map((row) => row.value), ["0.05 ratio", "19 / 30", "95 pg/mL", "2 alleles", "Female"]);
  assert.deepEqual(rows.map((row) => row.points), [17.1, 11.8, 10.7, 4, 0.5]);
  assert.equal(rows[0].share, 1);
  assert.ok(rows.every((row, index) => index === 0 || row.share <= rows[index - 1].share));
});

test("at most five rows; unknown and non-positive factors are dropped", () => {
  const rows = buildTopFactorRows(prediction({
    top_factors: [
      { feature: "age", value: 74, contribution: 0.2 },
      { feature: "not_a_model_input", value: 1, contribution: 0.15 },
      { feature: "bmi", value: 24, contribution: 0 },
      ...["nfl_pg_ml", "gfap_pg_ml", "t_tau_pg_ml", "gdf15_pg_ml", "crp40_copy_number"].map((feature, index) => ({ feature, value: 10, contribution: 0.1 - index * 0.01 })),
    ],
  }));
  assert.equal(rows.length, 5);
  assert.ok(!rows.some((row) => row.key === "not_a_model_input" || row.key === "bmi"));
});

test("results saved before attribution return null; an empty ranking returns no rows", () => {
  assert.equal(buildTopFactorRows(prediction()), null);
  assert.deepEqual(buildTopFactorRows(prediction({ top_factors: [] })), []);
});

test("the interpretation no longer reveals other class probabilities", () => {
  const interpretation = buildGeneralInterpretation(prediction());
  assert.equal(interpretation.selectedName, "Alzheimer-type");
  assert.match(interpretation.separation, /led the next-closest disease class by 49 points/);
  assert.doesNotMatch(interpretation.separation, /19%|healthy-reference/);
});
