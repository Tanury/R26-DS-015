import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import ts from "typescript";

const source = fs.readFileSync(new URL("../lib/voice-recording.ts", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2021 } });
const recordingModule = { exports: {} };
new Function("module", "exports", compiled.outputText)(recordingModule, recordingModule.exports);
const { createRecordedFile, selectRecordingMimeType } = recordingModule.exports;

test("selects a supported microphone container across browser capabilities", () => {
  assert.equal(selectRecordingMimeType((type) => type.startsWith("audio/webm")), "audio/webm;codecs=opus");
  assert.equal(selectRecordingMimeType((type) => type === "audio/mp4"), "audio/mp4");
  assert.equal(selectRecordingMimeType((type) => type === "audio/ogg;codecs=opus"), "audio/ogg;codecs=opus");
  assert.equal(selectRecordingMimeType(() => false), undefined);
});

for (const [recordedType, expectedType, extension] of [
  ["audio/webm;codecs=opus", "audio/webm", "webm"],
  ["audio/mp4;codecs=mp4a.40.2", "audio/mp4", "m4a"],
  ["audio/ogg;codecs=opus", "audio/ogg", "ogg"],
]) {
  test(`preserves recording bytes and submits the correct type for ${recordedType}`, async () => {
    const file = createRecordedFile([new Blob(["first"]), new Blob(["last"])], recordedType);
    assert.equal(file.type, expectedType);
    assert.ok(file.name.endsWith(`.${extension}`));
    assert.equal(await file.text(), "firstlast");
    const form = new FormData();
    form.append("file", file);
    const request = new Request("http://localhost/voice-assessments/", { method: "POST", body: form });
    assert.ok((await request.text()).includes(`Content-Type: ${expectedType}\r\n`));
  });
}

test("uses the captured chunk type when the recorder does not report one", () => {
  const file = createRecordedFile([new Blob(["audio"], { type: "audio/mp4;codecs=mp4a.40.2" })], "");
  assert.equal(file.type, "audio/mp4");
  assert.ok(file.name.endsWith(".m4a"));
});

test("rejects empty and unknown recordings instead of mislabelling them as WebM", () => {
  assert.throws(() => createRecordedFile([], "audio/webm"), /No audio was captured/);
  assert.throws(() => createRecordedFile([new Blob(["audio"])], ""), /unsupported audio format/);
  assert.throws(() => createRecordedFile([new Blob(["audio"])], "audio/unknown"), /unsupported audio format/);
});
