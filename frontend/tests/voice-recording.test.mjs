import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import ts from "typescript";

const source = fs.readFileSync(new URL("../lib/voice-recording.ts", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2021 } });
const recordingModule = { exports: {} };
new Function("module", "exports", compiled.outputText)(recordingModule, recordingModule.exports);
const { createRecordedFile, encodeWav, needsWavConversion, selectRecordingMimeType } = recordingModule.exports;

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

test("encodes mono 16-bit PCM WAV that the backend signature check accepts", () => {
  const bytes = new Uint8Array(encodeWav(new Float32Array([0, 1, -1, 2]), 16000));
  const view = new DataView(bytes.buffer);
  const text = (start, end) => String.fromCharCode(...bytes.slice(start, end));
  assert.equal(text(0, 4), "RIFF");
  assert.equal(text(8, 12), "WAVE");
  assert.equal(view.getUint16(22, true), 1);
  assert.equal(view.getUint32(24, true), 16000);
  assert.equal(view.getUint32(40, true), 8);
  assert.deepEqual([1, 2, 3, 4].map((i) => view.getInt16(42 + i * 2, true)), [0, 32767, -32768, 32767]);
});

test("only compressed browser containers are converted to WAV", () => {
  assert.equal(needsWavConversion(new File(["x"], "a.webm", { type: "audio/webm;codecs=opus" })), true);
  assert.equal(needsWavConversion(new File(["x"], "a.m4a", { type: "audio/x-m4a" })), true);
  assert.equal(needsWavConversion(new File(["x"], "a.wav", { type: "audio/wav" })), false);
  assert.equal(needsWavConversion(new File(["x"], "a.mp3", { type: "audio/mpeg" })), false);
});
