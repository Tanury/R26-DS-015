const recordingMimeTypes = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/mp4",
  "audio/ogg;codecs=opus",
  "audio/ogg",
];

export function selectRecordingMimeType(isSupported: (type: string) => boolean) {
  return recordingMimeTypes.find(isSupported);
}

const extensions: Record<string, string> = {
  "audio/webm": "webm",
  "audio/mp4": "m4a",
  "audio/x-m4a": "m4a",
  "audio/ogg": "ogg",
  "audio/wav": "wav",
  "audio/x-wav": "wav",
  "audio/mpeg": "mp3",
};

export function createRecordedFile(chunks: Blob[], recorderMimeType: string): File {
  const recordedType = recorderMimeType || chunks.find((chunk) => chunk.type)?.type || "";
  const mimeType = recordedType.split(";", 1)[0].trim().toLowerCase();
  const extension = extensions[mimeType];
  if (!extension) {
    throw new Error("This browser recorded an unsupported audio format. Try another browser or upload a WAV, MP3, M4A, OGG, or WebM file.");
  }
  const file = new File(chunks, `voice-${Date.now()}.${extension}`, { type: mimeType });
  if (!file.size) {
    throw new Error("No audio was captured. Please record your speech again.");
  }
  return file;
}

// Compressed containers the backend can only read with FFmpeg. Converting them
// in the browser means the server never needs FFmpeg for recordings.
const browserDecodedTypes = new Set(["audio/webm", "audio/ogg", "audio/mp4", "audio/x-m4a"]);
const WAV_SAMPLE_RATE = 16000;

export function needsWavConversion(file: File) {
  return browserDecodedTypes.has(file.type.split(";", 1)[0].trim().toLowerCase());
}

/** Encode mono float samples (-1..1) as 16-bit PCM WAV bytes. */
export function encodeWav(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeText = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };
  writeText(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeText(8, "WAVE");
  writeText(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeText(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
  return buffer;
}

/** Decode a browser recording and re-encode it as 16 kHz mono WAV. */
export async function convertToWav(file: File): Promise<File> {
  const context = new AudioContext();
  let decoded: AudioBuffer;
  try {
    decoded = await context.decodeAudioData(await file.arrayBuffer());
  } catch {
    throw new Error("This recording could not be decoded in the browser. Record again or upload a WAV file.");
  } finally {
    void context.close().catch(() => {});
  }
  // Rendering into a one-channel context downmixes and resamples in one pass.
  const offline = new OfflineAudioContext(1, Math.max(1, Math.ceil(decoded.duration * WAV_SAMPLE_RATE)), WAV_SAMPLE_RATE);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);
  source.start();
  const rendered = await offline.startRendering();
  const name = file.name.replace(/\.[^.]+$/, "") + ".wav";
  return new File([encodeWav(rendered.getChannelData(0), WAV_SAMPLE_RATE)], name, { type: "audio/wav" });
}
