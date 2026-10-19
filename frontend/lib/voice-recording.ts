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
