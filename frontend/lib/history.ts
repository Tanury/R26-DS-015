"use client";

import { useMemo, useSyncExternalStore } from "react";
import type { EegRiskReport } from "@/lib/eeg-types";
import type {
  BiomedicalFeatures,
  HistoryItem,
  Prediction,
  SpeechFeatures,
} from "@/lib/types";

const HISTORY_KEY = "neurorisk-assessments";
const RESULT_KEY = "neurorisk-current-result";
const HISTORY_EVENT = "neurorisk-history-change";
const RESULT_EVENT = "neurorisk-result-change";
const LEGACY_HISTORY_KEY = "neurovoice-assessments";
const LEGACY_RESULT_KEY = "neurovoice-current-result";
const EMPTY_ARRAY = "[]";

function historySnapshot() {
  return localStorage.getItem(HISTORY_KEY) ?? localStorage.getItem(LEGACY_HISTORY_KEY) ?? EMPTY_ARRAY;
}

function resultSnapshot() {
  return sessionStorage.getItem(RESULT_KEY) ?? sessionStorage.getItem(LEGACY_RESULT_KEY) ?? "";
}

function subscribeTo(key: "history" | "result", callback: () => void) {
  const eventName = key === "history" ? HISTORY_EVENT : RESULT_EVENT;
  const storageListener = (event: StorageEvent) => {
    const storageKeys = key === "history"
      ? [HISTORY_KEY, LEGACY_HISTORY_KEY]
      : [RESULT_KEY, LEGACY_RESULT_KEY];
    if (event.key && storageKeys.includes(event.key)) callback();
  };
  window.addEventListener(eventName, callback);
  window.addEventListener("storage", storageListener);
  return () => {
    window.removeEventListener(eventName, callback);
    window.removeEventListener("storage", storageListener);
  };
}

export function readHistory(): HistoryItem[] {
  if (typeof window === "undefined") return [];
  try {
    return JSON.parse(historySnapshot()) as HistoryItem[];
  } catch {
    return [];
  }
}

export function saveAssessment(
  type: HistoryItem["type"],
  prediction: Prediction,
  extras: {
    features?: SpeechFeatures;
    biomarkers?: BiomedicalFeatures;
    transcript?: string;
  } = {},
) {
  const item: HistoryItem = {
    id: `NSRA-${String(Date.now()).slice(-6)}`,
    createdAt: new Date().toISOString(),
    type,
    prediction,
    ...extras,
  };
  const history = [item, ...readHistory()].slice(0, 100);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  sessionStorage.setItem(RESULT_KEY, JSON.stringify(item));
  window.dispatchEvent(new Event(HISTORY_EVENT));
  window.dispatchEvent(new Event(RESULT_EVENT));
  return item;
}

/** EEG results carry a report instead of a speech `Prediction`. */
export function saveEegAssessment(report: EegRiskReport) {
  const item: HistoryItem = {
    id: `NEEG-${String(Date.now()).slice(-6)}`,
    createdAt: new Date().toISOString(),
    type: "EEG",
    eegReport: report,
  };
  const history = [item, ...readHistory()].slice(0, 100);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  sessionStorage.setItem(RESULT_KEY, JSON.stringify(item));
  window.dispatchEvent(new Event(HISTORY_EVENT));
  window.dispatchEvent(new Event(RESULT_EVENT));
  return item;
}

export function readCurrentResult(): HistoryItem | null {
  if (typeof window === "undefined") return null;
  try {
    const value = resultSnapshot();
    return value ? (JSON.parse(value) as HistoryItem) : null;
  } catch {
    return null;
  }
}

export function setCurrentResult(item: HistoryItem) {
  sessionStorage.setItem(RESULT_KEY, JSON.stringify(item));
  window.dispatchEvent(new Event(RESULT_EVENT));
}

export function useStoredHistory() {
  const snapshot = useSyncExternalStore(
    (callback) => subscribeTo("history", callback),
    historySnapshot,
    () => EMPTY_ARRAY,
  );
  return useMemo(() => {
    try { return JSON.parse(snapshot) as HistoryItem[]; }
    catch { return []; }
  }, [snapshot]);
}

export function useCurrentResult() {
  const snapshot = useSyncExternalStore(
    (callback) => subscribeTo("result", callback),
    resultSnapshot,
    () => "",
  );
  return useMemo(() => {
    try { return snapshot ? (JSON.parse(snapshot) as HistoryItem) : null; }
    catch { return null; }
  }, [snapshot]);
}
