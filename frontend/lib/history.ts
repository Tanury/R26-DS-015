"use client";

import { useMemo, useSyncExternalStore } from "react";
import type { HistoryItem, Prediction, SpeechFeatures } from "@/lib/types";

const HISTORY_KEY = "neurovoice-assessments";
const RESULT_KEY = "neurovoice-current-result";
const HISTORY_EVENT = "neurovoice-history-change";
const RESULT_EVENT = "neurovoice-result-change";
const EMPTY_ARRAY = "[]";

function subscribeTo(key: "history" | "result", callback: () => void) {
  const eventName = key === "history" ? HISTORY_EVENT : RESULT_EVENT;
  const storageListener = (event: StorageEvent) => {
    const storageKey = key === "history" ? HISTORY_KEY : RESULT_KEY;
    if (event.key === storageKey) callback();
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
    return JSON.parse(localStorage.getItem(HISTORY_KEY) ?? "[]") as HistoryItem[];
  } catch {
    return [];
  }
}

export function saveAssessment(
  type: HistoryItem["type"],
  prediction: Prediction,
  extras: { features?: SpeechFeatures; transcript?: string } = {},
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

export function readCurrentResult(): HistoryItem | null {
  if (typeof window === "undefined") return null;
  try {
    const value = sessionStorage.getItem(RESULT_KEY);
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
    () => localStorage.getItem(HISTORY_KEY) ?? EMPTY_ARRAY,
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
    () => sessionStorage.getItem(RESULT_KEY) ?? "",
    () => "",
  );
  return useMemo(() => {
    try { return snapshot ? (JSON.parse(snapshot) as HistoryItem) : null; }
    catch { return null; }
  }, [snapshot]);
}
