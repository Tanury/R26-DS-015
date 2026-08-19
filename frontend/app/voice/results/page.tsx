"use client";
import { ResultsView } from "@/components/results-view";
import { useCurrentResult } from "@/lib/history";
export default function VoiceResultsPage(){ return <ResultsView item={useCurrentResult()} expectedType="Voice" />; }
