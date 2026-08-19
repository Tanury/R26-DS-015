"use client";
import { ResultsView } from "@/components/results-view";
import { useCurrentResult } from "@/lib/history";
export default function GeneralResultsPage(){ return <ResultsView item={useCurrentResult()} expectedType="General" />; }
