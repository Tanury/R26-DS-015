from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd

from app.services.audio_features import extract_audio_features
from app.services.audio_predictor import AUDIO_FEATURE_COLUMNS, predict_audio_features


ROOT_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT_DIR / "notebooks" / "reports"
AUDIO_TEST_PATH = ROOT_DIR / "notebooks" / "data" / "splits" / "audio_test.csv"
AUDIO_PREDICTIONS_PATH = REPORT_DIR / "audio_test_predictions.csv"
SUMMARY_PATH = REPORT_DIR / "final_evaluation_summary.json"
METADATA_PATH = REPORT_DIR / "audio_model_metadata.json"

CLINICAL_NOTE = "Research-based dementia risk screening only. Not a clinical diagnosis."


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_audio_rows() -> pd.DataFrame:
    if not AUDIO_TEST_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(AUDIO_TEST_PATH)


def load_audio_predictions() -> pd.DataFrame:
    if not AUDIO_PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(AUDIO_PREDICTIONS_PATH)


def as_percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%" if float(value) <= 1 else f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def metric_value(summary: dict[str, Any], key: str) -> Any:
    for model in summary.get("models_evaluated", []):
        if model.get("model_type") == "Audio-only":
            return model.get(key)
    return None


def build_metric_strip() -> str:
    summary = load_json(SUMMARY_PATH, {})
    metadata = load_json(METADATA_PATH, {})
    model_name = metadata.get("model_name", "Audio model")
    cards = [
        ("Model", model_name),
        ("Test accuracy", as_percent(metric_value(summary, "accuracy"))),
        ("Dementia recall", as_percent(metric_value(summary, "recall_dementia"))),
        ("AUC-ROC", as_percent(metric_value(summary, "auc_roc"))),
    ]
    card_html = "".join(
        f"""
        <div class="metric-card">
            <span>{html.escape(label)}</span>
            <strong>{html.escape(str(value))}</strong>
        </div>
        """
        for label, value in cards
    )
    return f"""<div class="metric-strip">{card_html}</div>"""


def prediction_badge(result: dict[str, Any], actual_label: str | None) -> tuple[str, str]:
    predicted = result["prediction"]["predicted_class"]
    if not actual_label or actual_label == "Unknown":
        return "No actual label", "neutral"
    return ("Correct", "good") if predicted == actual_label else ("Incorrect", "bad")


def result_html(result: dict[str, Any], actual_label: str | None = None) -> str:
    prediction = result["prediction"]
    probabilities = prediction["probabilities"]
    badge, badge_class = prediction_badge(result, actual_label)
    dementia_prob = float(probabilities.get("Dementia", 0))
    control_prob = float(probabilities.get("Control", 0))
    actual = actual_label or "Unknown"

    return f"""
    <section class="result-shell">
        <div class="result-topline">
            <div>
                <span class="eyebrow">Prediction</span>
                <h2>{html.escape(prediction["predicted_class"])}</h2>
            </div>
            <div class="risk-pill {html.escape(prediction["risk_level"].lower())}">
                {html.escape(prediction["risk_level"])} risk
            </div>
        </div>
        <div class="prob-grid">
            <div>
                <div class="bar-label"><span>Control</span><b>{control_prob:.1f}%</b></div>
                <div class="bar-track"><div class="bar-fill control" style="width:{control_prob:.2f}%"></div></div>
            </div>
            <div>
                <div class="bar-label"><span>Dementia</span><b>{dementia_prob:.1f}%</b></div>
                <div class="bar-track"><div class="bar-fill dementia" style="width:{dementia_prob:.2f}%"></div></div>
            </div>
        </div>
        <div class="status-row">
            <div><span>Actual label</span><strong>{html.escape(actual)}</strong></div>
            <div><span>Accuracy label</span><strong class="{badge_class}">{html.escape(badge)}</strong></div>
            <div><span>File ID</span><strong>{html.escape(str(result["input_metadata"].get("file_id", "N/A")))}</strong></div>
        </div>
        <p class="clinical-note">{html.escape(result.get("clinical_note", CLINICAL_NOTE))}</p>
    </section>
    """


def details_html(result: dict[str, Any]) -> str:
    metadata = result.get("input_metadata", {})
    explainability = result.get("explainability", {})
    fields = [
        ("Input type", metadata.get("input_type", "N/A")),
        ("Model type", metadata.get("model_type", "audio-only")),
        ("Task", metadata.get("task", "Cookie Theft picture description")),
        ("Explainability", explainability.get("method") or explainability.get("explanation_type", "audio feature importance")),
    ]
    rows = "".join(
        f"<li><span>{html.escape(label)}</span><b>{html.escape(str(value))}</b></li>"
        for label, value in fields
    )
    return f"""
    <div class="detail-panel">
        <h3>Inference context</h3>
        <ul>{rows}</ul>
        <p>{html.escape(explainability.get("note", "Prediction is based on acoustic features such as MFCC, pause ratio, speech ratio, energy, pitch, and spectral features."))}</p>
    </div>
    """


def indicators_table(result: dict[str, Any]) -> pd.DataFrame:
    indicators = result.get("audio_indicators", {})
    return pd.DataFrame(
        [{"indicator": key, "value": value} for key, value in indicators.items()]
    )


def feature_preview(features: dict[str, Any]) -> pd.DataFrame:
    priority = [
        "duration_sec",
        "speech_ratio",
        "pause_ratio",
        "rms_energy_mean",
        "zero_crossing_rate_mean",
        "spectral_centroid_mean",
        "pitch_f0_mean",
        "voiced_frame_ratio",
        "mfcc_1_mean",
        "mfcc_2_mean",
        "mfcc_delta_1_mean",
        "mel_band_g1_mean",
    ]
    rows = []
    for key in priority:
        if key in features:
            rows.append({"feature": key, "value": features[key]})
    return pd.DataFrame(rows)


def outputs_for(result: dict[str, Any], features: dict[str, Any], actual_label: str | None):
    return (
        result_html(result, actual_label),
        details_html(result),
        indicators_table(result),
        feature_preview(features),
        result,
    )


def normalize_sample_choice(choice: str | None) -> str:
    if not choice:
        raise gr.Error("Select an audio feature row first.")
    return choice.split("|", 1)[0].strip()


def predict_from_dataset_row(choice: str):
    file_id = normalize_sample_choice(choice)
    audio_rows = load_audio_rows()
    match = audio_rows[audio_rows["file_id"].astype(str) == file_id]
    if match.empty:
        raise gr.Error(f"No audio feature row found for file_id {file_id}.")

    row = match.iloc[0]
    actual_label = str(row.get("label", "Unknown"))
    features = row.to_dict()
    result = predict_audio_features(
        features=features,
        file_id=file_id,
        input_type="audio features",
    )
    return outputs_for(result, features, actual_label)


def predict_from_feature_json(feature_json: str, file_id: str, actual_label: str):
    try:
        features = json.loads(feature_json)
    except json.JSONDecodeError as exc:
        raise gr.Error(f"Feature JSON is not valid: {exc}") from exc
    if not isinstance(features, dict):
        raise gr.Error("Feature JSON must be an object of feature_name: value pairs.")

    result = predict_audio_features(
        features=features,
        file_id=file_id or "manual_audio_features",
        input_type="audio features",
    )
    return outputs_for(result, features, actual_label or "Unknown")


def predict_from_audio_file(audio_path: str | None, actual_label: str):
    if not audio_path:
        raise gr.Error("Upload an audio file first.")
    path = Path(audio_path)
    features = extract_audio_features(str(path))
    if not features.get("audio_load_success"):
        raise gr.Error(f"Audio could not be processed: {features.get('audio_error', 'unknown error')}")

    result = predict_audio_features(
        features=features,
        file_id=path.name,
        input_type="uploaded audio file",
    )
    return outputs_for(result, features, actual_label or "Unknown")


def sample_choices() -> list[str]:
    audio_rows = load_audio_rows()
    if audio_rows.empty:
        return []
    return [
        f"{row.file_id} | {row.label}"
        for row in audio_rows[["file_id", "label"]].itertuples(index=False)
    ]


def default_feature_json() -> str:
    audio_rows = load_audio_rows()
    if audio_rows.empty:
        return "{}"
    row = audio_rows[audio_rows["file_id"].astype(str) == "035-1"]
    if row.empty:
        row = audio_rows.head(1)
    features = {}
    for key in AUDIO_FEATURE_COLUMNS:
        if key not in row.columns:
            continue
        value = row.iloc[0][key]
        if isinstance(value, np.generic):
            value = value.item()
        features[key] = value
    return json.dumps(features, indent=2)


CUSTOM_CSS = """
:root {
    --bg: #090d12;
    --panel: #101720;
    --panel-soft: #151f2a;
    --line: #263241;
    --text: #edf2f7;
    --muted: #9aa8b6;
    --accent: #34d399;
    --accent-2: #60a5fa;
    --risk: #fb7185;
}
.gradio-container {
    background: radial-gradient(circle at 20% 0%, rgba(52, 211, 153, 0.10), transparent 28%),
                linear-gradient(180deg, #090d12 0%, #0d1218 100%) !important;
    color: var(--text) !important;
}
.hero {
    padding: 28px 4px 10px;
}
.hero h1 {
    font-size: 34px;
    line-height: 1.1;
    margin: 0 0 10px;
}
.hero p {
    color: var(--muted);
    max-width: 920px;
    margin: 0;
}
.metric-strip {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin: 14px 0 18px;
}
.metric-card,
.result-shell,
.detail-panel {
    border: 1px solid var(--line);
    background: linear-gradient(180deg, rgba(21,31,42,0.96), rgba(12,18,25,0.96));
    border-radius: 8px;
    padding: 16px;
}
.metric-card span,
.status-row span,
.eyebrow,
.detail-panel li span {
    display: block;
    color: var(--muted);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0;
}
.metric-card strong {
    display: block;
    margin-top: 6px;
    font-size: 22px;
}
.result-topline {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 16px;
}
.result-topline h2 {
    margin: 4px 0 0;
    font-size: 42px;
    line-height: 1;
}
.risk-pill {
    border-radius: 999px;
    padding: 8px 12px;
    font-weight: 700;
    border: 1px solid var(--line);
}
.risk-pill.high { color: #fecdd3; background: rgba(251, 113, 133, .15); }
.risk-pill.medium { color: #fde68a; background: rgba(245, 158, 11, .15); }
.risk-pill.low { color: #bbf7d0; background: rgba(34, 197, 94, .15); }
.prob-grid {
    display: grid;
    gap: 14px;
    margin: 22px 0;
}
.bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 14px;
    margin-bottom: 7px;
}
.bar-track {
    height: 13px;
    border-radius: 999px;
    overflow: hidden;
    background: #1f2937;
}
.bar-fill {
    height: 100%;
    border-radius: 999px;
}
.bar-fill.control { background: var(--accent-2); }
.bar-fill.dementia { background: var(--risk); }
.status-row {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
}
.status-row div {
    background: rgba(255,255,255,.035);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 8px;
    padding: 12px;
}
.status-row strong {
    display: block;
    margin-top: 5px;
    font-size: 18px;
}
.good { color: var(--accent); }
.bad { color: var(--risk); }
.neutral { color: var(--muted); }
.clinical-note {
    color: var(--muted);
    border-top: 1px solid var(--line);
    margin: 16px 0 0;
    padding-top: 12px;
}
.detail-panel h3 {
    margin: 0 0 12px;
}
.detail-panel ul {
    list-style: none;
    padding: 0;
    margin: 0;
    display: grid;
    gap: 10px;
}
.detail-panel li {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    border-bottom: 1px solid rgba(255,255,255,.06);
    padding-bottom: 8px;
}
.detail-panel p {
    color: var(--muted);
    margin: 14px 0 0;
}
@media (max-width: 820px) {
    .metric-strip,
    .status-row {
        grid-template-columns: 1fr;
    }
    .result-topline {
        display: block;
    }
    .result-topline h2 {
        font-size: 34px;
    }
}
"""


def build_demo() -> gr.Blocks:
    choices = sample_choices()
    default_choice = next((c for c in choices if c.startswith("035-1 |")), choices[0] if choices else None)

    with gr.Blocks(title="Speech Dementia Risk Screening") as demo:
        gr.HTML(
            """
            <div class="hero">
                <h1>Speech-Based Dementia Risk Screening</h1>
                <p>Phase 1 dashboard for audio feature inference, uploaded speech analysis, actual-label comparison, and model result review.</p>
            </div>
            """
        )
        gr.HTML(build_metric_strip())

        with gr.Row():
            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.Tab("Audio feature row"):
                        feature_choice = gr.Dropdown(
                            choices=choices,
                            value=default_choice,
                            label="Select input feature row",
                            info="Rows come from notebooks/data/splits/audio_test.csv.",
                        )
                        run_feature = gr.Button("Run Feature Inference", variant="primary")
                    with gr.Tab("Upload audio"):
                        audio_file = gr.Audio(
                            sources=["upload"],
                            type="filepath",
                            label="Speech audio file",
                        )
                        audio_actual = gr.Dropdown(
                            choices=["Unknown", "Control", "Dementia"],
                            value="Unknown",
                            label="Actual label",
                        )
                        run_audio = gr.Button("Extract Features and Predict", variant="primary")
                    with gr.Tab("Manual features"):
                        manual_file_id = gr.Textbox(value="035-1", label="File ID")
                        manual_actual = gr.Dropdown(
                            choices=["Unknown", "Control", "Dementia"],
                            value="Dementia",
                            label="Actual label",
                        )
                        manual_json = gr.Code(
                            value=default_feature_json(),
                            language="json",
                            label="Audio feature JSON",
                            lines=16,
                        )
                        run_manual = gr.Button("Run JSON Inference", variant="primary")

            with gr.Column(scale=7):
                result_panel = gr.HTML()
                detail_panel = gr.HTML()

        with gr.Row():
            indicators = gr.Dataframe(
                headers=["indicator", "value"],
                label="Audio indicators",
                interactive=False,
                wrap=True,
            )
            feature_table = gr.Dataframe(
                headers=["feature", "value"],
                label="Selected feature preview",
                interactive=False,
                wrap=True,
            )

        raw_json = gr.JSON(label="Model output JSON")

        run_feature.click(
            predict_from_dataset_row,
            inputs=[feature_choice],
            outputs=[result_panel, detail_panel, indicators, feature_table, raw_json],
        )
        run_audio.click(
            predict_from_audio_file,
            inputs=[audio_file, audio_actual],
            outputs=[result_panel, detail_panel, indicators, feature_table, raw_json],
        )
        run_manual.click(
            predict_from_feature_json,
            inputs=[manual_json, manual_file_id, manual_actual],
            outputs=[result_panel, detail_panel, indicators, feature_table, raw_json],
        )
        demo.load(
            predict_from_dataset_row,
            inputs=[feature_choice],
            outputs=[result_panel, detail_panel, indicators, feature_table, raw_json],
        )

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        prevent_thread_lock=True,
        theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="slate"),
        css=CUSTOM_CSS,
    )
    while True:
        time.sleep(3600)
