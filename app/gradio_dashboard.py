import json
import os
import sys
import time
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_SITE_PACKAGES = PROJECT_ROOT / "venv" / "Lib" / "site-packages"
if VENV_SITE_PACKAGES.exists() and str(VENV_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(VENV_SITE_PACKAGES))

import gradio as gr

from app.services.prediction_service import NeurologicalPredictionService


SERVICE = NeurologicalPredictionService()


FEATURES: List[Tuple[str, str]] = [
    ("age", "Age"),
    ("sex", "Sex"),
    ("moca_total_score", "MoCA total score"),
    ("updrs_part_i", "UPDRS Part I"),
    ("updrs_part_ii", "UPDRS Part II"),
    ("updrs_part_iii", "UPDRS Part III"),
    ("updrs_part_iv", "UPDRS Part IV"),
    ("disease_duration_years", "Disease duration years"),
    ("amyloid_beta_42_pg_ml", "Amyloid-beta 42"),
    ("amyloid_beta_40_pg_ml", "Amyloid-beta 40"),
    ("p_tau181_pg_ml", "p-tau181"),
    ("t_tau_pg_ml", "Total tau"),
    ("nfl_pg_ml", "NfL"),
    ("gfap_pg_ml", "GFAP"),
    ("alpha_synuclein_pg_ml", "Alpha-synuclein"),
    ("neuroinflam_score", "Neuroinflammation score"),
    ("tau_amyloid_ratio", "Tau-amyloid ratio"),
]


DEFAULT_PROFILE = [
    72,
    "Male",
    22,
    14,
    26,
    58,
    9,
    8.0,
    690,
    6800,
    1.4,
    180,
    29.0,
    130,
    1350,
    0.48,
    0.002,
]

DEFAULT_JSON = json.dumps(
    {
        "age": 72,
        "sex": "Male",
        "moca_total_score": 22,
        "updrs_part_i": 14,
        "updrs_part_ii": 26,
        "updrs_part_iii": 58,
        "updrs_part_iv": 9,
        "disease_duration_years": 8.0,
        "amyloid_beta_42_pg_ml": 690,
        "amyloid_beta_40_pg_ml": 6800,
        "p_tau181_pg_ml": 1.4,
        "t_tau_pg_ml": 180,
        "nfl_pg_ml": 29.0,
        "gfap_pg_ml": 130,
        "alpha_synuclein_pg_ml": 1350,
        "neuroinflam_score": 0.48,
        "tau_amyloid_ratio": 0.002,
    },
    indent=2,
)


def _clean_value(value: Any) -> Any:
    if value == "" or value == "Not available":
        return None
    return value


def _build_model_payload(values: Tuple[Any, ...]) -> Dict[str, Any]:
    payload = {
        key: cleaned_value
        for (key, _), value in zip(FEATURES, values)
        if (cleaned_value := _clean_value(value)) is not None
    }

    amyloid_42 = payload.get("amyloid_beta_42_pg_ml")
    amyloid_40 = payload.get("amyloid_beta_40_pg_ml")
    if amyloid_42 is not None and amyloid_40:
        payload["amyloid_beta_42_40_ratio"] = float(amyloid_42) / float(amyloid_40)

    return payload


def _confidence_label(confidence: float) -> Tuple[str, str]:
    if confidence >= 90:
        return "Very high", "#22c55e"
    if confidence >= 75:
        return "High", "#84cc16"
    if confidence >= 60:
        return "Moderate", "#f59e0b"
    return "Review carefully", "#ef4444"


def _risk_color(label: str) -> str:
    label = str(label).lower()
    if label == "high":
        return "#ef4444"
    if label == "medium":
        return "#f59e0b"
    if label == "low":
        return "#22c55e"
    return "#38bdf8"


def _probability_bars(probabilities: Dict[str, float], active_label: str) -> str:
    if not probabilities:
        return "<p class='muted'>No class probabilities were returned by this model.</p>"

    rows = []
    for label, value in sorted(probabilities.items(), key=lambda item: item[1], reverse=True):
        color = _risk_color(label) if label in {"Low", "Medium", "High"} else "#38bdf8"
        is_active = " active" if str(label) == str(active_label) else ""
        rows.append(
            f"""
            <div class="prob-row{is_active}">
                <div class="prob-top"><span>{escape(str(label))}</span><strong>{value:.2f}%</strong></div>
                <div class="prob-track"><div class="prob-fill" style="width:{max(0, min(value, 100)):.2f}%; background:{color};"></div></div>
            </div>
            """
        )
    return "\n".join(rows)


def _feature_gallery(features: Iterable[Dict[str, Any]]) -> str:
    cards = []
    for index, feature in enumerate(features, 1):
        name = escape(str(feature.get("display_name") or feature.get("feature") or "Feature"))
        feature_type = escape(str(feature.get("feature_type", "feature")).replace("_", " "))
        importance = float(feature.get("importance", 0.0))
        cards.append(
            f"""
            <article class="feature-card">
                <span class="rank">#{index}</span>
                <h3>{name}</h3>
                <p>{feature_type}</p>
                <div class="importance"><span style="width:{min(100, importance * 40):.1f}%"></span></div>
                <strong>{importance:.4f}</strong>
            </article>
            """
        )
    return "<div class='feature-gallery'>" + "\n".join(cards) + "</div>"


def _summary_html(result: Dict[str, Any], latency_ms: float, observed_count: int) -> str:
    disease_conf = float(result.get("disease_confidence", 0.0))
    risk_conf = float(result.get("risk_confidence", 0.0))
    accuracy_label, accuracy_color = _confidence_label(max(disease_conf, risk_conf))
    risk_color = _risk_color(result.get("predicted_risk", ""))
    disease = escape(str(result.get("disease_full_name", "")))
    disease_code = escape(str(result.get("predicted_disease", "")))
    risk = escape(str(result.get("predicted_risk", "")))

    return f"""
    <section class="hero-result">
        <div>
            <p class="eyebrow">Phase 1 inference result</p>
            <h1>{risk} neurological risk</h1>
            <p class="subtitle">Most consistent disease pattern: <strong>{disease}</strong> ({disease_code})</p>
        </div>
        <div class="metric-grid">
            <div class="metric-card">
                <span>Disease confidence</span>
                <strong>{disease_conf:.2f}%</strong>
            </div>
            <div class="metric-card">
                <span>Risk confidence</span>
                <strong style="color:{risk_color};">{risk_conf:.2f}%</strong>
            </div>
            <div class="metric-card">
                <span>Accuracy label</span>
                <strong style="color:{accuracy_color};">{accuracy_label}</strong>
            </div>
            <div class="metric-card">
                <span>Observed inputs</span>
                <strong>{observed_count}</strong>
            </div>
        </div>
        <p class="latency">Inference completed in {latency_ms:.1f} ms. Embedding dimension: {result.get("embedding_info", {}).get("z_bio_dimension", 256)}.</p>
    </section>
    """


def _details_html(result: Dict[str, Any]) -> str:
    return f"""
    <div class="result-columns">
        <section class="panel">
            <h2>Disease probabilities</h2>
            {_probability_bars(result.get("disease_probabilities", {}), result.get("predicted_disease", ""))}
        </section>
        <section class="panel">
            <h2>Risk probabilities</h2>
            {_probability_bars(result.get("risk_probabilities", {}), result.get("predicted_risk", ""))}
        </section>
    </div>
    """


def _interpretation_html(result: Dict[str, Any]) -> str:
    conclusion = escape(str(result.get("conclusion", "")))
    recommendation = escape(str(result.get("clinical_recommendation", "")))
    disclaimer = escape(str(result.get("disclaimer", "")))

    return f"""
    <section class="panel interpretation">
        <h2>Clinical interpretation</h2>
        <p>{conclusion}</p>
        <h2>Recommendation</h2>
        <p>{recommendation}</p>
        <p class="disclaimer">{disclaimer}</p>
    </section>
    """


def predict(*values: Any) -> Tuple[str, str, str, str]:
    payload = _build_model_payload(values)
    start = time.perf_counter()
    result = SERVICE.predict(payload)
    latency_ms = (time.perf_counter() - start) * 1000

    return (
        _summary_html(result, latency_ms, observed_count=len(payload)),
        _details_html(result),
        _feature_gallery(result.get("important_features", [])),
        _interpretation_html(result),
    )


def apply_json_to_dashboard(json_text: str, *current_values: Any) -> Tuple[Any, ...]:
    try:
        data = json.loads(json_text or "{}")
    except json.JSONDecodeError as exc:
        return (*current_values, _status_html(f"Invalid JSON: {exc.msg}", "error"))

    if not isinstance(data, dict):
        return (*current_values, _status_html("JSON input must be an object.", "error"))

    next_values = []
    applied = 0
    for index, (key, _) in enumerate(FEATURES):
        if key not in data:
            next_values.append(current_values[index])
            continue

        value = data[key]
        if key == "sex":
            value = value if value in {"Female", "Male", "Other", "Not available"} else "Not available"
        next_values.append(value)
        applied += 1

    return (
        *next_values,
        _status_html(f"Applied {applied} JSON fields to the dashboard inputs.", "success"),
    )


def _status_html(message: str, status: str) -> str:
    color = "#22c55e" if status == "success" else "#ef4444"
    return f"""
    <div class="json-status" style="border-color:{color};">
        <strong style="color:{color};">{escape(message)}</strong>
    </div>
    """


def clear_outputs() -> Tuple[str, str, str, str]:
    return "", "", "", ""


CSS = """
:root {
    color-scheme: dark;
}
.gradio-container {
    background:
        radial-gradient(circle at top left, rgba(20, 184, 166, 0.14), transparent 28rem),
        linear-gradient(145deg, #070b13 0%, #111827 45%, #0b1120 100%) !important;
}
.app-shell {
    max-width: 1220px;
    margin: 0 auto;
}
.title-block {
    padding: 14px 0 8px;
}
.title-block h1 {
    margin: 0;
    font-size: 2.1rem;
    line-height: 1.1;
    letter-spacing: 0;
}
.title-block p {
    max-width: 820px;
    color: #a7b2c3;
    font-size: 1rem;
}
.hero-result,
.panel,
.feature-card {
    border: 1px solid rgba(148, 163, 184, 0.18);
    background: rgba(15, 23, 42, 0.82);
    box-shadow: 0 18px 45px rgba(0, 0, 0, 0.22);
    border-radius: 8px;
}
.hero-result {
    padding: 24px;
}
.eyebrow {
    margin: 0 0 8px;
    color: #67e8f9;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
}
.hero-result h1 {
    margin: 0;
    font-size: 2rem;
    letter-spacing: 0;
}
.subtitle,
.latency,
.muted,
.feature-card p,
.disclaimer {
    color: #a7b2c3;
}
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin-top: 20px;
}
.metric-card {
    padding: 14px;
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 8px;
    background: rgba(2, 6, 23, 0.48);
}
.metric-card span {
    display: block;
    color: #94a3b8;
    font-size: 0.78rem;
}
.metric-card strong {
    display: block;
    margin-top: 6px;
    color: #f8fafc;
    font-size: 1.35rem;
}
.result-columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
}
.panel {
    padding: 18px;
}
.panel h2 {
    margin: 0 0 12px;
    font-size: 1rem;
    letter-spacing: 0;
}
.prob-row {
    margin: 13px 0;
}
.prob-top {
    display: flex;
    justify-content: space-between;
    gap: 14px;
    margin-bottom: 6px;
    color: #dbeafe;
}
.prob-track {
    height: 9px;
    overflow: hidden;
    background: rgba(71, 85, 105, 0.5);
    border-radius: 999px;
}
.prob-fill {
    height: 100%;
    border-radius: 999px;
}
.prob-row.active .prob-top span {
    color: #f8fafc;
    font-weight: 700;
}
.feature-gallery {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 12px;
}
.feature-card {
    padding: 14px;
    min-height: 150px;
}
.rank {
    color: #67e8f9;
    font-size: 0.8rem;
    font-weight: 800;
}
.feature-card h3 {
    margin: 10px 0 6px;
    font-size: 1rem;
    letter-spacing: 0;
}
.importance {
    height: 7px;
    margin: 14px 0 10px;
    border-radius: 999px;
    background: rgba(71, 85, 105, 0.55);
    overflow: hidden;
}
.importance span {
    display: block;
    height: 100%;
    background: #38bdf8;
}
.interpretation p {
    line-height: 1.55;
}
.disclaimer {
    margin-top: 16px;
    font-size: 0.86rem;
}
.json-status {
    margin-top: 12px;
    padding: 12px 14px;
    border: 1px solid;
    border-radius: 8px;
    background: rgba(2, 6, 23, 0.48);
}
@media (max-width: 900px) {
    .metric-grid,
    .result-columns,
    .feature-gallery {
        grid-template-columns: 1fr;
    }
}
"""

THEME = gr.themes.Soft(primary_hue="cyan", neutral_hue="slate")


def build_dashboard() -> gr.Blocks:
    with gr.Blocks(
        title="Neurological Risk Assessment Dashboard",
    ) as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(
                """
                <div class="title-block">
                    <h1>Neurological Risk Assessment</h1>
                    <p>Phase 1 dashboard for the selected clinical and fluid biomarker profile. All visible inputs are editable.</p>
                </div>
                """
            )

            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    with gr.Tabs():
                        with gr.Tab("Dashboard Input"):
                            with gr.Accordion("Input profile", open=True):
                                with gr.Row():
                                    age = gr.Number(label="Age", value=72)
                                    sex = gr.Dropdown(
                                        label="Sex",
                                        choices=["Female", "Male", "Other", "Not available"],
                                        value="Male",
                                    )
                                    moca_total_score = gr.Number(label="MoCA total score", value=22)
                                with gr.Row():
                                    updrs_part_i = gr.Number(label="UPDRS Part I", value=14)
                                    updrs_part_ii = gr.Number(label="UPDRS Part II", value=26)
                                    updrs_part_iii = gr.Number(label="UPDRS Part III", value=58)
                                    updrs_part_iv = gr.Number(label="UPDRS Part IV", value=9)
                                with gr.Row():
                                    disease_duration_years = gr.Number(
                                        label="Disease duration years", value=8.0
                                    )
                                    amyloid_beta_42_pg_ml = gr.Number(
                                        label="Amyloid-beta 42", value=690
                                    )
                                    amyloid_beta_40_pg_ml = gr.Number(
                                        label="Amyloid-beta 40", value=6800
                                    )
                                with gr.Row():
                                    p_tau181_pg_ml = gr.Number(label="p-tau181", value=1.4)
                                    t_tau_pg_ml = gr.Number(label="Total tau", value=180)
                                    nfl_pg_ml = gr.Number(label="NfL", value=29.0)
                                    gfap_pg_ml = gr.Number(label="GFAP", value=130)
                                with gr.Row():
                                    alpha_synuclein_pg_ml = gr.Number(
                                        label="Alpha-synuclein", value=1350
                                    )
                                    neuroinflam_score = gr.Number(
                                        label="Neuroinflammation score", value=0.48
                                    )
                                    tau_amyloid_ratio = gr.Number(
                                        label="Tau-amyloid ratio", value=0.002
                                    )

                            inputs = [
                                age,
                                sex,
                                moca_total_score,
                                updrs_part_i,
                                updrs_part_ii,
                                updrs_part_iii,
                                updrs_part_iv,
                                disease_duration_years,
                                amyloid_beta_42_pg_ml,
                                amyloid_beta_40_pg_ml,
                                p_tau181_pg_ml,
                                t_tau_pg_ml,
                                nfl_pg_ml,
                                gfap_pg_ml,
                                alpha_synuclein_pg_ml,
                                neuroinflam_score,
                                tau_amyloid_ratio,
                            ]

                            with gr.Row():
                                predict_btn = gr.Button("Run inference", variant="primary")
                                clear_btn = gr.Button("Clear results", variant="secondary")

                            gr.Examples(
                                examples=[DEFAULT_PROFILE],
                                inputs=inputs,
                                label="Example profiles",
                            )

                        with gr.Tab("JSON Input"):
                            json_input = gr.Textbox(
                                label="JSON input",
                                value=DEFAULT_JSON,
                                lines=22,
                                max_lines=28,
                            )
                            apply_json_btn = gr.Button("Apply JSON to dashboard", variant="primary")
                            json_status = gr.HTML()

                with gr.Column(scale=6):
                    summary_output = gr.HTML()
                    probability_output = gr.HTML()
                    feature_output = gr.HTML(label="Important features")
                    interpretation_output = gr.HTML()

            outputs = [
                summary_output,
                probability_output,
                feature_output,
                interpretation_output,
            ]
            apply_json_btn.click(
                fn=apply_json_to_dashboard,
                inputs=[json_input, *inputs],
                outputs=[*inputs, json_status],
            )
            predict_btn.click(fn=predict, inputs=inputs, outputs=outputs)
            clear_btn.click(fn=clear_outputs, inputs=None, outputs=outputs)

    return demo


dashboard = build_dashboard()


if __name__ == "__main__":
    port = int(os.getenv("GRADIO_PORT", "7860"))
    dashboard.queue(default_concurrency_limit=4).launch(
        server_name=os.getenv("GRADIO_HOST", "127.0.0.1"),
        server_port=port,
        theme=THEME,
        css=CSS,
    )
