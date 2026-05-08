from __future__ import annotations

import os

import gradio as gr

from app.demo_utils import checkpoint_exists, run_demo_subject
from app.subject_options import load_demo_subjects
from src.inference.output_schema import CLINICAL_DISCLAIMER
from src.utils.config_loader import load_config, resolve_path


APP_THEME = gr.themes.Soft(primary_hue="teal", neutral_hue="gray")
APP_CSS = """
.gradio-container {
  max-width: 1440px !important;
  margin: 0 auto !important;
  background: #f6f8fb;
}
.app-hero {
  background:
    linear-gradient(120deg, rgba(13, 148, 136, 0.96), rgba(15, 23, 42, 0.94)),
    radial-gradient(circle at top right, rgba(244, 63, 94, 0.18), transparent 38%);
  color: white;
  border-radius: 14px;
  padding: 28px 32px;
  margin-bottom: 18px;
  box-shadow: 0 18px 40px rgba(15, 23, 42, 0.16);
}
.app-hero h1 {
  margin: 0 0 6px;
  font-size: 34px;
  letter-spacing: 0;
}
.app-hero p {
  margin: 0;
  color: #dbeafe;
  max-width: 980px;
}
.result-shell {
  border: 1px solid #d8dee9;
  background: white;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
}
.hero {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  padding: 22px 24px;
  color: white;
}
.hero.hc { background: linear-gradient(120deg, #0f766e, #2563eb); }
.hero.ad { background: linear-gradient(120deg, #7f1d1d, #be123c); }
.hero.uncertain { background: linear-gradient(120deg, #92400e, #7c3aed); }
.hero h2 {
  margin: 3px 0;
  font-size: 28px;
  letter-spacing: 0;
}
.hero p {
  margin: 0;
  font-size: 17px;
  color: #eff6ff;
}
.eyebrow {
  text-transform: uppercase;
  font-size: 12px;
  letter-spacing: .08em;
  color: #bfdbfe;
  font-weight: 700;
}
.hero-badge {
  min-width: 128px;
  text-align: right;
}
.hero-badge span {
  display: block;
  color: #dbeafe;
  font-size: 12px;
  font-weight: 700;
}
.hero-badge strong {
  display: block;
  font-size: 30px;
}
.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px;
  padding: 16px;
}
.metric-card {
  background: #f8fafc;
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  padding: 13px 14px;
  min-height: 92px;
}
.metric-card span {
  display: block;
  color: #667085;
  font-size: 12px;
  font-weight: 700;
}
.metric-card strong {
  display: block;
  margin-top: 5px;
  color: #111827;
  font-size: 20px;
  line-height: 1.15;
}
.metric-card small {
  display: block;
  margin-top: 5px;
  color: #667085;
}
.note-row {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  border-top: 1px solid #e5e7eb;
  padding: 14px 16px 16px;
  color: #172033 !important;
  background: #ffffff;
}
.note-row span {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: #ecfdf5 !important;
  border: 1px solid #99f6e4;
  border-radius: 8px;
  color: #172033 !important;
  font-size: 14px;
  font-weight: 650;
  line-height: 1.3;
  padding: 7px 11px;
  min-height: 34px;
}
.note-row span strong {
  color: #0f766e !important;
  font-weight: 800;
}
.details-panel {
  border: 1px solid #d8dee9;
  border-radius: 12px;
  background: white;
  padding: 14px 18px;
}
.details-panel summary {
  cursor: pointer;
  font-weight: 800;
  color: #164e63;
}
.details-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 12px;
  margin-top: 14px;
}
.details-grid div {
  border: 1px solid #e5e7eb;
  background: #f8fafc;
  border-radius: 9px;
  padding: 11px 12px;
}
.details-grid b {
  display: block;
  color: #334155;
  margin-bottom: 4px;
}
.details-grid span {
  color: #667085;
  word-break: break-word;
}
.error-box {
  background: #fef2f2;
  border: 1px solid #fecaca;
  color: #991b1b;
  border-radius: 10px;
  padding: 14px;
  white-space: pre-wrap;
}
.download-row button {
  width: 100%;
}
.wrap .progress-text,
.wrap .progress-text *,
.wrap .progress-level,
.wrap .eta-bar,
.wrap .status,
.wrap .status *,
.wrap .meta-text,
.wrap .meta-text *,
[data-testid="progress-text"],
[data-testid="progress-text"] * {
  display: none !important;
  visibility: hidden !important;
}
.wrap .meta-text,
.wrap .meta-text * {
  display: none !important;
}
.wrap .progress-bar,
.wrap .progress-bar-wrap,
.wrap .generating {
  display: none !important;
}
.wrap .progress-bar span,
.wrap .progress-bar > div {
  display: none !important;
}
.clinical-note {
  margin: 16px 0 0;
  color: #344054 !important;
  font-size: 14px;
  line-height: 1.45;
}
.clinical-note strong {
  color: #111827 !important;
}
"""


def _allowed_paths() -> list[str]:
    cfg = load_config("configs/training.yaml")
    path_keys = [
        "figures_dir",
        "reports_dir",
        "logs_dir",
        "processed_epochs_dir",
        "embeddings_dir",
    ]
    paths = [str(resolve_path(cfg["paths"][key])) for key in path_keys if key in cfg["paths"]]
    paths.append(str(resolve_path("outputs/gradio_cache")))
    return paths


def _setup_message() -> str:
    return """
    <div class="error-box">
      No trained model checkpoint found.<br><br>
      Run training before using this demo:<br>
      <code>python -m src.training.train_eegnet --config configs/training.yaml --fold 2</code>
    </div>
    """


def build_app() -> gr.Blocks:
    subjects = load_demo_subjects()
    checkpoint_ok = checkpoint_exists()
    with gr.Blocks(
        title="EEG AD/HC Phase 1 Research Demo",
        fill_width=True,
    ) as demo:
        gr.HTML(
            """
            <div class="app-hero">
              <h1>EEG AD/HC Phase 1 Research Demo</h1>
              <p>Subject-level EEGNet inference with fold calibration, signal-quality review, visual EEG evidence, and downloadable JSON/HTML reports.</p>
            </div>
            """
        )

        if not checkpoint_ok:
            gr.HTML(_setup_message())

        with gr.Row():
            with gr.Column(scale=2, min_width=360):
                subject = gr.Dropdown(
                    choices=subjects,
                    value=subjects[0],
                    label="Subject",
                    info="Validation subjects are shown for the active fold when splits are available.",
                )
            with gr.Column(scale=1, min_width=260):
                run = gr.Button("Run EEG Analysis", variant="primary", interactive=checkpoint_ok)
            with gr.Column(scale=1, min_width=260, elem_classes=["download-row"]):
                json_download = gr.DownloadButton("Download result JSON", value=None, interactive=False)
                html_download = gr.DownloadButton("Download designed HTML", value=None, interactive=False)

        result_panel = gr.HTML(label="Subject-level result")

        more_panel = gr.HTML(label="View more")

        with gr.Accordion("Full JSON report", open=False):
            report = gr.Code(label="Structured JSON", language="json", lines=24)

        with gr.Row():
            with gr.Column(scale=1):
                raw_plot = gr.Image(label="Raw EEG", type="filepath", height=320)
            with gr.Column(scale=1):
                filtered_plot = gr.Image(label="Filtered EEG", type="filepath", height=320)

        with gr.Row():
            with gr.Column(scale=1):
                prob_chart = gr.Image(label="Class Probability", type="filepath", height=320)
            with gr.Column(scale=1):
                epoch_dist = gr.Image(label="Epoch Probability Distribution", type="filepath", height=320)

        gr.HTML(
            f"""
            <p class="clinical-note">
              <strong>Clinical disclaimer:</strong> {CLINICAL_DISCLAIMER}
            </p>
            """
        )

        run.click(
            run_demo_subject,
            inputs=subject,
            outputs=[
                raw_plot,
                filtered_plot,
                prob_chart,
                epoch_dist,
                result_panel,
                more_panel,
                report,
                json_download,
                html_download,
            ],
        ).then(
            lambda: (gr.update(interactive=True), gr.update(interactive=True)),
            outputs=[json_download, html_download],
        )
    return demo


if __name__ == "__main__":
    build_app().launch(
        share=os.environ.get("GRADIO_SHARE", "0") == "1",
        allowed_paths=_allowed_paths(),
        theme=APP_THEME,
        css=APP_CSS,
    )
