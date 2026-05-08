from __future__ import annotations

import base64
import json
import mimetypes
import shutil
from html import escape
from pathlib import Path

from src.inference.predict_subject import predict_subject
from src.utils.config_loader import load_config, resolve_data_file, resolve_path
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)
_CONFIG_PATH = "configs/training.yaml"


def checkpoint_exists() -> bool:
    try:
        cfg = load_config(_CONFIG_PATH)
        checkpoints_dir = resolve_path(cfg["paths"]["checkpoints_dir"])
        return any(checkpoints_dir.glob("eegnet_fold*_best.pth"))
    except Exception:
        return False


def _subject_split_membership(subject_id: str) -> dict:
    try:
        cfg = load_config(_CONFIG_PATH)
        split_file = resolve_data_file(cfg, "split_file", "splits_dir")
        active_fold = int(cfg["data"].get("active_fold", 0))
        if not split_file.exists():
            return {"active_fold": active_fold, "split": "unknown"}
        import pandas as pd

        splits = pd.read_csv(split_file)
        match = splits[
            (splits["fold"].astype(int) == active_fold) &
            (splits["subject_id"].astype(str) == str(subject_id))
        ]
        split = str(match.iloc[0]["split"]) if len(match) else "unknown"
        return {"active_fold": active_fold, "split": split}
    except Exception:
        return {"active_fold": None, "split": "unknown"}


def _copy_visual_to_gradio_cache(path: str | None, subject_id: str) -> str | None:
    if not path:
        return None
    src_path = Path(path)
    if not src_path.exists():
        return None
    cache_dir = Path("outputs") / "gradio_cache" / str(subject_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst_path = cache_dir / src_path.name
    shutil.copy2(src_path, dst_path)
    return str(dst_path.resolve())


def _safe_pct(value) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "n/a"


def _safe_float(value, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


def _load_fold_metrics(active_fold: int | None) -> dict:
    reports_dir = Path("outputs") / "reports"
    crossval_path = reports_dir / "phase1_crossval_metrics.json"
    if not crossval_path.exists() or active_fold is None:
        return {}
    try:
        payload = json.loads(crossval_path.read_text(encoding="utf-8"))
        per_fold = payload.get("per_fold", [])
        fold_metrics = {}
        for item in per_fold:
            if int(item.get("fold")) == int(active_fold):
                fold_metrics = item.get("metrics", {})
                break
        aggregate = payload.get("aggregate_metrics", {})
        return {
            "fold": fold_metrics,
            "aggregate": aggregate,
        }
    except Exception:
        LOGGER.exception("Failed to load cross-validation metrics")
        return {}


def _metric_card(label: str, value: str, hint: str = "") -> str:
    return (
        "<div class='metric-card'>"
        f"<span>{escape(label)}</span>"
        f"<strong>{escape(value)}</strong>"
        f"<small>{escape(hint)}</small>"
        "</div>"
    )


def _build_summary_html(report: dict, demo_split: dict) -> str:
    pred = report["subject_level_prediction"]
    prep = report["preprocessing_summary"]
    emb = report["embedding_output"]
    meta = report["input_metadata"]
    fold = demo_split.get("active_fold")
    metrics = _load_fold_metrics(fold)
    fold_metrics = metrics.get("fold", {})
    aggregate = metrics.get("aggregate", {})
    prediction = str(pred["prediction"])
    is_uncertain = bool(pred.get("is_uncertain", False))
    tone = "uncertain" if is_uncertain else ("ad" if int(pred.get("predicted_label", 0)) == 1 else "hc")
    real_data = bool(meta.get("real_data", not meta.get("synthetic_data", False)))

    cards = "".join(
        [
            _metric_card("Prediction", prediction, "subject-level output"),
            _metric_card("Confidence", _safe_pct(pred.get("subject_level_confidence")), "aggregated probability"),
            _metric_card("AD Probability", _safe_pct(pred.get("ad_eeg_pattern_probability")), f"threshold {_safe_float(pred.get('decision_threshold'), 4)}"),
            _metric_card("Signal Quality", str(prep.get("signal_quality", "n/a")), f"{prep.get('clean_epochs_used')} clean epochs"),
            _metric_card("Fold Accuracy", _safe_pct(fold_metrics.get("accuracy")), f"fold {fold} validation"),
            _metric_card("CV AUC", _safe_pct((aggregate.get("auc") or {}).get("mean")), "all-fold mean"),
            _metric_card("Embedding", _safe_float(emb.get("embedding_consistency")), "consistency score"),
            _metric_card("Split", f"fold {fold} / {demo_split.get('split')}", "subject membership"),
        ]
    )
    return f"""
    <div class="result-shell">
      <div class="hero {tone}">
        <div>
          <div class="eyebrow">EEG AD/HC Phase 1</div>
          <h2>{escape(str(meta.get("subject_id", "subject")))}</h2>
          <p>{escape(prediction)}</p>
        </div>
        <div class="hero-badge">
          <span>{'Uncertain' if is_uncertain else 'Result'}</span>
          <strong>{_safe_pct(pred.get("subject_level_confidence"))}</strong>
        </div>
      </div>
      <div class="metric-grid">{cards}</div>
      <div class="note-row">
        <span>Real data: <strong>{real_data}</strong></span>
        <span>Risk level: <strong>{escape(str(pred.get("risk_level", "n/a")))}</strong></span>
        <span>Margin: <strong>{_safe_float(pred.get("margin_from_threshold"), 4)}</strong></span>
        <span>Epoch rejection: <strong>{_safe_pct(prep.get("rejection_rate"))}</strong></span>
      </div>
    </div>
    """


def _build_details_html(report: dict, demo_split: dict) -> str:
    pred = report["subject_level_prediction"]
    prep = report["preprocessing_summary"]
    epoch = report["epoch_probability_summary"]
    artifact = report.get("model_artifact", {})
    ica = prep.get("ica_summary", {})
    return f"""
    <div class="details-panel">
      <details open>
        <summary>View more subject-level details</summary>
        <div class="details-grid">
          <div><b>Checkpoint</b><span>{escape(str(artifact.get("checkpoint", "n/a")))}</span></div>
          <div><b>Fold</b><span>{escape(str(artifact.get("fold", demo_split.get("active_fold"))))}</span></div>
          <div><b>Threshold source</b><span>{escape(str(pred.get("threshold_source", "n/a")))}</span></div>
          <div><b>Uncertainty margin</b><span>{_safe_float(pred.get("uncertainty_margin"), 4)}</span></div>
          <div><b>Epoch probability std</b><span>{_safe_float(epoch.get("std_ad_probability"), 4)}</span></div>
          <div><b>Epoch min / max AD</b><span>{_safe_float(epoch.get("min_ad_probability"), 4)} / {_safe_float(epoch.get("max_ad_probability"), 4)}</span></div>
          <div><b>ICA</b><span>{'applied' if ica.get('applied') else 'not applied'}; excluded {escape(str(ica.get('excluded_components', [])))}</span></div>
          <div><b>Clean ratio</b><span>{_safe_pct(prep.get("clean_epoch_ratio"))}</span></div>
        </div>
      </details>
    </div>
    """


def _json_tree_html(value, level: int = 0) -> str:
    if isinstance(value, dict):
        rows = []
        for key, item in value.items():
            rows.append(
                "<div class='json-row'>"
                f"<span class='json-key'>{escape(str(key))}</span>"
                f"<span class='json-val'>{_json_tree_html(item, level + 1)}</span>"
                "</div>"
            )
        return "<div class='json-object'>" + "".join(rows) + "</div>"
    if isinstance(value, list):
        if len(value) > 12:
            preview = value[:12] + [f"... {len(value) - 12} more"]
        else:
            preview = value
        return "<span class='json-list'>" + escape(json.dumps(preview, ensure_ascii=False)) + "</span>"
    return "<code>" + escape(json.dumps(value, ensure_ascii=False)) + "</code>"


def _write_report_files(report: dict, subject_id: str) -> tuple[str, str]:
    cache_dir = Path("outputs") / "gradio_cache" / str(subject_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    json_path = cache_dir / f"{subject_id}_result.json"
    html_path = cache_dir / f"{subject_id}_result.html"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    pred = report["subject_level_prediction"]
    prep = report["preprocessing_summary"]
    epoch = report["epoch_probability_summary"]
    meta = report["input_metadata"]
    emb = report["embedding_output"]
    artifact = report["model_artifact"]
    protocol = report["evaluation_protocol"]
    visuals = report.get("visual_outputs", {})
    confidence = _safe_pct(pred.get("subject_level_confidence"))
    ad_probability = _safe_pct(pred.get("ad_eeg_pattern_probability"))
    hc_probability = _safe_pct((pred.get("class_probabilities") or {}).get("Healthy Control"))
    prediction = str(pred.get("prediction", "n/a"))
    predicted_label = int(pred.get("predicted_label", 0))
    tone_class = "ad" if predicted_label == 1 else "hc"
    if pred.get("is_uncertain"):
        tone_class = "uncertain"

    def metric(label: str, value: str, hint: str = "") -> str:
        return (
            "<div class='metric'>"
            f"<span>{escape(label)}</span>"
            f"<strong>{escape(value)}</strong>"
            f"<small>{escape(hint)}</small>"
            "</div>"
        )

    def row(label: str, value) -> str:
        return (
            "<tr>"
            f"<th>{escape(label)}</th>"
            f"<td>{escape(str(value))}</td>"
            "</tr>"
        )

    def image_tile(title: str, path: str | None) -> str:
        if not path:
            return ""
        filename = Path(path).name
        local_copy = cache_dir / filename
        image_path = local_copy if local_copy.exists() else Path(path)
        if not image_path.exists():
            return ""
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return (
            "<figure class='plot'>"
            f"<figcaption>{escape(title)}</figcaption>"
            f"<img src='data:{escape(mime_type)};base64,{encoded}' alt='{escape(title)}'>"
            "</figure>"
        )

    metric_cards = "".join(
        [
            metric("Prediction", prediction, "subject-level model output"),
            metric("Confidence", confidence, "aggregated across clean epochs"),
            metric("AD probability", ad_probability, f"threshold {_safe_float(pred.get('decision_threshold'), 4)}"),
            metric("HC probability", hc_probability, "complementary class probability"),
            metric("Signal quality", str(prep.get("signal_quality", "n/a")), f"{prep.get('clean_epochs_used')} clean epochs"),
            metric("Embedding", _safe_float(emb.get("embedding_consistency")), "epoch embedding consistency"),
            metric("Risk level", str(pred.get("risk_level", "n/a")), f"margin {_safe_float(pred.get('margin_from_threshold'), 4)}"),
            metric("Epoch rejection", _safe_pct(prep.get("rejection_rate")), f"{prep.get('rejected_epochs')} rejected epochs"),
        ]
    )

    quality_rows = "".join(
        [
            row("Input file", meta.get("input_file", "n/a")),
            row("Sampling rate", f"{meta.get('actual_sampling_rate_hz', 'n/a')} Hz"),
            row("Duration", f"{_safe_float(meta.get('duration_seconds'), 1)} seconds"),
            row("Channels", meta.get("number_of_channels", "n/a")),
            row("Clean epochs", prep.get("clean_epochs_used", "n/a")),
            row("Total epochs", prep.get("total_epochs_generated", "n/a")),
            row("Clean epoch ratio", _safe_pct(prep.get("clean_epoch_ratio"))),
            row("ICA", "applied" if (prep.get("ica_summary") or {}).get("applied") else "not applied"),
        ]
    )

    model_rows = "".join(
        [
            row("Model", artifact.get("model_name", "n/a")),
            row("Checkpoint", artifact.get("checkpoint", "n/a")),
            row("Fold", artifact.get("fold", "n/a")),
            row("Split strategy", protocol.get("split_strategy", "n/a")),
            row("No epoch leakage", protocol.get("no_epoch_leakage", "n/a")),
            row("Threshold source", pred.get("threshold_source", "n/a")),
            row("Uncertainty margin", pred.get("uncertainty_margin", "n/a")),
            row("Aggregation", pred.get("aggregation_method", "n/a")),
        ]
    )

    epoch_rows = "".join(
        [
            row("Epochs used", epoch.get("epochs_used_for_prediction", "n/a")),
            row("Mean AD probability", epoch.get("mean_ad_probability", "n/a")),
            row("Std AD probability", epoch.get("std_ad_probability", "n/a")),
            row("Min AD probability", epoch.get("min_ad_probability", "n/a")),
            row("Max AD probability", epoch.get("max_ad_probability", "n/a")),
        ]
    )

    plot_tiles = "".join(
        [
            image_tile("Raw EEG", visuals.get("raw_eeg_plot")),
            image_tile("Filtered EEG", visuals.get("filtered_eeg_plot")),
            image_tile("Class Probability", visuals.get("probability_bar_chart")),
            image_tile("Epoch Probability Distribution", visuals.get("epoch_probability_distribution")),
        ]
    )

    raw_json = escape(json.dumps(report, indent=2))
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(subject_id)} EEG Result</title>
  <style>
    :root {{
      --ink:#172033;
      --muted:#667085;
      --line:#d8dee9;
      --soft:#f6f8fb;
      --teal:#0f766e;
      --rose:#be123c;
      --amber:#b45309;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0;
      font-family:Segoe UI, Arial, sans-serif;
      background:var(--soft);
      color:var(--ink);
      line-height:1.45;
    }}
    main {{ max-width:1180px; margin:0 auto; padding:32px 22px 48px; }}
    .report {{
      background:white;
      border:1px solid var(--line);
      border-radius:14px;
      overflow:hidden;
      box-shadow:0 16px 40px rgba(15,23,42,.09);
    }}
    .top {{
      display:flex;
      justify-content:space-between;
      gap:24px;
      padding:28px 32px;
      color:white;
      background:linear-gradient(120deg,#0f766e,#172033);
    }}
    .top.ad {{ background:linear-gradient(120deg,#991b1b,#be123c); }}
    .top.uncertain {{ background:linear-gradient(120deg,#92400e,#334155); }}
    .eyebrow {{
      font-size:12px;
      font-weight:800;
      letter-spacing:.08em;
      text-transform:uppercase;
      opacity:.86;
    }}
    h1 {{ margin:5px 0 6px; font-size:34px; line-height:1.1; }}
    h2 {{ margin:0 0 14px; font-size:20px; }}
    p {{ margin:0; }}
    .subtle {{ color:#dbeafe; }}
    .score {{ text-align:right; min-width:150px; }}
    .score span {{ display:block; font-weight:800; opacity:.88; }}
    .score strong {{ display:block; font-size:38px; line-height:1; margin-top:8px; }}
    .section {{ padding:24px 32px; border-top:1px solid #edf1f7; }}
    .grid {{
      display:grid;
      grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
      gap:14px;
    }}
    .metric {{
      border:1px solid #e3e8ef;
      background:#f8fafc;
      border-radius:10px;
      padding:14px;
      min-height:102px;
    }}
    .metric span {{ display:block; color:#53657d; font-size:12px; font-weight:800; }}
    .metric strong {{ display:block; margin-top:6px; font-size:22px; line-height:1.15; }}
    .metric small {{ display:block; margin-top:6px; color:var(--muted); }}
    .status-row {{
      display:flex;
      flex-wrap:wrap;
      gap:10px;
      margin-top:16px;
    }}
    .pill {{
      border:1px solid #99f6e4;
      background:#ecfdf5;
      color:#134e4a;
      border-radius:8px;
      padding:7px 11px;
      font-weight:750;
    }}
    .two-col {{
      display:grid;
      grid-template-columns:repeat(auto-fit,minmax(340px,1fr));
      gap:18px;
    }}
    .panel {{
      border:1px solid #e3e8ef;
      border-radius:12px;
      overflow:hidden;
      background:#fff;
    }}
    .panel h2 {{
      padding:15px 16px;
      background:#f8fafc;
      border-bottom:1px solid #e3e8ef;
    }}
    table {{ width:100%; border-collapse:collapse; table-layout:fixed; }}
    th,td {{
      padding:10px 14px;
      border-bottom:1px solid #edf1f7;
      vertical-align:top;
      text-align:left;
      word-break:break-word;
    }}
    th {{ width:34%; color:#164e63; font-size:13px; }}
    td {{ color:#27364a; font-size:13px; }}
    .plots {{
      display:grid;
      grid-template-columns:repeat(auto-fit,minmax(360px,1fr));
      gap:16px;
    }}
    .plot {{
      margin:0;
      border:1px solid #e3e8ef;
      background:#fff;
      border-radius:12px;
      overflow:hidden;
    }}
    .plot figcaption {{
      padding:10px 12px;
      background:#172033;
      color:white;
      font-weight:800;
    }}
    .plot img {{
      display:block;
      width:100%;
      height:auto;
      background:white;
    }}
    details {{
      border:1px solid #e3e8ef;
      border-radius:12px;
      background:#fff;
      overflow:hidden;
    }}
    summary {{
      cursor:pointer;
      padding:15px 16px;
      background:#172033;
      color:white;
      font-weight:800;
    }}
    pre {{
      margin:0;
      padding:18px;
      overflow:auto;
      max-height:640px;
      background:#0f172a;
      color:#e5eefb;
      font-size:12px;
      line-height:1.55;
    }}
    .disclaimer {{
      margin-top:16px;
      color:#344054;
      font-size:13px;
    }}
    @media (max-width:720px) {{
      main {{ padding:18px 10px 34px; }}
      .top {{ display:block; padding:24px 20px; }}
      .score {{ text-align:left; margin-top:18px; }}
      .section {{ padding:20px; }}
      h1 {{ font-size:28px; }}
    }}
  </style>
</head>
<body>
  <main>
    <div class="report">
      <section class="top {tone_class}">
        <div>
          <div class="eyebrow">EEG AD/HC Phase 1 Research Report</div>
          <h1>{escape(subject_id)}</h1>
          <p class="subtle">{escape(prediction)}</p>
        </div>
        <div class="score">
          <span>Subject confidence</span>
          <strong>{confidence}</strong>
        </div>
      </section>

      <section class="section">
        <div class="grid">{metric_cards}</div>
        <div class="status-row">
          <span class="pill">Real data: {escape(str(meta.get("real_data", "n/a")))}</span>
          <span class="pill">Input status: {escape(str(meta.get("input_status", "n/a")))}</span>
          <span class="pill">No epoch leakage: {escape(str(protocol.get("no_epoch_leakage", "n/a")))}</span>
        </div>
      </section>

      <section class="section two-col">
        <div class="panel">
          <h2>Input And Preprocessing</h2>
          <table>{quality_rows}</table>
        </div>
        <div class="panel">
          <h2>Model And Decision</h2>
          <table>{model_rows}</table>
        </div>
      </section>

      <section class="section">
        <div class="panel">
          <h2>Epoch Probability Summary</h2>
          <table>{epoch_rows}</table>
        </div>
      </section>

      <section class="section">
        <h2>Visual Outputs</h2>
        <div class="plots">{plot_tiles}</div>
      </section>

      <section class="section">
        <details>
          <summary>Full JSON Report</summary>
          <pre>{raw_json}</pre>
        </details>
        <p class="disclaimer"><strong>Clinical disclaimer:</strong> {escape(str(report.get("clinical_disclaimer", "")))}</p>
      </section>
    </div>
  </main>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    return str(json_path.resolve()), str(html_path.resolve())


def run_demo_subject(subject_id: str):
    """Run the EEG analysis pipeline and return Gradio-friendly outputs."""
    if not checkpoint_exists():
        message = (
            f"No trained model checkpoint found for subject {subject_id}.\n\n"
            "Run training first:\n"
            "  python -m src.training.train_eegnet --config configs/training.yaml --fold 2\n\n"
            "Then restart the Gradio demo."
        )
        return None, None, None, None, f"<div class='error-box'>{escape(message)}</div>", "", "{}", None, None

    try:
        report = predict_subject(subject_id, auto_train=False)
    except Exception as exc:
        LOGGER.exception("Demo pipeline failed for subject %s", subject_id)
        no_image = None
        error_summary = (
            f"Analysis failed for {subject_id}.\n"
            f"Error: {type(exc).__name__}: {exc}\n\n"
            "Check that preprocessing has been run and a model checkpoint exists.\n"
            "See logs for the full traceback."
        )
        return no_image, no_image, no_image, no_image, f"<div class='error-box'>{escape(error_summary)}</div>", "", "{}", None, None

    visuals = report["visual_outputs"]
    demo_split = _subject_split_membership(subject_id)

    # Gradio only allows serving files from cwd or temp (unless allowed_paths is set).
    # Copy Modal volume artifacts into a cwd-local cache for reliable display.
    raw_plot_path = _copy_visual_to_gradio_cache(visuals.get("raw_eeg_plot"), subject_id)
    filtered_plot_path = _copy_visual_to_gradio_cache(visuals.get("filtered_eeg_plot"), subject_id)
    prob_chart_path = _copy_visual_to_gradio_cache(visuals.get("probability_bar_chart"), subject_id)
    epoch_dist_path = _copy_visual_to_gradio_cache(visuals.get("epoch_probability_distribution"), subject_id)

    summary_html = _build_summary_html(report, demo_split)
    details_html = _build_details_html(report, demo_split)
    json_file, html_file = _write_report_files(report, subject_id)
    return (
        raw_plot_path,
        filtered_plot_path,
        prob_chart_path,
        epoch_dist_path,
        summary_html,
        details_html,
        json.dumps(report, indent=2),
        json_file,
        html_file,
    )
