from __future__ import annotations

import argparse
import csv
import json
from html import escape
from pathlib import Path

import yaml


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _fmt(value, digits: int = 3) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _pct(value) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(value)


def _overfit_level(auc_gap: float, acc_gap: float, loss_gap: float) -> tuple[str, str]:
    if auc_gap < 0.08 and acc_gap < 0.12 and loss_gap < 0.20:
        return "Low", "ok"
    if auc_gap < 0.15 and acc_gap < 0.25 and loss_gap < 0.40:
        return "Mild", "warn"
    if auc_gap < 0.25 and acc_gap < 0.35:
        return "Moderate", "risk"
    return "High", "bad"


def _config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _fold_training(root: Path, fold: int) -> dict:
    summary = _load_json(root / "outputs" / "reports" / f"fold{fold}_training_summary.json") or {}
    history = _load_csv(root / "outputs" / "reports" / f"fold{fold}_training_history.csv")
    best = summary.get("best_metrics") or {}
    last = summary.get("last_metrics") or {}
    auc_gap = float(best.get("train_subject_auc", 0) or 0) - float(best.get("val_subject_auc", 0) or 0)
    acc_gap = float(best.get("train_subject_accuracy", 0) or 0) - float(best.get("val_subject_accuracy", 0) or 0)
    loss_gap = float(best.get("val_loss", 0) or 0) - float(best.get("train_loss", 0) or 0)
    overfit, badge = _overfit_level(auc_gap, acc_gap, loss_gap)
    return {
        "fold": fold,
        "summary": summary,
        "history": history,
        "best": best,
        "last": last,
        "auc_gap": auc_gap,
        "acc_gap": acc_gap,
        "loss_gap": loss_gap,
        "overfit": overfit,
        "badge": badge,
        "curve": f"../figures/fold{fold}_training_curve.png",
        "cm": f"../figures/fold{fold}_confusion_matrix.png",
        "roc": f"../figures/fold{fold}_roc_curve.png",
    }


def _cv_metrics(root: Path) -> dict:
    return _load_json(root / "outputs" / "reports" / "phase1_crossval_metrics.json") or {}


def _inference_10(root: Path) -> dict:
    return _load_json(root / "outputs" / "reports" / "inference_10_subjects_folds_0_4.json") or {}


def _per_fold_cv(cv: dict) -> dict[int, dict]:
    result = {}
    for item in cv.get("per_fold", []):
        result[int(item["fold"])] = item.get("metrics", {})
    return result


def _best_fold(folds: list[dict], cv_by_fold: dict[int, dict]) -> int | None:
    if not folds:
        return None
    scored = []
    for fold in folds:
        fid = int(fold["fold"])
        cv = cv_by_fold.get(fid, {})
        score = (
            0.35 * float(cv.get("auc", 0) or 0)
            + 0.25 * float(cv.get("accuracy", 0) or 0)
            + 0.20 * float(cv.get("balanced_accuracy", 0) or 0)
            + 0.10 * float(cv.get("f1", 0) or 0)
            + 0.10 * max(float(cv.get("embedding_silhouette", 0) or 0), 0)
        )
        scored.append((score, fid))
    scored.sort(reverse=True)
    return scored[0][1]


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build_dashboard(config_path: str = "configs/training.yaml") -> Path:
    root = Path.cwd()
    cfg = _config(root / config_path)
    reports_dir = root / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    n_splits = int(cfg.get("data", {}).get("n_splits", 5))
    active_fold = int(cfg.get("data", {}).get("active_fold", 0))
    folds = [_fold_training(root, fold) for fold in range(n_splits)]
    cv = _cv_metrics(root)
    cv_by_fold = _per_fold_cv(cv)
    inf10 = _inference_10(root)
    best_fold = _best_fold(folds, cv_by_fold)
    aggregate = cv.get("aggregate_metrics", {})

    fold_rows = []
    for fold in folds:
        fid = fold["fold"]
        cvm = cv_by_fold.get(fid, {})
        best = fold["best"]
        best_marker = " <span class='pill best'>Best</span>" if fid == best_fold else ""
        active_marker = " <span class='pill active'>Gradio</span>" if fid == active_fold else ""
        fold_rows.append([
            f"Fold {fid}{best_marker}{active_marker}",
            _pct(cvm.get("accuracy")),
            _pct(cvm.get("balanced_accuracy")),
            _pct(cvm.get("auc")),
            _pct(cvm.get("f1")),
            _fmt(cvm.get("embedding_silhouette")),
            _fmt(best.get("epoch"), 0),
            _pct(best.get("val_subject_auc")),
            _pct(best.get("val_subject_accuracy")),
            f"<span class='pill {fold['badge']}'>{fold['overfit']}</span>",
        ])

    overfit_rows = []
    for fold in folds:
        overfit_rows.append([
            f"Fold {fold['fold']}",
            _pct(fold["best"].get("train_subject_auc")),
            _pct(fold["best"].get("val_subject_auc")),
            _fmt(fold["auc_gap"]),
            _pct(fold["best"].get("train_subject_accuracy")),
            _pct(fold["best"].get("val_subject_accuracy")),
            _fmt(fold["acc_gap"]),
            _fmt(fold["loss_gap"]),
            f"<span class='pill {fold['badge']}'>{fold['overfit']}</span>",
        ])

    inference_rows = []
    for fold, metrics in sorted((inf10.get("per_fold_metrics") or {}).items(), key=lambda x: int(x[0])):
        inference_rows.append([
            f"Fold {fold}",
            str(metrics.get("n", "n/a")),
            _pct(metrics.get("accuracy")),
            _pct(metrics.get("balanced_accuracy")),
            _pct(metrics.get("f1")),
            _pct(metrics.get("precision")),
            _pct(metrics.get("recall")),
        ])

    fold_sections = []
    for fold in folds:
        fid = fold["fold"]
        cvm = cv_by_fold.get(fid, {})
        fold_sections.append(f"""
        <section class="fold-panel">
          <div class="fold-title">
            <h3>Fold {fid}</h3>
            <span class="pill {fold['badge']}">{fold['overfit']} overfit risk</span>
          </div>
          <div class="metric-strip">
            <div><span>CV Accuracy</span><strong>{_pct(cvm.get("accuracy"))}</strong></div>
            <div><span>CV AUC</span><strong>{_pct(cvm.get("auc"))}</strong></div>
            <div><span>Best Epoch</span><strong>{_fmt(fold["best"].get("epoch"), 0)}</strong></div>
            <div><span>Val Loss</span><strong>{_fmt(fold["best"].get("val_loss"))}</strong></div>
          </div>
          <div class="images">
            <figure><img src="{escape(fold['curve'])}" alt="Fold {fid} training curve"><figcaption>Training curve</figcaption></figure>
            <figure><img src="{escape(fold['cm'])}" alt="Fold {fid} confusion matrix"><figcaption>Confusion matrix</figcaption></figure>
            <figure><img src="{escape(fold['roc'])}" alt="Fold {fid} ROC curve"><figcaption>ROC curve</figcaption></figure>
          </div>
        </section>
        """)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EEG AD/HC Training Dashboard</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --ink: #1b2430;
      --muted: #667085;
      --line: #d8dee9;
      --panel: #ffffff;
      --blue: #2563eb;
      --green: #15803d;
      --amber: #b45309;
      --red: #b42318;
      --cyan: #0e7490;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "Segoe UI", Arial, sans-serif;
      line-height: 1.5;
    }}
    header {{
      background: linear-gradient(120deg, #0f172a, #164e63 55%, #1e40af);
      color: white;
      padding: 48px 36px;
    }}
    header h1 {{ margin: 0 0 10px; font-size: 34px; letter-spacing: 0; }}
    header p {{ max-width: 980px; margin: 0; color: #dbeafe; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 28px; }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 22px;
      margin: 0 0 22px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }}
    h2, h3 {{ margin: 0 0 14px; letter-spacing: 0; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 14px;
      margin-top: -48px;
    }}
    .stat {{
      background: white;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
    }}
    .stat span {{ display: block; color: var(--muted); font-size: 13px; }}
    .stat strong {{ display: block; font-size: 28px; margin-top: 6px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      overflow: hidden;
    }}
    th, td {{
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      vertical-align: middle;
    }}
    th {{ background: #eef4ff; color: #243b53; font-weight: 650; }}
    tr:hover td {{ background: #f8fafc; }}
    .pill {{
      display: inline-block;
      border-radius: 999px;
      padding: 3px 9px;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }}
    .best {{ background: #dcfce7; color: var(--green); }}
    .active {{ background: #dbeafe; color: var(--blue); }}
    .ok {{ background: #dcfce7; color: var(--green); }}
    .warn {{ background: #fef3c7; color: var(--amber); }}
    .risk {{ background: #ffedd5; color: #c2410c; }}
    .bad {{ background: #fee2e2; color: var(--red); }}
    .fold-title {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; }}
    .metric-strip {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 10px;
      margin: 10px 0 18px;
    }}
    .metric-strip div {{
      background: #f8fafc;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 12px;
    }}
    .metric-strip span {{ display: block; color: var(--muted); font-size: 12px; }}
    .metric-strip strong {{ display: block; font-size: 20px; margin-top: 3px; }}
    .images {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 14px;
    }}
    figure {{ margin: 0; }}
    img {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      display: block;
    }}
    figcaption {{ color: var(--muted); font-size: 12px; margin-top: 6px; }}
    .note {{
      border-left: 4px solid var(--cyan);
      background: #ecfeff;
      padding: 12px 14px;
      border-radius: 0 8px 8px 0;
      color: #164e63;
    }}
    footer {{ color: var(--muted); padding: 24px 0 10px; font-size: 13px; }}
  </style>
</head>
<body>
  <header>
    <h1>EEG AD/HC Phase 1 Training Dashboard</h1>
    <p>Fold-wise training, validation metrics, overfitting check, and best single-fold model selection for the EEGNet baseline.</p>
  </header>
  <main>
    <div class="summary">
      <div class="stat"><span>Best single fold</span><strong>Fold {best_fold if best_fold is not None else "n/a"}</strong></div>
      <div class="stat"><span>Gradio active fold</span><strong>Fold {active_fold}</strong></div>
      <div class="stat"><span>CV accuracy mean</span><strong>{_pct((aggregate.get("accuracy") or {}).get("mean"))}</strong></div>
      <div class="stat"><span>CV AUC mean</span><strong>{_pct((aggregate.get("auc") or {}).get("mean"))}</strong></div>
      <div class="stat"><span>CV F1 mean</span><strong>{_pct((aggregate.get("f1") or {}).get("mean"))}</strong></div>
    </div>

    <section>
      <h2>Final Recommendation</h2>
      <p class="note">Use fold {best_fold if best_fold is not None else "n/a"} as the best single demo checkpoint, but report all-fold cross-validation mean and standard deviation for final performance. Gradio currently follows <code>configs/training.yaml -> data.active_fold</code>.</p>
    </section>

    <section>
      <h2>Fold Comparison</h2>
      {_html_table(["Fold", "Accuracy", "Balanced Acc", "AUC", "F1", "Silhouette", "Best Epoch", "Best Val AUC", "Best Val Acc", "Overfit"], fold_rows)}
    </section>

    <section>
      <h2>Overfitting Review</h2>
      {_html_table(["Fold", "Train AUC", "Val AUC", "AUC Gap", "Train Acc", "Val Acc", "Acc Gap", "Loss Gap", "Risk"], overfit_rows)}
    </section>

    <section>
      <h2>10-Subject Inference Test</h2>
      <p>Generated from <code>outputs/reports/inference_10_subjects_folds_0_4.json</code>.</p>
      {_html_table(["Fold", "N", "Accuracy", "Balanced Acc", "F1", "Precision", "Recall"], inference_rows)}
    </section>

    {"".join(fold_sections)}

    <footer>
      Generated by <code>python -m scripts.build_training_dashboard --config configs/training.yaml</code>.
      This dashboard is for research review only and is not a clinical report.
    </footer>
  </main>
</body>
</html>
"""
    out = reports_dir / "training_dashboard.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training.yaml")
    args = parser.parse_args()
    path = build_dashboard(args.config)
    print(path)


if __name__ == "__main__":
    main()
