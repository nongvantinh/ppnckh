"""Xuất kết quả chạy notebook ra thư mục cố định (CSV/JSON/hình) để báo cáo / LaTeX."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _flatten_summary(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out


def save_run_outputs(
    repo_root: Path,
    *,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    display_tbl: pd.DataFrame | None = None,
    ci_f1: pd.DataFrame | None = None,
    wilcoxon_summary: str = "",
    ablation_lr_f1: pd.DataFrame | None = None,
    lr_coefs: pd.Series | None = None,
    rf_importance: pd.Series | None = None,
    model_pkl_relative: str = "resources/diabetes_prediction_model.pkl",
    extra_manifest: dict[str, Any] | None = None,
) -> Path:
    """
    Ghi artifact vào ``resources/outputs/latest/`` (ghi đè mỗi lần chạy) và
    bản lưu theo timestamp ``resources/outputs/runs/YYYYMMDD_HHMMSS/``.

    Trả về đường dẫn thư mục ``latest``.
    """
    repo_root = repo_root.resolve()
    out_latest = repo_root / "resources" / "outputs" / "latest"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_run = repo_root / "resources" / "outputs" / "runs" / ts

    for d in (out_latest, out_run):
        d.mkdir(parents=True, exist_ok=True)
        (d / "figures").mkdir(exist_ok=True)

    results.to_csv(out_latest / "results_by_split.csv", index=False)
    _flatten_summary(summary).to_csv(out_latest / "summary_mean_std.csv")
    if display_tbl is not None:
        display_tbl.to_csv(out_latest / "metrics_mean_pm_std_formatted.csv")
    if ci_f1 is not None:
        ci_f1.to_csv(out_latest / "f1_ci95_by_model.csv", index=False)
    if ablation_lr_f1 is not None:
        ablation_lr_f1.to_csv(out_latest / "ablation_lr_scaler.csv", index=False)
    if lr_coefs is not None:
        lr_coefs.to_csv(out_latest / "lr_coefficients_scaled_space.csv", header=["value"])
    if rf_importance is not None:
        rf_importance.to_csv(out_latest / "rf_feature_importance.csv", header=["value"])
    if wilcoxon_summary.strip():
        (out_latest / "wilcoxon_f1.txt").write_text(wilcoxon_summary.strip() + "\n", encoding="utf-8")

    image_dir = repo_root / "image"
    if image_dir.is_dir():
        for src in sorted(image_dir.glob("roc_confusion*.png")):
            shutil.copy2(src, out_latest / "figures" / src.name)

    manifest: dict[str, Any] = {
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "model_pkl": model_pkl_relative,
    }
    try:
        import sklearn

        manifest["sklearn"] = sklearn.__version__
    except ImportError:
        pass
    try:
        import numpy as np

        manifest["numpy"] = np.__version__
    except ImportError:
        pass
    if extra_manifest:
        manifest.update(extra_manifest)

    (out_latest / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Snapshot giống latest vào runs/<ts>/
    for name in out_latest.iterdir():
        if name.is_dir():
            dest = out_run / name.name
            shutil.copytree(name, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(name, out_run / name.name)

    return out_latest
