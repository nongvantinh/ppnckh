#!/usr/bin/env python3
"""Generate resources/diabetes.ipynb aligned with main.tex methodology."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "resources" / "diabetes.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells: list[dict] = []

cells.append(
    md(
        """# Dự đoán tiểu đường (Pima Indians) — notebook khớp `main.tex`

Notebook này triển khai **đúng giao thức** mô tả trong báo cáo LaTeX (`main.tex`):

- **Dữ liệu:** Pima Indians Diabetes (UCI), không dùng SMOTE/làm giàu **trước** khi chia tập.
- **Mất cân bằng lớp:** chỉ **stratify** khi `train_test_split` và khi CV; `class_weight=None` (có thể đổi thành `'balanced'` trong lưới nếu muốn — ghi rõ trong Phụ lục).
- **Chống leakage:** `SimpleImputer` + `StandardScaler` nằm trong `Pipeline`, chỉ `fit` trên **train** hoặc **train fold** (qua `GridSearchCV`).
- **Selection vs evaluation:** `GridSearchCV` (vòng trong, 5-fold stratified) chỉ trên tập train; đánh giá trên **test hold-out 20%** (lặp với nhiều `random_state` split để có mean±std).
- **Mô hình:** Logistic Regression, SVM (RBF), Random Forest, KNN + baseline **Majority** (`DummyClassifier`).
- **Chỉ số:** Accuracy, Precision, Recall, F1, ROC–AUC; tùy chọn kiểm định cặp (Wilcoxon) và khoảng tin cậy 95% trên các lần split.
- **RQ2:** hệ số LR (đã chuẩn hóa), `feature_importances_` của RF; SHAP nếu cài extra `shap` (`uv sync --extra shap`).
- **Đặc trưng:** dùng đủ 8 biến gốc của Pima; không PCA / không loại biến theo tương quan (khớp mục Lựa chọn đặc trưng trong `main.tex` khi ghi ``giữ toàn bộ đặc trưng'')."""
    )
)

cells.append(
    md(
        """## Môi trường (`uv`)

Từ thư mục gốc repo (`ppnckh/`):

```bash
uv sync
uv run jupyter lab resources/diabetes.ipynb
```

Hoặc chỉ chạy toàn bộ notebook:

```bash
uv run jupyter execute resources/diabetes.ipynb
```"""
    )
)

cells.append(
    code(
        """from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)

# --- Tái lập (khớp Phụ lục D / main.tex) ---
RANDOM_STATE_CV = 42       # StratifiedKFold + GridSearchCV
SEED_MODEL = 42            # LR, SVM, RF
SPLIT_SEEDS = [0, 1, 2, 3, 4]  # Độ ổn định qua nhiều lần chia train/test (cùng tỷ lệ 80/20)
TEST_SIZE = 0.2
N_INNER_SPLITS = 5

print("Python", sys.version.split()[0])
print("NumPy", np.__version__)
print("pandas", pd.__version__)
import sklearn

print("scikit-learn", sklearn.__version__)

# Jupyter: không có __file__ — tìm thư mục gốc repo (có main.tex)
def _repo_root() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "main.tex").is_file():
        return cwd
    if (cwd.parent / "main.tex").is_file():
        return cwd.parent
    return cwd


REPO_ROOT = _repo_root()
DATA_DIR = REPO_ROOT / "resources" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = DATA_DIR / "pima-indians-diabetes.data.csv"
# UCI / Jason Brownlee mirror (8 đặc trưng + nhãn, không header)
DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

COLS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

# Pima: giá trị 0 thường là thiếu cho các biến sinh hóa (trừ Pregnancies & Outcome)
ZERO_TO_NA = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]"""
    )
)

cells.append(
    code(
        """def load_pima(path: Path = DATA_PATH, url: str = DATA_URL) -> pd.DataFrame:
    if not path.exists():
        import urllib.request

        print("Đang tải", url, "->", path)
        urllib.request.urlretrieve(url, path)
    df = pd.read_csv(path, header=None, names=COLS)
    for c in ZERO_TO_NA:
        df.loc[df[c] == 0, c] = np.nan
    return df


diabetes = load_pima()
X = diabetes.drop(columns=["Outcome"])
y = diabetes["Outcome"].astype(int)

print("Shape", X.shape, "Positive rate", f"{y.mean():.3f}")
diabetes.head()"""
    )
)

cells.append(code("diabetes.describe()"))

cells.append(
    code(
        """fig, ax = plt.subplots(1, 2, figsize=(10, 4))
y.value_counts().plot(kind="bar", ax=ax[0], title="Outcome (0/1)")
ax[0].set_xlabel("Outcome")
sns.heatmap(diabetes.drop(columns=["Outcome"]).corr(), ax=ax[1], cmap="vlag", center=0)
ax[1].set_title("Correlation (sau khi 0->NaN)")
plt.tight_layout()
plt.show()"""
    )
)

cells.append(
    md(
        """## Pipeline & siêu tham số (khớp Phụ lục A / `main.tex`)

- Lưới dưới đây là **ví dụ hợp lý**; có thể thu hẹp khi cần chạy nhanh.
- Mọi bước `imputer` / `scaler` nằm trong `Pipeline` → không fit trên test."""
    )
)

cells.append(
    code(
        """def make_base_pipeline(classifier) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", classifier),
        ]
    )


inner_cv = StratifiedKFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE_CV)

MODEL_GRIDS: list[tuple[str, object, dict]] = [
    (
        "Logistic Regression",
        LogisticRegression(max_iter=10_000, random_state=SEED_MODEL, class_weight=None),
        {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__solver": ["lbfgs"],
        },
    ),
    (
        "SVM (RBF)",
        SVC(kernel="rbf", probability=True, random_state=SEED_MODEL, class_weight=None),
        {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", 0.01, 0.1]},
    ),
    (
        "Random Forest",
        RandomForestClassifier(random_state=SEED_MODEL, class_weight=None, n_jobs=-1),
        {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 8, 16],
            "clf__min_samples_leaf": [1, 2],
        },
    ),
    (
        "KNN",
        KNeighborsClassifier(),
        {
            "clf__n_neighbors": [3, 5, 7, 9],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [2],
        },
    ),
    (
        "Majority (baseline)",
        DummyClassifier(strategy="most_frequent"),
        {},  # không tune
    ),
]


def evaluate_split(X_tr, X_te, y_tr, y_te, name, base_estimator, param_grid, inner_cv):
    pipe = make_base_pipeline(base_estimator)
    if not param_grid:
        pipe.fit(X_tr, y_tr)
        best = pipe
    else:
        search = GridSearchCV(
            pipe,
            param_grid,
            cv=inner_cv,
            scoring="f1",
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
    y_pred = best.predict(X_te)
    proba = None
    if hasattr(best.named_steps["clf"], "predict_proba"):
        proba = best.predict_proba(X_te)[:, 1]
    elif hasattr(best.named_steps["clf"], "decision_function"):
        proba = best.decision_function(X_te)
    out = {
        "model": name,
        "accuracy": accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "recall": recall_score(y_te, y_pred, zero_division=0),
        "f1": f1_score(y_te, y_pred, zero_division=0),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_te, proba) if proba is not None else np.nan
    except ValueError:
        out["roc_auc"] = np.nan
    return out, best, y_pred, proba


rows = []
fitted_by_seed: dict[int, dict[str, object]] = {}

for split_seed in SPLIT_SEEDS:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=split_seed
    )
    fitted_by_seed[split_seed] = {}
    for name, est, grid in MODEL_GRIDS:
        m, fitted, _, _ = evaluate_split(
            X_train, X_test, y_train, y_test, name, clone(est), grid, inner_cv
        )
        m["split_seed"] = split_seed
        rows.append(m)
        fitted_by_seed[split_seed][name] = fitted

results = pd.DataFrame(rows)
summary = results.groupby("model")[["accuracy", "precision", "recall", "f1", "roc_auc"]].agg(["mean", "std"])
summary"""
    )
)

cells.append(
    code(
        """# Bảng mean ± std theo các split_seed (báo cáo trong main.tex)
def fmt_mean_std(g):
    m = g["mean"]
    s = g["std"]
    return (m.map(lambda x: f"{x:.4f}") + " ± " + s.map(lambda x: f"{x:.4f}")).rename("value")


display_tbl = pd.DataFrame(
    {
        c: fmt_mean_std(summary[c])
        for c in summary.columns.levels[0]
    }
)
display_tbl"""
    )
)

cells.append(
    md(
        """### Kiểm định cặp (tùy chọn): Wilcoxon trên F1 giữa hai mô hình theo từng `split_seed`

Chỉ mang tính minh họa: so sánh hai mô hình có F1 gần nhau trên cùng các split."""
    )
)

cells.append(
    code(
        """def wilcoxon_f1(model_a: str, model_b: str) -> tuple[float, float]:
    a = results.loc[results["model"] == model_a, "f1"].values
    b = results.loc[results["model"] == model_b, "f1"].values
    if len(a) != len(b):
        raise ValueError("length mismatch")
    stat, p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return stat, p


# Đổi tên mô hình nếu cần (phải khớp chuỗi trong MODEL_GRIDS)
best_two = summary["f1"]["mean"].sort_values(ascending=False).index[:2].tolist()
print("Hai mô hình F1 cao nhất (mean):", best_two)
wilcoxon_summary = ""
if len(best_two) == 2:
    stat, p = wilcoxon_f1(best_two[0], best_two[1])
    wilcoxon_summary = f"Models: {best_two[0]} vs {best_two[1]}\\nWilcoxon statistic={stat:.6f}, p-value={p:.6f}\\n"
    print(wilcoxon_summary.strip())"""
    )
)

cells.append(
    code(
        """# Khoảng tin cậy 95% (t-interval) cho mean F1 của từng mô hình qua len(SPLIT_SEEDS) split


def t_ci_95(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return float(np.mean(values)), float(np.mean(values))
    m = np.mean(values)
    sem = stats.sem(values)
    h = sem * stats.t.ppf((1 + 0.95) / 2.0, n - 1)
    return m - h, m + h


ci_rows = []
for name in results["model"].unique():
    vals = results.loc[results["model"] == name, "f1"].values
    lo, hi = t_ci_95(vals)
    ci_rows.append({"model": name, "f1_mean": np.mean(vals), "f1_ci95_low": lo, "f1_ci95_high": hi})

ci_f1_df = pd.DataFrame(ci_rows).sort_values("f1_mean", ascending=False)
ci_f1_df"""
    )
)

cells.append(
    md(
        """## Diễn giải đặc trưng (RQ2) — trên một split cố định (`split_seed` = phần tử đầu của `SPLIT_SEEDS`)

Hệ số logistic trên dữ liệu **đã impute + scale**; RF: `feature_importances_`. **Không** nhân quả."""
    )
)

cells.append(
    code(
        """REF_SEED = SPLIT_SEEDS[0]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=REF_SEED
)

lr_pipe = make_base_pipeline(
    LogisticRegression(max_iter=10_000, random_state=SEED_MODEL, class_weight=None)
)
lr_search = GridSearchCV(
    lr_pipe,
    {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__solver": ["lbfgs"]},
    cv=inner_cv,
    scoring="f1",
    n_jobs=-1,
)
lr_search.fit(X_train, y_train)
lr_best = lr_search.best_estimator_
coefs = pd.Series(
    lr_best.named_steps["clf"].coef_.ravel(),
    index=X.columns,
    name="LR coefficient (scaled space)",
).sort_values(key=abs, ascending=False)
print(coefs)

rf_pipe = make_base_pipeline(
    RandomForestClassifier(random_state=SEED_MODEL, class_weight=None, n_jobs=-1)
)
rf_search = GridSearchCV(
    rf_pipe,
    {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 8, 16],
        "clf__min_samples_leaf": [1, 2],
    },
    cv=inner_cv,
    scoring="f1",
    n_jobs=-1,
)
rf_search.fit(X_train, y_train)
rf_best = rf_search.best_estimator_
imp = pd.Series(
    rf_best.named_steps["clf"].feature_importances_,
    index=X.columns,
    name="RF importance",
).sort_values(ascending=False)
print(imp)"""
    )
)

cells.append(
    code(
        """# SHAP (tùy chọn): uv sync --extra shap
try:
    import shap

    X_tr_imp = rf_best.named_steps["imputer"].transform(X_train)
    X_tr_imp = rf_best.named_steps["scaler"].transform(X_tr_imp)
    explainer = shap.TreeExplainer(rf_best.named_steps["clf"])
    sv = explainer.shap_values(X_tr_imp)
    if isinstance(sv, list):
        sv = sv[1]
    shap.summary_plot(sv, pd.DataFrame(X_tr_imp, columns=X.columns), show=True)
except ImportError:
    print("Bỏ qua SHAP (chưa cài). Cài: uv sync --extra shap")"""
    )
)

cells.append(
    md(
        """## ROC & ma trận nhầm lẫn (`split_seed=REF_SEED`, mô hình F1 cao nhất theo mean)

Lưu hình vào `image/roc_confusion_test_seed*.png` để chèn LaTeX (`\\includegraphics`)."""
    )
)

cells.append(
    code(
        """REF_SEED = SPLIT_SEEDS[0]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=REF_SEED
)

mean_f1 = results.groupby("model")["f1"].mean().sort_values(ascending=False)
best_name = mean_f1.index[0]
print("Best by mean F1:", best_name)

best_model = fitted_by_seed[REF_SEED][best_name]
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax[0].plot(fpr, tpr, label=f"{best_name} AUC={roc_auc_score(y_test, y_proba):.3f}")
ax[0].plot([0, 1], [0, 1], "k--")
ax[0].set_xlabel("FPR")
ax[0].set_ylabel("TPR")
ax[0].set_title(f"ROC (test, seed={REF_SEED})")
ax[0].legend()

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[1])
ax[1].set_xlabel("Predicted")
ax[1].set_ylabel("True")
ax[1].set_title("Confusion matrix")
plt.tight_layout()
_img_dir = REPO_ROOT / "image"
_img_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(_img_dir / f"roc_confusion_test_seed{REF_SEED}.png", dpi=200, bbox_inches="tight")
plt.show()
print(classification_report(y_test, y_pred, digits=4))"""
    )
)

cells.append(
    md(
        """## Ablation (khớp gợi ý `main.tex`): có / không `StandardScaler` — logistic cố định (`C=1`), một split

So sánh nhanh F1 trên cùng tập test; không thay thế lưới siêu tham số đầy đủ ở trên."""
    )
)

cells.append(
    code(
        """REF_SEED = SPLIT_SEEDS[0]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=REF_SEED
)
lr_a = LogisticRegression(max_iter=10_000, random_state=SEED_MODEL, C=1.0)
pipe_scaled = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", lr_a),
    ]
)
lr_b = LogisticRegression(max_iter=10_000, random_state=SEED_MODEL, C=1.0)
pipe_no_scale = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", lr_b),
    ]
)
pipe_scaled.fit(X_train, y_train)
pipe_no_scale.fit(X_train, y_train)
_f1_sc = f1_score(y_test, pipe_scaled.predict(X_test))
_f1_ns = f1_score(y_test, pipe_no_scale.predict(X_test))
ablation_lr_f1 = pd.DataFrame(
    [
        {"variant": "LR + impute + StandardScaler", "f1_test": _f1_sc},
        {"variant": "LR + impute (no scaler)", "f1_test": _f1_ns},
    ]
)
print("F1 (LR + impute + scale):", _f1_sc)
print("F1 (LR + impute, không scale):", _f1_ns)"""
    )
)

cells.append(md("""## Lưu mô hình tốt nhất (theo mean F1) — `joblib`"""))

cells.append(
    code(
        """out_path = REPO_ROOT / "resources" / "diabetes_prediction_model.pkl"
joblib.dump(best_model, out_path)
print("Đã lưu:", out_path.resolve())"""
    )
)

cells.append(
    md(
        """## Xuất kết quả ra `resources/outputs/` (CSV, JSON, hình)

Hàm `save_run_outputs` trong `resources/export_run.py` ghi:

- `resources/outputs/latest/` — bản mới nhất (ghi đè mỗi lần **Run All**),
- `resources/outputs/runs/<timestamp>/` — bản lưu theo thời điểm.

Có thể trích dữ liệu từ đây vào bảng LaTeX (`main.tex`)."""
    )
)

cells.append(
    code(
        """import importlib.util

_spec = importlib.util.spec_from_file_location(
    "export_run", REPO_ROOT / "resources" / "export_run.py"
)
_export = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_export)

saved_to = _export.save_run_outputs(
    REPO_ROOT,
    results=results,
    summary=summary,
    display_tbl=display_tbl,
    ci_f1=ci_f1_df,
    wilcoxon_summary=wilcoxon_summary,
    ablation_lr_f1=ablation_lr_f1,
    lr_coefs=coefs,
    rf_importance=imp,
    model_pkl_relative="resources/diabetes_prediction_model.pkl",
    extra_manifest={
        "RANDOM_STATE_CV": RANDOM_STATE_CV,
        "SEED_MODEL": SEED_MODEL,
        "SPLIT_SEEDS": list(SPLIT_SEEDS),
        "TEST_SIZE": TEST_SIZE,
        "N_INNER_SPLITS": N_INNER_SPLITS,
        "best_model_by_mean_f1": str(
            results.groupby("model")["f1"].mean().sort_values(ascending=False).index[0]
        ),
    },
)
print("Đã xuất artifact tới:", saved_to.resolve())"""
    )
)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Wrote", OUT)
