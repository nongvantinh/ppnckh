"""PR-AUC, bootstrap CI cho ROC-AUC (split 0), Cliff's delta trên F1 — cùng pipeline `diabetes.ipynb`."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Đồng bộ compute_calibration.py
RANDOM_STATE_CV = 42
SEED_MODEL = 42
SPLIT_SEEDS = [0, 1, 2, 3, 4]
TEST_SIZE = 0.2
N_INNER_SPLITS = 5
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42

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
ZERO_TO_NA = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def _repo_root() -> Path:
    p = Path(__file__).resolve().parent.parent
    if (p / "main.tex").is_file():
        return p
    return Path.cwd()


REPO_ROOT = _repo_root()
DATA_PATH = REPO_ROOT / "resources" / "data" / "pima-indians-diabetes.data.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
)


def load_pima(path: Path = DATA_PATH, url: str = DATA_URL) -> pd.DataFrame:
    if not path.exists():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        print("Đang tải", url, "->", path, file=sys.stderr)
        urllib.request.urlretrieve(url, path)
    df = pd.read_csv(path, header=None, names=COLS)
    for c in ZERO_TO_NA:
        df.loc[df[c] == 0, c] = np.nan
    return df


def make_base_pipeline(classifier) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", classifier),
        ]
    )


MODEL_GRIDS: list[tuple[str, object, dict]] = [
    (
        "Logistic Regression",
        LogisticRegression(max_iter=10_000, random_state=SEED_MODEL, class_weight=None),
        {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__solver": ["lbfgs"]},
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
]

MODEL_ORDER = ["Logistic Regression", "Random Forest", "SVM (RBF)", "KNN"]


def evaluate_split_proba(
    X_tr, X_te, y_tr, y_te, name, base_estimator, param_grid, inner_cv
):
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
    proba = best.predict_proba(X_te)[:, 1]
    return best, proba


def bootstrap_roc_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Trả về (auc điểm, phân vị 2.5%, 97.5%) từ bootstrap chỉ số mẫu trên tập kiểm thử."""
    rng = rng or np.random.default_rng(BOOTSTRAP_SEED)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    aucs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        if np.unique(yt).size < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    if not aucs:
        auc0 = roc_auc_score(y_true, y_score)
        return float(auc0), float(auc0), float(auc0)
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(roc_auc_score(y_true, y_score)), float(lo), float(hi)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta (hai mẫu độc lập); x,y là vector F1 theo cùng thứ tự split."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    nx, ny = len(x), len(y)
    more = np.sum(x[:, None] > y[None, :])
    less = np.sum(x[:, None] < y[None, :])
    return float((more - less) / (nx * ny))


def interpret_cliff(d: float) -> str:
    a = abs(d)
    if a < 0.147:
        return "không đáng kể"
    if a < 0.33:
        return "nhỏ"
    if a < 0.474:
        return "vừa"
    return "lớn"


def main() -> None:
    diabetes = load_pima()
    X = diabetes.drop(columns=["Outcome"])
    y = diabetes["Outcome"].astype(int)
    inner_cv = StratifiedKFold(
        n_splits=N_INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE_CV
    )

    rows_pr = []
    boot_rows = []
    fitted_seed0: dict[str, tuple[object, np.ndarray]] = {}
    y_test_seed0: pd.Series | None = None

    for split_seed in SPLIT_SEEDS:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=split_seed
        )
        if split_seed == 0:
            y_test_seed0 = y_test
        for name, est, grid in MODEL_GRIDS:
            _, proba = evaluate_split_proba(
                X_train, X_test, y_train, y_test, name, clone(est), grid, inner_cv
            )
            pr_auc = average_precision_score(y_test, proba)
            rows_pr.append(
                {"model": name, "split_seed": split_seed, "pr_auc": pr_auc}
            )
            if split_seed == 0:
                fitted_seed0[name] = (_, proba)

    assert y_test_seed0 is not None
    y_te = y_test_seed0.values

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for name in MODEL_ORDER:
        _, proba = fitted_seed0[name]
        auc_pt, lo, hi = bootstrap_roc_auc_ci(y_te, proba, rng=rng)
        boot_rows.append(
            {
                "model": name,
                "roc_auc_split0": auc_pt,
                "ci95_low": lo,
                "ci95_high": hi,
                "n_bootstrap": N_BOOTSTRAP,
            }
        )

    out_dir = REPO_ROOT / "resources" / "outputs" / "latest"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_pr = pd.DataFrame(rows_pr)
    df_pr.to_csv(out_dir / "pr_auc_by_split.csv", index=False)
    pr_summary = (
        df_pr.groupby("model")["pr_auc"].agg(["mean", "std"]).reset_index()
    )
    pr_summary = pr_summary.sort_values("mean", ascending=False)
    pr_summary.to_csv(out_dir / "pr_auc_mean_std.csv", index=False)

    df_boot = pd.DataFrame(boot_rows)
    df_boot.to_csv(out_dir / "roc_auc_bootstrap_ci_seed0.csv", index=False)

    # Cliff's delta: F1 theo split từ cùng pipeline (đọc results hoặc tính lại)
    results_path = out_dir / "results_by_split.csv"
    if not results_path.exists():
        raise FileNotFoundError(
            f"Cần {results_path} — chạy notebook export trước, hoặc tính F1 trong script."
        )
    r = pd.read_csv(results_path)

    def f1_vec(name: str) -> np.ndarray:
        return r[r["model"] == name].sort_values("split_seed")["f1"].values

    # Thứ tự báo cáo: delta > 0 nghĩa là model_a (cột trái) có xu hướng F1 cao hơn model_b
    PAIR_ORDER: list[tuple[str, str]] = [
        ("Random Forest", "Logistic Regression"),
        ("Random Forest", "KNN"),
        ("Random Forest", "SVM (RBF)"),
        ("Logistic Regression", "SVM (RBF)"),
        ("Logistic Regression", "KNN"),
        ("SVM (RBF)", "KNN"),
    ]
    cliff_rows = []
    for a, b in PAIR_ORDER:
        fa = f1_vec(a)
        fb = f1_vec(b)
        d = cliffs_delta(fa, fb)
        cliff_rows.append(
            {
                "model_a": a,
                "model_b": b,
                "cliffs_delta": round(d, 3),
                "interpretation": interpret_cliff(d),
            }
        )
    pd.DataFrame(cliff_rows).to_csv(out_dir / "cliffs_delta_f1_pairs.csv", index=False)

    # Hình PR (split 0)
    colors = {
        "Logistic Regression": "#1f77b4",
        "Random Forest": "#2ca02c",
        "SVM (RBF)": "#ff7f0e",
        "KNN": "#d62728",
    }
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    baseline = y_te.mean()
    for name in MODEL_ORDER:
        _, proba = fitted_seed0[name]
        prec, rec, _ = precision_recall_curve(y_te, proba)
        ap = average_precision_score(y_te, proba)
        ax.plot(rec, prec, label=f"{name} (AP $= {ap:.3f}$)", color=colors[name], lw=1.8)
    ax.axhline(baseline, color="gray", ls=":", lw=1.2, label=f"Tỷ lệ dương tính $= {baseline:.3f}$")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Đường Precision--Recall — tập kiểm thử, split_seed=0")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig_path = REPO_ROOT / "image" / "pr_curves_test_seed0.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figures" / "pr_curves_test_seed0.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(pr_summary.to_string(index=False))
    print(df_boot.to_string(index=False))
    print(pd.DataFrame(cliff_rows).to_string(index=False))
    print("Đã ghi artifact trong", out_dir)


if __name__ == "__main__":
    main()
