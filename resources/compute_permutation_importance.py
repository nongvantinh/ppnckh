"""Permutation importance (F1 trên tập kiểm thử) — cùng pipeline `diabetes.ipynb`; trung bình qua năm split."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

RANDOM_STATE_CV = 42
SEED_MODEL = 42
SPLIT_SEEDS = [0, 1, 2, 3, 4]
TEST_SIZE = 0.2
N_INNER_SPLITS = 5
N_REPEATS = 30
PI_SEED = 42

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

FEATURE_NAMES = COLS[:-1]


def fit_best(
    X_tr, y_tr, name: str, base_estimator, param_grid, inner_cv
) -> Pipeline:
    pipe = make_base_pipeline(base_estimator)
    if not param_grid:
        pipe.fit(X_tr, y_tr)
        return pipe
    search = GridSearchCV(
        pipe,
        param_grid,
        cv=inner_cv,
        scoring="f1",
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_tr, y_tr)
    return search.best_estimator_


def main() -> None:
    diabetes = load_pima()
    X = diabetes.drop(columns=["Outcome"])
    y = diabetes["Outcome"].astype(int)
    inner_cv = StratifiedKFold(
        n_splits=N_INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE_CV
    )

    rows = []
    importances_by_split: dict[int, dict[str, np.ndarray]] = {}

    for split_seed in SPLIT_SEEDS:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=split_seed
        )
        importances_by_split[split_seed] = {}
        for name, est, grid in MODEL_GRIDS:
            best = fit_best(X_train, y_train, name, clone(est), grid, inner_cv)
            result = permutation_importance(
                best,
                X_test,
                y_test,
                n_repeats=N_REPEATS,
                random_state=PI_SEED + split_seed,
                scoring="f1",
                n_jobs=-1,
            )
            importances_by_split[split_seed][name] = result.importances_mean
            for j, feat in enumerate(FEATURE_NAMES):
                rows.append(
                    {
                        "model": name,
                        "split_seed": split_seed,
                        "feature": feat,
                        "importance_mean": result.importances_mean[j],
                        "importance_std": result.importances_std[j],
                    }
                )

    df = pd.DataFrame(rows)
    out_dir = REPO_ROOT / "resources" / "outputs" / "latest"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "permutation_importance_by_split.csv", index=False)

    agg = (
        df.groupby(["model", "feature"])[["importance_mean", "importance_std"]]
        .agg({"importance_mean": ["mean", "std"], "importance_std": "mean"})
        .reset_index()
    )
    agg.columns = [
        "model",
        "feature",
        "importance_mean_over_splits",
        "importance_mean_std_across_splits",
        "importance_std_within_pi_mean",
    ]
    agg.to_csv(out_dir / "permutation_importance_mean_over_splits.csv", index=False)

    # Trung bình trên mô hình (thứ hạng chung)
    cons = (
        df.groupby(["feature", "split_seed"])["importance_mean"]
        .mean()
        .reset_index()
        .groupby("feature")["importance_mean"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    cons.to_csv(out_dir / "permutation_importance_consensus.csv", index=False)

    # Hình: grouped bar — trung bình qua 5 split, 4 mô hình, 8 đặc trưng
    pivot = (
        df.groupby(["model", "feature"])["importance_mean"]
        .mean()
        .unstack(level=0)
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    x = np.arange(len(FEATURE_NAMES))
    w = 0.2
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    for i, m in enumerate([t[0] for t in MODEL_GRIDS]):
        vals = [pivot.loc[f, m] for f in FEATURE_NAMES]
        ax.bar(x + (i - 1.5) * w, vals, width=w, label=m, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURE_NAMES, rotation=25, ha="right")
    ax.set_ylabel(r"Giảm F1 trung bình (permutation, $n_{\mathrm{rep}}=30$)")
    ax.set_title(
        "Độ quan trọng hoán vị (trung bình qua năm lần chia train/test; đo trên tập kiểm thử)"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.grid(True, axis="y", alpha=0.35)
    fig.tight_layout()
    fp = REPO_ROOT / "image" / "permutation_importance_mean5splits.png"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figures" / "permutation_importance_mean5splits.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(cons.to_string(index=False))
    print("Đã ghi:", out_dir)
    print("Hình:", fp)


if __name__ == "__main__":
    main()
