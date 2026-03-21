"""Tính Brier score và vẽ đường hiệu chuẩn — cùng pipeline với `diabetes.ipynb` / `main.tex`."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
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


def evaluate_split_proba(X_tr, X_te, y_tr, y_te, name, base_estimator, param_grid, inner_cv):
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


def main() -> None:
    diabetes = load_pima()
    X = diabetes.drop(columns=["Outcome"])
    y = diabetes["Outcome"].astype(int)
    inner_cv = StratifiedKFold(
        n_splits=N_INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE_CV
    )

    rows = []
    fitted_seed0: dict[str, object] = {}
    y_test_seed0 = None

    for split_seed in SPLIT_SEEDS:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=split_seed
        )
        if split_seed == 0:
            y_test_seed0 = y_test
        for name, est, grid in MODEL_GRIDS:
            best, proba = evaluate_split_proba(
                X_train, X_test, y_train, y_test, name, clone(est), grid, inner_cv
            )
            brier = brier_score_loss(y_test, proba)
            rows.append(
                {"model": name, "split_seed": split_seed, "brier": brier}
            )
            if split_seed == 0:
                fitted_seed0[name] = (best, proba)

    df = pd.DataFrame(rows)
    out_dir = REPO_ROOT / "resources" / "outputs" / "latest"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "brier_by_split.csv", index=False)

    summary = (
        df.groupby("model")["brier"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean")
    )
    summary.to_csv(out_dir / "brier_mean_std.csv", index=False)

    # Hình: đường hiệu chuẩn (split_seed=0), 4 mô hình học
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    colors = {
        "Logistic Regression": "#1f77b4",
        "Random Forest": "#2ca02c",
        "SVM (RBF)": "#ff7f0e",
        "KNN": "#d62728",
    }
    assert y_test_seed0 is not None
    y_te = y_test_seed0.values

    for name in ["Logistic Regression", "Random Forest", "SVM (RBF)", "KNN"]:
        _, proba = fitted_seed0[name]
        prob_true, prob_pred = calibration_curve(
            y_te, proba, n_bins=10, strategy="uniform"
        )
        ax.plot(
            prob_pred,
            prob_true,
            marker="o",
            label=name,
            color=colors[name],
            linewidth=1.8,
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Hiệu chuẩn hoàn hảo")
    ax.set_xlabel("Xác suất dự đoán trung bình (theo bin)")
    ax.set_ylabel("Tỷ lệ dương tính quan sát")
    ax.set_title("Đường hiệu chuẩn — tập kiểm thử, split_seed=0")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()

    fig_path = REPO_ROOT / "image" / "calibration_curves_test_seed0.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    out_fig = out_dir / "figures" / "calibration_curves_test_seed0.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_string(index=False))
    print("Đã ghi:", out_dir / "brier_by_split.csv")
    print("Đã ghi:", out_dir / "brier_mean_std.csv")
    print("Đã ghi:", fig_path)


if __name__ == "__main__":
    main()
