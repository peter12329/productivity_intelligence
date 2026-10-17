"""Evaluate the saved models on the hold-out people and explain them.

Run from the project root, after `python -m ml.train`:
    python -m ml.evaluate

Writes to artifacts/figures/ and artifacts/evaluation_report.md:
    confusion_matrix.png          which levels get confused with which
    regression_pred_vs_actual.png predicted vs real score on unseen people
    permutation_importance.png    what each model relies on
    partial_dependence.png        how the predicted score moves with each input

READ THIS BEFORE QUOTING ANY NUMBER
    * The data is synthetic: results show how well the models recover the rules
      in dummydataset.py, not how well they predict real people.
    * Importance / partial dependence describe the MODEL, not cause and effect.
    * With only a few hold-out people the numbers are noisy; the report says so.
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, mean_absolute_error,
                             r2_score, roc_auc_score)

from ml.settings import (ARTIFACT_DIR, CLASSIFICATION_TARGET, FEATURES, LEVELS,
                         RANDOM_STATE, REGRESSION_TARGET)
from ml.train import load_data, split_data

FIG_DIR = ARTIFACT_DIR / "figures"


def main() -> None:
    for name in ("regressor.joblib", "classifier.joblib", "metadata.json"):
        if not (ARTIFACT_DIR / name).exists():
            raise SystemExit("Models not found. Run: python -m ml.train")

    reg = joblib.load(ARTIFACT_DIR / "regressor.joblib")
    clf = joblib.load(ARTIFACT_DIR / "classifier.joblib")
    meta = json.loads((ARTIFACT_DIR / "metadata.json").read_text())

    _, test_df = split_data(load_data())
    if test_df is None:
        raise SystemExit("Dataset too small for a hold-out set (< 100 rows).")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    X = test_df[FEATURES]
    y_reg = test_df[REGRESSION_TARGET]
    y_clf = test_df[CLASSIFICATION_TARGET].map({n: i for i, n in enumerate(LEVELS)})

    # ---- regression ----
    pred = np.clip(reg.predict(X), 0, 100)
    r2, mae = r2_score(y_reg, pred), mean_absolute_error(y_reg, pred)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_reg, pred, s=6, alpha=0.35)
    ax.plot([0, 100], [0, 100], "k--", lw=1)
    ax.set(xlabel="actual score", ylabel="predicted score",
           title=f"Unseen people: R2={r2:.2f}, MAE={mae:.1f}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "regression_pred_vs_actual.png", dpi=130)
    plt.close(fig)

    # ---- classification ----
    pred_c = clf.predict(X)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ConfusionMatrixDisplay.from_predictions(
        y_clf, pred_c, labels=list(range(len(LEVELS))), display_labels=LEVELS,
        normalize="true", values_format=".0%", ax=ax, colorbar=False)
    ax.set_title("Confusion matrix (share of each true level)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=130)
    plt.close(fig)
    auc = roc_auc_score(y_clf, clf.predict_proba(X), multi_class="ovr",
                        average="macro", labels=list(range(len(LEVELS))))

    # ---- permutation importance (on hold-out, 15 shuffles) ----
    imp_r = permutation_importance(reg, X, y_reg, scoring="neg_mean_absolute_error",
                                   n_repeats=15, random_state=RANDOM_STATE)
    imp_c = permutation_importance(clf, X, y_clf, scoring="f1_macro",
                                   n_repeats=15, random_state=RANDOM_STATE)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, imp, title in ((axes[0], imp_r, "Regression: MAE increase when shuffled"),
                           (axes[1], imp_c, "Classification: macro-F1 drop when shuffled")):
        order = np.argsort(imp.importances_mean)
        ax.barh(np.array(FEATURES)[order], imp.importances_mean[order],
                xerr=imp.importances_std[order])
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "permutation_importance.png", dpi=130)
    plt.close(fig)

    # ---- partial dependence of the regressor ----
    fig, ax = plt.subplots(figsize=(13, 3.6))
    # Newer scikit-learn refuses integer columns here, so plot from a float copy.
    PartialDependenceDisplay.from_estimator(reg, X.astype(float), FEATURES, ax=ax, grid_resolution=25)
    fig.suptitle("Partial dependence of predicted score (model behaviour, not causation)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "partial_dependence.png", dpi=130)
    plt.close(fig)

    # ---- markdown report ----
    split = meta["split"]
    lines = [
        "# Evaluation report (synthetic data)",
        "",
        meta["disclaimer"],
        "",
        f"Hold-out: {split.get('test_people', '?')} people never seen in training "
        f"({split['test_rows']} rows). {split.get('warning', '')}",
        "",
        f"## Regression ({meta['regression']['selected']})",
        f"- Hold-out R2 = {r2:.3f}, MAE = {mae:.2f} points (score range 0-100)",
        "- Cross-validated numbers (more stable) are in metadata.json under "
        "regression.candidates",
        "",
        f"## Classification ({meta['classification']['selected']})",
        f"- Hold-out accuracy = {np.mean(pred_c == y_clf):.3f}, "
        f"macro one-vs-rest ROC-AUC = {auc:.3f}",
        "- Read confusion_matrix.png: the 'low' level is rare, so expect it to be "
        "the hardest class.",
        "",
        "## Permutation importance (hold-out)",
        "| feature | regression MAE increase | classification F1 drop |",
        "|---|---|---|",
    ]
    for i in np.argsort(-imp_r.importances_mean):
        lines.append(f"| {FEATURES[i]} | {imp_r.importances_mean[i]:.3f} | "
                     f"{imp_c.importances_mean[i]:.3f} |")
    lines += ["", "Importance describes what the fitted model uses. It is not causal "
              "and correlated features share credit.", ""]
    (ARTIFACT_DIR / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"Figures saved in {FIG_DIR}")


if __name__ == "__main__":
    main()