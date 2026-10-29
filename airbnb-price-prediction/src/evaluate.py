"""Step 6 - Final evaluation on the held-out test set (run once, at the end).

Loads every saved pipeline from outputs/models/, predicts the test split and
reports metrics in EUR (predictions are back-transformed with exp). Produces

  10_model_comparison.png       test MAE per model incl. a naive baseline
  11_pred_vs_actual.png         predicted vs. true price
  12_residuals.png              where the model over/under-predicts
  13_error_by_segment.png       MAE per neighbourhood and room type
  14_permutation_importance.png which inputs the best model relies on
  test_results.csv, worst_predictions.csv

Run:  python src/evaluate.py
"""

import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (mean_absolute_error, median_absolute_error,
                             r2_score, root_mean_squared_error)

import config

sns.set_theme(style="whitegrid", context="talk")


def euro_metrics(y_true_eur, y_pred_log):
    pred_eur = np.exp(y_pred_log)
    return dict(
        MAE_eur=mean_absolute_error(y_true_eur, pred_eur),
        MedAE_eur=median_absolute_error(y_true_eur, pred_eur),
        RMSE_eur=root_mean_squared_error(y_true_eur, pred_eur),
        R2_eur=r2_score(y_true_eur, pred_eur),
    )


def save(fig, name):
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def main():
    train = pd.read_csv(config.PROCESSED_DIR / "train.csv")
    test = pd.read_csv(config.PROCESSED_DIR / "test.csv")
    y_eur = test[config.PRICE_COL].values
    y_log = np.log(y_eur)

    # naive baseline every model has to beat: always predict the train median
    baseline = np.full(len(test), np.log(train[config.PRICE_COL].median()))
    rows = [dict(model="baseline_median", **euro_metrics(y_eur, baseline))]

    preds = {}
    for path in sorted(config.MODEL_DIR.glob("*.joblib")):
        pipe = joblib.load(path)
        p = pipe.predict(test)
        preds[path.stem] = (pipe, p)
        rows.append(dict(model=path.stem, **euro_metrics(y_eur, p)))

    results = pd.DataFrame(rows).sort_values("MAE_eur").reset_index(drop=True)
    results.to_csv(config.METRICS_DIR / "test_results.csv", index=False)
    print("\n", results.round(3).to_string(index=False))

    best_name = results[results.model != "baseline_median"].iloc[0]["model"]
    best_pipe, best_pred_log = preds[best_name]
    best_pred_eur = np.exp(best_pred_log)
    print(f"\nBest model on test: {best_name}")

    # 10 - model comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=results, y="model", x="MAE_eur", ax=ax, color="#4C72B0")
    ax.set(title="Test-set MAE (EUR / night)", xlabel="MAE (EUR)", ylabel="")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f", padding=3)
    save(fig, "10_model_comparison.png")

    # 11 - predicted vs. actual
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_eur, best_pred_eur, s=10, alpha=0.35)
    lim = [0, max(y_eur.max(), best_pred_eur.max()) * 1.02]
    ax.plot(lim, lim, "r--", lw=2, label="perfect prediction")
    ax.set(title=f"Predicted vs. actual - {best_name}",
           xlabel="actual (EUR)", ylabel="predicted (EUR)", xlim=lim, ylim=lim)
    ax.legend()
    save(fig, "11_pred_vs_actual.png")

    # 12 - residuals on the log scale
    resid = best_pred_log - y_log
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(resid, bins=60, ax=axes[0], color="#C44E52")
    axes[0].axvline(0, color="black")
    axes[0].set(title="Residuals (log scale)", xlabel="pred - actual")
    axes[1].scatter(y_log, resid, s=8, alpha=0.3)
    axes[1].axhline(0, color="red", ls="--")
    axes[1].set(title="Residual vs. actual", xlabel="log(actual)", ylabel="residual")
    save(fig, "12_residuals.png")

    # 13 - error by segment
    seg = test.copy()
    seg["abs_err_eur"] = np.abs(best_pred_eur - y_eur)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, col, color in [(axes[0], "neighbourhood_cleansed", "#55A868"),
                           (axes[1], "room_type", "#8172B2")]:
        seg.groupby(col)["abs_err_eur"].mean().sort_values().plot.barh(ax=ax, color=color)
        ax.set(title=f"MAE by {col.split('_')[0]}", xlabel="MAE (EUR)", ylabel="")
    save(fig, "13_error_by_segment.png")

    # 14 - permutation importance on raw input columns (shuffle one column,
    #      measure how much the error grows)
    check_cols = ["accommodates", "bedrooms", "bathrooms", "room_type",
                  "neighbourhood_cleansed", "latitude", "longitude",
                  "dist_center_km", "description", "amenity_count",
                  "review_scores_rating", "minimum_nights",
                  "availability_365", "host_is_superhost"]
    rng = np.random.default_rng(config.SEED)
    base_mae = mean_absolute_error(y_log, best_pred_log)
    imp = {}
    for col in check_cols:
        Xp = test.copy()
        Xp[col] = rng.permutation(Xp[col].values)
        imp[col] = mean_absolute_error(y_log, best_pipe.predict(Xp)) - base_mae
    imp = pd.Series(imp).sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    imp.plot.barh(ax=ax, color="#DD8452")
    ax.set(title=f"Permutation importance - {best_name}",
           xlabel="increase in MAE(log) when column is shuffled")
    save(fig, "14_permutation_importance.png")

    # worst predictions - concrete failure cases for the discussion
    seg["predicted_eur"] = best_pred_eur.round(1)
    worst = seg.nlargest(10, "abs_err_eur")[
        ["name", "neighbourhood_cleansed", "room_type", "accommodates",
         "price", "predicted_eur", "abs_err_eur"]]
    worst.round(1).to_csv(config.METRICS_DIR / "worst_predictions.csv", index=False)
    print("saved worst_predictions.csv")


if __name__ == "__main__":
    main()
