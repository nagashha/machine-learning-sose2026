"""Step 4 - Model selection with cross-validation.

Runs a grid of {model family} x {modality combination} with K-fold CV on the
TRAINING split only and saves

  outputs/metrics/cv_results.csv              all CV scores (= ablation table)
  outputs/models/<model>__<feature_set>.joblib  best pipeline per model family,
                                                refit on the full training split

The test set is not touched here - that happens once, in evaluate.py.

Run:  python src/train.py            (all models)
      python src/train.py ridge      (only one model family)
"""

import sys
import time

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

import config
import features


def make_models() -> dict:
    """Model zoo with sensible defaults (the winner is tuned in tune.py)."""
    return {
        "ridge": Ridge(alpha=1.0, random_state=config.SEED),
        "random_forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3, max_features=0.3,
            n_jobs=-1, random_state=config.SEED),
        "hist_gbdt": HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=config.SEED),
    }


def make_pipeline(train: pd.DataFrame, feature_set: str, model) -> Pipeline:
    return Pipeline([
        ("features", features.build_feature_union(train, features.FEATURE_SETS[feature_set])),
        ("model", model),
    ])


def main(only_model: str | None = None):
    train = pd.read_csv(config.PROCESSED_DIR / "train.csv")
    y = train[config.TARGET].values
    cv = KFold(n_splits=config.N_CV_FOLDS, shuffle=True, random_state=config.SEED)
    results_path = config.METRICS_DIR / "cv_results.csv"

    rows = []
    for fs_name in features.FEATURE_SETS:
        for model_name, model in make_models().items():
            if only_model and model_name != only_model:
                continue
            t0 = time.time()
            scores = cross_validate(
                make_pipeline(train, fs_name, model), train, y, cv=cv, n_jobs=1,
                scoring={"mae": "neg_mean_absolute_error",
                         "rmse": "neg_root_mean_squared_error",
                         "r2": "r2"})
            mae = -scores["test_mae"]
            rows.append(dict(
                feature_set=fs_name, model=model_name,
                cv_mae_log=mae.mean(), cv_mae_log_std=mae.std(),
                cv_rmse_log=-scores["test_rmse"].mean(),
                cv_r2=scores["test_r2"].mean(),
                fit_seconds=(time.time() - t0) / config.N_CV_FOLDS,
            ))
            print(f"{fs_name:16s} | {model_name:14s} | "
                  f"MAE(log)={mae.mean():.4f} +- {mae.std():.4f} | "
                  f"R2={scores['test_r2'].mean():.3f}", flush=True)

    results = pd.DataFrame(rows)
    if only_model and results_path.exists():          # merge partial runs
        prev = pd.read_csv(results_path)
        results = pd.concat([prev[prev.model != only_model], results])
    results = results.sort_values("cv_mae_log").reset_index(drop=True)
    results.to_csv(results_path, index=False)
    print(f"\nCV table saved to {results_path.relative_to(config.ROOT)}")

    # refit the best feature set of every model family on the full training split
    for model_name in [only_model] if only_model else make_models():
        best_fs = results[results.model == model_name].iloc[0].feature_set
        pipe = make_pipeline(train, best_fs, make_models()[model_name]).fit(train, y)
        out = config.MODEL_DIR / f"{model_name}__{best_fs}.joblib"
        joblib.dump(pipe, out)
        print("saved", out.name)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
