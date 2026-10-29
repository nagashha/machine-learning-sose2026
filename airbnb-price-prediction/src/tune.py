"""Step 5 - Hyperparameter tuning of the gradient-boosting model.

Randomised search (CV on the training split only) over the GBDT
hyperparameters and the size of the text embedding.

Run:  python src/tune.py        (slow: 30 candidates x 5 folds)
"""

import joblib
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

import config
import features

PARAM_DISTRIBUTIONS = {
    "model__learning_rate": loguniform(0.01, 0.3),
    "model__max_iter": randint(200, 1200),
    "model__max_leaf_nodes": randint(15, 127),
    "model__min_samples_leaf": randint(5, 60),
    "model__l2_regularization": loguniform(1e-3, 10),
    "features__text__svd__n_components": randint(30, 200),
}


def main():
    train = pd.read_csv(config.PROCESSED_DIR / "train.csv")
    y = train[config.TARGET].values

    pipe = Pipeline([
        ("features", features.build_feature_union(
            train, features.FEATURE_SETS["all_modalities"])),
        ("model", HistGradientBoostingRegressor(random_state=config.SEED)),
    ])
    search = RandomizedSearchCV(
        pipe, PARAM_DISTRIBUTIONS, n_iter=30,
        cv=KFold(config.N_CV_FOLDS, shuffle=True, random_state=config.SEED),
        scoring="neg_mean_absolute_error",
        random_state=config.SEED, n_jobs=-1, verbose=1,
    )
    search.fit(train, y)

    print("best CV MAE(log):", round(-search.best_score_, 4))
    print("best params     :", search.best_params_)

    cols = ["mean_test_score", "std_test_score", "rank_test_score"] + \
           [f"param_{p}" for p in PARAM_DISTRIBUTIONS]
    (pd.DataFrame(search.cv_results_)[cols]
       .sort_values("rank_test_score")
       .to_csv(config.METRICS_DIR / "tuning_results.csv", index=False))
    joblib.dump(search.best_estimator_, config.MODEL_DIR / "hist_gbdt__tuned.joblib")
    print("saved outputs/models/hist_gbdt__tuned.joblib")


if __name__ == "__main__":
    main()
