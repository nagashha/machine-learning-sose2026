"""Step 3 - Feature engineering: one sklearn block per data modality.

Everything is an sklearn transformer, so imputation, scaling, TF-IDF, SVD and
KMeans are fitted inside the Pipeline, i.e. only on the training folds during
cross-validation (no leakage).

  TABULAR : numeric columns (median-imputed, scaled) + one-hot categoricals
  TEXT    : name + description + neighborhood_overview
            -> TF-IDF (uni+bigrams) -> TruncatedSVD (dense "embedding")
  SPATIAL : lat/lon, distance to centre, KMeans location-cluster one-hots
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config


# ------------------------------------------------------------------ helpers
def numeric_feature_names(df: pd.DataFrame) -> list[str]:
    """All numeric columns that are legal features (no IDs, no target)."""
    drop = set(config.DROP_ALWAYS) | {"latitude", "longitude"}
    return [c for c in df.select_dtypes(include=np.number).columns if c not in drop]


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.cols]


class TextJoiner(BaseEstimator, TransformerMixin):
    """Concatenate several text columns into one string per row."""

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return (X[self.cols].fillna("")
                .agg(" ".join, axis=1)
                .str.replace(r"<[^>]+>", " ", regex=True)   # strip HTML tags
                .values)


class GeoClusterer(BaseEstimator, TransformerMixin):
    """KMeans on (lat, lon); outputs one-hot cluster membership."""

    def __init__(self, k=config.N_GEO_CLUSTERS, seed=config.SEED):
        self.k, self.seed = k, seed

    def fit(self, X, y=None):
        self.km_ = KMeans(n_clusters=self.k, random_state=self.seed, n_init=10)
        self.km_.fit(X[["latitude", "longitude"]])
        return self

    def transform(self, X):
        labels = self.km_.predict(X[["latitude", "longitude"]])
        return np.eye(self.k)[labels]


# ---------------------------------------------------------- modality blocks
def tabular_block(df: pd.DataFrame) -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", numeric, numeric_feature_names(df)),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         config.CATEGORICAL),
    ])


def text_block() -> Pipeline:
    return Pipeline([
        ("join", TextJoiner(config.TEXT_COLS)),
        ("tfidf", TfidfVectorizer(max_features=20_000, ngram_range=(1, 2),
                                  min_df=3, stop_words="english",
                                  sublinear_tf=True)),
        ("svd", TruncatedSVD(n_components=config.N_TEXT_SVD,
                             random_state=config.SEED)),
    ])


def spatial_block() -> FeatureUnion:
    return FeatureUnion([
        ("coords", Pipeline([
            ("sel", ColumnSelector(["latitude", "longitude", "dist_center_km"])),
            ("scale", StandardScaler()),
        ])),
        ("clusters", GeoClusterer()),
    ])


# ------------------------------------------------------------- feature sets
def build_feature_union(df: pd.DataFrame, modalities: tuple[str, ...]) -> FeatureUnion:
    """Compose the requested modalities into one FeatureUnion."""
    blocks = []
    if "tabular" in modalities:
        blocks.append(("tabular", tabular_block(df)))
    if "text" in modalities:
        blocks.append(("text", text_block()))
    if "spatial" in modalities:
        blocks.append(("spatial", spatial_block()))
    return FeatureUnion(blocks)


FEATURE_SETS = {
    "tabular":         ("tabular",),
    "tabular+spatial": ("tabular", "spatial"),
    "tabular+text":    ("tabular", "text"),
    "all_modalities":  ("tabular", "text", "spatial"),
}
