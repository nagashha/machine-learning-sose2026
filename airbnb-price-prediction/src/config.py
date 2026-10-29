"""Central configuration: paths, seed and column groups used by every script."""

import os
from pathlib import Path

# ----------------------------------------------------------------- paths ----
ROOT = Path(__file__).resolve().parents[1]

# Raw Inside Airbnb export. Can be overridden, e.g. to try the small sample:
#   AIRBNB_RAW_CSV=data/sample/listings_sample.csv python src/data_prep.py
RAW_CSV = Path(os.environ.get("AIRBNB_RAW_CSV", ROOT / "data" / "raw" / "listings.csv"))
if not RAW_CSV.is_absolute():
    RAW_CSV = (Path.cwd() / RAW_CSV).resolve()

PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"
MODEL_DIR = ROOT / "outputs" / "models"
METRICS_DIR = ROOT / "outputs" / "metrics"

for d in (PROCESSED_DIR, FIG_DIR, MODEL_DIR, METRICS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------ experiment ----
SEED = 42                 # one seed everywhere -> reproducible results
TEST_SIZE = 0.20          # 80/20 train/test split
N_CV_FOLDS = 5            # cross-validation folds used for model selection

TARGET = "log_price"      # we model log(price); metrics are reported in EUR
PRICE_COL = "price"
MAX_PRICE = 1_000         # listings above this nightly price are treated as outliers

# Dublin city centre (O'Connell Bridge), used for the distance feature
CITY_CENTER = (53.3473, -6.2591)

# --------------------------------------------------------- column groups ----
# Never used as plain numeric features: IDs, the target, free text (handled
# by the text pipeline) and raw columns that were replaced by derived ones.
DROP_ALWAYS = [
    "id", "host_id", "host_name", "host_location", "host_verifications",
    "price", "log_price", "amenities",
    "name", "description", "neighborhood_overview", "host_about",
    "host_since", "first_review", "last_review",
    "bathrooms_text", "property_type",
]

CATEGORICAL = ["neighbourhood_cleansed", "room_type", "property_category",
               "host_response_time"]

TEXT_COLS = ["name", "description", "neighborhood_overview"]

N_TEXT_SVD = 100          # dimensionality of the TF-IDF -> SVD text embedding
N_GEO_CLUSTERS = 12       # k for the KMeans location clusters
