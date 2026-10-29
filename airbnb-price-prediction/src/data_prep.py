"""Step 1 - Cleaning, feature derivation and the frozen train/test split.

Only deterministic transformations happen here (parsing, flags, derived
columns). Anything that is *learned* from the data, e.g. median imputation or
scaling, lives inside the sklearn pipelines in features.py, so it is fitted on
the training folds only.

Run:  python src/data_prep.py
"""

import ast
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config

# Columns that are useless for modelling (URLs, IDs, scrape metadata), empty,
# redundant (min/max night variants) or leak the target (estimated revenue).
OBSOLETE_COLS = [
    "listing_url", "scrape_id", "last_scraped", "source",
    "picture_url", "host_url", "host_thumbnail_url", "host_picture_url",
    "calendar_last_scraped", "calendar_updated", "neighbourhood_group_cleansed",
    "license", "estimated_revenue_l365d",
    "minimum_minimum_nights", "maximum_minimum_nights",
    "minimum_maximum_nights", "maximum_maximum_nights",
    "minimum_nights_avg_ntm", "maximum_nights_avg_ntm",
    "neighbourhood", "host_neighbourhood",
]

BOOL_COLS = ["host_is_superhost", "host_has_profile_pic",
             "host_identity_verified", "instant_bookable", "has_availability"]

# amenity flag -> keywords that count as "has it"
TARGET_AMENITIES = {
    "wifi":             ["wifi"],
    "air_conditioning": ["air conditioning"],
    "heating":          ["central heating", "heating"],
    "kitchen":          ["kitchen", "cooking basics", "dishes and silverware",
                         "refrigerator", "microwave", "oven", "dishwasher"],
    "washer":           ["washer"],
    "dryer":            ["dryer"],
    "tv":               ["tv"],
    "parking":          ["parking"],
    "pool":             ["pool"],
    "gym":              ["gym", "fitness"],
    "breakfast":        ["breakfast"],
    "self_checkin":     ["self check-in", "lockbox", "smart lock", "keypad"],
    "elevator":         ["elevator", "lift"],
    "ev_charger":       ["ev charger"],
    "hot_tub":          ["hot tub"],
    "fireplace":        ["fireplace"],
    "garden":           ["garden", "backyard", "patio"],
    "workspace":        ["dedicated workspace", "laptop-friendly workspace"],
}


def parse_bathrooms(text):
    """'1.5 shared baths' -> 1.5,  'Half-bath' -> 0.5,  NaN -> NaN"""
    if pd.isna(text):
        return np.nan
    if "half" in str(text).lower():
        return 0.5
    m = re.search(r"(\d+\.?\d*)", str(text))
    return float(m.group(1)) if m else np.nan


def categorise_property(ptype):
    """Collapse ~50 raw property types into a handful of broad groups."""
    if pd.isna(ptype):
        return "other"
    pt = str(ptype).lower()
    if "entire" in pt:
        if any(x in pt for x in ["house", "home", "villa", "cottage", "bungalow"]):
            return "entire_house"
        if any(x in pt for x in ["condo", "apartment", "unit", "flat", "loft"]):
            return "entire_apartment"
        return "entire_other"
    if "private room" in pt:
        return "private_room"
    if "shared room" in pt:
        return "shared_room"
    if "hotel" in pt or "room in" in pt:
        return "hotel_room"
    return "other"


def parse_amenities(raw):
    try:
        return [a.lower() for a in ast.literal_eval(raw)]
    except (ValueError, SyntaxError):
        return []


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    scrape_date = pd.to_datetime(df["last_scraped"]).max()
    df.attrs["scrape_date"] = scrape_date

    df = df.drop(columns=[c for c in OBSOLETE_COLS if c in df.columns])

    # target: "$1,234.00" -> 1234.0, drop rows without a price and outliers
    df["price"] = df["price"].str.replace(r"[\$,]", "", regex=True).astype(float)
    n0 = len(df)
    df = df[df["price"].notna() & (df["price"] <= config.MAX_PRICE)].copy()
    print(f"Dropped {n0 - len(df):,} rows without price or above "
          f"EUR {config.MAX_PRICE:,}; {len(df):,} rows left")
    df["log_price"] = np.log(df["price"])

    # 't'/'f' -> 1/0; unknown is treated as "no"
    for col in BOOL_COLS:
        df[col] = df[col].map({"t": 1, "f": 0}).fillna(0)

    # "95%" -> 0.95 (missing values are imputed later, inside the pipeline)
    for col in ["host_response_rate", "host_acceptance_rate"]:
        df[col] = df[col].str.replace("%", "", regex=False).astype(float) / 100

    # bathrooms is often empty while bathrooms_text ("1.5 baths") is filled
    df["bathrooms"] = df["bathrooms"].fillna(df["bathrooms_text"].apply(parse_bathrooms))

    df["host_response_time"] = df["host_response_time"].fillna("unknown")
    df["property_category"] = df["property_type"].apply(categorise_property)

    # amenities: JSON-like list -> one binary column per amenity group
    amenity_lists = df["amenities"].apply(parse_amenities)
    for name, keywords in TARGET_AMENITIES.items():
        df[f"amenity_{name}"] = amenity_lists.apply(
            lambda lst: int(any(kw in item for kw in keywords for item in lst)))
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ref = df.attrs["scrape_date"]

    df["host_tenure_days"] = (ref - pd.to_datetime(df["host_since"], errors="coerce")).dt.days

    # "never reviewed" is informative, so we flag it before filling anything
    df["never_reviewed"] = df["first_review"].isna().astype(int)
    for col, new in [("first_review", "days_since_first_review"),
                     ("last_review", "days_since_last_review")]:
        df[new] = (ref - pd.to_datetime(df[col], errors="coerce")).dt.days.fillna(-1)
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

    df["beds_per_person"] = df["beds"] / df["accommodates"].clip(lower=1)
    df["baths_per_person"] = df["bathrooms"] / df["accommodates"].clip(lower=1)
    df["amenity_count"] = df.filter(like="amenity_").sum(axis=1)
    df["desc_length"] = df["description"].fillna("").str.len()
    df["has_neighborhood_overview"] = df["neighborhood_overview"].notna().astype(int)
    df["has_host_about"] = df["host_about"].notna().astype(int)

    # haversine distance to the city centre
    lat0, lon0 = np.radians(config.CITY_CENTER)
    lat, lon = np.radians(df["latitude"]), np.radians(df["longitude"])
    a = (np.sin((lat - lat0) / 2) ** 2
         + np.cos(lat0) * np.cos(lat) * np.sin((lon - lon0) / 2) ** 2)
    df["dist_center_km"] = 2 * 6371.0 * np.arcsin(np.sqrt(a))

    # text columns: empty string instead of NaN (needed by TF-IDF)
    for c in config.TEXT_COLS:
        df[c] = df[c].fillna("")
    return df


def main():
    df = pd.read_csv(config.RAW_CSV)
    print(f"Loaded {config.RAW_CSV.name}: {df.shape[0]:,} rows, {df.shape[1]} cols")

    df = add_derived_features(clean_data(df))

    train, test = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.SEED,
        stratify=df["neighbourhood_cleansed"],   # same area mix in both splits
    )
    train.to_csv(config.PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(config.PROCESSED_DIR / "test.csv", index=False)
    print(f"train: {train.shape}   test: {test.shape}   -> {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
