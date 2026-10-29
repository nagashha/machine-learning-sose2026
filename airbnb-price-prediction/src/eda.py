"""Step 2 - Exploratory data analysis (figures 01-06).

Runs on the TRAINING split only, so nothing we look at is informed by the
test data.

Run:  python src/eda.py        (after data_prep.py)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config

sns.set_theme(style="whitegrid", context="talk")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def main():
    df = pd.read_csv(config.PROCESSED_DIR / "train.csv")

    # 01 - target distribution: raw vs log
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["price"], bins=60, ax=axes[0], color="#4C72B0")
    axes[0].set(title="Nightly price (EUR)", xlabel="price")
    sns.histplot(df["log_price"], bins=60, ax=axes[1], color="#55A868")
    axes[1].set(title="log(price) - modelling target", xlabel="log_price")
    save(fig, "01_target_distribution.png")

    # 02 - price by room type & neighbourhood
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, col, title in [(axes[0], "room_type", "Price by room type"),
                           (axes[1], "neighbourhood_cleansed", "Price by neighbourhood")]:
        order = df.groupby(col)["price"].median().sort_values().index
        sns.boxplot(data=df, x=col, y="price", order=order, ax=ax)
        ax.set(title=title, xlabel="")
        ax.tick_params(axis="x", rotation=20)
    save(fig, "02_price_by_category.png")

    # 03 - spatial view: listings coloured by price
    fig, ax = plt.subplots(figsize=(9, 9))
    sc = ax.scatter(df["longitude"], df["latitude"], c=df["log_price"],
                    cmap="viridis", s=8, alpha=0.6)
    ax.scatter(*config.CITY_CENTER[::-1], marker="*", s=350, c="red",
               edgecolor="black", label="city centre", zorder=5)
    ax.set(title="Listings coloured by log(price)", xlabel="longitude", ylabel="latitude")
    ax.legend()
    fig.colorbar(sc, ax=ax, label="log_price")
    save(fig, "03_spatial_price_map.png")

    # 04 - price vs. distance to centre
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(data=df.sample(min(len(df), 2000), random_state=0),
                x="dist_center_km", y="log_price",
                scatter_kws=dict(s=10, alpha=0.3), line_kws=dict(color="red"), ax=ax)
    ax.set(title="log(price) vs. distance to city centre", xlabel="distance to centre (km)")
    save(fig, "04_price_vs_distance.png")

    # 05 - strongest linear correlations with log_price
    corr = (df.select_dtypes(include=np.number).corr()["log_price"]
              .drop(["log_price", "price", "id", "host_id"]).dropna().sort_values())
    top = pd.concat([corr.head(10), corr.tail(10)])
    fig, ax = plt.subplots(figsize=(9, 8))
    top.plot.barh(ax=ax, color=np.where(top > 0, "#55A868", "#C44E52"))
    ax.set(title="Correlation with log(price) - top +/-10 features", xlabel="Pearson r")
    save(fig, "05_feature_correlations.png")

    # 06 - missingness in the raw file (before any cleaning)
    raw = pd.read_csv(config.RAW_CSV)
    miss = raw.isna().mean().sort_values(ascending=False) * 100
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(miss))))
    miss.plot.barh(ax=ax, color="#8172B2")
    ax.set(title="Missing values in the raw data", xlabel="% missing")
    save(fig, "06_missingness.png")

    print("\n--- headline numbers (train split) ---")
    print(f"n listings         : {len(df):,}")
    print(f"median price       : {df['price'].median():.0f} EUR")
    print(f"mean price         : {df['price'].mean():.0f} EUR")
    print(f"share entire homes : {(df['room_type'] == 'Entire home/apt').mean():.1%}")
    print(f"share private rooms: {(df['room_type'] == 'Private room').mean():.1%}")


if __name__ == "__main__":
    main()
