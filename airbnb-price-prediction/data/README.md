# Data

| Folder | Content | In git? |
|---|---|---|
| `raw/` | `listings.csv` - the full Inside Airbnb export for Dublin (scraped 16 Sep 2025, 6,945 listings, ~14 MB) | no |
| `processed/` | frozen `train.csv` / `test.csv`, written by `src/data_prep.py` | no |
| `sample/` | `listings_sample.csv` - 318 rows of the raw file, enough to test that the pipeline runs | yes |

## Getting the full data

1. Go to <https://insideairbnb.com/get-the-data/> and find **Dublin, Leinster, Ireland**.
2. Download `listings.csv.gz` (the *detailed* listings file, 79 columns) and unzip it to
   `data/raw/listings.csv`.

Inside Airbnb publishes a new snapshot every few months, so a newer download will give
slightly different numbers than the ones in the README.

Data © Inside Airbnb, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
