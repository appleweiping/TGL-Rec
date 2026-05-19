# Manual Data Steps

These steps are only needed if automatic download is unavailable or blocked.
Do not commit raw datasets to this repository.

## MovieLens-1M

- Official source page: https://grouplens.org/datasets/movielens/1m/
- Official archive URL: https://files.grouplens.org/datasets/movielens/ml-1m.zip
- Official README: https://files.grouplens.org/datasets/movielens/ml-1m-README.txt
- Access date: 2026-04-29
- License/use note: GroupLens permits research use with acknowledgement, disallows endorsement claims, and requires permission for commercial or revenue-bearing use.

Steps:

1. Download `ml-1m.zip` from the official archive URL.
2. Keep the archive outside git, for example at `data/raw/movielens_1m/ml-1m.zip`.
3. Either pass the zip directly:

```bash
python -m tglrec.cli preprocess movielens-1m --zip-path data/raw/movielens_1m/ml-1m.zip --output-dir artifacts/datasets/movielens_1m
```

4. Or unzip it so the files exist under `data/raw/movielens_1m/ml-1m/`, then run:

```bash
python -m tglrec.cli preprocess movielens-1m --raw-dir data/raw/movielens_1m/ml-1m --output-dir artifacts/datasets/movielens_1m
```

Expected raw files:

```text
ratings.dat  UserID::MovieID::Rating::Timestamp
movies.dat   MovieID::Title::Genres
users.dat    UserID::Gender::Age::Occupation::Zip-code
```

The current preprocessing path reads `ratings.dat` and `movies.dat`, normalizes raw IDs to stable integer IDs, and writes deterministic temporal leave-one-out and global-time splits.

## Amazon Reviews 2023

- Source page: https://amazon-reviews-2023.github.io/main.html
- Hugging Face dataset page: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- Access date: 2026-04-29
- License/use note: verify the current dataset card and source-page terms before redistribution
  or paper-grade release artifacts. Do not commit raw reviews or metadata.

Steps:

1. Download one category review file and its metadata file through the official source page or
   Hugging Face dataset page. For the first ecommerce benchmark, use a bounded category such as
   `All_Beauty`.
2. Keep the local files outside git, for example:

```text
data/raw/amazon_reviews_2023/All_Beauty.jsonl.gz
data/raw/amazon_reviews_2023/meta_All_Beauty.jsonl.gz
```

3. Run local preprocessing:

```bash
python -m tglrec.cli preprocess amazon-reviews-2023 \
  --reviews-path data/raw/amazon_reviews_2023/All_Beauty.jsonl.gz \
  --metadata-path data/raw/amazon_reviews_2023/meta_All_Beauty.jsonl.gz \
  --category all_beauty \
  --output-dir artifacts/datasets/amazon_reviews_2023_all_beauty \
  --min-user-interactions 5 \
  --min-item-interactions 5 \
  --global-train-ratio 0.8 \
  --global-val-ratio 0.1 \
  --seed 2026
```

Expected review fields for Amazon Reviews 2023 category JSONL files include `user_id`,
`parent_asin`, `asin`, `rating`, and `timestamp`. The CLI also supports older/local variants with
`unixReviewTime` via timestamp auto-detection or `--timestamp-col`.

Expected metadata fields include `parent_asin`, `title`, `main_category`, `categories`, `store`,
`description`, `features`, `average_rating`, and `rating_number`. Missing optional metadata fields
are written as empty strings; missing required review user, item, or timestamp fields fail loudly.
The preprocessing output intentionally excludes full-horizon aggregate fields such as
`average_rating` and `rating_number` from `items.csv` so downstream item text/features do not
receive future rating/popularity signal.

By default the loader collapses repeated `(user_id, parent_asin)` interactions to the first
observed event and rejects same-user identical timestamps. The former keeps held-out targets out of
prior history under seen-item filtering; the latter avoids inventing a temporal order from tied
timestamps. The manifest records raw file SHA256 hashes and byte sizes, but paper-grade releases
should also record the exact source URL or Hugging Face revision used to obtain the files.
