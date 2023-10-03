"""Merge multiple datasets into one, and split into train, val, and test sets.

Usage:
    python src/scripts/merge_data.py
"""

import pandas as pd
import numpy as np

# Set seed such that the split is the same every time.
SEED = 42
np.random.seed(SEED)

DATA_RAW_PATHS = {
    "da_reviews": "data/raw/da-reviews.tsv",
    "danish-some-sentiment": "data/raw/danish-some-sentiment.tsv",
    "da": "data/raw/da.csv",
    "en": "data/raw/en.csv",
    "no": "data/raw/no.csv",
    "sv": "data/raw/sv.csv",
}

DATA_FINAL_PATHS = {
    "train": "data/processed/train.csv",
    "val": "data/processed/val.csv",
    "test": "data/processed/test.csv",
}

COLUMNS = ["text", "label", "language", "dataset"]


TEST_FRACTION = 0.005  # And val fraction


def process_da_reviews() -> pd.DataFrame:
    df_da_reviews = pd.read_csv(DATA_RAW_PATHS["da_reviews"], sep="\t")
    # The data has labels 1-5.
    # Based on manual inspection, we consider 4 and 5 to be positive, and 1-3 to be negative.
    binary_labels = (df_da_reviews["ratingValue"] > 3) * 1
    df_da_reviews["label"] = binary_labels
    df_da_reviews["text"] = df_da_reviews["reviewBody"]
    df_da_reviews["language"] = "da"
    df_da_reviews["dataset"] = "da-reviews.tsv"
    return df_da_reviews[COLUMNS]


def process_danish_some_sentiment() -> pd.DataFrame:
    df_danish_some_sentiment = pd.read_csv(
        DATA_RAW_PATHS["danish-some-sentiment"], sep="\t", header=None
    )
    df_danish_some_sentiment.columns = ["label", "text"]
    # The data has labels [-2, -1, 0, 1, 2].
    # Based on manual inspection, we consider 1 and 2 to be positive, and -2, -1, and 0 to be negative.
    binary_labels = (df_danish_some_sentiment["label"] > 0) * 1
    df_danish_some_sentiment["label"] = binary_labels
    df_danish_some_sentiment["language"] = "da"
    df_danish_some_sentiment["dataset"] = "danish-some-sentiment.tsv"
    return df_danish_some_sentiment[COLUMNS]


def process_da() -> pd.DataFrame:
    df_da = pd.read_csv(DATA_RAW_PATHS["da"])
    df_da["language"] = "da"
    df_da["dataset"] = "da.csv"
    return df_da[COLUMNS]


def process_en() -> pd.DataFrame:
    df_en = pd.read_csv(DATA_RAW_PATHS["en"])
    df_en["language"] = "en"
    df_en["dataset"] = "en.csv"
    return df_en[COLUMNS]


def process_no() -> pd.DataFrame:
    df_no = pd.read_csv(DATA_RAW_PATHS["no"])
    df_no["language"] = "no"
    df_no["dataset"] = "no.csv"
    return df_no[COLUMNS]


def process_sv() -> pd.DataFrame:
    df_sv = pd.read_csv(DATA_RAW_PATHS["sv"])
    df_sv["language"] = "sv"
    df_sv["dataset"] = "sv.csv"
    return df_sv[COLUMNS]


def merge(dfs) -> pd.DataFrame:
    df = pd.concat(dfs)[COLUMNS]
    df = df.dropna(subset=["text"])
    return df


def split_df_and_save(df):
    n = len(df)
    test_size = int(n * TEST_FRACTION)
    train_size = n - test_size - test_size
    df_train, df_val, df_test = np.split(
        df.sample(frac=1), [train_size, train_size + test_size]
    )
    df_train.to_csv(DATA_FINAL_PATHS["train"], index=False)
    df_val.to_csv(DATA_FINAL_PATHS["val"], index=False)
    df_test.to_csv(DATA_FINAL_PATHS["test"], index=False)

    print("\n")
    print(f"Train size: {len(df_train)}")
    print(f"Val size: {len(df_val)}")
    print(f"Test size: {len(df_test)}")
    print("\n")
    print("Train distribution:")
    print(df_train["label"].value_counts())
    print("\n")
    print("Val distribution:")
    print(df_val["label"].value_counts())
    print("\n")
    print("Test distribution:")
    print(df_test["label"].value_counts())


def main():
    dfs = [
        process_da_reviews(),
        process_danish_some_sentiment(),
        process_da(),
        process_en(),
        process_no(),
        process_sv(),
    ]
    df = merge(dfs)
    split_df_and_save(df)


if __name__ == "__main__":
    main()
