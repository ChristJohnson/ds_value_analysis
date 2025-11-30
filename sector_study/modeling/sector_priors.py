from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sector_study.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR


def assign_sectors_to_transcript_clusters(
    transcript_clusters_path: Path = INTERIM_DATA_DIR
    / "transcript_clusters.csv",
    company_table_path: Path = PROCESSED_DATA_DIR / "company_data_table.csv",
    output_path: Path | None = None,
):
    """Attach sector from company_data_table.csv to each ticker in transcript_clusters.csv."""
    clusters = pd.read_csv(transcript_clusters_path)
    companies = pd.read_csv(company_table_path)

    # Clean tickers
    clusters["ticker"] = clusters["ticker"].astype(str).str.strip().str.upper()
    companies["ticker"] = (
        companies["ticker"].astype(str).str.strip().str.upper()
    )

    merged = clusters.merge(
        companies[["ticker", "sector"]],
        on="ticker",
        how="left",
    )

    # Optionally write out
    if output_path:
        merged.to_csv(output_path, index=False)

    return merged


def build_sector_training_data(
    transcript_clusters_df: pd.DataFrame | None = None,
    transcript_clusters_path: Path = INTERIM_DATA_DIR
    / "transcript_clusters.csv",
    company_table_path: Path = PROCESSED_DATA_DIR / "company_data_table.csv",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Assemble (cluster_id, sector) pairs and split into train/test.

    Returns:
        X_train, X_test, y_train, y_test, label_encoder, merged_df
    """
    # Load and merge if a DF wasn’t provided
    if transcript_clusters_df is None:
        merged_df = assign_sectors_to_transcript_clusters(
            transcript_clusters_path=transcript_clusters_path,
            company_table_path=company_table_path,
        )
    else:
        merged_df = transcript_clusters_df.copy()
        # ensure sector is present
        if "sector" not in merged_df.columns:
            companies = pd.read_csv(company_table_path)
            merged_df["ticker"] = (
                merged_df["ticker"].astype(str).str.strip().str.upper()
            )
            companies["ticker"] = (
                companies["ticker"].astype(str).str.strip().str.upper()
            )
            merged_df = merged_df.merge(
                companies[["ticker", "sector"]], on="ticker", how="left"
            )

    # Drop rows without features or labels
    merged_df = merged_df.dropna(subset=["cluster_id", "sector"])
    if merged_df.empty:
        raise ValueError(
            "No training rows remain after attaching sectors to clusters."
        )

    # One-hot encode cluster IDs
    X = pd.get_dummies(merged_df["cluster_id"], prefix="cluster").to_numpy()

    # Encode sector labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(merged_df["sector"])

    # Guard stratification for small/imbalanced sets
    from collections import Counter

    counts = Counter(y)  # type: ignore
    stratify_arg = y if all(c >= 2 for c in counts.values()) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )

    return X_train, X_test, y_train, y_test, label_encoder, merged_df


# Example usage: fit a logistic regression on the one-hot cluster features
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, le, merged = build_sector_training_data(
        transcript_clusters_path=INTERIM_DATA_DIR / "transcript_clusters.csv",
        company_table_path=PROCESSED_DATA_DIR / "company_data_table.csv",
    )

    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_train, y_train)
    print(
        "Train acc:",
        clf.score(X_train, y_train),
        "Test acc:",
        clf.score(X_test, y_test),
    )
    print("Classes:", le.classes_)
