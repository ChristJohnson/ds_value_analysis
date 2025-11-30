from collections import Counter
from itertools import product

import joblib
import pandas as pd

from sector_study.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from sector_study.modeling.topic_clustering.cluster_forming import (
    get_nl_topics_from_transcripts,
)


def main():
    round_2_testing()


def round_2_testing():
    n_features = [
        2**17,
        2**18,
    ]
    m_clusters = [210, 340]
    b_batch_size = [
        2**6,
        2**7,
    ]
    n_rows = 10000

    grid = list(product(n_features, m_clusters, b_batch_size))

    # comment out Parallel(...) if satisfied with sanity-... files
    joblib.Parallel(n_jobs=-1, prefer="threads")(
        joblib.delayed(test_cluster_hyperparams)(
            n_feat, n_cluster, batch_size, n_rows
        )
        for n_feat, n_cluster, batch_size in grid
    )

    for n_feat, n_cluster, batch_size in grid:
        # call sanity check
        check_param_test(n_feat, n_cluster, batch_size)


def round_1_testing():
    n_features = [
        2**14,
        2**15,
        2**16,
        2**17,
        2**18,
    ]
    m_clusters = [10, 20, 30, 50, 80, 130, 210, 340]
    b_batch_size = [
        2**6,
        2**7,
        2**8,
        2**9,
    ]

    grid = list(product(n_features, m_clusters, b_batch_size))

    # # comment out Parallel(...) if satisfied with sanity-... files

    # joblib.Parallel(n_jobs=-1, prefer="threads")(
    #     joblib.delayed(test_cluster_hyperparams)(n_feat, n_cluster, batch_size)
    #     for n_feat, n_cluster, batch_size in grid
    # )

    for n_feat, n_cluster, batch_size in grid:
        # call sanity check
        check_param_test(n_feat, n_cluster, batch_size)


def check_param_test(
    n_features: int,
    n_cluster: int,
    b_batch_size: int,
):
    target = (
        INTERIM_DATA_DIR
        / f"sanity-{n_features}f-{n_cluster}c-{b_batch_size}b.csv"
    )

    merged = pd.read_csv(target).merge(
        pd.read_csv(PROCESSED_DATA_DIR / "company_data_table.csv")[
            ["ticker", "sector"]
        ],
        on="ticker",
        how="left",
    )
    merged = merged.dropna(subset=["cluster_id", "sector"])

    # majority sector per cluster
    majority = (
        merged.groupby("cluster_id")["sector"]
        .agg(lambda s: Counter(s).most_common(1)[0][0])
        .to_dict()
    )

    # assign majority predictions
    merged["pred_sector"] = merged["cluster_id"].map(majority)

    # compute purity/accuracy
    acc = (merged["sector"] == merged["pred_sector"]).mean()
    print(
        f"sanity-{n_features}f-{n_cluster}c-{b_batch_size}b.csv",
        "Cluster majority-vote accuracy:",
        acc,
    )


def test_cluster_hyperparams(
    n_features: int,
    n_cluster: int,
    b_batch_size: int,
    n_rows: int = 1000,
):
    get_nl_topics_from_transcripts(
        # features_path: Path = INTERIM_DATA_DIR / "earnings_transcripts.csv",
        # model_path: Path = MODELS_DIR / "topic_clustering.pkl",
        # n_features: int = 2**18,  # 262,144 dims
        n_features=n_features,
        # n_clusters: int = 50,
        n_clusters=n_cluster,
        n_init=5,
        # batch_size: int = 1024,
        batch_size=b_batch_size,
        # observation_sample_limit: int | None = None,
        observation_sample_limit=n_rows,
    )

    # out_path=INTERIM_DATA_DIR / f"sanity-{n_features}f-{n_clusters}c-{batch_size}b.csv",


if __name__ == "__main__":
    main()
