from pathlib import Path

import joblib
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import HashingVectorizer
from tqdm import tqdm

from sector_study.config import INTERIM_DATA_DIR, MODELS_DIR
from sector_study.modeling.topic_clustering import preprocess


def get_nl_topics_from_transcripts(
    # -----------------------------------------
    features_path: Path = INTERIM_DATA_DIR / "earnings_transcripts.csv",
    model_path: Path = MODELS_DIR / "topic_clustering.pkl",
    n_features: int = 2**18,  # 262,144 dims
    n_clusters: int = 48,
    n_init: int = 8,
    batch_size: int = 1024,
    observation_sample_limit: int | None = None,
    # -----------------------------------------
):
    """Stream transcripts, fit the hashing vectorizer + MiniBatchKMeans model, normalize centroids for cosine-ish inference, and persist both to disk."""
    # -----------------------------------------
    logger.info("Initializing variables...")
    # A cursed artifact that attempted to allocate 8.29 GiB for a mutual reachability matrix
    # clusterer = HDBSCAN(metric="euclidean")

    vectorizer = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        stop_words="english",
    )

    clusterer = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        n_init=n_init,
    )

    # -----------------------------------------
    logger.info(
        f"Modelling {observation_sample_limit} transcripts on {n_features} features, {n_clusters} clusters, batch_size {batch_size}, with {n_init} n_inits..."
    )
    df_sample = None
    if observation_sample_limit:
        # df_sample = fit_transcript_clusters_sample(
        #     clusterer=clusterer,
        #     vectorizer=vectorizer,
        #     csv_path=features_path,
        #     text_column="content",
        #     observation_sample_limit=observation_sample_limit,
        # )
        df_sample = fit_transcript_clusters_sample_streaming(
            clusterer=clusterer,
            vectorizer=vectorizer,
            csv_path=features_path,
            text_column="content",
            sample_limit=observation_sample_limit,
            chunksize=512,
        )
    else:
        fit_transcript_clusters_streaming(
            clusterer=clusterer,
            vectorizer=vectorizer,
            csv_path=features_path,
            text_column="content",
            tqdm_label="earnings_transcript.csv",
        )

    # -----------------------------------------
    logger.info(
        "Finished training; normalizing centroids for cosine-ish inference..."
    )
    clusterer.cluster_centers_ = normalize_centers(clusterer.cluster_centers_)

    # -----------------------------------------
    logger.info("Saving transcript clusters...")
    if observation_sample_limit and df_sample is not None:
        save_sanity_clusters(
            df=df_sample,
            vectorizer=vectorizer,
            clusterer=clusterer,
            out_path=INTERIM_DATA_DIR
            / f"sanity-{n_features}f-{n_clusters}c-{batch_size}b.csv",
        )
    else:
        export_transcript_clusters_streaming(
            csv_path=features_path,
            vectorizer=vectorizer,
            kmeans=clusterer,
            out_path=INTERIM_DATA_DIR / "transcript_clusters.csv",
        )

    # -----------------------------------------
    if not observation_sample_limit and df_sample is None:
        logger.info("Saving model...")
        joblib.dump((vectorizer, clusterer), model_path)
        logger.info(f"Model saved to: {model_path}")


def fit_transcript_clusters_streaming(
    clusterer,
    vectorizer,
    csv_path: Path,
    text_column: str = "content",
    chunksize: int = 2048,
    tqdm_label: str | None = None,
):
    """Stream batches through MiniBatchKMeans to avoid materializing the entire dataset."""
    for X_chunk in iterate_vectorized_chunks(
        csv_path=csv_path,
        text_column=text_column,
        vectorizer=vectorizer,
        chunksize=chunksize,
        tqdm_label=tqdm_label,
    ):
        clusterer.partial_fit(X_chunk)


def fit_transcript_clusters_sample(
    clusterer,
    vectorizer,
    csv_path: Path,
    text_column: str,
    observation_sample_limit: int,
) -> pd.DataFrame:
    """Fit KMeans on a materialized sample for quicker experimentation."""
    logger.trace(f"Training on {observation_sample_limit} samples...")
    df_sample = pd.read_csv(csv_path, nrows=observation_sample_limit)
    texts = preprocess(df_sample[text_column])
    X_sample = vectorizer.transform(texts)
    clusterer.fit(X_sample)
    return df_sample


def fit_transcript_clusters_sample_streaming(
    clusterer,
    vectorizer,
    csv_path,
    text_column,
    sample_limit,
    chunksize=2048,
):
    """Partial-fit on up to sample_limit rows while streaming; also return the sampled df."""
    seen = 0
    sampled_chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        chunk = chunk.dropna(subset=[text_column])
        take = min(sample_limit - seen, len(chunk))
        if take <= 0:
            break
        # keep a copy of what we used
        sampled_chunks.append(chunk.iloc[:take].copy())
        texts = preprocess(chunk[text_column].iloc[:take])
        X_chunk = vectorizer.transform(texts)
        clusterer.partial_fit(X_chunk)
        seen += take
        if seen >= sample_limit:
            break
    if not sampled_chunks:
        return None
    return pd.concat(sampled_chunks, ignore_index=True)


def normalize_centers(centers: np.ndarray) -> np.ndarray:
    """Normalize cluster centroids to unit length to approximate cosine similarity during inference."""
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0  # avoid divide-by-zero
    return centers / norms


def save_sanity_clusters(
    df: pd.DataFrame,
    vectorizer,
    clusterer,
    out_path: Path,
    text_column: str = "content",
):
    """Persist a lightweight sanity-check clustering sample."""
    X_sanity = vectorizer.transform(df[text_column])
    df["cluster_id"] = clusterer.predict(X_sanity)
    cols = [
        c
        for c in ["ticker", "cluster_id", "year", "quarter"]
        if c in df.columns
    ]
    df[cols].to_csv(out_path)


def export_transcript_clusters_streaming(
    csv_path, vectorizer, kmeans, out_path, chunksize=2048
):
    first = True
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        chunk = chunk.dropna(subset=["ticker", "content"])
        chunk["ticker"] = chunk["ticker"].astype(str).str.strip().str.upper()
        X_chunk = vectorizer.transform(chunk["content"])
        chunk["cluster_id"] = kmeans.predict(X_chunk)
        cols = [
            c
            for c in ["ticker", "cluster_id", "year", "quarter"]
            if c in chunk.columns
        ]
        chunk[cols].to_csv(
            out_path, mode="w" if first else "a", header=first, index=False
        )
        first = False


def iterate_vectorized_chunks(
    csv_path: Path,
    text_column: str,
    vectorizer,
    chunksize: int = 512,
    tqdm_label=None,
):
    """Yield vectorized text chunks to avoid loading the entire CSV."""
    reader = pd.read_csv(csv_path, index_col=0, chunksize=chunksize)
    # for chunk in tqdm(reader, tqdm_label or Path(csv_path).name):
    for chunk in reader:
        texts = preprocess(chunk[text_column])
        yield vectorizer.transform(texts)
