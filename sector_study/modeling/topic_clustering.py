import html
from pathlib import Path
import re

import joblib
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import HashingVectorizer
from tqdm import tqdm
import typer

from sector_study.config import INTERIM_DATA_DIR, MODELS_DIR

app = typer.Typer()


@app.command(name="headline-to-topic-cluster")
def assign_headline_topic_clusters_to_csv(
    # -----------------------------------------
    features_path: Path = INTERIM_DATA_DIR / "headlines.csv",
    model_path: Path = MODELS_DIR / "topic_clustering.pkl",
    # -----------------------------------------
):
    """Load a saved vectorizer/clusterer, assign cluster labels to headlines, and overwrite the CSV with a new `cluster` column."""
    # -----------------------------------------
    logger.info("Loading features and model...")

    headlines = pd.read_csv(features_path)
    vectorizer, clusterer = joblib.load(model_path)

    # -----------------------------------------
    logger.info("Vectorizing input text...")

    texts = preprocess(headlines["headline"])
    X_headlines = vectorizer.transform(texts)

    # -----------------------------------------
    logger.info("Headline topic assignment...")

    headline_labels = clusterer.predict(X_headlines)
    headlines["cluster"] = headline_labels

    # -----------------------------------------
    logger.info("Saving feature set with cluster assignments...")
    headlines.to_csv(features_path, index=False)


@app.command(name="find-topic-clusters")
def train_transcript_topic_cluster_model(
    # -----------------------------------------
    features_path: Path = INTERIM_DATA_DIR / "earnings_transcripts.csv",
    model_path: Path = MODELS_DIR / "topic_clustering.pkl",
    # -----------------------------------------
):
    """Stream transcripts, fit the hashing vectorizer + MiniBatchKMeans model, normalize centroids for cosine-ish inference, and persist both to disk."""
    # -----------------------------------------
    logger.info("Initializing variables...")
    # A cursed artifact that attempted to allocate 8.29 GiB for a mutual reachability matrix
    # clusterer = HDBSCAN(metric="euclidean")

    vectorizer = HashingVectorizer(
        n_features=2**18,  # 262,144 dims
        alternate_sign=False,
        norm="l2",
        stop_words="english",
    )

    clusterer = MiniBatchKMeans(n_clusters=50, batch_size=2048)

    # -----------------------------------------
    logger.info("Training on transcripts...")
    # We use partial_fit so we don't need to materialize all transcripts in memory
    for X_chunk in iterate_vectorized_chunks(
        csv_path=features_path,
        text_column="content",
        vectorizer=vectorizer,
        chunksize=2048,
        tqdm_label="earnings_transcript.csv",
    ):
        clusterer.partial_fit(X_chunk)

    # -----------------------------------------
    logger.info(
        "Finished training; normalizing centroids for cosine-ish inference..."
    )

    centers = clusterer.cluster_centers_
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0  # avoid divide-by-zero
    clusterer.cluster_centers_ = centers / norms

    # -----------------------------------------
    logger.info("Saving model...")
    joblib.dump((vectorizer, clusterer), model_path)
    logger.info(f"Model saved to: {model_path}")


def iterate_vectorized_chunks(
    csv_path: Path,
    text_column: str,
    vectorizer,
    chunksize: int = 2048,
    tqdm_label=None,
):
    """Yield vectorized text chunks to avoid loading the entire CSV."""
    reader = pd.read_csv(csv_path, index_col=0, chunksize=chunksize)
    for chunk in tqdm(reader, tqdm_label or Path(csv_path).name):
        texts = preprocess(chunk[text_column])
        yield vectorizer.transform(texts)


def preprocess(series):
    texts = (
        series.fillna("")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .tolist()  # keep as list for downstream preprocessing
    )
    texts = [preprocess_token(t) for t in texts]
    return texts


def preprocess_token(t):
    t = html.unescape(str(t))
    t = t.lower()
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"&?#?[a-z0-9]+;s?", " ", t)  # kill &gt; &lt; &amp; etc
    t = re.sub(r"[^a-z0-9\s\-\.\,]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


if __name__ == "__main__":
    app()
