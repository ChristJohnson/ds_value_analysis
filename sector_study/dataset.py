import os
from pathlib import Path
import re
import shutil

from datasets import load_dataset
import kagglehub
from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer

from sector_study.config import (
    EXTERNAL_DATA_DIR,
    HF_DATASETS,
    KAGGLE_DATASETS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

app = typer.Typer()


@app.callback()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    logger.info("Processing dataset...")
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")

    # -----------------------------------------
    logger.success("Processing dataset complete.")


@app.command()
def generate_flat_raw():
    news_keys = [
        "cnbc_news_datase",
        "cnbc_headlines",
        "guardian_headlines",
        "reuters_headlines",
        "test",
        "train",
    ]

    cols = {
        "cnbc_news_datase": ["title", "published_at"],
        "cnbc_headlines": ["Headlines", "Time"],
        "guardian_headlines": ["Headlines", "Time"],
        "reuters_headlines": ["Headlines", "Time"],
        "test": ["Title"],
        "train": ["Title"],
    }

    ext_data: dict[str, pd.DataFrame] = {}
    load_external(ext_data)

    df = extract_columns(ext_data, news_keys, cols)
    df.columns = ["headline", "date", "source"]

    logger.info(f"{df.shape}\n{df.head()}")
    logger.trace(
        f"Writing dataframe to {RAW_DATA_DIR.joinpath('headlines.csv')}"
    )
    df.to_csv(RAW_DATA_DIR.joinpath("headlines.csv"), index=False)

    df = ext_data["all_stocks_5yr"]
    df = df.merge(
        ext_data["sector_industry_info"], left_on="Name", right_on="Ticker"
    )
    df = df.drop(columns=["Unnamed: 0", "Ticker"])

    logger.info(f"{df.shape}\n{df.head()}")
    logger.trace(f"Writing dataframe to {RAW_DATA_DIR.joinpath('stocks.csv')}")
    df.to_csv(RAW_DATA_DIR.joinpath("stocks.csv"))


def extract_columns(external_data, frame_targets, to_extract):
    dfs = []
    for key in frame_targets:
        if key in external_data and key in to_extract:
            df = external_data[key]
            cols = to_extract[key]

            available = [c for c in cols if c in df.columns]
            if not available:
                continue

            # Standardize to the target column names
            temp = df[available].copy()
            temp.columns = ["col_" + str(i) for i in range(len(available))]
            temp["source"] = key
            dfs.append(temp)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_external(external_data: dict[str, pd.DataFrame]):
    # EXTERNAL_DATA_DIR + "all_stock_5yr.csv"
    external_files = Path(EXTERNAL_DATA_DIR).rglob("*csv")

    for file in external_files:
        external_data[file.stem] = pd.read_csv(file)

    logger.success("External data loaded!")


@app.command()
def install_datasets(output_path: Path = EXTERNAL_DATA_DIR):
    """
    Installs all datasets from `KAGGLE_DATASETS` and `HF_DATASETS`
    variables in `sector_study.config`.
    """
    logger.info("installing kaggle datasets")
    logger.trace(", ".join(KAGGLE_DATASETS))

    for dataset in tqdm(
        KAGGLE_DATASETS, "Kaggle datasets", total=len(KAGGLE_DATASETS)
    ):
        install_kaggle_dataset(dataset, output_path)

    logger.info("Installing HuggingFace datasets")
    logger.trace(", ".join(HF_DATASETS))
    # Login using e.g. `huggingface-cli login` to access this dataset
    df = pd.read_csv(
        "hf://datasets/yeong-hwan/ticker-sector-industry/sector_industry_info.csv"
    )
    df.to_csv("../data/raw/sector_industry_info.csv")


def install_kaggle_dataset(dataset: str, output_path: Path = EXTERNAL_DATA_DIR):
    if re.match(r"^[A-z0-9-]+/[A-z0-9-]+$", dataset) is None:
        logger.error(f"{dataset} is not a valid kaggle dataset! Skipping...")
        return

    logger.info("downloading " + dataset)

    path = kagglehub.dataset_download(dataset, force_download=False)
    files = Path(path).rglob(
        "*csv" if dataset != "camnugent/sandp500" else "*all_stocks*"
    )

    logger.trace(dataset + " was downloaded to " + path)

    for origin_file in files:
        link_dest = output_path.joinpath(origin_file.relative_to(path))
        logger.info(f"copying file {origin_file} -> {link_dest}")
        if not os.path.exists(output_path.parent):
            logger.warning(
                f"{output_path.parent} does not exist! Creating now..."
            )
            os.makedirs(output_path.parent, exist_ok=True)
        shutil.copy(origin_file, link_dest)  # windows doesn't like symlinks >:(


def install_hf_dataset(dataset: str, output_path: Path = EXTERNAL_DATA_DIR):
    df = load_dataset("Adilbai/stock-dataset")
    df.with_format("pandas").to_csv(output_path.joinpath("stock_dataset.csv"))


if __name__ == "__main__":
    app()
