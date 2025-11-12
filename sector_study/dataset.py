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
    # for i in tqdm(range(10), total=10):
    #     if i == 5:
    #         logger.info("Something happened for iteration 5.")

    # -----------------------------------------
    logger.success("Processing dataset complete.")


@app.command()
def combine_datasets():
    combine_headlines()
    combine_datasets()


def combine_headlines():
    headlines = pd.read_csv(RAW_DATA_DIR.joinpath("headlines.csv"))
    headlines.to_csv(PROCESSED_DATA_DIR.joinpath("headlines.csv"))


def combine_stocks():
    all_stocks_5yr = pd.read_csv(RAW_DATA_DIR.joinpath("all_stocks_5yr.csv"))
    all_stocks_5yr.date = pd.to_datetime(all_stocks_5yr.date, utc=True)
    stock_dataset = pd.read_csv(RAW_DATA_DIR.joinpath("stock_dataset.csv"))
    sector_industry = pd.read_csv(
        RAW_DATA_DIR.joinpath("sector_industry_info.csv")
    )

    df = pd.concat([all_stocks_5yr, stock_dataset])
    df.date = pd.to_datetime(df.date)

    df = pd.merge(df, sector_industry, on="ticker")

    df.to_csv(PROCESSED_DATA_DIR.joinpath("stocks.csv"), index=False)


@app.command()
def clean_datasets(output_path: Path = RAW_DATA_DIR):
    """Go through each dataset, one-by-one, for column-wise cleaning.

    Standard column names:
    - `date`: the date-time column
    - `headline`: contains headline / article title
    - `source`: name of original dataset / repo
    - `open`: the opening value of a stock
    - `close`: the closing value of a stock
    - `ticker`: the label for a stock
    - `volume`: the number of stocks traded in an observation
    """
    clean_cnbc_news_datase()
    clean_fin_news_headlines()
    clean_all_stocks_5yr()
    clean_stock_dataset()
    clean_news_classification()
    clean_sector_industry_info()
    clean_earnings_transcript()


def clean_cnbc_news_datase():
    df = pd.read_csv(
        EXTERNAL_DATA_DIR.joinpath("cnbc_news_datase.csv"), index_col=0
    )

    df.columns = [
        "index",
        "headline",
        "url",
        "date",
        "author",
        "publisher",
        "short_description",
        "keywords",
        "header_image",
        "raw_description",
        "description",
        "scraped_at",
    ]

    intersting_columns = [
        "headline",
        "date",
    ]

    df = df[intersting_columns]

    df["source"] = pd.Series(
        [
            "thedevastator/cnbc-business-and-financial-news-dataset-450k"
            for _ in range(df.shape[0])
        ]
    )

    df["date"] = pd.to_datetime(df["date"])

    df.to_csv(RAW_DATA_DIR.joinpath("cnbc_news_dataset.csv"))


def clean_fin_news_headlines():
    cnbc = pd.read_csv(EXTERNAL_DATA_DIR.joinpath("cnbc_headlines.csv"))
    cnbc.columns = ["headline", "date", "description"]
    guard = pd.read_csv(EXTERNAL_DATA_DIR.joinpath("guardian_headlines.csv"))
    guard.columns = ["date", "headline"]
    reut = pd.read_csv(EXTERNAL_DATA_DIR.joinpath("reuters_headlines.csv"))
    reut.columns = ["headline", "date", "description"]

    reut.date = pd.to_datetime(reut.date, format="%b %d %Y", utc=True)

    # lord forgive me
    cnbc.date = pd.to_datetime(
        (
            cnbc.date.str.replace(
                r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun)", "", regex=True
            )  # remove weekday
            .str.replace(r"ET", "", regex=True)  # remove ET
            .str.replace(r"[\s,\.]+", " ", regex=True)  # remove excess space
            .str.replace(
                r"(?<!\d)(\d)(?=[:\s])", r"0\1", regex=True
            )  # pad 0 for %I if needed
            .str.replace(
                r"(?<=\d{2}\s)(\w{3}).*(?=\s\d{4})", r"\1", regex=True
            )  # replace month with first three letters
            .str.strip()
        ),
        format="%I:%M %p %d %b %Y",
        utc=True,
    )

    bad_dates = guard.date[guard.date.str.match(r"^[A-z]")].count()
    logger.warning(
        f"{bad_dates} entries with no day... Assigning to first day of month"
    )

    guard.date = pd.to_datetime(
        (
            guard.date.str.replace(
                r"^([A-z]+)", r"1-\1", regex=True
            ).str.replace(  # DECISION: map all dates
                r"\-", " ", regex=True
            )  # replace
        ),
        format="%d %b %y",
        utc=True,
    )

    df = pd.concat([cnbc, guard, reut])

    intersting_columns = [
        "headline",
        "date",
    ]

    df = df[intersting_columns]

    df["source"] = pd.Series(
        ["notlucasp/financial-news-headlines" for _ in range(df.shape[0])]
    )

    df.to_csv(RAW_DATA_DIR.joinpath("fin_news_headlines.csv"))


def clean_all_stocks_5yr():
    df = pd.read_csv(EXTERNAL_DATA_DIR.joinpath("all_stocks_5yr.csv"))
    df.date = pd.to_datetime(df.date)
    df.columns = ["date", "open", "high", "low", "close", "volume", "ticker"]

    df["source"] = pd.Series(["camnugent/sandp500" for _ in range(df.shape[0])])

    df.to_csv(RAW_DATA_DIR.joinpath("all_stocks_5yr.csv"), index=False)


def clean_stock_dataset():
    df = pd.read_csv(EXTERNAL_DATA_DIR.joinpath("stock_dataset.csv"))
    interested_variables = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Ticker",
    ]

    df = df[interested_variables]
    df.columns = ["date", "open", "high", "low", "close", "volume", "ticker"]

    df.date = pd.to_datetime(df.date, utc=True)

    df["source"] = pd.Series(
        ["Adilbai/stock-dataset" for _ in range(df.shape[0])]
    )
    df.to_csv(RAW_DATA_DIR.joinpath("stock_dataset.csv"), index=False)


def clean_news_classification():
    df_test = pd.read_csv(EXTERNAL_DATA_DIR.joinpath("test.csv"))
    df_test.columns = ["class", "headline", "description"]
    df_test["subset"] = pd.Series(["test" for _ in range(df_test.shape[0])])

    df_train = pd.read_csv(EXTERNAL_DATA_DIR.joinpath("train.csv"))
    df_train.columns = ["class", "headline", "description"]
    df_train["subset"] = pd.Series(["train" for _ in range(df_train.shape[0])])

    interested_columns = ["headline", "subset"]

    df_test = df_test[interested_columns]
    df_train = df_train[interested_columns]

    df = pd.concat([df_test, df_train])

    df["source"] = pd.Series(
        [
            "amananandrai/ag-news-classification-dataset"
            for _ in range(df_train.shape[0])
        ]
    )

    df.to_csv(PROCESSED_DATA_DIR.joinpath("test_train.csv"), index=False)


def clean_sector_industry_info():
    df = pd.read_csv(EXTERNAL_DATA_DIR.joinpath("sector_industry_info.csv"))
    interested_columns = ["Ticker", "Sector", "Industry"]
    df = df[interested_columns]
    df.columns = ["ticker", "sector", "industry"]
    df.to_csv(RAW_DATA_DIR.joinpath("sector_industry_info.csv"), index=False)


# WARN: This one kinda chugs
def clean_earnings_transcript():
    df = pd.read_csv(EXTERNAL_DATA_DIR.joinpath("earnings_transcripts.csv"))
    df.to_csv(RAW_DATA_DIR.joinpath("earnings_transcripts.csv"))

    interested_rows = [
        "symbol",
        "quarter",
        "year",
        "date",
        "content",
        "structured_content",
        "company_name",
        "company_id",
    ]

    df = df[interested_rows]
    df.columns = [
        "ticker",
        "quarter",
        "year",
        "date",
        "content",
        "structured_content",
        "company_name",
        "company_id",
    ]

    df["source"] = pd.Series(
        ["kurry/sp500_earnings_transcripts" for _ in range(df.shape[0])]
    )

    df.to_csv(PROCESSED_DATA_DIR.joinpath("earnings_transcripts.csv"))


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
    for dataset in tqdm(
        HF_DATASETS, "HuggingFace datasets", total=len(HF_DATASETS)
    ):
        install_hf_dataset(dataset, output_path)

    logger.success(f"All raw datasets installed to {output_path}")


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
    ds = load_dataset(dataset)

    file_name = re.search(r"(?<=/)([\w-]+)", dataset)
    if not file_name:
        file_name = "failed_parse"
    else:
        file_name = file_name.group(0)

    for key, df in ds.items():  # type: ignore
        df.to_pandas().to_csv(output_path.joinpath(f"{file_name}-{key}.csv"))


if __name__ == "__main__":
    app()
