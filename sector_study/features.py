from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import pandas as pd

from sector_study.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


@app.command()
def company_data(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "stocks.csv",
    output_path: Path = PROCESSED_DATA_DIR / "company_data_table.csv",
    # -----------------------------------------
):
    """
    Extract features from company:
      - Ticker (ID)
      - Sector
      - Industry

    Using following files:
      - "sector_industry_info.csv"
      - "stocks.csv"
    """

    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info(
        "Generating company data features from sector_industry_info.csv..."
    )

    # load sector_industry_info
    # Columns: ticker,sector,industry
    df = pd.read_csv(RAW_DATA_DIR.joinpath("sector_industry_info.csv"))
    df = df.set_index("ticker")

    logger.info("Adding entries from stocks.csv")

    # Columns: date,open,high,low,close,volume,ticker,source,sector,industry
    stocks = pd.read_csv(input_path)
    # only these columns for lookup table
    stocks_tickers = stocks["ticker"].unique()
    all_tickers = pd.Index(df.index).union(stocks_tickers)

    df = df.reindex(all_tickers)

    logger.info(f"shape of ticker-sector lookup table: {df.shape}")

    df.to_csv(output_path)
    # -----------------------------------------


if __name__ == "__main__":
    app()
