# sector_study

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Analysis of relationship between index fund sectors and financial news coverage.

## Progress

- [x] Complete [requirements.txt](./requirements.txt)
- [x] Set up virtual environment
- [x] Data install
- [ ] Hypothesis assertions
- [x] Data cleaning
  - [x] date-time standardization
  - [x] {maybe} source stock prices 2018-2020
- [x] Exploratory data analysis
- [x] Data pipelines
- [ ] Statistical analysis
  - [ ] independent variable definitions
- [ ] Feature analysis
- [ ] Feature creation

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         sector_study and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── sector_study   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes sector_study a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

---

# Using the Repo

## Data Sources

Data used can be sourced from:

- [thedevastator/cnbc-business-and-financial-news-dataset-450k](https://www.kaggle.com/datasets/thedevastator/cnbc-business-and-financial-news-dataset-450k)
- [notlucasp/financial-news-headlines](https://www.kaggle.com/datasets/notlucasp/financial-news-headlines)
- [camnugent/sandp500](https://www.kaggle.com/datasets/camnugent/sandp500)
- [amananandrai/ag-news-classification-dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
- [yeong-hwan/ticker-sector-industry](https://huggingface.co/datasets/yeong-hwan/ticker-sector-industry)
- [Adilbai/stock-dataset](https://huggingface.co/datasets/Adilbai/stock-dataset)
- {unused} [kurry/sp500_earnings_transcripts](https://huggingface.co/datasets/kurry/sp500_earnings_transcripts)

Place the csv files into `data/external/` and the `dataset.py` file should be able to prepare the data for this repository.

## Data Pipelines

Data pipelines can be viewed via the terminal (after activating the virtual environment) using

```
python sector_study/datasets.py --help
```

Further documentation about each pipeline is provided through the command-line toolset provided by the `typer` library. Implementation is still a work-in-progress.
