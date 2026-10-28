"""
LAYER 1 - Step 1 & 2: Data Collection and Data Integration.

Covers the "Collect and combine loan-related data from CSV/Excel files, APIs and
web sources" objective from the proposal. The integrate() function accepts a
list of heterogeneous sources and returns one unified DataFrame with a
`data_source` column so you can trace every row back to where it came from.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config


def _read_one(path: str | Path, usecols=None, nrows=None) -> pd.DataFrame:
    """Read a single source. Dispatches on file extension."""
    path = Path(path)
    suffix = "".join(path.suffixes).lower()

    if suffix.endswith((".csv", ".csv.gz", ".gz", ".zip")):
        # low_memory=False stops pandas from guessing different dtypes per chunk,
        # which is what produces the DtypeWarning spam on this dataset.
        available = pd.read_csv(path, nrows=0).columns.tolist()
        cols = [c for c in usecols if c in available] if usecols else None
        missing = sorted(set(usecols) - set(available)) if usecols else []
        if missing:
            print(f"  ! columns not present in {path.name}: {missing}")
        if nrows:
            return pd.read_csv(path, usecols=cols, nrows=nrows, low_memory=False)
        chunks = []
        for chunk in pd.read_csv(path, usecols=cols, chunksize=250000, low_memory=False):
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)

    if suffix.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, nrows=nrows)
        if usecols:
            df = df[[c for c in usecols if c in df.columns]]
        return df

    if suffix.endswith(".parquet"):
        df = pd.read_parquet(path)
        if usecols:
            df = df[[c for c in usecols if c in df.columns]]
        return df.head(nrows) if nrows else df

    if suffix.endswith(".json"):
        return pd.read_json(path, lines=True, nrows=nrows)

    raise ValueError(f"Unsupported file type: {path}")


def integrate(sources, usecols=None, nrows=None) -> pd.DataFrame:
    """
    Combine several sources into one DataFrame.

    sources : list of paths, or a single path
    Returns a DataFrame with an added `data_source` column.
    """
    if isinstance(sources, (str, Path)):
        sources = [sources]

    frames = []
    for src in sources:
        print(f"  reading {Path(src).name} ...")
        df = _read_one(src, usecols=usecols, nrows=nrows)
        df["data_source"] = Path(src).name
        frames.append(df)
        print(f"    -> {len(df):,} rows x {df.shape[1]} cols")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    print(f"  integrated: {len(combined):,} rows x {combined.shape[1]} cols")
    return combined


def load_raw(path=None, usecols=None, nrows=None) -> pd.DataFrame:
    """Convenience wrapper for the single-file default case."""
    path = path or config.RAW_DATA_PATH
    usecols = usecols if usecols is not None else config.USECOLS
    nrows = nrows if nrows is not None else config.NROWS

    if not Path(path).exists():
        raise FileNotFoundError(
            f"\nCould not find {path}\n"
            "Download the dataset from:\n"
            "  https://www.kaggle.com/datasets/wordsforthewise/lending-club\n"
            "and place accepted_2007_to_2018Q4.csv.gz in the data/ folder.\n"
            "Or run:  python scripts/make_sample_data.py   to generate synthetic "
            "test data with the same schema."
        )

    return integrate([path], usecols=usecols, nrows=nrows)


def fetch_from_api(url: str, params: dict | None = None) -> pd.DataFrame:
    """
    Optional third source to satisfy the 'APIs and web sources' objective.

    Example uses: FRED unemployment rate by state, or a public macro indicator
    that you then join onto the loans by addr_state / issue month. Keeping this
    here means you can demo genuine multi-source integration rather than just
    reading one CSV.
    """
    import requests

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    return pd.json_normalize(payload)


def enrich_with_state_macro(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join an external macro table onto the loan book.

    `macro` is expected to have columns [addr_state, year, <indicator...>].
    """
    df = df.copy()
    if "issue_d" in df.columns:
        df["issue_year"] = pd.to_datetime(
            df["issue_d"], format="%b-%Y", errors="coerce"
        ).dt.year
    return df.merge(macro, on=["addr_state", "issue_year"], how="left")


if __name__ == "__main__":
    frame = load_raw()
    print(frame.head())
