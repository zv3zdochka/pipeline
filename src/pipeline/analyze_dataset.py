# analyze_dataset.py

import pandas as pd
from typing import Optional, Dict, Any


def analyze_df(df: pd.DataFrame) -> None:
    """
    Analyze a loaded DataFrame:
     - column dtypes
     - number of non-null vs null
     - number of unique values
     - memory usage
     - basic examples (first 2 non-null for each)
    """
    # overall memory
    total_mem_mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"DataFrame memory footprint: {total_mem_mb:.2f} MB\n")

    # per-column summary
    info = []
    for col in df.columns:
        s = df[col]
        non_null = s.notna().sum()
        nulls = len(s) - non_null
        unique = s.nunique(dropna=True)
        example_vals = s.dropna().head(2).tolist() or [None]
        info.append({
            "column": col,
            "dtype": str(s.dtype),
            "non_null": non_null,
            "nulls": nulls,
            "unique": unique,
            "examples": example_vals
        })
    summary = pd.DataFrame(info)
    print(summary.to_markdown(index=False))


def analyze_csv(
        path: str,
        dtypes: Optional[Dict[str, Any]] = None,
        **read_csv_kwargs
) -> pd.DataFrame:
    """
    Read a CSV with optional explicit dtypes, analyze it,
    and return the loaded DataFrame.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    dtypes : dict, optional
        Column name → dtype mapping for pd.read_csv.
    **read_csv_kwargs
        Any additional kwargs to pass to pd.read_csv
        (e.g. parse_dates, nrows, usecols).
    """
    print(f"Loading '{path}' with dtypes={bool(dtypes)} and options={read_csv_kwargs}")
    df = pd.read_csv(path, dtype=dtypes or {}, **read_csv_kwargs)
    print("\n=== DataFrame Overview ===")
    analyze_df(df)
    print("\n=== First 2 Rows Preview ===")
    print(df.head(2).to_markdown())
    return df
