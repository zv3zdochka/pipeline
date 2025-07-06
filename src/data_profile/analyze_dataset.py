# analyze_dataset.py
import pandas as pd


def analyze_df(df: pd.DataFrame) -> None:
    """
    Анализирует уже загруженный DataFrame (dtype, пропуски, уникальные, пример).
    """
    info = []
    for col in df.columns:
        s = df[col]
        info.append({
            "column": col,
            "dtype": str(s.dtype),
            "non_null": s.notna().sum(),
            "nulls": len(s) - s.notna().sum(),
            "unique": s.nunique(dropna=True),
            "example": (s.dropna().iloc[0] if s.notna().any() else None)
        })
    summary = pd.DataFrame(info)
    print(summary.to_markdown(index=False))


def analyze_csv(path: str) -> None:
    """
    Читает CSV и вызывает analyze_df.
    """
    df = pd.read_csv(path)
    analyze_df(df)
