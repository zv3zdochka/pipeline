import pandas as pd
from tabulate import tabulate


def print_dataset_overview(df: pd.DataFrame) -> None:
    """
    Print detailed DataFrame overview:
    - shape and memory usage
    - summary by dtype: number of columns, total and percent missing, average and median unique values
    - top 5 columns by missing values
    - total missing and unique values
    """
    rows, cols = df.shape
    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"[INFO] DataFrame shape: {rows} rows, {cols} columns")
    print(f"[INFO] Memory usage: {mem_mb:.2f} MB")

    col_info = pd.DataFrame({
        'column': df.columns,
        'dtype': df.dtypes.astype(str).values,
        'missing': df.isna().sum().values,
        'unique': df.nunique(dropna=False).values
    })

    summary = (
        col_info.groupby('dtype')
        .agg(
            num_cols=('column', 'count'),
            total_missing=('missing', 'sum'),
            avg_unique=('unique', 'mean'),
            med_unique=('unique', 'median')
        )
        .reset_index()
    )
    summary['pct_missing'] = (summary['total_missing'] / (rows * summary['num_cols']) * 100).round(2)

    print("\n[SUMMARY BY DTYPE]")
    print(tabulate(summary, headers='keys', tablefmt='github', showindex=False))

    top_missing = col_info.sort_values('missing', ascending=False).head(5)
    print("\n[TOP 5 COLUMNS BY MISSING VALUES]")
    print(tabulate(
        top_missing[['column', 'dtype', 'missing', 'unique']],
        headers=['Column', 'Dtype', 'Missing', 'Unique'],
        tablefmt='github',
        showindex=False
    ))

    total_missing = col_info['missing'].sum()
    total_unique = col_info['unique'].sum()
    print(f"\n[INFO] Total missing values: {total_missing}")
    print(f"[INFO] Total unique values across all columns: {total_unique}")
