import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

def expand_dataset(
    df: pd.DataFrame,
    n: float = 6.0,
    noise_sigma: float = 0.0,
    random_state: int | None = 42,
) -> pd.DataFrame:
    label_col = "microtrend_label"
    if label_col not in df.columns:
        raise KeyError(f"'{label_col}' not found in dataframe")

    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if c != label_col
    ]
    X_tmp = (
        df[numeric_cols]
        .ffill()
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )
    y = df[label_col].to_numpy()

    counts = pd.Series(y).value_counts()
    majority = counts.idxmax()
    strategy = {lbl: int(counts[lbl] * n) for lbl in counts.index if lbl != majority}
    if not strategy:
        return df.copy()

    k_neighbors = max(1, min(5, min(counts[lbl] for lbl in strategy) - 1))
    smote = SMOTE(
        sampling_strategy=strategy,
        k_neighbors=k_neighbors,
        random_state=random_state,
    )
    X_res, y_res = smote.fit_resample(X_tmp, y)

    orig_len = len(df)
    X_syn = X_res[orig_len:]
    y_syn = y_res[orig_len:]

    if noise_sigma > 0 and len(X_syn):
        rng = np.random.default_rng(random_state)
        X_syn += rng.normal(0.0, noise_sigma, X_syn.shape)

    syn_df = pd.DataFrame(X_syn, columns=numeric_cols, dtype=np.float32)
    syn_df[label_col] = y_syn

    if "ts" in df.columns:
        ts_dtype = df["ts"].dtype
        last_ts = df["ts"].dropna().iloc[-1]
        non_null_ts = df["ts"].dropna()
        if len(non_null_ts) >= 2:
            step = non_null_ts.diff().iloc[-20:].median()
            if pd.isna(step) or step == pd.Timedelta(0):
                step = pd.Timedelta(minutes=5)
        else:
            step = pd.Timedelta(minutes=5)
        synth_ts = [last_ts + step * (i + 1) for i in range(len(syn_df))]
        syn_df.insert(0, "ts", pd.Series(synth_ts, dtype=ts_dtype))

    non_num_cols = [
        c for c in df.columns if c not in numeric_cols + ["ts", label_col]
    ]
    for col in non_num_cols:
        base_col = df[col]
        if pd.api.types.is_categorical_dtype(base_col):
            if len(base_col.mode()) > 0:
                default_val = base_col.mode().iloc[0]
            elif len(base_col.cat.categories) > 0:
                default_val = base_col.cat.categories[0]
            else:
                default_val = "unknown"
            syn_df[col] = pd.Series([default_val] * len(syn_df), dtype=base_col.dtype)
        elif pd.api.types.is_numeric_dtype(base_col):
            default_val = base_col.mode().iloc[0] if len(base_col.mode()) else 0
            syn_df[col] = pd.Series([default_val] * len(syn_df), dtype=base_col.dtype)
        else:
            default_val = base_col.mode().iloc[0] if len(base_col.mode()) else ""
            syn_df[col] = pd.Series([default_val] * len(syn_df), dtype=base_col.dtype)

    result = pd.concat([df, syn_df], ignore_index=True).reindex(columns=df.columns)

    if result.isna().values.any():
        raise ValueError("expand_dataset produced missing values – проверьте заполнение признаков.")

    return result
