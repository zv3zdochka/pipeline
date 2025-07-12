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

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != label_col]
    X_tmp = df[numeric_cols].ffill().fillna(0).to_numpy(dtype=np.float32)
    y_tmp = df[label_col].to_numpy()

    counts = pd.Series(y_tmp).value_counts()
    majority = counts.idxmax()
    strategy = {lbl: int(counts[lbl] * n) for lbl in counts.index if lbl != majority}
    if not strategy:
        return df.copy()

    k_neighbors = max(1, min(5, min(counts[lbl] for lbl in strategy) - 1))
    smote = SMOTE(sampling_strategy=strategy, k_neighbors=k_neighbors, random_state=random_state)
    X_res, y_res = smote.fit_resample(X_tmp, y_tmp)

    orig_len = len(df)
    X_syn = X_res[orig_len:]
    y_syn = y_res[orig_len:]

    if noise_sigma > 0 and len(X_syn):
        rng = np.random.default_rng(random_state)
        X_syn += rng.normal(0.0, noise_sigma, X_syn.shape)

    syn = pd.DataFrame(X_syn, columns=numeric_cols, dtype=np.float32)
    syn[label_col] = y_syn

    if "ts" in df.columns:
        ts = df["ts"].dropna()
        last = ts.iloc[-1]
        if len(ts) >= 2:
            step = ts.diff().iloc[-20:].median()
            if pd.isna(step) or step == pd.Timedelta(0):
                step = pd.Timedelta(minutes=5)
        else:
            step = pd.Timedelta(minutes=5)
        new_ts = [last + step * (i + 1) for i in range(len(syn))]
        syn.insert(0, "ts", pd.Series(new_ts, dtype=df["ts"].dtype))

    other_cols = [c for c in df.columns if c not in numeric_cols + ["ts", label_col]]
    for col in other_cols:
        base = df[col]
        if pd.api.types.is_categorical_dtype(base):
            default = base.mode().iloc[0] if not base.mode().empty else base.cat.categories[0]
        elif pd.api.types.is_numeric_dtype(base):
            default = base.mode().iloc[0] if not base.mode().empty else 0
        else:
            default = base.mode().iloc[0] if not base.mode().empty else ""
        syn[col] = pd.Series([default] * len(syn), dtype=base.dtype)

    result = pd.concat([df, syn], ignore_index=True).reindex(columns=df.columns)

    if len(result) % 2 != 0:
        pad = result.iloc[[-1]].copy()
        result = pd.concat([result, pad], ignore_index=True)

    if result.isna().any().any():
        raise ValueError("После expand_dataset остались NaN/None!")

    return result
