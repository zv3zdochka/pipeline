import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE


def expand_dataset(
    df: pd.DataFrame,
    n: float = 6.0,
    noise_sigma: float = 0.0,
    random_state: int | None = 42,
) -> pd.DataFrame:
    """
    Upsample minority classes (+1 / -1) using SMOTE and optional Gaussian noise augmentation.

    Args:
        df: Source DataFrame containing columns 'microtrend_label' and 'ts'.
        n: Multiplicative factor for increasing the number of minority class samples.
        noise_sigma: Standard deviation of Gaussian noise added to synthetic numeric features.
        random_state: Seed for reproducibility.

    Returns:
        New DataFrame with synthetic rows (no NaN/None), length padded to a multiple of 8.
    """
    label_col = 'microtrend_label'
    if label_col not in df.columns:
        raise KeyError(f"'{label_col}' not found in dataframe")

    numeric_cols = [
        c for c in df.select_dtypes(include=np.number).columns
        if c != label_col
    ]

    X = df[numeric_cols].ffill().bfill().to_numpy(dtype=np.float32)
    y = df[label_col].to_numpy()

    counts = pd.Series(y).value_counts()
    majority = counts.idxmax()
    strategy = {
        lbl: int(counts[lbl] * n)
        for lbl in counts.index if lbl != majority
    }
    if not strategy:
        result = df.copy().reset_index(drop=True)
    else:
        min_count = min(counts[lbl] for lbl in strategy)
        k_neighbors = max(1, min(5, min_count - 1))
        smote = SMOTE(
            sampling_strategy=strategy,
            k_neighbors=k_neighbors,
            random_state=random_state
        )
        X_res, y_res = smote.fit_resample(X, y)

        orig_len = len(df)
        X_syn = X_res[orig_len:]
        y_syn = y_res[orig_len:]

        if noise_sigma > 0 and len(X_syn) > 0:
            rng = np.random.default_rng(random_state)
            X_syn += rng.normal(0, noise_sigma, X_syn.shape)

        syn_num = pd.DataFrame(X_syn, columns=numeric_cols, dtype=np.float32)
        syn_num[label_col] = y_syn.astype(df[label_col].dtype)

        other_cols = [c for c in df.columns if c not in numeric_cols + [label_col, 'ts']]
        if other_cols:
            base = df[other_cols].sample(
                n=len(syn_num), replace=True, random_state=random_state
            ).reset_index(drop=True)
            syn_num = pd.concat([syn_num.reset_index(drop=True), base], axis=1)

        if 'ts' in df.columns:
            ts = pd.to_datetime(df['ts'], utc=True)
            last = ts.iloc[-1]
            if len(ts) >= 2:
                step = ts.diff().iloc[-20:].median()
                if pd.isna(step) or step == pd.Timedelta(0):
                    step = pd.Timedelta(minutes=5)
            else:
                step = pd.Timedelta(minutes=5)
            new_ts = [last + step * (i + 1) for i in range(len(syn_num))]
            syn_num.insert(0, 'ts', pd.Series(new_ts, dtype=df['ts'].dtype))

        result = pd.concat([df.reset_index(drop=True), syn_num], ignore_index=True)

    result = result.sample(frac=1, random_state=random_state).reset_index(drop=True)

    target_mod = 8
    rem = len(result) % target_mod
    if rem != 0:
        pad_count = target_mod - rem
        pad = result.iloc[-1:].copy()
        result = pd.concat(
            [result, pd.concat([pad] * pad_count, ignore_index=True)],
            ignore_index=True
        )

    if result.isna().any().any():
        raise ValueError('NaN/None values remain after expand_dataset!')

    return result
