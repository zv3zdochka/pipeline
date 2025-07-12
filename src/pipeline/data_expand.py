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
    Увеличить выборку редких классов (+1/-1) с помощью SMOTE и аугментации шума.

    Args:
        df: исходный DataFrame, содержащий столбец 'microtrend_label' и 'ts'.
        n: множитель для увеличения числа примеров редких классов.
        noise_sigma: std для гауссовского шума, добавляемого к синтетическим признакам.
        random_state: seed для воспроизводимости.

    Returns:
        Новый DataFrame с синтетическими строками, без NaN/None/0 в новых столбцах,
        и длиной, кратной 2**level (здесь 8).
    """
    label_col = 'microtrend_label'
    if label_col not in df.columns:
        raise KeyError(f"'{label_col}' not found in dataframe")

    # Чисто числовые колонки для SMOTE
    numeric_cols = [
        c for c in df.select_dtypes(include=np.number).columns
        if c != label_col
    ]
    # Подготовка данных для SMOTE
    X = df[numeric_cols].ffill().bfill().to_numpy(dtype=np.float32)
    y = df[label_col].to_numpy()

    # Стратегия для SMOTE: для каждого не-модального класса увеличить в n раз
    counts = pd.Series(y).value_counts()
    majority = counts.idxmax()
    strategy = {
        lbl: int(counts[lbl] * n)
        for lbl in counts.index if lbl != majority
    }
    if not strategy:
        result = df.copy().reset_index(drop=True)
    else:
        # Вычислить k_neighbors
        min_count = min(counts[lbl] for lbl in strategy)
        k_neighbors = max(1, min(5, min_count - 1))
        smote = SMOTE(sampling_strategy=strategy,
                      k_neighbors=k_neighbors,
                      random_state=random_state)
        X_res, y_res = smote.fit_resample(X, y)

        # Выделить только синтетические строки
        orig_len = len(df)
        X_syn = X_res[orig_len:]
        y_syn = y_res[orig_len:]

        # Добавить шум, если задано
        if noise_sigma > 0 and len(X_syn) > 0:
            rng = np.random.default_rng(random_state)
            X_syn += rng.normal(0, noise_sigma, X_syn.shape)

        # Собрать синтетический DataFrame только с числовыми
        syn_num = pd.DataFrame(X_syn, columns=numeric_cols, dtype=np.float32)
        syn_num[label_col] = y_syn.astype(df[label_col].dtype)

        # Копирование остальных столбцов из случайных строк исходника
        other_cols = [c for c in df.columns if c not in numeric_cols + [label_col, 'ts']]
        if other_cols:
            base = df[other_cols].sample(
                n=len(syn_num), replace=True, random_state=random_state
            ).reset_index(drop=True)
            syn_num = pd.concat([syn_num.reset_index(drop=True), base], axis=1)

        # Генерация новых timestamp: копируем из конца, сохраняя шаг
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

        # Итоговый DataFrame
        result = pd.concat([df.reset_index(drop=True), syn_num], ignore_index=True)

    # Перемешать, чтобы синтетика не шла блоком
    result = result.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Добить до длины, кратной 8 (2**level)
    target_mod = 8
    rem = len(result) % target_mod
    if rem != 0:
        pad_count = target_mod - rem
        pad = result.iloc[-1:].copy()
        result = pd.concat([result, pd.concat([pad] * pad_count, ignore_index=True)],
                           ignore_index=True)

    # Проверить на NaN/None
    if result.isna().any().any():
        raise ValueError('После expand_dataset остались NaN/None!')

    return result
