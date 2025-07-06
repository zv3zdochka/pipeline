# prepare_dataset_GRU.py
import pathlib
import joblib
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler  # для нормализации

# Основные «событийные» фичи
EVENT_COLS = [
    "whale_position_unrealizedPnl",
    "whale_position_size",
    "whale_position_liquidationPx",
    "whale_position_marginUsed",
    "whale_position_leverage",
    "whale_position_entryPx",
    "whale_position_positionValue",
    "whale_position_cumFunding",
    "liquidation_history_longTurnover",
    "liquidation_history_shortTurnover"
]

# Автоматически добавляем для каждой EVENT_COLS соответствующий missing-флаг
MISSING_COLS = [f"{c}_was_missing" for c in EVENT_COLS]

# Всё вместе — фичи для GRU
FEATURE_COLS = EVENT_COLS + MISSING_COLS


class GRUSequenceDataset(Dataset):
    """
    Готовит данные для GRU:
     - джойнит эмбедды CNN с EOS-фичами и их флагами пропусков
     - нормализует непрерывные признаки
     - строит последовательности фиксированной длины
     - отдаёт (x: FloatTensor[T, D], y: LongTensor)
    """

    def __init__(
            self,
            events_pkl: str | pathlib.Path,
            emb_path: str | pathlib.Path,
            seq_len: int = 96,
            scaler_path: str | pathlib.Path | None = None,  # куда (опционально) сохранить/откуда загрузить scaler
    ):
        # 1) загрузка событийных данных (impute_missing → joblib)
        events = joblib.load(events_pkl)
        events = events.set_index("ts").sort_index()

        # 1b) на всякий случай добавляем отсутствующие колонки и флаги
        for c in EVENT_COLS:
            if c not in events.columns:
                events[c] = 0.0
        for c in MISSING_COLS:
            if c not in events.columns:
                events[c] = 0

        # 2) проверяем, что теперь есть все нужные колонки (+ микротренд)
        needed = FEATURE_COLS + ["microtrend_label"]
        missing = [c for c in needed if c not in events.columns]
        if missing:
            raise KeyError(f"В events по-прежнему отсутствуют колонки: {missing}")

        df_ev = events[needed].copy()

        # 3) загрузка эмбеддингов CNN (parquet или csv)
        emb_path = pathlib.Path(emb_path)
        if emb_path.suffix == ".parquet":
            emb = pd.read_parquet(emb_path)
        else:
            emb = pd.read_csv(emb_path, index_col=0, parse_dates=True)
        emb = emb.sort_index()

        # 4) джоин по таймстемпу, inner чтобы убрать рассинхроны
        df_all = df_ev.join(emb, how="inner").dropna()

        # 5) нормализация: масштабируем все непрерывные фичи (события и эмбеддинги)
        feature_columns = EVENT_COLS + MISSING_COLS + emb.columns.tolist()
        scaler = None
        if scaler_path and pathlib.Path(scaler_path).exists():
            # если передан путь и файл есть — загружаем
            scaler = joblib.load(scaler_path)
        else:
            # иначе — создаём и обучаем новый
            scaler = StandardScaler()
            scaler.fit(df_all[feature_columns].values)

            if scaler_path:
                joblib.dump(scaler, scaler_path)

        df_all[feature_columns] = scaler.transform(df_all[feature_columns].values)

        # 6) формируем X, y
        # метка: {-1,0,1} → {0,1,2}
        self.y = (df_all["microtrend_label"] + 1).astype("int64").values

        # все фичи (события + missing-флаги + embedding-колонки) после нормализации
        self.X = (
            df_all
            .drop(columns=["microtrend_label"])
            .astype("float32")
            .values
        )
        self.seq_len = seq_len

    def __len__(self):
        # число полных последовательностей длины seq_len
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx):
        # последовательность [idx : idx+seq_len)
        x = self.X[idx: idx + self.seq_len]  # (T, D)
        y = self.y[idx + self.seq_len - 1]  # метка на последнем этапе
        return (
            torch.from_numpy(x),  # FloatTensor (T, D)
            torch.tensor(y, dtype=torch.long)  # LongTensor ()
        )


def prepare_gru_dataset(
        events_pkl: str,
        emb_path: str,
        seq_len: int = 96,
        out_path: str | None = None,
        scaler_path: str | None = None
) -> GRUSequenceDataset:
    """
    Обёртка для создания и (опционально) сохранения готового GRUSequenceDataset.
    """
    ds = GRUSequenceDataset(events_pkl, emb_path, seq_len, scaler_path)
    if out_path:
        joblib.dump(ds, out_path)
    return ds
