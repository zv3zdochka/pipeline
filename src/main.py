# main.py
import pathlib
import pandas as pd
import joblib

from data_profile import (
    impute_missing,
    prepare_tft_features,
    label_microtrend,
    prepare_1dcnn_df,
    train_wavecnn,
    prepare_gru_dataset,
    train_gru,
    prepare_timesnet_dataset,
)
from data_profile.train_TimesNet import train_timesnet  # наша реализация

CACHE_DIR = pathlib.Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = pathlib.Path(__file__).parent.parent / "data" / "BTCUSDT_merge_30d.csv"
TFT_CSV = CACHE_DIR / "TFT.csv"


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Загрузка сырых данных и imputation
    # ------------------------------------------------------------------ #
    df = pd.read_csv(CSV_PATH)
    df = impute_missing(df)

    # ------------------------------------------------------------------ #
    # 2. TFT-фичи и метки микротренда
    # ------------------------------------------------------------------ #
    df_tft = prepare_tft_features(df)
    df_tft["microtrend_label"] = label_microtrend(df_tft)
    df_tft.to_csv(TFT_CSV, index=False)
    joblib.dump(df_tft, CACHE_DIR / "imputed_events.pkl")
    print("[MAIN] Prepared TFT features and labels.")

    # ------------------------------------------------------------------ #
    # 3. Wavelet + 1D-CNN
    # ------------------------------------------------------------------ #
    prepare_1dcnn_df(
        df_tft,
        wavelet="db4",
        level=3,
        window_size=24,
        raw_scaler_path=CACHE_DIR / "scaler_raw.pkl",
        wave_scaler_path=CACHE_DIR / "scaler_wave.pkl",
        dataset_path=CACHE_DIR / "wavecnn_dataset.pkl",
        class_freq_path=CACHE_DIR / "class_freqs.pt",
    )
    train_wavecnn(
        dataset_pkl=str(CACHE_DIR / "wavecnn_dataset.pkl"),
        class_freqs_pt=str(CACHE_DIR / "class_freqs.pt"),
        model_out=str(CACHE_DIR / "wavecnn_model.pt"),
        emb_out=str(CACHE_DIR / "cnn_embeddings.parquet"),
        window=24,
        epochs=1,
        batch=256,
    )

    # ------------------------------------------------------------------ #
    # 4. GRU
    # ------------------------------------------------------------------ #
    prepare_gru_dataset(
        events_pkl=str(CACHE_DIR / "imputed_events.pkl"),
        emb_path=str(CACHE_DIR / "cnn_embeddings.parquet"),
        seq_len=96,
        out_path=str(CACHE_DIR / "gru_dataset.pkl"),
    )
    train_gru(
        dataset_pkl=str(CACHE_DIR / "gru_dataset.pkl"),
        class_freqs_pt=str(CACHE_DIR / "class_freqs.pt"),
        events_pkl=str(CACHE_DIR / "imputed_events.pkl"),
        emb_path=str(CACHE_DIR / "cnn_embeddings.parquet"),
        model_out=str(CACHE_DIR / "gru_model.pt"),
        emb_out=str(CACHE_DIR / "gru_embeddings.parquet"),
        seq_len=96,
        epochs=1,
        batch_size=128,
        lr=3e-4,
    )

    # ------------------------------------------------------------------ #
    # 5. TimesNet: подготовка датасета
    # ------------------------------------------------------------------ #
    tn_ds, _ = prepare_timesnet_dataset(
        df_tft,
        seq_len=288,
        horizon=288,
        scaler_path=CACHE_DIR / "timesnet_scaler.pkl",
        dataset_path=CACHE_DIR / "timesnet_dataset.pt",
    )
    print(f"[MAIN] TimesNet dataset prepared: {len(tn_ds)} windows.")

    # ------------------------------------------------------------------ #
    # 6. TimesNet: обучение, сохранение модели, эмбеддингов и прогнозов
    # ------------------------------------------------------------------ #
    train_timesnet(
        dataset_pt=str(CACHE_DIR / "timesnet_dataset.pt"),
        events_pkl=str(CACHE_DIR / "imputed_events.pkl"),
        model_out=str(CACHE_DIR / "timesnet_model.pt"),
        embed_out=str(CACHE_DIR / "timesnet_embeddings.parquet"),
        forecast_out=str(CACHE_DIR / "timesnet_forecast.parquet"),
        seq_len=288,
        epochs=1,
        batch_size=128,
        lr=3e-4,
    )

    # ------------------------------------------------------------------ #
    # 7. Интеграция прогнозов и эмбеддингов TimesNet
    # ------------------------------------------------------------------ #
    tn_pred = pd.read_parquet(CACHE_DIR / "timesnet_forecast.parquet")
    tn_emb = pd.read_parquet(CACHE_DIR / "timesnet_embeddings.parquet")
    # Объединяем по индексу ts
    df_tft = (
        df_tft
        .set_index("ts")
        .join(tn_pred)
        .join(tn_emb)
        .reset_index()
    )
    print("[MAIN] Added timesnet_pred and timesnet_emb to df_tft.")


if __name__ == "__main__":
    main()
