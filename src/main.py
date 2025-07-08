# main.py
import pathlib
import pandas as pd
import joblib

from pipeline import (
    impute_missing,
    prepare_tft_features,
    label_microtrend,
    prepare_1dcnn_df,
    train_wavecnn,
    prepare_gru_dataset,
    train_gru,
    prepare_timesnet_dataset,
    train_timesnet,
    prepare_tft_dataset,
    train_tft,
)
from pipeline.train_PPO import train_ppo  # ← добавлено

CACHE_DIR = pathlib.Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = pathlib.Path(__file__).parent.parent / "data" / "BTCUSDT_merge_30d.csv"

def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Загрузка сырых данных и imputation
    # ------------------------------------------------------------------ #
    df = pd.read_csv(CSV_PATH, parse_dates=["ts"])
    df = impute_missing(df)

    # ------------------------------------------------------------------ #
    # 2. TFT-фичи и метки микротренда
    # ------------------------------------------------------------------ #
    df_tft = prepare_tft_features(df)
    df_tft["microtrend_label"] = label_microtrend(df_tft)
    df_tft.to_csv(CACHE_DIR / "TFT.csv", index=False)
    joblib.dump(df_tft, CACHE_DIR / "imputed_events.pkl")
    print("[MAIN] Prepared features and labels.")

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
    # 5. TimesNet: подготовка и обучение
    # ------------------------------------------------------------------ #
    _, _ = prepare_timesnet_dataset(
        df_tft,
        seq_len=288,
        horizon=288,
        scaler_path=CACHE_DIR / "timesnet_scaler.pkl",
        dataset_path=CACHE_DIR / "timesnet_dataset.pt",
    )
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
    # 6. Собираем TFT-датасет и обучаем TFT
    # ------------------------------------------------------------------ #
    # интеграция TimesNet
    tn_pred = pd.read_parquet(CACHE_DIR / "timesnet_forecast.parquet")
    tn_emb = pd.read_parquet(CACHE_DIR / "timesnet_embeddings.parquet")
    df_tft = (
        df_tft
        .set_index("ts")
        .join(tn_pred)
        .join(tn_emb)
        .reset_index()
    )

    ds, groups = prepare_tft_dataset(
        df_events=CACHE_DIR / "imputed_events.pkl",
        cnn_emb_path=CACHE_DIR / "cnn_embeddings.parquet",
        gru_emb_path=CACHE_DIR / "gru_embeddings.parquet",
        timesnet_emb_path=CACHE_DIR / "timesnet_embeddings.parquet",
        timesnet_pred_path=CACHE_DIR / "timesnet_forecast.parquet",
        seq_len=96,
        dataset_path=CACHE_DIR / "tft_dataset.pt",
        scaler_path=CACHE_DIR / "tft_scaler.pkl",
    )
    joblib.dump(groups, CACHE_DIR / "tft_feature_groups.pkl")

    train_tft(
        dataset_pt=CACHE_DIR / "tft_dataset.pt",
        feature_groups_pkl=CACHE_DIR / "tft_feature_groups.pkl",
        class_freqs_pt=CACHE_DIR / "class_freqs.pt",
        model_out=CACHE_DIR / "tft_model.pt",
        emb_out=CACHE_DIR / "tft_embeddings.parquet",
        epochs=5,
        batch_size=128,
        lr=3e-4,
    )

    # ------------------------------------------------------------------ #
    # 7. PPO + Kelly Criterion
    # ------------------------------------------------------------------ #
    acc, kelly = train_ppo(
        emb_path=CACHE_DIR / "tft_embeddings.parquet",
        csv_path=CSV_PATH,
        price_col="ohlcv_5m_close",
        model_out=CACHE_DIR / "ppo_trading.zip",
        total_timesteps=20000,
    )
    print(f"[MAIN] PPO direction-accuracy: {acc*100:.2f}%")
    print(f"[MAIN] PPO Kelly fraction  : {kelly:.3f}")

if __name__ == "__main__":
    main()
