# main.py
import pathlib
import pandas as pd
import joblib

from pipeline import (
    impute_missing,
    prepare_features,
    label_microtrend,
    prepare_1dcnn_df,
    train_wavecnn,
    prepare_gru_dataset,
    train_gru,
    prepare_timesnet_dataset,
    train_timesnet,
    prepare_tft_dataset,
    train_tft,
    print_dataset_overview,
    save_dataset_and_distribution,
    train_ppo
)

CACHE_DIR = pathlib.Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = pathlib.Path(__file__).parent.parent / "data" / "XRPUSDT_merge_180d.csv"


def main() -> None:
    print("[MAIN] Step 1: Loading raw data and imputing missing values...")
    df = pd.read_csv(CSV_PATH, parse_dates=["ts"], low_memory=False)
    print(f"[MAIN] Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

    df = impute_missing(df)
    print(f"[PREPROCESS] After imputation: {df.isna().sum().sum()} total missing values remain")

    df.to_csv(CACHE_DIR / "data.csv", index=False)
    print(f"[MAIN] Cached imputed data to {CACHE_DIR / 'data.csv'}")

    print_dataset_overview(df)

    # Prepare features and microtrend labels
    print("[MAIN] Step 2: Preparing features and microtrend labels...")
    dataset = prepare_features(df)
    print(f"[FEATURES] Features prepared: {dataset.shape[1]} features for {dataset.shape[0]} rows")

    dataset['microtrend_label'] = label_microtrend(dataset, window=36, threshold=0.03)
    print(f"[MICROTREND] Assigned microtrend labels: {dataset['microtrend_label'].nunique()} unique labels")

    # Save outputs
    save_dataset_and_distribution(dataset)

    # Wavelet + 1D-CNN
    print(f"[WAVECNN] Starting CNN dataset preparation")
    df_train, df_test, scaler_raw, scaler_wave = prepare_1dcnn_df(
        dataset,
        wavelet="db4",
        level=3,
        window_size=24,
        train_frac=0.8,
        raw_scaler_path=CACHE_DIR / "scaler_raw.pkl",
        wave_scaler_path=CACHE_DIR / "scaler_wave.pkl",
        dataset_train_path=CACHE_DIR / "wavecnn_dataset_train.pkl",
        dataset_test_path=CACHE_DIR / "wavecnn_dataset_test.pkl",
        class_freq_path=CACHE_DIR / "class_freqs.pt",
    )
    print(f"[WAVECNN] Prepared datasets: "
          f"{df_train.shape[0]} train samples, {df_test.shape[0]} test samples")
    print(f"[WAVECNN] Saved raw scaler → {CACHE_DIR / 'scaler_raw.pkl'}, "
          f"wavelet scaler → {CACHE_DIR / 'scaler_wave.pkl'}")
    print(f"[WAVECNN] Saved datasets → "
          f"{CACHE_DIR / 'wavecnn_dataset_train.pkl'} & {CACHE_DIR / 'wavecnn_dataset_test.pkl'}")
    print(f"[WAVECNN] Class frequencies written to → {CACHE_DIR / 'class_freqs.pt'}")

    train_wavecnn(
        train_pkl=str(CACHE_DIR / "wavecnn_dataset_train.pkl"),
        test_pkl=str(CACHE_DIR / "wavecnn_dataset_test.pkl"),
        class_freqs_pt=str(CACHE_DIR / "class_freqs.pt"),
        model_out=str(CACHE_DIR / "wavecnn_model.pt"),
        emb_out=str(CACHE_DIR / "cnn_embeddings.parquet"),
        window=24,
        epochs=1,
        batch=256,
        lr=3e-4,
    )

    # train_wavecnn(
    #     dataset_pkl=str(CACHE_DIR / "wavecnn_dataset.pkl"),
    #     class_freqs_pt=str(CACHE_DIR / "class_freqs.pt"),
    #     model_out=str(CACHE_DIR / "wavecnn_model.pt"),
    #     emb_out=str(CACHE_DIR / "cnn_embeddings.parquet"),
    #     window=24,
    #     epochs=1,
    #     batch=256,
    # )

    print("[MAIN] All steps completed successfully.")

    exit()

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
        dataset,
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
    dataset = (
        dataset
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
    print(f"[MAIN] PPO direction-accuracy: {acc * 100:.2f}%")
    print(f"[MAIN] PPO Kelly fraction  : {kelly:.3f}")


if __name__ == "__main__":
    main()
