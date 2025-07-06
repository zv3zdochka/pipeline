# main.py
import pathlib
import pandas as pd
import joblib
from data_profile import (
    analyze_df,
    impute_missing,
    prepare_tft_features,
    label_microtrend,
    train_wavecnn,
    prepare_gru_dataset,
    prepare_1dcnn_df,
    train_gru,
    prepare_timesnet_dataset,
    train_timesnet,
)

CACHE_DIR: pathlib.Path = pathlib.Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH: pathlib.Path = pathlib.Path(__file__).parent.parent / "data" / "BTCUSDT_merge_30d.csv"
TFT_CSV: pathlib.Path = CACHE_DIR / "TFT.csv"


def main() -> None:
    df: pd.DataFrame = pd.read_csv(CSV_PATH)
    print("Before imputation:")
    df_clean: pd.DataFrame = impute_missing(df)
    print("\nAfter imputation:")
    df_tft: pd.DataFrame = prepare_tft_features(df_clean)
    print("\nAfter TFT features:")
    df_tft["microtrend_label"] = label_microtrend(df_tft)
    print("\nAfter labeling microtrends:")
    df_tft.to_csv(TFT_CSV, index=False)
    print(f"\nSaved TFT features to {TFT_CSV}")
    joblib.dump(df_tft, CACHE_DIR / "imputed_events.pkl")
    df_ready, scaler_raw, scaler_wave = prepare_1dcnn_df(
        df_tft,
        wavelet="db4",
        level=3,
        window_size=24,
        raw_scaler_path=CACHE_DIR / "scaler_raw.pkl",
        wave_scaler_path=CACHE_DIR / "scaler_wave.pkl",
        dataset_path=CACHE_DIR / "wavecnn_dataset.pkl",
        class_freq_path=CACHE_DIR / "class_freqs.pt",
    )
    print(f"[INFO] Prepared 1D-CNN dataset: {df_ready.shape}")
    print("[INFO] Saved scalers and serialized dataset.")
    train_wavecnn(
        dataset_pkl=str(CACHE_DIR / "wavecnn_dataset.pkl"),
        class_freqs_pt=str(CACHE_DIR / "class_freqs.pt"),
        model_out=str(CACHE_DIR / "wavecnn_model.pt"),
        emb_out=str(CACHE_DIR / "cnn_embeddings.parquet"),
        window=24,
        epochs=3,
        batch=256,
    )
    gru_ds = prepare_gru_dataset(
        events_pkl=str(CACHE_DIR / "imputed_events.pkl"),
        emb_path=str(CACHE_DIR / "cnn_embeddings.parquet"),
        seq_len=96,
        out_path=str(CACHE_DIR / "gru_dataset.pkl"),
    )
    print(f"[INFO] GRU dataset size: {len(gru_ds)} sequences of length {gru_ds.seq_len}")
    train_gru(
        dataset_pkl=str(CACHE_DIR / "gru_dataset.pkl"),
        class_freqs_pt=str(CACHE_DIR / "class_freqs.pt"),
        events_pkl=str(CACHE_DIR / "imputed_events.pkl"),
        emb_path=str(CACHE_DIR / "cnn_embeddings.parquet"),
        model_out=str(CACHE_DIR / "gru_model.pt"),
        emb_out=str(CACHE_DIR / "gru_embeddings.parquet"),
        seq_len=96,
        epochs=3,
        batch_size=128,
        lr=3e-4,
    )
    tn_ds, tn_scaler = prepare_timesnet_dataset(
        df_tft,
        seq_len=288,
        horizon=288,
        scaler_path=CACHE_DIR / "timesnet_scaler.pkl",
        dataset_path=CACHE_DIR / "timesnet_dataset.pt",
    )
    print(f"[INFO] TimesNet dataset: {len(tn_ds)} windows of 288×{tn_ds[0][0].shape[1]}")
    train_timesnet(
        dataset_path=str(CACHE_DIR / "timesnet_dataset.pt"),
        model_out=str(CACHE_DIR / "timesnet_model.pt"),
        scaler_path=str(CACHE_DIR / "timesnet_scaler.pkl"),
        seq_len=288,
        horizon=288,
        batch=128,
        epochs=3,
        lr=1e-3,
    )


if __name__ == "__main__":
    main()
