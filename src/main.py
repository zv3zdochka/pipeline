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
    train_gru
)

CSV_PATH = pathlib.Path(__file__).parent.parent / "data" / "BTCUSDT_merge_30d.csv"
TFT_CSV = pathlib.Path(__file__).parent / "TFT.csv"


def main():
    # 1) Чтение и профилирование
    df = pd.read_csv(CSV_PATH)
    print("Before imputation:")
    # analyze_df(df)

    # 2) Заполнение пропусков
    df_clean = impute_missing(df)
    print("\nAfter imputation:")
    # analyze_df(df_clean)

    # 3) TFT-признаки
    df_tft = prepare_tft_features(df_clean)
    print("\nAfter TFT features:")
    # analyze_df(df_tft)

    # 4) Разметка целевой переменной
    df_tft["microtrend_label"] = label_microtrend(df_tft)
    print("\nAfter labeling microtrends:")
    # analyze_df(df_tft[["ts", "ohlcv_5m_close", "microtrend_label"]])

    # 5) Сохраняем TFT.csv
    df_tft.to_csv(TFT_CSV, index=False)
    print(f"\nSaved TFT features to {TFT_CSV}")

    joblib.dump(
        df_tft,
        "imputed_events.pkl"
    )
    # 6) Подготовка датасета для 1D-CNN / WaveNet
    df_ready, scaler_raw, scaler_wave = prepare_1dcnn_df(
        df_tft,
        wavelet="db4",
        level=3,
        window_size=24,
        raw_scaler_path="scaler_raw.pkl",
        wave_scaler_path="scaler_wave.pkl",
        dataset_path="wavecnn_dataset.pkl",
        class_freq_path="class_freqs.pt",
    )
    print(f"[INFO] Prepared 1D-CNN dataset: {df_ready.shape}")
    print("[INFO] Saved scalers and serialized dataset.")

    # 7) Обучаем WaveNet+1D-CNN и сохраняем эмбеддинги
    train_wavecnn(
        dataset_pkl="wavecnn_dataset.pkl",
        class_freqs_pt="class_freqs.pt",
        model_out="wavecnn_model.pt",
        emb_out="cnn_embeddings.parquet",
        window=24,
        epochs=1,
        batch=256,
    )

    gru_ds = prepare_gru_dataset(
        events_pkl="imputed_events.pkl",
        emb_path="cnn_embeddings.parquet",
        seq_len=96,
        out_path="gru_dataset.pkl"
    )
    print(f"[INFO] GRU dataset size: {len(gru_ds)} sequences of length {gru_ds.seq_len}")

    # 9) Обучаем GRU
    train_gru(
        dataset_pkl="gru_dataset.pkl",
        class_freqs_pt="class_freqs.pt",
        events_pkl="imputed_events.pkl",
        emb_path="cnn_embeddings.parquet",
        model_out="gru_model.pt",
        emb_out="gru_embeddings.parquet",
        seq_len=96,
        epochs=1,
        batch_size=128,
        lr=3e-4
    )

    from data_profile import prepare_timesnet_dataset, train_timesnet

    # 10) TimesNet dataset
    tn_ds, tn_scaler = prepare_timesnet_dataset(
        df_tft,
        seq_len=288,  # 24h окно
        horizon=288,  # 24h прогноз
        scaler_path="timesnet_scaler.pkl",
        dataset_path="timesnet_dataset.pt"
    )
    print(f"[INFO] TimesNet dataset: {len(tn_ds)} windows of 288×{tn_ds[0][0].shape[1]}")

    # 11) Обучаем TimesNet (в одном звонке; GPU найдётся — ок)
    train_timesnet(
        dataset_path="timesnet_dataset.pt",
        model_out="timesnet_model.pt",
        scaler_path="timesnet_scaler.pkl",
        seq_len=288,
        horizon=288,
        batch=128,
        epochs=20,
        lr=1e-3
    )


if __name__ == "__main__":
    main()
