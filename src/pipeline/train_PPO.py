# src/pipeline/train_PPO.py
"""
PPO-обучение на TFT-эмбеддингах.

▪ inner-join по ts, если есть
▪ иначе выравнивание рядов «по порядку»
"""

from __future__ import annotations
import argparse
import pathlib

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .rl.trading_env import TradingEnv


def _read_embeddings(emb_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_parquet(emb_path)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={df.index.name or "index": "ts"})
    return df


def _read_price_csv(csv_path: pathlib.Path, price_col: str) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        usecols=["ts", price_col],
        parse_dates=["ts"],
    )
    # локализуем к UTC, если нет tz
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    return df


def _align(emb: pd.DataFrame, px: pd.DataFrame, price_col: str) -> tuple[np.ndarray, np.ndarray]:
    if "ts" in emb.columns:
        merged = emb.merge(px, on="ts", how="inner", sort=True).reset_index(drop=True)
        if merged.empty:
            raise ValueError("Нет общих меток времени для объединения")
        X = merged.drop(columns=["ts", price_col]).values.astype(np.float32)
        y = merged[price_col].values.astype(np.float32)
        return X, y

    # fallback: по порядку
    X = emb.values.astype(np.float32)
    y = px[price_col].values.astype(np.float32)
    if len(y) > len(X):
        y = y[len(y) - len(X):]
    elif len(X) > len(y):
        X = X[len(X) - len(y):]
    if len(X) != len(y):
        raise ValueError("Не удалось выровнять длины")
    return X, y


def _make_vec_env(X: np.ndarray, y: np.ndarray):
    return DummyVecEnv([lambda: TradingEnv(X, y)])


def train_ppo(
    emb_path: str | pathlib.Path,
    csv_path: str | pathlib.Path,
    price_col: str,
    model_out: str | pathlib.Path = "ppo_trading.zip",
    total_timesteps: int = 200_000,
):
    emb_path = pathlib.Path(emb_path)
    csv_path = pathlib.Path(csv_path)

    print(f"[train_PPO] Загрузка эмбеддингов из {emb_path}")
    df_emb = _read_embeddings(emb_path)

    print(f"[train_PPO] Загрузка цен из {csv_path}")
    df_px  = _read_price_csv(csv_path, price_col)

    X, y = _align(df_emb, df_px, price_col)
    print(f"[train_PPO] Выравнено: {len(y)} шагов, dim_emb={X.shape[1]}")

    env = _make_vec_env(X, y)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=512,
        batch_size=512,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(model_out)
    print(f"[train_PPO] ✅ Сохранено в {model_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--emb",       required=True, help="Parquet с TFT-эмбеддингами")
    p.add_argument("--csv",       required=True, help="CSV с колонками ts и ценой")
    p.add_argument("--price_col", default="ohlcv_5m_close")
    p.add_argument("--out",       default="ppo_trading.zip")
    p.add_argument("--steps", type=int, default=200_000)
    args = p.parse_args()

    train_ppo(
        emb_path=args.emb,
        csv_path=args.csv,
        price_col=args.price_col,
        model_out=args.out,
        total_timesteps=args.steps,
    )
