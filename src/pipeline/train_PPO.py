# src/pipeline/train_PPO.py
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from .rl.trading_env import TradingEnv
from .kelly import compute_kelly_fraction


def _read_emb(emb_path: str | pathlib.Path) -> pd.DataFrame:
    df = pd.read_parquet(emb_path)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index(names="ts")
    return df

def _read_price_csv(csv_path: str | pathlib.Path, price_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["ts"])
    if price_col not in df.columns:
        raise KeyError(f"[train_PPO] В CSV нет колонки '{price_col}'")
    return df[["ts", price_col]].copy()

def _align(df_emb: pd.DataFrame, df_px: pd.DataFrame, price_col: str) -> tuple[np.ndarray, np.ndarray]:
    if "ts" in df_emb.columns:
        merged = df_emb.merge(df_px, on="ts", how="inner", sort=True).reset_index(drop=True)
        emb = merged.drop(columns=["ts", price_col]).values.astype(np.float32)
        px  = merged[price_col].values.astype(np.float32)
        return emb, px
    emb = df_emb.values.astype(np.float32)
    px  = df_px[price_col].values.astype(np.float32)
    if len(px) > len(emb):
        px = px[-len(emb):]
    else:
        emb = emb[-len(px):]
    if len(emb) != len(px):
        raise ValueError("[train_PPO] Длины эмб и цен не совпадают")
    return emb, px

def _make_env(emb: np.ndarray, price: np.ndarray):
    return DummyVecEnv([lambda: TradingEnv(emb, price)])

def train_ppo(
    emb_path: str | pathlib.Path,
    csv_path: str | pathlib.Path,
    price_col: str,
    model_out: str | pathlib.Path = "ppo_trading.zip",
    total_timesteps: int = 500_000,
) -> tuple[float, float]:
    print(f"[train_PPO] Загрузка эмбеддингов из {emb_path}")
    df_emb = _read_emb(emb_path)

    print(f"[train_PPO] Загрузка цен из {csv_path}")
    df_px  = _read_price_csv(csv_path, price_col)

    emb, px = _align(df_emb, df_px, price_col)
    print(f"[train_PPO] Выравнено: {len(px)} точек, dim_emb={emb.shape[1]}")

    # --- тренировка ---
    env = _make_env(emb, px)
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
        ent_coef=0.01,         # повысим энтропию для исследования
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(model_out)
    print(f"[train_PPO] ✅ Модель сохранена в {model_out}")

    # --- оценка и сбор P/L ---
    eval_env = _make_env(emb, px)
    obs = eval_env.reset()
    preds, rets = [], []
    for t in range(len(px)-1):
        action, _ = model.predict(obs, deterministic=True)
        a = int(action[0])
        preds.append(a)
        next_obs, _, done, _ = eval_env.step(action)
        ret = ((px[t+1] - px[t]) / px[t]) * (1 if a==1 else -1 if a==2 else 0)
        rets.append(ret)
        obs = next_obs
        if done[0]:
            break

    act_sign = np.where(np.array(preds)==1, 1, np.where(np.array(preds)==2, -1, 0))
    moves = np.sign(px[1:1+len(act_sign)] - px[:len(act_sign)])
    accuracy = float((act_sign==moves).mean())

    kelly = compute_kelly_fraction(np.array(rets, dtype=float))

    print(f"[train_PPO] Directional accuracy = {accuracy*100:.2f}%")
    print(f"[train_PPO] Kelly fraction     = {kelly:.3f}")

    return accuracy, kelly


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Train PPO on TFT embeddings")
    ap.add_argument("--emb", required=True, help="parquet с TFT-эмбеддингами")
    ap.add_argument("--csv", required=True, help="CSV с ts + ценой")
    ap.add_argument("--price_col", default="ohlcv_5m_close")
    ap.add_argument("--out", default="ppo_trading.zip")
    ap.add_argument("--steps", type=int, default=500_000)
    args = ap.parse_args()

    acc, kelly = train_ppo(
        emb_path=args.emb,
        csv_path=args.csv,
        price_col=args.price_col,
        model_out=args.out,
        total_timesteps=args.steps,
    )
    print(f"Accuracy: {acc*100:.2f}%, Kelly: {kelly:.3f}")
