# src/pipeline/train_PPO.py
from __future__ import annotations
import os, pathlib, numpy as np, pandas as pd, gym, torch
from gym import spaces
from typing import Tuple, Union
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

# ----------------------------- Hyperparameters ----------------------------- #
_N_ENVS       = int(os.getenv("PPO_N_ENVS", 6))        # parallel environments
_BATCH        = int(os.getenv("PPO_BATCH", 4096))      # PPO minibatch size
_N_STEPS      = int(os.getenv("PPO_N_STEPS", 2048))    # rollout length per env
_REWARD_K     = 1_000.0                                # reward scale factor
_HOLD_PENALTY = 0.001                                  # penalty per idle step
_CLOSE_BONUS  = 0.01                                   # bonus for successful exit
_MAX_EPISODE  = 1_000                                  # max steps per episode
_EPS          = 1e-6                                   # numerical epsilon
# --------------------------------------------------------------------------- #


def _sanitize(x: np.ndarray) -> np.ndarray:
    """Replace NaN / Inf with 0 and cast to float32."""
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _embeddings(path: Union[str, pathlib.Path]) -> np.ndarray:
    df = pd.read_parquet(path)
    return _sanitize(
        np.vstack(df["embedding"].values)
        if "embedding" in df.columns
        else df.to_numpy()
    )


def _prices(path: Union[str, pathlib.Path], col: str) -> np.ndarray:
    pr = _sanitize(pd.read_csv(path, usecols=[col])[col].to_numpy())
    bad = pr <= 0
    if bad.any():  # simple forward fill fallback + epsilon
        pr[bad] = np.maximum.accumulate(pr)[bad] + 1e-3
    return pr


class TradingEnv(gym.Env):
    """
    Discrete trading environment.

    Actions:
        0: Open Long
        1: Open Short
        2: Close Long
        3: Close Short
        4: Hold
    """
    metadata = {"render.modes": []}

    def __init__(self, emb: np.ndarray, pr: np.ndarray):
        super().__init__()
        assert len(emb) == len(pr)
        self.emb, self.pr = emb, pr
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(emb.shape[1],), dtype=np.float32
        )
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.i = 0        # absolute index
        self.t = 0        # index within current episode
        self.pos = 0      # 1 long, -1 short, 0 flat
        self.entry = 0.0
        return self.emb[self.i], {}

    def step(self, action: int):
        p_now = float(max(self.pr[self.i], _EPS))
        p_next = float(max(self.pr[self.i + 1], _EPS))
        r = 0.0

        # actions
        if action == 0 and self.pos == 0:          # open long
            self.pos, self.entry = 1, p_now
        elif action == 1 and self.pos == 0:        # open short
            self.pos, self.entry = -1, p_now
        elif action == 2 and self.pos == 1:        # close long
            r += (p_now - self.entry) / self.entry + _CLOSE_BONUS
            self.pos, self.entry = 0, 0.0
        elif action == 3 and self.pos == -1:       # close short
            r += (self.entry - p_now) / self.entry + _CLOSE_BONUS
            self.pos, self.entry = 0, 0.0

        # mark-to-market
        if self.pos == 1:
            r += (p_next - p_now) / p_now
        elif self.pos == -1:
            r += (p_now - p_next) / p_now

        # idle penalty
        if self.pos == 0:
            r -= _HOLD_PENALTY

        # scale reward
        r *= _REWARD_K

        # advance
        self.i += 1
        self.t += 1
        done = self.i >= len(self.pr) - 1 or self.t >= _MAX_EPISODE
        return self.emb[self.i], float(r), done, False, {}


def _vec_env(emb: np.ndarray, pr: np.ndarray, n: int) -> SubprocVecEnv:
    def _factory():
        return TradingEnv(emb, pr)
    start = "fork" if os.name == "posix" else "spawn"
    return SubprocVecEnv([_factory] * n, start_method=start)


def train_ppo(
    emb_path: str | pathlib.Path,
    csv_path: str | pathlib.Path,
    price_col: str,
    *,
    model_out: str | pathlib.Path = "ppo_trading_best.zip",
    total_timesteps: int = 1_000_000,
) -> Tuple[float, float]:
    """
    Train a PPO agent on the trading environment and return (avg_reward, win_rate)
    evaluated on a hold-out split.

    Args:
        emb_path: Path to parquet file with embeddings (rows aligned with prices).
        csv_path: Path to CSV containing price column.
        price_col: Name of price column in CSV.
        model_out: Output path for the best model (.zip).
        total_timesteps: Total training timesteps.

    Returns:
        avg_reward: Mean episode reward on evaluation.
        win_rate: Fraction of evaluation episodes with positive reward.
    """
    emb = _embeddings(emb_path)
    pr = _prices(csv_path, price_col)
    if len(emb) != len(pr):
        m = min(len(emb), len(pr))
        print(f"[WARN] truncate → {m}")
        emb, pr = emb[:m], pr[:m]

    split = int(len(pr) * 0.8)
    train_env = _vec_env(emb[:split], pr[:split], _N_ENVS)
    eval_env = _vec_env(emb[split:], pr[split:], _N_ENVS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"[PPO] device={device}  envs={_N_ENVS}  steps={_N_STEPS}  "
        f"batch={_BATCH}  total={total_timesteps:,}"
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        batch_size=_BATCH,
        n_steps=_N_STEPS,
        gamma=0.99,
        gae_lambda=0.90,
        ent_coef=0.02,
        clip_range=0.30,
        policy_kwargs=dict(net_arch=[512, 512, 256]),
        device=device,
        verbose=0,
    )

    cb = EvalCallback(
        eval_env,
        eval_freq=10_000,
        best_model_save_path=str(pathlib.Path(model_out).parent),
        n_eval_episodes=10,
        deterministic=True,
    )

    print("[PPO] training …")
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=cb)

    best = pathlib.Path(model_out).parent / "best_model.zip"
    (best if best.exists() else pathlib.Path(model_out)).rename(model_out)
    model = PPO.load(model_out, env=eval_env, device=device)

    mean_r, _ = evaluate_policy(model, eval_env, 20, deterministic=True)
    ep_r, _ = evaluate_policy(
        model, eval_env, 20, deterministic=True, return_episode_rewards=True
    )
    win_rate = float((np.array(ep_r) > 0).mean())
    print(f"[RESULT] avg_reward={mean_r:.3f} | win_rate={win_rate:.1%}")

    return float(mean_r), win_rate
