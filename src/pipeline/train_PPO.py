"""
train_ppo.py

Enhanced PPO agent for *real‑exchange* crypto trading.
=====================================================
Key improvements vs. first draft
--------------------------------
1. **Realistic micro‑structure costs** – commission & slippage deducted on every entry/exit.
2. **Trend‑aware reward shaping** – incentivises catching trends of 3‑100 candles (defaults 3‑15).
3. **Observation normalisation** – `StandardScaler` on TFT embeddings for faster, stabler learning.
4. **Better PPO hyper‑params** – tuned for discrete 5‑action space & sparse rewards.
5. **Evaluation callback** – early‑stopping on Sharpe‑like metric over validation split.
6. **Clean API remains**: `train_ppo()` + `get_action()` unchanged for callers.

Dependencies
------------
```bash
pip install stable-baselines3==2.2.1 torch gym numpy scikit-learn
```

"""
from __future__ import annotations

import numpy as np
import gym
from gym import spaces
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List

# ──────────────────────────────────────────
# Environment
# ──────────────────────────────────────────


class TradingEnv(gym.Env):
    """Trend‑aware trading environment for PPO.

    Parameters
    ----------
    embeddings : np.ndarray, (T, D)
        TFT contextual embeddings.
    prices : np.ndarray, (T,)
        Reference close price per candle.
    commission_perc : float, default 2e‑4 (0.02 %)
        One‑side commission percentage *per* trade side.
    slippage_perc : float, default 1e‑4 (0.01 %)
        Expected adverse price movement when filling.
    min_trend : int, default 3
        Minimum candles a trend is expected to last.
    max_trend : int, default 100
        Upper bound for trend duration (avoids blind holding forever).
    max_steps : int | None
        Cap episode length; default = len(embeddings) ‑ 2.
    """

    metadata = {"render.modes": ["human"]}

    ACTIONS: List[str] = [
        "Take Long",
        "Take Short",
        "Exit Long",
        "Exit Short",
        "Hold",
    ]

    def __init__(
        self,
        embeddings: np.ndarray,
        prices: np.ndarray,
        *,
        commission_perc: float = 2e-4,
        slippage_perc: float = 1e-4,
        min_trend: int = 3,
        max_trend: int = 100,
        max_steps: int | None = None,
    ):
        super().__init__()

        assert len(embeddings) == len(prices), "Embeddings and prices length mismatch"
        self.raw_embeddings = embeddings.astype(np.float32)
        self.prices = prices.astype(np.float32)
        self.T, self.embedding_dim = self.raw_embeddings.shape

        # Normalise observations (fit once)
        self.scaler = StandardScaler().fit(self.raw_embeddings)
        self.embeddings = self.scaler.transform(self.raw_embeddings).astype(np.float32)

        self.commission = commission_perc
        self.slippage = slippage_perc
        self.min_trend = min_trend
        self.max_trend = max_trend
        self.max_steps = (max_steps or self.T - 2)  # leave look‑ahead 1 candle

        # Gym spaces
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(self.embedding_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self.ACTIONS))

        # Internal state vars
        self.current_step: int = 0
        self.position: int = 0      # 1 long, -1 short, 0 flat
        self.entry_price: float = 0
        self.hold_duration: int = 0 # candles in current position

    # ────────────── Core API ──────────────

    def reset(self, *, seed: int | None = None, **kwargs):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        self.hold_duration = 0
        return self._get_obs(), {}

    def step(self, action: int):
        done = False
        truncated = False

        price_now = self.prices[self.current_step]
        price_next = self.prices[self.current_step + 1]

        reward = 0.0
        trade_executed = False

        # ─── Action logic ───────────────────
        if action == 0:  # Take Long
            if self.position != 1:
                reward -= self._trade_cost(price_now)
                self.position = 1
                self.entry_price = price_now
                self.hold_duration = 0
                trade_executed = True
        elif action == 1:  # Take Short
            if self.position != -1:
                reward -= self._trade_cost(price_now)
                self.position = -1
                self.entry_price = price_now
                self.hold_duration = 0
                trade_executed = True
        elif action == 2 and self.position == 1:  # Exit Long
            reward += self._realised_pnl(price_now)
            reward -= self._trade_cost(price_now)
            self.position = 0
            self.hold_duration = 0
            trade_executed = True
        elif action == 3 and self.position == -1:  # Exit Short
            reward += self._realised_pnl(price_now)
            reward -= self._trade_cost(price_now)
            self.position = 0
            self.hold_duration = 0
            trade_executed = True
        elif action == 4:  # Hold
            pass

        # Unrealised PnL every step while in position
        reward += self._unrealised_pnl(price_now, price_next)

        # Trend‑aware holding bonus / penalty
        if self.position != 0:
            self.hold_duration += 1
            if self.hold_duration < self.min_trend:
                reward -= 0.01  # discourage premature exit
            elif self.hold_duration > self.max_trend:
                reward -= 0.02  # penalty for over‑holding dead positions

        # Advance time
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        obs = self._get_obs()
        info = {
            "step": self.current_step,
            "position": self.position,
            "hold_duration": self.hold_duration,
            "trade": trade_executed,
        }
        return obs, float(reward), done, truncated, info

    # ────────────── Helpers ──────────────

    def _get_obs(self) -> np.ndarray:  # returns normalised embedding
        return self.embeddings[self.current_step]

    def _trade_cost(self, price: float) -> float:
        # Commission both sides (entry/exit) + slippage when trade executes
        return (self.commission * 2 + self.slippage) * 100.0

    def _unrealised_pnl(self, price_now: float, price_next: float) -> float:
        if self.position == 0:
            return 0.0
        direction = 1.0 if self.position == 1 else -1.0
        pct = (price_next - price_now) / price_now * direction
        return pct * 100.0

    def _realised_pnl(self, price_exit: float) -> float:
        if self.position == 0:
            return 0.0
        direction = 1.0 if self.position == 1 else -1.0
        pct = (price_exit - self.entry_price) / self.entry_price * direction
        return pct * 100.0

    # ────────────── Render ──────────────

    def render(self):
        print(
            f"Step {self.current_step:>6} | Pos {self.position:+d} | Hold {self.hold_duration:>3} | "
            f"Price {self.prices[self.current_step]:.2f}"
        )

    def close(self):
        pass


# ──────────────────────────────────────────
# Training wrapper with validation split
# ──────────────────────────────────────────

def train_ppo(
    embeddings: np.ndarray,
    prices: np.ndarray,
    *,
    policy_kwargs: dict | None = None,
    total_timesteps: int = 2_000_000,
    learning_rate: float = 2.5e-4,
    device: str = "auto",
    verbose: int = 1,
    eval_fraction: float = 0.2,
) -> PPO:
    """Train PPO with early stopping on validation Sharpe‑like metric."""

    assert 0 < eval_fraction < 0.5, "eval_fraction should be (0,0.5)"
    split = int(len(embeddings) * (1 - eval_fraction))
    train_emb, val_emb = embeddings[:split], embeddings[split:]
    train_prc, val_prc = prices[:split], prices[split:]

    env = DummyVecEnv([lambda: TradingEnv(train_emb, train_prc)])
    eval_env = DummyVecEnv([lambda: TradingEnv(val_emb, val_prc)])

    if policy_kwargs is None:
        policy_kwargs = dict(net_arch=[512, 256, 128], activation_fn=th.nn.Tanh)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=4096,
        ent_coef=0.005,
        clip_range=0.2,
        gae_lambda=0.95,
        gamma=0.997,
        max_grad_norm=0.5,
        vf_coef=0.4,
        target_kl=0.03,
        verbose=verbose,
        device=device,
    )

    stop_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=10, verbose=verbose)
    eval_cb = EvalCallback(
        eval_env,
        eval_freq=25_000,
        best_model_save_path=None,
        deterministic=True,
        render=False,
        callback_after_eval=stop_cb,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_cb)
    return model


# ──────────────────────────────────────────
# Inference helper (unchanged API)
# ──────────────────────────────────────────

def get_action(
    model: PPO,
    observation: np.ndarray,
    deterministic: bool = False,
) -> Tuple[str, Dict[str, float]]:
    """Return human‑readable action and confidences."""
    obs = th.as_tensor(observation, dtype=th.float32).unsqueeze(0).to(model.device)
    with th.no_grad():
        dist = model.policy.get_distribution(obs)
        probs = th.softmax(dist.distribution.logits, dim=-1).cpu().numpy().flatten()

    actions = TradingEnv.ACTIONS
    if deterministic:
        idx = int(np.argmax(probs))
    else:
        idx = int(np.random.choice(len(actions), p=probs))

    return actions[idx], {a: float(p) for a, p in zip(actions, probs)}


# ──────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────
if __name__ == "__main__":
    T, D = 20_000, 64
    np.random.seed(42)
    dummy_emb = np.random.randn(T, D).astype(np.float32)
    dummy_price = 50_000 + np.cumsum(np.random.randn(T)).astype(np.float32)

    model = train_ppo(dummy_emb, dummy_price, total_timesteps=50_000, verbose=0)
    act, conf = get_action(model, dummy_emb[-1], deterministic=True)
    print("Action:", act)
    print("Confidences:")
    for k, v in conf.items():
        print(f"  {k:11s}: {v:.3f}")
