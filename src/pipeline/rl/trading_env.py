# src/pipeline/rl/trading_env.py
from __future__ import annotations
import numpy as np
import gym
from gym import spaces


class TradingEnv(gym.Env):
    """
    Gymnasium-style среда для PPO:
    - observation: TFT-embedding + текущая позиция
    - action_space: Discrete(4): 0=hold, 1=long, 2=short, 3=close
    - reward: PnL между шагами минус комиссия
    """
    metadata = {"render.modes": []}

    def __init__(
        self,
        embeddings: np.ndarray,
        prices: np.ndarray,
        initial_balance: float = 1.0,
        fee: float = 0.0004,
    ):
        super().__init__()
        assert len(embeddings) == len(prices), "embeddings и prices разной длины"
        self.embeddings = embeddings.astype(np.float32)
        self.prices = prices.astype(np.float32)
        self.n_steps = len(prices)

        obs_dim = self.embeddings.shape[1] + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        self.initial_balance = initial_balance
        self.fee = fee
        self.reset()

    def reset(self, *, seed: int | None = None, **kwargs):
        super().reset(seed=seed)
        self.step_idx = 0
        self.position = 0       # -1, 0, +1
        self.entry_price = 0.0
        obs = self._get_obs()
        return obs, {}          # Gymnasium API: (obs, info)

    def step(self, action: int):
        price = self.prices[self.step_idx]
        reward = 0.0

        # выполнение действия
        if action == 1:  # open long
            if self.position == 0:
                self.entry_price = price
            self.position = 1
        elif action == 2:  # open short
            if self.position == 0:
                self.entry_price = price
            self.position = -1
        elif action == 3 and self.position != 0:  # close position
            reward += self._pnl(price) - self.fee
            self.position = 0
            self.entry_price = 0.0

        # отметим переход
        self.step_idx += 1
        terminated = self.step_idx >= self.n_steps - 1

        # марк-то-маркет PnL, если остались в позиции
        if not terminated and self.position != 0:
            reward += self._pnl(self.prices[self.step_idx]) - self.fee

        obs = self._get_obs()
        info = {}

        # Gymnasium API expects (obs, reward, terminated, truncated, info)
        truncated = False
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        emb = self.embeddings[self.step_idx]
        return np.concatenate([emb, [self.position]], dtype=np.float32)

    def _pnl(self, exit_price: float) -> float:
        if self.position == 0:
            return 0.0
        direction = 1.0 if self.position == 1 else -1.0
        return direction * (exit_price / self.entry_price - 1.0)
