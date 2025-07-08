# src/pipeline/rl/trading_env.py

from __future__ import annotations
import gym
import numpy as np
from gym import spaces

from ..kelly import kelly_fraction


class TradingEnv(gym.Env):
    """
    Мини-среда для PPO:
      • obs      = TFT-эмбеддинг (np.ndarray dim_emb)
      • price    = close-price (float32)
      • action   = {0=flat,1=long,2=short}
      • reward   = ΔPNL (% от equity) с учётом Kelly-фракции
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        embeddings: np.ndarray,  # (T, D)
        prices: np.ndarray,      # (T,)
        *,
        odds_estimate: float = 2.0,
        kelly_coeff: float = 0.5,
        p_smooth: float = 0.95,
    ):
        super().__init__()
        assert len(embeddings) == len(prices)
        self.emb = embeddings.astype(np.float32)
        self.price = prices.astype(np.float32)
        self.T, self.D = self.emb.shape

        # action / obs spaces
        self.action_space = spaces.Discrete(3)  # 0=flat,1=long,2=short
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.D,), dtype=np.float32
        )

        # Kelly params
        self.odds = odds_estimate
        self.kelly_coeff = kelly_coeff
        self.p_smooth = p_smooth
        self._p_win = 0.5

        # internal
        self._idx = 0
        self._pos = 0           # -1,0,+1
        self._entry_price = 0.0
        self._kelly_f = 0.0

    def _update_p_win(self, reward: float):
        success = 1.0 if reward > 0 else 0.0
        self._p_win = self.p_smooth * self._p_win + (1 - self.p_smooth) * success

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self._idx = 0
        self._pos = 0
        self._entry_price = 0.0
        self._kelly_f = 0.0
        self._p_win = 0.5
        return self.emb[0], {}

    def step(self, action_int: int):
        """
        Теперь сначала применяем action, потом переходим к next_idx
        и считаем reward по новой позиции.
        Возвращаем (obs, reward, terminated, truncated, info).
        """
        # исходная цена текущего шага
        cur_price = self.price[self._idx]

        # применяем действие
        if action_int == 0:
            self._pos = 0
        elif action_int == 1:
            if self._pos != 1:
                self._pos = 1
                self._entry_price = cur_price
        else:  # action_int == 2
            if self._pos != -1:
                self._pos = -1
                self._entry_price = cur_price

        # переходим к следующей свече
        next_idx = self._idx + 1
        done = next_idx >= self.T - 1
        new_price = self.price[next_idx]

        # считаем P/L с учётом Kelly-фракции
        if self._pos == 0:
            reward = 0.0
        else:
            pnl = (new_price - self._entry_price) / self._entry_price
            pnl *= -1.0 if self._pos == -1 else 1.0
            reward = pnl * self._kelly_f

        # обновляем оценку p_win и Kelly-фракцию для следующих шагов
        self._update_p_win(reward)
        self._kelly_f = (
            kelly_fraction(self._p_win, self.odds, self.kelly_coeff)
            if self._pos != 0
            else 0.0
        )

        # сохраняем индекс, подготавливаем obs/info
        self._idx = next_idx
        obs = self.emb[self._idx]
        info = {"kelly_f": self._kelly_f, "pos": int(self._pos), "step": int(self._idx)}

        return obs, float(reward), done, False, info

    def render(self):
        pass
