"""
extract_features.py

Provides functions to compute multi-timeframe technical indicators
and label 5-minute microtrends, plus utilities to save feature
datasets and label distributions for downstream training.
"""

import json
from pathlib import Path

import pandas as pd
import joblib

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) for a price series.

    Args:
        series: Price series.
        period: Lookback period for RSI (default=14).

    Returns:
        RSI values.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build multi-timeframe features for 5m/15m/30m/60m, cross-TF ratios,
    missing-value flags, category codes, and fill gaps.

    Args:
        df: Input DataFrame with columns
            ['ts', 'ohlcv_5m_close', 'ohlcv_5m_vol',
             'whale_action_actionType', 'trades_side'].

    Returns:
        DataFrame of features, with 'ts' as a column.
    """
    df_feat = df.copy()
    df_feat['ts'] = pd.to_datetime(df_feat['ts'], utc=True)
    df_feat = df_feat.set_index('ts')

    close = df_feat['ohlcv_5m_close']
    vol = df_feat['ohlcv_5m_vol']
    ret = close.shift(1).pct_change()

    df_feat['vol_ma_5_10'] = vol.shift(1).rolling(10, min_periods=1).mean()
    df_feat['vol_std_5_10'] = vol.shift(1).rolling(10, min_periods=1).std(ddof=0)
    df_feat['ret_ma_5_10'] = ret.rolling(10, min_periods=1).mean()
    df_feat['ret_std_5_10'] = ret.rolling(10, min_periods=1).std(ddof=0)
    df_feat['ret_ewm_5m'] = ret.ewm(span=10, adjust=False).mean()

    df15 = pd.DataFrame({
        'close_15m': close
        .resample('15min', closed='right', label='right')
        .last()
        .shift(1),
        'vol_15m': vol
        .resample('15min', closed='right', label='right')
        .sum()
        .shift(1),
        'ret_15m': ret
        .resample('15min', closed='right', label='right')
        .std(ddof=0)
        .shift(1),
    })
    df15['ema20_15m'] = df15['close_15m'].ewm(span=20, adjust=False).mean()
    df15['ema_slope_15m'] = (df15['ema20_15m'] - df15['ema20_15m'].shift(5)) / 5
    df15['rsi14_15m'] = _rsi(df15['close_15m'], period=14)

    df30 = pd.DataFrame({
        'close_30m': close.resample('30min').last(),
        'vol_30m': vol.resample('30min').sum(),
        'ret_30m': ret.resample('30min').std(ddof=0),
    })
    ema12 = df30['close_30m'].ewm(span=12, adjust=False).mean()
    ema26 = df30['close_30m'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    df30['macd_12_26'] = macd
    df30['macd_signal_9'] = sig
    df30['macd_hist'] = macd - sig
    df30['rsi14_30m'] = _rsi(df30['close_30m'], period=14)

    df60 = pd.DataFrame({'close_60m': close.resample('60min').last()})
    df60['ema20_60m'] = df60['close_60m'].ewm(span=20, adjust=False).mean()
    df60['ema_slope_60m'] = (df60['ema20_60m'] - df60['ema20_60m'].shift(5)) / 5

    df_merged = (
        df_feat
        .join(df15.drop(columns=['close_15m']), how='left')
        .join(df30.drop(columns=['close_30m']), how='left')
        .join(df60.drop(columns=['close_60m']), how='left')
    )

    real_feats = [
        'vol_ma_5_10', 'vol_std_5_10', 'ret_ma_5_10', 'ret_std_5_10', 'ret_ewm_5m',
        'vol_15m', 'ret_15m', 'ema20_15m', 'ema_slope_15m', 'rsi14_15m',
        'vol_30m', 'ret_30m', 'macd_12_26', 'macd_signal_9', 'macd_hist', 'rsi14_30m',
        'ema20_60m', 'ema_slope_60m'
    ]
    for feat in real_feats:
        df_merged[f'{feat}_na'] = df_merged[feat].isna().astype('int8')

    df_merged[real_feats] = df_merged[real_feats].fillna(0).astype('float32')

    df_merged['ema_slope_acc'] = df_merged['ema_slope_15m'] - df_merged['ema_slope_60m']
    df_merged['rsi_ratio_15_30'] = df_merged['rsi14_15m'] / (df_merged['rsi14_30m'] + 1e-9)
    for feat in ['ema_slope_acc', 'rsi_ratio_15_30']:
        df_merged[f'{feat}_na'] = df_merged[feat].isna().astype('int8')
    df_merged[['ema_slope_acc', 'rsi_ratio_15_30']] = (
        df_merged[['ema_slope_acc', 'rsi_ratio_15_30']]
        .fillna(0)
        .astype('float32')
    )

    for col in ['whale_action_actionType', 'trades_side']:
        df_merged[col] = df_merged[col].astype('category')
        df_merged[f'{col}_code'] = df_merged[col].cat.codes.astype('int8')
        df_merged[f'{col}_na'] = df_merged[col].isna().astype('int8')

    return df_merged.reset_index()


import pandas as pd


def label_microtrend(
        df: pd.DataFrame,
        price_col: str = 'ohlcv_5m_close',
        min_candles: int = 3,
        profit_thr: float = 0.03,
        stop_loss_thr: float = 0.02
) -> pd.Series:
    close = df[price_col].values
    n = len(close)
    labels = pd.Series(0, index=df.index, dtype='int8')

    for i in range(n):
        entry_price = close[i]
        if entry_price == 0:
            continue
        min_ret = 0.0
        max_ret = 0.0

        for j in range(i + 1, n):
            ret = close[j] / entry_price - 1
            min_ret = min(min_ret, ret)
            max_ret = max(max_ret, ret)
            length = j - i + 1

            if length < min_candles:
                continue
            if ret >= profit_thr and min_ret >= -stop_loss_thr:
                labels.iat[i] = 1
                break
            if ret <= -profit_thr and max_ret <= stop_loss_thr:
                labels.iat[i] = -1
                break
            if min_ret < -stop_loss_thr or max_ret > stop_loss_thr:
                break

    return labels
