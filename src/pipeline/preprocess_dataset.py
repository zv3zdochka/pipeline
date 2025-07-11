from typing import Any, Dict

import json
import pandas as pd


def _parse_depth_json(raw: Any, levels: int = 3) -> Dict[str, float]:
    """
    Convert one depth_raw JSON string to numerical features.

    Parameters
    ----------
    raw : str | Any
        The JSON payload from the exchange or non-string when empty.
    levels : int, default 3
        How many book levels to keep.

    Returns
    -------
    dict
        Numerical features; missing values are returned as None.
    """
    if not isinstance(raw, str):
        return {}
    try:
        d = json.loads(raw)
        bids = [(float(p), float(q)) for p, q in d.get("b", [])[:levels]]
        asks = [(float(p), float(q)) for p, q in d.get("a", [])[:levels]]
        if not bids or not asks:
            return {}
        best_bid, _ = bids[0]
        best_ask, _ = asks[0]
        bid2_qty = bids[1][1] if len(bids) > 1 else None
        ask2_qty = asks[1][1] if len(asks) > 1 else None
        bid3_qty = bids[2][1] if len(bids) > 2 else None
        ask3_qty = asks[2][1] if len(asks) > 2 else None
        spread = best_ask - best_bid
        mid = (best_ask + best_bid) / 2
        bid_vol_sum = sum(q for _, q in bids)
        ask_vol_sum = sum(q for _, q in asks)
        vol_imbalance = (bid_vol_sum - ask_vol_sum) / (bid_vol_sum + ask_vol_sum + 1e-9)
        return {
            "ob_best_bid": best_bid,
            "ob_best_ask": best_ask,
            "ob_spread": spread,
            "ob_mid": mid,
            "ob_bid_vol_sum": bid_vol_sum,
            "ob_ask_vol_sum": ask_vol_sum,
            "ob_vol_imbalance": vol_imbalance,
            "ob_lvl2_bid_qty": bid2_qty,
            "ob_lvl2_ask_qty": ask2_qty,
            "ob_lvl3_bid_qty": bid3_qty,
            "ob_lvl3_ask_qty": ask3_qty,
        }
    except (json.JSONDecodeError, KeyError, ValueError, IndexError):
        return {}


def impute_missing(df: pd.DataFrame, *, cum_spread_window: int = 12) -> pd.DataFrame:
    """
    End-to-end cleaning for TFT:
      1. Interpolate dense metrics.
      2. Flag missing whale/trades data.
      3. Parse depth_raw to order-book features, deltas, rolling spread.
      4. Forward-fill and then fill NaN with zero.

    Parameters
    ----------
    df : pd.DataFrame
        Original raw market data.
    cum_spread_window : int
        Rolling-window length to accumulate spread.

    Returns
    -------
    pd.DataFrame
        Ready-to-model dataset: no NaN, explicit missing flags,
        float32 dense numbers, depth_raw removed.
    """
    out = df.copy()
    out = out.dropna(subset=["ts"])
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    out = out.set_index("ts").sort_index()

    depth_parsed = (
        out["depth_raw"]
        .apply(_parse_depth_json)
        .apply(pd.Series, dtype="float64")
    )
    depth_parsed["depth_was_missing"] = depth_parsed["ob_mid"].isna().astype("uint8")
    depth_cols = depth_parsed.columns.difference(["depth_was_missing"])
    depth_parsed[depth_cols] = depth_parsed[depth_cols].ffill().astype("float32")
    depth_parsed["ob_mid_diff"] = depth_parsed["ob_mid"].diff().fillna(0).astype("float32")
    depth_parsed[f"ob_spread_cum_{cum_spread_window}"] = (
        depth_parsed["ob_spread"]
        .rolling(window=cum_spread_window, min_periods=1)
        .sum()
        .astype("float32")
    )

    out = pd.concat([out.drop(columns=["depth_raw"]), depth_parsed], axis=1)

    always_empty = [
        "mkt_order_buy_sell_cnt_deltaCnt",
        "mkt_order_buy_sell_val_deltaUsd",
        "mkt_order_buy_sell_vol_deltaVol",
    ]
    out = out.drop(columns=[c for c in always_empty if c in out.columns])

    whale_cols = [
        "whale_action_actionType",
        "whale_action_sideVal",
        "whale_action_positionValue",
        "whale_action_price",
        "whale_action_qty",
        "whale_position_price",
        "whale_position_positionValue",
        "whale_position_size",
        "whale_position_unrealizedPnl",
        "whale_position_liquidationPx",
        "whale_position_marginUsed",
        "whale_position_leverage",
        "whale_position_cumFunding",
        "whale_position_entryPx",
    ]
    sparse_prefixes = ("trades_,")
    numeric_cols = out.select_dtypes(include="number").columns.tolist()
    depth_feature_cols = [c for c in out.columns if c.startswith("ob_")]
    skip = set(whale_cols + depth_feature_cols + ["depth_was_missing"])
    skip.update(c for c in numeric_cols if any(c.startswith(pref) for pref in sparse_prefixes))
    to_interpolate = [c for c in numeric_cols if c not in skip]

    out[to_interpolate] = out[to_interpolate].interpolate(
        method="linear", limit_direction="both"
    ).astype("float32")

    for col in whale_cols:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].isna().astype("uint8")

    for col in out.columns:
        if any(col.startswith(pref) for pref in sparse_prefixes):
            out[f"{col}_was_missing"] = out[col].isna().astype("uint8")

    out = out.reset_index()
    fill_cols = [c for c in out.columns if not c.startswith("depth_")]
    out[fill_cols] = out[fill_cols].fillna(0)

    if out.isna().any().any():
        raise ValueError("Detected NaN values after imputation")

    float64_cols = out.select_dtypes(include="float64").columns
    out = out.astype({c: "float32" for c in float64_cols})

    return out
