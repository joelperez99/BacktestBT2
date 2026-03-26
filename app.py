#!/usr/bin/env python3
"""
BitPredict Backtester v2 — Streamlit Edition
=============================================
Lógica idéntica a btc_live_predictor v2:
  predice al CIERRE de la vela → evalúa en la SIGUIENTE vela 5m
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import calendar as cal_lib
import io
import plotly.graph_objects as go

# ── Config ─────────────────────────────────────────────────────────────────────
SYMBOL   = "BTCUSDT"
BINANCE  = "https://api.binance.com/api/v3"
BET_SIZE = 10.0

TIER_DEF = {
    "S": {"min_volume": 500, "min_abs_move": 0.3},
    "A": {"min_volume": 300, "min_abs_move": 0.2},
    "B": {"min_volume": 200, "min_abs_move": 0.1},
    "C": {"min_volume": 100, "min_abs_move": 0.0},
}
TIER_ORDER = {"S": 0, "A": 1, "B": 2, "C": 3, "D": 4}
TIER_COLORS_HEX = {
    "S": "#ffd700", "A": "#00d68f", "B": "#7c6fff",
    "C": "#f7931a", "D": "#ff4757",
}
_CST_SHIFT = pd.Timedelta(hours=6)
_CST_DELTA = timedelta(hours=-6)


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

def fetch_klines_range(interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    all_rows = []
    current  = start_ms
    while current < end_ms:
        r = requests.get(
            f"{BINANCE}/klines",
            params={"symbol": SYMBOL, "interval": interval,
                    "startTime": current, "endTime": end_ms, "limit": 1000},
            timeout=15,
        )
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        all_rows.extend(rows)
        current = rows[-1][6] + 1
        if len(rows) < 1000:
            break
    if not all_rows:
        return pd.DataFrame()
    cols = ["open_time","open","high","low","close","volume",
            "close_time","qvol","trades","tb","tq","ignore"]
    df = pd.DataFrame(all_rows, columns=cols)
    for c in ("open","high","low","close","volume"):
        df[c] = df[c].astype(float)
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# INDICADORES  (idénticos al live v2)
# ══════════════════════════════════════════════════════════════════════════════

def _ema(s, n):   return s.ewm(span=n, adjust=False).mean()

def _rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return float((100 - 100 / (1 + g / l.replace(0, np.nan))).iloc[-1])

def _macd_hist(s):
    m = _ema(s, 12) - _ema(s, 26)
    return float((m - _ema(m, 9)).iloc[-1])

def _bb_pct(s, n=20):
    mid = s.rolling(n).mean(); std = s.rolling(n).std()
    lo  = (mid - 2*std).iloc[-1]; hi = (mid + 2*std).iloc[-1]
    return float((s.iloc[-1] - lo) / ((hi - lo) + 1e-9))

def _mom(s, n=5):
    return float((s.iloc[-1] - s.iloc[-(n+1)]) / s.iloc[-(n+1)] * 100)

def _vol_ratio(v, recent=5, hist=20):
    return float(v.iloc[-recent:].mean() / (v.iloc[-(hist+recent):-recent].mean() + 1e-9))


def predict_from_df(df1m: pd.DataFrame):
    c, v = df1m["close"], df1m["volume"]
    votes = []

    r_val = _rsi(c)
    if   r_val < 42: votes.append(("RSI", "UP",      min((42 - r_val) / 42, 1)))
    elif r_val > 58: votes.append(("RSI", "DOWN",    min((r_val - 58) / 42, 1)))
    else:            votes.append(("RSI", "NEUTRAL", 0.0))

    mh = _macd_hist(c)
    votes.append(("MACD", "UP" if mh > 0 else "DOWN", min(abs(mh) / 30, 1)))

    e9  = float(_ema(c, 9).iloc[-1]); e21 = float(_ema(c, 21).iloc[-1])
    dp  = (e9 - e21) / e21 * 100
    votes.append(("EMA", "UP" if e9 > e21 else "DOWN", min(abs(dp) / 0.4, 1)))

    bb = _bb_pct(c)
    if   bb < 0.30: votes.append(("Bollinger", "UP",      (0.30 - bb) / 0.30))
    elif bb > 0.70: votes.append(("Bollinger", "DOWN",    (bb - 0.70) / 0.30))
    else:           votes.append(("Bollinger", "NEUTRAL", 0.0))

    mom_v = _mom(c, 5)
    if   mom_v >  0.04: votes.append(("Momentum", "UP",   min(mom_v / 0.3, 1)))
    elif mom_v < -0.04: votes.append(("Momentum", "DOWN", min(abs(mom_v) / 0.3, 1)))
    else:               votes.append(("Momentum", "NEUTRAL", 0.0))

    vr = _vol_ratio(v)
    if vr > 1.25:
        votes.append(("Volume", "UP" if mom_v >= 0 else "DOWN", min((vr - 1) / 1.5, 1)))

    e50 = float(_ema(c, 50).iloc[-1])
    votes.append(("Trend50", "UP" if c.iloc[-1] > e50 else "DOWN", 0.4))

    up_s  = sum(s for _, d, s in votes if d == "UP")
    dn_s  = sum(s for _, d, s in votes if d == "DOWN")
    total = up_s + dn_s or 1e-9
    up_pct = up_s / total * 100; dn_pct = dn_s / total * 100
    direction = "UP" if up_s >= dn_s else "DOWN"
    return direction, max(up_pct, dn_pct), up_pct, dn_pct, votes


# ══════════════════════════════════════════════════════════════════════════════
# TIER / FILTER
# ══════════════════════════════════════════════════════════════════════════════

def get_trade_tier(volume: float, abs_move: float) -> str:
    if volume >= TIER_DEF["S"]["min_volume"] and abs_move >= TIER_DEF["S"]["min_abs_move"]:
        return "S"
    if volume >= TIER_DEF["A"]["min_volume"] and abs_move >= TIER_DEF["A"]["min_abs_move"]:
        return "A"
    if volume >= TIER_DEF["B"]["min_volume"] and abs_move >= TIER_DEF["B"]["min_abs_move"]:
        return "B"
    if volume >= TIER_DEF["C"]["min_volume"]:
        return "C"
    return "D"


def passes_filter(volume: float, abs_move: float, hour: int,
                  confidence: float, filters: dict) -> bool:
    if filters["min_volume"]     and volume     < filters["min_volume"]:     return False
    if filters["min_abs_move"]   and abs_move   < filters["min_abs_move"]:   return False
    if filters["allowed_hours"]  and hour not in filters["allowed_hours"]:   return False
    if filters["min_confidence"] and confidence < filters["min_confidence"]: return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# PROCESAMIENTO DE VELAS
# ══════════════════════════════════════════════════════════════════════════════

def _process_candles(df5m_range: pd.DataFrame, df1m_full: pd.DataFrame,
                     filters: dict) -> list:
    results = []
    n = len(df5m_range)
    for i in range(n - 1):
        c5      = df5m_range.iloc[i]
        c5_next = df5m_range.iloc[i + 1]

        ctx = df1m_full[df1m_full["open_time"] < c5["close_time"]].tail(120)
        if len(ctx) < 50:
            continue

        direction, conf, up_pct, dn_pct, votes = predict_from_df(ctx)

        sig_vol  = float(c5["volume"])
        sig_move = abs((float(c5["close"]) - float(c5["open"])) / float(c5["open"]) * 100)

        open_price  = float(c5["close"])
        close_price = float(c5_next["close"])
        actual  = "UP" if close_price >= open_price else "DOWN"
        correct = direction == actual
        pct_move = (close_price - open_price) / open_price * 100

        tier   = get_trade_tier(sig_vol, sig_move)
        c_open = c5["close_time"]
        hour   = c_open.to_pydatetime().astimezone().hour
        in_flt = passes_filter(sig_vol, sig_move, hour, conf, filters)

        m1s = df1m_full[
            (df1m_full["open_time"] >= c5_next["open_time"]) &
            (df1m_full["open_time"] <  c5_next["close_time"])
        ].reset_index(drop=True)
        min_correct = {}
        for mi, m1r in m1s.iterrows():
            mn = mi + 1
            mc = float(m1r["close"])
            min_correct[mn] = ("UP" if mc >= open_price else "DOWN") == direction

        sigs = "|".join(f"{v[0]}:{v[1]}" for v in votes)
        loc  = c5_next["open_time"].to_pydatetime().astimezone()
        utc  = c5_next["open_time"].to_pydatetime()

        results.append({
            "date":            loc.strftime("%Y-%m-%d"),
            "time":            loc.strftime("%H:%M"),
            "hour":            loc.hour,
            "timestamp_utc":   utc.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_local": loc.strftime("%Y-%m-%d %H:%M:%S"),
            "open_price":      open_price,
            "close_price":     close_price,
            "pct_move":        pct_move,
            "abs_move":        abs(pct_move),
            "prediction":      direction,
            "actual":          actual,
            "correct":         correct,
            "confidence":      conf,
            "up_pct":          up_pct,
            "dn_pct":          dn_pct,
            "signals":         sigs,
            "minute_correct":  min_correct,
            "high":            float(c5_next["high"]),
            "low":             float(c5_next["low"]),
            "volume":          float(c5_next["volume"]),
            "signal_volume":   sig_vol,
            "signal_move":     sig_move,
            "tier":            tier,
            "in_filter":       in_flt,
        })
    return results


# ══════════════════════════════════════════════════════════════════════════════
# P&L
# ══════════════════════════════════════════════════════════════════════════════

def simulate_pnl(df: pd.DataFrame, bet_size: float = BET_SIZE,
                 min_tier: str = "B") -> dict:
    filtered = df[df["tier"].map(TIER_ORDER) <= TIER_ORDER.get(min_tier, 4)]
    n = len(filtered)
    if n == 0:
        return dict(total_trades=0, wins=0, losses=0, accuracy=0,
                    total_pnl=0.0, pnl_per_day=0.0, roi_pct=0.0, days=0)
    wins      = int(filtered["correct"].sum())
    losses    = n - wins
    total_pnl = (wins - losses) * bet_size
    days      = max(1, filtered["date"].nunique())
    return dict(
        total_trades=n, wins=wins, losses=losses,
        accuracy=round(wins / n * 100, 2),
        total_pnl=round(total_pnl, 2),
        pnl_per_day=round(total_pnl / days, 2),
        roi_pct=round(total_pnl / (n * bet_size) * 100, 2),
        days=days,
    )


def filtered_pnl(df: pd.DataFrame, tiers: list, bet_size: float = BET_SIZE) -> dict:
    sub = df[df["tier"].isin(tiers)] if tiers else df.iloc[0:0]
    n   = len(sub)
    if n == 0:
        return dict(total_trades=0, wins=0, losses=0, accuracy=0,
                    total_pnl=0.0, pnl_per_day=0.0, roi_pct=0.0, days=0)
    wins      = int(sub["correct"].sum())
    losses    = n - wins
    total_pnl = (wins - losses) * bet_size
    days      = max(1, sub["date"].nunique())
    return dict(
        total_trades=n, wins=wins, losses=losses,
        accuracy=round(wins / n * 100, 2),
        total_pnl=round(total_pnl, 2),
        pnl_per_day=round(total_pnl / days, 2),
        roi_pct=round(total_pnl / (n * bet_size) * 100, 2),
        days=days,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ESTADÍSTICAS
# ══════════════════════════════════════════════════════════════════════════════

def _compute_stats(df: pd.DataFrame) -> dict:
    total  = len(df)
    wins   = int(df["correct"].sum())
    losses = total - wins
    acc    = wins / total * 100

    by_hour = df.groupby("hour")["correct"].agg(["sum","count"]).rename(
        columns={"sum":"wins","count":"total"})
    by_hour["accuracy"] = by_hour["wins"] / by_hour["total"] * 100

    bins_c = [50,55,60,65,70,75,80,85,90,95,101]
    labs_c = ["50-55","55-60","60-65","65-70","70-75",
              "75-80","80-85","85-90","90-95","95-100"]
    df2 = df.copy()
    df2["conf_bin"] = pd.cut(df2["confidence"], bins=bins_c, labels=labs_c, right=False)
    by_conf = df2.groupby("conf_bin", observed=False)["correct"].agg(["sum","count"]).rename(
        columns={"sum":"wins","count":"total"})
    by_conf["accuracy"] = by_conf.apply(
        lambda r: r["wins"]/r["total"]*100 if r["total"]>0 else float("nan"), axis=1)

    minute_acc = {}
    for mn in range(1, 6):
        col   = df["minute_correct"].apply(lambda d: d.get(mn))
        valid = col.dropna()
        if len(valid):
            minute_acc[mn] = {"correct": int(valid.sum()), "total": int(len(valid)),
                               "accuracy": valid.mean() * 100}
    best_minute = max(minute_acc, key=lambda m: abs(minute_acc[m]["accuracy"]-50), default=None)

    all_sigs = []
    for sig_str in df["signals"]:
        for part in sig_str.split("|"):
            if ":" in part:
                sn, sd = part.split(":", 1)
                all_sigs.append({"signal": sn, "direction": sd})
    sig_df = pd.DataFrame(all_sigs)
    signal_bias = {}
    if not sig_df.empty:
        for sn, grp in sig_df.groupby("signal"):
            signal_bias[sn] = {
                "UP":      int((grp["direction"]=="UP").sum()),
                "DOWN":    int((grp["direction"]=="DOWN").sum()),
                "NEUTRAL": int((grp["direction"]=="NEUTRAL").sum()),
                "total":   int(len(grp)),
            }

    by_pred = df.groupby("prediction")["correct"].agg(["sum","count"])
    by_pred["accuracy"] = by_pred["sum"] / by_pred["count"] * 100

    max_ws = max_ls = cw = cl = 0
    for c in df["correct"]:
        if c: cw += 1; cl = 0
        else: cl += 1; cw = 0
        max_ws = max(max_ws, cw); max_ls = max(max_ls, cl)

    pct_desc  = df["pct_move"].describe()
    conf_win  = df[df["correct"]]["confidence"].mean()  if wins   else float("nan")
    conf_loss = df[~df["correct"]]["confidence"].mean() if losses else float("nan")

    bhs = by_hour[by_hour["total"] >= 3].sort_values("accuracy", ascending=False)
    best_hour  = int(bhs.index[0])  if len(bhs) else None
    worst_hour = int(bhs.index[-1]) if len(bhs) else None

    vol_win  = df[df["correct"]]["signal_volume"].mean()  if wins   else float("nan")
    vol_loss = df[~df["correct"]]["signal_volume"].mean() if losses else float("nan")

    df2["range_pct"] = (df2["high"] - df2["low"]) / df2["open_price"] * 100
    range_win  = df2[df2["correct"]]["range_pct"].mean()  if wins   else float("nan")
    range_loss = df2[~df2["correct"]]["range_pct"].mean() if losses else float("nan")

    by_day = None
    if "date" in df.columns and df["date"].nunique() > 1:
        by_day = df.groupby("date")["correct"].agg(["sum","count"]).rename(
            columns={"sum":"wins","count":"total"})
        by_day["accuracy"] = by_day["wins"] / by_day["total"] * 100
        by_day = by_day.sort_index()

    by_tier = df.groupby("tier")["correct"].agg(["sum","count"]).rename(
        columns={"sum":"wins","count":"total"})
    by_tier["accuracy"] = by_tier["wins"] / by_tier["total"] * 100
    by_tier = by_tier.reindex([t for t in ["S","A","B","C","D"] if t in by_tier.index])

    pnl_rows = []
    for mt in ["S", "A", "B", "C", "D"]:
        p = simulate_pnl(df, bet_size=BET_SIZE, min_tier=mt)
        p["min_tier"] = mt
        pnl_rows.append(p)
    pnl_table = pd.DataFrame(pnl_rows).set_index("min_tier")

    vol_bins = [0, 50, 100, 200, 300, 500, 1000, float("inf")]
    vol_labs = ["0-50","50-100","100-200","200-300","300-500","500-1000","1000+"]
    df2["sig_vol_bin"] = pd.cut(df2["signal_volume"], bins=vol_bins, labels=vol_labs, right=False)
    by_signal_vol = df2.groupby("sig_vol_bin", observed=False)["correct"].agg(["sum","count"]).rename(
        columns={"sum":"wins","count":"total"})
    by_signal_vol["accuracy"] = by_signal_vol.apply(
        lambda r: r["wins"]/r["total"]*100 if r["total"]>0 else float("nan"), axis=1)

    df2["tgt_vol_bin"] = pd.cut(df2["volume"], bins=vol_bins, labels=vol_labs, right=False)
    by_target_vol = df2.groupby("tgt_vol_bin", observed=False)["correct"].agg(["sum","count"]).rename(
        columns={"sum":"wins","count":"total"})
    by_target_vol["accuracy"] = by_target_vol.apply(
        lambda r: r["wins"]/r["total"]*100 if r["total"]>0 else float("nan"), axis=1)

    filt_df    = df[df["tier"].map(TIER_ORDER) <= TIER_ORDER["B"]]
    filt_acc   = filt_df["correct"].mean() * 100 if len(filt_df) else float("nan")
    filt_total = len(filt_df)
    n_days     = max(1, df["date"].nunique())
    filt_per_day = filt_total / n_days

    return dict(
        total=total, wins=wins, losses=losses, accuracy=acc,
        by_hour=by_hour, by_conf=by_conf,
        minute_acc=minute_acc, best_minute=best_minute,
        signal_bias=signal_bias, by_pred=by_pred,
        max_win_streak=max_ws, max_loss_streak=max_ls,
        pct_desc=pct_desc, conf_win=conf_win, conf_loss=conf_loss,
        best_hour=best_hour, worst_hour=worst_hour,
        vol_win=vol_win, vol_loss=vol_loss,
        range_win=range_win, range_loss=range_loss,
        by_day=by_day, by_tier=by_tier, pnl_table=pnl_table,
        by_signal_vol=by_signal_vol, by_target_vol=by_target_vol,
        filt_acc=filt_acc, filt_total=filt_total, filt_per_day=filt_per_day,
    )


# ══════════════════════════════════════════════════════════════════════════════
# CST
# ══════════════════════════════════════════════════════════════════════════════

def _to_cst(utc_str: str) -> str:
    dt  = datetime.strptime(utc_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    cst = dt + _CST_DELTA
    return cst.strftime("%Y-%m-%d %H:%M:%S")


# ══════════════════════════════════════════════════════════════════════════════
# MOTORES
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(date_str: str, filters: dict) -> dict:
    day_start    = pd.Timestamp(date_str, tz="UTC") + _CST_SHIFT
    day_end      = day_start + pd.Timedelta(days=1)
    warmup_start = day_start - pd.Timedelta(hours=2)
    fetch_end    = day_end   + pd.Timedelta(minutes=10)
    start_ms = int(warmup_start.value // 1_000_000)
    end_ms   = int(fetch_end.value    // 1_000_000)

    df1m = fetch_klines_range("1m", start_ms, end_ms)
    df5m = fetch_klines_range("5m", start_ms, end_ms)
    if df1m.empty or df5m.empty:
        raise ValueError("No se obtuvieron datos para esa fecha.")

    df5m_day = df5m[(df5m["open_time"] >= day_start) &
                    (df5m["open_time"] <  day_end)].reset_index(drop=True)
    last_signal_close = df5m_day.iloc[-1]["close_time"] if not df5m_day.empty else day_end
    df5m_window = df5m[(df5m["open_time"] >= day_start) &
                       (df5m["open_time"] <= last_signal_close)].reset_index(drop=True)

    records = _process_candles(df5m_window, df1m, filters)
    if not records:
        raise ValueError("No hay velas 5m procesables para ese día.")

    df = pd.DataFrame(records)
    df = df[df["timestamp_utc"].apply(lambda u: _to_cst(u)[:10] == date_str)].reset_index(drop=True)
    if df.empty:
        raise ValueError("No hay velas 5m en fecha CST para ese día.")

    stats = _compute_stats(df)
    return {"records": df.to_dict("records"), "df": df, "stats": stats,
            "label": date_str, "mode": "day"}


def run_backtest_month(year_str: str, month_str: str, filters: dict,
                       progress_cb=None) -> dict:
    year  = int(year_str); month = int(month_str)
    month_start  = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    last_day     = cal_lib.monthrange(year, month)[1]
    month_end    = month_start + pd.Timedelta(days=last_day)
    warmup_start = month_start - pd.Timedelta(hours=2)
    fetch_end    = month_end   + pd.Timedelta(minutes=10)
    start_ms = int(warmup_start.value // 1_000_000)
    end_ms   = int(fetch_end.value    // 1_000_000)

    df1m = fetch_klines_range("1m", start_ms, end_ms)
    df5m = fetch_klines_range("5m", start_ms, end_ms)
    if df1m.empty or df5m.empty:
        raise ValueError("No se obtuvieron datos para ese mes.")

    df5m_month = df5m[(df5m["open_time"] >= month_start) &
                      (df5m["open_time"] <  fetch_end)].reset_index(drop=True)
    all_records = []
    for idx in range(last_day):
        day_ts     = month_start + pd.Timedelta(days=idx)
        day_end_ts = day_ts + pd.Timedelta(days=1)
        df5m_day = df5m_month[
            (df5m_month["open_time"] >= day_ts) &
            (df5m_month["open_time"] <  day_end_ts + pd.Timedelta(minutes=5))
        ].reset_index(drop=True)
        if not df5m_day.empty:
            all_records.extend(_process_candles(df5m_day, df1m, filters))
        if progress_cb:
            progress_cb(idx + 1, last_day)

    if not all_records:
        raise ValueError("No hay velas 5m procesables para ese mes.")

    df    = pd.DataFrame(all_records)
    stats = _compute_stats(df)
    return {"records": all_records, "df": df, "stats": stats,
            "label": f"{year_str}-{month_str}", "mode": "month"}


# ══════════════════════════════════════════════════════════════════════════════
# EXCEL (bytes para download_button)
# ══════════════════════════════════════════════════════════════════════════════

def save_excel_bytes(result: dict, bet_size: float = BET_SIZE) -> bytes:
    df = result["df"].copy()
    s  = result["stats"]
    pred_cols = ["date","time","hour","tier","in_filter","prediction","actual","correct",
                 "confidence","pct_move","abs_move","signal_volume","signal_move",
                 "volume","open_price","close_price","high","low"]
    n_days = max(1, df["date"].nunique())
    resumen_rows = [
        ("Período",                  result["label"]),
        ("Lógica",                   "v2 — predice al cierre, evalúa en siguiente vela"),
        ("Días analizados",          n_days),
        ("Total velas",              s["total"]),
        ("Aciertos",                 s["wins"]),
        ("Fallos",                   s["losses"]),
        ("Precisión global %",       round(s["accuracy"], 2)),
        ("── Stats S+A+B ──",        ""),
        ("Trades filtrados",         s["filt_total"]),
        ("Trades filtrados / día",   round(s["filt_per_day"], 1)),
        ("Precisión filtrada %",     round(s["filt_acc"], 2) if not np.isnan(s["filt_acc"]) else ""),
        ("── Métricas ──",           ""),
        ("Racha max aciertos",       s["max_win_streak"]),
        ("Racha max fallos",         s["max_loss_streak"]),
        ("Conf. media aciertos",     round(s["conf_win"],  2) if not np.isnan(s["conf_win"])  else ""),
        ("Conf. media fallos",       round(s["conf_loss"], 2) if not np.isnan(s["conf_loss"]) else ""),
    ]
    df_resumen = pd.DataFrame(resumen_rows, columns=["Métrica", "Valor"])
    min_rows = [
        {"Minuto": mn, "Aciertos": info["correct"], "Total": info["total"],
         "Precisión%": round(info["accuracy"], 2),
         "★": "★" if mn == s["best_minute"] else ""}
        for mn, info in sorted(s["minute_acc"].items())
    ]
    df_min  = pd.DataFrame(min_rows)
    df_sigs = pd.DataFrame([
        {"Señal": sn, "UP": v["UP"], "DOWN": v["DOWN"],
         "NEUTRAL": v["NEUTRAL"], "Total": v["total"]}
        for sn, v in s["signal_bias"].items()
    ])
    df_tier = s["by_tier"].copy().reset_index()
    df_tier.columns = ["Tier", "Aciertos", "Total", "Precisión%"]
    pnl_df = s["pnl_table"].reset_index()
    pnl_df.columns = ["Min Tier","Trades","Wins","Losses","Accuracy%","P&L Total($)","P&L/Día($)","ROI%","Días"]
    df_sigvol = s["by_signal_vol"].reset_index()
    df_sigvol.columns = ["Vol Vela Señal (BTC)","Aciertos","Total","Precisión%"]

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_resumen.to_excel(writer,                  sheet_name="Resumen",        index=False)
        df[pred_cols].to_excel(writer,               sheet_name="Predicciones",   index=False)
        s["by_hour"].reset_index().to_excel(writer,  sheet_name="Por Hora",       index=False)
        s["by_conf"].reset_index().to_excel(writer,  sheet_name="Por Confianza",  index=False)
        df_min.to_excel(writer,                      sheet_name="Por Minuto",     index=False)
        s["by_pred"].reset_index().to_excel(writer,  sheet_name="Por Direccion",  index=False)
        df_sigs.to_excel(writer,                     sheet_name="Señales",        index=False)
        if s["by_day"] is not None:
            s["by_day"].reset_index().to_excel(writer, sheet_name="Por Dia",      index=False)
        df_tier.to_excel(writer,                     sheet_name="Por Tier",       index=False)
        pnl_df.to_excel(writer,                      sheet_name="P&L Simulado",   index=False)
        df_sigvol.to_excel(writer,                   sheet_name="Vol Vela Señal", index=False)
    return output.getvalue()


def save_excel_detail_bytes(result: dict) -> bytes:
    df = result["df"].copy()
    detail_rows = []
    for _, r in df.iterrows():
        utc_str = r.get("timestamp_utc", f"{r['date']} {r['time']}:00")
        cst_str = _to_cst(utc_str)
        detail_rows.append({
            "Timestamp CST":        cst_str,
            "Hora Local":           r.get("timestamp_local", ""),
            "Open Price (to beat)": round(r["open_price"],    2),
            "Prediccion":           r["prediction"],
            "Confianza %":          round(r["confidence"],    2),
            "UP %":                 round(r["up_pct"],        2),
            "DOWN %":               round(r["dn_pct"],        2),
            "Vol Señal BTC":        round(r["signal_volume"], 2),
            "Move Señal %":         round(r["signal_move"],   4),
            "Tier":                 r["tier"],
            "En Filtro":            "SI" if r["in_filter"] else "NO",
            "Close Price":          round(r["close_price"],   2),
            "Direccion Real":       r["actual"],
            "Correcto":             "SI" if r["correct"] else "NO",
            "Pct Move %":           round(r["pct_move"],      4),
            "Señales":              r["signals"],
        })
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(detail_rows).to_excel(writer, sheet_name="Detalle", index=False)
    return output.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPER — DETALLE DE DÍA
# ══════════════════════════════════════════════════════════════════════════════

def render_day_detail(df_main: pd.DataFrame, date_str: str,
                      sel_tiers: list, stake: float):
    day_df = df_main[
        (df_main["date"] == date_str) & (df_main["tier"].isin(sel_tiers))
    ].reset_index(drop=True)
    if day_df.empty:
        st.info("Sin trades para los tiers seleccionados en este día.")
        return

    wins     = int(day_df["correct"].sum())
    losses   = len(day_df) - wins
    total    = len(day_df)
    net_pnl  = (wins - losses) * stake
    win_rate = wins / total * 100 if total else 0.0

    BANCO_INI = max(100_000.0, stake * 100)
    running   = BANCO_INI
    curve     = [running]
    min_bank = max_bank = running
    min_idx  = max_idx  = 0
    for i, row in day_df.iterrows():
        running += stake if row["correct"] else -stake
        curve.append(running)
        if running < min_bank: min_bank, min_idx = running, int(i) + 1
        if running > max_bank: max_bank, max_idx = running, int(i) + 1

    gain_color = "#00d68f" if net_pnl > 0 else ("#ff4757" if net_pnl < 0 else "#a0a3b1")
    tier_str   = "+".join(sel_tiers) if sel_tiers else "ninguno"

    st.markdown(
        f"<div class='section-hdr'>🔍 Detalle — {date_str} &nbsp;·&nbsp; "
        f"Tiers {tier_str} &nbsp;·&nbsp; ${stake:,.0f}/trade</div>",
        unsafe_allow_html=True)

    # Fila 1
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total operaciones", total)
    d2.metric("✅ Gana (SI)", wins)
    d3.metric("❌ Pierde (NO)", losses)
    d4.metric("Balance neto", f"${net_pnl:+,.0f}")

    # Fila 2
    d5, d6, d7, d8 = st.columns(4)
    d5.metric("Banco inicial", f"${BANCO_INI:,.0f}")
    d6.metric("Banco final",   f"${curve[-1]:,.0f}")
    d7.metric("Win rate",      f"{win_rate:.1f}%")
    d8.metric("Resultado neto",f"${net_pnl:+,.0f}")

    # Fila 3
    d9, d10 = st.columns(2)
    d9.metric( f"Banco mínimo (op #{min_idx})", f"${min_bank:,.0f}")
    d10.metric(f"Banco máximo (op #{max_idx})", f"${max_bank:,.0f}")

    # Curva equity
    line_color = "#00d68f" if net_pnl >= 0 else "#ff4757"
    fig = go.Figure()
    x = list(range(len(curve)))
    fig.add_trace(go.Scatter(
        x=x, y=curve, mode="lines",
        line=dict(color=line_color, width=2),
        fill="tozeroy",
        fillcolor="rgba(0,214,143,0.08)" if net_pnl >= 0 else "rgba(255,71,87,0.08)",
        hovertemplate="Op #%{x}<br>Banco: $%{y:,.0f}<extra></extra>",
        name="Banco",
    ))
    fig.add_hline(y=BANCO_INI, line_dash="dash", line_color="#6e7191", line_width=1)
    fig.update_layout(
        plot_bgcolor="#0d0f1a", paper_bgcolor="#0d0f1a",
        font_color="#e8eaf6", height=280,
        margin=dict(l=20, r=20, t=10, b=30),
        yaxis=dict(gridcolor="#1e2236", tickprefix="$"),
        xaxis=dict(gridcolor="#1e2236", title="Operación #"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Secuencia de resultados
    st.markdown("**Secuencia de resultados**")
    badges = ""
    for i, (_, row) in enumerate(day_df.iterrows()):
        ok  = bool(row["correct"])
        clr = "#00d68f" if ok else "#ff4757"
        bg  = "#0a2e1e" if ok else "#2e0a0a"
        lbl = "SI" if ok else "NO"
        badges += (f"<span style='background:{bg};color:{clr};font-weight:bold;"
                   f"padding:3px 7px;border-radius:4px;margin:2px;font-size:11px;"
                   f"display:inline-block;'>{lbl}</span>")
    st.markdown(f"<div style='line-height:2.2;'>{badges}</div>", unsafe_allow_html=True)

    # Tabla de operaciones
    st.markdown("**Operaciones del día**")
    tbl_rows = []
    for _, row in day_df.iterrows():
        ok    = bool(row["correct"])
        t_pnl = stake if ok else -stake
        tbl_rows.append({
            "Hora":        row["time"],
            "Predicción":  row["prediction"],
            "Real":        row["actual"],
            "Correcto":    "✅ SI" if ok else "❌ NO",
            "Confianza %": f"{row['confidence']:.1f}",
            "Tier":        row["tier"],
            "En Filtro":   "SI" if row["in_filter"] else "NO",
            f"P&L (${stake:.0f})": f"${t_pnl:+,.0f}",
        })
    st.dataframe(pd.DataFrame(tbl_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="BitPredict Backtester v2",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background-color: #0d0f1a; }
section[data-testid="stSidebar"] { background-color: #161929; }
div[data-testid="metric-container"] {
    background-color: #161929;
    border: 1px solid #1e2236;
    border-radius: 8px;
    padding: 12px 16px;
}
.section-hdr {
    font-size: 14px; font-weight: bold; color: #7c6fff;
    border-bottom: 1px solid #1e2236;
    padding-bottom: 6px; margin: 20px 0 12px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-size:20px;font-weight:bold;color:#f7931a;margin-bottom:4px;'>"
        "₿ BitPredict Backtester v2</div>"
        "<div style='font-size:11px;color:#6e7191;margin-bottom:16px;'>"
        "Predice al cierre → evalúa siguiente vela</div>",
        unsafe_allow_html=True)
    st.divider()

    mode = st.radio("Modo", ["Por día", "Por mes"], horizontal=True)

    today  = datetime.now()
    years  = list(range(today.year - 2, today.year + 1))
    months = list(range(1, 13))

    cy, cm = st.columns(2)
    with cy:
        year_sel  = st.selectbox("Año",  years,  index=len(years)-1)
    with cm:
        month_sel = st.selectbox("Mes",  months, index=today.month - 1,
                                  format_func=lambda m: f"{m:02d}")

    if mode == "Por día":
        days_in_month = cal_lib.monthrange(year_sel, month_sel)[1]
        default_day   = min(today.day - 1, days_in_month - 1)
        day_sel = st.selectbox("Día", list(range(1, days_in_month + 1)),
                                index=max(0, default_day - 1),
                                format_func=lambda d: f"{d:02d}")

    st.divider()
    st.markdown("**Filtros de señal**")
    cv, cm2 = st.columns(2)
    with cv:
        flt_vol  = st.number_input("Vol ≥ (BTC)", value=200.0, step=50.0, min_value=0.0)
    with cm2:
        flt_move = st.number_input("|move| ≥ %",  value=0.1,   step=0.05, min_value=0.0)

    st.divider()
    stake = st.number_input("Stake $ / trade", value=10.0, step=5.0, min_value=1.0)

    st.divider()
    st.markdown("**Tiers activos**")
    tcols = st.columns(5)
    defaults_t = {"S": True, "A": True, "B": True, "C": False, "D": False}
    tier_sel = {}
    for i, tier in enumerate(["S","A","B","C","D"]):
        with tcols[i]:
            tier_sel[tier] = st.checkbox(
                tier, value=defaults_t[tier], key=f"tier_{tier}",
                help=f"Tier {tier}")
    selected_tiers = [t for t, v in tier_sel.items() if v]

    st.divider()
    run_btn = st.button("▶  Ejecutar Backtest", use_container_width=True, type="primary")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex;align-items:center;gap:14px;margin-bottom:10px;'>
  <div style='width:44px;height:44px;background:#f7931a;border-radius:50%;
              display:flex;align-items:center;justify-content:center;
              font-size:22px;font-weight:bold;color:white;flex-shrink:0;'>₿</div>
  <div>
    <div style='font-size:22px;font-weight:bold;color:#e8eaf6;line-height:1.2;'>
      Backtester v2 — Lógica Live</div>
    <div style='font-size:12px;color:#6e7191;'>
      Tier = vol+|move| de la vela SEÑAL (cerrada) &nbsp;•&nbsp;
      contexto 1m hasta cierre señal &nbsp;•&nbsp;
      evalúa en la SIGUIENTE vela 5m</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Run ─────────────────────────────────────────────────────────────────────────
filters = {
    "min_volume":     flt_vol,
    "min_abs_move":   flt_move,
    "allowed_hours":  None,
    "min_confidence": None,
}

if run_btn:
    if mode == "Por día":
        date_str = f"{year_sel:04d}-{month_sel:02d}-{day_sel:02d}"
        with st.spinner(f"Descargando y procesando {date_str}…"):
            try:
                result = run_backtest(date_str, filters)
                st.session_state["result"] = result
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        year_s  = f"{year_sel:04d}"
        month_s = f"{month_sel:02d}"
        prog    = st.progress(0, text="Descargando datos del mes…")
        def _prog_cb(done, total):
            prog.progress(done / total, text=f"Procesando día {done}/{total}…")
        try:
            result = run_backtest_month(year_s, month_s, filters, _prog_cb)
            st.session_state["result"] = result
            prog.empty()
        except Exception as e:
            prog.empty()
            st.error(f"Error: {e}")

# ── Display ─────────────────────────────────────────────────────────────────────
if "result" in st.session_state:
    result  = st.session_state["result"]
    s       = result["stats"]
    df_main = result["df"]
    mode_r  = result["mode"]
    n_days  = max(1, df_main["date"].nunique())
    pnl_sel = filtered_pnl(df_main, selected_tiers, stake)
    tier_label = "+".join(selected_tiers) if selected_tiers else "ninguno"

    # ── Resumen global ──────────────────────────────────────────────────────
    st.markdown(f"<div class='section-hdr'>📊 Backtest v2 — {result['label']}"
                f"{'  ·  '+str(n_days)+' días' if mode_r=='month' else ''}</div>",
                unsafe_allow_html=True)
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total velas",          s["total"])
    c2.metric("Aciertos",             s["wins"])
    c3.metric("Fallos",               s["losses"])
    c4.metric("Precisión global",     f"{s['accuracy']:.1f}%")
    c5.metric("Racha max. aciertos",  s["max_win_streak"])
    c6.metric("Racha max. fallos",    s["max_loss_streak"])

    # ── Rendimiento filtrado ────────────────────────────────────────────────
    filt_acc_dyn = pnl_sel["accuracy"] if pnl_sel["total_trades"] else float("nan")
    st.markdown(
        f"<div class='section-hdr'>🎯 Rendimiento Filtrado &nbsp;·&nbsp; "
        f"Tiers: {tier_label} &nbsp;·&nbsp; Vol≥{flt_vol} &nbsp;·&nbsp; |move|≥{flt_move}%</div>",
        unsafe_allow_html=True)
    cf1,cf2,cf3,cf4,cf5,cf6,cf7 = st.columns(7)
    cf1.metric(f"Trades ({tier_label})", pnl_sel["total_trades"])
    cf2.metric("Precisión",  f"{filt_acc_dyn:.1f}%" if not np.isnan(filt_acc_dyn) else "n/a")
    cf3.metric("Trades/día", f"{pnl_sel['total_trades']/n_days:.1f}")
    cf4.metric(f"P&L (${stake:.0f}/tr)", f"${pnl_sel['total_pnl']:+,.0f}")
    cf5.metric("P&L/día",   f"${pnl_sel['pnl_per_day']:+.1f}")
    cf6.metric("ROI%",      f"{pnl_sel['roi_pct']:+.1f}%")
    cf7.metric("Wins/Losses", f"{pnl_sel['wins']}/{pnl_sel['losses']}")

    # ── Tier breakdown ──────────────────────────────────────────────────────
    st.markdown(
        "<div class='section-hdr'>🏆 Rendimiento por Tier &nbsp;·&nbsp;"
        " S: Vol≥500 &amp; |mov|≥0.3%  A: Vol≥300 &amp; |mov|≥0.2%"
        "  B: Vol≥200 &amp; |mov|≥0.1%  C: Vol≥100  D: resto</div>",
        unsafe_allow_html=True)
    tcols5 = st.columns(5)
    for i, tier in enumerate(["S","A","B","C","D"]):
        sub  = df_main[df_main["tier"] == tier]
        n_t  = len(sub)
        w_n  = int(sub["correct"].sum()) if n_t > 0 else 0
        w_pct = w_n / n_t * 100 if n_t > 0 else 0.0
        pnl  = (w_n - (n_t - w_n)) * stake
        col  = TIER_COLORS_HEX[tier]
        acc_col = "#00d68f" if w_pct >= 55 else ("#ff4757" if w_pct < 45 else "#a0a3b1")
        pnl_col = "#00d68f" if pnl   >= 0  else "#ff4757"
        with tcols5[i]:
            st.markdown(f"""
            <div style='background:#161929;border:2px solid {col};border-radius:10px;
                        padding:14px 8px;text-align:center;'>
              <div style='color:{col};font-size:17px;font-weight:bold;'>Tier {tier}</div>
              <div style='color:{acc_col};font-size:28px;font-weight:bold;'>{w_pct:.1f}%</div>
              <div style='color:#6e7191;font-size:12px;'>{w_n}/{n_t}</div>
              <div style='color:{pnl_col};font-size:13px;font-weight:bold;'>P&L ${pnl:+.0f}</div>
              <div style='color:#6e7191;font-size:11px;'>{n_t/n_days:.1f}/día</div>
            </div>""", unsafe_allow_html=True)

    # ── P&L Simulation Table ────────────────────────────────────────────────
    st.markdown(f"<div class='section-hdr'>💰 Simulación P&L — ${stake:.0f} por trade, apuesta fija</div>",
                unsafe_allow_html=True)
    pnl_rows_list = []
    for mt in ["S","A","B","C","D"]:
        p = simulate_pnl(df_main, stake, mt)
        pnl_rows_list.append({
            "Min Tier": f"≥ {mt}",
            "Trades": p["total_trades"],
            "Acc%": f"{p['accuracy']:.1f}%",
            f"P&L Total": f"${p['total_pnl']:+,.0f}",
            "P&L/Día": f"${p['pnl_per_day']:+.1f}",
            "ROI%": f"{p['roi_pct']:+.1f}%",
            "Wins": p["wins"],
            "Losses": p["losses"],
        })
    st.dataframe(pd.DataFrame(pnl_rows_list), use_container_width=True, hide_index=True)

    # ── Por Volumen Señal ───────────────────────────────────────────────────
    st.markdown("<div class='section-hdr'>📦 Precisión por Volumen de la Vela SEÑAL (BTC)</div>",
                unsafe_allow_html=True)
    by_svol   = s["by_signal_vol"]
    non_zero  = [(vb, rd) for vb, rd in by_svol.iterrows() if int(rd["total"]) > 0]
    if non_zero:
        vcols = st.columns(len(non_zero))
        for i, (vbin, rd) in enumerate(non_zero):
            av  = rd["accuracy"]
            clr = "#00d68f" if av >= 60 else ("#ff4757" if av < 50 else "#a0a3b1")
            with vcols[i]:
                st.markdown(f"""
                <div style='background:#161929;border-radius:8px;padding:10px 6px;text-align:center;'>
                  <div style='color:#6e7191;font-size:10px;font-weight:bold;'>{vbin}</div>
                  <div style='color:{clr};font-size:20px;font-weight:bold;'>{av:.0f}%</div>
                  <div style='color:#6e7191;font-size:10px;'>{int(rd["wins"])}/{int(rd["total"])}</div>
                </div>""", unsafe_allow_html=True)

    # ── Por Hora ────────────────────────────────────────────────────────────
    st.markdown("<div class='section-hdr'>🕐 Precisión por Hora (hora local, apertura vela TARGET)</div>",
                unsafe_allow_html=True)
    by_h = s["by_hour"].reset_index()
    clrs_h = ["#00d68f" if a >= 60 else ("#ff4757" if a < 50 else "#a0a3b1")
               for a in by_h["accuracy"]]
    fig_h = go.Figure(go.Bar(
        x=[f"{int(h):02d}h" for h in by_h["hour"]],
        y=by_h["accuracy"],
        marker_color=clrs_h,
        text=[f"{a:.0f}%" for a in by_h["accuracy"]],
        textposition="outside",
        customdata=by_h["total"],
        hovertemplate="%{x}<br>Acc: %{y:.1f}%<br>n=%{customdata}<extra></extra>",
    ))
    fig_h.add_hline(y=50, line_dash="dash", line_color="#6e7191", line_width=1)
    fig_h.update_layout(
        plot_bgcolor="#0d0f1a", paper_bgcolor="#0d0f1a", font_color="#e8eaf6",
        height=300, showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis=dict(range=[0, 110], gridcolor="#1e2236"),
        xaxis=dict(gridcolor="#1e2236"),
    )
    st.plotly_chart(fig_h, use_container_width=True)

    # ── Por Minuto ──────────────────────────────────────────────────────────
    if s["minute_acc"]:
        st.markdown("<div class='section-hdr'>⏱ Precisión por Minuto intra-vela TARGET</div>",
                    unsafe_allow_html=True)
        mcols = st.columns(5)
        for mn, info in sorted(s["minute_acc"].items()):
            av  = info["accuracy"]
            clr = "#00d68f" if av >= 60 else ("#ff4757" if av < 50 else "#a0a3b1")
            is_best = mn == s["best_minute"]
            bdr = "2px solid #ffd700" if is_best else "1px solid #1e2236"
            with mcols[mn - 1]:
                st.markdown(f"""
                <div style='background:#161929;border:{bdr};border-radius:8px;
                            padding:12px 6px;text-align:center;'>
                  <div style='color:{"#ffd700" if is_best else "#6e7191"};font-size:11px;font-weight:bold;'>
                    {"★ " if is_best else ""}min {mn}</div>
                  <div style='color:{clr};font-size:22px;font-weight:bold;'>{av:.1f}%</div>
                  <div style='color:#6e7191;font-size:10px;'>{info["correct"]}/{info["total"]}</div>
                </div>""", unsafe_allow_html=True)

    # ── Calendario (mes) ────────────────────────────────────────────────────
    if mode_r == "month":
        parts  = result["label"].split("-")
        yr_cal = int(parts[0]); mo_cal = int(parts[1])

        filt_df_cal = (df_main[df_main["tier"].isin(selected_tiers)].copy()
                       if selected_tiers else df_main.iloc[0:0].copy())
        by_date_cal: dict = {}
        for ds, grp in filt_df_cal.groupby("date"):
            w = int(grp["correct"].sum()); tot = len(grp)
            by_date_cal[ds] = {"wins": w, "losses": tot-w, "total": tot,
                                "pnl": (w-(tot-w))*stake}

        if by_date_cal:
            all_pnl_c   = [v["pnl"] for v in by_date_cal.values()]
            total_pnl_c = sum(all_pnl_c)
            days_gain_c = sum(1 for p in all_pnl_c if p > 0)
            days_loss_c = sum(1 for p in all_pnl_c if p < 0)
            best_date_c  = max(by_date_cal, key=lambda d: by_date_cal[d]["pnl"])
            worst_date_c = min(by_date_cal, key=lambda d: by_date_cal[d]["pnl"])
            best_pnl_c   = by_date_cal[best_date_c]["pnl"]
            worst_pnl_c  = by_date_cal[worst_date_c]["pnl"]
        else:
            total_pnl_c = days_gain_c = days_loss_c = 0
            best_date_c = worst_date_c = None
            best_pnl_c = worst_pnl_c = 0.0

        st.markdown(
            f"<div class='section-hdr'>📅 Calendario P&L — "
            f"${stake:.0f}/trade · Tiers: {tier_label}</div>",
            unsafe_allow_html=True)
        cc1,cc2,cc3,cc4,cc5 = st.columns(5)
        cc1.metric("P&L Total Mes",  f"${total_pnl_c:+,.0f}")
        cc2.metric("Días ganancia",  days_gain_c)
        cc3.metric("Días pérdida",   days_loss_c)
        cc4.metric("Mejor día",
                   (f"Día {best_date_c.split('-')[2]} (${best_pnl_c:+,.0f})"
                    if best_date_c else "—"))
        cc5.metric("Peor día",
                   (f"Día {worst_date_c.split('-')[2]} (${worst_pnl_c:+,.0f})"
                    if worst_date_c else "—"))

        # Grid HTML del calendario
        day_names_es = ["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"]
        month_weeks  = cal_lib.monthcalendar(yr_cal, mo_cal)
        cal_html = ("<div style='display:grid;grid-template-columns:repeat(7,1fr);"
                    "gap:6px;margin-top:10px;'>")
        for dn in day_names_es:
            cal_html += (f"<div style='text-align:center;color:#6e7191;"
                         f"font-size:12px;font-weight:bold;padding:4px;'>{dn}</div>")
        for week in month_weeks:
            for day in week:
                if day == 0:
                    cal_html += "<div></div>"
                    continue
                ds   = f"{yr_cal:04d}-{mo_cal:02d}-{day:02d}"
                info = by_date_cal.get(ds)
                bdr  = ("#00d68f" if ds == best_date_c else
                        ("#ff4757" if ds == worst_date_c else "#1e2236"))
                if info:
                    pv  = info["pnl"]
                    pc  = "#00d68f" if pv > 0 else ("#ff4757" if pv < 0 else "#a0a3b1")
                    cal_html += (
                        f"<div style='background:#161929;border:2px solid {bdr};"
                        f"border-radius:8px;padding:8px 4px;text-align:center;min-height:78px;'>"
                        f"<div style='color:#6e7191;font-size:10px;text-align:right;'>{day:02d}</div>"
                        f"<div style='color:{pc};font-size:15px;font-weight:bold;'>${pv:+,.0f}</div>"
                        f"<div style='color:#6e7191;font-size:9px;'>"
                        f"{info['wins']}W/{info['losses']}L·{info['total']}tr</div></div>")
                else:
                    cal_html += (
                        f"<div style='background:#0d0f1a;border:1px solid #1e2236;"
                        f"border-radius:8px;padding:8px 4px;text-align:center;"
                        f"min-height:78px;opacity:0.35;'>"
                        f"<div style='color:#6e7191;font-size:10px;text-align:right;'>{day:02d}</div>"
                        f"<div style='color:#6e7191;font-size:13px;'>—</div></div>")
        cal_html += "</div>"
        st.markdown(cal_html, unsafe_allow_html=True)

        # Selector de día para detalle
        st.markdown("<div class='section-hdr'>🔍 Detalle de Día</div>",
                    unsafe_allow_html=True)
        available_days = sorted(by_date_cal.keys())
        if available_days:
            sel_day = st.selectbox(
                "Seleccionar día:", ["— elige un día —"] + available_days,
                key="day_detail_sel")
            if sel_day != "— elige un día —":
                render_day_detail(df_main, sel_day, selected_tiers, stake)

    # ── Detalle automático (modo día) ────────────────────────────────────────
    if mode_r == "day":
        render_day_detail(df_main, result["label"], selected_tiers, stake)

    # ── Distribución de Señales ─────────────────────────────────────────────
    if s["signal_bias"]:
        st.markdown("<div class='section-hdr'>📡 Distribución de Señales</div>",
                    unsafe_allow_html=True)
        sig_rows = [
            {"Señal": sn, "UP": v["UP"], "DOWN": v["DOWN"],
             "NEUTRAL": v["NEUTRAL"], "Total": v["total"]}
            for sn, v in sorted(s["signal_bias"].items())
        ]
        st.dataframe(pd.DataFrame(sig_rows), use_container_width=True, hide_index=True)

    # ── Predicciones raw ────────────────────────────────────────────────────
    with st.expander("📋 Ver todas las predicciones"):
        show_cols = ["date","time","tier","prediction","actual","correct",
                     "confidence","signal_volume","signal_move","pct_move"]
        st.dataframe(df_main[show_cols], use_container_width=True, hide_index=True)

    # ── Descargas ───────────────────────────────────────────────────────────
    st.markdown("<div class='section-hdr'>⬇ Exportar resultados</div>",
                unsafe_allow_html=True)
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "📂 Descargar Excel Completo (12 hojas)",
            data=save_excel_bytes(result, stake),
            file_name=f"backtest_v2_{result['label']}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "📋 Descargar Excel Detalle (CST)",
            data=save_excel_detail_bytes(result),
            file_name=f"detalle_v2_{result['label']}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
