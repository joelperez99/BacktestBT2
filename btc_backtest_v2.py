#!/usr/bin/env python3
"""
BitPredict Backtester  v2  — Lógica idéntica a btc_live_predictor v2
=====================================================================
DIFERENCIA CLAVE vs backtest original (v1):
  v1: predice al INICIO de cada vela → usa volumen de la vela ANTERIOR
  v2: predice al CIERRE de la vela → usa volumen de la vela RECIÉN CERRADA
      luego evalúa el resultado en la SIGUIENTE vela 5m

  Contexto 1m: últimas 120 velas ANTES del cierre de la vela señal
  Open price to beat: close de la vela señal (= open de la siguiente)
  Tier/Filtro: calculados con vol+|move| de la vela señal cerrada

Esto replica exactamente lo que hace el live en producción.

Uso:  pip install requests pandas numpy openpyxl
      python btc_backtest_v2.py
"""

import tkinter as tk
from tkinter import ttk
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar
import threading
import os
import subprocess
import random
import string

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOL     = "BTCUSDT"
BINANCE    = "https://api.binance.com/api/v3"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

FILTERS = {
    "min_volume":     200,    # BTC mínimo en la vela señal (cerrada)
    "min_abs_move":   0.1,    # |pct_move| mínimo % de la vela señal
    "allowed_hours":  None,   # None = todas; ej: [1, 15, 17, 21]
    "min_confidence": None,   # None = ignorar confianza (recomendado)
}

TIER_DEF = {
    "S": {"min_volume": 500, "min_abs_move": 0.3},
    "A": {"min_volume": 300, "min_abs_move": 0.2},
    "B": {"min_volume": 200, "min_abs_move": 0.1},
    "C": {"min_volume": 100, "min_abs_move": 0.0},
}

TIER_ORDER = {"S": 0, "A": 1, "B": 2, "C": 3, "D": 4}
BET_SIZE   = 10.0

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = "#0d0f1a"
CARD    = "#161929"
CARD2   = "#1e2236"
UP      = "#00d68f"
DOWN    = "#ff4757"
NEUTRAL = "#a0a3b1"
TEXT    = "#e8eaf6"
TEXT2   = "#6e7191"
ACCENT  = "#7c6fff"
ORANGE  = "#f7931a"
WHITE   = "#ffffff"
YELLOW  = "#ffd700"
TIER_COLORS = {"S": "#ffd700", "A": "#00d68f", "B": "#7c6fff",
               "C": "#f7931a", "D": "#ff4757"}


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
# INDICADORES  (idénticos al live v2 — no modificar)
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
    mid = s.rolling(n).mean();  std = s.rolling(n).std()
    lo  = (mid - 2*std).iloc[-1];  hi = (mid + 2*std).iloc[-1]
    return float((s.iloc[-1] - lo) / ((hi - lo) + 1e-9))

def _mom(s, n=5):
    return float((s.iloc[-1] - s.iloc[-(n+1)]) / s.iloc[-(n+1)] * 100)

def _vol_ratio(v, recent=5, hist=20):
    return float(v.iloc[-recent:].mean() / (v.iloc[-(hist+recent):-recent].mean() + 1e-9))


def predict_from_df(df1m: pd.DataFrame):
    """
    Predice dirección usando los últimos ~120 minutos de velas 1m.
    Retorna: (direction, confidence, up_pct, dn_pct, votes_list)
    """
    c, v = df1m["close"], df1m["volume"]
    votes = []

    r_val = _rsi(c)
    if   r_val < 42: votes.append(("RSI",       "UP",      min((42 - r_val) / 42, 1)))
    elif r_val > 58: votes.append(("RSI",       "DOWN",    min((r_val - 58) / 42, 1)))
    else:            votes.append(("RSI",       "NEUTRAL", 0.0))

    mh = _macd_hist(c)
    votes.append(("MACD", "UP" if mh > 0 else "DOWN", min(abs(mh) / 30, 1)))

    e9  = float(_ema(c, 9).iloc[-1]);  e21 = float(_ema(c, 21).iloc[-1])
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
    up_pct = up_s / total * 100;  dn_pct = dn_s / total * 100
    direction = "UP" if up_s >= dn_s else "DOWN"
    return direction, max(up_pct, dn_pct), up_pct, dn_pct, votes


# ══════════════════════════════════════════════════════════════════════════════
# TIER
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


def passes_filter(volume: float, abs_move: float, hour: int, confidence: float) -> bool:
    if FILTERS["min_volume"]     and volume     < FILTERS["min_volume"]:     return False
    if FILTERS["min_abs_move"]   and abs_move   < FILTERS["min_abs_move"]:   return False
    if FILTERS["allowed_hours"]  and hour not in FILTERS["allowed_hours"]:   return False
    if FILTERS["min_confidence"] and confidence < FILTERS["min_confidence"]: return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# PROCESAMIENTO DE VELAS  (lógica v2 — idéntica al live)
# ══════════════════════════════════════════════════════════════════════════════

def _process_candles(df5m_range: pd.DataFrame, df1m_full: pd.DataFrame) -> list:
    """
    Para cada vela señal c5[i]:
      - Contexto 1m: velas ANTES del CIERRE de c5[i]  (= lo que tiene el live)
      - Tier/Filtro: vol + |move| de c5[i]
      - Open price to beat: close de c5[i]
      - Resultado: cierre de c5[i+1] vs close de c5[i]
    """
    results = []
    n = len(df5m_range)

    for i in range(n - 1):          # necesitamos i+1 → hasta penúltima vela
        c5      = df5m_range.iloc[i]
        c5_next = df5m_range.iloc[i + 1]

        # ── Contexto 1m: hasta el cierre de la vela señal ─────────────────
        ctx = df1m_full[df1m_full["open_time"] < c5["close_time"]].tail(120)
        if len(ctx) < 50:
            continue

        direction, conf, up_pct, dn_pct, votes = predict_from_df(ctx)

        # ── Métricas de la vela SEÑAL (cerrada) ───────────────────────────
        sig_vol   = float(c5["volume"])
        sig_move  = abs((float(c5["close"]) - float(c5["open"])) / float(c5["open"]) * 100)

        # ── Open price = close de la vela señal = open real de la siguiente ─
        open_price  = float(c5["close"])
        close_price = float(c5_next["close"])

        actual   = "UP" if close_price >= open_price else "DOWN"
        correct  = direction == actual
        pct_move = (close_price - open_price) / open_price * 100

        tier    = get_trade_tier(sig_vol, sig_move)
        c_open  = c5["close_time"]           # timestamp de referencia = cierre vela señal
        hour    = c_open.to_pydatetime().astimezone().hour
        in_flt  = passes_filter(sig_vol, sig_move, hour, conf)

        # ── Minutos intra-vela TARGET ──────────────────────────────────────
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
            "signal_volume":   sig_vol,     # vol de la vela señal (para análisis)
            "signal_move":     sig_move,    # |move| de la vela señal
            "tier":            tier,
            "in_filter":       in_flt,
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# P&L SIMULACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def simulate_pnl(df: pd.DataFrame, bet_size: float = BET_SIZE,
                 min_tier: str = "B") -> dict:
    """P&L acumulado para todos los tiers ≤ min_tier."""
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
        total_trades=n,
        wins=wins,
        losses=losses,
        accuracy=round(wins / n * 100, 2),
        total_pnl=round(total_pnl, 2),
        pnl_per_day=round(total_pnl / days, 2),
        roi_pct=round(total_pnl / (n * bet_size) * 100, 2),
        days=days,
    )


def filtered_pnl(df: pd.DataFrame, tiers: list, bet_size: float = BET_SIZE) -> dict:
    """P&L para un conjunto exacto de tiers seleccionados."""
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
        total_trades=n,
        wins=wins,
        losses=losses,
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
        max_ws = max(max_ws, cw);  max_ls = max(max_ls, cl)

    pct_desc  = df["pct_move"].describe()
    conf_win  = df[df["correct"]]["confidence"].mean()  if wins   else float("nan")
    conf_loss = df[~df["correct"]]["confidence"].mean() if losses else float("nan")

    bhs = by_hour[by_hour["total"] >= 3].sort_values("accuracy", ascending=False)
    best_hour  = int(bhs.index[0])  if len(bhs) else None
    worst_hour = int(bhs.index[-1]) if len(bhs) else None

    # Volumen de señal (vela cerrada usada para tier/filtro)
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

    # Volumen de señal por bins
    vol_bins = [0, 50, 100, 200, 300, 500, 1000, float("inf")]
    vol_labs = ["0-50","50-100","100-200","200-300","300-500","500-1000","1000+"]
    df2["sig_vol_bin"] = pd.cut(df2["signal_volume"], bins=vol_bins, labels=vol_labs, right=False)
    by_signal_vol = df2.groupby("sig_vol_bin", observed=False)["correct"].agg(["sum","count"]).rename(
        columns={"sum":"wins","count":"total"})
    by_signal_vol["accuracy"] = by_signal_vol.apply(
        lambda r: r["wins"]/r["total"]*100 if r["total"]>0 else float("nan"), axis=1)

    # Volumen de vela target por bins (para comparación con v1)
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
        by_day=by_day,
        by_tier=by_tier,
        pnl_table=pnl_table,
        by_signal_vol=by_signal_vol,
        by_target_vol=by_target_vol,
        filt_acc=filt_acc,
        filt_total=filt_total,
        filt_per_day=filt_per_day,
    )


# ══════════════════════════════════════════════════════════════════════════════
# EXCEL EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def _unique_path(filepath: str) -> str:
    if not os.path.exists(filepath):
        return filepath
    base, ext = os.path.splitext(filepath)
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"{base}-{suffix}{ext}"


def save_excel(result: dict) -> str:
    label    = result["label"].replace("-", "_")
    filepath = _unique_path(os.path.join(OUTPUT_DIR, f"backtest_v2_{label}.xlsx"))

    df = result["df"].copy()
    s  = result["stats"]

    pred_cols = ["date","time","hour","tier","in_filter",
                 "prediction","actual","correct","confidence",
                 "pct_move","abs_move","signal_volume","signal_move",
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
        ("── Filtros activos ──",    ""),
        ("Filtro min_volume (vela señal)", FILTERS["min_volume"]),
        ("Filtro min_abs_move",      FILTERS["min_abs_move"]),
        ("Filtro allowed_hours",     str(FILTERS["allowed_hours"])),
        ("── Stats filtradas (S+A+B) ──", ""),
        ("Trades filtrados",         s["filt_total"]),
        ("Trades filtrados / día",   round(s["filt_per_day"], 1)),
        ("Precisión filtrada %",     round(s["filt_acc"], 2) if not np.isnan(s["filt_acc"]) else ""),
        ("── Métricas generales ──", ""),
        ("Racha max aciertos",       s["max_win_streak"]),
        ("Racha max fallos",         s["max_loss_streak"]),
        ("Conf. media aciertos",     round(s["conf_win"],  2) if not np.isnan(s["conf_win"])  else ""),
        ("Conf. media fallos",       round(s["conf_loss"], 2) if not np.isnan(s["conf_loss"]) else ""),
        ("Mejor hora",               f"{s['best_hour']:02d}h"  if s["best_hour"]  is not None else ""),
        ("Peor hora",                f"{s['worst_hour']:02d}h" if s["worst_hour"] is not None else ""),
        ("Vol. señal prom. aciertos", round(s["vol_win"],  0)  if not np.isnan(s["vol_win"])  else ""),
        ("Vol. señal prom. fallos",   round(s["vol_loss"], 0)  if not np.isnan(s["vol_loss"]) else ""),
        ("Pct_move min",             round(s["pct_desc"]["min"],  4)),
        ("Pct_move media",           round(s["pct_desc"]["mean"], 4)),
        ("Pct_move max",             round(s["pct_desc"]["max"],  4)),
        ("Pct_move std",             round(s["pct_desc"]["std"],  4)),
    ]
    df_resumen = pd.DataFrame(resumen_rows, columns=["Métrica", "Valor"])

    min_rows = [
        {"Minuto": mn,
         "Aciertos":   info.get("correct", 0),
         "Total":      info.get("total", 0),
         "Precisión%": round(info.get("accuracy", float("nan")), 2),
         "★":          "★" if mn == s["best_minute"] else ""}
        for mn, info in sorted(s["minute_acc"].items())
    ]
    df_min = pd.DataFrame(min_rows)

    df_sigs = pd.DataFrame([
        {"Señal": sn, "UP": v["UP"], "DOWN": v["DOWN"],
         "NEUTRAL": v["NEUTRAL"], "Total": v["total"]}
        for sn, v in s["signal_bias"].items()
    ])

    df_tier = s["by_tier"].copy().reset_index()
    df_tier.columns = ["Tier", "Aciertos", "Total", "Precisión%"]
    tier_pnl_exact = {}
    for t in ["S","A","B","C","D"]:
        sub = df[df["tier"] == t]
        n   = len(sub)
        if n:
            w = int(sub["correct"].sum())
            tier_pnl_exact[t] = {
                "P&L($10/trade)": round((w - (n-w)) * BET_SIZE, 2),
                "ROI%":           round((w - (n-w)) * BET_SIZE / (n * BET_SIZE) * 100, 2),
                "Trades/día":     round(n / n_days, 1),
            }
        else:
            tier_pnl_exact[t] = {"P&L($10/trade)": 0, "ROI%": 0, "Trades/día": 0}
    for _, row in df_tier.iterrows():
        t = row["Tier"]
        if t in tier_pnl_exact:
            for k, v in tier_pnl_exact[t].items():
                df_tier.loc[df_tier["Tier"] == t, k] = v

    pnl_df = s["pnl_table"].reset_index()
    pnl_df.columns = ["Min Tier","Trades","Wins","Losses",
                       "Accuracy%","P&L Total($)","P&L/Día($)","ROI%","Días"]

    df_sigvol = s["by_signal_vol"].reset_index()
    df_sigvol.columns = ["Vol Vela Señal (BTC)","Aciertos","Total","Precisión%"]

    df_tgtvol = s["by_target_vol"].reset_index()
    df_tgtvol.columns = ["Vol Vela Target (BTC)","Aciertos","Total","Precisión%"]

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df_resumen.to_excel(writer,               sheet_name="Resumen",           index=False)
        df[pred_cols].to_excel(writer,            sheet_name="Predicciones",      index=False)
        s["by_hour"].reset_index().to_excel(writer, sheet_name="Por Hora",        index=False)
        s["by_conf"].reset_index().to_excel(writer, sheet_name="Por Confianza",   index=False)
        df_min.to_excel(writer,                   sheet_name="Por Minuto",        index=False)
        s["by_pred"].reset_index().to_excel(writer, sheet_name="Por Direccion",   index=False)
        df_sigs.to_excel(writer,                  sheet_name="Señales",           index=False)
        if s["by_day"] is not None:
            s["by_day"].reset_index().to_excel(writer, sheet_name="Por Dia",      index=False)
        df_tier.to_excel(writer,                  sheet_name="Por Tier",          index=False)
        pnl_df.to_excel(writer,                   sheet_name="P&L Simulado",      index=False)
        df_sigvol.to_excel(writer,                sheet_name="Vol Vela Señal",    index=False)
        df_tgtvol.to_excel(writer,                sheet_name="Vol Vela Target",   index=False)

    return filepath


# ══════════════════════════════════════════════════════════════════════════════
# EXCEL DETALLE
# ══════════════════════════════════════════════════════════════════════════════

_CST = timedelta(hours=-6)

def _to_cst(utc_str: str) -> str:
    from datetime import timezone
    dt  = datetime.strptime(utc_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    cst = dt + _CST
    return cst.strftime("%Y-%m-%d %H:%M:%S")


def save_excel_detail(result: dict) -> str:
    label    = result["label"].replace("-", "_")
    filepath = _unique_path(os.path.join(OUTPUT_DIR, f"detalle_v2_{label}.xlsx"))

    df = result["df"].copy()
    detail_rows = []
    for _, r in df.iterrows():
        utc_str = r.get("timestamp_utc", f"{r['date']} {r['time']}:00")
        cst_str = _to_cst(utc_str)
        detail_rows.append({
            "Timestamp CST":        cst_str,
            "Hora Local":           r.get("timestamp_local", f"{r['date']} {r['time']}:00"),
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

    detail_df = pd.DataFrame(detail_rows)

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        detail_df.to_excel(writer, sheet_name="Detalle", index=False)
        ws = writer.sheets["Detalle"]
        col_widths = {
            "A": 22, "B": 22, "C": 16, "D": 12, "E": 12,
            "F": 10, "G": 10, "H": 14, "I": 14, "J":  8,
            "K": 10, "L": 12, "M": 14, "N": 10, "O": 12,
            "P": 80,
        }
        for col_letter, width in col_widths.items():
            ws.column_dimensions[col_letter].width = width

    return filepath


# ══════════════════════════════════════════════════════════════════════════════
# MOTORES
# ══════════════════════════════════════════════════════════════════════════════

_CST_SHIFT = pd.Timedelta(hours=6)

def run_backtest(date_str: str) -> dict:
    day_start    = pd.Timestamp(date_str, tz="UTC") + _CST_SHIFT
    day_end      = day_start + pd.Timedelta(days=1)
    # Warmup: 2h antes + 1 vela 5m extra al final para tener la "next candle"
    warmup_start = day_start - pd.Timedelta(hours=2)
    fetch_end    = day_end   + pd.Timedelta(minutes=10)
    start_ms = int(warmup_start.value // 1_000_000)
    end_ms   = int(fetch_end.value    // 1_000_000)

    df1m = fetch_klines_range("1m", start_ms, end_ms)
    df5m = fetch_klines_range("5m", start_ms, end_ms)
    if df1m.empty or df5m.empty:
        raise ValueError("No se obtuvieron datos para esa fecha.")

    # Velas señal = las del día; velas target pueden extenderse un poco
    df5m_day = df5m[(df5m["open_time"] >= day_start) &
                    (df5m["open_time"] <  day_end)].reset_index(drop=True)

    # Aseguramos que haya una vela target para la última vela señal del día
    last_signal_close = df5m_day.iloc[-1]["close_time"] if not df5m_day.empty else day_end
    df5m_window = df5m[(df5m["open_time"] >= day_start) &
                       (df5m["open_time"] <= last_signal_close)].reset_index(drop=True)

    records = _process_candles(df5m_window, df1m)
    if not records:
        raise ValueError("No hay velas 5m procesables para ese día.")

    df = pd.DataFrame(records)
    df = df[df["timestamp_utc"].apply(lambda u: _to_cst(u)[:10] == date_str)].reset_index(drop=True)
    if df.empty:
        raise ValueError("No hay velas 5m en fecha CST para ese día.")

    stats = _compute_stats(df)
    return {"records": df.to_dict("records"), "df": df, "stats": stats,
            "label": date_str, "mode": "day"}


def run_backtest_month(year_str: str, month_str: str, progress_cb=None) -> dict:
    year  = int(year_str);  month = int(month_str)
    month_start  = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    last_day     = calendar.monthrange(year, month)[1]
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
        # Incluir una vela extra al final como posible target
        df5m_day = df5m_month[(df5m_month["open_time"] >= day_ts) &
                               (df5m_month["open_time"] <  day_end_ts + pd.Timedelta(minutes=5))
                               ].reset_index(drop=True)
        if not df5m_day.empty:
            all_records.extend(_process_candles(df5m_day, df1m))
        if progress_cb:
            progress_cb(idx + 1, last_day)

    if not all_records:
        raise ValueError("No hay velas 5m procesables para ese mes.")

    df    = pd.DataFrame(all_records)
    stats = _compute_stats(df)
    return {"records": all_records, "df": df, "stats": stats,
            "label": f"{year_str}-{month_str}", "mode": "month"}


def run_backtest_year(year_str: str, progress_cb=None) -> dict:
    year         = int(year_str)
    year_start   = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
    year_end     = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")
    warmup_start = year_start - pd.Timedelta(hours=2)
    fetch_end    = year_end   + pd.Timedelta(minutes=10)
    start_ms     = int(warmup_start.value // 1_000_000)
    end_ms       = int(fetch_end.value    // 1_000_000)

    if progress_cb:
        progress_cb(0, 12)
    df1m = fetch_klines_range("1m", start_ms, end_ms)
    df5m = fetch_klines_range("5m", start_ms, end_ms)
    if df1m.empty or df5m.empty:
        raise ValueError("No se obtuvieron datos para ese año.")

    all_records = []
    for month in range(1, 13):
        month_start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
        last_day    = calendar.monthrange(year, month)[1]
        month_end_t = month_start + pd.Timedelta(days=last_day)

        df5m_month = df5m[
            (df5m["open_time"] >= month_start) &
            (df5m["open_time"] <  month_end_t + pd.Timedelta(minutes=10))
        ].reset_index(drop=True)

        for idx in range(last_day):
            day_ts     = month_start + pd.Timedelta(days=idx)
            day_end_ts = day_ts + pd.Timedelta(days=1)
            df5m_day   = df5m_month[
                (df5m_month["open_time"] >= day_ts) &
                (df5m_month["open_time"] <  day_end_ts + pd.Timedelta(minutes=5))
            ].reset_index(drop=True)
            if not df5m_day.empty:
                all_records.extend(_process_candles(df5m_day, df1m))

        if progress_cb:
            progress_cb(month, 12)

    if not all_records:
        raise ValueError("No hay velas procesables para ese año.")

    df    = pd.DataFrame(all_records)
    stats = _compute_stats(df)
    return {"records": all_records, "df": df, "stats": stats,
            "label": year_str, "mode": "year"}


# ══════════════════════════════════════════════════════════════════════════════
# FILTRO LÍMITE DIARIO
# ══════════════════════════════════════════════════════════════════════════════

def find_daily_limit_hit(day_df: pd.DataFrame, stake: float, limit: float) -> dict:
    """
    Simula el P&L acumulado trade a trade en un día.
    Retorna qué límite (+ o -) se alcanzó primero, en qué operación y a qué hora.
    """
    running = 0.0
    for i, (_, row) in enumerate(day_df.iterrows()):
        running += stake if row["correct"] else -stake
        if running >= limit:
            return {"hit": "UP",   "amount": running, "op": i + 1, "time": row["time"]}
        if running <= -limit:
            return {"hit": "DOWN", "amount": running, "op": i + 1, "time": row["time"]}
    return {"hit": None, "amount": running, "op": None, "time": None}


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _lbl(parent, text, fg=None, font=None, **kw):
    return tk.Label(parent, text=text, bg=parent["bg"],
                    fg=fg or TEXT, font=font or ("Segoe UI", 9), **kw)

def _card_row(parent, items, font_val=("Segoe UI", 16, "bold"), pady_bottom=8):
    row = tk.Frame(parent, bg=BG)
    row.pack(fill="x", pady=(0, pady_bottom))
    for i, (label, val, color) in enumerate(items):
        row.columnconfigure(i, weight=1)
        card = tk.Frame(row, bg=CARD)
        card.grid(row=0, column=i, padx=5, pady=2, sticky="nsew")
        _lbl(card, label, fg=TEXT2, font=("Segoe UI", 8, "bold")).pack(pady=(8, 2))
        _lbl(card, str(val), fg=color, font=font_val).pack(pady=(0, 8))


# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════

class BacktestApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("BitPredict  —  Backtester v2  (lógica live)")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.state("zoomed")
        self._excel_path  = None
        self._detail_path = None
        self._last_result = None
        self._build_ui()

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=20, pady=(14, 6))
        cv = tk.Canvas(hdr, width=38, height=38, bg=BG, highlightthickness=0)
        cv.pack(side="left")
        cv.create_oval(1, 1, 37, 37, fill=ORANGE, outline="")
        cv.create_text(19, 19, text="₿", fill=WHITE, font=("Segoe UI", 16, "bold"))
        tf = tk.Frame(hdr, bg=BG)
        tf.pack(side="left", padx=10)
        _lbl(tf, "Backtester v2  — Lógica Live (predice al cierre → evalúa siguiente vela)",
             fg=TEXT, font=("Segoe UI", 14, "bold")).pack(anchor="w")
        _lbl(tf, "Tier = vol+|move| de la vela SEÑAL (cerrada)  •  contexto 1m hasta cierre señal",
             fg=TEXT2, font=("Segoe UI", 9)).pack(anchor="w")

        # Control panel
        ctrl = tk.Frame(self, bg=CARD)
        ctrl.pack(fill="x", padx=20, pady=8)
        row = tk.Frame(ctrl, bg=CARD)
        row.pack(padx=18, pady=12, anchor="w")

        # Modo
        self._mode = tk.StringVar(value="day")
        mf = tk.Frame(row, bg=CARD)
        mf.pack(side="left", padx=(0, 14))
        _lbl(mf, "Modo:", fg=TEXT2, font=("Segoe UI", 10, "bold")).pack(side="left", padx=(0,6))
        for txt, val in [("Por día", "day"), ("Por mes", "month"), ("Por año", "year")]:
            tk.Radiobutton(mf, text=txt, variable=self._mode, value=val,
                           command=self._on_mode_change,
                           bg=CARD, fg=TEXT, selectcolor=CARD2,
                           activebackground=CARD, activeforeground=WHITE,
                           font=("Segoe UI", 10, "bold"),
                           relief="flat", cursor="hand2").pack(side="left", padx=3)

        tk.Frame(row, bg=CARD2, width=1).pack(side="left", fill="y", padx=10)

        # Fecha
        df_ = tk.Frame(row, bg=CARD)
        df_.pack(side="left")
        _lbl(df_, "Fecha:", fg=TEXT2, font=("Segoe UI", 10, "bold")).pack(side="left", padx=(0,6))
        today  = datetime.now()
        years  = [str(y) for y in range(today.year - 2, today.year + 1)]
        months = [f"{m:02d}" for m in range(1, 13)]
        days   = [f"{d:02d}" for d in range(1, 32)]
        self._year  = tk.StringVar(value=str(today.year))
        self._month = tk.StringVar(value=f"{today.month:02d}")
        self._day   = tk.StringVar(value=f"{(today - timedelta(days=1)).day:02d}")
        ttk.Combobox(df_, textvariable=self._year,  values=years,  width=6, state="readonly").pack(side="left", padx=2)
        self._sep_mon = _lbl(df_, "-", fg=TEXT2, font=("Segoe UI", 12))
        self._sep_mon.pack(side="left")
        self._cb_mon = ttk.Combobox(df_, textvariable=self._month, values=months, width=4, state="readonly")
        self._cb_mon.pack(side="left", padx=2)
        self._sep_day = _lbl(df_, "-", fg=TEXT2, font=("Segoe UI", 12))
        self._sep_day.pack(side="left")
        self._cb_day = ttk.Combobox(df_, textvariable=self._day, values=days, width=4, state="readonly")
        self._cb_day.pack(side="left", padx=2)

        tk.Frame(row, bg=CARD2, width=1).pack(side="left", fill="y", padx=10)

        # Filtros rápidos
        ff = tk.Frame(row, bg=CARD)
        ff.pack(side="left")
        _lbl(ff, "Vol señal≥", fg=TEXT2, font=("Segoe UI", 9)).pack(side="left")
        self._flt_vol = tk.StringVar(value=str(FILTERS["min_volume"]))
        tk.Entry(ff, textvariable=self._flt_vol, width=5,
                 bg=CARD2, fg=TEXT, insertbackground=TEXT,
                 relief="flat", font=("Segoe UI", 9)).pack(side="left", padx=2)
        _lbl(ff, " |move|≥", fg=TEXT2, font=("Segoe UI", 9)).pack(side="left")
        self._flt_move = tk.StringVar(value=str(FILTERS["min_abs_move"]))
        tk.Entry(ff, textvariable=self._flt_move, width=5,
                 bg=CARD2, fg=TEXT, insertbackground=TEXT,
                 relief="flat", font=("Segoe UI", 9)).pack(side="left", padx=2)

        tk.Frame(row, bg=CARD2, width=1).pack(side="left", fill="y", padx=10)

        # Stake
        sf = tk.Frame(row, bg=CARD)
        sf.pack(side="left")
        _lbl(sf, "Stake $", fg=TEXT2, font=("Segoe UI", 9)).pack(side="left")
        self._stake_var = tk.StringVar(value=str(int(BET_SIZE)))
        tk.Entry(sf, textvariable=self._stake_var, width=6,
                 bg=CARD2, fg=YELLOW, insertbackground=TEXT,
                 relief="flat", font=("Segoe UI", 9, "bold")).pack(side="left", padx=2)
        self._sim_btn = tk.Button(sf, text="↻ Simular",
                                   command=self._re_simulate,
                                   bg=CARD2, fg=YELLOW, relief="flat", cursor="hand2",
                                   font=("Segoe UI", 9, "bold"), padx=6, pady=3,
                                   state="disabled")
        self._sim_btn.pack(side="left", padx=2)

        # Botones
        self._run_btn = tk.Button(row, text="▶  Ejecutar",
                                   command=self._run, bg=ACCENT, fg=WHITE,
                                   relief="flat", cursor="hand2",
                                   font=("Segoe UI", 11, "bold"), padx=12, pady=5)
        self._run_btn.pack(side="left", padx=(12, 0))

        self._excel_btn = tk.Button(row, text="📂  Excel",
                                     command=self._open_excel,
                                     bg=CARD2, fg=TEXT2, relief="flat", cursor="hand2",
                                     font=("Segoe UI", 10), padx=8, pady=5, state="disabled")
        self._excel_btn.pack(side="left", padx=4)

        self._detail_btn = tk.Button(row, text="📋  Detalle",
                                      command=self._open_detail_excel,
                                      bg=CARD2, fg=TEXT2, relief="flat", cursor="hand2",
                                      font=("Segoe UI", 10), padx=8, pady=5, state="disabled")
        self._detail_btn.pack(side="left", padx=2)

        self._status_lbl = _lbl(row, "", fg=TEXT2, font=("Segoe UI", 9))
        self._status_lbl.pack(side="left", padx=8)

        # Segunda fila: selector de tiers
        row2 = tk.Frame(ctrl, bg=CARD)
        row2.pack(padx=18, pady=(0, 10), anchor="w")
        _lbl(row2, "Tiers:", fg=TEXT2, font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 8))
        self._tier_vars = {}
        for tier in ["S", "A", "B", "C", "D"]:
            var = tk.BooleanVar(value=tier in ("S", "A", "B"))
            self._tier_vars[tier] = var
            color = TIER_COLORS[tier]
            tk.Checkbutton(
                row2, text=f"  {tier}  ", variable=var,
                bg=CARD, fg=color, selectcolor=CARD2,
                activebackground=CARD, activeforeground=color,
                font=("Segoe UI", 10, "bold"),
                relief="flat", cursor="hand2",
            ).pack(side="left", padx=2)
        tk.Frame(row2, bg=CARD2, width=1).pack(side="left", fill="y", padx=8)
        # Botones de selección rápida
        for lbl_txt, tiers_set in [("S+A+B", ("S","A","B")), ("Todos", ("S","A","B","C","D")), ("Limpiar", ())]:
            t = tiers_set
            tk.Button(
                row2, text=lbl_txt,
                command=lambda ts=t: self._set_tiers(ts),
                bg=CARD2, fg=TEXT2, relief="flat", cursor="hand2",
                font=("Segoe UI", 8), padx=6, pady=2,
            ).pack(side="left", padx=2)

        tk.Frame(row2, bg=CARD2, width=1).pack(side="left", fill="y", padx=8)

        # Filtro límite diario
        lf = tk.Frame(row2, bg=CARD)
        lf.pack(side="left")
        self._use_limit = tk.BooleanVar(value=False)
        tk.Checkbutton(
            lf, text="Filtro ±$", variable=self._use_limit,
            command=self._re_simulate,
            bg=CARD, fg=YELLOW, selectcolor=CARD2,
            activebackground=CARD, activeforeground=YELLOW,
            font=("Segoe UI", 9, "bold"), relief="flat", cursor="hand2",
        ).pack(side="left", padx=(0, 4))
        self._limit_var = tk.StringVar(value="500")
        tk.Entry(lf, textvariable=self._limit_var, width=7,
                 bg=CARD2, fg=YELLOW, insertbackground=TEXT,
                 relief="flat", font=("Segoe UI", 9, "bold")).pack(side="left", padx=2)

        # Scroll area
        outer = tk.Frame(self, bg=BG)
        outer.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        self._canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        self._canvas.pack(side="left", fill="both", expand=True)
        vsb = tk.Scrollbar(outer, orient="vertical", command=self._canvas.yview)
        vsb.pack(side="right", fill="y")
        self._canvas.configure(yscrollcommand=vsb.set)
        self._sf = tk.Frame(self._canvas, bg=BG)
        self._sf_id = self._canvas.create_window((0, 0), window=self._sf, anchor="nw")
        self._sf.bind("<Configure>", lambda _e: self._canvas.configure(
            scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>", lambda e: self._canvas.itemconfig(
            self._sf_id, width=e.width))
        self.bind("<MouseWheel>", lambda e: self._canvas.yview_scroll(
            int(-1 * (e.delta / 120)), "units"))

    def _get_selected_tiers(self) -> list:
        return [t for t, v in self._tier_vars.items() if v.get()]

    def _set_tiers(self, tiers: tuple):
        for t, v in self._tier_vars.items():
            v.set(t in tiers)

    def _on_mode_change(self):
        mode = self._mode.get()
        if mode == "year":
            self._sep_day.pack_forget();   self._cb_day.pack_forget()
            self._sep_mon.pack_forget();   self._cb_mon.pack_forget()
        elif mode == "month":
            self._sep_day.pack_forget();   self._cb_day.pack_forget()
            self._sep_mon.pack(side="left")
            self._cb_mon.pack(side="left", padx=2)
        else:
            self._sep_mon.pack(side="left")
            self._cb_mon.pack(side="left", padx=2)
            self._sep_day.pack(side="left")
            self._cb_day.pack(side="left", padx=2)

    def _open_excel(self):
        if self._excel_path and os.path.exists(self._excel_path):
            subprocess.Popen(["start", "", self._excel_path], shell=True)
            return
        if not self._last_result:
            return
        self._excel_btn.config(state="disabled", fg=ORANGE)
        self._status_lbl.config(text="Guardando Excel…", fg=ORANGE)
        result = self._last_result

        def save_and_open():
            try:
                path = save_excel(result)
                self.after(0, lambda: self._on_excel_saved(path))
            except Exception as ex:
                self.after(0, lambda: self._status_lbl.config(
                    text=f"Excel error: {ex}", fg=DOWN))
                self.after(0, lambda: self._excel_btn.config(state="normal", fg=TEXT))

        threading.Thread(target=save_and_open, daemon=True).start()

    def _on_excel_saved(self, path):
        self._excel_path = path
        self._excel_btn.config(state="normal", fg=UP)
        self._status_lbl.config(
            text=f"Excel: {os.path.basename(path)}  •  12 hojas guardadas", fg=UP)
        subprocess.Popen(["start", "", path], shell=True)

    def _open_detail_excel(self):
        if self._detail_path and os.path.exists(self._detail_path):
            subprocess.Popen(["start", "", self._detail_path], shell=True)
            return
        if not self._last_result:
            return
        self._detail_btn.config(state="disabled", fg=ORANGE)
        self._status_lbl.config(text="Generando Detalle…", fg=ORANGE)
        result = self._last_result

        def save_and_open():
            try:
                path = save_excel_detail(result)
                self.after(0, lambda: self._on_detail_saved(path))
            except Exception as ex:
                self.after(0, lambda: self._status_lbl.config(
                    text=f"Detalle error: {ex}", fg=DOWN))
                self.after(0, lambda: self._detail_btn.config(state="normal", fg=TEXT))

        threading.Thread(target=save_and_open, daemon=True).start()

    def _on_detail_saved(self, path):
        self._detail_path = path
        self._detail_btn.config(state="normal", fg=UP)
        self._status_lbl.config(
            text=f"Detalle: {os.path.basename(path)}  •  {len(self._last_result['df'])} pronósticos", fg=UP)
        subprocess.Popen(["start", "", path], shell=True)

    def _run(self):
        try:
            FILTERS["min_volume"]   = float(self._flt_vol.get())
            FILTERS["min_abs_move"] = float(self._flt_move.get())
        except ValueError:
            pass

        self._last_result = None
        self._excel_path  = None
        self._detail_path = None
        for w in self._sf.winfo_children():
            w.destroy()
        self._run_btn.config(state="disabled")
        self._excel_btn.config(state="disabled", fg=TEXT2)
        self._detail_btn.config(state="disabled", fg=TEXT2)
        self._sim_btn.config(state="disabled")
        self._status_lbl.config(text="Descargando datos…", fg=ORANGE)

        mode = self._mode.get()
        if mode == "day":
            date_str = f"{self._year.get()}-{self._month.get()}-{self._day.get()}"
            def worker():
                try:
                    self.after(0, lambda: self._on_result(run_backtest(date_str)))
                except Exception as ex:
                    self.after(0, lambda: self._on_error(str(ex)))

        elif mode == "month":
            ys = self._year.get();  ms = self._month.get()
            def progress_cb_m(done, total):
                self.after(0, lambda: self._status_lbl.config(
                    text=f"Procesando día {done}/{total}…", fg=ORANGE))
            def worker():
                try:
                    self.after(0, lambda: self._on_result(
                        run_backtest_month(ys, ms, progress_cb_m)))
                except Exception as ex:
                    self.after(0, lambda: self._on_error(str(ex)))

        else:  # year
            ys = self._year.get()
            def progress_cb_y(done, total):
                mnames = ["","Ene","Feb","Mar","Abr","May","Jun",
                          "Jul","Ago","Sep","Oct","Nov","Dic"]
                self.after(0, lambda: self._status_lbl.config(
                    text=f"Procesando {mnames[done] if done <= 12 else 'mes'} {done}/{total}…",
                    fg=ORANGE))
            def worker():
                try:
                    self.after(0, lambda: self._on_result(
                        run_backtest_year(ys, progress_cb_y)))
                except Exception as ex:
                    self.after(0, lambda: self._on_error(str(ex)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_error(self, msg):
        self._status_lbl.config(text=f"Error: {msg}", fg=DOWN)
        self._run_btn.config(state="normal")

    def _on_result(self, result):
        s = result["stats"]
        self._last_result = result
        self._excel_path  = None
        self._detail_path = None
        self._status_lbl.config(
            text=f"Completado  •  {s['total']} velas  •  {s['wins']} acc. ({s['accuracy']:.1f}%)  "
                 f"•  Tier S+A+B: {s['filt_total']} trades ({s['filt_acc']:.1f}% acc)",
            fg=UP)
        self._run_btn.config(state="normal")
        self._excel_btn.config(state="normal", fg=TEXT)
        self._sim_btn.config(state="normal")
        if result["mode"] == "day":
            self._detail_btn.config(state="normal", fg=TEXT)
        else:
            self._detail_btn.config(state="disabled", fg=TEXT2)
        self.update_idletasks()
        self._render(result)

    def _re_simulate(self):
        if not self._last_result:
            return
        for w in self._sf.winfo_children():
            w.destroy()
        self.update_idletasks()
        self._render(self._last_result)

    def _get_stake(self) -> float:
        try:
            v = float(self._stake_var.get())
            return v if v > 0 else BET_SIZE
        except ValueError:
            return BET_SIZE

    def _get_limit(self) -> float:
        try:
            v = float(self._limit_var.get())
            return v if v > 0 else 500.0
        except ValueError:
            return 500.0

    # ══════════════════════════════════════════════════════════════════════════
    # RENDER
    # ══════════════════════════════════════════════════════════════════════════

    def _section(self, parent, title):
        f = tk.Frame(parent, bg=BG)
        f.pack(fill="x", pady=(14, 4))
        tk.Frame(f, bg=CARD2, height=1).pack(fill="x")
        _lbl(f, title, fg=ACCENT, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(4, 0))

    # ── helpers ───────────────────────────────────────────────────────────────

    def _bind_cell(self, widget, callback):
        """Bind click + hand cursor recursively to all children of a cell."""
        try:
            widget.configure(cursor="hand2")
        except tk.TclError:
            pass
        widget.bind("<Button-1>", callback)
        for child in widget.winfo_children():
            self._bind_cell(child, callback)

    def _show_day_detail(self, date_str: str, sel_tiers: list, stake: float):
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            has_mpl = True
        except ImportError:
            has_mpl = False

        df_main  = self._last_result["df"]
        day_df   = df_main[
            (df_main["date"] == date_str) & (df_main["tier"].isin(sel_tiers))
        ].reset_index(drop=True)

        if day_df.empty:
            return

        day_num   = int(date_str.split("-")[2])
        wins      = int(day_df["correct"].sum())
        losses    = len(day_df) - wins
        total     = len(day_df)
        net_pnl   = (wins - losses) * stake

        # Running bank starting at 100× stake (min $10 000)
        BANCO_INI = max(100_000.0, stake * 100)
        running   = BANCO_INI
        curve     = [running]
        min_bank, max_bank, min_idx, max_idx = running, running, 0, 0
        for i, row in day_df.iterrows():
            running += stake if row["correct"] else -stake
            curve.append(running)
            if running < min_bank:
                min_bank, min_idx = running, int(i) + 1
            if running > max_bank:
                max_bank, max_idx = running, int(i) + 1
        banco_final = running
        win_rate    = wins / total * 100 if total else 0.0

        # ── Ventana ────────────────────────────────────────────────────────
        top = tk.Toplevel(self)
        gain_txt = "+ Ganancia" if net_pnl > 0 else ("- Pérdida" if net_pnl < 0 else "Neutro")
        top.title(f"Detalle — Día {day_num:02d}  {gain_txt}  ({date_str})")
        top.configure(bg=BG)
        top.geometry("980x800")
        top.grab_set()

        outer_f = tk.Frame(top, bg=BG)
        outer_f.pack(fill="both", expand=True)
        cv2 = tk.Canvas(outer_f, bg=BG, highlightthickness=0)
        cv2.pack(side="left", fill="both", expand=True)
        vsb2 = tk.Scrollbar(outer_f, orient="vertical", command=cv2.yview)
        vsb2.pack(side="right", fill="y")
        cv2.configure(yscrollcommand=vsb2.set)
        sf2 = tk.Frame(cv2, bg=BG)
        sf2_id = cv2.create_window((0, 0), window=sf2, anchor="nw")
        sf2.bind("<Configure>", lambda _: cv2.configure(scrollregion=cv2.bbox("all")))
        cv2.bind("<Configure>", lambda e: cv2.itemconfig(sf2_id, width=e.width))
        top.bind("<MouseWheel>", lambda e: cv2.yview_scroll(int(-1*(e.delta/120)), "units"))

        # ── Header ─────────────────────────────────────────────────────────
        hdr = tk.Frame(sf2, bg=BG)
        hdr.pack(fill="x", padx=20, pady=(14, 6))
        gain_clr = UP if net_pnl > 0 else (DOWN if net_pnl < 0 else NEUTRAL)
        tk.Label(hdr, text=f"Detalle — día {day_num:02d}   ",
                 bg=BG, fg=TEXT, font=("Segoe UI", 14, "bold")).pack(side="left")
        tk.Label(hdr, text=gain_txt, bg=BG, fg=gain_clr,
                 font=("Segoe UI", 14, "bold")).pack(side="left")
        tier_str = "+".join(sel_tiers)
        tk.Label(hdr, text=f"   tiers {tier_str}  ·  ${stake:,.0f}/trade",
                 bg=BG, fg=TEXT2, font=("Segoe UI", 9)).pack(side="left")

        def _card(parent, label, value, color, sub=None):
            c = tk.Frame(parent, bg=CARD)
            c.grid_propagate(False)
            tk.Label(c, text=label, bg=CARD, fg=TEXT2,
                     font=("Segoe UI", 8)).pack(pady=(10, 2))
            tk.Label(c, text=value, bg=CARD, fg=color,
                     font=("Segoe UI", 18, "bold")).pack()
            if sub:
                tk.Label(c, text=sub, bg=CARD, fg=TEXT2,
                         font=("Segoe UI", 8)).pack(pady=(2, 8))
            else:
                tk.Frame(c, bg=CARD, height=8).pack()
            return c

        # ── Fila 1 ─────────────────────────────────────────────────────────
        r1 = tk.Frame(sf2, bg=BG)
        r1.pack(fill="x", padx=20, pady=(8, 4))
        for ci, (lbl, val, clr) in enumerate([
            ("Total resultados", str(total), TEXT),
            ("SI  (gana)",       str(wins),  UP),
            ("NO  (pierde)",     str(losses), DOWN),
            ("Balance neto",     f"${net_pnl:+,.0f}", gain_clr),
        ]):
            r1.columnconfigure(ci, weight=1)
            _card(r1, lbl, val, clr).grid(row=0, column=ci, padx=5, sticky="nsew")

        # ── Fila 2 ─────────────────────────────────────────────────────────
        r2 = tk.Frame(sf2, bg=BG)
        r2.pack(fill="x", padx=20, pady=4)
        fin_clr = UP if banco_final >= BANCO_INI else DOWN
        for ci, (lbl, val, clr, sub) in enumerate([
            ("Banco inicial",        f"${BANCO_INI:,.0f}", TEXT, None),
            ("Banco final",          f"${banco_final:,.0f}", fin_clr, f"{net_pnl:+,.0f}"),
            ("Win rate",             f"{win_rate:.1f}%",
             UP if win_rate >= 55 else (DOWN if win_rate < 50 else NEUTRAL), None),
        ]):
            r2.columnconfigure(ci, weight=1)
            _card(r2, lbl, val, clr, sub).grid(row=0, column=ci, padx=5, sticky="nsew")

        # ── Fila 3 min/max ─────────────────────────────────────────────────
        r3 = tk.Frame(sf2, bg=BG)
        r3.pack(fill="x", padx=20, pady=4)
        for ci, (lbl, val, clr, sub) in enumerate([
            ("Banco mínimo del día", f"${min_bank:,.0f}", DOWN, f"Resultado #{min_idx}"),
            ("Banco máximo del día", f"${max_bank:,.0f}", UP,   f"Resultado #{max_idx}"),
        ]):
            r3.columnconfigure(ci, weight=1)
            _card(r3, lbl, val, clr, sub).grid(row=0, column=ci, padx=5, sticky="nsew")

        # ── Gráfico ────────────────────────────────────────────────────────
        if has_mpl:
            # Metadatos por punto de la curva (índice 0 = inicio, 1..N = trades)
            pt_times  = ["Inicio"] + list(day_df["time"])
            pt_deltas = [0.0] + [stake if r["correct"] else -stake
                                 for _, r in day_df.iterrows()]
            pt_dirs   = [""] + list(day_df["prediction"])

            fig = Figure(figsize=(9, 3.0), facecolor=BG, tight_layout=True)
            ax  = fig.add_subplot(111, facecolor=BG)
            x   = list(range(len(curve)))
            line_clr = "#00d68f" if net_pnl >= 0 else "#ff4757"
            ax.plot(x, curve, color=line_clr, linewidth=1.8, zorder=3)
            ax.fill_between(x, BANCO_INI, curve,
                            where=[v >= BANCO_INI for v in curve],
                            color="#00d68f", alpha=0.12, zorder=2)
            ax.fill_between(x, BANCO_INI, curve,
                            where=[v < BANCO_INI for v in curve],
                            color="#ff4757", alpha=0.12, zorder=2)
            ax.axhline(BANCO_INI, color="#6e7191", linewidth=0.8, linestyle="--", zorder=1)
            ax.set_xlim(0, max(1, len(x) - 1))
            for spine in ax.spines.values():
                spine.set_color("#1e2236")
            ax.tick_params(colors="#6e7191", labelsize=7)
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}K"))
            ax.set_xlabel("Operación #", color="#6e7191", fontsize=7)

            # ── Tooltip interactivo ────────────────────────────────────────
            dot, = ax.plot([], [], "o", markersize=7, zorder=5,
                           color="#ffffff", markeredgewidth=0)
            vline = ax.axvline(x=0, color="#7c6fff", linewidth=1,
                               linestyle=":", alpha=0, zorder=4)
            annot = ax.annotate(
                "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", fc="#161929",
                          ec="#7c6fff", alpha=0.95, linewidth=1.2),
                arrowprops=dict(arrowstyle="->", color="#7c6fff", lw=1.2),
                color="#e8eaf6", fontsize=8, zorder=6,
            )
            annot.set_visible(False)

            def _on_move(event, _ax=ax, _curve=curve, _vline=vline,
                         _annot=annot, _dot=dot,
                         _times=pt_times, _deltas=pt_deltas,
                         _banco_ini=BANCO_INI):
                if event.inaxes != _ax:
                    _annot.set_visible(False)
                    _vline.set_alpha(0)
                    _dot.set_data([], [])
                    fc.draw_idle()
                    return
                xm = event.xdata
                if xm is None:
                    return
                idx = max(0, min(int(round(xm)), len(_curve) - 1))
                y_pt    = _curve[idx]
                delta   = _deltas[idx]
                hora    = _times[idx]
                acum    = y_pt - _banco_ini

                # Color del borde según resultado
                if delta > 0:
                    ec = "#00d68f"
                elif delta < 0:
                    ec = "#ff4757"
                else:
                    ec = "#7c6fff"
                _annot.get_bbox_patch().set_edgecolor(ec)

                if idx == 0:
                    txt = f"Inicio\nBanco: ${_banco_ini:,.0f}"
                else:
                    sign = "+" if delta >= 0 else ""
                    txt = (f"Op #{idx}  ·  {hora}\n"
                           f"Trade: {sign}${delta:,.0f}\n"
                           f"Banco: ${y_pt:,.0f}\n"
                           f"Acum:  {'+' if acum>=0 else ''}${acum:,.0f}")

                _annot.set_text(txt)
                _annot.xy = (idx, y_pt)

                # Ajustar xytext para que no salga del área
                x_frac = idx / max(1, len(_curve) - 1)
                _annot.set_position((-80, 12) if x_frac > 0.75 else (12, 12))

                _dot.set_data([idx], [y_pt])
                _dot.set_color(ec)
                _vline.set_xdata([idx])
                _vline.set_alpha(0.45)
                _annot.set_visible(True)
                fc.draw_idle()

            chart_f = tk.Frame(sf2, bg=BG)
            chart_f.pack(fill="x", padx=20, pady=(6, 4))
            fc = FigureCanvasTkAgg(fig, master=chart_f)
            fc.draw()
            fc.get_tk_widget().pack(fill="x")
            fc.mpl_connect("motion_notify_event", _on_move)

        # ── Secuencia ──────────────────────────────────────────────────────
        tk.Label(sf2, text="Secuencia de resultados", bg=BG, fg=TEXT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=20, pady=(10, 4))
        seq_outer = tk.Frame(sf2, bg=BG)
        seq_outer.pack(fill="x", padx=20, pady=(0, 8))
        cur_row_f = None
        ROW_SIZE  = 25
        for i, (_, row) in enumerate(day_df.iterrows()):
            if i % ROW_SIZE == 0:
                cur_row_f = tk.Frame(seq_outer, bg=BG)
                cur_row_f.pack(anchor="w", pady=1)
            ok    = bool(row["correct"])
            color = UP if ok else DOWN
            bg_c  = "#0a2e1e" if ok else "#2e0a0a"
            tk.Label(cur_row_f, text="SI" if ok else "NO",
                     bg=bg_c, fg=color, font=("Segoe UI", 8, "bold"),
                     padx=5, pady=2).pack(side="left", padx=2)

        # ── Tabla de operaciones ───────────────────────────────────────────
        tk.Label(sf2, text="Operaciones del día", bg=BG, fg=TEXT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=20, pady=(10, 4))
        tbl = tk.Frame(sf2, bg=CARD)
        tbl.pack(fill="x", padx=20, pady=(0, 24))

        col_defs = [("Hora", 8), ("Predicción", 11), ("Real", 10),
                    ("Correcto", 10), ("Confianza %", 12), ("Tier", 7),
                    ("En Filtro", 9), ("P&L", 10)]
        for ci, (h, w) in enumerate(col_defs):
            tbl.columnconfigure(ci, weight=1)
            tk.Label(tbl, text=h, bg=CARD2, fg=TEXT2,
                     font=("Segoe UI", 8, "bold"), width=w, anchor="center").grid(
                row=0, column=ci, padx=1, pady=3, sticky="nsew")

        for ri, (_, row) in enumerate(day_df.iterrows(), start=1):
            bg_c     = CARD if ri % 2 == 0 else CARD2
            ok       = bool(row["correct"])
            t_pnl    = stake if ok else -stake
            pred_clr = UP if row["prediction"] == "UP" else DOWN
            real_clr = UP if row["actual"]     == "UP" else DOWN
            vals = [
                (row["time"],                     TEXT2,  col_defs[0][1]),
                (row["prediction"],               pred_clr, col_defs[1][1]),
                (row["actual"],                   real_clr, col_defs[2][1]),
                ("SI" if ok else "NO",            UP if ok else DOWN, col_defs[3][1]),
                (f"{row['confidence']:.1f}",      TEXT2,  col_defs[4][1]),
                (row["tier"],                     TIER_COLORS.get(row["tier"], TEXT2), col_defs[5][1]),
                ("SI" if row["in_filter"] else "NO",
                 UP if row["in_filter"] else TEXT2, col_defs[6][1]),
                (f"${t_pnl:+,.0f}",               UP if t_pnl > 0 else DOWN, col_defs[7][1]),
            ]
            for ci, (txt, clr, w) in enumerate(vals):
                tk.Label(tbl, text=txt, bg=bg_c, fg=clr,
                         font=("Segoe UI", 8), width=w, anchor="center").grid(
                    row=ri, column=ci, padx=1, pady=2, sticky="nsew")

    def _render_calendar(self, f, result, stake):
        import calendar as cal_lib

        df_main = result["df"]
        label   = result["label"]
        parts   = label.split("-")
        year, month = int(parts[0]), int(parts[1])

        # Filtrar por tiers seleccionados
        sel_tiers = self._get_selected_tiers()
        filt_df   = df_main[df_main["tier"].isin(sel_tiers)].copy() if sel_tiers else df_main.iloc[0:0].copy()

        use_limit   = self._use_limit.get()
        daily_limit = self._get_limit()

        # P&L por día + límite diario
        by_date: dict = {}
        limit_results: dict = {}
        for date_str, grp in filt_df.groupby("date"):
            grp_sorted = grp.sort_values("time").reset_index(drop=True)
            wins   = int(grp_sorted["correct"].sum())
            total  = len(grp_sorted)
            losses = total - wins
            pnl    = (wins - losses) * stake
            by_date[date_str] = {"wins": wins, "losses": losses, "total": total, "pnl": pnl}
            if use_limit:
                limit_results[date_str] = find_daily_limit_hit(grp_sorted, stake, daily_limit)

        # Stats resumen
        if by_date:
            all_pnl      = [v["pnl"]   for v in by_date.values()]
            all_trades   = [v["total"] for v in by_date.values()]
            total_pnl    = sum(all_pnl)
            days_gain    = sum(1 for p in all_pnl if p > 0)
            days_loss    = sum(1 for p in all_pnl if p < 0)
            best_date    = max(by_date, key=lambda d: by_date[d]["pnl"])
            worst_date   = min(by_date, key=lambda d: by_date[d]["pnl"])
            maxrisk_date = max(by_date, key=lambda d: by_date[d]["total"])
            best_pnl     = by_date[best_date]["pnl"]
            worst_pnl    = by_date[worst_date]["pnl"]
            maxrisk_tr   = by_date[maxrisk_date]["total"]
            maxrisk_pnl  = maxrisk_tr * stake
            best_day_n   = int(best_date.split("-")[2])
            worst_day_n  = int(worst_date.split("-")[2])
            maxrisk_day_n = int(maxrisk_date.split("-")[2])
        else:
            total_pnl = days_gain = days_loss = 0
            best_day_n = worst_day_n = maxrisk_day_n = None
            best_pnl = worst_pnl = maxrisk_pnl = 0.0
            maxrisk_tr = 0

        # ── Sección ─────────────────────────────────────────────────────
        tier_label = "+".join(sel_tiers) if sel_tiers else "ninguno"
        lim_txt = f"  ·  Filtro ±${daily_limit:,.0f}" if use_limit else ""
        self._section(f, f"Calendario de Ganancias por Día  (${stake:.0f}/trade · tiers {tier_label}{lim_txt})")

        # ── Fila de resumen ──────────────────────────────────────────────
        best_lbl    = f"{best_day_n:02d}  (${best_pnl:+,.0f})"   if best_day_n    else "—"
        worst_lbl   = f"{worst_day_n:02d}  (${worst_pnl:+,.0f})" if worst_day_n   else "—"
        maxrisk_lbl = (f"${maxrisk_pnl:,.0f}  (día {maxrisk_day_n:02d} · {maxrisk_tr}tr)"
                       if maxrisk_day_n else "—")
        summary_items = [
            ("P&L Total Mes",     f"${total_pnl:+,.0f}", UP if total_pnl >= 0 else DOWN),
            ("Días con ganancia", str(days_gain),         UP),
            ("Días con pérdida",  str(days_loss),         DOWN),
            ("Mejor día",         best_lbl,               UP),
            ("Peor día",          worst_lbl,              DOWN),
            ("Máx. Riesgo/día",   maxrisk_lbl,            YELLOW),
        ]
        if use_limit and limit_results:
            hit_up   = sum(1 for v in limit_results.values() if v["hit"] == "UP")
            hit_down = sum(1 for v in limit_results.values() if v["hit"] == "DOWN")
            hit_none = sum(1 for v in limit_results.values() if v["hit"] is None)
            summary_items += [
                (f"+${daily_limit:,.0f} primero", str(hit_up),   UP),
                (f"-${daily_limit:,.0f} primero", str(hit_down), DOWN),
                ("Sin límite",                    str(hit_none), NEUTRAL),
            ]
        _card_row(f, summary_items, font_val=("Segoe UI", 13, "bold"))

        # ── Grid del calendario ──────────────────────────────────────────
        cal_frame = tk.Frame(f, bg=BG)
        cal_frame.pack(fill="x", pady=(0, 14))

        day_names = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
        for ci in range(7):
            cal_frame.columnconfigure(ci, weight=1)
            tk.Label(cal_frame, text=day_names[ci], bg=BG, fg=TEXT2,
                     font=("Segoe UI", 8, "bold")).grid(
                row=0, column=ci, padx=3, pady=(0, 4), sticky="ew")

        month_weeks = cal_lib.monthcalendar(year, month)

        for ri, week in enumerate(month_weeks, start=1):
            for ci, day in enumerate(week):
                if day == 0:
                    tk.Frame(cal_frame, bg=BG, height=72).grid(
                        row=ri, column=ci, padx=3, pady=3, sticky="nsew")
                    continue

                date_str = f"{year:04d}-{month:02d}-{day:02d}"
                info     = by_date.get(date_str)
                lr       = limit_results.get(date_str) if use_limit else None

                is_best  = (best_day_n  is not None and date_str == best_date)
                is_worst = (worst_day_n is not None and date_str == worst_date)

                # Border color: límite tiene prioridad sobre mejor/peor si está activo
                if use_limit and lr:
                    if   lr["hit"] == "UP":   border_color = UP
                    elif lr["hit"] == "DOWN": border_color = DOWN
                    else:                     border_color = NEUTRAL
                else:
                    border_color = UP if is_best else (DOWN if is_worst else CARD)

                outer = tk.Frame(cal_frame, bg=border_color)
                outer.grid(row=ri, column=ci, padx=3, pady=3, sticky="nsew")
                inn_bg = CARD2
                inn = tk.Frame(outer, bg=inn_bg)
                inn.pack(fill="both", expand=True, padx=2, pady=2)

                # Número del día
                tk.Label(inn, text=f"{day:02d}", bg=inn_bg, fg=TEXT2,
                         font=("Segoe UI", 8, "bold"),
                         anchor="e").pack(fill="x", padx=6, pady=(4, 0))

                if info:
                    pnl    = info["pnl"]
                    wins   = info["wins"]
                    losses = info["losses"]
                    total  = info["total"]
                    pnl_color = UP if pnl > 0 else (DOWN if pnl < 0 else NEUTRAL)
                    tk.Label(inn, text=f"${pnl:+,.0f}", bg=inn_bg, fg=pnl_color,
                             font=("Segoe UI", 12, "bold")).pack()
                    tk.Label(inn, text=f"{wins}W/{losses}L · {total}tr",
                             bg=inn_bg, fg=TEXT2,
                             font=("Segoe UI", 7)).pack()

                    # ── Bloque límite diario ──────────────────────────
                    if use_limit and lr:
                        if lr["hit"] == "UP":
                            lbg = "#0a2e1e"
                            lc  = UP
                            ltxt = f"🟢 +${daily_limit:,.0f} op#{lr['op']} {lr['time']}"
                        elif lr["hit"] == "DOWN":
                            lbg = "#2e0a0a"
                            lc  = DOWN
                            ltxt = f"🔴 -${daily_limit:,.0f} op#{lr['op']} {lr['time']}"
                        else:
                            lbg = CARD
                            lc  = TEXT2
                            ltxt = "⚪ Sin límite"
                        lim_fr = tk.Frame(inn, bg=lbg)
                        lim_fr.pack(fill="x", padx=3, pady=(3, 4))
                        tk.Label(lim_fr, text=ltxt, bg=lbg, fg=lc,
                                 font=("Segoe UI", 7, "bold"),
                                 wraplength=110, justify="center").pack(pady=2)
                    else:
                        tk.Frame(inn, bg=inn_bg, height=4).pack()

                    click_cb = lambda e, d=date_str, st=list(sel_tiers), sk=stake: \
                        self._show_day_detail(d, st, sk)
                    self._bind_cell(outer, click_cb)
                else:
                    tk.Label(inn, text="—", bg=inn_bg, fg=TEXT2,
                             font=("Segoe UI", 12)).pack()
                    tk.Label(inn, text="0 trades", bg=inn_bg, fg=TEXT2,
                             font=("Segoe UI", 7)).pack(pady=(0, 6))

    def _render_year_calendar(self, f, result, stake):
        MONTH_NAMES = ["Enero","Febrero","Marzo","Abril","Mayo","Junio",
                       "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"]
        MONTH_SHORT = ["Ene","Feb","Mar","Abr","May","Jun",
                       "Jul","Ago","Sep","Oct","Nov","Dic"]

        df_main   = result["df"]
        year_str  = result["label"]
        sel_tiers = self._get_selected_tiers()
        filt_df   = (df_main[df_main["tier"].isin(sel_tiers)].copy()
                     if sel_tiers else df_main.iloc[0:0].copy())

        by_month: dict = {}
        filt_df["_mo"] = filt_df["date"].str[:7]
        for mo_key, grp in filt_df.groupby("_mo"):
            w = int(grp["correct"].sum()); t = len(grp)
            by_month[mo_key] = {
                "wins": w, "losses": t - w, "total": t,
                "pnl":  (w - (t - w)) * stake,
                "acc":  w / t * 100 if t > 0 else 0,
            }

        # Resumen
        tier_label = "+".join(sel_tiers) if sel_tiers else "ninguno"
        self._section(f, f"Calendario Anual {year_str}  (${stake:.0f}/trade · tiers {tier_label})")

        if by_month:
            total_pnl_y = sum(v["pnl"] for v in by_month.values())
            mo_gain_y   = sum(1 for v in by_month.values() if v["pnl"] > 0)
            mo_loss_y   = sum(1 for v in by_month.values() if v["pnl"] < 0)
            best_mo     = max(by_month, key=lambda k: by_month[k]["pnl"])
            worst_mo    = min(by_month, key=lambda k: by_month[k]["pnl"])
            best_mi     = int(best_mo.split("-")[1]) - 1
            worst_mi    = int(worst_mo.split("-")[1]) - 1
            best_lbl    = f"{MONTH_SHORT[best_mi]}  (${by_month[best_mo]['pnl']:+,.0f})"
            worst_lbl   = f"{MONTH_SHORT[worst_mi]}  (${by_month[worst_mo]['pnl']:+,.0f})"
        else:
            total_pnl_y = mo_gain_y = mo_loss_y = 0
            best_mo = worst_mo = None
            best_lbl = worst_lbl = "—"

        _card_row(f, [
            ("P&L Total Año",      f"${total_pnl_y:+,.0f}", UP if total_pnl_y >= 0 else DOWN),
            ("Meses con ganancia", str(mo_gain_y),           UP),
            ("Meses con pérdida",  str(mo_loss_y),           DOWN),
            ("Mejor mes",          best_lbl,                 UP),
            ("Peor mes",           worst_lbl,                DOWN),
        ], font_val=("Segoe UI", 13, "bold"))

        # Grid 4 filas × 3 cols
        grid = tk.Frame(f, bg=BG)
        grid.pack(fill="x", pady=(0, 14))
        for ci in range(3):
            grid.columnconfigure(ci, weight=1)

        for row_i in range(4):
            for col_i in range(3):
                m      = row_i * 3 + col_i + 1
                mo_key = f"{year_str}-{m:02d}"
                info   = by_month.get(mo_key)

                is_best  = (best_mo  == mo_key)
                is_worst = (worst_mo == mo_key)
                border   = UP if is_best else (DOWN if is_worst else CARD)

                outer = tk.Frame(grid, bg=border)
                outer.grid(row=row_i, column=col_i, padx=5, pady=5, sticky="nsew")
                inn = tk.Frame(outer, bg=CARD2)
                inn.pack(fill="both", expand=True, padx=2, pady=2)

                _lbl(inn, MONTH_NAMES[m - 1], fg=ACCENT,
                     font=("Segoe UI", 10, "bold")).pack(pady=(8, 2))

                if info:
                    pnl = info["pnl"]
                    acc = info["acc"]
                    pc  = UP if pnl > 0 else (DOWN if pnl < 0 else NEUTRAL)
                    ac  = UP if acc >= 55 else (DOWN if acc < 50 else NEUTRAL)
                    _lbl(inn, f"${pnl:+,.0f}", fg=pc,
                         font=("Segoe UI", 16, "bold")).pack()
                    _lbl(inn, f"{acc:.1f}%", fg=ac,
                         font=("Segoe UI", 11)).pack()
                    _lbl(inn, f"{info['wins']}W / {info['losses']}L · {info['total']}tr",
                         fg=TEXT2, font=("Segoe UI", 7)).pack(pady=(0, 8))
                else:
                    _lbl(inn, "—", fg=TEXT2, font=("Segoe UI", 14)).pack()
                    _lbl(inn, "sin datos", fg=TEXT2, font=("Segoe UI", 7)).pack(pady=(0, 8))

    def _render(self, result):
        f     = self._sf
        s     = result["stats"]
        mode  = result["mode"]
        stake = self._get_stake()

        title = f"Backtest v2  •  {result['label']}"
        if mode == "month":
            title += f"  ({s['total']} velas  •  {result['df']['date'].nunique()} días)"
        self._section(f, title)

        # ── Resumen global ────────────────────────────────────────────────
        _card_row(f, [
            ("Total velas",         s["total"],                  TEXT),
            ("Aciertos",            s["wins"],                   UP),
            ("Fallos",              s["losses"],                 DOWN),
            ("Precisión global",    f"{s['accuracy']:.1f}%",     ACCENT),
            ("Racha max. aciertos", s["max_win_streak"],         UP),
            ("Racha max. fallos",   s["max_loss_streak"],        DOWN),
        ], font_val=("Segoe UI", 18, "bold"))

        # ── Stats filtradas (tiers seleccionados) ─────────────────────────
        n_days       = max(1, result["df"]["date"].nunique())
        sel_tiers    = self._get_selected_tiers()
        tier_label   = "+".join(sel_tiers) if sel_tiers else "ninguno"
        pnl_sel      = filtered_pnl(result["df"], sel_tiers, stake)
        filt_acc_dyn = pnl_sel["accuracy"] if pnl_sel["total_trades"] else float("nan")

        self._section(f, f"Rendimiento Filtrado  ·  Tiers: {tier_label}  ·  Vol≥{FILTERS['min_volume']} BTC  |move|≥{FILTERS['min_abs_move']}%")
        _card_row(f, [
            (f"Trades ({tier_label})",    pnl_sel["total_trades"],                                       ACCENT),
            (f"Precisión ({tier_label})", f"{filt_acc_dyn:.1f}%" if not np.isnan(filt_acc_dyn) else "n/a", UP),
            ("Trades/día",               f"{pnl_sel['total_trades']/n_days:.1f}",                        TEXT),
            (f"P&L total (${stake:.0f}/tr)", f"${pnl_sel['total_pnl']:+,.0f}",
             UP if pnl_sel["total_pnl"] >= 0 else DOWN),
            ("P&L/día",                  f"${pnl_sel['pnl_per_day']:+.1f}",
             UP if pnl_sel["pnl_per_day"] >= 0 else DOWN),
            ("ROI%",                     f"{pnl_sel['roi_pct']:+.1f}%",
             UP if pnl_sel["roi_pct"] >= 0 else DOWN),
            ("Wins / Losses",            f"{pnl_sel['wins']} / {pnl_sel['losses']}",                     TEXT2),
        ])

        # ── Tier Summary ──────────────────────────────────────────────────
        self._section(f, "Rendimiento por Tier  (Tier = vol+|move| de la vela SEÑAL)")
        _lbl(f,
             "S: Vol≥500 & |mov|≥0.3%  •  A: Vol≥300 & |mov|≥0.2%  •  B: Vol≥200 & |mov|≥0.1%  •  C: Vol≥100  •  D: resto",
             fg=TEXT2, font=("Segoe UI", 8)).pack(anchor="w", pady=(0, 6))

        tier_frame = tk.Frame(f, bg=BG)
        tier_frame.pack(fill="x", pady=(0, 8))
        df_main = result["df"]

        for ci, tier in enumerate(["S", "A", "B", "C", "D"]):
            tier_frame.columnconfigure(ci, weight=1)
            sub = df_main[df_main["tier"] == tier]
            n   = len(sub)
            if n == 0:
                w_pct = 0.0;  w_n = 0
            else:
                w_n   = int(sub["correct"].sum())
                w_pct = w_n / n * 100
            pnl   = (w_n - (n - w_n)) * stake
            color = TIER_COLORS[tier]

            outer_bdr = tk.Frame(tier_frame, bg=color)
            outer_bdr.grid(row=0, column=ci, padx=5, sticky="nsew")
            inn = tk.Frame(outer_bdr, bg=CARD)
            inn.pack(fill="both", expand=True, padx=2, pady=2)

            _lbl(inn, f"Tier {tier}", fg=color,  font=("Segoe UI", 12, "bold")).pack(pady=(10,2))
            acc_clr = UP if w_pct >= 55 else (DOWN if w_pct < 45 else NEUTRAL)
            _lbl(inn, f"{w_pct:.1f}%",  fg=acc_clr, font=("Segoe UI", 20, "bold")).pack()
            _lbl(inn, f"{w_n}/{n}",     fg=TEXT2,   font=("Segoe UI", 8)).pack()
            pnl_clr = UP if pnl >= 0 else DOWN
            _lbl(inn, f"P&L ${pnl:+.0f}", fg=pnl_clr, font=("Segoe UI", 9, "bold")).pack()
            _lbl(inn, f"{n/max(1,n_days):.1f}/día", fg=TEXT2, font=("Segoe UI", 7)).pack(pady=(0,8))

        # ── P&L acumulado ─────────────────────────────────────────────────
        self._section(f, f"Simulación P&L  (${stake:.0f} por trade, apuesta fija)")
        pnl_tbl_frame = tk.Frame(f, bg=CARD)
        pnl_tbl_frame.pack(fill="x", pady=(0, 8))
        hdrs   = ["Min Tier", "Trades", "Acc%", "P&L Total", "P&L/Día", "ROI%"]
        widths = [8, 8, 8, 10, 10, 8]
        for ci, (h, w) in enumerate(zip(hdrs, widths)):
            tk.Label(pnl_tbl_frame, text=h, bg=CARD2, fg=TEXT2,
                     font=("Segoe UI", 9, "bold"), width=w, anchor="center").grid(
                row=0, column=ci, padx=2, pady=4, sticky="nsew")
        for ri, mt in enumerate(["S","A","B","C","D"], start=1):
            p    = simulate_pnl(result["df"], stake, mt)
            bg_c = CARD if ri % 2 == 0 else CARD2
            pclr = UP if p["total_pnl"] >= 0 else DOWN
            rclr = UP if p["roi_pct"]   >= 0 else DOWN
            row_vals = [
                (f"≥ {mt}",            TEXT,   widths[0]),
                (str(p["total_trades"]), TEXT2, widths[1]),
                (f"{p['accuracy']:.1f}%", UP if p['accuracy'] >= 55 else (DOWN if p['accuracy'] < 50 else NEUTRAL), widths[2]),
                (f"${p['total_pnl']:+.0f}", pclr, widths[3]),
                (f"${p['pnl_per_day']:+.1f}", pclr, widths[4]),
                (f"{p['roi_pct']:+.1f}%", rclr, widths[5]),
            ]
            for ci, (txt, clr, w) in enumerate(row_vals):
                tk.Label(pnl_tbl_frame, text=txt, bg=bg_c, fg=clr,
                         font=("Segoe UI", 9), width=w, anchor="center").grid(
                    row=ri, column=ci, padx=2, pady=3, sticky="nsew")

        # ── Por Volumen Señal ─────────────────────────────────────────────
        self._section(f, "Precisión por Volumen de la Vela SEÑAL (BTC) — usado para Tier/Filtro")
        vf = tk.Frame(f, bg=BG)
        vf.pack(fill="x", pady=(0, 8))
        by_svol = s["by_signal_vol"]
        for ci, (vbin, rd) in enumerate(by_svol.iterrows()):
            vf.columnconfigure(ci, weight=1)
            tot_v = int(rd["total"])
            if tot_v == 0:
                continue
            av  = rd["accuracy"]
            clr = UP if av >= 60 else (DOWN if av < 50 else NEUTRAL)
            card = tk.Frame(vf, bg=CARD)
            card.grid(row=0, column=ci, padx=3, pady=2, sticky="nsew")
            _lbl(card, str(vbin), fg=TEXT2, font=("Segoe UI", 7, "bold")).pack(pady=(6,0))
            _lbl(card, f"{av:.0f}%", fg=clr, font=("Segoe UI", 13, "bold")).pack()
            _lbl(card, f"{int(rd['wins'])}/{tot_v}", fg=TEXT2,
                 font=("Segoe UI", 7)).pack(pady=(0,6))

        # ── Por Hora ──────────────────────────────────────────────────────
        self._section(f, "Precisión por Hora (hora local de apertura de vela TARGET)")
        hf = tk.Frame(f, bg=BG)
        hf.pack(fill="x", pady=(0, 8))
        by_h = s["by_hour"].reset_index()
        for ci, (_, rd) in enumerate(by_h.iterrows()):
            hf.columnconfigure(ci, weight=1)
            tot_h = int(rd["total"])
            if tot_h == 0:
                continue
            av  = rd["accuracy"]
            clr = UP if av >= 60 else (DOWN if av < 50 else NEUTRAL)
            card = tk.Frame(hf, bg=CARD)
            card.grid(row=0, column=ci, padx=2, pady=2, sticky="nsew")
            _lbl(card, f"{int(rd['hour']):02d}h", fg=TEXT2,
                 font=("Segoe UI", 7, "bold")).pack(pady=(5,0))
            _lbl(card, f"{av:.0f}%", fg=clr, font=("Segoe UI", 11, "bold")).pack()
            _lbl(card, f"n={tot_h}", fg=TEXT2, font=("Segoe UI", 6)).pack(pady=(0,5))

        # ── Por Minuto intra-vela ─────────────────────────────────────────
        if s["minute_acc"]:
            self._section(f, "Precisión por Minuto intra-vela TARGET")
            mf = tk.Frame(f, bg=BG)
            mf.pack(fill="x", pady=(0, 8))
            for ci, (mn, info) in enumerate(sorted(s["minute_acc"].items())):
                mf.columnconfigure(ci, weight=1)
                av  = info["accuracy"]
                clr = UP if av >= 60 else (DOWN if av < 50 else NEUTRAL)
                is_best = mn == s["best_minute"]
                card = tk.Frame(mf, bg=CARD if not is_best else CARD2)
                card.grid(row=0, column=ci, padx=4, pady=2, sticky="nsew")
                star = "★ " if is_best else ""
                _lbl(card, f"{star}min {mn}", fg=YELLOW if is_best else TEXT2,
                     font=("Segoe UI", 8, "bold")).pack(pady=(6,0))
                _lbl(card, f"{av:.1f}%", fg=clr, font=("Segoe UI", 14, "bold")).pack()
                _lbl(card, f"{info['correct']}/{info['total']}", fg=TEXT2,
                     font=("Segoe UI", 7)).pack(pady=(0,6))

        # ── Calendario P&L ───────────────────────────────────────────────
        if mode == "month":
            self._render_calendar(f, result, stake)
        elif mode == "year":
            self._render_year_calendar(f, result, stake)

        # ── Señales ───────────────────────────────────────────────────────
        if s["signal_bias"]:
            self._section(f, "Distribución de Señales")
            sg_f = tk.Frame(f, bg=CARD)
            sg_f.pack(fill="x", pady=(0, 8))
            hdrs_s = ["Señal", "UP", "DOWN", "NEUTRAL", "Total"]
            wds_s  = [12, 8, 8, 10, 8]
            for ci, (h, w) in enumerate(zip(hdrs_s, wds_s)):
                tk.Label(sg_f, text=h, bg=CARD2, fg=TEXT2,
                         font=("Segoe UI", 8, "bold"), width=w, anchor="center").grid(
                    row=0, column=ci, padx=2, pady=3, sticky="nsew")
            for ri, (sn, v) in enumerate(sorted(s["signal_bias"].items()), start=1):
                bg_c = CARD if ri % 2 == 0 else CARD2
                vals = [(sn, TEXT), (str(v["UP"]), UP), (str(v["DOWN"]), DOWN),
                        (str(v["NEUTRAL"]), NEUTRAL), (str(v["total"]), TEXT2)]
                for ci, ((txt, c), w) in enumerate(zip(vals, wds_s)):
                    tk.Label(sg_f, text=txt, bg=bg_c, fg=c,
                             font=("Segoe UI", 8), width=w, anchor="center").grid(
                        row=ri, column=ci, padx=2, pady=2, sticky="nsew")


def main():
    app = BacktestApp()
    app.mainloop()


if __name__ == "__main__":
    main()
