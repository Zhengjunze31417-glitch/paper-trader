#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
thermo_core.py
--------------
Shared foundation imported by both thermo_trainer.py and thermo_scanner.py.

Contains:
  Config        — single frozen config dataclass
  ScanResult    — typed output record
  MarketData    — universe fetch + yfinance download
  FeatureEngine — all signal calculations (original + 3 new features)
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from io import StringIO
from typing import Iterator

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import tempfile, os as _os
_yf_cache = _os.path.join(_os.path.expanduser("~"), ".yf_cache_thermo")
_os.makedirs(_yf_cache, exist_ok=True)
yf.set_tz_cache_location(_yf_cache)
from scipy.stats import entropy as scipy_entropy

warnings.filterwarnings("ignore")

log = logging.getLogger("thermo")


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Config:
    """Single source of truth for every tunable value. Immutable at runtime."""

    # ── Model ──────────────────────────────────────────────────────────────────
    model_path: str = str(__import__('pathlib').Path(__file__).parent / "thermo_ai_brain.joblib")
    min_confidence:    float = 0.53
    top_n:             int   = 10

    # -- Label
    forward_window:    int   = 20

    # -- Features
    features: tuple = (
        "dist_20ma_pct",
        "energy_rank",
        "vol_contraction",
        "mom_5d",
        "mom_20d",
        "range_ratio",
    )

    # -- Signal windows
    vol_window:        int   = 5
    ent_window:        int   = 14
    ent_order:         int   = 3
    flow_lookback:     int   = 15
    atr_window:        int   = 14
    ma_window:         int   = 20
    rs_window:         int   = 20
    vol_bias_window:   int   = 10
    vcontract_fast:    int   = 5
    vcontract_slow:    int   = 30
    rank_window:       int   = 200
    rank_min_periods:  int   = 50

    # -- Filters
    min_dollar_vol:    float = 20_000_000.0
    min_price:         float = 5.0
    min_history_bars:  int   = 252
    lookback_days:     int   = 2600
    stale_data_days:   int   = 3


    # ── Fallback universe ──────────────────────────────────────────────────────
    fallback_tickers: tuple = (
        "NVDA", "AMD", "AAPL", "MSFT",
        "AMZN", "GOOGL", "META", "TSLA",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA TYPES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
@dataclass
class ScanResult:
    """One row of scanner output."""
    ticker:        str
    close:         float
    stop_loss:     float
    ai_confidence: float
    dist_20ma_pct: float
    energy_rank:   float
    vol_contraction: float
    mom_5d:        float
    mom_20d:       float
    range_ratio:   float

    def to_dict(self) -> dict:
        return {
            "ticker":        self.ticker,
            "Close":         self.close,
            "StopLoss":      self.stop_loss,
            "AI_Confidence": self.ai_confidence,
            "dist_20ma_pct": self.dist_20ma_pct,
            "energy_rank":   self.energy_rank,
            "vol_contraction": self.vol_contraction,
            "mom_5d":        self.mom_5d,
            "mom_20d":       self.mom_20d,
            "range_ratio":   self.range_ratio,
        }

# ══════════════════════════════════════════════════════════════════════════════
# 3. MARKET DATA
# ══════════════════════════════════════════════════════════════════════════════

class MarketData:
    """Fetch universe and download OHLCV data including SPY benchmark."""

    SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def fetch_universe(self) -> list:
        log.info("Fetching S&P 500 constituents ...")
        try:
            resp = requests.get(
                self.SP500_URL,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            resp.raise_for_status()
            tables  = pd.read_html(StringIO(resp.text))
            df      = next(t for t in tables if "Symbol" in t.columns or "Ticker" in t.columns)
            col     = "Symbol" if "Symbol" in df.columns else "Ticker"
            tickers = sorted(set(df[col].astype(str).str.replace(".", "-", regex=False)))
            log.info("Universe: %d tickers", len(tickers))
            return tickers
        except Exception as exc:
            log.warning(
                "Wikipedia fetch failed (%s). Falling back to %d tickers — INCOMPLETE.",
                exc, len(self.cfg.fallback_tickers),
            )
            return list(self.cfg.fallback_tickers)

    def download(self, tickers: list) -> tuple[dict, pd.Series]:
        """Download in small batches to avoid overwhelming DNS resolver."""
        import time
        start = (datetime.now() - timedelta(days=self.cfg.lookback_days)).strftime("%Y-%m-%d")
        all_tickers = list(set(tickers) | {"SPY"})
        batch_size = 50
        result = {}

        log.info("Downloading %d tickers in batches of %d ...", len(all_tickers), batch_size)

        for i in range(0, len(all_tickers), batch_size):
            batch = all_tickers[i: i + batch_size]
            log.info("Batch %d/%d ...", i // batch_size + 1, -(-len(all_tickers) // batch_size))
            try:
                raw = yf.download(
                    batch,
                    start=start,
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=True,
                    threads=False,  # sequential within batch — avoids thread exhaustion
                    progress=False,
                )
                result.update(self._unpack(raw, batch))
            except Exception as exc:
                log.warning("Batch %d failed: %s", i // batch_size + 1, exc)
            time.sleep(1)  # brief pause between batches

        spy_close = result.pop("SPY", pd.DataFrame())
        spy_series = spy_close["Close"] if not spy_close.empty else pd.Series(dtype=float)

        log.info("Received data for %d tickers (+SPY benchmark).", len(result))
        self._check_freshness(result)
        return result, spy_series

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _unpack(raw: pd.DataFrame, tickers: list) -> dict:
        if isinstance(raw.columns, pd.MultiIndex):
            available = raw.columns.get_level_values(0).unique()
            return {t: raw[t].copy() for t in available if t in tickers}
        if len(tickers) == 1:
            return {tickers[0]: raw.copy()}
        log.warning("Unexpected flat DataFrame — only '%s' recovered.", tickers[0])
        return {tickers[0]: raw.copy()}

    def _check_freshness(self, ticker_dfs: dict) -> None:
        if not ticker_dfs:
            return
        latest = max(df.index[-1] for df in ticker_dfs.values() if not df.empty)
        age    = (datetime.now() - latest.to_pydatetime().replace(tzinfo=None)).days
        if age > self.cfg.stale_data_days:
            log.warning("Newest bar is %d days old (%s) — data may be stale.", age, latest.date())
        else:
            log.info("Data freshness OK — newest bar: %s", latest.date())


# ══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class FeatureEngine:
    """
    Stateless signal calculator.
    compute() returns a ScanResult (scanner mode) or a feature dict (trainer mode).
    All methods are pure functions — no side effects, no stored state.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    # ── public entry point ────────────────────────────────────────────────────

    def compute(
        self,
        ticker: str,
        df: pd.DataFrame,
        spy_close: pd.Series,
        as_history: bool = False,
    ):
        """
        Compute features for one ticker.

        as_history=False (scanner):  returns a single ScanResult for today, or None.
        as_history=True  (trainer):  returns a DataFrame of all historical rows,
                                     aligned to df's index, for label generation.
        """
        df = df.dropna(subset=["Close"]).copy()
        if not self._passes_filters(df):
            return None

        df = self._add_all_features(df, spy_close)

        if as_history:
            return df  # trainer will add labels and sample weights

        # Scanner: return just the latest bar as a typed ScanResult
        last      = df.iloc[-1]
        price     = float(last["Close"])
        stop_loss = price - 1.5 * self._atr_series(df).iloc[-1]

        return ScanResult(
            ticker        = ticker,
            close         = price,
            stop_loss     = round(stop_loss, 2),
            ai_confidence = 0.0,
            dist_20ma_pct = float(last['dist_20ma_pct']),
            energy_rank   = float(last['energy_rank']),
            vol_contraction = float(last['vol_contraction']),
            mom_5d        = float(last['mom_5d']),
            mom_20d       = float(last['mom_20d']),
            range_ratio   = float(last['range_ratio']),
        )

    # ── full feature matrix ───────────────────────────────────────────────────

    def _add_all_features(self, df: pd.DataFrame, spy_close: pd.Series) -> pd.DataFrame:
        """Add all feature columns to df in-place and return it."""
        cfg = self.cfg

        df["ret"]    = np.log(df["Close"] / df["Close"].shift(1))
        energy       = df["ret"].rolling(cfg.vol_window).std() * (df["Close"] * df["Volume"])
        atr          = self._atr_series(df)

        # ── original 5 features ───────────────────────────────────────────────
        df["energy_rank"]   = energy.rolling(cfg.rank_window, min_periods=cfg.rank_min_periods).rank(pct=True) * 100
        df["vol_contraction"] = df["ret"].rolling(5).std() / (df["ret"].rolling(60).std() + 1e-9)
        df["entropy_score"] = self._entropy_score_series(df["ret"])
        df["sprint_score"]  = (
            0.4 * df["entropy_score"]
            + 0.4 * (1 - df["vol_contraction"].clip(0, 2) / 2) * 100
            + 0.2 * df["energy_rank"]
        )
        ma20                = df["Close"].rolling(cfg.ma_window).mean()
        df["dist_20ma_pct"] = ((df["Close"] - ma20) / ma20 * 100).abs()

        # ── NEW: relative strength vs SPY ─────────────────────────────────────
        # How much has this stock outperformed SPY over the past rs_window days?
        spy_aligned       = spy_close.reindex(df.index).ffill()
        stock_ret         = df["Close"].pct_change(cfg.rs_window)
        spy_ret           = spy_aligned.pct_change(cfg.rs_window)
        df["rs_vs_spy"]   = (stock_ret - spy_ret) * 100   # percentage points

        # ── NEW: volume bias (up-day vol / down-day vol) ──────────────────────
        # Values > 1 mean buyers are more aggressive than sellers.
        up_mask           = df["ret"] > 0
        up_vol            = df["Volume"].where(up_mask).rolling(cfg.vol_bias_window, min_periods=1).mean()
        down_vol          = df["Volume"].where(~up_mask).rolling(cfg.vol_bias_window, min_periods=1).mean()
        df["vol_bias"]    = (up_vol / (down_vol + 1e-9)).clip(0, 5)

        # ── NEW: ATR contraction ratio (coiling / compression signal) ─────────
        # Short ATR / Long ATR < 1 means volatility is compressing — often
        # precedes a breakout as energy builds before release.
        atr_fast              = self._atr_series(df, window=cfg.vcontract_fast)
        atr_slow              = self._atr_series(df, window=cfg.vcontract_slow)
        df["atr_contraction"] = atr_fast / (atr_slow + 1e-9)

        # ── NEW: multi-timeframe momentum ─────────────────────────────────────
        df["mom_5d"]  = df["Close"].pct_change(5)  * 100
        df["mom_20d"] = df["Close"].pct_change(20) * 100

        # ── NEW: range compression (coiling signal) ───────────────────────────
        daily_range        = df["High"] - df["Low"]
        df["range_ratio"]  = (
            daily_range.rolling(5).mean() /
            (daily_range.rolling(30).mean() + 1e-9)
        )

        return df

    # ── helpers ───────────────────────────────────────────────────────────────

    def _passes_filters(self, df: pd.DataFrame) -> bool:
        cfg = self.cfg
        if len(df) < cfg.min_history_bars:
            return False
        if float(df["Close"].iloc[-1]) < cfg.min_price:
            return False
        return float((df["Close"] * df["Volume"]).tail(20).mean()) >= cfg.min_dollar_vol

    def _atr_series(self, df: pd.DataFrame, window: int = None) -> pd.Series:
        w         = window or self.cfg.atr_window
        high, low, close = df["High"], df["Low"], df["Close"]
        tr = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        return tr.rolling(w).mean()

    # flow_z replaced by vol_contraction





    def _entropy_score_series(self, ret: pd.Series) -> pd.Series:
        perm_ent = self._rolling_permutation_entropy(ret)
        rank     = perm_ent.rolling(
            self.cfg.rank_window, min_periods=self.cfg.rank_min_periods
        ).rank(pct=True)
        return (1 - rank) * 100   # inverted: low entropy = high score

    def _rolling_permutation_entropy(self, series: pd.Series) -> pd.Series:
        """
        Permutation entropy — O(n * window).
        Normalised to [0, 1]. Lower = more ordered = pre-breakout signal.
        """
        window    = self.cfg.ent_window
        order     = self.cfg.ent_order
        log_denom = math.log(math.factorial(order))
        n         = len(series)
        out       = np.full(n, np.nan)
        vals      = series.values

        for i in range(window - 1, n):
            seg      = vals[i - window + 1 : i + 1]
            patterns = np.array([
                tuple(np.argsort(seg[j : j + order]))
                for j in range(len(seg) - order + 1)
            ])
            _, counts = np.unique(patterns, axis=0, return_counts=True)
            p         = counts / counts.sum()
            out[i]    = scipy_entropy(p) / log_denom

        return pd.Series(out, index=series.index)
