#!/usr/bin/env python3
"""
data_prep.py - Download market data and generate scenario-return matrices for quantum_finance.

This script produces a portable .npz file that main.py can load via EngineConfig.returns_npz_path.

Outputs (.npz):
  - returns: (S, N) float64 matrix of log-returns scenarios r_{s,i}
  - p_s:     (S,) float64 scenario probabilities (uniform by default)
  - tickers: (N,) array of ticker strings (order matches columns of returns)
  - meta:    JSON metadata string describing the scenario construction

Example:
  python data_prep.py --tickers AAPL MSFT NVDA AMZN META \
      --start 2018-01-01 --end 2026-03-01 \
      --scenario bootstrap --S 50 --out data/scenarios_bootstrap.npz

Scenario methods:
  - rolling:         last S daily returns
  - historical:      sample S distinct daily returns (no replacement)
  - bootstrap:       sample S daily returns with replacement
  - block_bootstrap: sample contiguous blocks (length L) until S is reached
  - gaussian:        fit mu, Omega to full return history and sample MVN returns
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np

try:
    import pandas as pd
except Exception as e:
    raise ImportError("This script requires pandas. pip install pandas") from e

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    # Optional fallback provider
    import pandas_datareader.data as pdr
except Exception:
    pdr = None


def download_adjclose_yfinance(
    tickers: List[str],
    start: str,
    end: str,
    interval: str = "1d",
    threads: bool = False,
    retries: int = 3,
    retry_sleep_s: float = 2.0,
) -> "pd.DataFrame":
    """Download adjusted close prices using yfinance (Yahoo Finance).

    Notes:
    - Yahoo may intermittently throttle or return empty/invalid JSON.
      We therefore retry a few times and default to threads=False to reduce burstiness.
    """
    if yf is None:
        raise ImportError("yfinance is not installed. pip install yfinance")

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=threads,
            )
            break
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(retry_sleep_s * attempt)
            else:
                raise
    else:
        raise last_err  # pragma: no cover

    # yfinance returns different shapes for single vs multi ticker
    if isinstance(df.columns, pd.MultiIndex):
        # Prefer "Close" if present after auto_adjust, else "Adj Close"
        if ("Close" in df.columns.get_level_values(1)):
            adj = df.xs("Close", axis=1, level=1, drop_level=True)
        else:
            adj = df.xs("Adj Close", axis=1, level=1, drop_level=True)
    else:
        # Single ticker, single column series
        adj = df.rename(columns={df.columns[0]: tickers[0]}) if len(tickers) == 1 else df

    # Ensure column order exactly matches tickers
    adj = adj.loc[:, tickers]
    return adj


def download_adjclose_stooq(tickers: List[str], start: str, end: str) -> "pd.DataFrame":
    """Download close prices using Stooq via pandas-datareader.

    Stooq uses symbols like 'AAPL.US' for US equities. For convenience, if a ticker
    has no suffix, we append '.US'.
    """
    if pdr is None:
        raise ImportError("pandas-datareader is required for provider=stooq. pip install pandas-datareader")

    def _to_stooq(sym: str) -> str:
        return sym if "." in sym else f"{sym}.US"

    stooq_syms = [_to_stooq(t) for t in tickers]
    frames = []
    for sym, orig in zip(stooq_syms, tickers):
        df = pdr.DataReader(sym, "stooq")
        # stooq returns descending dates; sort ascending
        df = df.sort_index()
        close = df[["Close"]].rename(columns={"Close": orig})
        frames.append(close)

    prices = pd.concat(frames, axis=1)
    prices = prices.loc[(prices.index >= pd.to_datetime(start)) & (prices.index < pd.to_datetime(end))]
    return prices


def log_returns(prices: "pd.DataFrame") -> "pd.DataFrame":
    """Compute log returns r_t = log P_t - log P_{t-1}."""
    lr = np.log(prices).diff()
    lr = lr.dropna(how="any")
    return lr


def make_scenarios(returns: np.ndarray, S: int, method: str, seed: int, block_len: int = 5) -> np.ndarray:
    """
    Build scenario matrix (S, N) from a time series of returns (T, N).
    """
    rng = np.random.default_rng(seed)
    T, N = returns.shape
    if S <= 0:
        raise ValueError("S must be positive.")
    if T < 2:
        raise ValueError("Not enough return observations to form scenarios.")

    method = method.lower()

    if method == "rolling":
        if T < S:
            raise ValueError(f"Not enough data for rolling window: T={T} < S={S}")
        return returns[-S:, :].copy()

    if method == "historical":
        if T < S:
            raise ValueError(f"Not enough data for historical sample without replacement: T={T} < S={S}")
        idx = rng.choice(T, size=S, replace=False)
        return returns[idx, :].copy()

    if method == "bootstrap":
        idx = rng.choice(T, size=S, replace=True)
        return returns[idx, :].copy()

    if method == "block_bootstrap":
        if block_len <= 0:
            raise ValueError("block_len must be positive.")
        # Sample blocks until length S reached
        out = np.zeros((S, N), dtype=float)
        filled = 0
        while filled < S:
            start_idx = int(rng.integers(0, max(1, T - block_len)))
            block = returns[start_idx : start_idx + block_len, :]
            take = min(block.shape[0], S - filled)
            out[filled : filled + take, :] = block[:take, :]
            filled += take
        return out

    if method == "gaussian":
        # Fit mu and Omega to entire return history and sample MVN
        mu = returns.mean(axis=0)
        centered = returns - mu[None, :]
        Omega = (centered.T @ centered) / float(T - 1)
        # Symmetrize for numerical stability
        Omega = 0.5 * (Omega + Omega.T)
        return rng.multivariate_normal(mean=mu, cov=Omega, size=S).astype(float)

    raise ValueError(f"Unknown scenario method: {method}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols (space separated).")
    ap.add_argument("--provider", default="yfinance", choices=["yfinance", "stooq"],
                    help="Data provider (default: yfinance). Use stooq if Yahoo blocks requests.")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD.")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD (exclusive-ish depending on provider).")
    ap.add_argument("--interval", default="1d", help="Data interval (default 1d).")
    ap.add_argument("--threads", action="store_true", help="Enable yfinance threading (can trigger throttling).")
    ap.add_argument("--retries", type=int, default=3, help="Retries for yfinance downloads.")
    ap.add_argument("--scenario", default="bootstrap",
                    choices=["rolling", "historical", "bootstrap", "block_bootstrap", "gaussian"],
                    help="Scenario construction method.")
    ap.add_argument("--S", type=int, required=True, help="Number of scenarios to output.")
    ap.add_argument("--seed", type=int, default=123, help="Random seed for scenario construction.")
    ap.add_argument("--block-len", type=int, default=5, help="Block length for block_bootstrap.")
    ap.add_argument("--out", required=True, help="Output .npz path.")
    ap.add_argument("--drop-na", action="store_true", help="Drop any date rows with NaNs (recommended).")
    ap.add_argument("--save-returns-csv", default=None, help="Optional path to save full return history as CSV.")

    args = ap.parse_args()
    tickers = args.tickers
    out_path = Path(args.out)

    if args.provider == "yfinance":
        prices = download_adjclose_yfinance(
            tickers,
            start=args.start,
            end=args.end,
            interval=args.interval,
            threads=bool(args.threads),
            retries=int(args.retries),
        )
        source_note = "yfinance(auto_adjust=True)"
    else:
        prices = download_adjclose_stooq(tickers, start=args.start, end=args.end)
        source_note = "stooq(pandas-datareader)"

    if args.drop_na:
        prices = prices.dropna(how="any")

    lr = log_returns(prices)
    if lr.shape[0] < 2:
        raise RuntimeError("After cleaning, not enough price history to compute returns.")

    if args.save_returns_csv:
        Path(args.save_returns_csv).parent.mkdir(parents=True, exist_ok=True)
        lr.to_csv(args.save_returns_csv, index=True)

    returns_hist = lr.to_numpy(dtype=float)  # (T, N)
    scen = make_scenarios(returns_hist, S=args.S, method=args.scenario, seed=args.seed, block_len=args.block_len)

    # Uniform scenario probabilities (you can change later)
    p_s = np.full(args.S, 1.0 / args.S, dtype=float)

    meta = {
        "tickers": tickers,
        "start": args.start,
        "end": args.end,
        "interval": args.interval,
        "scenario_method": args.scenario,
        "S": args.S,
        "seed": args.seed,
        "block_len": args.block_len,
        "source": source_note,
        "note": "returns are log returns; scenario matrix is (S,N) in ticker order",
        "T_available": int(returns_hist.shape[0]),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, returns=scen, p_s=p_s, tickers=np.array(tickers, dtype=object), meta=json.dumps(meta))
    print(f"Wrote {out_path} with returns shape {scen.shape} (S,N).")
    print("Tickers order:", tickers)


if __name__ == "__main__":
    main()
