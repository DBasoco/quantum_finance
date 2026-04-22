from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import yfinance as yf



def _create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS universe (
            ticker      TEXT PRIMARY KEY,
            added_at    TEXT NOT NULL,
            note        TEXT
        );

        CREATE TABLE IF NOT EXISTS prices (
            ticker      TEXT NOT NULL,
            date        TEXT NOT NULL,
            adj_close   REAL NOT NULL,
            PRIMARY KEY (ticker, date),
            FOREIGN KEY (ticker) REFERENCES universe(ticker)
        );

        CREATE TABLE IF NOT EXISTS returns (
            ticker      TEXT NOT NULL,
            date        TEXT NOT NULL,
            log_return  REAL NOT NULL,
            PRIMARY KEY (ticker, date),
            FOREIGN KEY (ticker) REFERENCES universe(ticker)
        );

        CREATE TABLE IF NOT EXISTS ingest_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            tickers     TEXT NOT NULL,
            start_date  TEXT NOT NULL,
            end_date    TEXT NOT NULL,
            source      TEXT NOT NULL,
            interval    TEXT NOT NULL,
            n_prices    INTEGER,
            n_returns   INTEGER,
            ingested_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_prices_ticker  ON prices(ticker);
        CREATE INDEX IF NOT EXISTS idx_prices_date    ON prices(date);
        CREATE INDEX IF NOT EXISTS idx_returns_ticker ON returns(ticker);
        CREATE INDEX IF NOT EXISTS idx_returns_date   ON returns(date);
    """)
    conn.commit()


def open_db(path: str) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    _create_tables(conn)
    return conn


def upsert_universe(conn: sqlite3.Connection, tickers: List[str]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        "INSERT INTO universe(ticker, added_at) VALUES(?,?) ON CONFLICT(ticker) DO NOTHING",
        [(t, now) for t in tickers],
    )
    conn.commit()


def upsert_prices(conn: sqlite3.Connection, prices_df: "pd.DataFrame") -> int:
    rows = []
    for date_idx, row in prices_df.iterrows():
        date_str = str(date_idx.date())
        for ticker in prices_df.columns:
            val = row[ticker]
            if pd.isna(val):
                continue
            rows.append((ticker, date_str, float(val)))
    conn.executemany(
        "INSERT OR REPLACE INTO prices(ticker, date, adj_close) VALUES(?,?,?)",
        rows,
    )
    conn.commit()
    return len(rows)


def upsert_returns(conn: sqlite3.Connection, returns_df: "pd.DataFrame") -> int:
    rows = []
    for date_idx, row in returns_df.iterrows():
        date_str = str(date_idx.date())
        for ticker in returns_df.columns:
            val = row[ticker]
            if pd.isna(val):
                continue
            rows.append((ticker, date_str, float(val)))
    conn.executemany(
        "INSERT OR REPLACE INTO returns(ticker, date, log_return) VALUES(?,?,?)",
        rows,
    )
    conn.commit()
    return len(rows)


def log_ingest(
    conn: sqlite3.Connection,
    tickers: List[str],
    start: str,
    end: str,
    source: str,
    interval: str,
    n_prices: int,
    n_returns: int,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO ingest_log"
        "(tickers,start_date,end_date,source,interval,n_prices,n_returns,ingested_at) "
        "VALUES(?,?,?,?,?,?,?,?)",
        (json.dumps(tickers), start, end, source, interval, n_prices, n_returns, now),
    )
    conn.commit()







def query_available_tickers(conn: sqlite3.Connection, min_rows: int = 20) -> List[str]:
    cur = conn.execute(
        "SELECT ticker, COUNT(*) AS n FROM returns "
        "GROUP BY ticker HAVING n >= ? ORDER BY ticker",
        (min_rows,),
    )
    return [row[0] for row in cur.fetchall()]


def query_returns_for_tickers(
    conn: sqlite3.Connection,
    tickers: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> "pd.DataFrame":
    placeholders = ",".join("?" * len(tickers))
    params: list = list(tickers)
    where_extra = ""
    if start:
        where_extra += " AND date >= ?"
        params.append(start)
    if end:
        where_extra += " AND date < ?"
        params.append(end)

    query = (
        f"SELECT ticker, date, log_return FROM returns "
        f"WHERE ticker IN ({placeholders}){where_extra} ORDER BY date, ticker"
    )
    cur = conn.execute(query, params)
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=tickers)

    df = pd.DataFrame(rows, columns=["ticker", "date", "log_return"])
    pivot = df.pivot(index="date", columns="ticker", values="log_return")
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.sort_index()

    # NaN check
    for t in tickers:
        if t not in pivot.columns:
            pivot[t] = np.nan
    pivot = pivot[tickers]
    pivot = pivot.dropna(how="any")
    return pivot




def download_adjclose_yfinance(
    tickers: List[str],
    start: str,
    end: str,
    interval: str = "1d",
    threads: bool = False,
    retries: int = 3,
    retry_sleep_s: float = 2.0,
) -> "pd.DataFrame":
    last_err: Optional[Exception] = None
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

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(1):
            adj = df.xs("Close", axis=1, level=1, drop_level=True)
        else:
            adj = df.xs("Adj Close", axis=1, level=1, drop_level=True)
    else:
        adj = df.rename(columns={df.columns[0]: tickers[0]}) if len(tickers) == 1 else df

    adj = adj.loc[:, tickers]
    return adj


def download_adjclose_stooq(tickers: List[str], start: str, end: str) -> "pd.DataFrame":
    def _to_stooq(sym: str) -> str:
        return sym if "." in sym else f"{sym}.US"

    stooq_syms = [_to_stooq(t) for t in tickers]
    frames = []
    for sym, orig in zip(stooq_syms, tickers):
        df = pdr.DataReader(sym, "stooq")
        df = df.sort_index()
        close = df[["Close"]].rename(columns={"Close": orig})
        frames.append(close)

    prices = pd.concat(frames, axis=1)
    prices = prices.loc[
        (prices.index >= pd.to_datetime(start)) & (prices.index < pd.to_datetime(end))
    ]
    return prices


def log_returns(prices: "pd.DataFrame") -> "pd.DataFrame":
    # Compute log returns r_t = log(P_t) - log(P_{t-1})
    lr = np.log(prices).diff()
    lr = lr.dropna(how="any")
    return lr



def make_scenarios(
    returns: np.ndarray,
    S: int,
    method: str,
    seed: int,
    block_len: int = 5,
) -> np.ndarray:
    # (S x N)
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
            raise ValueError(
                f"Not enough data for historical sample without replacement: T={T} < S={S}"
            )
        idx = rng.choice(T, size=S, replace=False)
        return returns[idx, :].copy()

    if method == "bootstrap":
        idx = rng.choice(T, size=S, replace=True)
        return returns[idx, :].copy()

    if method == "block_bootstrap":
        if block_len <= 0:
            raise ValueError("block_len must be positive.")
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
        mu = returns.mean(axis=0)
        centered = returns - mu[None, :]
        Omega = (centered.T @ centered) / float(T - 1)
        Omega = 0.5 * (Omega + Omega.T)
        return rng.multivariate_normal(mean=mu, cov=Omega, size=S).astype(float)

    raise ValueError(f"Unknown scenario method: {method!r}")


# I'm only human after all
def inspect_db(conn: sqlite3.Connection) -> None:
    print("=== Database Inspection ===\n")

    cur = conn.execute("SELECT COUNT(*) FROM universe")
    n_tickers = cur.fetchone()[0]
    print(f"Universe: {n_tickers} tickers")
    for row in conn.execute("SELECT ticker, added_at FROM universe ORDER BY ticker"):
        print(f"  {row[0]}  (added {row[1][:10]})")

    print()
    rows = conn.execute(
        "SELECT ticker, COUNT(*) AS n, MIN(date) AS first, MAX(date) AS last "
        "FROM prices GROUP BY ticker ORDER BY ticker"
    ).fetchall()
    if rows:
        print(f"Prices: {sum(r[1] for r in rows)} total rows")
        for r in rows:
            print(f"  {r[0]:10s}  {r[1]:5d} rows  {r[2]} – {r[3]}")

    print()
    rows = conn.execute(
        "SELECT ticker, COUNT(*) AS n, MIN(date) AS first, MAX(date) AS last "
        "FROM returns GROUP BY ticker ORDER BY ticker"
    ).fetchall()
    if rows:
        print(f"Returns: {sum(r[1] for r in rows)} total rows")
        for r in rows:
            print(f"  {r[0]:10s}  {r[1]:5d} rows  {r[2]} – {r[3]}")

    print()
    rows = conn.execute(
        "SELECT id, tickers, start_date, end_date, source, ingested_at "
        "FROM ingest_log ORDER BY id DESC LIMIT 5"
    ).fetchall()
    if rows:
        print("Recent ingest runs (latest 5):")
        for r in rows:
            print(f"  [{r[0]}] {r[5][:19]}  tickers={r[1]}  {r[2]}–{r[3]}  source={r[4]}")



# CSV for tabluea
def export_returns_csv(conn: sqlite3.Connection, csv_path: str) -> None:
    rows = conn.execute(
        "SELECT ticker, date, log_return FROM returns ORDER BY date, ticker"
    ).fetchall()
    if not rows:
        print("No returns data found in database.")
        return
    import pandas as pd  # local import for optional usage
    df = pd.DataFrame(rows, columns=["ticker", "date", "log_return"])
    pivot = df.pivot(index="date", columns="ticker", values="log_return")
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.sort_index()
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(csv_path)
    print(f"Exported returns to {csv_path}  shape={pivot.shape}")



def main():
    ap = argparse.ArgumentParser(
        description="""
Download market data, store in SQLite, and optionally emit .npz scenario files.

Primary persistent storage is a SQLite database that main.py can query at runtime to
randomly sample asset subsets and construct scenario matrices on the fly.

SQLite schema:
  universe   - ticker metadata and universe membership
  prices     - adjusted-close price history (ticker x date)
  returns    - daily log-return history (ticker x date)
  ingest_log - metadata about each ingest run

Usage examples:
  # Ingest data into SQLite (primary usage)
  python data_prep.py --tickers AAPL MSFT NVDA AMZN META \\
      --start 2018-01-01 --end 2026-03-01 \\
      --db data/market.db

  # Also emit a scenario .npz file (backwards-compatible; optional)
  python data_prep.py --tickers AAPL MSFT NVDA AMZN META \\
      --start 2018-01-01 --end 2026-03-01 \\
      --db data/market.db \\
      --scenario bootstrap --S 50 --out data/scenarios_bootstrap.npz

  # Inspect the database
  python data_prep.py --db data/market.db --inspect

  # Export returns to CSV for manual inspection
  python data_prep.py --db data/market.db --export-returns-csv data/returns.csv
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument("--db", default="data/market.db",
                    help="Path to SQLite database (default: data/market.db).")

    # Inspect / export-only modes (no download needed)
    ap.add_argument("--inspect", action="store_true",
                    help="Print database summary and exit.")
    ap.add_argument("--export-returns-csv", default=None, metavar="PATH",
                    help="Export return history to CSV and exit.")

    # Download arguments
    ap.add_argument("--tickers", nargs="+", default=None,
                    help="Ticker symbols (space separated).")
    ap.add_argument("--provider", default="yfinance", choices=["yfinance", "stooq"],
                    help="Data provider.")
    ap.add_argument("--start", default=None, help="Start date YYYY-MM-DD.")
    ap.add_argument("--end", default=None, help="End date YYYY-MM-DD.")
    ap.add_argument("--interval", default="1d", help="Data interval (default 1d).")
    ap.add_argument("--threads", action="store_true",
                    help="Enable yfinance threading (can trigger throttling).")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--drop-na", action="store_true",
                    help="Drop date rows with any NaN before storing.")

    # Optional .npz export
    ap.add_argument("--scenario", default="bootstrap",
                    choices=["rolling", "historical", "bootstrap", "block_bootstrap", "gaussian"],
                    help="Scenario method for optional .npz output.")
    ap.add_argument("--S", type=int, default=None,
                    help="Number of scenarios for optional .npz output.")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--block-len", type=int, default=5)
    ap.add_argument("--out", default=None,
                    help="Output .npz path (requires --S).")
    ap.add_argument("--save-returns-csv", default=None,
                    help="Save downloaded return history as CSV.")

    args = ap.parse_args()
    conn = open_db(args.db)

    if args.inspect:
        inspect_db(conn)
        conn.close()
        return

    if args.export_returns_csv:
        export_returns_csv(conn, args.export_returns_csv)
        conn.close()
        return

    # --- Download + store ---
    if not args.tickers:
        ap.error("--tickers is required for download mode.")
    if not args.start or not args.end:
        ap.error("--start and --end are required for download mode.")

    tickers = args.tickers
    print(f"Downloading data for: {tickers}  ({args.start} – {args.end})  provider={args.provider}")

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

    # if args.drop_na:
    #     prices = prices.dropna(how="any")

    # lr = log_returns(prices)
    lr = np.log(prices).diff() 
    if lr.shape[0] < 2:
        raise RuntimeError("After cleaning, not enough price history to compute returns.")

    print(f"  {prices.shape[0]} price rows → {lr.shape[0]} return rows per ticker.")

    upsert_universe(conn, tickers)
    n_prices = upsert_prices(conn, prices)
    n_returns = upsert_returns(conn, lr)
    log_ingest(conn, tickers, args.start, args.end, source_note, args.interval, n_prices, n_returns)
    print(f"Stored in {args.db}: {n_prices} price rows, {n_returns} return rows.")

    if args.save_returns_csv:
        Path(args.save_returns_csv).parent.mkdir(parents=True, exist_ok=True)
        lr.to_csv(args.save_returns_csv, index=True)
        print(f"Saved return CSV → {args.save_returns_csv}")

    # Optional .npz scenario export
    if args.S is not None:
        if args.out is None:
            print("Warning: --S provided but --out not specified; skipping .npz export.")
        else:
            returns_hist = lr.to_numpy(dtype=float)
            scen = make_scenarios(
                returns_hist, S=args.S, method=args.scenario,
                seed=args.seed, block_len=args.block_len,
            )
            p_s = np.full(args.S, 1.0 / args.S, dtype=float)
            meta = {
                "tickers": tickers, "start": args.start, "end": args.end,
                "interval": args.interval, "scenario_method": args.scenario,
                "S": args.S, "seed": args.seed, "block_len": args.block_len,
                "source": source_note,
                "note": "returns are log returns; scenario matrix is (S,N) in ticker order",
                "T_available": int(returns_hist.shape[0]),
            }
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_path, returns=scen, p_s=p_s,
                tickers=np.array(tickers, dtype=object), meta=json.dumps(meta),
            )
            print(f"Wrote {out_path}  shape={scen.shape}")

    conn.close()


if __name__ == "__main__":
    main()