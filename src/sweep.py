from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import re
from dataclasses import dataclass, fields, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import json



@dataclass
class SweepConfig:
    db: str = "../data/market.db"
    params: str = "../data/params_hybrid.npy"
    output_csv: str = "../data/sweep_results.csv"
    output_plot: str = "../data/sweep_results.png"

    N: int = 28
    p: int = 2
    k: int = 12
    alpha: float = 0.99
    C: float = 0.03
    spsa_iters: int = 50
    shots_train: int = 2048
    shots_eval: int = 8192

    init_seed: int = 0
    n_seeds: int = 100
    rolling_warm_start: bool = False  # if True, each seed updates the params file



@dataclass
class RunResult:
    seed: int
    selected_tickers: str          
    selected_indices: str          
    weights: str                   
    expected_return: Optional[float]
    empirical_cvar: Optional[float]
    exact_k_fraction: Optional[float]
    total_eval_shots: Optional[int]
    classical_expected_return: Optional[float]
    classical_empirical_cvar: Optional[float]
    return_gap: Optional[float]    
    best_outcome: Optional[int]
    best_outcome_energy: Optional[float]
    best_outcome_fraction: Optional[float]
    warm_started: bool
    status: str                    
    timestamp: str
    
    
def parse_output(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("SWEEP_JSON:"):
            try:
                return json.loads(line[len("SWEEP_JSON:"):])
            except json.JSONDecodeError:
                pass
    return {}

# def parse_output(stdout: str) -> dict:
#     def find_float(pattern: str) -> Optional[float]:
#         m = re.search(pattern, stdout)
#         return float(m.group(1)) if m else None

#     def find_int(pattern: str) -> Optional[int]:
#         m = re.search(pattern, stdout)
#         return int(m.group(1)) if m else None

#     def find_str(pattern: str) -> Optional[str]:
#         m = re.search(pattern, stdout)
#         return m.group(1).strip() if m else None

#     result = {}

#     # tickers and indices
#     result["selected_tickers"] = find_str(r"Selected tickers\s*:\s*(\[.*?\])")
#     result["selected_indices"] = find_str(r"Selected indices\s*:\s*(\[.*?\])")
#     result["weights"] = find_str(r"Weights\s*:\s*(\[.*?\])")

#     # quantum metrics
#     result["expected_return"] = find_float(
#         r"Expected return\s*:\s*([-\d.eE+]+)"
#     )
#     result["empirical_cvar"] = find_float(
#         r"Empirical CVaR\s*:\s*([-\d.eE+]+)"
#     )
#     result["exact_k_fraction"] = find_float(
#         r"Exact-k fraction\s*:\s*([\d.]+)"
#     )
#     result["total_eval_shots"] = find_int(
#         r"Total eval shots\s*:\s*(\d+)"
#     )
#     result["shot_fraction"] = find_float(
#         r"Shot fraction\s*:\s*([\d.]+)"
#     )

#     # classical baseline
#     result["classical_expected_return"] = find_float(
#         r"Upper bound.*?\n.*?Expected return\s*:\s*([-\d.eE+]+)"
#     )
#     result["classical_empirical_cvar"] = find_float(
#         r"Upper bound.*?\n.*?Expected return.*?\n.*?Empirical CVaR\s*:\s*([-\d.eE+]+)"
#     )
#     result["return_gap"] = find_float(
#         r"Return gap \(classical B - quantum\)\s*:\s*([-\d.eE+]+)"
#     )

#     # best outcome from distribution table (rank 1)
#     result["best_outcome"] = find_int(
#         r"^\s*1\s+(\d+)\s+\d+\s+\d+", 
#     )
#     result["best_outcome_energy"] = find_float(
#         r"^\s*1\s+\d+\s+\d+\s+\d+\s+[\d.]+\s+([-\d.eE+]+)",
#     )

#     return result


def run_single(
    cfg: SweepConfig,
    seed: int,
    warm_start: bool,
    save_params: bool,
    python_bin: str = "python3",
) -> RunResult:
    timestamp = datetime.now().isoformat(timespec="seconds")

    cmd = [
        python_bin, "main.py",
        "--db", cfg.db,
        "--mode", "hybrid",
        "--N", str(cfg.N),
        "--p", str(cfg.p),
        "--k", str(cfg.k),
        "--alpha", str(cfg.alpha),
        "--C", str(cfg.C),
        "--spsa-iters", str(cfg.spsa_iters),
        "--shots-train", str(cfg.shots_train),
        "--shots-eval", str(cfg.shots_eval),
        "--seed", str(seed),
    ]
    if warm_start:
        cmd += ["--warm-start", cfg.params]
    if save_params:
        cmd += ["--save-params", cfg.params]

    print(f"  cmd: {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  
        )
        stdout = proc.stdout
        stderr = proc.stderr

        if proc.returncode != 0:
            print(f"  [WARN] Non-zero return code: {proc.returncode}")
            print(f"  stderr: {stderr[:300]}")
            status = f"error_rc{proc.returncode}"
        else:
            status = "ok"

        # print live so you can watch progress
        print(stdout)

        parsed = parse_output(stdout)

    except subprocess.TimeoutExpired:
        parsed = {}
        status = "timeout"
        print(f"  [ERROR] Run timed out (seed={seed})")
    except Exception as e:
        parsed = {}
        status = f"exception: {e}"
        print(f"  [ERROR] {e}")

    return RunResult(
        seed=seed,
        selected_tickers=parsed.get("selected_tickers") or "",
        selected_indices=parsed.get("selected_indices") or "",
        weights=parsed.get("weights") or "",
        expected_return=parsed.get("expected_return"),
        empirical_cvar=parsed.get("empirical_cvar"),
        exact_k_fraction=parsed.get("exact_k_fraction"),
        total_eval_shots=parsed.get("total_eval_shots"),
        classical_expected_return=parsed.get("classical_expected_return"),
        classical_empirical_cvar=parsed.get("classical_empirical_cvar"),
        return_gap=parsed.get("return_gap"),
        best_outcome=parsed.get("best_outcome"),
        best_outcome_energy=parsed.get("best_outcome_energy"),
        best_outcome_fraction=parsed.get("shot_fraction"),
        warm_started=warm_start,
        status=status,
        timestamp=timestamp,
    )



def write_csv(results: list[RunResult], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in fields(RunResult)])
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"\nCSV written -> {out}  ({len(results)} rows)")



def plot_results(results: list[RunResult], path: str, cfg: SweepConfig) -> None:
    ok = [r for r in results if r.status == "ok" and r.expected_return is not None]
    if not ok:
        print("No successful results to plot.")
        return

    seeds = [r.seed for r in ok]
    returns = [r.expected_return for r in ok]          # already annualised %
    cvars = [r.empirical_cvar for r in ok]             # already %
    exact_k = [r.exact_k_fraction * 100 for r in ok]  # fraction -> %
    gaps = [r.return_gap if r.return_gap else 0 for r in ok]
    cl_returns = [r.classical_expected_return for r in ok if r.classical_expected_return]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"QAOA Hybrid Sweep  |  N={cfg.N}  k={cfg.k}  p={cfg.p}  "
        f"Seeds 1-{max(seeds)}",
        fontsize=13, fontweight="bold"
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # annualised return per seed 
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(seeds, returns, color="steelblue", alpha=0.7, label="Quantum")
    if any(c is not None for c in cl_returns):
        cl_valid = [c for c in cl_returns if c is not None]
        ax1.axhline(
            np.mean(cl_valid), color="crimson", linestyle="--",
            linewidth=1.5, label=f"Classical mean ({np.mean(cl_valid):.1f}%)"
        )
    ax1.set_title("Annualised Expected Return (%)")
    ax1.set_xlabel("Seed")
    ax1.set_ylabel("%")
    ax1.legend(fontsize=8)

    # empirical CVaR per seed 
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(seeds, cvars, color="darkorange", alpha=0.7)
    ax2.axhline(3.0, color="red", linestyle="--", linewidth=1.5, label="Budget 3%")
    ax2.set_title("Empirical CVaR (%) — worst 1% daily loss")
    ax2.set_xlabel("Seed")
    ax2.set_ylabel("%")
    ax2.legend(fontsize=8)

    # exact-k fraction per seed
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(seeds, exact_k, marker="o", markersize=4,
             color="seagreen", linewidth=1.5)
    ax3.set_title("Exact-k Fraction (%) — valid cardinality shots")
    ax3.set_xlabel("Seed")
    ax3.set_ylabel("%")
    ax3.set_ylim(0, 100)

    # return gap distribution 
    ax4 = fig.add_subplot(gs[1, 1])
    colors = ["steelblue" if g <= 0 else "crimson" for g in gaps]
    ax4.bar(seeds, gaps, color=colors, alpha=0.7)
    ax4.axhline(0, color="black", linewidth=1.0)
    ax4.set_title("Return Gap — Classical B minus Quantum (annualised %)\nRed = quantum beat classical")
    ax4.set_xlabel("Seed")
    ax4.set_ylabel("%")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved -> {out}")
    plt.close()



def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="""
Run main.py hybrid QAOA across multiple seeds and collect results.

Outputs:
  - results CSV for Tableau or plotting
  - optional matplotlib summary plot

Usage:
  python3 sweep.py
  python3 sweep.py --plot
  python3 sweep.py --N 28 --p 2 --k 12 --seeds 100 --plot
""")
    ap.add_argument("--db", default="../data/market.db")
    ap.add_argument("--params", default="../data/params_hybrid.npy")
    ap.add_argument("--output-csv", default="../data/sweep_results.csv")
    ap.add_argument("--output-plot", default="../data/sweep_results.png")
    ap.add_argument("--N", type=int, default=28)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.99)
    ap.add_argument("--C", type=float, default=0.03)
    ap.add_argument("--spsa-iters", type=int, default=80)
    ap.add_argument("--shots-train", type=int, default=2048)
    ap.add_argument("--shots-eval", type=int, default=16384)
    ap.add_argument("--init-seed", type=int, default=420,
                    help="Seed for initial warm-start generation run.")
    ap.add_argument("--seeds", type=int, default=100,
                    help="Number of seeds to sweep (1 through N).")
    ap.add_argument("--rolling", action="store_true",
                    help="Each seed updates the params file for the next run.")
    ap.add_argument("--plot", action="store_true",
                    help="Generate matplotlib summary plot.")
    ap.add_argument("--python", default="python3",
                    help="Python interpreter to use.")
    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    cfg = SweepConfig(
        db=args.db,
        params=args.params,
        output_csv=args.output_csv,
        output_plot=args.output_plot,
        N=args.N,
        p=args.p,
        k=args.k,
        alpha=args.alpha,
        C=args.C,
        spsa_iters=args.spsa_iters,
        shots_train=args.shots_train,
        shots_eval=args.shots_eval,
        init_seed=args.init_seed,
        n_seeds=args.seeds,
        rolling_warm_start=args.rolling,
    )

    results: list[RunResult] = []

    # initial run
    print(f"\n{'='*60}")
    print(f"INITIAL RUN  seed={cfg.init_seed}  (generates warm-start)")
    print(f"{'='*60}")
    init_result = run_single(
        cfg, seed=cfg.init_seed,
        warm_start=False, save_params=True,
        python_bin=args.python,
    )
    results.append(init_result)
    write_csv(results, cfg.output_csv)  # write after each run

    for i, seed in enumerate(range(1, cfg.n_seeds + 1), start=1):
        print(f"\n{'='*60}")
        print(f"SEED {seed}  ({i}/{cfg.n_seeds})")
        print(f"{'='*60}")

        result = run_single(
            cfg, seed=seed,
            warm_start=True,
            save_params=cfg.rolling_warm_start,
            python_bin=args.python,
        )
        results.append(result)

        # write CSV after every run so partial results are never lost
        write_csv(results, cfg.output_csv)

        # running summary
        ok_so_far = [r for r in results if r.status == "ok" and r.expected_return is not None]
        if ok_so_far:
            mean_ret = np.mean([r.expected_return for r in ok_so_far]) 
            mean_ekf = np.mean([r.exact_k_fraction for r in ok_so_far
                                if r.exact_k_fraction is not None])
            print(f"\n  Running mean annualised return : {mean_ret:.2f}%")
            print(f"  Running mean exact-k fraction  : {mean_ekf:.1f}%")
            print(f"  Completed {i}/{cfg.n_seeds}  "
                  f"({sum(1 for r in results if r.status == 'ok')} ok, "
                  f"{sum(1 for r in results if r.status != 'ok')} failed)")

    print(f"\n{'='*60}")
    print("SWEEP COMPLETE")
    print(f"{'='*60}")
    ok = [r for r in results if r.status == "ok" and r.expected_return is not None]
    print(f"Successful runs : {len(ok)} / {len(results)}")
    if ok:
        rets = [r.expected_return for r in ok]
        ekfs = [r.exact_k_fraction for r in ok if r.exact_k_fraction is not None]
        gaps = [r.return_gap for r in ok if r.return_gap is not None]
        print(f"Return (ann%)   : mean={np.mean(rets):.2f}  "
              f"min={np.min(rets):.2f}  max={np.max(rets):.2f}  std={np.std(rets):.2f}")
        print(f"Exact-k (%)     : mean={np.mean(ekfs):.1f}  "
              f"min={np.min(ekfs):.1f}  max={np.max(ekfs):.1f}")
        if gaps:
            print(f"Return gap      : mean={np.mean(gaps):.4f}%  "
                  f"quantum beat classical in "
                  f"{sum(1 for g in gaps if g < 0)}/{len(gaps)} runs")

    write_csv(results, cfg.output_csv)

    if args.plot:
        plot_results(results, cfg.output_plot, cfg=cfg)


if __name__ == "__main__":
    main()
    #TODO: do post processing to get rid of all seeds that resulted in ridiculous weight values