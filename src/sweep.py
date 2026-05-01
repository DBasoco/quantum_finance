from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass, fields, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


SCENARIO_CHOICES = [
    "rolling", "historical", "bootstrap",
    "block_bootstrap", "gaussian",
    "clustered", "clustered_blocks",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SweepConfig:
    db: str = "../data/market.db"
    params_dir: str = "../data/params"
    output_csv: str = "../data/sweep_results.csv"
    output_plot_dir: str = "../data/plots"

    N: int = 28
    S: int = 100
    p: int = 2
    k_min: int = 3
    k_max: int = -1
    alpha: float = 0.99
    C: float = 0.03
    spsa_iters: int = 50
    shots_train: int = 2048
    shots_eval: int = 8192

    init_seed: int = 0
    # Seed range for this job — enables parallel HPC execution.
    # Each SLURM array task sets a different seed_start/seed_end.
    seed_start: int = 1
    seed_end: int = 100
    rolling_warm_start: bool = False

    # Scenario construction
    scenario: str = "block_bootstrap"
    block_len: int = 5

    # Asset sampling
    sampling: str = "random"
    pca_components: int = 10


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    seed: int           # human-readable seed (seed_start..seed_end) — used for plots and CSV
    effective_seed: int # actual seed passed to main.py (seed * 1000 + k) — for reproducibility
    k: int
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
    sampling: str
    scenario: str


# ---------------------------------------------------------------------------
# JSON parser
# ---------------------------------------------------------------------------

def parse_output(stdout: str) -> dict:
    """Read the SWEEP_JSON line emitted by main.py."""
    for line in stdout.splitlines():
        if line.startswith("SWEEP_JSON:"):
            try:
                return json.loads(line[len("SWEEP_JSON:"):])
            except json.JSONDecodeError:
                pass
    return {}


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(
    cfg: SweepConfig,
    seed: int,           # human-readable seed stored in CSV and used for plots
    k: int,
    warm_start: bool,
    save_params: bool,
    python_bin: str = "python3",
    effective_seed: Optional[int] = None,  # seed passed to main.py; defaults to seed
) -> RunResult:
    timestamp = datetime.now().isoformat(timespec="seconds")
    params_path = str(Path(cfg.params_dir) / f"params_k{k}.npy")
    # Use effective_seed for main.py if provided, otherwise use seed directly
    _main_seed = effective_seed if effective_seed is not None else seed

    cmd = [
        python_bin, "main.py",
        "--db", cfg.db,
        "--mode", "hybrid",
        "--N", str(cfg.N),
        "--S", str(cfg.S),
        "--p", str(cfg.p),
        "--k", str(k),
        "--alpha", str(cfg.alpha),
        "--C", str(cfg.C),
        "--spsa-iters", str(cfg.spsa_iters),
        "--shots-train", str(cfg.shots_train),
        "--shots-eval", str(cfg.shots_eval),
        "--seed", str(_main_seed),
        "--scenario", cfg.scenario,
        "--block-len", str(cfg.block_len),
        "--sampling", cfg.sampling,
        "--pca-components", str(cfg.pca_components),
    ]
    if warm_start:
        cmd += ["--warm-start", params_path]
    if save_params:
        cmd += ["--save-params", params_path]

    print(f"  cmd: {' '.join(cmd)}")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        stdout = proc.stdout
        stderr = proc.stderr

        if proc.returncode != 0:
            print(f"  [WARN] rc={proc.returncode}")
            print(f"  stderr: {stderr[:300]}")
            status = f"error_rc{proc.returncode}"
        else:
            status = "ok"

        print(stdout)
        parsed = parse_output(stdout)

    except subprocess.TimeoutExpired:
        parsed = {}
        status = "timeout"
        print(f"  [ERROR] Timed out (seed={seed} k={k})")
    except Exception as e:
        parsed = {}
        status = f"exception: {e}"
        print(f"  [ERROR] {e}")

    return RunResult(
        seed=seed,
        effective_seed=_main_seed,
        k=k,
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
        sampling=parsed.get("sampling") or cfg.sampling,
        scenario=parsed.get("scenario") or cfg.scenario,
    )


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def write_csv(results: list[RunResult], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in fields(RunResult)])
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"CSV written -> {out}  ({len(results)} rows)")


# ---------------------------------------------------------------------------
# Merge helper — combine CSVs from multiple parallel jobs
# ---------------------------------------------------------------------------

def merge_csvs(input_pattern: str, output_path: str) -> None:
    """
    Merge multiple per-job CSV files into a single sorted CSV.

    Example:
        merge_csvs("../data/sweep_results_s*.csv",
                   "../data/sweep_results_merged.csv")
    """
    import glob
    files = sorted(glob.glob(input_pattern))
    if not files:
        print(f"No files matched pattern: {input_pattern}")
        return
    print(f"Merging {len(files)} files ...")
    all_rows: list[RunResult] = []
    for fpath in files:
        with open(fpath, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Cast numeric fields back to correct types
                typed = {}
                for field in fields(RunResult):
                    val = row.get(field.name, "")
                    if val == "" or val is None:
                        typed[field.name] = None
                    elif field.type in ("Optional[float]", "float"):
                        try:
                            typed[field.name] = float(val)
                        except (ValueError, TypeError):
                            typed[field.name] = None
                    elif field.type in ("Optional[int]", "int"):
                        try:
                            typed[field.name] = int(float(val))
                        except (ValueError, TypeError):
                            typed[field.name] = None
                    elif field.type == "bool":
                        typed[field.name] = str(val).lower() in ("true", "1", "yes")
                    else:
                        typed[field.name] = str(val)
                all_rows.append(RunResult(**typed))
    # Sort by k then seed
    all_rows.sort(key=lambda r: (r.k, r.seed))
    write_csv(all_rows, output_path)
    print(f"Merged {len(all_rows)} total rows -> {output_path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_results_for_k(
    results: list[RunResult],
    k: int,
    plot_dir: str,
    cfg: SweepConfig,
) -> None:
    ok = [r for r in results if r.k == k and r.status == "ok" and r.expected_return is not None]
    if not ok:
        print(f"  No successful results for k={k}, skipping plot.")
        return

    seeds = [r.seed for r in ok]
    returns = [r.expected_return for r in ok]
    cvars = [r.empirical_cvar for r in ok]
    exact_k = [r.exact_k_fraction * 100 for r in ok]
    gaps = [r.return_gap if r.return_gap is not None else 0.0 for r in ok]
    cl_returns = [r.classical_expected_return for r in ok if r.classical_expected_return]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"QAOA Hybrid Sweep  |  N={cfg.N}  k={k}  p={cfg.p}  "
        f"alpha={cfg.alpha}  C={cfg.C}  sampling={cfg.sampling}  "
        f"scenario={cfg.scenario}  Seeds {min(seeds)}-{max(seeds)}",
        fontsize=11, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(seeds, returns, color="steelblue", alpha=0.7, label="Quantum")
    if cl_returns:
        cl_mean = np.mean(cl_returns)
        ax1.axhline(cl_mean, color="crimson", linestyle="--", linewidth=1.5,
                    label=f"Classical mean ({cl_mean:.3f}%)")
    ax1.set_title("Daily Expected Return (%)")
    ax1.set_xlabel("Seed"); ax1.set_ylabel("%")
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(seeds, cvars, color="darkorange", alpha=0.7)
    ax2.axhline(cfg.C * 100, color="red", linestyle="--", linewidth=1.5,
                label=f"Budget {cfg.C*100:.0f}%")
    ax2.set_title("Empirical CVaR (%) — worst 1% daily loss")
    ax2.set_xlabel("Seed"); ax2.set_ylabel("%")
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(seeds, exact_k, marker="o", markersize=4, color="seagreen", linewidth=1.5)
    ax3.set_title(f"Exact-k={k} Fraction (%) — valid cardinality shots")
    ax3.set_xlabel("Seed"); ax3.set_ylabel("%")
    ax3.set_ylim(0, 100)

    ax4 = fig.add_subplot(gs[1, 1])
    colors = ["crimson" if g < 0 else "steelblue" for g in gaps]
    ax4.bar(seeds, gaps, color=colors, alpha=0.7)
    ax4.axhline(0, color="black", linewidth=1.0)
    ax4.set_title("Return Gap — Classical minus Quantum (%)\nRed = quantum beat classical")
    ax4.set_xlabel("Seed"); ax4.set_ylabel("%")

    out = Path(plot_dir) / f"sweep_k{k}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved -> {out}")
    plt.close()


def plot_k_comparison(
    results: list[RunResult],
    plot_dir: str,
    cfg: SweepConfig,
    k_values: list[int],
    seed_start: int,
    seed_end: int,
) -> None:
    k_means_ret, k_means_cvar, k_means_ekf, k_std_ret, k_beat_classical = [], [], [], [], []

    for k in k_values:
        ok = [r for r in results if r.k == k and r.status == "ok" and r.expected_return is not None]
        if not ok:
            for lst in [k_means_ret, k_means_cvar, k_means_ekf, k_std_ret, k_beat_classical]:
                lst.append(np.nan)
            continue
        rets = [r.expected_return for r in ok]
        cvars = [r.empirical_cvar for r in ok]
        ekfs = [r.exact_k_fraction * 100 for r in ok if r.exact_k_fraction is not None]
        gaps = [r.return_gap for r in ok if r.return_gap is not None]
        k_means_ret.append(float(np.mean(rets)))
        k_means_cvar.append(float(np.mean(cvars)))
        k_means_ekf.append(float(np.mean(ekfs)) if ekfs else np.nan)
        k_std_ret.append(float(np.std(rets)))
        k_beat_classical.append(
            100.0 * sum(1 for g in gaps if g < 0) / len(gaps) if gaps else np.nan
        )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"QAOA Hybrid Sweep — k Comparison  |  N={cfg.N}  p={cfg.p}  "
        f"sampling={cfg.sampling}  scenario={cfg.scenario}  "
        f"Seeds {seed_start}-{seed_end}",
        fontsize=11, fontweight="bold",
    )

    ax = axes[0, 0]
    ax.errorbar(k_values, k_means_ret, yerr=k_std_ret,
                marker="o", color="steelblue", linewidth=2, capsize=4)
    ax.set_title("Mean Daily Return (%) vs k")
    ax.set_xlabel("k (cardinality)"); ax.set_ylabel("%")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(k_values, k_means_cvar, marker="o", color="darkorange", linewidth=2)
    ax.axhline(cfg.C * 100, color="red", linestyle="--", linewidth=1.5,
               label=f"Budget {cfg.C*100:.0f}%")
    ax.set_title("Mean Empirical CVaR (%) vs k")
    ax.set_xlabel("k (cardinality)"); ax.set_ylabel("%")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(k_values, k_means_ekf, marker="o", color="seagreen", linewidth=2)
    ax.set_title("Mean Exact-k Fraction (%) vs k")
    ax.set_xlabel("k (cardinality)"); ax.set_ylabel("%")
    ax.set_ylim(0, 100); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.bar(k_values, k_beat_classical, color="crimson", alpha=0.7)
    ax.set_title("% Runs Where Quantum Beat Classical vs k")
    ax.set_xlabel("k (cardinality)"); ax.set_ylabel("% of seeds")
    ax.set_ylim(0, 100); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(plot_dir) / "sweep_k_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"k-comparison plot saved -> {out}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="""
Sweep main.py hybrid QAOA over multiple seeds AND cardinality values k.

Seed range design:
  Use --seed-start and --seed-end to define this job's seed range.
  Each parallel HPC job gets a different range and writes its own CSV.
  Merge results afterwards with --merge.

  Example: 4 jobs covering seeds 1-100 (25 each):
    Job 0: --seed-start 1  --seed-end 25  --output-csv results_s1-25.csv
    Job 1: --seed-start 26 --seed-end 50  --output-csv results_s26-50.csv
    Job 2: --seed-start 51 --seed-end 75  --output-csv results_s51-75.csv
    Job 3: --seed-start 76 --seed-end 100 --output-csv results_s76-100.csv

  Then merge:
    python3 sweep.py --merge "../data/sweep_results_s*.csv" \\
                     --merge-output "../data/sweep_results_merged.csv"

Warm-start note:
  All jobs share the same --params-dir. The job whose seed range includes
  --init-seed generates the params files; all others wait and reuse them.
  With SLURM array jobs use a dependency or stagger start times.

Asset sampling methods (--sampling):
  random     - uniform random draw (default)
  clustered  - k-means on PCA-reduced correlation structure

Scenario methods (--scenario):
  block_bootstrap   - contiguous blocks, uniform p_s  [default]
  clustered_blocks  - k-means regime clustering with blocks
  clustered         - k-means, one day per cluster
  bootstrap / historical / rolling / gaussian
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument("--db", default="../data/market.db")
    ap.add_argument("--params-dir", default="../data/params",
                    help="Shared directory for warm-start .npy files. "
                         "All parallel jobs must point to the same location.")
    ap.add_argument("--output-csv", default=None,
                    help="CSV output path. Defaults to "
                         "sweep_results_s{seed_start}-{seed_end}.csv")
    ap.add_argument("--output-plot-dir", default="../data/plots")

    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--S", type=int, default=80)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--k-min", type=int, default=3)
    ap.add_argument("--k-max", type=int, default=-1,
                    help="Maximum cardinality to sweep. Defaults to N.")
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--C", type=float, default=0.03)
    ap.add_argument("--spsa-iters", type=int, default=50)
    ap.add_argument("--shots-train", type=int, default=2048)
    ap.add_argument("--shots-eval", type=int, default=8192)

    # Seed control
    ap.add_argument("--init-seed", type=int, default=0,
                    help="Seed used to generate warm-start params for each k. "
                         "Should be identical across all parallel jobs.")
    ap.add_argument("--seed-start", type=int, default=1,
                    help="First seed in this job's range (inclusive). Default: 1.")
    ap.add_argument("--seed-end", type=int, default=100,
                    help="Last seed in this job's range (inclusive). Default: 100.")

    ap.add_argument("--rolling", action="store_true",
                    help="Each seed updates the params file for the next seed.")
    ap.add_argument("--plot", action="store_true",
                    help="Generate per-k plots and k-comparison plot.")
    ap.add_argument("--python", default="python3")

    # Scenario
    ap.add_argument("--scenario", default="block_bootstrap", choices=SCENARIO_CHOICES,
                    help="Scenario construction method (default: block_bootstrap).")
    ap.add_argument("--block-len", type=int, default=5)

    # Asset sampling
    ap.add_argument("--sampling", default="random", choices=["random", "clustered"],
                    help="Asset sampling method (default: random).")
    ap.add_argument("--pca-components", type=int, default=10)

    # Merge mode — combine CSVs from parallel jobs
    ap.add_argument("--merge", default=None, metavar="GLOB_PATTERN",
                    help="Glob pattern matching CSV files to merge "
                         "(e.g. '../data/sweep_results_s*.csv'). "
                         "When set, skips the sweep and only merges.")
    ap.add_argument("--merge-output", default="../data/sweep_results_merged.csv",
                    help="Output path for merged CSV (default: sweep_results_merged.csv).")

    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    # --- Merge-only mode ---
    if args.merge:
        merge_csvs(args.merge, args.merge_output)
        return

    # --- Validate seed range ---
    if args.seed_start > args.seed_end:
        ap.error(f"--seed-start ({args.seed_start}) must be <= --seed-end ({args.seed_end})")

    k_max = args.k_max if args.k_max > 0 else args.N
    k_values = list(range(args.k_min, k_max + 1))

    # Default CSV name encodes the seed range so parallel jobs never collide
    if args.output_csv is None:
        output_csv = f"../data/sweep_results_s{args.seed_start}-{args.seed_end}.csv"
    else:
        output_csv = args.output_csv

    # Default plot dir also encodes seed range to avoid parallel write conflicts
    output_plot_dir = args.output_plot_dir
    if output_plot_dir == "../data/plots" and (args.seed_start != 1 or args.seed_end != 100):
        output_plot_dir = f"../data/plots_s{args.seed_start}-{args.seed_end}"

    Path(args.params_dir).mkdir(parents=True, exist_ok=True)

    cfg = SweepConfig(
        db=args.db,
        params_dir=args.params_dir,
        output_csv=output_csv,
        output_plot_dir=output_plot_dir,
        N=args.N, S=args.S, p=args.p,
        k_min=args.k_min, k_max=k_max,
        alpha=args.alpha, C=args.C,
        spsa_iters=args.spsa_iters,
        shots_train=args.shots_train,
        shots_eval=args.shots_eval,
        init_seed=args.init_seed,
        seed_start=args.seed_start,
        seed_end=args.seed_end,
        rolling_warm_start=args.rolling,
        scenario=args.scenario,
        block_len=args.block_len,
        sampling=args.sampling,
        pca_components=args.pca_components,
    )

    n_seeds_this_job = cfg.seed_end - cfg.seed_start + 1
    total_runs = len(k_values) * n_seeds_this_job
    # The init run only fires if this job's range includes the init seed
    # or if no params file exists yet for a given k.
    generates_warmstart = (cfg.seed_start <= cfg.init_seed <= cfg.seed_end)

    print(f"\nSweep plan : {len(k_values)} k values × {n_seeds_this_job} seeds "
          f"= {total_runs} runs")
    print(f"Seed range : {cfg.seed_start} – {cfg.seed_end}")
    print(f"Init seed  : {cfg.init_seed}  "
          f"({'this job generates warm-start' if generates_warmstart else 'reusing existing warm-start'})")
    print(f"k values   : {k_values}")
    print(f"Sampling   : {cfg.sampling}   Scenario : {cfg.scenario}")
    print(f"Output CSV : {output_csv}")
    print()

    all_results: list[RunResult] = []
    run_count = 0

    for k in k_values:
        print(f"\n{'#'*65}")
        print(f"# k = {k}  ({k_values.index(k)+1}/{len(k_values)})")
        print(f"{'#'*65}")

        params_path = Path(cfg.params_dir) / f"params_k{k}.npy"

        # Generate warm-start if:
        #   (a) this job's seed range includes the init seed, OR
        #   (b) the params file doesn't exist yet (first job to reach this k)
        if generates_warmstart or not params_path.exists():
            print(f"\n{'='*60}")
            print(f"WARM-START RUN  k={k}  seed={cfg.init_seed}")
            print(f"{'='*60}")
            init_result = run_single(
                cfg, seed=cfg.init_seed, k=k,
                warm_start=False, save_params=True,
                python_bin=args.python,
                effective_seed=cfg.init_seed,
            )
            all_results.append(init_result)
            run_count += 1
            write_csv(all_results, cfg.output_csv)
        else:
            print(f"\n  [k={k}] Warm-start params exist at {params_path}, skipping init run.")

        # Seed sweep for this job's range
        seeds_this_k = list(range(cfg.seed_start, cfg.seed_end + 1))
        for i, seed in enumerate(seeds_this_k, start=1):
            print(f"\n{'='*60}")
            print(f"k={k}  SEED {seed}  ({i}/{n_seeds_this_job})  "
                  f"[overall {run_count+1}/{total_runs}]")
            print(f"{'='*60}")

            # Effective seed passed to main.py includes k to ensure different
            # asset draws across k values for the same seed number.
            # The human-readable seed (used in CSV and plots) stays as-is.
            effective_seed = seed * 1000 + k

            result = run_single(
                cfg, seed=seed, k=k,
                warm_start=True,
                save_params=cfg.rolling_warm_start,
                python_bin=args.python,
                effective_seed=effective_seed,
            )
            all_results.append(result)
            run_count += 1
            write_csv(all_results, cfg.output_csv)

            ok_k = [r for r in all_results
                    if r.k == k and r.status == "ok" and r.expected_return is not None]
            if ok_k:
                mean_ret = np.mean([r.expected_return for r in ok_k])
                mean_ekf = np.mean([r.exact_k_fraction for r in ok_k
                                    if r.exact_k_fraction is not None])
                print(f"  [k={k}] mean return={mean_ret:.4f}%  "
                      f"mean exact-k={mean_ekf*100:.1f}%  "
                      f"runs this job={len(ok_k)}")

        if args.plot:
            plot_results_for_k(all_results, k=k,
                               plot_dir=cfg.output_plot_dir, cfg=cfg)

    # --- Final summary ---
    print(f"\n{'='*65}")
    print("SWEEP COMPLETE")
    print(f"{'='*65}")
    print(f"Seed range this job : {cfg.seed_start} – {cfg.seed_end}")
    ok_all = [r for r in all_results if r.status == "ok" and r.expected_return is not None]
    print(f"Successful runs : {len(ok_all)} / {len(all_results)}")

    for k in k_values:
        ok_k = [r for r in ok_all if r.k == k]
        if not ok_k:
            print(f"  k={k:3d} : no successful runs")
            continue
        rets = [r.expected_return for r in ok_k]
        ekfs = [r.exact_k_fraction * 100 for r in ok_k if r.exact_k_fraction is not None]
        gaps = [r.return_gap for r in ok_k if r.return_gap is not None]
        beat = sum(1 for g in gaps if g < 0)
        print(
            f"  k={k:3d} : "
            f"ret mean={np.mean(rets):7.4f}%  std={np.std(rets):.4f}%  "
            f"min={np.min(rets):.4f}%  max={np.max(rets):.4f}%  |  "
            f"exact-k={np.mean(ekfs):.1f}%  |  "
            f"beat classical={beat}/{len(gaps)}"
        )

    write_csv(all_results, cfg.output_csv)

    if args.plot:
        plot_k_comparison(
            all_results, plot_dir=cfg.output_plot_dir,
            cfg=cfg, k_values=k_values,
            seed_start=cfg.seed_start, seed_end=cfg.seed_end,
        )

    print(f"\nTo merge results from multiple jobs:")
    print(f"  python3 sweep.py --merge '../data/sweep_results_s*.csv' "
          f"--merge-output '../data/sweep_results_merged.csv'")


if __name__ == "__main__":
    main()
