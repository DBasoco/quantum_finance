from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cvxpy as cp

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2


try:
    from data_prep import (
        open_db,
        query_available_tickers,
        query_returns_for_tickers,
        make_scenarios as _dp_make_scenarios,
    )
    _HAS_DATA_PREP = True
except ImportError:
    _HAS_DATA_PREP = False



def make_scenarios(
    returns: np.ndarray, S: int, method: str, seed: int, block_len: int = 5,
) -> np.ndarray:
    if _HAS_DATA_PREP:
        return _dp_make_scenarios(returns, S=S, method=method, seed=seed, block_len=block_len)
    rng = np.random.default_rng(seed)
    T, N = returns.shape
    m = method.lower()
    if m == "rolling":
        return returns[-S:, :].copy()
    if m == "historical":
        return returns[rng.choice(T, S, replace=False), :].copy()
    if m == "bootstrap":
        return returns[rng.choice(T, S, replace=True), :].copy()
    if m == "block_bootstrap":
        out = np.zeros((S, N), dtype=float)
        filled = 0
        while filled < S:
            si = int(rng.integers(0, max(1, T - block_len)))
            blk = returns[si : si + block_len, :]
            take = min(blk.shape[0], S - filled)
            out[filled : filled + take, :] = blk[:take, :]
            filled += take
        return out
    if m == "gaussian":
        mu = returns.mean(0)
        c = returns - mu
        Om = (c.T @ c) / float(T - 1)
        return rng.multivariate_normal(mu, 0.5 * (Om + Om.T), size=S).astype(float)
    raise ValueError(f"Unknown scenario method: {method!r}")



@dataclass(frozen=True)
class EngineConfig:
    # Problem dimensions
    N: int = 16
    S: int = 50
    k: int = 6

    # CVaR parameters
    alpha: float = 0.99
    C: float = 0.03

    # Selector QUBO weights
    beta_return: float = 1.0
    gamma_risk: float = 1.0

    # Full-QUBO penalty weights
    lambda_V: float = 10.0
    lambda_B: float = 10.0
    lambda_K: float = 10.0
    lambda_L: float = 10.0
    lambda_C: float = 10.0
    lambda_T: float = 10.0

    # QAOA
    p: int = 2
    shots_train: int = 1024
    shots_eval: int = 8192
    gamma_bounds: Tuple[float, float] = (0.0, 2.0 * math.pi)
    beta_bounds: Tuple[float, float] = (0.0, math.pi)

    # Training CVaR aggregation (hybrid only).
    # 1.0 = mean energy; smaller = focus on lowest-energy shots.
    train_cvar_alpha: float = 0.25

    # SPSA
    spsa_iters: int = 80
    spsa_a: float = 0.2
    spsa_c: float = 0.1
    spsa_A: float = 10.0
    spsa_alpha: float = 0.602
    spsa_gamma: float = 0.101
    spsa_eval_every: int = 5

    seed: int = 125

    # Synthetic fallback
    ret_mu: float = 0.0005
    ret_sigma: float = 0.02

    # Full-QUBO bit widths
    Bw: int = 2
    Bt: int = 2
    Bv: int = 2
    Bxi: int = 2
    Beta: int = 2

    # Fixed-point encoding ranges
    w_max: float = 1.0
    v_max: float = 0.20
    xi_max: float = 0.20
    eta_max: float = 0.20
    auto_t_range: bool = True
    t_min: float = -0.10
    t_max: float = 0.10

    max_unique_subsets_to_score: int = 50
    DRAW_CIRCUIT: bool = False


@dataclass
class BitLayout:
    n_total: int
    x_idx: List[int]
    w_idx: List[List[int]]
    t_idx: List[int]
    v_idx: List[List[int]]
    xi_idx: List[int]
    eta_idx: List[List[int]]
    w0: np.ndarray
    w_delta: np.ndarray
    t0: float
    t_delta: np.ndarray
    v0: np.ndarray
    v_delta: np.ndarray
    xi0: float
    xi_delta: np.ndarray
    eta0: np.ndarray
    eta_delta: np.ndarray


def fixed_point_deltas(vmax: float, B: int) -> np.ndarray:
    if B <= 0:
        return np.array([], dtype=float)
    step = vmax / (2**B - 1)
    return np.array([step * (2**b) for b in range(B)], dtype=float)



def sample_assets_from_db(
    db_path: str,
    N: int,
    S: int,
    scenario_method: str,
    scenario_seed: int,
    start: Optional[str] = None,
    end: Optional[str] = None,
    block_len: int = 5,
    min_history: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if not _HAS_DATA_PREP:
        raise ImportError("data_prep.py is not importable.")
    if min_history is None:
        min_history = max(S + 10, 30)

    conn = open_db(db_path)
    try:
        all_tickers = query_available_tickers(conn, min_rows=min_history)
        if len(all_tickers) < N:
            raise RuntimeError(
                f"Database has only {len(all_tickers)} tickers with >= {min_history} "
                f"return rows, but N={N} was requested.  Reduce --N or ingest more data."
            )
        rng = np.random.default_rng(scenario_seed)
        chosen = sorted(rng.choice(all_tickers, size=N, replace=False).tolist())
        ret_df = query_returns_for_tickers(conn, chosen, start=start, end=end)
        if ret_df.shape[0] < min_history:
            raise RuntimeError(
                f"Only {ret_df.shape[0]} aligned return rows after date filtering "
                f"(need >= {min_history})."
            )
        returns_hist = ret_df.to_numpy(dtype=float)
        scen = make_scenarios(returns_hist, S=S, method=scenario_method,
                              seed=scenario_seed, block_len=block_len)
        p_s = np.full(S, 1.0 / S, dtype=float)
        return scen, p_s, chosen
    finally:
        conn.close()



def _empirical_cvar(losses: np.ndarray, alpha: float) -> float:
    if losses.size == 0:
        return float("nan")
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    return float(tail.mean()) if tail.size > 0 else var


def solve_cvar_weights(
    returns: np.ndarray, p_s: np.ndarray, alpha: float, C: float,
) -> Dict[str, object]:
    S, N = returns.shape
    mu = (p_s[:, None] * returns).sum(axis=0)
    w = cp.Variable(N, nonneg=True)
    t = cp.Variable()
    v = cp.Variable(S, nonneg=True)
    losses = -returns @ w
    prob = cp.Problem(
        cp.Maximize(mu @ w),
        [v >= losses - t, cp.sum(w) == 1.0,
         t + (1.0 / (1.0 - alpha)) * (p_s @ v) <= C],
    )
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)
    status = prob.status
    if status not in ("optimal", "optimal_inaccurate"):
        return {"status": status}
    w_val = np.clip(np.array(w.value, dtype=float).reshape(-1), 0.0, None)
    w_val /= max(w_val.sum(), 1e-12)
    port_ret = returns @ w_val
    return {
        "status": status,
        "weights": w_val,
        "expected_return": float((p_s * port_ret).sum()),
        "empirical_cvar": _empirical_cvar(-port_ret, alpha),
    }



class PortfolioQAOAEngine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.sampler = AerSamplerV2(seed=cfg.seed)
        self.tickers: Optional[List[str]] = None
        # Initialise with synthetic data; override with set_scenarios().
        self.returns = self.rng.normal(cfg.ret_mu, cfg.ret_sigma, (cfg.S, cfg.N)).astype(float)
        self.p_s = np.full(cfg.S, 1.0 / cfg.S, dtype=float)
        self.mu, self.Omega = self._estimate_mu_omega(self.returns, self.p_s)
        self._update_t_range()


    @staticmethod
    def _estimate_mu_omega(r: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu = (p[:, None] * r).sum(0)
        c = r - mu[None, :]
        Om = (p[:, None, None] * (c[:, :, None] * c[:, None, :])).sum(0)
        return mu, 0.5 * (Om + Om.T)

    def _update_t_range(self) -> None:
        cfg = self.cfg
        if cfg.auto_t_range:
            w_eq = np.full(cfg.N, 1.0 / cfg.N)
            losses = -self.returns @ w_eq
            pad = 0.05 * (losses.max() - losses.min() + 1e-12)
            self.t_min = float(losses.min() - pad)
            self.t_max = float(losses.max() + pad)
        else:
            self.t_min, self.t_max = cfg.t_min, cfg.t_max

    def set_scenarios(
        self, returns: np.ndarray,
        p_s: Optional[np.ndarray] = None,
        tickers: Optional[List[str]] = None,
    ) -> None:
        S_new, N_new = returns.shape
        cfg = self.cfg
        if S_new != cfg.S or N_new != cfg.N:
            raise ValueError(
                f"Scenario shape ({S_new},{N_new}) does not match config (S={cfg.S}, N={cfg.N})."
            )
        self.returns = np.asarray(returns, dtype=float)
        if p_s is None:
            self.p_s = np.full(cfg.S, 1.0 / cfg.S, dtype=float)
        else:
            p_arr = np.asarray(p_s, dtype=float).reshape(-1)
            self.p_s = p_arr / p_arr.sum()
        self.mu, self.Omega = self._estimate_mu_omega(self.returns, self.p_s)
        self._update_t_range()
        if tickers is not None:
            self.tickers = list(tickers)



    def build_selector_qubo(self) -> Tuple[np.ndarray, float]:
        cfg = self.cfg
        N, k = cfg.N, cfg.k
        lamK, beta, gamma = cfg.lambda_K, cfg.beta_return, cfg.gamma_risk
        Q = np.zeros((N, N), dtype=float)
        for i in range(N):
            Q[i, i] = lamK * (1.0 - 2.0 * k) - beta * self.mu[i] + (gamma / (k * k)) * self.Omega[i, i]
        for i in range(N):
            for j in range(i + 1, N):
                Q[i, j] = 2.0 * lamK + (2.0 * gamma / (k * k)) * self.Omega[i, j]
        return Q, lamK * (k * k)

    def make_full_bit_layout(self) -> BitLayout:
        cfg = self.cfg
        idx = 0
        x_idx = list(range(idx, idx + cfg.N)); idx += cfg.N
        w_idx: List[List[int]] = []
        for _ in range(cfg.N):
            w_idx.append(list(range(idx, idx + cfg.Bw))); idx += cfg.Bw
        t_idx = list(range(idx, idx + cfg.Bt)); idx += cfg.Bt
        v_idx: List[List[int]] = []
        for _ in range(cfg.S):
            v_idx.append(list(range(idx, idx + cfg.Bv))); idx += cfg.Bv
        xi_idx = list(range(idx, idx + cfg.Bxi)); idx += cfg.Bxi
        eta_idx: List[List[int]] = []
        for _ in range(cfg.S):
            eta_idx.append(list(range(idx, idx + cfg.Beta))); idx += cfg.Beta
        w0 = np.zeros(cfg.N, dtype=float)
        w_delta = np.stack([fixed_point_deltas(cfg.w_max, cfg.Bw)] * cfg.N, axis=0)
        t0 = float(self.t_min)
        t_delta = fixed_point_deltas(float(self.t_max - self.t_min), cfg.Bt)
        v0 = np.zeros(cfg.S, dtype=float)
        v_delta = fixed_point_deltas(cfg.v_max, cfg.Bv)
        xi0 = 0.0
        xi_delta = fixed_point_deltas(cfg.xi_max, cfg.Bxi)
        eta0 = np.zeros(cfg.S, dtype=float)
        eta_delta = fixed_point_deltas(cfg.eta_max, cfg.Beta)
        return BitLayout(
            n_total=idx, x_idx=x_idx, w_idx=w_idx, t_idx=t_idx,
            v_idx=v_idx, xi_idx=xi_idx, eta_idx=eta_idx,
            w0=w0, w_delta=w_delta, t0=t0, t_delta=t_delta,
            v0=v0, v_delta=v_delta, xi0=xi0, xi_delta=xi_delta,
            eta0=eta0, eta_delta=eta_delta,
        )

    def build_full_qubo(self) -> Tuple[np.ndarray, float, BitLayout]:
        cfg = self.cfg
        layout = self.make_full_bit_layout()
        M = layout.n_total
        Q = np.zeros((M, M), dtype=float)
        kappa = 0.0

        def add_linear(i: int, a: float) -> None:
            Q[i, i] += a

        def add_quadratic(i: int, j: int, a: float) -> None:
            if i == j:
                Q[i, i] += a
            else:
                if i > j: i, j = j, i
                Q[i, j] += a

        def add_affine_square(lam: float, g0: float, coeffs: Dict[int, float]) -> None:
            nonlocal kappa
            kappa += lam * g0 * g0
            for i, gi in coeffs.items():
                add_linear(i, lam * (2.0 * g0 * gi + gi * gi))
            il = list(coeffs.keys())
            for ai in range(len(il)):
                i = il[ai]; gi = coeffs[i]
                for aj in range(ai + 1, len(il)):
                    j = il[aj]; gj = coeffs[j]
                    add_quadratic(i, j, 2.0 * lam * gi * gj)

        # Return term
        for i in range(cfg.N):
            kappa += -self.mu[i] * layout.w0[i]
            for b, bi in enumerate(layout.w_idx[i]):
                add_linear(bi, -self.mu[i] * layout.w_delta[i, b])

        # Volatility penalty
        lamV = cfg.lambda_V
        for i in range(cfg.N):
            for j in range(i, cfg.N):
                coef = self.Omega[i, j] * (2.0 if j > i else 1.0)
                if abs(coef) < 1e-15: continue
                kappa += lamV * coef * layout.w0[i] * layout.w0[j]
                if layout.w0[i] != 0.0:
                    for b, bj in enumerate(layout.w_idx[j]):
                        add_linear(bj, lamV * coef * layout.w0[i] * layout.w_delta[j, b])
                if layout.w0[j] != 0.0:
                    for b, bi in enumerate(layout.w_idx[i]):
                        add_linear(bi, lamV * coef * layout.w0[j] * layout.w_delta[i, b])
                for b, bi in enumerate(layout.w_idx[i]):
                    di = layout.w_delta[i, b]
                    for bp, bj in enumerate(layout.w_idx[j]):
                        add_quadratic(bi, bj, lamV * coef * di * layout.w_delta[j, bp])

        # Budget
        gB0 = float(layout.w0.sum() - 1.0)
        coeffs_B: Dict[int, float] = {}
        for i in range(cfg.N):
            for b, bi in enumerate(layout.w_idx[i]):
                coeffs_B[bi] = coeffs_B.get(bi, 0.0) + float(layout.w_delta[i, b])
        add_affine_square(cfg.lambda_B, gB0, coeffs_B)

        # Cardinality
        add_affine_square(cfg.lambda_K, float(-cfg.k), {i: 1.0 for i in layout.x_idx})

        # Link 
        lamL = cfg.lambda_L
        for i in range(cfg.N):
            x_i = layout.x_idx[i]
            kappa += lamL * layout.w0[i]
            for b, bw in enumerate(layout.w_idx[i]):
                add_linear(bw, lamL * layout.w_delta[i, b])
            add_linear(x_i, -lamL * layout.w0[i])
            for b, bw in enumerate(layout.w_idx[i]):
                add_quadratic(x_i, bw, -lamL * layout.w_delta[i, b])

        # CVaR constraint
        a = self.p_s / (1.0 - cfg.alpha)
        gC0 = float(layout.t0 + float((a * layout.v0).sum()) + layout.xi0 - cfg.C)
        coeffs_C: Dict[int, float] = {}
        for b, bt in enumerate(layout.t_idx):
            coeffs_C[bt] = coeffs_C.get(bt, 0.0) + float(layout.t_delta[b])
        for s in range(cfg.S):
            for b, bv in enumerate(layout.v_idx[s]):
                coeffs_C[bv] = coeffs_C.get(bv, 0.0) + float(a[s] * layout.v_delta[b])
        for b, bxi in enumerate(layout.xi_idx):
            coeffs_C[bxi] = coeffs_C.get(bxi, 0.0) + float(layout.xi_delta[b])
        add_affine_square(cfg.lambda_C, gC0, coeffs_C)

        # Tail excess
        lamT = cfg.lambda_T
        for s in range(cfg.S):
            r_s = self.returns[s, :]
            gT0 = float(layout.v0[s] + np.dot(r_s, layout.w0) + layout.t0 - layout.eta0[s])
            coeffs_T: Dict[int, float] = {}
            for b, bv in enumerate(layout.v_idx[s]):
                coeffs_T[bv] = coeffs_T.get(bv, 0.0) + float(layout.v_delta[b])
            for b, bt in enumerate(layout.t_idx):
                coeffs_T[bt] = coeffs_T.get(bt, 0.0) + float(layout.t_delta[b])
            for b, be in enumerate(layout.eta_idx[s]):
                coeffs_T[be] = coeffs_T.get(be, 0.0) - float(layout.eta_delta[b])
            for i in range(cfg.N):
                for b, bw in enumerate(layout.w_idx[i]):
                    coeffs_T[bw] = coeffs_T.get(bw, 0.0) + float(r_s[i] * layout.w_delta[i, b])
            add_affine_square(lamT, gT0, coeffs_T)

        return Q, float(kappa), layout



    @staticmethod
    def qubo_to_ising(
        Q_upper: np.ndarray, kappa: float,
    ) -> Tuple[np.ndarray, Dict[Tuple[int, int], float], float]:
        n = Q_upper.shape[0]
        a = np.diag(Q_upper).copy()
        offset = float(kappa + 0.5 * a.sum())
        for i in range(n):
            for j in range(i + 1, n):
                offset += float(Q_upper[i, j] / 4.0)
        h = -0.5 * a
        for i in range(n):
            s = 0.0
            for j in range(n):
                if i == j: continue
                ii, jj = (i, j) if i < j else (j, i)
                s += float(Q_upper[ii, jj]) / 4.0
            h[i] -= s
        J: Dict[Tuple[int, int], float] = {}
        for i in range(n):
            for j in range(i + 1, n):
                val = float(Q_upper[i, j] / 4.0)
                if abs(val) > 1e-15:
                    J[(i, j)] = val
        return h, J, offset

    @staticmethod
    def build_qaoa_circuit_from_ising(
        n_qubits: int, h: np.ndarray,
        J: Dict[Tuple[int, int], float], p: int,
    ) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector]:
        gammas = ParameterVector("gamma", p)
        betas = ParameterVector("beta", p)
        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits))
        for layer in range(p):
            for i in range(n_qubits):
                hi = float(h[i])
                if abs(hi) > 1e-15:
                    qc.rz(2.0 * hi * gammas[layer], i)
            for (i, j), Jij in J.items():
                if abs(Jij) > 1e-15:
                    qc.rzz(2.0 * float(Jij) * gammas[layer], i, j)
            for i in range(n_qubits):
                qc.rx(2.0 * betas[layer], i)
        qc.measure_all()
        return qc, gammas, betas


    @staticmethod
    def int_to_bits(x: int, n: int) -> np.ndarray:
        return np.array([(x >> i) & 1 for i in range(n)], dtype=np.int8)

    @staticmethod
    def precompute_qubo_terms(
        Q_upper: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        diag = np.diag(Q_upper).astype(float).copy()
        iu, ju = np.triu_indices_from(Q_upper, k=1)
        vals = Q_upper[iu, ju].astype(float)
        mask = np.abs(vals) > 1e-15
        return diag, iu[mask], ju[mask], vals[mask]

    @staticmethod
    def energy_from_qubo_terms(
        diag: np.ndarray, iu: np.ndarray, ju: np.ndarray,
        vals: np.ndarray, kappa: float, z_bits: np.ndarray,
    ) -> float:
        zb = z_bits.astype(np.int8, copy=False)
        e = float(kappa + np.dot(diag, zb.astype(float)))
        if vals.size:
            e += float(np.dot(vals, (zb[iu] & zb[ju]).astype(float)))
        return e


    def _convert_counts(self, counts_raw) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for k, v in counts_raw.items():
            out[int(k, 2) if isinstance(k, str) else int(k)] = int(v)
        return out

    def sample_bitstrings(
        self, circuit: QuantumCircuit,
        gammas: ParameterVector, betas: ParameterVector,
        params: np.ndarray, shots: int,
    ) -> Dict[int, int]:
        return self.sample_bitstrings_batch(circuit, gammas, betas, [params], shots=shots)[0]

    def sample_bitstrings_batch(
        self, circuit: QuantumCircuit,
        gammas: ParameterVector, betas: ParameterVector,
        params_batch: List[np.ndarray], shots: int,
    ) -> List[Dict[int, int]]:
        p = self.cfg.p
        pubs = []
        for params in params_batch:
            params = np.asarray(params, dtype=float)
            param_values = params[:p].tolist() + params[p:].tolist()
            if circuit.num_parameters != len(param_values):
                raise ValueError(
                    f"Parameter mismatch: circuit expects {circuit.num_parameters}, "
                    f"got {len(param_values)}."
                )
            pubs.append((circuit, param_values))
        job = self.sampler.run(pubs, shots=shots)
        result = job.result()
        return [
            self._convert_counts(result[i].data.meas.get_counts())
            for i in range(len(result))
        ]


    def aggregate_energy_from_counts(
        self,
        counts: Dict[int, int],
        n: int,
        diag: np.ndarray,
        iu: np.ndarray,
        ju: np.ndarray,
        vals: np.ndarray,
        kappa: float,
        alpha: float,
    ) -> float:
        if not counts:
            return float("inf")
        alpha = float(min(max(alpha, 0.0), 1.0))
        ec: List[Tuple[float, int]] = []
        total_shots = 0
        for outcome, c in counts.items():
            z = self.int_to_bits(int(outcome), n)
            e = self.energy_from_qubo_terms(diag, iu, ju, vals, kappa, z)
            ec.append((float(e), int(c)))
            total_shots += int(c)
        if total_shots == 0:
            return float("inf")
        if alpha >= 1.0 - 1e-12:
            return sum(e * c for e, c in ec) / total_shots
        target = max(alpha * total_shots, 1.0)
        ec.sort(key=lambda t: t[0])
        used = 0.0; total = 0.0
        for e, c in ec:
            take = min(float(c), target - used)
            if take > 0.0:
                total += e * take; used += take
            if used >= target - 1e-12:
                break
        return (total / used) if used > 0 else float("inf")


    def spsa_optimize_params(
        self,
        objective_fn,
        dim: int,
        iters: int,
        shots: int,
        init: Optional[np.ndarray] = None,
        objective_batch_fn=None,
    ) -> Tuple[np.ndarray, List[float]]:
        cfg = self.cfg
        p = cfg.p
        assert dim == 2 * p

        if init is None:
            theta = np.concatenate([
                self.rng.uniform(cfg.gamma_bounds[0], cfg.gamma_bounds[1], size=p),
                self.rng.uniform(cfg.beta_bounds[0], cfg.beta_bounds[1], size=p),
            ]).astype(float)
        else:
            theta = np.asarray(init, dtype=float).copy()
            if theta.size != dim:
                raise ValueError(f"Warm-start shape {theta.size} != dim={dim}.")

        def project(th: np.ndarray) -> np.ndarray:
            th = th.copy()
            th[:p] = np.mod(th[:p], cfg.gamma_bounds[1])
            th[p:] = np.mod(th[p:], cfg.beta_bounds[1])
            return th

        theta = project(theta)
        best_theta = theta.copy()
        best_val = float("inf")
        hist: List[float] = []

        for it in range(iters):
            ak = cfg.spsa_a / ((it + 1.0 + cfg.spsa_A) ** cfg.spsa_alpha)
            ck = cfg.spsa_c / ((it + 1.0) ** cfg.spsa_gamma)
            delta = self.rng.choice([-1.0, 1.0], size=dim)
            tp = project(theta + ck * delta)
            tm = project(theta - ck * delta)
            eval_now = (it % cfg.spsa_eval_every) == 0

            if objective_batch_fn is not None:
                batch = [tp, tm] + ([theta] if eval_now else [])
                values = objective_batch_fn(batch, shots)
                fp, fm = float(values[0]), float(values[1])
                fc = float(values[2]) if eval_now else 0.5 * (fp + fm)
            else:
                fp = float(objective_fn(tp, shots))
                fm = float(objective_fn(tm, shots))
                fc = float(objective_fn(theta, shots)) if eval_now else 0.5 * (fp + fm)

            ghat = ((fp - fm) / (2.0 * ck)) * delta
            theta = project(theta - ak * ghat)
            hist.append(float(fc))
            if fc < best_val:
                best_val = float(fc)
                best_theta = theta.copy()

        return best_theta, hist



    def solve_cvar_weights_cvxpy(self, subset: np.ndarray) -> Dict[str, object]:
        cfg = self.cfg
        sel = np.asarray(subset, dtype=int)
        if sel.size == 0:
            return {"status": "empty_subset"}
        r_sub = self.returns[:, sel]
        mu_sub = self.mu[sel]
        w = cp.Variable(sel.size, nonneg=True)
        t = cp.Variable()
        v = cp.Variable(cfg.S, nonneg=True)
        losses = -r_sub @ w
        prob = cp.Problem(
            cp.Maximize(mu_sub @ w),
            [v >= losses - t, cp.sum(w) == 1.0,
             t + (1.0 / (1.0 - cfg.alpha)) * (self.p_s @ v) <= cfg.C],
        )
        try:
            prob.solve(solver=cp.ECOS, verbose=False)
        except Exception:
            prob.solve(solver=cp.SCS, verbose=False)
        status = prob.status
        if status not in ("optimal", "optimal_inaccurate"):
            return {"status": status}
        w_val = np.clip(np.array(w.value, dtype=float).reshape(-1), 0.0, None)
        port_losses = -(r_sub @ w_val)
        return {
            "status": status,
            "weights": w_val,
            "expected_return": float(mu_sub @ w_val),
            "empirical_cvar": _empirical_cvar(port_losses, cfg.alpha),
        }


    @staticmethod
    def _decode_affine(bits: np.ndarray, offset: float, deltas: np.ndarray) -> float:
        if deltas.size == 0:
            return float(offset)
        return float(offset + np.dot(deltas.astype(float), bits.astype(float)))

    def decode_full_solution(self, z: np.ndarray, layout: BitLayout) -> Dict[str, object]:
        cfg = self.cfg
        x = z[layout.x_idx].astype(int)
        w = np.array([
            self._decode_affine(z[layout.w_idx[i]], float(layout.w0[i]), layout.w_delta[i])
            for i in range(cfg.N)
        ])
        t = self._decode_affine(z[layout.t_idx], layout.t0, layout.t_delta)
        v = np.array([
            self._decode_affine(z[layout.v_idx[s]], float(layout.v0[s]), layout.v_delta)
            for s in range(cfg.S)
        ])
        xi = self._decode_affine(z[layout.xi_idx], layout.xi0, layout.xi_delta)
        eta = np.array([
            self._decode_affine(z[layout.eta_idx[s]], float(layout.eta0[s]), layout.eta_delta)
            for s in range(cfg.S)
        ])
        return {
            "x": x, "w": w, "t": float(t), "v": v,
            "xi": float(xi), "eta": eta,
            "sum_w": float(w.sum()), "sum_x": int(x.sum()),
        }







    def run_hybrid_selector(
        self, warm_start: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        cfg = self.cfg
        Q, kappa = self.build_selector_qubo()
        scale = float(np.max(np.abs(Q)) + 1e-12)
        Qs = Q / scale; kappas = kappa / scale
        h, J, offset = self.qubo_to_ising(Qs, kappas)
        n = cfg.N
        qc, gammas, betas = self.build_qaoa_circuit_from_ising(n, h, J, cfg.p)
        diag, iu, ju, vals = self.precompute_qubo_terms(Qs)

        if cfg.DRAW_CIRCUIT:
            print(qc.draw())

        def objective_batch(theta_batch: List[np.ndarray], shots: int) -> List[float]:
            counts_batch = self.sample_bitstrings_batch(qc, gammas, betas, theta_batch, shots)
            return [
                self.aggregate_energy_from_counts(
                    cnt, n, diag, iu, ju, vals, kappas, cfg.train_cvar_alpha)
                for cnt in counts_batch
            ]

        def objective(theta: np.ndarray, shots: int) -> float:
            return objective_batch([theta], shots)[0]

        best_theta, hist = self.spsa_optimize_params(
            objective_fn=objective,
            objective_batch_fn=objective_batch,
            dim=2 * cfg.p, iters=cfg.spsa_iters, shots=cfg.shots_train,
            init=warm_start,
        )

        counts_eval = self.sample_bitstrings(qc, gammas, betas, best_theta, cfg.shots_eval)
        total_eval = int(sum(counts_eval.values()))

        candidates: List[Tuple[int, np.ndarray, int, float]] = []
        exact_k_shots = 0
        for outcome, c in counts_eval.items():
            xb = self.int_to_bits(int(outcome), n)
            card = int(xb.sum())
            e = self.energy_from_qubo_terms(diag, iu, ju, vals, kappas, xb)
            candidates.append((int(outcome), xb, int(c), float(e)))
            if card == cfg.k:
                exact_k_shots += int(c)

        candidates.sort(key=lambda d: (0 if int(d[1].sum()) == cfg.k else 1, d[3], -d[2]))

        seen: set = set()
        best_port = None
        best_ret = -float("inf")
        scored = 0
        for outcome_int, xb, c, e in candidates:
            if outcome_int in seen: continue
            seen.add(outcome_int)
            subset = np.where(xb == 1)[0]
            if subset.size != cfg.k: continue
            scored += 1
            if scored > cfg.max_unique_subsets_to_score: break
            sol = self.solve_cvar_weights_cvxpy(subset)
            if sol.get("status") not in ("optimal", "optimal_inaccurate"): continue
            exp_ret = float(sol["expected_return"])
            if exp_ret > best_ret:
                best_ret = exp_ret
                best_port = {
                    "subset_bits": xb,
                    "subset_idx": subset,
                    "subset_tickers": (
                        [str(self.tickers[i]) for i in subset]
                        if self.tickers is not None else None
                    ),
                    "selector_energy_scaled": float(e * scale),
                    "selector_energy_internal": float(e),
                    "count": int(c),
                    "fraction": float(c) / total_eval if total_eval else 0.0,
                    "weights": sol["weights"],
                    "expected_return": exp_ret,
                    "empirical_cvar": float(sol["empirical_cvar"]),
                    "status": sol["status"],
                }

        return {
            "mode": "hybrid_selector",
            "N": cfg.N, "S": cfg.S, "k": cfg.k,
            "alpha": cfg.alpha, "C": cfg.C,
            "train_cvar_alpha": cfg.train_cvar_alpha,
            "best_params": best_theta,
            "training_history": hist,
            "eval_counts": counts_eval,
            "total_eval_shots": total_eval,
            "exact_k_shots": exact_k_shots,
            "exact_k_fraction": float(exact_k_shots) / total_eval if total_eval else 0.0,
            "best_portfolio": best_port,
            "qubo_scale": scale,
            "ising_offset_internal": offset,
            "candidates": candidates,
        }






    def run_full_penalized(
        self, warm_start: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        cfg = self.cfg
        Q, kappa, layout = self.build_full_qubo()
        scale = float(np.max(np.abs(Q)) + 1e-12)
        Qs = Q / scale; kappas = kappa / scale
        h, J, offset = self.qubo_to_ising(Qs, kappas)
        n = layout.n_total
        qc, gammas, betas = self.build_qaoa_circuit_from_ising(n, h, J, cfg.p)
        diag, iu, ju, vals = self.precompute_qubo_terms(Qs)

        if cfg.DRAW_CIRCUIT:
            print(qc.draw())

        def objective(theta: np.ndarray, shots: int) -> float:
            counts = self.sample_bitstrings(qc, gammas, betas, theta, shots)
            total = sum(counts.values())
            if total == 0: return float("inf")
            return sum(
                float(c) * self.energy_from_qubo_terms(diag, iu, ju, vals, kappas,
                                                        self.int_to_bits(int(oc), n))
                for oc, c in counts.items()
            ) / total

        best_theta, hist = self.spsa_optimize_params(
            objective_fn=objective, dim=2 * cfg.p,
            iters=cfg.spsa_iters, shots=cfg.shots_train, init=warm_start,
        )

        counts_eval = self.sample_bitstrings(qc, gammas, betas, best_theta, cfg.shots_eval)
        total_eval = int(sum(counts_eval.values()))

        all_outcomes: List[Dict] = []
        for oc, c in counts_eval.items():
            z = self.int_to_bits(int(oc), n)
            e = self.energy_from_qubo_terms(diag, iu, ju, vals, kappas, z)
            dec = self.decode_full_solution(z, layout)
            all_outcomes.append({
                "outcome": int(oc),
                "count": int(c),
                "fraction": float(c) / total_eval if total_eval else 0.0,
                "energy_internal": float(e),
                "energy_scaled": float(e * scale),
                "sum_x": int(dec["sum_x"]),
                "exact_k": int(dec["sum_x"]) == cfg.k,
                "decoded": dec,
                "z": z,
            })
        all_outcomes.sort(key=lambda d: d["energy_internal"])
        best = all_outcomes[0] if all_outcomes else None

        return {
            "mode": "full_penalized",
            "N": cfg.N, "S": cfg.S, "k": cfg.k,
            "alpha": cfg.alpha, "C": cfg.C,
            "n_qubits": n,
            "best_params": best_theta,
            "training_history": hist,
            "total_eval_shots": total_eval,
            "best_sample": {
                "energy_scaled": best["energy_scaled"],
                "energy_internal": best["energy_internal"],
                "count": best["count"],
                "fraction": best["fraction"],
                "outcome": best["outcome"],
            } if best else None,
            "decoded": best["decoded"] if best else None,
            "all_outcomes": all_outcomes,
            "qubo_scale": scale,
            "ising_offset_internal": offset,
        }








def compute_classical_benchmark(
    engine: PortfolioQAOAEngine, mode: str, results: Dict,
) -> Dict[str, object]:
    cfg = engine.cfg
    returns = engine.returns
    p_s = engine.p_s
    alpha, C = cfg.alpha, cfg.C

    full_sol = solve_cvar_weights(returns, p_s, alpha, C)
    benchmark: Dict[str, object] = {"full_universe_classical": full_sol}

    if mode == "hybrid":
        best = results.get("best_portfolio")
        if best is not None:
            subset_idx = best["subset_idx"]
            # Quantum portfolio metrics
            w_q = np.zeros(cfg.N, dtype=float)
            w_q[subset_idx] = best["weights"]
            port_ret = returns @ w_q
            benchmark["quantum_expected_return"] = float((p_s * port_ret).sum())
            benchmark["quantum_empirical_cvar"] = _empirical_cvar(-port_ret, alpha)
            # Classical on same subset
            benchmark["quantum_subset_classical"] = engine.solve_cvar_weights_cvxpy(subset_idx)

    elif mode == "full":
        decoded = results.get("decoded")
        if decoded is not None:
            subset_idx = np.where(decoded["x"].astype(int) == 1)[0]
            w_q = np.asarray(decoded["w"], dtype=float)
            port_ret = returns @ w_q
            benchmark["quantum_expected_return"] = float((p_s * port_ret).sum())
            benchmark["quantum_empirical_cvar"] = _empirical_cvar(-port_ret, alpha)
            if subset_idx.size > 0:
                benchmark["quantum_decoded_subset_classical"] = engine.solve_cvar_weights_cvxpy(subset_idx)

    return benchmark



def print_distribution(results: Dict, mode: str, tickers: Optional[List[str]], top_k: int = 10) -> None:
    print(f"\n{'='*65}")
    print(f"MEASURED OUTCOME DISTRIBUTION  (top {top_k})")
    print(f"{'='*65}")

    if mode == "hybrid":
        candidates = results.get("candidates", [])
        total = results.get("total_eval_shots", 0)
        k = results["k"]
        best_port = results.get("best_portfolio")
        print(f"Total eval shots : {total}")
        print(f"Exact-k fraction : {results.get('exact_k_fraction', 0.0):.4f}")
        print()
        header = f"{'Rk':>3}  {'Outcome':>12}  {'Card':>4}  {'Count':>6}  {'Frac':>7}  {'Energy':>12}  {'Exact-k':>7}"
        print(header)
        print("-" * len(header))
        seen: set = set()
        shown = 0
        for outcome_int, xb, c, e in candidates:
            if outcome_int in seen: continue
            seen.add(outcome_int)
            card = int(xb.sum())
            frac = float(c) / total if total else 0.0
            is_best = (best_port is not None and np.array_equal(xb, best_port["subset_bits"]))
            tag = " <-- SELECTED" if is_best else ""
            print(f"{shown+1:>3}  {outcome_int:>12}  {card:>4}  {c:>6}  {frac:>7.4f}  {e:>12.6f}  "
                  f"{'Yes' if card == k else 'No':>7}{tag}")
            shown += 1
            if shown >= top_k: break

    elif mode == "full":
        all_outcomes = results.get("all_outcomes", [])
        total = results.get("total_eval_shots", 0)
        k = results["k"]
        print(f"Total eval shots      : {total}")
        print(f"Unique outcome states : {len(all_outcomes)}")
        print()
        header = f"{'Rk':>3}  {'Count':>6}  {'Frac':>7}  {'Energy':>12}  {'SumX':>5}  {'SumW':>7}  {'ExactK':>7}"
        print(header)
        print("-" * len(header))
        for rank, d in enumerate(all_outcomes[:top_k]):
            dec = d["decoded"]
            tag = " <-- SELECTED" if rank == 0 else ""
            print(f"{rank+1:>3}  {d['count']:>6}  {d['fraction']:>7.4f}  "
                  f"{d['energy_internal']:>12.6f}  {dec['sum_x']:>5}  "
                  f"{dec['sum_w']:>7.4f}  {'Yes' if d['exact_k'] else 'No':>7}{tag}")


def print_benchmark_comparison(
    results: Dict, benchmark: Dict, mode: str,
    tickers: Optional[List[str]], cfg: EngineConfig,
) -> None:
    print(f"\n{'='*65}")
    print("BENCHMARK COMPARISON")
    print(f"{'='*65}")
    print(f"Mode         : {mode}")
    print(f"N={cfg.N}  S={cfg.S}  k={cfg.k}  alpha={cfg.alpha}  C={cfg.C}")
    if tickers:
        print(f"Tickers      : {tickers}")

    q_ret = benchmark.get("quantum_expected_return")
    q_cvar = benchmark.get("quantum_empirical_cvar")

    def _fmt(label: str, val) -> str:
        return f"  {label:30s}: {val:.6f}" if val is not None else f"  {label:30s}: N/A"

    if mode == "hybrid":
        best = results.get("best_portfolio")
        print("\n--- Quantum (hybrid selector) ---")
        if best is None:
            print("  No CVaR-feasible portfolio found.")
        else:
            if best.get("subset_tickers"):
                print(f"  Selected tickers : {best['subset_tickers']}")
            print(f"  Selected indices : {best['subset_idx'].tolist()}")
            print(f"  Weights          : {np.round(best['weights'], 5).tolist()}")
            print(_fmt("Expected return", q_ret * 25200))
            print(_fmt("Empirical CVaR", q_cvar * 100))
            print(_fmt("Shot fraction", best['fraction'] * 100))

        print("\n--- Classical baseline: full N-asset universe, CVaR-optimal ---")
        print("  [Upper bound given the same scenario data]")
        cl_b = benchmark.get("full_universe_classical", {})
        if cl_b.get("status") in ("optimal", "optimal_inaccurate"):
            print(_fmt("Expected return", cl_b.get('expected_return') * 25200))
            print(_fmt("Empirical CVaR", cl_b.get("empirical_cvar") * 100))
            if q_ret is not None:
                print(f"  Return gap (classical - quantum): {(cl_b['expected_return'] * 25200) - (q_ret* 25200):.6f}")
        else:
            print(f"  Infeasible / status: {cl_b.get('status','N/A')}")

    elif mode == "full":
        decoded = results.get("decoded")
        print("\n--- Quantum (full-penalised) ---")
        if decoded is None:
            print("  No decoded solution.")
        else:
            print(f"  Decoded x     : {decoded['x'].tolist()}")
            print(f"  sum_x (vs k)  : {decoded['sum_x']} (k={cfg.k})")
            print(f"  Decoded w     : {np.round(decoded['w'], 5).tolist()}")
            print(f"  sum_w         : {decoded['sum_w']:.4f}")
            print(_fmt("Expected return", q_ret * 25200))
            print(_fmt("Empirical CVaR", q_cvar * 25200))

        print("\n--- Classical baseline A: quantum-decoded subset, CVaR-optimal weights ---")
        cl_a = benchmark.get("quantum_decoded_subset_classical", {})
        if cl_a.get("status") in ("optimal", "optimal_inaccurate"):
            print(_fmt("Expected return", cl_a.get("expected_return") * 25200))
            print(_fmt("Empirical CVaR", cl_a.get("empirical_cvar") * 100))
            if q_ret is not None:
                print(f"  Return gap (classical A - quantum): {(cl_a['expected_return'] * 25200) - (q_ret* 25200):.6f}")
        else:
            print(f"  Infeasible / status: {cl_a.get('status','N/A')}")

        print("\n--- Classical baseline B: full N-asset universe, CVaR-optimal ---")
        cl_b = benchmark.get("full_universe_classical", {})
        if cl_b.get("status") in ("optimal", "optimal_inaccurate"):
            print(_fmt("Expected return", cl_b.get("expected_return") * 25200))
            print(_fmt("Empirical CVaR", cl_b.get("empirical_cvar") * 100))
            if q_ret is not None:
                print(f"  Return gap (classical B - quantum): {(cl_b['expected_return'] * 25200) - (q_ret* 25200):.6f}")
        else:
            print(f"  Infeasible / status: {cl_b.get('status','N/A')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_cli_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Quantum CVaR portfolio optimisation (QAOA).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--db", default=None,
                    help="SQLite database from data_prep.py. If omitted, synthetic data is used.")
    ap.add_argument("--start", default=None, help="Return history start date (YYYY-MM-DD).")
    ap.add_argument("--end", default=None, help="Return history end date (YYYY-MM-DD).")
    ap.add_argument("--scenario", default="bootstrap",
                    choices=["rolling", "historical", "bootstrap", "block_bootstrap", "gaussian"])
    ap.add_argument("--block-len", type=int, default=5)
    ap.add_argument("--mode", choices=["hybrid", "full"], default="hybrid")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--S", type=int, default=50)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.99)
    ap.add_argument("--C", type=float, default=0.03)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--shots-train", type=int, default=2048)
    ap.add_argument("--shots-eval", type=int, default=16384)
    ap.add_argument("--train-cvar-alpha", type=float, default=0.25)
    ap.add_argument("--spsa-iters", type=int, default=80)
    ap.add_argument("--spsa-a", type=float, default=0.2)
    ap.add_argument("--spsa-c", type=float, default=0.1)
    ap.add_argument("--warm-start", default=None, metavar="PATH",
                    help="Path to .npy file with initial QAOA parameters.")
    ap.add_argument("--save-params", default=None, metavar="PATH",
                    help="Save optimised QAOA parameters to .npy.")
    ap.add_argument("--Bw", type=int, default=2)
    ap.add_argument("--Bt", type=int, default=2)
    ap.add_argument("--Bv", type=int, default=2)
    ap.add_argument("--Bxi", type=int, default=2)
    ap.add_argument("--Beta", type=int, default=2)
    ap.add_argument("--lambda-K", type=float, default=10.0)
    ap.add_argument("--beta-return", type=float, default=1.0)
    ap.add_argument("--gamma-risk", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k-outcomes", type=int, default=10)
    ap.add_argument("--draw-circuit", action="store_true")
    return ap


def main() -> None:
    ap = build_cli_parser()
    args = ap.parse_args()

    print("=" * 65)
    print("Quantum CVaR Portfolio Optimisation")
    print("=" * 65)
    print(f"Mode : {args.mode}   N={args.N}  S={args.S}  k={args.k}  "
          f"alpha={args.alpha}  C={args.C}  seed={args.seed}")

    warm_start: Optional[np.ndarray] = None
    if args.warm_start:
        warm_start = np.load(args.warm_start)
        print(f"Warm-start loaded: {args.warm_start}  shape={warm_start.shape}")

    cfg = EngineConfig(
        N=args.N, S=args.S, k=args.k, alpha=args.alpha, C=args.C,
        p=args.p, shots_train=args.shots_train, shots_eval=args.shots_eval,
        train_cvar_alpha=args.train_cvar_alpha,
        spsa_iters=args.spsa_iters, spsa_a=args.spsa_a, spsa_c=args.spsa_c,
        lambda_K=args.lambda_K, beta_return=args.beta_return, gamma_risk=args.gamma_risk,
        Bw=args.Bw, Bt=args.Bt, Bv=args.Bv, Bxi=args.Bxi, Beta=args.Beta,
        seed=args.seed, DRAW_CIRCUIT=args.draw_circuit,
    )
    engine = PortfolioQAOAEngine(cfg)

    chosen_tickers: Optional[List[str]] = None
    if args.db:
        if not _HAS_DATA_PREP:
            print("ERROR: data_prep.py is not importable.", file=sys.stderr)
            sys.exit(1)
        print(f"\nQuerying {args.db} for {args.N} assets ...")
        scen, p_s, chosen_tickers = sample_assets_from_db(
            db_path=args.db, N=args.N, S=args.S,
            scenario_method=args.scenario, scenario_seed=args.seed,
            start=args.start, end=args.end, block_len=args.block_len,
        )
        print(f"Selected tickers : {chosen_tickers}")
        engine.set_scenarios(scen, p_s, tickers=chosen_tickers)
    else:
        print("\nNo --db specified; using synthetic data.")

    print(f"\nRunning SPSA ({args.spsa_iters} iters) ...")
    if args.mode == "hybrid":
        results = engine.run_hybrid_selector(warm_start=warm_start)
        best = results.get("best_portfolio")
        print(f"Exact-k fraction : {results['exact_k_fraction']:.4f}")
        if best:
            print(f"Best portfolio return : {best['expected_return']:.6f}")
        else:
            print("No CVaR-feasible exact-k subset found.")
    else:
        results = engine.run_full_penalized(warm_start=warm_start)
        print(f"Qubits used : {results['n_qubits']}")
        bs = results.get("best_sample")
        if bs:
            print(f"Best energy (internal) : {bs['energy_internal']:.6f}")

    if args.save_params:
        np.save(args.save_params, results["best_params"])
        print(f"\nSaved parameters -> {args.save_params}")

    print_distribution(results, mode=args.mode,
                       tickers=chosen_tickers, top_k=args.top_k_outcomes)

    benchmark = compute_classical_benchmark(engine, mode=args.mode, results=results)
    print_benchmark_comparison(results, benchmark, mode=args.mode,
                               tickers=chosen_tickers, cfg=cfg)

    if args.mode == "full":
        decoded = results.get("decoded")
        if decoded is not None:
            r, p_s = engine.returns, engine.p_s
            w = np.asarray(decoded["w"], dtype=float)
            v = np.asarray(decoded["v"], dtype=float)
            t = float(decoded["t"])
            eta = np.asarray(decoded["eta"], dtype=float)
            xi = float(decoded["xi"])
            print(f"\n--- Full-mode penalty residuals ---")
            print(f"  Budget residual (sum_w - 1)   : {decoded['sum_w'] - 1.0:.6f}")
            print(f"  Cardinality residual (sum_x-k): {decoded['sum_x'] - cfg.k}")
            cvar_res = t + (1.0 / (1.0 - cfg.alpha)) * np.dot(p_s, v) + xi - cfg.C
            tail_res = v + (r @ w) + t - eta
            print(f"  CVaR budget residual          : {float(cvar_res):.6f}")
            print(f"  Max abs tail residual         : {float(np.max(np.abs(tail_res))):.6f}")


if __name__ == "__main__":
    main()