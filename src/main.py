from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import math
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2 

import cvxpy as cp



@dataclass(frozen=True)
class EngineConfig:
    N: int = 16               # assets
    S: int = 50              # scenarios
    k: int = 6               # target cardinality

    # CVaR settings for classical stage
    alpha: float = 0.99        # confidence level
    C: float = 0.03          # CVaR budget (loss units)

    # Selector QUBO trade-off weights
    beta_return: float = 1.0
    gamma_risk: float = 1.0

    # Penalties 
    lambda_V: float = 10.0
    lambda_B: float = 10.0
    lambda_K: float = 10.0
    lambda_L: float = 10.0
    lambda_C: float = 10.0
    lambda_T: float = 10.0

    # QAOA settings 
    p: int = 2
    shots_train: int = 2048
    shots_eval: int = 8192
    gamma_bounds: Tuple[float, float] = (0.0, 2.0 * math.pi)
    beta_bounds: Tuple[float, float] = (0.0, math.pi)

    # SPSA settings 
    spsa_iters: int = 80
    spsa_a: float = 0.2
    spsa_c: float = 0.1
    spsa_A: float = 10.0
    spsa_alpha: float = 0.602
    spsa_gamma: float = 0.101

    # Synthetic data parameters
    seed: int = 125
    # Place holders for real data
    ret_mu: float = 0.0005   # mean return per scenario
    ret_sigma: float = 0.02  # std dev per scenario

    # Real-data input (optional)
    # If provided, main.py will load scenario returns from this .npz file.
    # Expected keys: 'returns' (S,N), optional 'p_s' (S,), optional 'tickers' (N,)
    returns_npz_path: Optional[str] = None

    # Full-QUBO discretization 
    Bw: int = 2       # bits per weight w_i
    Bt: int = 2       # bits for t
    Bv: int = 2       # bits per tail excess v_s
    Bxi: int = 2      # bits for slack xi
    Beta: int = 2     # bits per slack eta_s

    # Ranges for fixed-point encodings
    w_max: float = 1.0
    v_max: float = 0.20
    xi_max: float = 0.20
    eta_max: float = 0.20

    # t range
    auto_t_range: bool = True       #  (if auto_t_range=True, computed from data)
    t_min: float = -0.10
    t_max: float = 0.10

    # Candidate subsets evaluated in hybrid mode 
    max_unique_subsets_to_score: int = 50
    
    # Toggle for circuit draw
    DRAW_CIRCUIT: bool = False


@dataclass
class BitLayout:
    n_total: int
    # Slices / index lists
    x_idx: List[int]                          # length N
    w_idx: List[List[int]]                    # N x Bw
    t_idx: List[int]                          # Bt
    v_idx: List[List[int]]                    # S x Bv
    xi_idx: List[int]                         # Bxi
    eta_idx: List[List[int]]                  # S x Beta

    # Encodings (offset + deltas) aligned with the above
    w0: np.ndarray                            # (N,)
    w_delta: np.ndarray                       # (N,Bw)
    
    t0: float
    t_delta: np.ndarray                       # (Bt,)
    
    v0: np.ndarray                            # (S,)
    v_delta: np.ndarray                       # (Bv,)
    
    xi0: float
    xi_delta: np.ndarray                      # (Bxi,)
    
    eta0: np.ndarray                          # (S,)
    eta_delta: np.ndarray                     # (Beta,)


def fixed_point_deltas(vmax: float, B: int) -> np.ndarray:
    if B <= 0:
        return np.array([], dtype=float)
    step = vmax / (2**B - 1)
    return np.array([step * (2**b) for b in range(B)], dtype=float)


class PortfolioQAOAEngine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # Scenario returns and probabilities.
        # If cfg.returns_npz_path is set, load real scenarios from disk (data_prep.py output).
        self.tickers = None
        if cfg.returns_npz_path:
            returns, p_s, tickers = self.load_returns_npz(cfg.returns_npz_path, expected_S=cfg.S, expected_N=cfg.N)
            self.tickers = tickers
            self.set_scenarios(returns, p_s)
        else:
            self.returns = self.generate_synthetic_scenarios()
            self.p_s = np.full(cfg.S, 1.0 / cfg.S, dtype=float)
            self.mu, self.Omega = self.estimate_mu_omega(self.returns, self.p_s)

            # Auto t-range for full-QUBO
            if cfg.auto_t_range:
                w_eq = np.full(cfg.N, 1.0 / cfg.N)
                losses = -self.returns @ w_eq
                pad = 0.05 * (losses.max() - losses.min() + 1e-12)
                self.t_min = float(losses.min() - pad)
                self.t_max = float(losses.max() + pad)
            else:
                self.t_min = cfg.t_min
                self.t_max = cfg.t_max

        # If we loaded real scenarios, set_scenarios already handled t-range; if not auto, respect config.
        if cfg.returns_npz_path and not cfg.auto_t_range:
            self.t_min = cfg.t_min
            self.t_max = cfg.t_max


    def generate_synthetic_scenarios(self) -> np.ndarray:
        cfg = self.cfg
        return self.rng.normal(loc=cfg.ret_mu, scale=cfg.ret_sigma, size=(cfg.S, cfg.N)).astype(float)

    @staticmethod
    def estimate_mu_omega(returns: np.ndarray, p_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu = (p_s[:, None] * returns).sum(axis=0)

        centered = returns - mu[None, :]
        Omega = (p_s[:, None, None] * (centered[:, :, None] * centered[:, None, :])).sum(axis=0)

        Omega = 0.5 * (Omega + Omega.T)

        return mu, Omega

    def load_returns_npz(self, path: str, expected_S: int, expected_N: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Load scenario returns from an .npz file produced by data_prep.py."""
        npz = np.load(Path(path), allow_pickle=True)
        if "returns" not in npz:
            raise KeyError(f"NPZ file {path} missing required key 'returns'")
        returns = np.asarray(npz["returns"], dtype=float)
        if returns.ndim != 2:
            raise ValueError(f"'returns' must be 2D (S,N). Got shape {returns.shape}")
        S, N = returns.shape
        if S != expected_S or N != expected_N:
            raise ValueError(f"returns shape mismatch. Expected (S,N)=({expected_S},{expected_N}), got ({S},{N}). "
                             f"Regenerate scenarios with matching N and S or update EngineConfig.")
        if "p_s" in npz:
            p_s = np.asarray(npz["p_s"], dtype=float).reshape(-1)
            if p_s.size != S:
                raise ValueError(f"'p_s' must have length S={S}. Got {p_s.size}")
            p_s = p_s / p_s.sum()
        else:
            p_s = np.full(S, 1.0 / S, dtype=float)

        tickers = None
        if "tickers" in npz:
            tickers = np.asarray(npz["tickers"])
            if tickers.size != N:
                tickers = None

        return returns, p_s, tickers

    def set_scenarios(self, returns: np.ndarray, p_s: Optional[np.ndarray] = None) -> None:
        """Replace scenarios and recompute moments (mu, Omega) and t-range if configured."""
        self.returns = np.asarray(returns, dtype=float)
        if p_s is None:
            self.p_s = np.full(self.returns.shape[0], 1.0 / self.returns.shape[0], dtype=float)
        else:
            p_s = np.asarray(p_s, dtype=float).reshape(-1)
            self.p_s = p_s / p_s.sum()

        self.mu, self.Omega = self.estimate_mu_omega(self.returns, self.p_s)

        # Update t-range bounds for full-QUBO encoding if requested
        if self.cfg.auto_t_range:
            w_eq = np.full(self.cfg.N, 1.0 / self.cfg.N)
            losses = -self.returns @ w_eq
            pad = 0.05 * (losses.max() - losses.min() + 1e-12)
            self.t_min = float(losses.min() - pad)
            self.t_max = float(losses.max() + pad)


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

        kappa = lamK * (k * k)
        return Q, kappa
    
    def make_full_bit_layout(self) -> BitLayout:
        cfg = self.cfg
        idx = 0

        # x bits
        x_idx = list(range(idx, idx + cfg.N))
        idx += cfg.N

        # w bits 
        w_idx: List[List[int]] = []
        for _ in range(cfg.N):
            w_bits = list(range(idx, idx + cfg.Bw))
            w_idx.append(w_bits)
            idx += cfg.Bw

        # t bits
        t_idx = list(range(idx, idx + cfg.Bt))
        idx += cfg.Bt

        # v bits 
        v_idx: List[List[int]] = []
        for _ in range(cfg.S):
            v_bits = list(range(idx, idx + cfg.Bv))
            v_idx.append(v_bits)
            idx += cfg.Bv

        # xi bits
        xi_idx = list(range(idx, idx + cfg.Bxi))
        idx += cfg.Bxi

        # eta bits 
        eta_idx: List[List[int]] = []
        for _ in range(cfg.S):
            eta_bits = list(range(idx, idx + cfg.Beta))
            eta_idx.append(eta_bits)
            idx += cfg.Beta

        # Encodings 
        w0 = np.zeros(cfg.N, dtype=float)
        w_delta = np.stack([fixed_point_deltas(cfg.w_max, cfg.Bw) for _ in range(cfg.N)], axis=0)

        # t is shifted-range
        t0 = float(self.t_min)
        t_delta = fixed_point_deltas(float(self.t_max - self.t_min), cfg.Bt)

        v0 = np.zeros(cfg.S, dtype=float)
        v_delta = fixed_point_deltas(cfg.v_max, cfg.Bv)

        xi0 = 0.0
        xi_delta = fixed_point_deltas(cfg.xi_max, cfg.Bxi)

        eta0 = np.zeros(cfg.S, dtype=float)
        eta_delta = fixed_point_deltas(cfg.eta_max, cfg.Beta)

        return BitLayout(
            n_total=idx,
            x_idx=x_idx,
            w_idx=w_idx,
            t_idx=t_idx,
            v_idx=v_idx,
            xi_idx=xi_idx,
            eta_idx=eta_idx,
            w0=w0,
            w_delta=w_delta,
            t0=t0,
            t_delta=t_delta,
            v0=v0,
            v_delta=v_delta,
            xi0=xi0,
            xi_delta=xi_delta,
            eta0=eta0,
            eta_delta=eta_delta,
        )

    def build_full_qubo(self) -> Tuple[np.ndarray, float, BitLayout]:
        cfg = self.cfg
        layout = self.make_full_bit_layout()

        M = layout.n_total
        Q = np.zeros((M, M), dtype=float)
        kappa = 0.0

        # Adds the values to QUBO form
        def add_linear(i: int, a: float) -> None:
            Q[i, i] += a

        def add_quadratic(i: int, j: int, a: float) -> None:
            if i == j:
                Q[i, i] += a  # z_i^2 = z_i
            else:
                if i > j:
                    i, j = j, i
                Q[i, j] += a

        def add_affine_square(lam: float, g0: float, coeffs: Dict[int, float]) -> None:
            nonlocal kappa
            kappa += lam * (g0 * g0)
            # Linear updates
            for i, gi in coeffs.items():
                add_linear(i, lam * (2.0 * g0 * gi + gi * gi))
            # Quadratic updates
            idx = list(coeffs.keys())
            for a_i in range(len(idx)):
                i = idx[a_i]
                gi = coeffs[i]
                for a_j in range(a_i + 1, len(idx)):
                    j = idx[a_j]
                    gj = coeffs[j]
                    add_quadratic(i, j, 2.0 * lam * gi * gj)

        # Return
        for i in range(cfg.N):
            mu_i = self.mu[i]
            # constant offset from w0
            kappa += -mu_i * layout.w0[i]
            # bit contributions
            for b, bit_idx in enumerate(layout.w_idx[i]):
                add_linear(bit_idx, -mu_i * layout.w_delta[i, b])

        # Volatility
        lamV = cfg.lambda_V
        for i in range(cfg.N):
            for j in range(i, cfg.N):
                coef = self.Omega[i, j] * (2.0 if j > i else 1.0)
                if abs(coef) < 1e-15:
                    continue
                # constant offset
                kappa += lamV * coef * layout.w0[i] * layout.w0[j]

                # linear parts 
                if layout.w0[i] != 0.0:
                    for b, bit_j in enumerate(layout.w_idx[j]):
                        add_linear(bit_j, lamV * coef * layout.w0[i] * layout.w_delta[j, b])
                if layout.w0[j] != 0.0:
                    for b, bit_i in enumerate(layout.w_idx[i]):
                        add_linear(bit_i, lamV * coef * layout.w0[j] * layout.w_delta[i, b])

                # quadratic bit-bit parts
                for b, bit_i in enumerate(layout.w_idx[i]):
                    di = layout.w_delta[i, b]
                    for bp, bit_j in enumerate(layout.w_idx[j]):
                        dj = layout.w_delta[j, bp]
                        add_quadratic(bit_i, bit_j, lamV * coef * di * dj)

        # Budget
        lamB = cfg.lambda_B
        gB0 = float(layout.w0.sum() - 1.0)
        coeffs_B: Dict[int, float] = {}
        for i in range(cfg.N):
            for b, bit_idx in enumerate(layout.w_idx[i]):
                coeffs_B[bit_idx] = coeffs_B.get(bit_idx, 0.0) + float(layout.w_delta[i, b])
        add_affine_square(lamB, gB0, coeffs_B)

        # Cardinality
        lamK = cfg.lambda_K
        gK0 = float(-cfg.k)
        coeffs_K: Dict[int, float] = {idx: 1.0 for idx in layout.x_idx}
        add_affine_square(lamK, gK0, coeffs_K)

        # Link
        lamL = cfg.lambda_L
        for i in range(cfg.N):
            x_i = layout.x_idx[i]
            # +lambda_L w_i term
            kappa += lamL * layout.w0[i]
            for b, bit_w in enumerate(layout.w_idx[i]):
                add_linear(bit_w, lamL * layout.w_delta[i, b])

            # -lambda_L w_i x_i term (bilinear)
            add_linear(x_i, -lamL * layout.w0[i])  # from offset w0_i * x_i
            for b, bit_w in enumerate(layout.w_idx[i]):
                add_quadratic(x_i, bit_w, -lamL * layout.w_delta[i, b])

        # CVaR
        lamC = cfg.lambda_C
        a = self.p_s / (1.0 - cfg.alpha)

        gC0 = float(layout.t0 + float((a * layout.v0).sum()) + layout.xi0 - cfg.C)
        coeffs_C: Dict[int, float] = {}

        # t bits
        for b, bit_t in enumerate(layout.t_idx):
            coeffs_C[bit_t] = coeffs_C.get(bit_t, 0.0) + float(layout.t_delta[b])

        # v bits 
        for s in range(cfg.S):
            for b, bit_v in enumerate(layout.v_idx[s]):
                coeffs_C[bit_v] = coeffs_C.get(bit_v, 0.0) + float(a[s] * layout.v_delta[b])

        # xi bits
        for b, bit_xi in enumerate(layout.xi_idx):
            coeffs_C[bit_xi] = coeffs_C.get(bit_xi, 0.0) + float(layout.xi_delta[b])

        add_affine_square(lamC, gC0, coeffs_C)

        # Tail
        lamT = cfg.lambda_T
        for s in range(cfg.S):
            r_s = self.returns[s, :]  # (N,)

            gT0 = float(layout.v0[s] + float(np.dot(r_s, layout.w0)) + layout.t0 - layout.eta0[s])
            coeffs_T: Dict[int, float] = {}

            # v_s bits
            for b, bit_v in enumerate(layout.v_idx[s]):
                coeffs_T[bit_v] = coeffs_T.get(bit_v, 0.0) + float(layout.v_delta[b])

            # t bits
            for b, bit_t in enumerate(layout.t_idx):
                coeffs_T[bit_t] = coeffs_T.get(bit_t, 0.0) + float(layout.t_delta[b])

            # eta_s bits
            for b, bit_eta in enumerate(layout.eta_idx[s]):
                coeffs_T[bit_eta] = coeffs_T.get(bit_eta, 0.0) - float(layout.eta_delta[b])

            # weight bits
            for i in range(cfg.N):
                for b, bit_w in enumerate(layout.w_idx[i]):
                    coeffs_T[bit_w] = coeffs_T.get(bit_w, 0.0) + float(r_s[i] * layout.w_delta[i, b])

            add_affine_square(lamT, gT0, coeffs_T)

        return Q, float(kappa), layout




    @staticmethod
    def qubo_to_ising(Q_upper: np.ndarray, kappa: float) -> Tuple[np.ndarray, Dict[Tuple[int, int], float], float]:
        n = Q_upper.shape[0]
        a = np.diag(Q_upper).copy()
        # Off-diagonal 
        # Constant offset
        offset = float(kappa + 0.5 * a.sum())
        
        # Add off-diagonal constant contributions
        for i in range(n):
            for j in range(i + 1, n):
                offset += float(Q_upper[i, j] / 4.0)

        h = -0.5 * a
        for i in range(n):
            s = 0.0
            for j in range(n):
                if i == j:
                    continue
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
    def ising_to_pauliop(h: np.ndarray, J: Dict[Tuple[int, int], float]) -> SparsePauliOp:
        n = len(h)
        paulis: List[Tuple[str, complex]] = []
        
        # Z terms
        for i, hi in enumerate(h):
            if abs(hi) < 1e-15:
                continue
            s = ["I"] * n
            s[i] = "Z"
            paulis.append(("".join(reversed(s)), complex(hi)))  # reversed for Qiskit's string order
            
        # ZZ terms
        for (i, j), Jij in J.items():
            if abs(Jij) < 1e-15:
                continue
            s = ["I"] * n
            s[i] = "Z"
            s[j] = "Z"
            paulis.append(("".join(reversed(s)), complex(Jij)))
        if not paulis:
            # Return a 0 operator if empty
            return SparsePauliOp.from_list([("I" * n, 0.0)])
        return SparsePauliOp.from_list(paulis)


    def build_qaoa_circuit_from_ising(
        self,
        n_qubits: int,
        h: np.ndarray,
        J: Dict[Tuple[int, int], float],
        p: int,
    ) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector]:
        gammas = ParameterVector("gamma", p)
        betas = ParameterVector("beta", p)

        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits))

        for l in range(p):
            # Cost: exp(-i gamma_l H_C) with H_C = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j
            for i in range(n_qubits):
                hi = float(h[i])
                if abs(hi) > 1e-15:
                    qc.rz(2.0 * hi * gammas[l], i)

            for (i, j), Jij in J.items():
                if abs(Jij) > 1e-15:
                    qc.rzz(2.0 * float(Jij) * gammas[l], i, j)

            # Mixer: exp(-i beta_l sum_i X_i)
            for i in range(n_qubits):
                qc.rx(2.0 * betas[l], i)

        if self.cfg.DRAW_CIRCUIT:
            print(qc.draw(output="text"))

        qc.measure_all()
        return qc, gammas, betas

    def get_sampler(self, shots: int) -> AerSamplerV2:
        return AerSamplerV2(default_shots=shots, seed=self.cfg.seed)

    @staticmethod
    def int_to_bits(x: int, n: int) -> np.ndarray:
        return np.array([(x >> i) & 1 for i in range(n)], dtype=np.int8)

    @staticmethod
    def energy_from_qubo(Q_upper: np.ndarray, kappa: float, z_bits: np.ndarray) -> float:
        z = z_bits.astype(float)
        E = float(kappa + np.dot(np.diag(Q_upper), z))
        n = len(z)
        # upper triangle off-diagonal
        for i in range(n):
            if z[i] == 0.0:
                continue
            zi = z[i]
            row = Q_upper[i]
            for j in range(i + 1, n):
                if z[j] != 0.0:
                    E += float(row[j] * zi * z[j])
        return E

    def sample_bitstrings(self, circuit: QuantumCircuit, gammas: ParameterVector, betas: ParameterVector, params: np.ndarray, shots: int) -> Dict[int, int]:
        sampler = self.get_sampler(shots=shots)

        p = self.cfg.p
        gamma_vals = params[:p].tolist()
        beta_vals = params[p:].tolist()
        param_values = gamma_vals + beta_vals

        if circuit.num_parameters != len(param_values):
            raise ValueError(
                f"Parameter length mismatch: circuit expects {circuit.num_parameters}, got {len(param_values)}."
            )


        pubs = [(circuit, param_values)]
        job = sampler.run(pubs, shots=shots)
        result = job.result()
        pub_result = result[0]

        counts = pub_result.data.meas.get_counts()

        # Convert keys to integers compatible with int_to_bits()
        out: Dict[int, int] = {}
        for k, v in counts.items():
            if isinstance(k, str):
                out[int(k, 2)] = int(v)
            else:
                out[int(k)] = int(v)
        return out

    # Had to find this
    def spsa_optimize_params(self, objective_fn, dim: int, iters: int, shots: int, init: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[float]]:
        cfg = self.cfg
        p = cfg.p
        assert dim == 2 * p, "For p layers, dim must be 2p (gammas + betas)."

        if init is None:
            # Random starting point within bounds
            gamma0 = self.rng.uniform(cfg.gamma_bounds[0], cfg.gamma_bounds[1], size=p)
            beta0 = self.rng.uniform(cfg.beta_bounds[0], cfg.beta_bounds[1], size=p)
            theta = np.concatenate([gamma0, beta0]).astype(float)
        else:
            theta = init.astype(float).copy()

        def project(theta_in: np.ndarray) -> np.ndarray:
            th = theta_in.copy()

            th[:p] = np.mod(th[:p], cfg.gamma_bounds[1])

            th[p:] = np.mod(th[p:], cfg.beta_bounds[1])
            return th

        theta = project(theta)
        best_theta = theta.copy()
        best_val = float("inf")
        hist: List[float] = []

        for k in range(iters):
            ak = cfg.spsa_a / ((k + 1.0 + cfg.spsa_A) ** cfg.spsa_alpha)
            ck = cfg.spsa_c / ((k + 1.0) ** cfg.spsa_gamma)

            # Bernoulli +/-1 perturbation
            delta = self.rng.choice([-1.0, 1.0], size=dim)
            theta_plus = project(theta + ck * delta)
            theta_minus = project(theta - ck * delta)

            f_plus = float(objective_fn(theta_plus, shots))
            f_minus = float(objective_fn(theta_minus, shots))

            # SPSA gradient estimate
            ghat = (f_plus - f_minus) / (2.0 * ck) * delta

            # Update
            theta = project(theta - ak * ghat)

            # Track
            f_curr = float(objective_fn(theta, shots))
            hist.append(f_curr)
            if f_curr < best_val:
                best_val = f_curr
                best_theta = theta.copy()

        return best_theta, hist

    # Had to find this
    @staticmethod
    def empirical_cvar(losses: np.ndarray, alpha: float) -> float:
        if losses.size == 0:
            return float("nan")
        var = float(np.quantile(losses, alpha))
        tail = losses[losses >= var]
        return float(tail.mean()) if tail.size > 0 else var

    # Had to find this
    def solve_cvar_weights_cvxpy(self, subset: np.ndarray) -> Dict[str, object]:
        cfg = self.cfg
        sel = np.asarray(subset, dtype=int)
        k = sel.size
        if k == 0:
            return {"status": "empty_subset"}

        r_sub = self.returns[:, sel]  
        mu_sub = self.mu[sel]         
        p_s = self.p_s                

        w = cp.Variable(k, nonneg=True)
        t = cp.Variable() 
        v = cp.Variable(cfg.S, nonneg=True)

        # Losses
        losses = -r_sub @ w

        constraints = [v >= losses - t, cp.sum(w) == 1.0, t + (1.0 / (1.0 - cfg.alpha)) * (p_s @ v) <= cfg.C]

        objective = cp.Maximize(mu_sub @ w)
        prob = cp.Problem(objective, constraints)

        # ECOS is a good LP/QP default; SCS is fallback
        try:
            prob.solve(solver=cp.ECOS, verbose=False)
        except Exception:
            prob.solve(solver=cp.SCS, verbose=False)

        status = prob.status
        if status not in ("optimal", "optimal_inaccurate"):
            return {"status": status}

        w_val = np.array(w.value, dtype=float).reshape(-1)
        # Compute achieved metrics on in-sample scenarios
        port_returns = r_sub @ w_val
        port_losses = -port_returns
        exp_ret = float(mu_sub @ w_val)
        cvar_hat = self.empirical_cvar(port_losses, cfg.alpha)

        return {"status": status, "weights": w_val, "expected_return": exp_ret, "empirical_cvar": cvar_hat}


    def run_hybrid_selector(self) -> Dict[str, object]:
        cfg = self.cfg
        Q, kappa = self.build_selector_qubo()

        scale = float(np.max(np.abs(Q)) + 1e-12)
        Qs = Q / scale
        kappas = kappa / scale

        h, J, offset = self.qubo_to_ising(Qs, kappas)
        n = cfg.N

        qc, gammas, betas = self.build_qaoa_circuit_from_ising(n, h, J, cfg.p)

        def objective(theta: np.ndarray, shots: int) -> float:
            counts = self.sample_bitstrings(qc, gammas, betas, theta, shots=shots)

            energies: List[float] = []
            for outcome, c in counts.items():
                z = self.int_to_bits(outcome, n)
                e = self.energy_from_qubo(Qs, kappas, z)
                energies.extend([e] * c)
            return float(np.mean(energies)) if energies else float("inf")

        best_theta, hist = self.spsa_optimize_params(objective_fn=objective, dim=2 * cfg.p, iters=cfg.spsa_iters, shots=cfg.shots_train, init=None)

        # Final evaluation
        counts_eval = self.sample_bitstrings(qc, gammas, betas, best_theta, shots=cfg.shots_eval)

        # Rank candidate subsets
        candidates: List[Tuple[int, np.ndarray, int, float]] = []
        for outcome, c in counts_eval.items():
            outcome_int = int(outcome)
            x_bits = self.int_to_bits(outcome_int, n)
            card = int(x_bits.sum())
            e = self.energy_from_qubo(Qs, kappas, x_bits)
            candidates.append((outcome_int, x_bits, int(c), float(e)))

        def key(item):
            outcome_int, x_bits, c, e = item
            card = int(x_bits.sum())
            return (0 if card == self.cfg.k else 1, e, -c)

        candidates.sort(key=key)

        seen: set[int] = set()
        best_port = None
        best_ret = -float("inf")
        scored = 0
        for outcome_int, x_bits, c, e in candidates:
            if outcome_int in seen:
                continue
            seen.add(outcome_int)

            subset = np.where(x_bits == 1)[0]
            if subset.size != self.cfg.k:
                continue

            scored += 1
            if scored > cfg.max_unique_subsets_to_score:
                break

            sol = self.solve_cvar_weights_cvxpy(subset)
            if sol.get("status") not in ("optimal", "optimal_inaccurate"):
                continue

            exp_ret = float(sol["expected_return"])
            if exp_ret > best_ret:
                best_ret = exp_ret
                best_port = {
                    "subset_bits": x_bits,
                    "subset_idx": subset,
                    "selector_energy_scaled": float(e * scale),
                    "selector_energy_internal": float(e),
                    "count": int(c),
                    "weights": sol["weights"],
                    "expected_return": exp_ret,
                    "empirical_cvar": float(sol["empirical_cvar"]),
                    "status": sol["status"]
                }

        return {
            "mode": "hybrid_selector",
            "N": cfg.N,
            "S": cfg.S,
            "k": cfg.k,
            "alpha": cfg.alpha,
            "C": cfg.C,
            "best_params": best_theta,
            "training_history": hist,
            "eval_counts": counts_eval,
            "best_portfolio": best_port,
            "qubo_scale": scale,
            "ising_offset_internal": offset
        }


    @staticmethod
    def decode_affine(bits: np.ndarray, offset: float, deltas: np.ndarray) -> float:
        """Decode y = offset + sum_b deltas[b] * bits[b]."""
        if deltas.size == 0:
            return float(offset)
        return float(offset + np.dot(deltas.astype(float), bits.astype(float)))

    def decode_full_solution(self, z: np.ndarray, layout: BitLayout) -> Dict[str, object]:
        """Decode the full bitstring into (x,w,t,v,xi,eta) using the layout encodings."""
        cfg = self.cfg

        # x
        x = z[layout.x_idx].astype(int)

        # w
        w = np.zeros(cfg.N, dtype=float)
        for i in range(cfg.N):
            bits = z[layout.w_idx[i]]
            w[i] = self.decode_affine(bits, float(layout.w0[i]), layout.w_delta[i])

        # t
        t_bits = z[layout.t_idx]
        t = self.decode_affine(t_bits, layout.t0, layout.t_delta)

        # v
        v = np.zeros(cfg.S, dtype=float)
        for s in range(cfg.S):
            bits = z[layout.v_idx[s]]
            v[s] = self.decode_affine(bits, float(layout.v0[s]), layout.v_delta)

        # xi
        xi_bits = z[layout.xi_idx]
        xi = self.decode_affine(xi_bits, layout.xi0, layout.xi_delta)

        # eta
        eta = np.zeros(cfg.S, dtype=float)
        for s in range(cfg.S):
            bits = z[layout.eta_idx[s]]
            eta[s] = self.decode_affine(bits, float(layout.eta0[s]), layout.eta_delta)

        return {
            "x": x,
            "w": w,
            "t": float(t),
            "v": v,
            "xi": float(xi),
            "eta": eta,
            "sum_w": float(w.sum()),
            "sum_x": int(x.sum()),
        }


    def run_full_penalized(self) -> Dict[str, object]:
        cfg = self.cfg
        Q, kappa, layout = self.build_full_qubo()

        # Normalize for conditioning
        scale = float(np.max(np.abs(Q)) + 1e-12)
        Qs = Q / scale
        kappas = kappa / scale

        h, J, offset = self.qubo_to_ising(Qs, kappas)
        n = layout.n_total

        qc, gammas, betas = self.build_qaoa_circuit_from_ising(n, h, J, cfg.p)

        def objective(theta: np.ndarray, shots: int) -> float:
            counts = self.sample_bitstrings(qc, gammas, betas, theta, shots=shots)
            energies: List[float] = []
            for outcome, c in counts.items():
                z = self.int_to_bits(outcome, n)
                e = self.energy_from_qubo(Qs, kappas, z)
                energies.extend([e] * c)
            return float(np.mean(energies)) if energies else float("inf")

        best_theta, hist = self.spsa_optimize_params(objective_fn=objective, dim=2 * cfg.p, iters=cfg.spsa_iters, shots=cfg.shots_train, init=None)

        counts_eval = self.sample_bitstrings(qc, gammas, betas, best_theta, shots=cfg.shots_eval)

        # Pick the single best-energy sample
        best = None
        for outcome, c in counts_eval.items():
            z = self.int_to_bits(outcome, n)
            e = self.energy_from_qubo(Qs, kappas, z)
            if (best is None) or (e < best["energy_internal"]):
                best = {"outcome": int(outcome), "count": int(c), "energy_internal": float(e), "z": z}

        decoded = self.decode_full_solution(best["z"], layout) if best is not None else None

        return {
            "mode": "full_penalized",
            "N": cfg.N,
            "S": cfg.S,
            "k": cfg.k,
            "alpha": cfg.alpha,
            "C": cfg.C,
            "n_qubits": n,
            "best_params": best_theta,
            "training_history": hist,
            "best_sample": {
                "energy_scaled": float(best["energy_internal"] * scale) if best else None,
                "energy_internal": float(best["energy_internal"]) if best else None,
                "count": int(best["count"]) if best else None,
                "outcome": int(best["outcome"]) if best else None,
            } if best else None,
            "decoded": decoded,
            "qubo_scale": scale,
            "ising_offset_internal": offset,
        }





if __name__ == "__main__":
    # cfg = EngineConfig(
    #     N=16,
    #     S=5,
    #     k=7,
    #     alpha=0.95,
    #     C=0.03,
    #     lambda_K=8.0,
    #     beta_return=1.0,
    #     gamma_risk=1.0,
    #     spsa_iters=10,  
    #     seed=13,
    #     DRAW_CIRCUIT=False,
    #     returns_npz_path="../data/scenario_1.npz"
    # )

    # engine = PortfolioQAOAEngine(cfg)
    # results = engine.run_hybrid_selector()
    

    # best = results["best_portfolio"]
    # print("=== HYBRID SELECTOR RESULTS ===")
    # print("Best QAOA params (gamma_1..gamma_p, beta_1..beta_p):")
    # print(results["best_params"])

    # if best is None:
    #     print("No CVaR-feasible portfolio found among sampled exact-k subsets.")
    # else:
    #     print("\nSelected subset indices:", best["subset_idx"].tolist())
    #     print("Subset bitstring x:", best["subset_bits"].astype(int).tolist())
    #     print("Classical weights on subset:", np.round(best["weights"], 6).tolist())
    #     print("Expected return:", best["expected_return"])
    #     print("Empirical CVaR:", best["empirical_cvar"])
    #     print("Selector energy:", best["selector_energy_scaled"])
    #     print("Sample count:", best["count"])
        
        
    cfg = EngineConfig(
        N=5,
        S=4,
        k=2,
        alpha=0.95,
        C=0.03,
        lambda_K=8.0,
        beta_return=1.0,
        gamma_risk=1.0,
        # Full QUBO bit
        Bw=5,
        Bt=1,
        Bv=1,
        Bxi=1,
        Beta=1,   
        spsa_iters=10,
        shots_train=512,
        shots_eval=1024,
        seed=7,
        DRAW_CIRCUIT=False,
        returns_npz_path="../data/scenario_2.npz"
    )

    engine = PortfolioQAOAEngine(cfg)
    results = engine.run_full_penalized()

    print("\n\n=== FULL-PENALIZED QAOA RESULTS ===")
    print("Best QAOA params (gamma_1..gamma_p, beta_1..beta_p):")
    print(results["best_params"])
    print("Number of qubits:", results["n_qubits"])

    best = results["best_sample"]
    decoded = results["decoded"]

    if best is None or decoded is None:
        print("No sampled solution returned.")
    else:
        print("\nBest sampled energy (scaled):", best["energy_scaled"])
        print("Best sampled energy (internal):", best["energy_internal"])
        print("Sample count:", best["count"])

        print("\nDecoded variables:")
        print("Subset bitstring x:", decoded["x"].astype(int).tolist())
        print("Quantum weights on subset:", np.round(decoded["w"], 6).tolist())
        print("t:", round(decoded["t"], 6))
        print("v:", np.round(decoded["v"], 6).tolist())
        print("xi:", round(decoded["xi"], 6))
        print("eta:", np.round(decoded["eta"], 6).tolist())
        print("sum_x:", decoded["sum_x"])
        print("sum_w:", round(decoded["sum_w"], 6))


        x = decoded["x"].astype(float)
        w = decoded["w"].astype(float)
        t = float(decoded["t"])
        v = decoded["v"].astype(float)
        xi = float(decoded["xi"])
        eta = decoded["eta"].astype(float)

        r = engine.returns            
        p_s = engine.p_s             
        alpha = cfg.alpha


        budget_residual = float(np.sum(w) - 1.0)
        cardinality_residual = float(np.sum(x) - cfg.k)
        cvar_budget_residual = float(t + (1.0 / (1.0 - alpha)) * np.dot(p_s, v) + xi - cfg.C)


        tail_residuals = v + (r @ w) + t - eta

        print("\nPenalty residual checks:")
        print("Budget residual (sum(w)-1):", round(budget_residual, 6))
        print("Cardinality residual (sum(x)-k):", round(cardinality_residual, 6))
        print("CVaR budget residual:", round(cvar_budget_residual, 6))
        print("Tail residuals:", np.round(tail_residuals, 6).tolist())
