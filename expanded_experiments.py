"""Expanded reproducible experiments for defect-sensitive PAVI.

The benchmark family is
    V_{a,J}(x) = sum_i (x_i^2/2 + a cos x_i)
                 + 0.5 tanh(x)^T J tanh(x),
where J is symmetric with zero diagonal and ||J||_op < 1.
The exact MFVI minimizer is pi_a^(tensor m).

The script generates all numerical figures and CSV tables used in the revised
manuscript.  It uses fixed seeds and only NumPy/Matplotlib.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import time
import numpy as np
import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Target1D:
    a: float
    grid: np.ndarray
    density: np.ndarray
    cdf: np.ndarray

    @classmethod
    def build(cls, a: float, radius: float = 8.0, size: int = 200_001) -> "Target1D":
        grid = np.linspace(-radius, radius, size)
        log_density = -0.5 * grid * grid - a * np.cos(grid)
        log_density -= np.max(log_density)
        density = np.exp(log_density)
        dx = grid[1] - grid[0]
        cdf = np.cumsum(density) * dx
        cdf /= cdf[-1]
        density /= np.trapezoid(density, grid)
        return cls(a=a, grid=grid, density=density, cdf=cdf)

    def quantiles(self, n: int) -> np.ndarray:
        probs = (np.arange(n) + 0.5) / n
        return np.interp(probs, self.cdf, self.grid)

    def quantiles_at(self, probs: np.ndarray) -> np.ndarray:
        return np.interp(probs, self.cdf, self.grid)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.interp(rng.random(n), self.cdf, self.grid)


def ring_interaction(m: int, strength: float) -> np.ndarray:
    """Symmetric ring matrix rescaled to have operator norm `strength`."""
    if m < 2:
        return np.zeros((m, m))
    if m == 2:
        return np.array([[0.0, strength], [strength, 0.0]])
    J = np.zeros((m, m), dtype=float)
    for i in range(m):
        J[i, (i - 1) % m] = 0.5 * strength
        J[i, (i + 1) % m] = 0.5 * strength
    return J


def sech2(x: np.ndarray) -> np.ndarray:
    t = np.tanh(x)
    return 1.0 - t * t


def product_w2(particles: np.ndarray, target_quantiles: np.ndarray) -> float:
    """Approximate W2 for a product empirical law and pi_a^{otimes m}."""
    if particles.ndim != 2 or particles.shape[1] != target_quantiles.size:
        raise ValueError("particles must have shape (m, len(target_quantiles)).")
    return float(np.sqrt(np.sum(np.mean((np.sort(particles, axis=1) - target_quantiles) ** 2, axis=1))))


def projected_drift(
    particles: np.ndarray,
    *,
    a: float,
    J: np.ndarray,
    batch_size: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Finite-batch projected drift; batch_size=None gives the exact empirical drift."""
    m, n = particles.shape
    if J.shape != (m, m):
        raise ValueError("J has incompatible shape.")
    if batch_size is None:
        mean_tanh = np.mean(np.tanh(particles), axis=1)
    else:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        indices = rng.integers(0, n, size=(m, batch_size))
        sampled = particles[np.arange(m)[:, None], indices]
        mean_tanh = np.mean(np.tanh(sampled), axis=1)
    interaction = J @ mean_tanh
    return particles - a * np.sin(particles) + sech2(particles) * interaction[:, None]


def pavi_step(
    particles: np.ndarray,
    *,
    a: float,
    J: np.ndarray,
    h: float,
    batch_size: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    drift = projected_drift(particles, a=a, J=J, batch_size=batch_size, rng=rng)
    return particles - h * drift + np.sqrt(2.0 * h) * rng.normal(size=particles.shape)


def rms_and_se(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sq = values * values
    mean_sq = np.mean(sq, axis=0)
    rms = np.sqrt(mean_sq)
    if values.shape[0] <= 1:
        return rms, np.zeros_like(rms)
    se_mean_sq = np.std(sq, axis=0, ddof=1) / np.sqrt(values.shape[0])
    se = np.divide(se_mean_sq, 2.0 * rms, out=np.zeros_like(rms), where=rms > 0)
    return rms, se


def run_pavi_trace(
    *,
    a: float,
    J: np.ndarray,
    n_particles: int,
    h: float,
    final_time: float,
    batch_size: int | None,
    replications: int,
    seed: int,
    record_count: int = 81,
    init_mode: str = "far",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Vectorized replications of PAVI for the benchmark family."""
    target = Target1D.build(a)
    q = target.quantiles(n_particles)
    steps = int(round(final_time / h))
    record_steps = np.unique(np.linspace(0, steps, record_count, dtype=int))
    times = h * record_steps
    traces = np.empty((replications, len(record_steps)), dtype=float)
    rng = np.random.default_rng(seed)
    m = J.shape[0]
    if init_mode == "far":
        X = rng.normal(loc=3.0, scale=0.35, size=(replications, m, n_particles))
    elif init_mode == "stationary":
        X = np.interp(rng.random((replications, m, n_particles)), target.cdf, target.grid)
    else:
        raise ValueError("init_mode must be far or stationary.")

    def errors(arr: np.ndarray) -> np.ndarray:
        sorted_arr = np.sort(arr, axis=2)
        sq = np.mean((sorted_arr - q[None, None, :]) ** 2, axis=2)
        return np.sqrt(np.sum(sq, axis=1))

    k = 0
    traces[:, k] = errors(X)
    k += 1
    start_time = time.perf_counter()
    for step in range(1, steps + 1):
        if batch_size is None:
            mean_tanh = np.mean(np.tanh(X), axis=2)
        else:
            indices = rng.integers(0, n_particles, size=(replications, m, batch_size))
            sampled = np.take_along_axis(X, indices, axis=2)
            mean_tanh = np.mean(np.tanh(sampled), axis=2)
        interaction = mean_tanh @ J.T
        drift = X - a * np.sin(X) + sech2(X) * interaction[:, :, None]
        X = X - h * drift + np.sqrt(2.0 * h) * rng.normal(size=X.shape)
        if k < len(record_steps) and step == record_steps[k]:
            traces[:, k] = errors(X)
            k += 1
    runtime = time.perf_counter() - start_time
    rms, se = rms_and_se(traces)
    return times, rms, se, float(runtime / replications)

def terminal_pavi_error(**kwargs: object) -> tuple[float, float, float]:
    t, rms, se, runtime = run_pavi_trace(record_count=2, **kwargs)
    del t
    return float(rms[-1]), float(se[-1]), runtime




def coupled_euler_error(
    *,
    a: float,
    J: np.ndarray,
    n_particles: int,
    h: float,
    h_ref: float,
    final_time: float,
    replications: int,
    seed: int,
) -> tuple[float, float, float]:
    """Coupled full-drift Euler error relative to a fine-step reference.

    Coarse Brownian increments are exact sums of the fine increments.  Both
    schemes start from the same stationary product-empirical array, so the
    diagnostic isolates time discretization rather than sampling error.
    """
    ratio_float = h / h_ref
    ratio = int(round(ratio_float))
    if ratio < 1 or not np.isclose(ratio_float, ratio):
        raise ValueError("h must be an integer multiple of h_ref.")
    coarse_steps = int(round(final_time / h))
    if not np.isclose(coarse_steps * h, final_time):
        raise ValueError("final_time must be an integer multiple of h.")

    target = Target1D.build(a)
    rng = np.random.default_rng(seed)
    m = J.shape[0]
    initial = np.interp(
        rng.random((replications, m, n_particles)), target.cdf, target.grid
    )
    coarse = initial.copy()
    fine = initial.copy()

    start = time.perf_counter()
    for _ in range(coarse_steps):
        brownian_sum = np.zeros_like(coarse)
        for _ in range(ratio):
            dW = np.sqrt(h_ref) * rng.normal(size=fine.shape)
            fine_means = np.mean(np.tanh(fine), axis=2)
            fine_interaction = fine_means @ J.T
            fine_drift = fine - a * np.sin(fine) + sech2(fine) * fine_interaction[:, :, None]
            fine = fine - h_ref * fine_drift + np.sqrt(2.0) * dW
            brownian_sum += dW
        coarse_means = np.mean(np.tanh(coarse), axis=2)
        coarse_interaction = coarse_means @ J.T
        coarse_drift = coarse - a * np.sin(coarse) + sech2(coarse) * coarse_interaction[:, :, None]
        coarse = coarse - h * coarse_drift + np.sqrt(2.0) * brownian_sum

    sorted_coarse = np.sort(coarse, axis=2)
    sorted_fine = np.sort(fine, axis=2)
    errors = np.sqrt(np.sum(np.mean((sorted_coarse - sorted_fine) ** 2, axis=2), axis=1))
    rms, se = rms_and_se(errors[:, None])
    runtime = (time.perf_counter() - start) / replications
    return float(rms[0]), float(se[0]), float(runtime)


def batch_estimator_rmse(
    *,
    a: float,
    J: np.ndarray,
    n_particles: int,
    batch_size: int,
    replications: int,
    seed: int,
) -> tuple[float, float, float]:
    """Root mean-square finite-batch drift error at stationary arrays."""
    target = Target1D.build(a)
    rng = np.random.default_rng(seed)
    m = J.shape[0]
    X = np.interp(rng.random((replications, m, n_particles)), target.cdf, target.grid)
    exact_means = np.mean(np.tanh(X), axis=2)
    exact_interaction = exact_means @ J.T
    exact = X - a * np.sin(X) + sech2(X) * exact_interaction[:, :, None]

    start = time.perf_counter()
    indices = rng.integers(0, n_particles, size=(replications, m, batch_size))
    sampled = np.take_along_axis(X, indices, axis=2)
    batch_means = np.mean(np.tanh(sampled), axis=2)
    batch_interaction = batch_means @ J.T
    estimate = X - a * np.sin(X) + sech2(X) * batch_interaction[:, :, None]
    per_rep = np.sqrt(np.mean(np.sum((estimate - exact) ** 2, axis=1), axis=1))
    rms, se = rms_and_se(per_rep[:, None])
    runtime = (time.perf_counter() - start) / replications
    return float(rms[0]), float(se[0]), float(runtime)


def density_quantiles(grid: np.ndarray, density: np.ndarray, probs: np.ndarray) -> np.ndarray:
    density = np.maximum(density, 0.0)
    dx = grid[1] - grid[0]
    cdf = np.cumsum(density) * dx
    cdf /= cdf[-1]
    return np.interp(probs, cdf, grid)


def run_cavi(
    *,
    a: float,
    J: np.ndarray,
    sweeps: int,
    grid_radius: float = 8.0,
    grid_size: int = 40_001,
    quantile_count: int = 2048,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sequential CAVI on a deterministic grid, initialized near N(3,0.35^2)."""
    m = J.shape[0]
    grid = np.linspace(-grid_radius, grid_radius, grid_size)
    dx = grid[1] - grid[0]
    base = np.exp(-0.5 * grid * grid - a * np.cos(grid))
    base /= np.sum(base) * dx
    init = np.exp(-0.5 * ((grid - 3.0) / 0.35) ** 2)
    init /= np.sum(init) * dx
    densities = np.tile(init[None, :], (m, 1))
    moments = np.sum(densities * np.tanh(grid)[None, :], axis=1) * dx
    probs = (np.arange(quantile_count) + 0.5) / quantile_count
    target = density_quantiles(grid, base, probs)
    errors = np.empty(sweeps + 1)

    def current_error() -> float:
        sq = 0.0
        for i in range(m):
            qi = density_quantiles(grid, densities[i], probs)
            sq += float(np.mean((qi - target) ** 2))
        return float(np.sqrt(sq))

    errors[0] = current_error()
    start = time.perf_counter()
    tanh_grid = np.tanh(grid)
    for sweep in range(1, sweeps + 1):
        for i in range(m):
            field = float(J[i] @ moments)
            dens = base * np.exp(-field * tanh_grid)
            dens /= np.sum(dens) * dx
            densities[i] = dens
            moments[i] = np.sum(dens * tanh_grid) * dx
        errors[sweep] = current_error()
    runtime = time.perf_counter() - start
    return np.arange(sweeps + 1), errors, runtime


def generic_drift_cost(
    n_particles: int,
    batch_size: int,
    *,
    a: float = 2.0,
    lam: float = 0.5,
    repeats: int = 30,
    seed: int = 123,
) -> float:
    """Measured cost of the generic O(m N B) drift evaluation for m=2."""
    rng = np.random.default_rng(seed + n_particles + batch_size)
    X = rng.normal(size=(2, n_particles))
    batch = X[np.arange(2)[:, None], rng.integers(0, n_particles, size=(2, batch_size))]
    start = time.perf_counter()
    for _ in range(repeats):
        drift = np.empty_like(X)
        # Coordinate 0 evaluated on all N x B hybrid states.
        x0 = X[0, :, None]
        y1 = batch[1, None, :]
        vals0 = x0 - a * np.sin(x0) + lam * sech2(x0) * np.tanh(y1)
        drift[0] = np.mean(vals0, axis=1)
        # Coordinate 1.
        x1 = X[1, :, None]
        y0 = batch[0, None, :]
        vals1 = x1 - a * np.sin(x1) + lam * sech2(x1) * np.tanh(y0)
        drift[1] = np.mean(vals1, axis=1)
        if not np.all(np.isfinite(drift)):
            raise FloatingPointError
    return (time.perf_counter() - start) / repeats


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path.with_suffix(".pdf"))
    plt.savefig(path.with_suffix(".png"), dpi=220)
    plt.close()


def make_all(root: Path) -> None:
    figures = root / "figures"
    results = root / "results"
    figures.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)

    a = 2.0
    lam = 0.5
    J = ring_interaction(2, lam)
    seed = 20260717

    # 1. Convergence of finite-batch and full-gradient PAVI.
    rows: list[list[object]] = []
    plt.figure(figsize=(6.2, 4.15))
    for idx, B in enumerate((1, 4, 16, None)):
        t, rms, se, runtime = run_pavi_trace(
            a=a, J=J, n_particles=384, h=0.004, final_time=6.0,
            batch_size=B, replications=12, seed=seed + 13 * idx,
        )
        label = "full empirical drift" if B is None else f"B={B}"
        plt.plot(t, rms, label=label)
        plt.fill_between(t, np.maximum(rms - 2 * se, 0.0), rms + 2 * se, alpha=0.12)
        rows.extend([[label, float(tt), float(rr), float(ss), runtime]
                     for tt, rr, ss in zip(t, rms, se)])
    plt.xlabel("physical time $nh$")
    plt.ylabel(r"root mean-square $W_2$ error")
    plt.legend(frameon=False)
    save_figure(figures / "convergence_methods")
    write_csv(results / "convergence_methods.csv",
              ["method", "time", "rms_w2", "standard_error", "mean_runtime_seconds"], rows)

    # 2. Actual terminal PAVI error versus N.
    rows = []
    Ns = np.array([64, 128, 256, 512, 768])
    vals, ses = [], []
    for idx, N in enumerate(Ns):
        val, se, runtime = terminal_pavi_error(
            a=a, J=J, n_particles=int(N), h=0.002, final_time=2.0,
            batch_size=16, replications=10, seed=seed + 100 + idx, init_mode="stationary",
        )
        vals.append(val); ses.append(se)
        rows.append([int(N), val, se, runtime])
    vals_arr = np.asarray(vals); ses_arr = np.asarray(ses)
    slope_N, _ = np.polyfit(np.log(Ns), np.log(vals_arr), 1)
    plt.figure(figsize=(5.7, 4.0))
    plt.errorbar(Ns, vals_arr, yerr=2 * ses_arr, fmt="o", capsize=3, label="terminal PAVI error")
    ref = vals_arr[0] * (Ns / Ns[0]) ** (-0.5)
    plt.plot(Ns, ref, "--", label=r"reference $N^{-1/2}$")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("particles per marginal $N$")
    plt.ylabel(r"terminal root mean-square $W_2$ error")
    plt.legend(frameon=False)
    save_figure(figures / "pavi_particle_scaling")
    for row in rows:
        row.append(float(slope_N))
    write_csv(results / "pavi_particle_scaling.csv",
              ["n_particles", "rms_w2", "standard_error", "runtime_seconds", "loglog_slope"], rows)

    # 3. Coupled coarse/fine Euler discretization error.
    rows = []
    hs = np.array([0.008, 0.004, 0.002, 0.001])
    h_ref = 0.0005
    vals, ses = [], []
    for idx, h in enumerate(hs):
        val, se, runtime = coupled_euler_error(
            a=a, J=J, n_particles=1024, h=float(h), h_ref=h_ref,
            final_time=0.512, replications=12, seed=seed + 900 + idx,
        )
        vals.append(val); ses.append(se); rows.append([h, h_ref, val, se, runtime])
    vals_arr_h = np.asarray(vals)
    slope_h, _ = np.polyfit(np.log(hs), np.log(vals_arr_h), 1)
    plt.figure(figsize=(5.7, 4.0))
    plt.errorbar(hs, vals_arr_h, yerr=2 * np.asarray(ses), fmt="o", capsize=3,
                 label="coupled coarse/fine error")
    ref_h = vals_arr_h[-1] * (hs / hs[-1])
    plt.plot(hs, ref_h, "--", label=r"reference $h$")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("coarse step size $h$")
    plt.ylabel(r"RMS $W_2$ error to $h_{\rm ref}=5\times10^{-4}$")
    plt.legend(frameon=False)
    save_figure(figures / "stepsize_scaling")
    for row in rows:
        row.append(float(slope_h))
    write_csv(results / "stepsize_scaling.csv",
              ["step_size", "reference_step", "rms_w2", "standard_error",
               "runtime_seconds", "loglog_slope"], rows)

    # 4. Finite-batch drift-estimator error.
    rows = []
    batch_values = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    vals, ses = [], []
    for idx, B in enumerate(batch_values):
        val, se, runtime = batch_estimator_rmse(
            a=a, J=J, n_particles=2048, batch_size=int(B),
            replications=256, seed=seed + 300 + idx,
        )
        vals.append(val); ses.append(se); rows.append([int(B), val, se, runtime])
    vals_arr_B = np.asarray(vals)
    slope_B, _ = np.polyfit(np.log(batch_values), np.log(vals_arr_B), 1)
    plt.figure(figsize=(5.9, 4.0))
    plt.errorbar(batch_values, vals_arr_B, yerr=2 * np.asarray(ses), fmt="o", capsize=3,
                 label="batch drift RMSE")
    ref_B = vals_arr_B[0] * batch_values ** (-0.5)
    plt.plot(batch_values, ref_B, "--", label=r"reference $B^{-1/2}$")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("batch size $B$")
    plt.ylabel("RMS projected-drift estimation error")
    plt.legend(frameon=False)
    save_figure(figures / "batch_scaling")
    for row in rows:
        row.append(float(slope_B))
    write_csv(results / "batch_scaling.csv",
              ["batch_size", "rms_drift_error", "standard_error", "runtime_seconds",
               "loglog_slope"], rows)

    # 5. Dependence on the conservative uniform defect bound through a.
    rows = []
    avals = np.array([1.05, 1.6, 2.2, 3.0])
    beta_bounds = 8.0 * (avals + lam) ** 2
    vals, ses = [], []
    for idx, aa in enumerate(avals):
        val, se, runtime = terminal_pavi_error(
            a=float(aa), J=J, n_particles=384, h=0.002, final_time=6.0,
            batch_size=16, replications=8, seed=seed + 400 + idx,
        )
        vals.append(val); ses.append(se)
        rows.append([aa, beta_bounds[idx], val, se, runtime])
    plt.figure(figsize=(5.9, 4.0))
    plt.errorbar(beta_bounds, vals, yerr=2 * np.asarray(ses), fmt="o-", capsize=3)
    plt.xlabel(r"conservative uniform-defect bound $\beta=8(a+\lambda)^2$")
    plt.ylabel(r"terminal root mean-square $W_2$ error")
    save_figure(figures / "defect_scaling")
    write_csv(results / "defect_scaling.csv",
              ["a", "beta_bound", "rms_w2", "standard_error", "runtime_seconds"], rows)

    # 6. CAVI comparison on the same exact-MFVI benchmark.
    sweeps, cavi_errors, cavi_runtime = run_cavi(a=a, J=J, sweeps=25)
    plt.figure(figsize=(5.8, 4.0))
    plt.semilogy(sweeps, cavi_errors, "o-")
    plt.xlabel("CAVI sweeps")
    plt.ylabel(r"$W_2$ error to $q^\star$")
    save_figure(figures / "cavi_convergence")
    write_csv(results / "cavi_convergence.csv",
              ["sweep", "w2_error", "total_runtime_seconds"],
              [[int(k), float(e), cavi_runtime] for k, e in zip(sweeps, cavi_errors)])

    # 7. Generic O(m N B) drift cost scaling.
    rows = []
    work, runtime_vals = [], []
    for N in (256, 512, 1024, 2048, 4096):
        for B in (8, 32, 128, 256):
            rt = generic_drift_cost(N, B, repeats=8, seed=seed)
            w = 2 * N * B
            work.append(w); runtime_vals.append(rt)
            rows.append([N, B, w, rt])
    work_arr = np.asarray(work, dtype=float)
    runtime_arr = np.asarray(runtime_vals)
    mask = work_arr >= 32768
    slope_cost, intercept = np.polyfit(np.log(work_arr[mask]), np.log(runtime_arr[mask]), 1)
    order = np.argsort(work_arr)
    plt.figure(figsize=(5.8, 4.0))
    plt.loglog(work_arr, runtime_arr, "o", label="measured")
    fitted = np.exp(intercept) * work_arr[order] ** slope_cost
    plt.loglog(work_arr[order], fitted, "--", label=f"fit slope {slope_cost:.2f}")
    plt.xlabel(r"partial-gradient workload $mNB$")
    plt.ylabel("seconds per drift evaluation")
    plt.legend(frameon=False)
    save_figure(figures / "cost_scaling")
    for row in rows:
        row.append(float(slope_cost))
    write_csv(results / "cost_scaling.csv",
              ["n_particles", "batch_size", "mNB", "seconds", "loglog_slope"], rows)

    # 8. Dimension scaling for the ring benchmark.
    rows = []
    dims = np.array([2, 4, 8, 16])
    vals, ses = [], []
    for idx, m in enumerate(dims):
        Jm = ring_interaction(int(m), lam)
        val, se, runtime = terminal_pavi_error(
            a=a, J=Jm, n_particles=256, h=0.002, final_time=2.0,
            batch_size=16, replications=8, seed=seed + 600 + idx, init_mode="stationary",
        )
        vals.append(val); ses.append(se); rows.append([int(m), val, se, runtime])
    plt.figure(figsize=(5.7, 4.0))
    plt.errorbar(dims, vals, yerr=2 * np.asarray(ses), fmt="o-", capsize=3)
    plt.xlabel("dimension $m$")
    plt.ylabel(r"terminal root mean-square $W_2$ error")
    save_figure(figures / "dimension_scaling")
    write_csv(results / "dimension_scaling.csv",
              ["dimension", "rms_w2", "standard_error", "runtime_seconds"], rows)

    print(f"PAVI N-scaling slope: {slope_N:.3f}")
    print(f"Euler coarse/fine slope: {slope_h:.3f}")
    print(f"Batch estimator slope: {slope_B:.3f}")
    print(f"Generic cost slope: {slope_cost:.3f}")
    print(f"CAVI runtime: {cavi_runtime:.3f} seconds")


if __name__ == "__main__":
    make_all(Path(__file__).resolve().parents[1])
