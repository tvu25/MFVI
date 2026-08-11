"""Reference NumPy implementation of finite-batch PAVI.

The implementation follows Algorithm 1 in the manuscript.  The user supplies
``partial_gradient(x, i)`` returning the i-th partial derivative of V at x.
Coordinates are indexed from zero in Python.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
PartialGradient = Callable[[Array, int], float]


@dataclass(frozen=True)
class PAVIConfig:
    step_size: float
    batch_size: int
    iterations: int
    seed: int | None = None

    def validate(self) -> None:
        if self.step_size <= 0:
            raise ValueError("step_size must be positive.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least one.")
        if self.iterations < 0:
            raise ValueError("iterations must be nonnegative.")


def sample_product_empirical(
    particles: Array,
    batch_size: int,
    rng: np.random.Generator,
) -> Array:
    """Draw samples from the product of the row empirical measures.

    Parameters
    ----------
    particles:
        Array of shape (m, N).
    batch_size:
        Number of product samples.
    rng:
        NumPy random generator.

    Returns
    -------
    Array of shape (m, batch_size).  Indices are drawn independently in each
    row.  Drawing common column indices would sample the column empirical law,
    not the product empirical law required by PAVI.
    """
    if particles.ndim != 2:
        raise ValueError("particles must have shape (m, N).")
    if batch_size < 1:
        raise ValueError("batch_size must be at least one.")
    m, n = particles.shape
    if n < 1:
        raise ValueError("particles must contain at least one particle per row.")
    row_indices = np.arange(m)[:, None]
    column_indices = rng.integers(0, n, size=(m, batch_size))
    return particles[row_indices, column_indices]


def estimate_projected_drifts(
    particles: Array,
    batch: Array,
    partial_gradient: PartialGradient,
) -> Array:
    """Evaluate the finite-batch projected drift at every particle."""
    if particles.ndim != 2 or batch.ndim != 2:
        raise ValueError("particles and batch must both be two-dimensional.")
    m, n = particles.shape
    if batch.shape[0] != m:
        raise ValueError("particles and batch must have the same number of rows.")
    bsize = batch.shape[1]
    if bsize < 1:
        raise ValueError("batch must contain at least one sample.")

    drift = np.empty_like(particles, dtype=float)
    work = np.empty(m, dtype=float)
    for i in range(m):
        for j in range(n):
            total = 0.0
            for b in range(bsize):
                work[:] = batch[:, b]
                work[i] = particles[i, j]
                value = float(partial_gradient(work, i))
                if not np.isfinite(value):
                    raise FloatingPointError("partial_gradient returned a nonfinite value.")
                total += value
            drift[i, j] = total / bsize
    return drift


def pavi_step(
    particles: Array,
    *,
    step_size: float,
    batch_size: int,
    partial_gradient: PartialGradient,
    rng: np.random.Generator,
) -> Array:
    """Perform one finite-batch PAVI update."""
    if step_size <= 0:
        raise ValueError("step_size must be positive.")
    batch = sample_product_empirical(particles, batch_size, rng)
    drift = estimate_projected_drifts(particles, batch, partial_gradient)
    noise = np.sqrt(2.0 * step_size) * rng.normal(size=particles.shape)
    updated = particles - step_size * drift + noise
    if not np.all(np.isfinite(updated)):
        raise FloatingPointError("PAVI produced nonfinite particles.")
    return updated


def run_pavi(
    initial_particles: Array,
    *,
    config: PAVIConfig,
    partial_gradient: PartialGradient,
    save_every: int | None = None,
) -> tuple[Array, list[Array]]:
    """Run finite-batch PAVI and optionally save a trajectory.

    Returns the final array and a list of saved copies.  The first saved array
    is the initial condition when ``save_every`` is provided.
    """
    config.validate()
    particles = np.asarray(initial_particles, dtype=float).copy()
    if particles.ndim != 2 or particles.shape[1] < 1:
        raise ValueError("initial_particles must have shape (m, N), N >= 1.")
    if not np.all(np.isfinite(particles)):
        raise ValueError("initial_particles must be finite.")
    if save_every is not None and save_every < 1:
        raise ValueError("save_every must be positive when provided.")

    rng = np.random.default_rng(config.seed)
    trajectory: list[Array] = []
    if save_every is not None:
        trajectory.append(particles.copy())

    for iteration in range(1, config.iterations + 1):
        particles = pavi_step(
            particles,
            step_size=config.step_size,
            batch_size=config.batch_size,
            partial_gradient=partial_gradient,
            rng=rng,
        )
        if save_every is not None and iteration % save_every == 0:
            trajectory.append(particles.copy())

    return particles, trajectory


if __name__ == "__main__":
    # Minimal smoke test with V(x)=||x||^2/2.
    def gaussian_partial_gradient(x: Array, i: int) -> float:
        return float(x[i])

    initial = np.zeros((3, 128), dtype=float)
    cfg = PAVIConfig(step_size=0.01, batch_size=1, iterations=100, seed=1234)
    final, _ = run_pavi(initial, config=cfg, partial_gradient=gaussian_partial_gradient)
    print(final.shape, np.mean(final), np.var(final))
