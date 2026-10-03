"""2D checkerboard distribution shared by the diffusion and flow-matching models.

The board covers [-half_width, half_width]^2 and is split into n_squares x n_squares
cells; samples are uniform over the "black" cells, i.e. cells (i, j) with (i + j) even.
With the defaults (4x4 board on [-2, 2]^2) the per-coordinate std is ~1.15, close
enough to the unit-variance Gaussian prior that no extra normalization is needed.

Sampling is exact (pick a black cell, then a uniform point inside it), so every
sample lies in the support. That makes `support_fraction` a clean quality metric:
a perfect generative model scores 1.0, while an isotropic Gaussian fit scores ~0.5.
"""

import numpy as np


def _black_cells(n_squares: int) -> np.ndarray:
    """Integer (i, j) indices of the cells that carry probability mass."""
    i, j = np.meshgrid(np.arange(n_squares), np.arange(n_squares), indexing="ij")
    mask = (i + j) % 2 == 0
    return np.stack([i[mask], j[mask]], axis=1)


def sample_checkerboard(
    n: int,
    rng: np.random.Generator,
    n_squares: int = 4,
    half_width: float = 2.0,
) -> np.ndarray:
    """Draw n points uniformly from the black squares. Returns float32 array (n, 2)."""
    cells = _black_cells(n_squares)
    side = 2.0 * half_width / n_squares
    picked = cells[rng.integers(len(cells), size=n)]
    offsets = rng.random((n, 2))
    x = -half_width + (picked + offsets) * side
    return x.astype(np.float32)


def in_support(
    x: np.ndarray, n_squares: int = 4, half_width: float = 2.0
) -> np.ndarray:
    """Boolean mask: which points fall inside a black square of the board."""
    side = 2.0 * half_width / n_squares
    idx = np.floor((x + half_width) / side).astype(int)
    inside_board = np.all((idx >= 0) & (idx < n_squares), axis=1)
    return inside_board & (idx.sum(axis=1) % 2 == 0)


def support_fraction(
    x: np.ndarray, n_squares: int = 4, half_width: float = 2.0
) -> float:
    """Fraction of generated samples that land on the true support (1.0 is perfect)."""
    return float(in_support(x, n_squares, half_width).mean())


def make_checkerboard_splits(
    n_train: int = 100_000,
    n_ref: int = 20_000,
    seed: int = 0,
    **board_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Training set plus an independent held-out reference set for sample-quality metrics."""
    rng = np.random.default_rng(seed)
    train = sample_checkerboard(n_train, rng, **board_kwargs)
    ref = sample_checkerboard(n_ref, rng, **board_kwargs)
    return train, ref
