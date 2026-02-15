import numpy as np
from scipy.optimize import least_squares
from typing import Optional

_DEFAULT_MICS = np.array([[0.0, 0.0, 0.0],
                          [-0.2, -0.4, 0.0],
                          [-0.2, 0.4, 0.0]])

_DEFAULT_MICS_2D = np.array([[0.0, 0.0], [-0.04, -0.02], [0.04, -0.02]])

def tdoa_using_ls(
    tdoa: np.ndarray,
    mic_positions: Optional[np.ndarray] = None,
    c: float = 343.0,
    prior_mean: Optional[np.ndarray] = None,
    prior_cov: Optional[np.ndarray] = None,   # covariance, not inverse
    x_min: Optional[float] = None,
    z_prior_sigma: Optional[float] = None,    # meters
    x0: Optional[np.ndarray] = None,
    robust: bool = True,
) -> np.ndarray:
    mics = np.asarray(mic_positions if mic_positions is not None else _DEFAULT_MICS, dtype=float)
    tdoa = np.asarray(tdoa, dtype=float).reshape(-1)

    if mics.shape != (3, 3):
        raise ValueError(f"mic_positions must be (3,3), got {mics.shape}")
    if tdoa.shape != (2,):
        raise ValueError(f"tdoa must be (2,), got {tdoa.shape}")

    if x0 is None:
        x0 = np.mean(mics, axis=0) + np.array([0.0, 0.5, 0.0])

    m0, m1, m2 = mics

    # Precompute whitening factor for prior if provided
    prior_L = None
    if prior_mean is not None:
        prior_mean = np.asarray(prior_mean, dtype=float).reshape(3)
        if prior_cov is None:
            raise ValueError("prior_cov must be provided when prior_mean is set")
        prior_cov = np.asarray(prior_cov, dtype=float)
        # prior_cov must be SPD
        prior_L = np.linalg.cholesky(prior_cov)

    def residuals(p: np.ndarray) -> np.ndarray:
        d0 = np.linalg.norm(p - m0)
        d1 = np.linalg.norm(p - m1)
        d2 = np.linalg.norm(p - m2)

        r = np.array([
            (d1 - d0) / c - tdoa[0],
            (d2 - d0) / c - tdoa[1],
        ], dtype=float)

        if prior_L is not None:
            diff = p - prior_mean
            # whiten: y = L^{-1} diff
            y = np.linalg.solve(prior_L, diff)
            r = np.concatenate([r, y])

        if z_prior_sigma is not None and z_prior_sigma > 0:
            r = np.concatenate([r, [p[2] / z_prior_sigma]])

        return r

    kw = {}
    if x_min is not None:
        kw["bounds"] = ([x_min, -np.inf, -np.inf], [np.inf, np.inf, np.inf])

    if robust:
        kw["loss"] = "soft_l1"

    res = least_squares(residuals, x0, **kw)
    return res.x


def tdoa_using_ls_2D(
    tdoa: np.ndarray,
    mic_positions: Optional[np.ndarray] = None,
    c: float = 343.0,
    prior_mean: Optional[np.ndarray] = None,
    prior_cov: Optional[np.ndarray] = None,
    x_min: Optional[float] = None,
    x0: Optional[np.ndarray] = None,
    robust: bool = True,
) -> np.ndarray:
    """TDOA source localization in 2D (x, y). Mics (3, 2), source (2,)."""
    mics = np.asarray(mic_positions if mic_positions is not None else _DEFAULT_MICS_2D, dtype=float)
    tdoa = np.asarray(tdoa, dtype=float).reshape(-1)
    if mics.shape != (3, 2):
        raise ValueError(f"mic_positions must be (3,2), got {mics.shape}")
    if tdoa.shape != (2,):
        raise ValueError(f"tdoa must be (2,), got {tdoa.shape}")

    if x0 is None:
        x0 = np.mean(mics, axis=0) + np.array([0.0, 0.5])
    m0, m1, m2 = mics

    prior_L = None
    if prior_mean is not None:
        prior_mean = np.asarray(prior_mean, dtype=float).reshape(2)
        if prior_cov is None:
            raise ValueError("prior_cov must be provided when prior_mean is set")
        prior_L = np.linalg.cholesky(np.asarray(prior_cov, dtype=float))

    def residuals(p: np.ndarray) -> np.ndarray:
        d0 = np.linalg.norm(p - m0)
        d1 = np.linalg.norm(p - m1)
        d2 = np.linalg.norm(p - m2)
        r = np.array([(d1 - d0) / c - tdoa[0], (d2 - d0) / c - tdoa[1]], dtype=float)
        if prior_L is not None:
            r = np.concatenate([r, np.linalg.solve(prior_L, p - prior_mean)])
        return r

    kw = {}
    if x_min is not None:
        kw["bounds"] = ([x_min, -np.inf], [np.inf, np.inf])
    if robust:
        kw["loss"] = "soft_l1"
    res = least_squares(residuals, x0, **kw)
    return res.x


def tdoa_using_grid_search(
    tdoa: np.ndarray,
    mic_positions: Optional[np.ndarray] = None,
    c: float = 343.0,
    distance: float = 1,
    n_points: int = 9,
) -> tuple[np.ndarray, float]:
    """
    SRP-PHAT style: grid of candidate positions at z=0, x>0, radius=distance.
    Returns the grid point whose theoretical TDOA best matches observed tdoa, and a similarity in [0,1].
    """
    mics = np.asarray(mic_positions if mic_positions is not None else _DEFAULT_MICS, dtype=float)
    tdoa = np.asarray(tdoa, dtype=float).reshape(2)
    if mics.shape != (3, 3):
        raise ValueError(f"mic_positions must be (3,3), got {mics.shape}")

    # 9 points at z=0, x>0, on circle of radius distance (angles -90 to +90 deg in xy)
    angles = np.linspace(-np.pi / 2, np.pi / 2, n_points)
    grid = np.column_stack([distance * np.cos(angles), distance * np.sin(angles), np.zeros(n_points)])
    print(grid)

    m0, m1, m2 = mics
    best_idx = 0
    best_err = np.inf
    for i, p in enumerate(grid):
        d0 = np.linalg.norm(p - m0)
        d1 = np.linalg.norm(p - m1)
        d2 = np.linalg.norm(p - m2)
        tdoa_pred = np.array([(d1 - d0) / c, (d2 - d0) / c])
        err = np.linalg.norm(tdoa_pred - tdoa)
        if err < best_err:
            best_err = err
            best_idx = i

    best_point = grid[best_idx]
    similarity = 1.0 / (1.0 + best_err)
    return best_point, similarity