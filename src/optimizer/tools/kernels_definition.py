import numpy as np

def K_beta_second_order(
    s: np.ndarray,
    beta: float,
    alpha: float,
) -> np.ndarray:
    r"""
    Closed form for K_beta(s),  s ≥ 0:
    Parameters
    ----------
    s : np.ndarray
        Nonnegative argument(s) s ≥ 0.
    beta : float
        Parameter beta.
    alpha : float
        Parameter alpha.
    """
    s = np.asarray(s, dtype=float)
    
    if beta > 0.5:
        kappa = np.sqrt(2 * beta - 1)
        return (2.0 / kappa) * np.exp(-s / alpha) * np.sinh((kappa / alpha) * s)

    if np.isclose(beta, 0.5):
        return (2.0 * s / alpha) * np.exp(-s / alpha)

    # beta < 1/2
    rho = np.sqrt(1 - 2 * beta)
    return (2.0 / rho) * np.exp(-s / alpha) * np.sin((rho / alpha) * s)


def K_beta_first_order(
    s: np.ndarray,
    beta: float,
    alpha: float,
) -> np.ndarray:
    r"""
    Closed form for K_beta(s),  s ≥ 0.
    Parameters
    ----------
    s : np.ndarray
        Nonnegative argument(s) s ≥ 0.
    beta : float
        Parameter beta.
    alpha : float
        Parameter alpha.
    """
    s = np.asarray(s, dtype=float)
    return np.exp(-((1.0 - beta) / alpha) * s)
