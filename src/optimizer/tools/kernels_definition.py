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

        # Original: (2/κ) × exp(-s/α) × sinh(κs/α)
        # Extended: (2/κ) × exp(-s/α) × [exp(κs/α) - exp(-κs/α)] / 2
        #          = [exp(s(κ-1)/α) - exp(-s(κ+1)/α)] / κ
    
        exp_arg_pos = s * (kappa - 1) / alpha  # s(κ-1)/α
        exp_arg_neg = -s * (kappa + 1) / alpha  # -s(κ+1)/α
        
        with np.errstate(over='ignore', invalid='ignore'):
            term_pos = np.exp(exp_arg_pos)
            term_neg = np.exp(exp_arg_neg)
        
        result = (term_pos - term_neg) / kappa
        
        overflow_mask = np.isinf(term_pos) | np.isnan(result)
        if np.any(overflow_mask):
            result[overflow_mask] = np.inf if kappa > 1 else 0.0
        
        return result

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
