from typing import Callable
import numpy as np
from scipy.integrate import fixed_quad

class IntegrationQuadrature:
    """
    Integration quadrature class using Gauss-Legendre quadrature.
    
    Parameters
    ----------
    n : int
        Quadrature order (number of nodes/weights).
    tol : float, optional
        Threshold below which intervals are treated as zero-length. Default is 1e-12.
    verbose : bool, optional
        If True, prints intermediate results. Default is False.
    """
    def __init__(self, n: int, tol: float = 1e-12, verbose: bool = False):
        self.n = n
        self.tol = tol
        self.verbose = verbose
    
    def integrate(self, fun: Callable, lo: float, hi: float) -> float:
        """
        Integrates function over a single interval [lo, hi].
        
        Parameters
        ----------
        fun : Callable
            Function to integrate.
        lo : float
            Lower bound.
        hi : float
            Upper bound.
            
        Returns
        -------
        float
            Integral value.
        """

        if np.abs(hi - lo) < self.tol:
            return 0.

        result, _ = fixed_quad(fun, lo, hi, n=self.n)
    
        return result