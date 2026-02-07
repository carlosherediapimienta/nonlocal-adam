"""
Explicit Euler methods for ODEs: y' = f(...) or y'' = f(...).
"""
from typing import Callable, Tuple, Any, Union
import numpy as np

DTYPE = np.float64

class EulerMethod:
    """
    Euler Integrator for ODEs.
    """
    
    def __init__(self):
        pass

    def _forward(self,
                 alpha: float,
                 y0: Union[float, np.ndarray],
                 t_vec: np.ndarray,
                 rhs: Callable,
                 func_rhs: Tuple[Any, ...]) -> np.ndarray:
        """Forward Euler: y_{n+1} = y_n + h * f(t_n, y_n)"""
        alpha = DTYPE(alpha)
        is_scalar = np.ndim(y0) == 0
        n = len(t_vec)
        
        # Initialize
        if is_scalar:
            y = np.zeros(n, dtype=DTYPE)
            y[0] = DTYPE(y0)
            dy_hist = np.zeros(n, dtype=DTYPE)
        else:
            y = np.zeros((n, len(y0)), dtype=DTYPE)
            y[0] = np.asarray(y0, dtype=DTYPE)
            dy_hist = np.zeros((n, len(y0)), dtype=DTYPE)
        
        # Integrate
        for i in range(n - 1):
            dy = rhs(y[i], i, *func_rhs)
            dy_hist[i] = dy
            y[i + 1] = y[i] + (alpha/5) * dy
                
        dy_hist[-1] = rhs(y[-1], n - 1, *func_rhs)
        
        return y, dy_hist
    
    def solve(self,
                  alpha: float,
                  y0: Union[float, np.ndarray],
                  t_vec: np.ndarray,
                  rhs: Callable,
                  func_rhs: Tuple[Any, ...] = ()) -> np.ndarray:
        """
        Integrates y' = rhs(t, y, ...) over t_vec.
        
        Parameters
        ----------
        alpha : float
            Time step.
        y0 : float or array
            Initial condition (scalar for y', vector [y, y'] for y'').
        t_vec : ndarray
            Evaluation times.
        rhs : Callable
            Right-hand side: rhs(y, idx, *func_rhs) -> dy/dt
        func_rhs : tuple
            Extra functions for rhs.
        
        Returns
        -------
        (y, dy) : tuple of ndarrays
            Returns state and derivatives.
        """
        return self._forward(alpha, y0, t_vec, rhs, func_rhs)
    

    