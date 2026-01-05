"""
Explicit Euler methods for ODEs: y' = f(...) or y'' = f(...).
"""
from typing import Callable, Tuple, Any, Union, Literal
import numpy as np

DTYPE = np.float64

class EulerMethod:
    """
    Integrador Euler explícito para EDOs.
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
        
        # Inicializar
        if is_scalar:
            y = np.zeros(n, dtype=DTYPE)
            y[0] = DTYPE(y0)
            dy_hist = np.zeros(n, dtype=DTYPE)
        else:
            y = np.zeros((n, len(y0)), dtype=DTYPE)
            y[0] = np.asarray(y0, dtype=DTYPE)
            dy_hist = np.zeros((n, len(y0)), dtype=DTYPE)
        
        # Integrar
        for i in range(n - 1):
            dy = rhs(y[i], i, *func_rhs)
            dy_hist[i] = dy
            y[i + 1] = y[i] + alpha * dy
                
        dy_hist[-1] = rhs(y[-1], n - 1, *func_rhs)
        
        return y, dy_hist
    
    def solve(self,
                  alpha: float,
                  y0: Union[float, np.ndarray],
                  t_vec: np.ndarray,
                  rhs: Callable,
                  func_rhs: Tuple[Any, ...] = ()) -> np.ndarray:
        """
        Integra y' = rhs(t, y, ...) sobre t_vec.
        
        Parameters
        ----------
        alpha : float
            Paso temporal.
        y0 : float or array
            Condición inicial (escalar para y', vector [y, y'] para y'').
        t_vec : ndarray
            Tiempos de evaluación.
        rhs : Callable
            Lado derecho: rhs(y, idx, *func_rhs) -> dy/dt
        func_rhs : tuple
            Funciones extra para rhs.
        
        Returns
        -------
        (y, dy) : Tupla de ndarrays
            Devuelve estado y derivadas.
        """
        return self._forward(alpha, y0, t_vec, rhs, func_rhs)
    

    