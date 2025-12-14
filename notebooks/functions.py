import numpy as np
from typing import Tuple

## Functions for the analysis
def dL_rosenbrock_1d(
    x: np.ndarray,
    function_parameters: Tuple[float, ...] = ()
) -> np.ndarray:
    """
    Gradient of Rosenbrock 1D:
    L(x) = (1 - x)^2 + a*(x^2 - 1)^2
    """
    (a,) = function_parameters
    return 2*(x - 1) + 4*a*x*(x**2 - 1)