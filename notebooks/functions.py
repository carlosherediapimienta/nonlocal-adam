import numpy as np
from typing import Tuple

## Functions for the analysis
def dL_rosenbrock_1d(
    x: np.ndarray,
    function_parameters: Tuple[float, ...] = ()
) -> np.ndarray:
    """
    Gradient of Rosenbrock 1D:
    L(x) = (1 - x)^2 + c*(x^2 - 1)^2
    """
    (c,) = function_parameters
    return 2*(x - 1) + 4*c*x*(x**2 - 1)