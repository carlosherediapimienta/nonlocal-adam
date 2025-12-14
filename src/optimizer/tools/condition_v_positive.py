import numpy as np
from scipy.ndimage import minimum_filter1d, maximum_filter1d

class ConditionVPositive:
    def __init__(self,  alpha: float):
        self.alpha = alpha       

    def __call__(self, beta2: float, r: np.ndarray, t: np.ndarray, v: np.ndarray) -> bool:
        viol = np.zeros_like(t, dtype=bool)

        if beta2 >= 0.5:
            return viol
        
        rho = np.sqrt(1.0 - 2.0 * beta2)          
        q = np.exp(-np.pi / rho)
        delta = np.pi * self.alpha / rho              
        W = int(np.ceil(delta / self.alpha))

        if W <= 0 or len(r) <= 1:
            return viol

        window_size = min(W + 1, len(r))
        r_padded = np.pad(r, (window_size-1, 0), mode='edge')
        rmin_padded = minimum_filter1d(r_padded, size=window_size, mode='nearest', origin=0)
        rmax_padded = maximum_filter1d(r_padded, size=window_size, mode='nearest', origin=0)
        rmin = rmin_padded[window_size-1:]
        rmax = rmax_padded[window_size-1:]

        return (rmin < q * rmax)



