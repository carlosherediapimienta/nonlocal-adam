import numpy as np

class ConditionVPositive:
    def __init__(self,  alpha: float):
        self.alpha = alpha       

    def __call__(self, beta2: float, r: np.ndarray, t: np.ndarray, v: np.ndarray) -> np.ndarray:
        out = np.zeros_like(t, dtype=bool)

        if beta2 >= 0.5:
            return out

        if len(r) != len(t):
            raise ValueError(f"r y t deben tener la misma longitud: len(r)={len(r)} vs len(t)={len(t)}")
        
        rho = np.sqrt(1.0 - 2.0 * beta2)          
        q = np.exp(-np.pi / rho)

        # Global + causal: min/max accumulated over [0..i]
        rmin = np.minimum.accumulate(r)
        rmax = np.maximum.accumulate(r)

        # Violation of the lemma criterion: rmin < q * rmax
        return (rmin < q * rmax)



