from typing import Callable, Tuple
import numpy as np

class AdamScratch1D:
    """
    Scalar Adam optimizer from scratch.

    Parameters
    ----------
    dL : Callable
        Gradient function dL(theta). 
    lr : float
        Base learning rate (alpha).
    beta1 : float
        Exponential decay rate for the first moment (0 < beta1 < 1).
    beta2 : float
        Exponential decay rate for the second moment (0 < beta2 < 1).
    epsilon : float
        Small constant in the denominator for numerical stability.
    epochs : int
        Number of iterations to run.

    Attributes
    ----------
    m, v : float
        First/second moment accumulators.
    iteration : int
        1-based iteration counter.
    theta_result, m_result, v_result : list[float]
        Per-epoch histories for analysis/plotting.
    """

    def __init__(self, dL: Callable, lr: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8, epochs: int = 1000):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.dL = dL
        self.epochs = epochs
        self.__reset_state__()

    # Private method to reset the optimizer state and histories
    def __reset_state__(self):
        """Reset optimizer state and histories."""
        self.m = 0.0
        self.v = 0.0
        self.iteration = 1
        self.theta_history = []
        self.m_history = []
        self.v_history = []

    @staticmethod
    def __global_error__(theta_new: float, theta_old: float) -> float:
        """Absolute difference |theta_new - theta_old|; handy as a progress metric."""
        return abs(theta_new - theta_old)

    # Public method to solve the optimization problem
    def solve(self,
    theta_initial: float,
    function_parameters: Tuple[float, ...] = ()
    ) -> Tuple[list[float], list[float], list[float], int]:
            """
            Run Adam for `epochs` steps starting from `theta_initial`.

            Parameters
            ----------
            theta_initial : list, tuple, or numpy.ndarray
                Initial values [theta1, theta2]
            function_parameters : Tuple[float, float]
                Parameters of the function to optimize.
            Notes
            -----
            - 2D implementation (always works with [theta1, theta2]).
            - Bias-corrected moments m, v are used.
            - If `weight_decay != 0`, apply decoupled weight decay before subtracting `update`.
            """
            self.__reset_state__()
            theta = float(theta_initial)

            while self.iteration <= self.epochs:
                # Log histories 
                self.theta_history.append(theta)
                self.m_history.append(self.m)
                self.v_history.append(self.v)

                theta_old = theta
                dL_value = self.dL(x=theta, function_parameters=function_parameters)

                # Adam moment updates
                self.m = self.beta1 * self.m + (1 - self.beta1) * dL_value
                self.v = self.beta2 * self.v + (1 - self.beta2) * (dL_value ** 2)

                # Bias corrections
                m_hat = self.m / (1 - self.beta1 ** self.iteration)
                v_hat = self.v / (1 - self.beta2 ** self.iteration)

                # Adam update (theta -= update)
                theta -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

                global_error = float(self.__global_error__(theta_new=theta, theta_old=theta_old))
                if self.iteration % 50 == 0:
                    print(f'Epoch: {self.iteration}, theta={theta:.6e}, Error: {global_error:.6e}')

                self.iteration += 1

            print(f'Last epoch: {self.iteration-1}, Error: {global_error:.6e}')

            return self.theta_history, self.m_history, self.v_history, self.iteration