from __future__ import annotations
from typing import Callable, Tuple, Any, Union
import numpy as np
from .euler_method import EulerMethod

DTYPE = np.float64

class AlgorithmIDE:
    """
    Algorithm for Integro-Differential Equations (IDE) solver.
    
    Shared boilerplate for non-local ODE-like solvers:
      - time grid construction
      - explicit Euler time-stepping
      - fixed-point relaxation with smoothing
      - global error monitoring and tolerance/iteration controls
      - integration via Gauss-Legendre quadrature

    Subclasses MUST implement:
      * _build_func_rhs(y) -> Tuple[Any, ...]
          Precompute any structures ("stats") needed to evaluate the RHS
          at all times, given the current iterate y(t).

      * _rhs(t, y_prev, idx, *stats) -> dy/dt
          Right-hand side function to advance one Euler step.
          `idx` is the integer time-index (0-based) corresponding to `t`.

    Attributes (key ones)
    ---------------------
    dL : Callable
        Additional functional/derivative used by specific non-local models.
    t : np.ndarray
        Time grid [t0, tf) with spacing `alpha` (tf is excluded).
    y0 : DTYPE
        Initial condition (scalar).
    alpha : DTYPE
        Time step size.
    lambda_ : DTYPE
        Optional model parameter exposed to subclasses.
    smoothing : DTYPE
        Current relaxation factor s in y_relax = s*y_cur + (1-s)*y_new.
    global_tol : float
        Convergence tolerance for the global error metric.
    max_iteration : int
        Safety cap on fixed-point iterations.
    verbose : bool
        If True, prints progress every 20 iterations.
    integrator : IntegrationQuadrature
        Gauss-Legendre quadrature integrator instance.
    """
    def __init__(self,
                 dL: Callable,
                 rhs: Callable,
                 build_func_rhs: Callable,
                 t_span: Tuple[float, float],
                 y0: Union[float, np.ndarray],
                 alpha: float,
                 lambda_: float = 0.,
                 verbose: bool = False,
                 quad_order: int = 1000):

        self.dL = dL
        self.rhs_initial = rhs  
        self.build_func_rhs = build_func_rhs

        y0_arr = np.asarray(y0, dtype=DTYPE)
        if y0_arr.ndim == 0 or len(y0_arr) == 1:
            self.equation_order = 1
            self.y0 = y0_arr.ravel()[0]
            self.is_scalar = True
        elif len(y0_arr) == 2:
            self.equation_order = 2
            self.y0 = y0_arr  
            self.is_scalar = False
        else:
            raise ValueError(f"Invalid initial condition length: {len(y0_arr)}. Must be 1 or 2.")

        self.rhs = self._create_system_rhs()
        self.euler_method = EulerMethod()

        self.alpha = DTYPE(alpha)
        self.verbose = verbose

        self.t0, self.tf = t_span
        self.t = np.arange(self.t0, self.tf, self.alpha, dtype=DTYPE)

        self.smoothing = DTYPE(0.5)
        self.smooth_max = DTYPE(0.9999)
        self.increments = np.linspace(self.smoothing,
                                      self.smooth_max,
                                      1_000, dtype=DTYPE)
        self._max_inc_hit = False

        self.global_tol = 1e-5
        self.max_iteration = int(3e3)

    def _create_system_rhs(self):
        if self.equation_order == 1:
            return self.rhs_initial
        elif self.equation_order == 2:
            def system_rhs(z, idx, *func_rhs):
                dy = z[1]
                ddy = self.rhs_initial(z, idx, *func_rhs)
                return np.array([dy, ddy], dtype=DTYPE)
            return system_rhs

    # ---------------------------------------------------------------
    def _integrate(self, alpha: float, y0, t_vec: np.ndarray, 
                   rhs: Callable, func_rhs: Tuple[Any, ...]) -> np.ndarray:
        """
        Integra usando el método Euler configurado. 
        """
        return self.euler_method.solve(alpha=alpha, y0=y0, t_vec=t_vec, rhs=rhs, func_rhs=func_rhs)

    # ------------- error & smoothing -----------------------------------
    @staticmethod
    def _global_error(a: np.ndarray, b: np.ndarray):
        """Global error metric: Euclidean norm over the full time series."""
        return np.linalg.norm(a - b)

    @staticmethod
    def _mix(s: DTYPE, a: np.ndarray, b: np.ndarray):
        """Convex combination for relaxation: s*a + (1-s)*b."""
        return s * a + (1. - s) * b

    # -------------------------------------------------------------------
    def _step(self, y_current: np.ndarray) -> np.ndarray:
        """Run one full pass of the explicit Euler integrator."""
        func_rhs = self.build_func_rhs(y_current)
        y_next, _ = self._integrate(self.alpha, self.y0, self.t, self.rhs, func_rhs)
        err = self._global_error(y_current, y_next)
        return y_next, err

    # -------------------------------------------------------------------
    def solve(self, order: int = 1):
        """
        Fixed-point outer loop with under-relaxation.

        Algorithm sketch
        ----------------
        1) Build a purely local baseline solution (no integral term)
           by integrating y' = f(t, y).
        2) Given the current iterate y_cur, build stats and integrate the
           full non-local RHS once to get y_new.
        3) Relax: y_relax = mix(smoothing, y_cur, y_new).
        4) Repeat until the global error ||y_relax - y_new|| < global_tol
           or safety limits are hit. The smoothing factor can increase
           when the error rises (simple backoff strategy).
        """
        self.iteration = 0

        # ---- convergence bookkeeping ----
        converged = False
        stop_reason = None
        last = np.inf

        # Baseline solution without the non-local term
        zeros = np.zeros_like(self.t)
        func_rhs_zero = (zeros, zeros, np.ones_like(self.t), 1e-8*np.ones_like(self.t))

        y_cur, _ = self._integrate(
            alpha=self.alpha,
            y0=self.y0,
            t_vec=self.t,
            rhs=self.rhs, 
            func_rhs=func_rhs_zero
        )

        y_new, err = self._step(y_cur)
        if self.verbose:
            print(f"Iter {self.iteration} – err {err}")

        last = err

        while err > self.global_tol:
            self.iteration += 1

            # Under-relaxed update
            y_relax = self._mix(self.smoothing, y_cur, y_new)

            # Re-integrate with func_rhs from the relaxed iterate
            y_new, err = self._step(y_relax)

            # ---------- safety / control logic ----------
            if np.isnan(err) or np.isinf(err):
                stop_reason = "Divergence (NaN/Inf)"
                break

            # If error worsens, increase smoothing factor (toward 1)
            if err > last:
                if self._max_inc_hit:
                    print("Maximum smoothing achieved. Stop.")
                    break
                try:
                    nxt = self.increments[
                        np.searchsorted(self.increments,
                                        self.smoothing, side="right")]
                except IndexError:
                    nxt = self.smooth_max
                    self._max_inc_hit = True
                
                if self.verbose:
                    print(f"Error increased, adjusting smoothing: {self.smoothing:.4f} → {nxt:.4f}")
                
                self.smoothing = np.minimum(self.smooth_max, nxt)
            last = err
            y_cur = y_relax
            

            if not self.verbose and self.iteration % 20 == 0:
                print(f"Iter {self.iteration:4d} │ err={err:.6e} │ smooth={self.smoothing:.4f}")
            if self.verbose:
                print(f"Iter {self.iteration:4d} │ err={err:.6e} │ smooth={self.smoothing:.4f}")

            if self.iteration >= self.max_iteration:
                stop_reason = "Max iterations"
                break

        converged = (err <= self.global_tol)
        print(f"\n{'='*60}")
        if converged:
            print("Convergence achieved!")
        else:
            print(f"Stopped: {stop_reason}")
        print(f"Final iteration: {self.iteration}")
        print(f"Final error: {err:.6e}")
        print(f"Final smoothing: {self.smoothing:.4f}")
        print(f"{'='*60}\n")

        self.y = y_new
        self.global_error = err
        
        func_rhs = self.build_func_rhs(y_new)
        y_final, dy_final = self._integrate(
            alpha=self.alpha,
            y0=self.y0,
            t_vec=self.t,
            rhs=self.rhs,
            func_rhs=func_rhs
        )

        if self.equation_order == 2:
            # Segundo orden: agregar aceleración
            if self.verbose:
                print("Agregando aceleración para segundo orden...")
            
            acceleration = dy_final[:, 1]  # d²θ/dt²
            y_extended = np.column_stack([y_new, acceleration])
            
            if self.verbose:
                print(f"Shape: {y_extended.shape} [θ, dθ/dt, d²θ/dt²]")
            
            return self.t, y_extended
        else:
            # Primer orden: agregar velocidad
            if self.verbose:
                print("Agregando velocidad para primer orden...")
            
            # dy_final es escalar: dθ/dt
            velocity = dy_final
            y_extended = np.column_stack([y_new, velocity])
            
            if self.verbose:
                print(f"Shape: {y_extended.shape} [θ, dθ/dt]")
            
            return self.t, y_extended