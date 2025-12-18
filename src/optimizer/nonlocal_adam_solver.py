import numpy as np
from typing import Callable, Tuple
from scipy.interpolate import interp1d as scipy_interp1d
from .tools.algorithm import AlgorithmIDE, DTYPE
from .tools.integration import IntegrationQuadrature
from .tools.kernels_definition import K_beta_first_order, K_beta_second_order
from .tools.condition_v_positive import ConditionVPositive
from .tools.paralelizacion import ParallelComputation

class NonlocalSolverMomentumAdam:
    """
    Continuous-time Adam-like nonlocal dynamics for a scalar parameter θ(t):

        dot theta(t) = - alpha(t) hatm(t) / ( sqrt{hatv(t) + eps(t)} )

    with hatm, hatv defined by exponential kernels K_1, K_2
           K_a(t) = (1 - beta_a)/alpha exp(-(1 - β_a) t / alpha),   a in {1,2}.

    Notes
    -----
    - Uses AlgorithmIDE for grid construction, Euler stepping, relaxation.
    - Uses IntegrationQuadrature for Gauss-Legendre quadrature.
    - alpha(t) and eps(t) are bias-correction factors (analogs to discrete Adam).
    """
    def __init__(self, 
                 dL: Callable,
                 t_span: Tuple[float, float],
                 y0: float,
                 alpha: float,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps_base: float = 1e-8,
                 verbose: bool = False,
                 quad_order: int = 50,
                 n_workers: int = -2):
        
        self.dL = dL
        self.beta1, self.beta2 = map(DTYPE, betas)
        self.eps_base = DTYPE(eps_base)
        self.verbose = verbose
        self.condition_v_positive = ConditionVPositive(alpha)
        
        y0_arr = np.asarray(y0, dtype=DTYPE)
        if y0_arr.ndim == 0 or len(y0_arr) == 1:
            # First order equation
            self.equation_order = 1
            self.y0 = y0_arr.ravel()[0]
        elif len(y0_arr) == 2:
            # Second order equation
            self.equation_order = 2
            self.y0 = y0_arr
        else:
            raise ValueError(f"y0 debe ser escalar o vector de 2 elementos, got shape {y0_arr.shape}")
        
        self.t0, self.tf = t_span
        self.alpha = DTYPE(alpha)
        
        # Setup integrator for quadrature computations
        self.integrator = IntegrationQuadrature(n=quad_order, tol=1e-12, verbose=verbose)

        if self.equation_order == 1:
            self.K1 = lambda s: K_beta_first_order(s, self.beta1, self.alpha)
            self.K2 = lambda s: K_beta_first_order(s, self.beta2, self.alpha)
            if verbose:
                print(f"First order equation --> using exponential kernels")
        else:
            self.K1 = lambda s: K_beta_second_order(s, self.beta1, self.alpha)
            self.K2 = lambda s: K_beta_second_order(s, self.beta2, self.alpha)
            if verbose:
                print(f"Second order equation --> using sinh/sin kernels")
        
        
        # Bias-correction factors (continuous-time analogs)
        self._alpha_t = lambda t: np.where(
            t <= 1e-12,
            1.,
            np.sqrt(1. - self.beta2 ** (t / self.alpha)) / (1. - self.beta1 ** (t / self.alpha))
        )
        self._eps_t = lambda t: np.where(
            t <= 1e-12,
            self.eps_base,                                   
            self.eps_base * np.sqrt(1. - self.beta2 ** (t / self.alpha))
        )

        # Lambda parameters for kernels
        self.lam1 = (1. - self.beta1) / self.alpha
        self.lam2 = (1. - self.beta2) / self.alpha
        
        rhs = self._rhs if self.equation_order == 1 else self._rhs_inertial
        # Create the IDE solver with our specific RHS and func_rhs builder
        self.solver = AlgorithmIDE(
            dL=self.dL,
            rhs=rhs,
            build_func_rhs=self._build_func_rhs,
            t_span=t_span,
            y0=y0,
            alpha=alpha,
            verbose=verbose,
            quad_order=quad_order
        )

        self.t = self.solver.t

        self.parallel = ParallelComputation(
            n_workers=n_workers,
            min_items_for_parallel=100,
            verbose=self.verbose
        )
        
    def _interp(self, y: np.ndarray):
        """Cubic interpolator over the current time grid."""
        kind = 'linear'
        return lambda t: scipy_interp1d(self.t, y, kind=kind, 
                                        fill_value='extrapolate')(t)

    def _build_func_rhs(self, y: np.ndarray) -> Tuple:
        """
        Precompute along the current iterate y(t):
          - m(t): exponential moving average of g(t)
          - v(t): exponential moving average of g(t)^2
          - sqrt(v(t)), alpha(t), eps(t): quantities needed by the RHS

        Returns
        -------
        (y, m, v_sqrt, a_t, eps_t)
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Building functional RHS (iteration {getattr(self.solver, 'iteration', 0)})...")
            print(f"{'='*60}")
            print(f"¿No finitos en y?: {~np.isfinite(y).all()} (NaN: {np.isnan(y).any()}, Inf: {np.isinf(y).any()})")

        if self.equation_order == 2:
            y_pos = y[:, 0]
        else: 
            y_pos = y

        interp = self._interp(y_pos)

        if self.verbose:
            print(f"\n--- DEBUG interpolador ---")
            print(f"y_pos shape: {y_pos.shape}, dtype: {y_pos.dtype}")
            print(f"y_pos[:5]: {y_pos[:5]}")
            print(f"y_pos finitos: {np.isfinite(y_pos).all()}")

            if not np.isfinite(y_pos).all():
                bad_idx = np.where(~np.isfinite(y_pos))[0]
                first_bad = bad_idx[0]
                print(f"  ⚠️ PRIMER NO-FINITO en idx={first_bad}, t={self.t[first_bad]:.6f}")
                print(f"    y_pos[{first_bad-2}:{first_bad+3}] = {y_pos[max(0,first_bad-2):first_bad+3]}")
            
            # Test interpolador en algunos puntos
            test_times = [self.alpha, self.alpha * 2, self.t[1], self.t[5]]
            for tt in test_times:
                if tt <= self.t[-1]:
                    y_interp = interp(tt)
                    g_val = self.dL(y_interp)
                    print(f"  t={tt:.6f}: interp(t)={y_interp:.6e}, dL(interp)={g_val:.6e}")
            print(f"--- FIN DEBUG ---\n")

        def g_fun(tau):
            # g(tau) = dL(y(tau))
            return self.dL(interp(tau)) 
        
        def _moments_single(t):
            """
            Compute (m(t), v(t)) for a single time t via GL quadrature.
            For very small t, short-circuit to (0, 0) to avoid boundary issues.
            """
            if t < 1e-12:
                return DTYPE(0.), DTYPE(0.), None
            
            f_m = lambda tau: self.K1(t - tau) * g_fun(tau)
            f_v = lambda tau: self.K2(t - tau) * g_fun(tau)**2
            
            m_k = self.lam1 * self.integrator.integrate(f_m, self.alpha, t)
            v_k = self.lam2 * self.integrator.integrate(f_v, self.alpha, t)

            diag = None
            if not np.isfinite(m_k) or not np.isfinite(v_k) or np.abs(m_k) > 1e10 or np.abs(v_k) > 1e10:
                g_at_t = g_fun(t)
                diag = {
                    't': t,
                    'm_k': m_k,
                    'v_k': v_k,
                    'g(t)': g_at_t,
                    'K1(0)': self.K1(0),
                    'K2(0)': self.K2(0),
                }

            return m_k, v_k, diag

        if self.verbose:
            print(f"Computing moments for {len(self.t)} time points...")
            import time
            start_time = time.time()

        ## Parallel computation of moments
        moments_list = self.parallel.map(_moments_single, self.t)
        moments = np.array([(r[0], r[1]) for r in moments_list])
        diagnostics = [r[2] for r in moments_list if r[2] is not None]
        ##

        if self.verbose and diagnostics:
            print(f"\n{len(diagnostics)} problemas detectados en momentos:")
            for i, d in enumerate(diagnostics[:5]):
                print(f"t={d['t']:.6f}: m={d['m_k']:.3e}, v={d['v_k']:.3e}, g(t)={d['g(t)']:.3e}")
            if len(diagnostics) > 5:
                print(f"... y {len(diagnostics)-5} más")

        if self.verbose:
            elapsed = time.time() - start_time
            print(f"Moments computed in {elapsed:.2f}s")

        m = moments[:, 0]
        v = moments[:, 1]

        r = g_fun(self.t)**2
        viol = self.condition_v_positive(beta2=self.beta2, r=r, t=self.t, v=v)

        if viol.any():
            n_viol = np.sum(viol)
            print(f"WARNING: {n_viol} violations of v>0 condition detected")
            if self.verbose:
                print(f"   First violation at t={self.t[viol][0]:.6f}")
                print(f"   Last violation at t={self.t[viol][-1]:.6f}")


        if self.verbose:
            print(f"m[0]={m[0]:.3e}, v[0]={v[0]:.3e}")
            print(f"m[1]={m[1]:.3e}, v[1]={v[1]:.3e}")
            print(f"m[-1]={m[-1]:.3e}, v[-1]={v[-1]:.3e}")

        # Save timeseries for inspection/plotting
        self._last_m = np.stack((self.t, m), axis=1)
        self._last_v = np.stack((self.t, v), axis=1)

        v_sqrt = np.sqrt(np.maximum(v, 0.))
        a_t = self._alpha_t(self.t)
        eps_t = self._eps_t(self.t)

        return (m, v_sqrt, a_t, eps_t)

    def _rhs(self, y_prev, idx: int,  m: np.ndarray, v_sqrt: np.ndarray, 
             a_t: np.ndarray, eps_t: np.ndarray) -> float:
        """
        Right-hand side for the explicit Euler step:

            .. = - alpha(t) m(t) / ( sqrt{v(t)} + eps(t) )

        Parameters
        ----------
        idx : int
            Index of `t` on the solver time grid.
        m : np.ndarray
            Samples of the first moment along the grid.
        v_sqrt : np.ndarray
            Samples of sqrt(second moment) along the grid.
        a_t : np.ndarray
            Bias-correction factor alpha(t) along the grid.
        eps_t : np.ndarray
            Scaled eps(t) along the grid.

        Returns
        -------
        float
            dy/dt - Instantaneous rate used by the Euler integrator.
        """
        # Denominator sqrt{v(t)} + eps(t) for stability
        denom = v_sqrt[idx] + eps_t[idx]

        # Combine local dynamics with normalized moment term
        return - a_t[idx] * (m[idx] / denom)

    def _rhs_inertial(self, z_prev, idx: int,
                  m: np.ndarray, v_sqrt: np.ndarray,
                  a_t: np.ndarray, eps_t: np.ndarray) -> float:

        theta, dtheta = z_prev[0], z_prev[1]

        denom = v_sqrt[idx] + eps_t[idx]
        T = m[idx] / denom  # m/(sqrt(v)+eps)

        update = dtheta + a_t[idx] * T
        result = 2.0 * update / self.alpha
        
        return result
    
    def solve(self):
        """
        Solve the nonlocal Adam ODE using the IDE solver.
        
        Returns
        -------
        t : np.ndarray
            Time grid.
        y : np.ndarray
            Solution values at each time point.
        """
        return self.solver.solve()