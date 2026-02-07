# Accelerated Continuous-Time Formulation of Adam

This repository contains the code accompanying the paper:

**"From Adam to Adam-Like Lagrangians: Second-Order Nonlocal Dynamics"**  
*Authors: Carlos Heredia*

## Abstract

In this paper, we derive an accelerated continuous-time formulation of Adam by modeling it as a second-order integro-differential dynamical system. We relate this inertial nonlocal model to an existing first-order nonlocal Adam flow through an alpha-refinement limit, and we provide Lyapunov-based stability and convergence analyses. We also introduce an Adam-inspired nonlocal Lagrangian formulation, offering a variational viewpoint. Numerical simulations on Rosenbrock-type examples show agreement between the proposed dynamics and discrete Adam.

## Repository Structure

```
.
├── src/
│   ├── optimizer/
│   │   ├── adam_scratch_1d.py           # Discrete Adam implementation from scratch
│   │   ├── nonlocal_adam_solver.py      # Main nonlocal Adam continuous solver
│   │   └── tools/
│   │       ├── algorithm.py             # Core optimization algorithms
│   │       ├── condition_v_positive.py  # Positive velocity conditions
│   │       ├── euler_method.py          # Euler numerical integration scheme
│   │       ├── integration.py           # Integro-differential equation solvers
│   │       ├── kernels_definition.py    # Nonlocal kernel functions
│   │       └── paralelizacion.py        # Parallelization utilities
│   └── utils/
│       └── ploting.py                   # Visualization and plotting tools
│
├── notebooks/
│   ├── error_phi.ipynb                  # Main phi-error analysis
│   ├── 1rst_analysis.ipynb              # First-order model analysis
│   ├── 2nd_analysis.ipynb               # Second-order model analysis
│   ├── analysis.ipynb                   # Discrete analysis
│   ├── initial_velocity.ipynb           # Initial velocity sensitivity study
│   ├── max_error.ipynb                  # Maximum error computations
│   ├── plot_convergence.ipynb           # Convergence visualizations
│   ├── transition_nonconvex.ipynb       # Non-convex transition analysis
│   ├── config.py                        # Experiment configuration
│   ├── functions.py                     # Test functions (Rosenbrock)
│   ├── errors/                          # Error metrics tables (CSV)
│   ├── plots/                           # Generated figures (PNG)
│   ├── results/                         # Experimental results (CSV)
│   ├── results_1rst_order/              # First-order model results
│   └── results_2nd_order/               # Second-order model results
│
├── requirements.txt                     # Python dependencies
├── LICENSE                              # MIT License
└── README.md                            # This file
```

## Algorithm

The core numerical method for solving the nonlocal integro-differential equations is based on an iterative fixed-point scheme with adaptive under-relaxation. Below is the pseudocode:

### Iterative Modified IDESolver Method

```
Algorithm: Iterative Modified IDESolver Method
────────────────────────────────────────────────────────────────

1:  Initialize the iteration counter k ← 0
2:  Compute the initial solution y_guess using the original differential equation
3:  Compute the initial guess y_current including the integral part with y_guess
4:  Calculate the initial global error: error ← ||y_current - y_guess||

5:  WHILE error > tolerance DO
6:      Compute new solution y_new using a smoothing factor with y_current and y_guess
7:      Update y_guess solving the ODE including the integral part with y_new
8:      Calculate the current global error: new_error ← ||y_new - y_guess||
9:      
10:     IF new_error > error THEN
11:         IF maximum smoothing factor reached THEN
12:             Exit the loop without achieving the desired tolerance
13:         ELSE
14:             Update the smoothing factor to the next value
15:         END IF
16:     END IF
17:     
18:     Update y_current ← y_new
19:     Increment the iteration counter k ← k + 1
20:     
21:     IF k > k_max THEN
22:         Exit the loop
23:     END IF
24:     
25:     Update error ← new_error
26: END WHILE

27: Set the final solution y ← y_guess
28: RETURN time values and the corresponding solution y
```

**Key features:**
- **Fixed-point iteration**: Solves the nonlocal equation iteratively
- **Adaptive smoothing**: Adjusts relaxation factor when error increases
- **Global error metric**: Uses L² norm over the entire time domain
- **Safety controls**: Maximum iterations and divergence detection
