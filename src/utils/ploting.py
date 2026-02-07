import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional

class AdamPlotter:
    """
    Class to visualize the evolution of theta, m and v during the optimization with Adam (1D).
    
    Parameters
    ----------
    theta_history : List[float]
        History of scalar values of theta by iteration.
    m_history : List[float]
        History of values of the first moment m (scalar).
    v_history : List[float]
        History of values of the second moment v (scalar).
    """
    
    def __init__(
        self, 
        theta_history: List[float], 
        m_history: List[float], 
        v_history: List[float],
        theta_dot_history: Optional[List[float]] = None,
        theta_ddot_history: Optional[List[float]] = None
    ):
        self.theta_history = np.array(theta_history)
        self.m_history = np.array(m_history)
        self.v_history = np.array(v_history)
        
        # No scaling
        self.theta_dot_history = np.array(theta_dot_history) if theta_dot_history is not None else None
        self.theta_ddot_history = np.array(theta_ddot_history) if theta_ddot_history is not None else None
        
        self.iterations = np.arange(len(theta_history))
        self.is_second_order = theta_dot_history is not None
    
    def __plot_theta__(self, figsize=(10, 6)):
        """Plot the evolution of theta."""
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(self.iterations, self.theta_history, linewidth=2, color='C0')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='θ = 0')
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Value of θ', fontsize=12)
        ax.set_title('Evolution of θ during the optimization', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def __plot_moments__(self, figsize=(12, 5)):
        """Plot the evolution of the moments m and v."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # First moment m
        ax1.plot(self.iterations, self.m_history, linewidth=2, color='C1')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Iteration', fontsize=11)
        ax1.set_ylabel('Value of m', fontsize=11)
        ax1.set_title('Evolution of the first moment m', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Second moment v
        ax2.plot(self.iterations, self.v_history, linewidth=2, color='C2')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Iteration', fontsize=11)
        ax2.set_ylabel('Value of v', fontsize=11)
        ax2.set_title('Evolution of the second moment v', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, (ax1, ax2)

    def __plot_phase_diagram__(self, figsize=(8, 8)):
        """
        Phase diagram: θ vs dθ/dt
        Shows the trajectory in the state space with temporal color code.
        """
        if not self.is_second_order:
            raise ValueError("Phase diagram requires second-order dynamics (velocity)")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot with temporal color code
        scatter = ax.scatter(
            self.theta_history, 
            self.theta_dot_history, 
            c=self.iterations, 
            cmap='viridis',
            s=20,
            alpha=0.6
        )
        
        # Connect points with lines
        ax.plot(self.theta_history, self.theta_dot_history, 
                'k-', alpha=0.2, linewidth=0.5)
        
        # Initial and final point
        ax.plot(self.theta_history[0], self.theta_dot_history[0], 
                'go', markersize=10, label='Start', zorder=5)
        ax.plot(self.theta_history[-1], self.theta_dot_history[-1], 
                'r*', markersize=15, label='End', zorder=5)
        
        # References
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        
        ax.set_xlabel('Position θ', fontsize=12)
        ax.set_ylabel('Velocity dθ/dt', fontsize=12)
        ax.set_title('Phase Diagram (State Space Trajectory)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Iteration', fontsize=11)
        
        plt.tight_layout()
        return fig, ax
    
    def __plot_state_variables__(self, figsize=(12, 9)):
        """
        Panel with θ, dθ/dt, and d²θ/dt² in separate subplots.
        """
        if not self.is_second_order:
            raise ValueError("State variables plot requires second-order dynamics")
        
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Theta θ
        axes[0].plot(self.iterations, self.theta_history, linewidth=2, color='C0')
        axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_ylabel('Position θ', fontsize=11)
        axes[0].set_title('Evolution of State Variables', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Velocity dθ/dt (dot theta)
        axes[1].plot(self.iterations, self.theta_dot_history, linewidth=2, color='C1')
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1].set_ylabel('Velocity dθ/dt', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        # Acceleration d²θ/dt² (ddot theta)
        if self.theta_ddot_history is not None:
            axes[2].plot(self.iterations, self.theta_ddot_history, linewidth=2, color='C3')
            axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[2].set_ylabel('Acceleration d²θ/dt²', fontsize=11)
        axes[2].set_xlabel('Iteration', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, axes
    
    def __plot_kinetic_energy__(self, figsize=(10, 6)):
        """
        Kinetic energy plot: (1/2)(dθ/dt)²
        Useful to see how the system "cools down".
        """
        if not self.is_second_order:
            raise ValueError("Kinetic energy requires velocity")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        kinetic_energy = 0.5 * self.theta_dot_history**2
        
        ax.plot(self.iterations, kinetic_energy, linewidth=2, color='C4')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Kinetic Energy (½(dθ/dt)²)', fontsize=12)
        ax.set_title('Evolution of Kinetic Energy', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        return fig, ax
    
    def __plot_dynamics_comparison__(self, figsize=(14, 10)):
        """
        Complete comparison: state, moments and dynamics in a single panel.
        Panel 2x3 with all relevant information.
        """
        if not self.is_second_order:
            raise ValueError("Dynamics comparison requires second-order system")
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Row 1: Complete state
        # Theta θ
        axes[0, 0].plot(self.iterations, self.theta_history, linewidth=2, color='C0')
        axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        axes[0, 0].set_ylabel('θ', fontsize=11)
        axes[0, 0].set_title('Position', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # dθ/dt (dot theta)
        axes[0, 1].plot(self.iterations, self.theta_dot_history, linewidth=2, color='C1')
        axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        axes[0, 1].set_ylabel('dθ/dt', fontsize=11)
        axes[0, 1].set_title('Velocity', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # d²θ/dt² (ddot theta)
        if self.theta_ddot_history is not None:
            axes[0, 2].plot(self.iterations, self.theta_ddot_history, linewidth=2, color='C3')
            axes[0, 2].axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        axes[0, 2].set_ylabel('d²θ/dt²', fontsize=11)
        axes[0, 2].set_title('Acceleration', fontsize=12, fontweight='bold')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Row 2: Analysis
        # Phase diagram (mini)
        axes[1, 0].scatter(self.theta_history, self.theta_dot_history, 
                          c=self.iterations, cmap='viridis', s=10, alpha=0.5)
        axes[1, 0].plot(self.theta_history[0], self.theta_dot_history[0], 
                       'go', markersize=8, label='Start')
        axes[1, 0].plot(self.theta_history[-1], self.theta_dot_history[-1], 
                       'r*', markersize=12, label='End')
        axes[1, 0].set_xlabel('θ', fontsize=11)
        axes[1, 0].set_ylabel('dθ/dt', fontsize=11)
        axes[1, 0].set_title('Phase Diagram', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(fontsize=9)
        
        # Kinetic energy
        kinetic_energy = 0.5 * self.theta_dot_history**2
        axes[1, 1].plot(self.iterations, kinetic_energy, linewidth=2, color='C4')
        axes[1, 1].set_xlabel('Iteration', fontsize=11)
        axes[1, 1].set_ylabel('½(dθ/dt)²', fontsize=11)
        axes[1, 1].set_title('Kinetic Energy', fontsize=12, fontweight='bold')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Moments (m and v)
        ax_twin = axes[1, 2].twinx()
        axes[1, 2].plot(self.iterations, self.m_history, linewidth=2, 
                       color='C1', label='m (1st moment)', alpha=0.7)
        ax_twin.plot(self.iterations, self.v_history, linewidth=2, 
                    color='C2', label='v (2nd moment)', alpha=0.7)
        axes[1, 2].set_xlabel('Iteration', fontsize=11)
        axes[1, 2].set_ylabel('m', fontsize=11, color='C1')
        ax_twin.set_ylabel('v', fontsize=11, color='C2')
        axes[1, 2].set_title('Adam Moments', fontsize=12, fontweight='bold')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].tick_params(axis='y', labelcolor='C1')
        ax_twin.tick_params(axis='y', labelcolor='C2')
        
        plt.suptitle('Complete Dynamics Analysis', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        return fig, axes
    
    
    def __plot_all__(self, figsize=None):
        """
        Plot all variables in a single figure.
        Layout adapts based on whether it's first or second order.
        """
        if self.is_second_order:
            # Second order: Panel with more vertical space
            if figsize is None:
                figsize = (18, 14)  # ← Increased height
            
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(3, 3, hspace=0.50, wspace=0.35)  # Increased hspace
            
            # ============ ROW 1: State variables ============
            # θ (Position)
            ax00 = fig.add_subplot(gs[0, 0])
            ax00.plot(self.iterations, self.theta_history, linewidth=2, color='C0')
            ax00.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax00.set_xlabel('Iteration', fontsize=11)
            ax00.set_ylabel('θ', fontsize=11)
            ax00.set_title('Position θ', fontsize=12, fontweight='bold', pad=10)  # Added pad
            ax00.grid(True, alpha=0.3)
            
            # dθ/dt (Velocity)
            ax01 = fig.add_subplot(gs[0, 1])
            ax01.plot(self.iterations, self.theta_dot_history, linewidth=2, color='C1')
            ax01.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax01.set_xlabel('Iteration', fontsize=11)
            ax01.set_ylabel('dθ/dt', fontsize=11)
            ax01.set_title('Velocity', fontsize=12, fontweight='bold', pad=10)
            ax01.grid(True, alpha=0.3)
            
            # d²θ/dt² (Acceleration)
            ax02 = fig.add_subplot(gs[0, 2])
            if self.theta_ddot_history is not None:
                ax02.plot(self.iterations, self.theta_ddot_history, linewidth=2, color='C3')
                ax02.set_yscale('symlog', linthresh=1e-1)
                ax02.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax02.set_xlabel('Iteration', fontsize=11)
            ax02.set_ylabel('d²θ/dt²', fontsize=11)
            ax02.set_title('Acceleration', fontsize=12, fontweight='bold', pad=10)
            ax02.grid(True, alpha=0.3)
            
            # ============ ROW 2: Adam moments ============
            # m (First moment)
            ax10 = fig.add_subplot(gs[1, 0])
            ax10.plot(self.iterations, self.m_history, linewidth=2, color='C1')
            ax10.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax10.set_xlabel('Iteration', fontsize=11)
            ax10.set_ylabel('m', fontsize=11)
            ax10.set_title('First moment m', fontsize=12, fontweight='bold', pad=10)
            ax10.grid(True, alpha=0.3)
            
            # v (Second moment)
            ax11 = fig.add_subplot(gs[1, 1])
            ax11.plot(self.iterations, self.v_history, linewidth=2, color='C2')
            ax11.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax11.set_xlabel('Iteration', fontsize=11)
            ax11.set_ylabel('v', fontsize=11)
            ax11.set_title('Second moment v', fontsize=12, fontweight='bold', pad=10)
            ax11.grid(True, alpha=0.3)
            
            # Effective scale
            ax12 = fig.add_subplot(gs[1, 2])
            eps = 1e-8
            effective_scale = np.abs(self.m_history) / (np.sqrt(self.v_history) + eps)
            ax12.plot(self.iterations, effective_scale, linewidth=2, color='C4')
            ax12.set_xlabel('Iteration', fontsize=11)
            ax12.set_ylabel('|m| / (√v + ε)', fontsize=11)
            ax12.set_title('Effective update scale', fontsize=12, fontweight='bold', pad=10)
            ax12.grid(True, alpha=0.3)
            ax12.set_yscale('log')
            
            # ============ ROW 3: Dynamic analysis ============
            # Phase diagram (occupies 2 columns)
            ax20 = fig.add_subplot(gs[2, :2])
            scatter = ax20.scatter(self.theta_history, self.theta_dot_history, 
                                c=self.iterations, cmap='viridis', s=20, alpha=0.6)
            ax20.plot(self.theta_history, self.theta_dot_history, 
                    'k-', alpha=0.15, linewidth=0.5)
            ax20.plot(self.theta_history[0], self.theta_dot_history[0], 
                    'go', markersize=10, label='Start', zorder=5)
            ax20.plot(self.theta_history[-1], self.theta_dot_history[-1], 
                    'r*', markersize=15, label='End', zorder=5)
            ax20.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
            ax20.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
            ax20.set_xlabel('θ', fontsize=11)
            ax20.set_ylabel('dθ/dt', fontsize=11)
            ax20.set_title('Phase Diagram', fontsize=12, fontweight='bold', pad=10)
            ax20.legend(fontsize=10, loc='best')
            ax20.grid(True, alpha=0.3)
            
            # Colorbar for phase diagram
            cbar = plt.colorbar(scatter, ax=ax20)
            cbar.set_label('Iteration', fontsize=10)
            
            # Kinetic energy (occupies 1 column)
            ax21 = fig.add_subplot(gs[2, 2])
            kinetic_energy = 0.5 * self.theta_dot_history**2
            ax21.plot(self.iterations, kinetic_energy, linewidth=2, color='C5')
            ax21.set_xlabel('Iteration', fontsize=11)
            ax21.set_ylabel('½(dθ/dt)²', fontsize=11)
            ax21.set_title('Kinetic Energy', fontsize=12, fontweight='bold', pad=10)
            ax21.set_yscale('log')
            ax21.grid(True, alpha=0.3)
            
            plt.suptitle('Complete Optimization Analysis', 
                        fontsize=16, fontweight='bold', y=0.996)  # Adjusted y
            
            return fig, fig.axes
            
        else:
            # First order: Panel 2x2 original
            if figsize is None:
                figsize = (12, 8)
            
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            
            # Theta
            axes[0, 0].plot(self.iterations, self.theta_history, linewidth=2, color='C0')
            axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            axes[0, 0].set_xlabel('Iteration', fontsize=11)
            axes[0, 0].set_ylabel('Value of θ', fontsize=11)
            axes[0, 0].set_title('Evolution of θ', fontsize=12, fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
            
            # First moment m
            axes[0, 1].plot(self.iterations, self.m_history, linewidth=2, color='C1')
            axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            axes[0, 1].set_xlabel('Iteration', fontsize=11)
            axes[0, 1].set_ylabel('Value of m', fontsize=11)
            axes[0, 1].set_title('Evolution of the first moment m', fontsize=12, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Second moment v
            axes[1, 0].plot(self.iterations, self.v_history, linewidth=2, color='C2')
            axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            axes[1, 0].set_xlabel('Iteration', fontsize=11)
            axes[1, 0].set_ylabel('Value of v', fontsize=11)
            axes[1, 0].set_title('Evolution of the second moment v', fontsize=12, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Effective scale
            eps = 1e-8
            effective_scale = np.abs(self.m_history) / (np.sqrt(self.v_history) + eps)
            axes[1, 1].plot(self.iterations, effective_scale, linewidth=2, color='C3')
            axes[1, 1].set_xlabel('Iteration', fontsize=11)
            axes[1, 1].set_ylabel('m / (sqrt(v) + epsilon)', fontsize=11)
            axes[1, 1].set_title('Effective scale of the update', fontsize=12, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_yscale('log')
            
            plt.tight_layout()
            return fig, axes
    
    def plot(self, which='all', figsize=None):
        """
        Main method to plot.
        
        Parameters
        ----------
        which : str
            What to plot:
            - 'theta': Only position
            - 'moments': Only Adam moments
            - 'all': Original 2x2 panel (first order)
            - 'phase': Phase diagram θ vs dθ/dt (second order)
            - 'state': State variables θ, dθ/dt, d²θ/dt² (second order)
            - 'kinetic': Kinetic energy (second order)
            - 'dynamics': Complete 2x3 panel (second order)
        
        Returns
        -------
        fig, axes
        """
        if figsize is None:
            figsize_map = {
                'theta': (10, 6),
                'moments': (12, 5),
                'all': (12, 8),
                'phase': (8, 8),
                'state': (12, 9),
                'kinetic': (10, 6),
                'dynamics': (14, 10)
            }
            figsize = figsize_map.get(which, (10, 6))
        
        if which == 'theta':
            return self.__plot_theta__(figsize)
        elif which == 'moments':
            return self.__plot_moments__(figsize)
        elif which == 'all':
            return self.__plot_all__(figsize)
        elif which == 'phase':
            return self.__plot_phase_diagram__(figsize)
        elif which == 'state':
            return self.__plot_state_variables__(figsize)
        elif which == 'kinetic':
            return self.__plot_kinetic_energy__(figsize)
        elif which == 'dynamics':
            return self.__plot_dynamics_comparison__(figsize)
        else:
            valid_options = "'theta', 'moments', 'all', 'phase', 'state', 'kinetic', 'dynamics'"
            raise ValueError(f"Option '{which}' is not valid. Use {valid_options}")
    
    def __repr__(self):
        return f"AdamPlotter(iterations={len(self.iterations)})"