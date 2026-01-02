import matplotlib.pyplot as plt
import numpy as np
from typing import List

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
        v_history: List[float]
    ):
        self.theta_history = np.array(theta_history)
        self.m_history = np.array(m_history)
        self.v_history = np.array(v_history)
        self.iterations = np.arange(len(theta_history))
    
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
    
    def __plot_all__(self, figsize=(12, 8)):
        """Plot all variables in a single figure."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Theta
        axes[0, 0].plot(self.iterations, self.theta_history, linewidth=2, color='C0')
        axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('Iteration', fontsize=11)
        axes[0, 0].set_ylabel('Value of θ', fontsize=11)
        axes[0, 0].set_title('Evolution of θ', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Primer momento m
        axes[0, 1].plot(self.iterations, self.m_history, linewidth=2, color='C1')
        axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Iteration', fontsize=11)
        axes[0, 1].set_ylabel('Value of m', fontsize=11)
        axes[0, 1].set_title('Evolution of the first moment m', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Segundo momento v
        axes[1, 0].plot(self.iterations, self.v_history, linewidth=2, color='C2')
        axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Iteration', fontsize=11)
        axes[1, 0].set_ylabel('Value of v', fontsize=11)
        axes[1, 0].set_title('Evolution of the second moment v', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Effective scale of the update: |m| / √(v + ε)
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
            What to plot: 'theta', 'moments', or 'all' (default: 'all')
        figsize : tuple, optional
            Size of the figure. If None, use default sizes.
        
        Returns
        -------
        fig, axes
            Figure and axes of matplotlib.
        
        Examples
        --------
        >>> plotter = AdamPlotter(theta_hist, m_hist, v_hist)
        >>> plotter.plot('theta')  # Only theta
        >>> plotter.plot('moments')  # Only moments m and v
        >>> plotter.plot('all')  # All in a single figure 2x2
        """
        if figsize is None:
            figsize_map = {
                'theta': (10, 6),
                'moments': (12, 5),
                'all': (12, 8)
            }
            figsize = figsize_map.get(which, (10, 6))
        
        if which == 'theta':
            return self.__plot_theta__(figsize)
        elif which == 'moments':
            return self.__plot_moments__(figsize)
        elif which == 'all':
            return self.__plot_all__(figsize)
        else:
            raise ValueError(f"Option '{which}' is not valid. Use 'theta', 'moments', or 'all'")
    
    def __repr__(self):
        return f"AdamPlotter(iterations={len(self.iterations)})"