import matplotlib.pyplot as plt
import numpy as np
from typing import List

class AdamPlotter:
    """
    Clase para visualizar la evolución de theta, m y v durante la optimización con Adam (1D).
    
    Parameters
    ----------
    theta_history : List[float]
        Historial de valores escalares de theta por iteración.
    m_history : List[float]
        Historial de valores del primer momento m (escalar).
    v_history : List[float]
        Historial de valores del segundo momento v (escalar).
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
        """Plotea la evolución de theta."""
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(self.iterations, self.theta_history, linewidth=2, color='C0')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='θ = 0')
        
        ax.set_xlabel('Iteración', fontsize=12)
        ax.set_ylabel('Valor de θ', fontsize=12)
        ax.set_title('Evolución de θ durante la optimización', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def __plot_moments__(self, figsize=(12, 5)):
        """Plotea la evolución de los momentos m y v."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Primer momento m
        ax1.plot(self.iterations, self.m_history, linewidth=2, color='C1')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Iteración', fontsize=11)
        ax1.set_ylabel('Valor de m', fontsize=11)
        ax1.set_title('Evolución del primer momento m', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Segundo momento v
        ax2.plot(self.iterations, self.v_history, linewidth=2, color='C2')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Iteración', fontsize=11)
        ax2.set_ylabel('Valor de v', fontsize=11)
        ax2.set_title('Evolución del segundo momento v', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    def __plot_all__(self, figsize=(12, 8)):
        """Plotea todas las variables en una sola figura."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Theta
        axes[0, 0].plot(self.iterations, self.theta_history, linewidth=2, color='C0')
        axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('Iteración', fontsize=11)
        axes[0, 0].set_ylabel('Valor de θ', fontsize=11)
        axes[0, 0].set_title('Evolución de θ', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Primer momento m
        axes[0, 1].plot(self.iterations, self.m_history, linewidth=2, color='C1')
        axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Iteración', fontsize=11)
        axes[0, 1].set_ylabel('Valor de m', fontsize=11)
        axes[0, 1].set_title('Evolución del primer momento m', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Segundo momento v
        axes[1, 0].plot(self.iterations, self.v_history, linewidth=2, color='C2')
        axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Iteración', fontsize=11)
        axes[1, 0].set_ylabel('Valor de v', fontsize=11)
        axes[1, 0].set_title('Evolución del segundo momento v', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Escala efectiva del update: |m| / √(v + ε)
        eps = 1e-8
        effective_scale = np.abs(self.m_history) / (np.sqrt(self.v_history) + eps)
        axes[1, 1].plot(self.iterations, effective_scale, linewidth=2, color='C3')
        axes[1, 1].set_xlabel('Iteración', fontsize=11)
        axes[1, 1].set_ylabel('|m| / √(v + ε)', fontsize=11)
        axes[1, 1].set_title('Escala efectiva del update', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        return fig, axes
    
    def plot(self, which='all', figsize=None):
        """
        Método principal para plotear.
        
        Parameters
        ----------
        which : str
            Qué plotear: 'theta', 'moments', o 'all' (default: 'all')
        figsize : tuple, optional
            Tamaño de la figura. Si None, usa tamaños por defecto.
        
        Returns
        -------
        fig, axes
            Figura y ejes de matplotlib.
        
        Examples
        --------
        >>> plotter = AdamPlotter(theta_hist, m_hist, v_hist)
        >>> plotter.plot('theta')  # Solo theta
        >>> plotter.plot('moments')  # Solo momentos m y v
        >>> plotter.plot('all')  # Todo en una figura 2x2
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
            raise ValueError(f"Opción '{which}' no válida. Usa 'theta', 'moments', o 'all'")
    
    def __repr__(self):
        return f"AdamPlotter(iterations={len(self.iterations)})"