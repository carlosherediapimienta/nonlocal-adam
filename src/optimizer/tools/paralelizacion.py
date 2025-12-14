from typing import Callable, Iterable, List, Any, Optional
import multiprocessing as mp

try:
    from joblib import Parallel, delayed
except ImportError:
    raise ImportError(
        "joblib is required for parallelization. Install with: pip install joblib"
    )


class ParallelComputation:
    """
    Manages the parallel execution of CPU-intensive functions using joblib.
    """
    
    def __init__(
        self,
        n_workers: int = -2,
        min_items_for_parallel: int = 100,
        verbose: bool = False,
        backend: str = 'loky'
    ):
        """
        Parameters
        ----------
        n_workers : int
            Number of parallel processes.
            - If > 0: uses that exact number of workers
            - If == -1: uses all available CPUs
            - If < -1: uses (cpu_count + 1 + n_workers)
                      example: -2 = leaves 1 CPU free, -3 = leaves 2 CPUs free
        min_items_for_parallel : int
            Minimum number of items to activate parallelization.
            If there are less items, uses sequential execution (more efficient).
        verbose : bool
            If True, prints information about the process.
        backend : str
            Backend of joblib:
            - 'loky' (default): Robust, supports closures
            - 'multiprocessing': Standard Python multiprocessing
            - 'threading': For I/O-bound (not recommended for CPU-bound)
        """
        self.min_items_for_parallel = min_items_for_parallel
        self.verbose = verbose
        self.backend = backend
        
        # Calculate number of workers
        cpu_count = mp.cpu_count()
        if n_workers == -1:
            self.n_workers = cpu_count
        elif n_workers < -1:
            self.n_workers = max(1, cpu_count + 1 + n_workers)
        else:
            self.n_workers = max(1, n_workers)
        
        if self.verbose:
            print(f"ParallelComputation: joblib backend='{self.backend}', "
                  f"workers={self.n_workers}/{cpu_count}")
    
    def map(
        self,
        func: Callable,
        items: Iterable,
        batch_size: Optional[str] = 'auto'
    ) -> List[Any]:
        """
        Applies func to each item in items, in parallel if beneficial.
        
        Parameters
        ----------
        func : Callable
            Function to apply to each item.
            Can be closure, nested function, lambda, etc.
        items : Iterable
            Collection of items to process.
        batch_size : str or int, optional
            Batch size for joblib:
            - 'auto' (default): joblib decides automatically
            - int: specific number of items per batch
        
        Returns
        -------
        List[Any]
            List of results, in the same order as items.
        
        Notes
        -----
        - If len(items) < min_items_for_parallel, uses sequential execution
        - If n_workers == 1, uses sequential execution
        - Supports any type of function (closures, nested, etc.)
        """
        items_list = list(items)
        n_items = len(items_list)
        
        # Decide if it's worth parallelizing
        use_parallel = (
            n_items >= self.min_items_for_parallel and
            self.n_workers > 1
        )
        
        if not use_parallel:
            if self.verbose:
                if n_items < self.min_items_for_parallel:
                    reason = f"{n_items} items < {self.min_items_for_parallel} threshold"
                else:
                    reason = f"only {self.n_workers} worker(s)"
                print(f"  Sequential execution ({reason})")
            return [func(item) for item in items_list]
        
        # Execute in parallel with joblib
        if self.verbose:
            print(f"  Parallel execution: {n_items} items → "
                  f"{self.n_workers} workers (joblib/{self.backend})")
        
        results = Parallel(
            n_jobs=self.n_workers,
            backend=self.backend,
            batch_size=batch_size,
            verbose=0  # joblib internal verbose (0=no output)
        )(delayed(func)(item) for item in items_list)
        
        return results
    
    def __repr__(self):
        return (f"ParallelComputation(backend='joblib/{self.backend}', "
                f"workers={self.n_workers}, "
                f"min_items={self.min_items_for_parallel})")