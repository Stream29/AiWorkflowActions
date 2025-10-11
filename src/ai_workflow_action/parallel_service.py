"""
Global Parallel Service.
Provides shared ThreadPoolExecutor for concurrent task execution.
"""

from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, Callable, TypeVar
from .config_loader import ConfigLoader

T = TypeVar('T')


class ParallelService:
    """Global parallel execution service using thread pool"""

    _executor: Optional[ThreadPoolExecutor] = None

    @classmethod
    def get_executor(cls) -> ThreadPoolExecutor:
        """
        Get global ThreadPoolExecutor instance.

        Returns:
            ThreadPoolExecutor configured from global config

        Notes:
            - Core threads: 0 (threads created on demand)
            - Max workers: from config.evaluation.parallel.max_workers
            - Shared across all phases for simplicity
        """
        if cls._executor is None:
            config = ConfigLoader.get_config()
            max_workers = config.evaluation.parallel.max_workers
            cls._executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="eval_worker"
            )
        return cls._executor

    @classmethod
    def submit(cls, fn: Callable[..., T], *args, **kwargs) -> Future[T]:
        """
        Submit a task to the thread pool.

        Args:
            fn: Callable to execute
            *args: Positional arguments for fn
            **kwargs: Keyword arguments for fn

        Returns:
            Future object representing the task
        """
        executor = cls.get_executor()
        return executor.submit(fn, *args, **kwargs)

    @classmethod
    def shutdown(cls) -> None:
        """Shutdown the executor (for cleanup/testing)"""
        if cls._executor is not None:
            cls._executor.shutdown(wait=True)
            cls._executor = None
