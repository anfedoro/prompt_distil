"""
Performance timing utilities for debugging.

This module provides decorators and utilities for measuring execution time
of key functions when PD_DEBUG environment variable is set.
"""

import functools
import os
import time
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)

# Check if debug mode is enabled
DEBUG_ENABLED = os.getenv("PD_DEBUG") == "1"


def timer(func: F) -> F:
    """
    Decorator that measures and logs execution time when PD_DEBUG=1.

    Args:
        func: Function to measure

    Returns:
        Wrapped function that logs timing if debug is enabled
    """
    if not DEBUG_ENABLED:
        # If debug is disabled, return function unchanged for minimal overhead
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000
        print(f"[PD_DEBUG] {func.__name__}: {elapsed_ms:.2f}ms")

        return result

    return wrapper
