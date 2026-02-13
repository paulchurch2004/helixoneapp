"""
Lazy loading utilities to improve startup time
"""

import importlib
import sys
from collections.abc import Callable
from typing import Any


class LazyModule:
    """
    Lazy module loader that delays import until first use
    """

    def __init__(self, module_name: str):
        self._module_name = module_name
        self._module: Any | None = None

    def __getattr__(self, name: str) -> Any:
        if self._module is None:
            print(f"[LazyLoader] Loading {self._module_name}...")
            self._module = importlib.import_module(self._module_name)
        return getattr(self._module, name)

    def __call__(self, *args, **kwargs):
        if self._module is None:
            self._module = importlib.import_module(self._module_name)
        return self._module(*args, **kwargs)


class LazyObject:
    """
    Lazy object loader that delays instantiation until first use
    """

    def __init__(self, factory: Callable, *args, **kwargs):
        self._factory = factory
        self._args = args
        self._kwargs = kwargs
        self._instance: Any | None = None

    def _ensure_loaded(self):
        if self._instance is None:
            print(f"[LazyLoader] Creating {self._factory.__name__}...")
            self._instance = self._factory(*self._args, **self._kwargs)

    def __getattr__(self, name: str) -> Any:
        self._ensure_loaded()
        return getattr(self._instance, name)

    def __call__(self, *args, **kwargs):
        self._ensure_loaded()
        return self._instance(*args, **kwargs)


# Lazy imports for heavy modules
lazy_numpy = LazyModule("numpy")
lazy_pandas = LazyModule("pandas")
lazy_yfinance = LazyModule("yfinance")
lazy_ta = LazyModule("ta")
lazy_opencv = LazyModule("cv2")


def preload_modules(module_names: list[str], callback: Callable[[str], None] | None = None):
    """
    Preload modules in the background

    Args:
        module_names: List of module names to preload
        callback: Optional callback to call after each module is loaded
    """
    import threading

    def _load_modules():
        for module_name in module_names:
            try:
                if module_name not in sys.modules:
                    print(f"[LazyLoader] Preloading {module_name}...")
                    importlib.import_module(module_name)
                    if callback:
                        callback(module_name)
            except ImportError as e:
                print(f"[LazyLoader] Failed to preload {module_name}: {e}")

    thread = threading.Thread(target=_load_modules, daemon=True)
    thread.start()


def get_startup_modules() -> list[str]:
    """
    Get list of modules to preload at startup
    """
    return [
        "customtkinter",
        "PIL",
        "requests",
        # Heavy modules - load later
        # "numpy",
        # "pandas",
        # "yfinance",
    ]


def get_background_modules() -> list[str]:
    """
    Get list of modules to load in background after UI is shown
    """
    return [
        "numpy",
        "pandas",
        "yfinance",
        "ta",
    ]


# Module cache for singleton-like behavior
_module_cache: dict[str, Any] = {}


def get_cached_module(module_name: str) -> Any:
    """
    Get a cached module, loading it if necessary

    Args:
        module_name: Name of the module to import

    Returns:
        The imported module
    """
    if module_name not in _module_cache:
        _module_cache[module_name] = importlib.import_module(module_name)
    return _module_cache[module_name]


def clear_module_cache():
    """Clear the module cache"""
    _module_cache.clear()


# Example usage
if __name__ == "__main__":
    import time

    print("Testing lazy loading...")

    # Test lazy module
    start = time.time()
    lazy_np = LazyModule("numpy")
    print(f"LazyModule created in {time.time() - start:.3f}s")

    # First use triggers actual import
    start = time.time()
    arr = lazy_np.array([1, 2, 3])
    print(f"First use (actual import) took {time.time() - start:.3f}s")

    # Second use is fast
    start = time.time()
    arr2 = lazy_np.array([4, 5, 6])
    print(f"Second use took {time.time() - start:.3f}s")

    print(f"Array: {arr}")
    print(f"Array2: {arr2}")
