from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_LIB: Optional[ctypes.CDLL] = None
_LIB_PATH: Optional[Path] = None


def _library_filename() -> str:
    if sys.platform.startswith("win"):
        return "linear_regression_cpp.dll"
    if sys.platform == "darwin":
        return "liblinear_regression_cpp.dylib"
    return "liblinear_regression_cpp.so"


def _resolve_library_path(explicit_path: Optional[os.PathLike] = None) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Linear regression library not found at {path}")
        return path

    env_path = os.environ.get("CPP_LINEAR_REGRESSION_LIB")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if path.is_file():
            return path

    repo_root = Path(__file__).resolve().parents[1]
    lib_name = _library_filename()
    candidate_dirs = [
        repo_root / "impl" / "cpp-v2-cpu" / "build",
        repo_root / "impl" / "cpp-v2-cpu" / "build" / "Release",
        repo_root / "impl" / "cpp-v2-cpu" / "build" / "Debug",
    ]

    for candidate_dir in candidate_dirs:
        candidate = candidate_dir / lib_name
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Unable to locate the C++ linear regression library. "
        "Build it with `cmake -S impl/cpp-v2-cpu -B impl/cpp-v2-cpu/build && "
        "cmake --build impl/cpp-v2-cpu/build --config Release` "
        "or set CPP_LINEAR_REGRESSION_LIB to the compiled shared library."
    )


def _configure_prototypes(lib: ctypes.CDLL) -> None:
    lib.lr_create.argtypes = [ctypes.c_size_t, ctypes.c_double]
    lib.lr_create.restype = ctypes.c_void_p

    lib.lr_destroy.argtypes = [ctypes.c_void_p]
    lib.lr_destroy.restype = None

    lib.lr_fit.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.lr_fit.restype = ctypes.c_int

    lib.lr_predict.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.lr_predict.restype = ctypes.c_int

    lib.lr_last_error.argtypes = []
    lib.lr_last_error.restype = ctypes.c_char_p


def _load_library(
    library_path: Optional[os.PathLike] = None,
) -> tuple[ctypes.CDLL, Path]:
    global _LIB, _LIB_PATH

    if library_path is None and _LIB is not None:
        return _LIB, _LIB_PATH  # type: ignore[misc]

    path = _resolve_library_path(library_path)
    lib = ctypes.CDLL(str(path))
    _configure_prototypes(lib)

    if library_path is None:
        _LIB = lib
        _LIB_PATH = path

    return lib, path


def _last_error(lib: ctypes.CDLL) -> str:
    err = lib.lr_last_error()
    if err:
        return err.decode("utf-8")
    return "Unknown error"


def _check_status(lib: ctypes.CDLL, status: int) -> None:
    if status != 0:
        raise RuntimeError(_last_error(lib))


class LinearRegressionV21:
    """
    Thin Python wrapper around the optimized C++ LinearRegressionV21 implementation.
    """

    def __init__(
        self,
        iterations: int = 1000,
        learning_rate: float = 0.05,
        library_path: Optional[os.PathLike] = None,
    ) -> None:
        self._lib, self._lib_path = _load_library(library_path)
        self._handle = self._lib.lr_create(
            ctypes.c_size_t(iterations),
            ctypes.c_double(learning_rate),
        )
        if not self._handle:
            raise RuntimeError(_last_error(self._lib))
        self._iterations = iterations
        self._learning_rate = learning_rate

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            self._lib.lr_destroy(handle)
            self._handle = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionV21":
        X_arr = np.ascontiguousarray(X, dtype=np.float64)
        y_arr = np.ascontiguousarray(y, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y_arr.ndim != 1:
            raise ValueError("y must be a 1D array")
        n_samples, n_features = X_arr.shape
        status = self._lib.lr_fit(
            self._handle,
            X_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_size_t(n_samples),
            ctypes.c_size_t(n_features),
        )
        _check_status(self._lib, status)
        self._n_features = n_features
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_n_features"):
            raise RuntimeError("Model must be fitted before calling predict()")
        X_arr = np.ascontiguousarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array")
        n_samples, n_features = X_arr.shape
        if n_features != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features but received {n_features}"
            )
        out = np.empty(n_samples, dtype=np.float64)
        status = self._lib.lr_predict(
            self._handle,
            X_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_size_t(n_samples),
            ctypes.c_size_t(n_features),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        _check_status(self._lib, status)
        return out

    @property
    def library_path(self) -> Optional[Path]:
        return self._lib_path

    @property
    def iterations(self) -> int:
        return self._iterations

    @property
    def learning_rate(self) -> float:
        return self._learning_rate
