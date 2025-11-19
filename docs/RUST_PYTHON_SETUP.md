## Setup for Rust to Python, with fluid Scikit-learn integration

### Data Flow

1.  `Python NumPy Array` --(zero-copy view)--> `Rust PyReadonlyArray`
2.  `Rust PyReadonlyArray` --(one-time copy for cache optimization/ownership)--> `Rust ndarray::Array` (owned data)
3.  `Rust ndarray::Array` (owned data) --(zero-copy view, ownership transfer)--> `Python NumPy Array`

### Implementation Steps

1.  **Rust Setup:**
    *   Add `pyo3`, `pyo3-numpy`, and `ndarray` dependencies to `Cargo.toml`.
2.  **Rust Model Logic (`src/lib.rs`):**
    *   Define `#[pyclass]` methods (`fit`, `predict`).
    *   Accept input `X` as `PyReadonlyArray2` and `y` as `PyReadonlyArray1`.
    *   Inside `fit` (or data loading), convert to owned `ndarray::Array` using `.to_owned_array()`.
    *   Inside `predict`, convert your result `ndarray::Array` to a `Py<PyArray1>` using `.into_pyarray(py)` for zero-copy return.
3.  **Build Python Module:**
    *   Use `maturin develop` (for development) or `maturin build --release` (for distribution) to compile and install.
4.  **Python Wrapper:**
    *   Create a Scikit-learn-compatible Python class (inheriting from `BaseEstimator` and `ClassifierMixin`/`RegressorMixin`).
    *   This class will instantiate and call our Rust model's methods.
    *   **Usefull:** (*Refer to the documentation to modify the template for your own scikit-learn contribution: https://contrib.scikit-learn.org/project-template*, from https://github.com/scikit-learn-contrib/project-template/)
      
  
