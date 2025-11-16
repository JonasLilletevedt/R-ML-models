# CoreFlux — High-Performance Machine Learning from Scratch (Rust → C++ → CUDA)

CoreFlux is a multi-backend machine learning engine built from the ground up.
The goal is to understand machine learning algorithms deeply by implementing them manually,
validate them against scikit-learn, and then optimize them for high performance.

The project includes:
- Rust implementations (correctness-first baseline)
- C++ OpenMP implementations (performance-first CPU backend)
- CUDA backend (planned for GPU acceleration)

Everything is implemented from scratch: gradient descent, KNN, memory layouts, threading, and Python integration.

------------------------------------------------------------
Project Goals
------------------------------------------------------------

1. Implement ML algorithms from first principles.
2. Achieve numerical equivalence with scikit-learn.
3. Optimize performance using:
   - Raw pointers and cache-aware memory layouts
   - OpenMP threading
   - SIMD (planned)
   - CUDA kernels (planned)
4. Provide Python bindings for all backends.

------------------------------------------------------------
Backends
------------------------------------------------------------

v1 — Rust (correctness-first)
- Safe memory model
- Clean, readable baseline implementations
- Python bindings using PyO3

v2 — C++ (performance-first)
- Raw pointers with predictable memory layout
- Parallel gradient descent using OpenMP
- Python bindings via ctypes (pybind11 planned)

v3 — CUDA (planned)
- GPU kernels for linear algebra
- Parallel gradient descent on GPU
- Benchmarks comparing CPU vs GPU vs scikit-learn

------------------------------------------------------------
Algorithms Implemented
------------------------------------------------------------

Linear Regression (Gradient Descent)
- Rust baseline
- C++ optimized version with OpenMP
- Python wrappers for both backends
- Detailed explanation in docs/LINEAR_REGRESSION.md

K-Nearest Neighbors (Regression and Classification)
- Rust implementation
- Python bindings
- Detailed explanation in docs/KNN.md

More algorithms will be ported to C++ and CUDA as the project evolves.

------------------------------------------------------------
Project Structure
------------------------------------------------------------

coreflux/
  rust/        Rust correctness baseline
  cpp/         C++ optimized backend
  cuda/        CUDA backend (planned)
  python/      Python wrappers and build scripts
  docs/        Algorithm explanations and derivations
  notebooks/   Tests, validation, and benchmarks

------------------------------------------------------------
Benchmarking
------------------------------------------------------------

Dataset:
- 200,000 training samples
- 10,000 test samples
- 40 features
- 1000 gradient descent iterations

Baseline Comparison (CPU):

Model                                   | Mean Time (5 runs)
--------------------------------------- | ------------------
scikit-learn SGDRegressor (1000 iters)  | 2.05 s
Rust Linear Regression                  | 18.17 s
C++ Parallel GD (OpenMP)                | 1.05 s

Notes:
The Rust implementation is intentionally naive and does not use BLAS, SIMD, or parallelism.
The C++ version significantly reduces the gap through:
- Thread-parallel gradient accumulation
- Cache-friendly memory access patterns
- Reduced branching
- Tighter loop structures

See notebooks/benchmarks for full results and commentary.

------------------------------------------------------------
Installation
------------------------------------------------------------

1. Clone the repository:
   ```bash
   git clone https://github.com/JonasLilletvedt/coreflux
   cd coreflux
   ```

3. Create a virtual environment (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # Linux/Mac
   .venv\Scripts\activate           # Windows
   ```

4. Install required packages:

   ```bash
   pip install maturin numpy scikit-learn
   ```

6. Build Rust backend:

   ```bash
   maturin develop --release
   ```

8. Build C++ backend:

   ```bash
   cmake -S cpp -B cpp/build
   cmake --build cpp/build --config Release
   ```

Python usage:

   ```python
   from coreflux_cpp import LinearRegression
   model = LinearRegression(iters=1000, lr=0.05)
   ```

If the shared library is in a custom path, set:
   ```python
   export CPP_LINEAR_REGRESSION_LIB=/path/to/lib
   ```

------------------------------------------------------------
Detailed Algorithm Documentation
------------------------------------------------------------

Linear Regression derivation:
  docs/LINEAR_REGRESSION.md

KNN explanation:
  docs/KNN.md

These documents include mathematical derivations, code explanations,
and performance considerations.

------------------------------------------------------------
Roadmap
------------------------------------------------------------

v1 — Rust (Complete)
- Linear Regression
- KNN
- Python bindings
- Validation against scikit-learn

v2 — C++ (In Progress)
- Parallelized Linear Regression
- SIMD optimizations
- pybind11 module
- Additional models

v3 — CUDA (Planned)
- GPU kernels for LR
- GPU matrix operations
- CPU vs GPU vs sklearn benchmarks
