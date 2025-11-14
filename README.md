# Rmodels: Machine Learning Algorithms Built From Scratch

Do you really understand an algorithm if you can’t build it yourself?
Probably not — and this project is my way of learning machine learning from the ground up,
by implementing core algorithms myself and validating them against scikit-learn.

This project is part of a long-term goal:
**Rust → C++ → CUDA → GPU-accelerated ML from scratch.**

---

# Project Goals

1. Understand ML algorithms deeply by implementing them entirely from scratch.
2. Build a correct and reproducible baseline that numerically matches scikit-learn.
3. Transition to high-performance computing by re-implementing models in:
   - **C++ (v2 – optimized CPU implementation)**
   - **CUDA (v3 – GPU execution)**

Rust is the correctness-first implementation.  
C++/CUDA will be the performance-first implementations.

---

# Why Rust (for v1)?

Rust was chosen because:

- Memory safety without garbage collection
- Performance close to C/C++
- Easy Python integration via PyO3

This makes Rust ideal for a clean, correct, baseline implementation of ML algorithms.

---

# Features Implemented

- **Linear Regression (Gradient Descent)**
- **K-Nearest Neighbors (Regression & Classification)**
- **Python bindings** with PyO3
- **Benchmark & validation notebooks** comparing Rust ↔ sklearn

---

# Installation & Running Locally

Below is the full workflow for cloning, building, and running the project.

## 1. Clone the repository

```bash
git clone https://github.com/JonasLilletevedt/R-ML-models
cd R-ML-models
```

## 2. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS & Linux
# OR
.venv\Scripts\activate   # Windows
```

## 3. Install Python dependencies

```bash
pip install maturin numpy scikit-learn
```

## 4. Build and install Rust module (release mode)

```bash
maturin develop --release
```

This compiles the Rust code into a Python extension module (`rust_core`) and installs it in your environment.

You can verify the installation with:

```python
import rust_core
print(rust_core.__file__)
```

You should see something like:

```
.../site-packages/rust_core.cpython-312-darwin.so
```

## 5. Run the Jupyter notebooks

```bash
pip install jupyter
jupyter notebook
```

Then open:

- `notebooks/tests/test_linear_regression_functional.ipynb`
- `notebooks/benchmarks/benchmark_linear_regression_speed.ipynb`

---

# Example Usage

### Python

```python
import rust_core

model = rust_core.MyRustLinearRegression(
    learning_rate=0.05,
    iterations=1000,
    mode=rust_core.Mode.Regression,
)

model.fit(X_train, y_train)
pred = model.predict(X_test)
```

### Rust

```rust
use Rmodels::linear_regression::MyRustLinearRegression;

let mut model = MyRustLinearRegression::new(1000, 0.05, Mode::Regression);
model.fit(&X_train, &y_train);
let pred = model.predict(&X_test);
```

---

# Benchmarks (current baseline)

Dataset:

- **200,000 training samples**
- **10,000 test samples**
- **40 features**
- **1,000 gradient descent iterations**

### Results

| Model                                   | Time (mean over 5 runs) |
| --------------------------------------- | ----------------------- |
| **sklearn.SGDRegressor (1000 iters)**   | **2.05 s**              |
| **Rust Linear Regression (1000 iters)** | **18.17 s**             |

### Interpretation

`SGDRegressor` uses highly optimized C/Fortran BLAS routines.  
The Rust version is a **naive educational implementation** without:

- SIMD
- threading
- BLAS
- cache-level optimizations

A ~9× slowdown at this stage is entirely expected.

The benchmarking is used to understand scaling before moving to C++/CUDA.

---

# Roadmap

### v1 — Rust (correctness baseline)

- Linear Regression
- KNN
- Python bindings
- Numerical equivalence with scikit-learn (**achieved**)

### v2 — C++ (optimized CPU)

- Rewrite GD in C++
- Add SIMD + multithreading
- Aim to match or exceed sklearn SGD on CPU
- **Note:** I’m a bit stupid and overly optimistic and actually think I can beat them here ;)

### v3 — CUDA (GPU)

- Implement GPU-accelerated matrix operations
- GPU-based gradient descent
- Compare CPU vs GPU scaling

---

If you try the project and want to contribute, feel free to open issues or PRs! 🚀
