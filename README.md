# Rmodels: Machine Learning Algorithms Built from Scratch

Do you really understand something if you cannot build it yourself? Maybe, maybe not, but it feels great to now you can.

I often use machine learning algorithms via libraries, but rarely stop to think about how they work under the hood. This project is my way of building them from the ground up in Rust and truly understanding their mechanics.

---

## Features Implemented

- **Linear Regression** with gradient descent
- **K-Nearest Neighbors (KNN)** for regression and classification
- **Python bindings** using PyO3, so you can use these algorithms directly in Python

---

## Quick Example

### Rust

```rust
use Rmodels::linear_regression::MyRustLinearRegression;

let mut model = LinearRegression::new();
model.fit(&X_train, &y_train);
let predictions = model.predict(&X_test);
```

### Python (via PyO3 bindings)

```python
import importlib
import Rmodels

model = Rmodels.MyRustLinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Coming Soon

- scikit-learn compatible wrapper
- Decision Tree algorithm

## Installation

```bash
git clone <repo-url>
cd Rmodels
maturin develop
```

## Additional Resources

- Detailed .md files explaining integration of Rust with Python
- Algorithm explanation .md files for Linear Regression and KNN
