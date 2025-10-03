# K-Nearest Neighbors (KNN)

The `MyRustKNN` model is a from-scratch implementation of the K-Nearest Neighbors algorithm, built in Rust for high performance and clarity.

- **Non-parametric**: It makes no assumptions about the underlying data distribution.
- **Lazy Learning**: The model memorizes the training data instead of learning a function from it. All calculations happen at prediction time.

## How It Works

### `fit` (The "Training" Phase)

- The `fit` method is simple and fast.
- It loads and stores the training dataset (`X_train`, `y_train`) in memory.
- No actual "learning" occurs in this step.

### `predict` (The Prediction Phase)

For each new data point, the algorithm will:

1.  **Calculate Distances**: Compute the Euclidean distance between the new point and every point in the training data.
2.  **Find K-Nearest Neighbors**: Identify the `k` training points with the smallest distances to the new point. `k` is set during model initialization.
3.  **Make a Prediction**:
    - **Classification**: The prediction is the most frequent class (the **mode**) among the `k` neighbors.
    - **Regression**: The prediction is the **average** of the values of the `k` neighbors.

## Usage in `Rmodels`

### Initialization

The model is initialized by specifying `k` and the `mode`.

```python
import Rmodels

# For Classification
knn_classifier = Rmodels.MyRustKNN(k=3, mode="classification")

# For Regression
knn_regressor = Rmodels.MyRustKNN(k=5, mode="regression")
```

- k (usize): The number of neighbors to use.
- mode (str): Either "classification" or "regression".

## Quick Example (Python)

### Regression

```python
import numpy as np
import Rmodels

# Sample Data
X_train = np.array([,,,], dtype=np.float64)
y_train = np.array([3.0, 5.0, 7.0, 11.0], dtype=np.float64) # Continuous values

# Fit and Predict
knn_regressor = Rmodels.MyRustKNN(k=3, mode="regression")
knn_regressor.fit(X_train, y_train)
predictions = knn_regressor.predict(X_pred)

print(f"Prediction: {predictions}")
# Expected Output: Prediction: [5.] (Average of 3.0, 5.0, 7.0)
```
