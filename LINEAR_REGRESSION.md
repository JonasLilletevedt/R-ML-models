# Linear Regression

The `MyRustLinearRegression` model is a from-scratch implementation of one of the most fundamental machine learning algorithms. It is built in Rust to leverage its performance for computationally intensive tasks like gradient descent.

Linear Regression is a supervised learning algorithm used to predict a continuous dependent variable (`y`) based on a set of independent variables (`X`). It works by finding the best-fitting linear relationship between the features and the target.

## How It Works

The model aims to find the optimal parameters for a linear equation:

`y = wx + b`

- `y`: The predicted value.
- `w` (Weights): A vector of coefficients, one for each feature. It represents the strength of the relationship between each feature and the target.
- `x` (Features): A vector of feature values, e.g, input data.
- `b` (Bias): The intercept term. It's the value of `y` when all features are zero.

### Training with Gradient Descent

The "training" process involves finding the best values for `w` and `b` that minimize the error between the model's predictions and the actual target values. This is achieved using **Gradient Descent**.

1.  **Initialization**: The `weights` and `bias` are initialized to zero.
2.  **Prediction**: The model makes initial predictions using the current `weights` and `bias`.
3.  **Calculate Error**: The difference (error) between the predicted values and the actual values is calculated.
4.  **Compute Gradients**: The algorithm calculates the gradient (partial derivatives) of a cost function (MSE) with respect to the `weights` and `bias`. The gradient indicates the direction of the steepest increase in error.
5.  **Update Parameters**: The `weights` and `bias` are updated by taking a small step in the _opposite_ direction of the gradient. The size of this step is controlled by the `learning_rate`.
    - `new_weight = old_weight - learning_rate * gradient_of_weight`
    - `new_bias = old_bias - learning_rate * gradient_of_bias`
6.  **Repeat**: Steps 2-5 are repeated for a fixed number of `iterations`. With each iteration, the model's parameters get closer to the optimal values that minimize the error.

## Usage in `Rmodels`

### Initialization

The model is initialized by specifying the number of `iterations` and the `learning_rate`.

```python
import Rmodels

# Initialize the model
lr_model = Rmodels.MyRustLinearRegression(
    iterations=1000,
    learning_rate=0.01
)
```

- iterations (usize): The number of times the gradient descent algorithm will run.
- learning_rate (f64): Controls the step size during parameter updates.

## Quick Example (Python)

```python
import numpy as np
import Rmodels

# 1. Sample Data (y = 2*x1 + 3*x2 + 1)
X_train = np.array([
    [1.0, 1.0],
    [2.0, 3.0],
    [4.0, 2.0],
    [5.0, 5.0]
], dtype=np.float64)

y_train = np.array([6.0, 14.0, 15.0, 26.0], dtype=np.float64)

# Data to predict
X_pred = np.array([[3.0, 4.0]], dtype=np.float64)
# Expected y = 2*3 + 3*4 + 1 = 19

# 2. Initialize and Fit the Model
lr_model = Rmodels.MyRustLinearRegression(iterations=1000, learning_rate=0.01)
lr_model.fit(X_train, y_train)

# 3. Predict
predictions = lr_model.predict(X_pred)

print(f"Prediction: {predictions}")
# The output should be close to [19.0]
```

**To Note:** For gradient based algorithms like our linear regression model, the data used needs to be scaled before the model is fitted.
