import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import rust_core


# scikit-learn wrapper for knn
class KNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.classes_ = unique_labels(y)

        self.rust_model_ = rust_core.MyRustKNN(k=self.k, mode=rust_core.Mode.Regression)
        self.rust_model_.fit(X, y)

        self.is_fitted_ = True

        return self

    def predict(self, X):
        check_is_fitted(self)

        X = check_array(X)

        return self.rust_model_.predict(X)

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)

        y_true_int = np.asarray(y, dtype=int)
        y_pred_int = np.asarray(y_pred, dtype=int)

        return accuracy_score(y_true_int, y_pred_int)


# scikit-learn wrapper for linear regression
class LinearRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.05, iterations=1000):
        self.learning_rate = learning_rate
        self.iteration = iterations

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.classes_ = unique_labels(y)

        self.rust_model_ = rust_core.MyRustLinearRegression(
            learning_rate=self.learning_rate,
            iterations=self.iterations,
            mode=rust_core.Mode.Regression,
        )

        self.rust_model_.fit(X, y)

        self.is_fitted_ = True

        return self

    def predict(self, X):
        check_is_fitted(self)

        X = check_array(X)

        return self.rust_model_.predict(X)
