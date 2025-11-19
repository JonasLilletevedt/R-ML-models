import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import coreflux_rust
from . import coreflux_cpp


# scikit-learn wrapper for knn
class KNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.classes_ = unique_labels(y)

        self.rust_model_ = coreflux_rust.MyRustKNN(
            k=self.k, mode=coreflux_rust.Mode.Regression
        )
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
        self.iterations = iterations

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.classes_ = unique_labels(y)

        self.rust_model_ = coreflux_rust.MyRustLinearRegression(
            learning_rate=self.learning_rate,
            iterations=self.iterations,
            mode=coreflux_rust.Mode.Regression,
        )

        self.rust_model_.fit(X, y)

        self.is_fitted_ = True

        return self

    def predict(self, X):
        check_is_fitted(self)

        X = check_array(X)

        return self.rust_model_.predict(X)


class CppLinearRegression(BaseEstimator, RegressorMixin):
    """
    scikit-learn compatible wrapper around the optimized C++ LinearRegression implementation.
    """

    def __init__(self, learning_rate=0.05, iterations=1000, library_path=None):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.library_path = library_path

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._cpp_model = coreflux_cpp.LinearRegression(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            library_path=self.library_path,
        )
        self._cpp_model.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self._cpp_model.predict(X)
