use ndarray::{Array, Array1, Array2};
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::model_base::{Mode, ModelBase};

#[pyclass]
pub struct LinearRegresssion {
    iterations: usize,
    learning_rate: f64,
    base: ModelBase,
    weights: Array1<f64>,
    bias: f64,
}

#[pymethods]
impl LinearRegresssion {
    #[new]
    fn new(iterations: usize, learning_rate: f64, mode: Mode) -> Self {
        LinearRegresssion {
            iterations: (iterations),
            learning_rate: (learning_rate),
            base: ModelBase::new(mode),
            weights: Array1::zeros(1), // Not worth the trouble to wrap in option, just start size to 1, not able to missuse due to it checking if fit before predict
            bias: 0.0,
        }
    }

    fn fit(&mut self, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        self.base.X = Some(X.as_array().to_owned());
        self.base.y = Some(y.as_array().to_owned());

        let X_train = self.base.X.as_ref().unwrap();
        let n_features = X_train.ncols();

        self.weights =
            Array1::<f64>::zeros(self.base.X.as_ref().expect("Model not fitted").ncols());

        let mut predictions = Array1::<f64>::zeros(X_train.nrows());
        let mut errors = Array1::<f64>::zeros(X_train.nrows());
        let mut grad = Array1::<f64>::zeros(n_features);

        for _ in 0..self.iterations {
            self.gradient_descent(&mut predictions, &mut errors, &mut grad);
        }
        Ok(())
    }

    fn predict(&mut self, py: Python, X_pred: PyReadonlyArray2<f64>) -> PyResult<Py<PyAny>> {
        // Check if model have been fitted
        let (X_train, _y_train) = match (&self.base.X, &self.base.y) {
            (Some(X), Some(y)) => (X, y),
            _ => {
                return Err(PyValueError::new_err(
                    "The model must be fitted before making predictions",
                ));
            }
        };

        let X_pred_view = X_pred.as_array().to_owned();

        match self.base.mode {
            Mode::Regression => {
                let res: Array1<f64> = make_predictions(&X_pred_view, &self.weights, &self.bias);
                let np_res = res.into_pyarray(py);
                Ok(np_res.into())
            }
            Mode::Classification => Err(PyValueError::new_err(
                "Classification mode is not implemented for linear regression",
            )),
        }
    }
}

impl LinearRegresssion {
    fn gradient_descent(
        &mut self,
        predictions: &mut Array1<f64>,
        errors: &mut Array1<f64>,
        grad: &mut Array1<f64>,
    ) {
        let X_train = self.base.X.as_ref().unwrap();
        let y_train = self.base.y.as_ref().unwrap();
        let m = X_train.nrows() as f64;

        // predictions
        *predictions = X_train.dot(&self.weights);
        *predictions += self.bias;

        // errors
        *errors = predictions.clone();
        *errors -= y_train;

        // gradients
        grad.fill(0.0);
        for (j, mut g_j) in grad.iter_mut().enumerate() {
            let col = X_train.column(j);
            *g_j = col.dot(errors) / m;
        }

        // weights -= learning_rate * grad
        // unngå (grad * lr) som også lager ny array:
        for (w, g) in self.weights.iter_mut().zip(grad.iter()) {
            *w -= self.learning_rate * *g;
        }

        // bias -= mean(errors)
        let bias_derivative = errors.mean().unwrap();
        self.bias -= bias_derivative * self.learning_rate;
    }
}

fn make_predictions(X_train: &Array2<f64>, weights: &Array1<f64>, bias: &f64) -> Array1<f64> {
    X_train.dot(weights) + *bias
}

fn calculate_weight_loss_derivative(errors: &Array1<f64>, X_train: &Array2<f64>) -> Array1<f64> {
    let m = X_train.nrows();

    let weights_derivatives = X_train
        .columns()
        .into_iter()
        .map(|feat| feat.dot(errors) / m as f64)
        .collect::<Array1<f64>>();

    weights_derivatives
}
