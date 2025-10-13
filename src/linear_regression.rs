use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::model_base::{Mode, ModelBase};

#[pyclass]
pub struct MyRustLinearRegression {
    iterations: usize,
    learning_rate: f64,
    base: ModelBase,
    weights: Array1<f64>,
    bias: f64,
}

#[pymethods]
impl MyRustLinearRegression {
    #[new]
    fn new(iterations: usize, learning_rate: f64, mode: Mode) -> Self {
        MyRustLinearRegression {
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
        self.weights =
            Array1::<f64>::zeros(self.base.X.as_ref().expect("Model not fitted").ncols());

        for _ in 0..self.iterations {
            self.gradient_descent();
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

impl MyRustLinearRegression {
    fn gradient_descent(&mut self) {
        let X_train = self
            .base
            .X
            .as_ref()
            .expect("Code Error [fn gradient_descent] -- self.base.X could not unwrap_failed");
        let y_train = self
            .base
            .y
            .as_ref()
            .expect("Code Error [fn gradient_descent] -- self.base.y could not unwrap_failed");

        // predictions
        let predictions = make_predictions(X_train, &self.weights, &self.bias);

        // errors
        let errors = &predictions - y_train;

        // gradients
        let weight_loss_derivatives = calculate_weight_loss_derivative(&errors, X_train);

        // update weights and bias
        self.weights -= &(weight_loss_derivatives * self.learning_rate);

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
