use ndarray::{Array1, Array2, ArrayBase, ArrayView2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::collections::HashMap;
use std::mem::zeroed;

use crate::model_base::{Labels, Mode, ModelBase};

#[pyclass]
pub struct MyRustLinearRegression {
    iterations: usize,
    base: ModelBase,
    weights: Array1<f64>,
    bias: f64,
}

#[pymethods]
impl MyRustLinearRegression {
    #[new]
    fn new(iterations: usize, mode: Mode) -> Self {
        MyRustLinearRegression {
            iterations: (iterations),
            base: ModelBase::new(mode),
            weights: Array1::zeros(1), // Not worth the trouble to wrap in option, just start size to 1, not able to missuse due to it checking if fit before predict
            bias: f64 = 0.0,
        }
    }

    fn fit(&mut self, X: PyReadonlyArray2<f64>, y: &Bound<'_, PyAny>) -> PyResult<()> {
        let labels = self.base.parse_and_validate(y)?;
        self.base.X = Some(X.as_array().to_owned());
        self.base.y = Some(labels);
        self.weights =
            Array1::<f64>::zeros(self.base.X.as_ref().expect("Model not fitted").nrows());
        Ok(())
    }

    fn predict(&mut self, py: Python, X_pred: PyReadonlyArray2<f64>) -> PyResult<Py<PyAny>> {
        // Check if model have been fitted
        let (X_train, y_train) = match (&self.base.X, &self.base.y) {
            (Some(X), Some(y)) => (X, y),
            _ => {
                return Err(PyValueError::new_err(
                    "The model must be fitted before making predictions",
                ));
            }
        };

        match y_train {
            Labels::Float(y_train_float) => {
                let res: Array1<f64> = todo!();
                let np_res = res.into_pyarray(py);
                Ok(np_res.into())
            }
            Labels::Int(y_train_int) => {
                // For classification, implement voting logic here
                let res: Array1<i64> = todo!();
                let np_res = res.into_pyarray(py);
                Ok(np_res.into())
            }
        }
    }
}

fn make_loss_predictions(
    X_train: &Array2<f64>,
    y_train: &Array1<f64>,
    weights: &Array1<f64>,
    bias: &f64,
) -> Array1<Mode> {
    let y_loss_predictions: Array1<f64> = Array1::zeros(y_train.len());
    for (i, (row_train, y_row)) in X_train.into_iter().zip(y_train.iter()).enumerate() {
        let row_loss = (*y_row - (weights * *row_train + *bias)).pow2();
    }
    todo!()
}
