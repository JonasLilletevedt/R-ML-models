use std::collections::btree_set::Difference;

// Import libraries
use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_stats::QuantileExt;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

#[pyclass]
#[derive(Clone, Copy)]
pub enum Mode {
    Classification,
    Regression,
}

enum Labels {
    Int(ndarray::Array1<i64>),
    Float(ndarray::Array1<f64>),
}

#[pyclass]
pub struct MyRustKNN {
    k: usize,
    mode: Mode,
    X: Option<ndarray::Array2<f64>>,
    y: Option<Labels>,
}

#[pymethods]
impl MyRustKNN {
    #[new]
    fn new(k: usize, mode: Mode) -> Self {
        MyRustKNN {
            k: (k),
            mode: (mode),
            X: (None),
            y: (None),
        }
    }

    fn fit(&mut self, X: PyReadonlyArray2<f64>, y: &Bound<'_, PyAny>) -> PyResult<()> {
        let labels = self.parse_and_validate(y)?;
        self.X = Some(X.as_array().to_owned());
        self.y = Some(labels);

        Ok(())
    }

    fn predict(&mut self, X_pred: PyReadonlyArray2<f64>) -> PyResult<PyObject> {
        // Check if model have been fitted
        let (X_train, y_train) = match (&self.X, &self.y) {
            (Some(X), Some(y)) => (X, y),
            _ => {
                return Err(PyValueError::new_err(
                    "The model must be fitted before making predictions",
                ));
            }
        };

        let X_pred_view = X_pred.as_array();

        let distances = calculate_distances(X_pred_view, X_train.view());

        match y_train {
            Labels::Float(y_train_float) => {
                let prediction = distances.
                Err(PyValueError::new_err("Regression not yet implemented"))
            }
            Labels::Int(y_train_int) => {
                // For classification, implement voting logic here
                Err(PyValueError::new_err("Classification not yet implemented"))
            }
        }
    }
}

impl MyRustKNN {
    fn parse_and_validate(&self, y_obj: &Bound<'_, PyAny>) -> PyResult<Labels> {
        match self.mode {
            Mode::Regression => {
                if let Ok(y_arr_mutable) = y_obj.downcast::<PyArray1<f64>>() {
                    let y_arr_read_only = y_arr_mutable.readonly();

                    Ok(Labels::Float(y_arr_read_only.as_array().to_owned()))
                } else {
                    Err(PyTypeError::new_err(
                        "For Regression mode, labels must be a NumPy array of type f64",
                    ))
                }
            }

            Mode::Classification => {
                if let Ok(y_arr_mutable) = y_obj.downcast::<PyArray1<i64>>() {
                    let y_arr_read_only = y_arr_mutable.readonly();

                    Ok(Labels::Int(y_arr_read_only.as_array().to_owned()))
                } else {
                    Err(PyTypeError::new_err(
                        "For classification mode, labels must be a NumPy array of type i64",
                    ))
                }
            }
        }
    }
}

fn calculate_distances(X_pred_view: ArrayView2<f64>, X_train_view: ArrayView2<f64>) -> Array2<f64> {
    let n_pred = X_pred_view.nrows();
    let n_train = X_train_view.nrows();
    let n_features = X_train_view.ncols();

    let mut distances = Array::<f64, ndarray::Ix2>::zeros((n_pred, n_train));

    for (i, pred_point) in X_pred_view.outer_iter().enumerate() {
        let differences = &pred_point - &X_train_view;

        let squared_differences = differences.mapv(|x| x.powi(2));

        let sum_squared_diffs = squared_differences.sum_axis(Axis(1));

        let final_distances = squared_differences.mapv(f64::sqrt);

        distances.row_mut(i).assign(&final_distances);
    }

    return distances;
}
