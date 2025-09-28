use std::collections::btree_set::Difference;

// Import libraries
use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_stats::QuantileExt;
use numpy::{
    IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray, PyReadonlyArray1, PyReadonlyArray2,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};

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

    fn predict(&mut self, py: Python, X_pred: PyReadonlyArray2<f64>) -> PyResult<PyObject> {
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

        let mut pred_closest_rowi = Array2::<usize>::zeros((X_pred_view.nrows(), self.k));

        for (i, row) in distances.axis_iter(Axis(0)).enumerate() {
            let mut indexed_row: Vec<(f64, usize)> = row
                .iter()
                .enumerate()
                .map(|(idx, &val)| (val, idx))
                .collect();

            indexed_row.select_nth_unstable_by(self.k, |a: &(f64, usize), b: &(f64, usize)| {
                a.0.partial_cmp(&b.0).unwrap()
            });

            let k_smallest_indices: Vec<usize> =
                indexed_row[..self.k].iter().map(|(_, idx)| *idx).collect();

            pred_closest_rowi
                .row_mut(i)
                .assign(&Array1::from(k_smallest_indices));
        }

        match y_train {
            Labels::Float(y_train_float) => {
                let k_closest_results = pred_closest_rowi.mapv(|idx| y_train_float[idx]);
                let row_means = k_closest_results.mean_axis(Axis(1)).unwrap();
                let np_res = row_means.into_pyarray(py);
                Ok(np_res.into())
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

    let mut distances = Array2::<f64>::zeros((n_pred, n_train));

    for (i, pred_point) in X_pred_view.outer_iter().enumerate() {
        // For each training point, compute the difference with pred_point
        for (j, train_point) in X_train_view.outer_iter().enumerate() {
            let diff = &pred_point - &train_point;
            let distance = diff.mapv(|x| x.powi(2)).sum().sqrt();
            distances[[i, j]] = distance;
        }
    }

    distances
}

#[pymodule]
fn Xmodels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MyRustKNN>()?;
    m.add_class::<Mode>()?;
    Ok(())
}
