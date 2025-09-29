use ndarray::{Array1, Array2, ArrayView2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::collections::HashMap;

use crate::model_base::{Labels, Mode, ModelBase};

#[pyclass]
pub struct MyRustKNN {
    k: usize,
    base: ModelBase,
}

#[pymethods]
impl MyRustKNN {
    #[new]
    fn new(k: usize, mode: Mode) -> Self {
        MyRustKNN {
            k: (k),
            base: ModelBase::new(mode),
        }
    }

    fn fit(&mut self, X: PyReadonlyArray2<f64>, y: &Bound<'_, PyAny>) -> PyResult<()> {
        let labels = self.base.parse_and_validate(y)?;
        self.base.X = Some(X.as_array().to_owned());
        self.base.y = Some(labels);

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
                let k_closest_results = pred_closest_rowi.mapv(|idx| y_train_int[idx]);
                let mut preds = Array1::<i64>::zeros(k_closest_results.nrows());
                for (i, row) in k_closest_results.rows().into_iter().enumerate() {
                    let mut frequencies: HashMap<i64, i64> = HashMap::new();

                    for item in row.iter() {
                        *frequencies.entry(item.clone()).or_insert(0) += 1;
                    }

                    let row_mode = frequencies
                        .into_iter()
                        .max_by_key(|&(_, count)| count)
                        .map(|(val, _)| val);

                    if let Some(mode) = row_mode {
                        preds[i] = mode;
                    } else {
                        preds[i] = -1;
                    }
                }
                let np_res = preds.into_pyarray(py);
                Ok(np_res.into())
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
