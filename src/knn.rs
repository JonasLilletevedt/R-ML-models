use crate::model_base::{Mode, ModelBase};
use ndarray::{Array1, Array2, ArrayView2, Axis};
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

#[pyclass]
pub struct KNN {
    k: usize,
    base: ModelBase,
}

#[pymethods]
impl KNN {
    #[new]
    fn new(k: usize, mode: Mode) -> Self {
        KNN {
            k: (k),
            base: ModelBase::new(mode),
        }
    }

    fn fit(&mut self, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        self.base.X = Some(X.as_array().to_owned());
        self.base.y = Some(y.as_array().to_owned());

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

        let n_train = X_train.nrows();
        if self.k == 0 {
            return Err(PyValueError::new_err(
                "k must be greater than 0 when performing predictions",
            ));
        }
        if self.k > n_train {
            return Err(PyValueError::new_err(format!(
                "k ({}) cannot be greater than the number of training samples ({})",
                self.k, n_train
            )));
        }

        let X_pred_view = X_pred.as_array();

        let distances = calculate_distances(X_pred_view, X_train.view());

        let mut pred_closest_rowi = Array2::<usize>::zeros((X_pred_view.nrows(), self.k));

        for (i, row) in distances.axis_iter(Axis(0)).enumerate() {
            let mut indexed_row: Vec<(f64, usize)> = row
                .iter()
                .enumerate()
                .map(|(idx, &val)| (val, idx))
                .collect();

            let kth_index = self.k - 1;
            indexed_row.select_nth_unstable_by(kth_index, |a: &(f64, usize), b: &(f64, usize)| {
                a.0.partial_cmp(&b.0).unwrap()
            });

            let k_smallest_indices: Vec<usize> =
                indexed_row[..self.k].iter().map(|(_, idx)| *idx).collect();

            pred_closest_rowi
                .row_mut(i)
                .assign(&Array1::from(k_smallest_indices));
        }

        match self.base.mode {
            Mode::Regression => {
                let k_closest_results = pred_closest_rowi.mapv(|idx| y_train[idx]);
                let row_means = k_closest_results.mean_axis(Axis(1)).unwrap();
                let np_res = row_means.into_pyarray(py);
                Ok(np_res.into())
            }
            Mode::Classification => {
                // For classification, implement voting logic here
                let k_closest_results = pred_closest_rowi.mapv(|idx| y_train[idx]);
                let mut preds = Array1::<f64>::zeros(k_closest_results.nrows());
                for (i, row) in k_closest_results.rows().into_iter().enumerate() {
                    let mut frequencies: Vec<(f64, usize)> = Vec::new();

                    for &item in row.iter() {
                        if let Some((_, count)) =
                            frequencies.iter_mut().find(|(val, _)| *val == item)
                        {
                            *count += 1;
                        } else {
                            frequencies.push((item, 1));
                        }
                    }

                    let row_mode = frequencies
                        .into_iter()
                        .max_by_key(|&(_, count)| count)
                        .map(|(val, _)| val);

                    if let Some(mode) = row_mode {
                        preds[i] = mode;
                    } else {
                        preds[i] = -1.0;
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
