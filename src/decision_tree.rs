use std::collections::HashSet;

use ndarray::{Array1, Array2, ArrayView2, Axis};
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::model_base::{Labels, Mode, ModelBase};

#[pyclass]
pub struct MyRustDecisionTree {
    base: ModelBase,
}

#[pymethods]
impl MyRustDecisionTree {
    #[new]
    fn new(k: usize, mode: Mode) -> Self {
        MyRustDecisionTree {
            base: ModelBase::new(mode),
        }
    }

    fn fit(&mut self, X: PyReadonlyArray2<f64>, y: &Bound<'_, PyAny>) -> PyResult<()> {
        let labels = self.base.parse_and_validate(y)?;
        self.base.X = Some(X.as_array().to_owned());
        self.base.y = Some(labels);

        let X_train = match &self.base.X {
            Some(X) => X,
            _ => {
                return Err(PyValueError::new_err(
                    "Error fitting train data [MyRustDecisionTree][fn fit()]",
                ));
            }
        };

        let uniq_sorted_rows: Vec<Vec<f64>> = X_train
            .axis_iter(ndarray::Axis(0))
            .map(|row| {
                let mut v: Vec<f64> = row.to_vec();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v.dedup_by(|a, b| a.to_bits() == b.to_bits());
                v
            })
            .collect();

        Ok(todo!())
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
                let np_res: Array1<f64> = todo!();
                let res = np_res.into_pyarray(py);
                Ok(np_res.into())
            }
            Labels::Int(y_train_int) => {
                let np_res = todo!();
                let res = np_res.into_pyarray(py);
                Ok(np_res.into())
            }
        }
    }
}

pub fn calculate_gini_index(
    split_val: f64,
    feat_index: usize,
    X_train: &Array2<f64>,
    y_train: &Array1<f64>,
) -> f64 {
    let n = y_train.len() as f64;
    let feat = X_train.column(feat_index);

    let mask_left: Array1<f64> = feat.mapv(|v| if v <= split_val { 1.0 } else { 0.0 });

    let y_pos: Array1<f64> = y_train.mapv(|y| if y != 0.0 { 1.0 } else { 0.0 });

    let left_total = mask_left.sum();
    let right_total = n - left_total;

    let left_pos = (&y_pos * &mask_left).sum();
    let right_pos = y_pos.sum() - left_pos;

    let gini_node = |pos: f64, total: f64| {
        if total <= 0.0 {
            0.0
        } else {
            let p = pos / total;
            2.0 * p * (1.0 - p)
        }
    };

    let g_left = gini_node(left_pos, left_total);
    let g_right = gini_node(right_pos, right_total);

    (left_total / n) * g_left + (right_total / n) * g_right
}

pub fn calculate_mse(
    split_val: f64,
    feat_index: usize,
    X_train: &Array2<f64>,
    y_train: &Array1<f64>,
) -> f64 {
    let feat = X_train.column(feat_index);
    let n = feat.len();
    let mask_left = feat.mapv(|v| if v <= split_val { 1.0 } else { 0.0 });
    let mask_right = feat.mapv(|v| if v > split_val { 1.0 } else { 0.0 });

    let n_left = mask_left.sum();
    let n_right = n as f64 - n_left;

    let left_mean = if n_left > 0.0 {
        &y_train.dot(&mask_left) / n_left
    } else {
        0.0
    };

    let right_mean = if n_right > 0.0 {
        &y_train.dot(&mask_right) / n_right
    } else {
        0.0
    };

    let mse = y_train
        .indexed_iter()
        .map(|(i, &val)| {
            if mask_left[i] != 0.0 {
                (val - left_mean).powi(2)
            } else {
                (val - right_mean).powi(2)
            }
        })
        .sum::<f64>()
        / n as f64;

    mse
}
