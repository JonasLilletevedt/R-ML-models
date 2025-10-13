use core::f64;

use ndarray::{Array1, Array2, ArrayView2, Axis};
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::model_base::{Mode, ModelBase};

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

    fn fit(&mut self, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        self.base.X = Some(X.as_array().to_owned());
        self.base.y = Some(y.as_array().to_owned());

        let X_train = match &self.base.X {
            Some(X) => X,
            _ => {
                return Err(PyValueError::new_err(
                    "Error fitting X_train data [MyRustDecisionTree][fn fit()]",
                ));
            }
        };

        let y_train = match &self.base.y {
            Some(y) => y,
            _ => {
                return Err(PyValueError::new_err(
                    "Error fitting y_train data [MyRustDecisionTree][fn fit()]",
                ))
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

        // Sudo code ---
        // check all possible values in train for each feature to find best gini index, store current best row, col
        // Add that question as first node, do that recursivley to set depth
        // We also need to check on only what is left in that node after the previous question
        let best_q = find_best_q(X_train, y_train, *&self.base.mode);

        todo!()
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
        todo!()
    }
}

pub fn find_best_q(X_train: &Array2<f64>, y_train: &Array1<f64>, mode: Mode) -> (f64, usize) {
    let split_function = get_split_function(mode);
    let rows = X_train.nrows();
    let cols = X_train.ncols();

    let mut best_q = (0.0, 0);
    let mut best_loss = f64::INFINITY;

    for row in 0..rows {
        for col in 0..cols {
            let split_val = X_train[[row, col]];
            let feat_index = col;
            let loss = split_function(split_val, feat_index, &X_train, &y_train);
            if loss < best_loss {
                best_loss = loss;
                best_q = (split_val, feat_index);
            }
        }
    }
    best_q
}

fn get_split_function(mode: Mode) -> fn(f64, usize, &Array2<f64>, &Array1<f64>) -> f64 {
    match mode {
        Mode::Classification => calculate_gini_index,
        Mode::Regression => calculate_mse,
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
