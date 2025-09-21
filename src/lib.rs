// Import libraries
use ndarray::{Array1, Array2};
use ndarray_stats::QuantileExt;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

pub enum Mode {
    Classification,
    Regression,
}

pub enum Labels {
    Int(PyReadonlyArray1<i64>),
    Float(PyReadonlyArray1<f64>),
}

#[pyclass]
pub struct MyRustKNN {
    k: usize,
    mode: Mode,
    X_train: Option<Array2<f64>>,
    y_train: Option<Labels>,
}

#[pymethods]
impl MyRustKNN {
    #[new]
    fn new(k: usize, mode: Mode) -> Self {
        MyRustKNN {
            k: (k),
            mode: (mode),
            X_train: (None),
            y_train: (None),
        }
    }

    fn fit(&mut self, X_train: PyReadonlyArray2<f64>, y_train: Labels) -> PyResult<()> {
        self.X_train = Some(X_train.to_owned().as_array().to_owned());
        self.y_train = Some(y_train.to_owned().as_array().to_owned());
        Ok(())
    }
}
