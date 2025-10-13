use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

#[pyclass]
#[derive(Clone, Copy)]
pub enum Mode {
    Classification,
    Regression,
}

pub struct ModelBase {
    pub mode: Mode,
    pub X: Option<Array2<f64>>,
    pub y: Option<Array1<f64>>,
}

impl ModelBase {
    pub fn new(mode: Mode) -> Self {
        Self {
            mode,
            X: None,
            y: None,
        }
    }
}
