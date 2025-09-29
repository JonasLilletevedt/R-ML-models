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

pub enum Labels {
    Int(Array1<i64>),
    Float(Array1<f64>),
}

pub struct ModelBase {
    pub mode: Mode,
    pub X: Option<Array2<f64>>,
    pub y: Option<Labels>,
}

impl ModelBase {
    pub fn new(mode: Mode) -> Self {
        Self {
            mode,
            X: None,
            y: None,
        }
    }

    pub fn parse_and_validate(&self, y_obj: &Bound<'_, PyAny>) -> PyResult<Labels> {
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
