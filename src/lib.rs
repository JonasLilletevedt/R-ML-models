use pyo3::prelude::*;

pub mod knn;
mod linear_regression;
pub mod model_base;

use knn::MyRustKNN;
use model_base::Mode;

#[pymodule]
fn Xmodels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MyRustKNN>()?;
    m.add_class::<Mode>()?;
    Ok(())
}
