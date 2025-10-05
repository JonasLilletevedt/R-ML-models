use pyo3::prelude::*;

pub mod knn;
pub mod linear_regression;
pub mod model_base;

use knn::MyRustKNN;
use linear_regression::MyRustLinearRegression;
use model_base::Mode;

#[pymodule]
fn rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MyRustKNN>()?;
    m.add_class::<MyRustLinearRegression>()?;
    m.add_class::<Mode>()?;
    Ok(())
}
