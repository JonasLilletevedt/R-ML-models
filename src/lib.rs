use pyo3::prelude::*;

pub mod knn;
pub mod linear_regression;
pub mod model_base;

use knn::KNN;
use linear_regression::LinearRegresssion;
use model_base::Mode;

#[pymodule]
fn coreflux_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KNN>()?;
    m.add_class::<LinearRegresssion>()?;
    m.add_class::<Mode>()?;
    Ok(())
}
