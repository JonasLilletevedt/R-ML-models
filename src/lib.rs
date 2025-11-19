use pyo3::prelude::*;

pub mod decision_tree;
pub mod knn;
pub mod linear_regression;
pub mod model_base;

use decision_tree::DecisionTree;
use knn::KNN;
use linear_regression::LinearRegresssion;
use model_base::Mode;

#[pymodule]
fn coreflux_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KNN>()?;
    m.add_class::<LinearRegresssion>()?;
    m.add_class::<DecisionTree>()?;
    m.add_class::<Mode>()?;
    Ok(())
}
