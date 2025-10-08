use pyo3::prelude::*;

pub mod decision_tree;
pub mod knn;
pub mod linear_regression;
pub mod model_base;

use decision_tree::MyRustDecisionTree;
use knn::MyRustKNN;
use linear_regression::MyRustLinearRegression;
use model_base::Mode;

#[pymodule]
fn rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MyRustKNN>()?;
    m.add_class::<MyRustLinearRegression>()?;
    m.add_class::<MyRustDecisionTree>()?;
    m.add_class::<Mode>()?;
    Ok(())
}
