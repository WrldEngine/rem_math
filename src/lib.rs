use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

pub mod native;

#[pyfunction]
#[pyo3(signature = (arr, simd = false))]
pub fn sum_nparr_int32<'py>(
    _py: Python<'py>,
    arr: PyReadonlyArray1<'py, i32>,
    simd: bool,
) -> PyResult<i64> {
    Ok(native::sum_arr_int32(arr.as_slice()?.to_vec(), simd))
}

#[pyfunction]
#[pyo3(signature = (arr, simd = false))]
pub fn sum_arr_int32(_py: Python, arr: Vec<i32>, simd: bool) -> PyResult<i64> {
    Ok(native::sum_arr_int32(arr, simd))
}

#[pyfunction]
pub fn sum_two_arr(_py: Python, arr_1: Vec<i32>, arr_2: Vec<i32>) -> PyResult<Vec<i64>> {
    Ok(vec![1; 100])
}

#[pymodule]
fn rmath(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_nparr_int32, m)?)?;
    m.add_function(wrap_pyfunction!(sum_arr_int32, m)?)?;
    Ok(())
}
