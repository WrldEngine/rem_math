extern crate rem_math_gpu;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{exceptions, prelude::*};

pub mod native;

#[pyfunction]
#[pyo3(signature = (arr, simd = false))]
pub fn sum_nparr_int32(_py: Python, arr: PyReadonlyArray1<i32>, simd: bool) -> PyResult<i64> {
    Ok(native::sum_arr_int32(arr.as_slice()?, simd))
}

#[pyfunction]
#[pyo3(signature = (arr, simd = false))]
pub fn sum_arr_int32(_py: Python, arr: Vec<i32>, simd: bool) -> PyResult<i64> {
    Ok(native::sum_arr_int32(arr.as_slice(), simd))
}

#[pyfunction]
#[pyo3(signature = (arr_1, arr_2, method = ""))]
pub fn sum_two_nparr_floats32<'py>(
    _py: Python<'py>,
    arr_1: PyReadonlyArray1<'py, f32>,
    arr_2: PyReadonlyArray1<'py, f32>,
    method: &'py str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if arr_1.len()? != arr_2.len()? {
        return Err(exceptions::PyBaseException::new_err(
            "Array lengths should be equal",
        ));
    }

    let result = native::sum_two_floats32(arr_1.as_slice()?, arr_2.as_slice()?, method);

    Ok(result.into_pyarray(_py))
}

#[pyfunction]
#[pyo3(signature = (arr_1, arr_2, method = ""))]
pub fn sum_two_floats32(
    _py: Python,
    arr_1: Vec<f32>,
    arr_2: Vec<f32>,
    method: &str,
) -> PyResult<Vec<f64>> {
    if arr_1.len() != arr_2.len() {
        return Err(exceptions::PyBaseException::new_err(
            "Array lengths should be equal",
        ));
    }

    Ok(native::sum_two_floats32(
        arr_1.as_slice(),
        arr_2.as_slice(),
        method,
    ))
}

#[pyfunction]
#[pyo3(signature = (arr_1, arr_2, method = ""))]
pub fn sum_two_nparr_ints32<'py>(
    _py: Python<'py>,
    arr_1: PyReadonlyArray1<'py, i32>,
    arr_2: PyReadonlyArray1<'py, i32>,
    method: &'py str,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    if arr_1.len()? != arr_2.len()? {
        return Err(exceptions::PyBaseException::new_err(
            "Array lengths should be equal",
        ));
    }

    let result = native::sum_two_ints32(arr_1.as_slice()?, arr_2.as_slice()?, method);

    match result {
        Ok(r) => Ok(r.into_pyarray(_py)),
        Err(e) => Err(exceptions::PyBaseException::new_err(format!(
            "failed to execute, reason: {}",
            e.to_string()
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (arr_1, arr_2, method = ""))]
pub fn sum_two_ints32(
    _py: Python,
    arr_1: Vec<i32>,
    arr_2: Vec<i32>,
    method: &str,
) -> PyResult<Vec<i64>> {
    if arr_1.len() != arr_2.len() {
        return Err(exceptions::PyBaseException::new_err(
            "Array lengths should be equal",
        ));
    }

    let result = native::sum_two_ints32(arr_1.as_slice(), arr_2.as_slice(), method);

    match result {
        Ok(r) => Ok(r),
        Err(e) => Err(exceptions::PyBaseException::new_err(format!(
            "failed to execute, reason: {}",
            e.to_string()
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (arr_1, arr_2, method = ""))]
pub fn multiply_two_nparr_ints32<'py>(
    _py: Python<'py>,
    arr_1: PyReadonlyArray1<'py, i32>,
    arr_2: PyReadonlyArray1<'py, i32>,
    method: &'py str,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    if arr_1.len()? != arr_2.len()? {
        return Err(exceptions::PyBaseException::new_err(
            "Array lengths should be equal",
        ));
    }

    let result = native::multiply_two_ints32(arr_1.as_slice()?, arr_2.as_slice()?, method);
    Ok(result.into_pyarray(_py))
}

#[pyfunction]
#[pyo3(signature = (arr_1, arr_2, method = ""))]
pub fn multiply_two_ints32(
    _py: Python,
    arr_1: Vec<i32>,
    arr_2: Vec<i32>,
    method: &str,
) -> PyResult<Vec<i64>> {
    if arr_1.len() != arr_2.len() {
        return Err(exceptions::PyBaseException::new_err(
            "Array lengths should be equal",
        ));
    }

    Ok(native::multiply_two_ints32(
        arr_1.as_slice(),
        arr_2.as_slice(),
        method,
    ))
}

#[pyfunction]
#[pyo3(signature = (arr_1, arr_2, method = ""))]
pub fn dot_two_nparr_floats32<'py>(
    _py: Python<'py>,
    arr_1: PyReadonlyArray1<'py, f32>,
    arr_2: PyReadonlyArray1<'py, f32>,
    method: &'py str,
) -> PyResult<f32> {
    if arr_1.len()? != arr_2.len()? {
        return Err(exceptions::PyBaseException::new_err(
            "Array lengths should be equal",
        ));
    }

    let result = native::dot_two_floats32(arr_1.as_slice()?, arr_2.as_slice()?, method);

    match result {
        Ok(r) => Ok(r),
        Err(e) => Err(exceptions::PyBaseException::new_err(format!(
            "failed to execute, reason: {}",
            e.to_string()
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (arr_1, arr_2, method = ""))]
pub fn dot_two_floats32(
    _py: Python,
    arr_1: Vec<f32>,
    arr_2: Vec<f32>,
    method: &str,
) -> PyResult<f32> {
    if arr_1.len() != arr_2.len() {
        return Err(exceptions::PyBaseException::new_err(
            "Array lengths should be equal",
        ));
    }

    let result = native::dot_two_floats32(arr_1.as_slice(), arr_2.as_slice(), method);

    match result {
        Ok(r) => Ok(r),
        Err(e) => Err(exceptions::PyBaseException::new_err(format!(
            "failed to execute, reason: {}",
            e.to_string()
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (arr_1, arr_2, method = ""))]
pub fn mul_mat_floats32(
    _py: Python,
    arr_1: Vec<Vec<f32>>,
    arr_2: Vec<Vec<f32>>,
    method: &str,
) -> PyResult<Vec<Vec<f32>>> {
    if arr_1.len() != arr_2.len() {
        return Err(exceptions::PyBaseException::new_err(
            "Array lengths should be equal",
        ));
    }

    let refs_arr_1: Vec<&[f32]> = arr_1.iter().map(|x| x.as_slice()).collect();
    let refs_arr_2: Vec<&[f32]> = arr_1.iter().map(|x| x.as_slice()).collect();

    let result = native::mul_matrix(&refs_arr_1, &refs_arr_2, method);

    match result {
        Ok(r) => Ok(r),
        Err(e) => Err(exceptions::PyBaseException::new_err(format!(
            "failed to execute, reason: {}",
            e.to_string()
        ))),
    }
}

#[pymodule]
fn _rem_math(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_nparr_int32, m)?)?;
    m.add_function(wrap_pyfunction!(sum_arr_int32, m)?)?;
    m.add_function(wrap_pyfunction!(sum_two_floats32, m)?)?;
    m.add_function(wrap_pyfunction!(sum_two_nparr_floats32, m)?)?;
    m.add_function(wrap_pyfunction!(sum_two_ints32, m)?)?;
    m.add_function(wrap_pyfunction!(sum_two_nparr_ints32, m)?)?;
    m.add_function(wrap_pyfunction!(multiply_two_ints32, m)?)?;
    m.add_function(wrap_pyfunction!(multiply_two_nparr_ints32, m)?)?;
    m.add_function(wrap_pyfunction!(dot_two_floats32, m)?)?;
    m.add_function(wrap_pyfunction!(dot_two_nparr_floats32, m)?)?;
    m.add_function(wrap_pyfunction!(mul_mat_floats32, m)?)?;
    Ok(())
}
