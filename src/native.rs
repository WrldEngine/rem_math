#![feature(stdsimd)]
#![feature(available_parallelism)]
use std::thread::available_parallelism;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use rayon::prelude::*;

const WAY_8_SZ: usize = 8;
const WAY_4_SZ: usize = 4;

/// # Sum of integer 32 arrays elements
///
/// Note: this function works with AVX2 instructions
/// # Example:
/// ```no_run
/// let arr1: Vec<i32> = vec![1; 10];
/// let arr2: i64 = unsafe { sum_arr_int32_simd(arr1) };
/// assert_eq!(arr2, 10);
/// ```
#[target_feature(enable = "avx2")]
unsafe fn sum_arr_int32_simd(arr: &[i32]) -> i64 {
    let mut sum = _mm256_setzero_si256();
    let chunks = arr.chunks_exact(WAY_8_SZ);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let ptr = chunk.as_ptr() as *const __m256i;
        let vec = _mm256_loadu_si256(ptr);
        sum = _mm256_add_epi32(sum, vec);
    }

    let mut temp = [0i32; WAY_8_SZ];
    _mm256_storeu_si256(temp.as_mut_ptr() as *mut __m256i, sum);
    let mut total = temp.iter().sum::<i32>();

    total += remainder.iter().sum::<i32>();

    return total as i64;
}

/// # Sum values of indexes of two float 32 arrays
///
/// Note: this function works with SSE instructions
/// Note: this function may not be more efficiently because LLVM compiler optimised naive function already in compile time
/// # Example:
/// ```no_run
/// let arr1: Vec<f32> = vec![1.0; 10];
/// let arr2: Vec<f32> = vec![1.0; 10];
/// let arr3: Vec<f64> = unsafe { sum_two_floats32_simd(arr1, arr2) };
/// assert_eq!(arr3, vec![2.0; 10]);
/// ```
#[target_feature(enable = "sse")]
unsafe fn sum_two_floats32_simd(arr_1: &[f32], arr_2: &[f32]) -> Vec<f64> {
    let arr_len = arr_1.len();
    let mut result = vec![0.0f64; arr_len];
    let chunks = arr_len / WAY_4_SZ;

    for i in 0..chunks {
        let offset = i * WAY_4_SZ;

        let a = _mm_loadu_ps(arr_1[offset..].as_ptr());
        let b = _mm_loadu_ps(arr_2[offset..].as_ptr());
        let sum = _mm_add_ps(a, b);

        let mut temp: [f32; WAY_4_SZ] = std::mem::zeroed();
        _mm_storeu_ps(temp.as_mut_ptr(), sum);

        result[offset] = temp[0] as f64;
        result[offset + 1] = temp[1] as f64;
        result[offset + 2] = temp[2] as f64;
        result[offset + 3] = temp[3] as f64;
    }

    for i in (chunks * WAY_4_SZ)..arr_len {
        result[i] = (arr_1[i] + arr_2[i]) as f64;
    }

    result
}

/// # Sum values of indexes of two integer 32 arrays
///
/// Note: this function works with AVX2 instructions
/// # Example:
/// ```no_run
/// let arr1: Vec<i32> = vec![1; 10];
/// let arr2: Vec<i32> = vec![1; 10];
/// let arr3: Vec<i32> = unsafe { sum_two_ints32_simd(arr1, arr2) };
/// assert_eq!(arr3, vec![2; 10]);
/// ```
#[target_feature(enable = "avx2")]
unsafe fn sum_two_ints32_simd(arr_1: &[i32], arr_2: &[i32]) -> Vec<i64> {
    let len = arr_1.len();
    let mut result = vec![0i64; len];
    let chunks = len / WAY_8_SZ;

    for i in 0..chunks {
        let offset = i * WAY_8_SZ;

        let a = _mm256_loadu_si256(arr_1[offset..].as_ptr() as *const __m256i);
        let b = _mm256_loadu_si256(arr_2[offset..].as_ptr() as *const __m256i);

        let sum = _mm256_add_epi32(a, b);

        let sum_lo = _mm256_extracti128_si256(sum, 0);
        let sum_hi = _mm256_extracti128_si256(sum, 1);

        let sum_lo_i64 = _mm_cvtepi32_epi64(sum_lo);
        let sum_hi_i64 = _mm_cvtepi32_epi64(sum_hi);

        _mm_storeu_si128(result[offset..].as_mut_ptr() as *mut __m128i, sum_lo_i64);
        _mm_storeu_si128(
            result[offset + WAY_4_SZ..].as_mut_ptr() as *mut __m128i,
            sum_hi_i64,
        );
    }

    for i in (chunks * WAY_8_SZ)..len {
        result[i] = (arr_1[i] + arr_2[i]) as i64;
    }

    result
}

/// # Multiply values of indexes of two integer 32 arrays
///
/// Note: this function works with AVX2 instructions
/// # Example:
/// ```no_run
/// let arr1 = [1; 10];
/// let arr2 = [1; 10];
/// let arr3 = unsafe { multiply_two_ints32_simd(arr1, arr2) };
/// assert_eq!(arr3, [1; 10]);
/// ```
#[target_feature(enable = "avx2")]
unsafe fn multiply_two_ints32_simd(arr_1: &[i32], arr_2: &[i32]) -> Vec<i64> {
    let arr_len = arr_1.len();
    let mut result: Vec<i64> = vec![0i64; arr_len];
    let chunks = arr_len / WAY_8_SZ;

    for i in 0..chunks {
        let offset = i * WAY_8_SZ;

        let a_1_vec = _mm256_loadu_si256(arr_1[offset..].as_ptr() as *const __m256i);
        let a_2_vec = _mm256_loadu_si256(arr_2[offset..].as_ptr() as *const __m256i);
        let result_vec = _mm256_mullo_epi32(a_1_vec, a_2_vec);

        let mut temp = [0i32; 8];
        _mm256_storeu_si256(temp.as_mut_ptr() as *mut __m256i, result_vec);

        for j in 0..WAY_8_SZ {
            result[offset + j] = temp[j] as i64;
        }
    }

    for i in (chunks * WAY_8_SZ)..arr_len {
        result[i] = (arr_1[i] as i64) * (arr_2[i] as i64);
    }

    result
}

#[inline(always)]
pub fn sum_arr_int32(arr: &[i32], simd: bool) -> i64 {
    if simd && arr.len() >= WAY_8_SZ {
        unsafe { return sum_arr_int32_simd(arr) }
    }

    let sum: i32 = arr.iter().sum();
    sum as i64
}

#[inline(always)]
pub fn sum_two_floats32(arr_1: &[f32], arr_2: &[f32], method: &str) -> Vec<f64> {
    match method {
        "simd" => unsafe { return sum_two_floats32_simd(arr_1, arr_2) },
        "threading" => {
            return arr_1
                .par_iter()
                .zip(arr_2.par_iter())
                .map(|(&a, &b)| (a + b) as f64)
                .collect()
        }
        &_ => {
            let mut result: Vec<f64> = vec![0.0; arr_1.len()];
            for ((arr_3_val, arr_1_val), arr_2_val) in result.iter_mut().zip(arr_1).zip(arr_2) {
                *arr_3_val = (arr_1_val + arr_2_val) as f64;
            }

            return result;
        }
    }
}

#[inline(always)]
pub fn sum_two_ints32(arr_1: &[i32], arr_2: &[i32], simd: bool) -> Vec<i64> {
    if simd {
        unsafe { return sum_two_ints32_simd(arr_1, arr_2) }
    }

    let mut result: Vec<i64> = vec![0; arr_1.len()];
    for ((arr_3_val, arr_1_val), arr_2_val) in result.iter_mut().zip(arr_1).zip(arr_2) {
        *arr_3_val = (arr_1_val + arr_2_val) as i64;
    }

    result
}

#[inline(always)]
pub fn multiply_two_ints32(arr_1: &[i32], arr_2: &[i32], method: &str) -> Vec<i64> {
    match method {
        "simd" => unsafe { return multiply_two_ints32_simd(arr_1, arr_2) },
        "threading" => {
            return arr_1
                .par_iter()
                .zip(arr_2.par_iter())
                .map(|(&a, &b)| (a * b) as i64)
                .collect()
        }
        &_ => {
            let mut result: Vec<i64> = vec![0; arr_1.len()];
            for i in 0..arr_1.len() {
                result[i] = (arr_1[i] * arr_2[i]) as i64
            }

            return result;
        }
    }
}
