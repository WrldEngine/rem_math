use std::arch::x86_64::*;

const WAY_8_SZ: usize = 8;
const WAY_4_SZ: usize = 4;

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn sum_arr_int32_simd(arr: Vec<i32>) -> i64 {
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

#[target_feature(enable = "sse")]
#[inline]
unsafe fn sum_two_floats32_simd(arr_1: Vec<f32>, arr_2: Vec<f32>) -> Vec<f64> {
    let arr_len = arr_1.len();

    let mut result = Vec::with_capacity(arr_len);
    let chunks = arr_len / WAY_4_SZ;

    for i in 0..chunks {
        let offset = i * WAY_4_SZ;

        let a_1_vec = _mm_loadu_ps(arr_1[offset..].as_ptr());
        let a_2_vec = _mm_loadu_ps(arr_2[offset..].as_ptr());
        let result_vec = _mm_add_ps(a_1_vec, a_2_vec);

        let mut store_buff = [0.0f32; WAY_4_SZ];
        _mm_storeu_ps(store_buff.as_mut_ptr(), result_vec);

        result.extend(store_buff.iter().map(|&x| x as f64));
    }

    for i in (chunks * WAY_4_SZ)..arr_len {
        result.push((arr_1[i] + arr_2[i]) as f64);
    }

    result
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn sum_two_ints32_simd(arr_1: Vec<i32>, arr_2: Vec<i32>) -> Vec<i64> {
    let arr_len = arr_1.len();

    let mut result: Vec<i64> = Vec::with_capacity(arr_len);
    let chunks = arr_len / WAY_8_SZ;

    for i in 0..chunks {
        let offset = i * WAY_8_SZ;

        let a_1_vec = _mm256_loadu_si256(arr_1[offset..].as_ptr() as *const __m256i);
        let a_2_vec = _mm256_loadu_si256(arr_2[offset..].as_ptr() as *const __m256i);
        let result_vec = _mm256_add_epi32(a_1_vec, a_2_vec);

        let mut store_buff = [0i32; WAY_8_SZ];
        _mm256_storeu_si256(store_buff.as_mut_ptr() as *mut __m256i, result_vec);

        result.extend(store_buff.iter().map(|&x| x as i64));
    }

    for i in (chunks * WAY_4_SZ)..arr_len {
        result.push((arr_1[i] + arr_2[i]) as i64);
    }

    result
}

#[inline]
pub fn sum_arr_int32(arr: Vec<i32>, simd: bool) -> i64 {
    if simd && arr.len() >= WAY_8_SZ {
        unsafe { return sum_arr_int32_simd(arr) }
    }

    let sum: i32 = arr.iter().sum();
    sum as i64
}

#[inline]
pub fn sum_two_floats32(arr_1: Vec<f32>, arr_2: Vec<f32>, simd: bool) -> Vec<f64> {
    if simd {
        unsafe { return sum_two_floats32_simd(arr_1, arr_2) }
    }

    let mut result: Vec<f64> = vec![0.0; arr_1.len()];
    for ((arr_3_val, arr_1_val), arr_2_val) in result.iter_mut().zip(&arr_1).zip(&arr_2) {
        *arr_3_val = (arr_1_val + arr_2_val) as f64;
    }

    result
}

#[inline]
pub fn sum_two_ints32(arr_1: Vec<i32>, arr_2: Vec<i32>, simd: bool) -> Vec<i64> {
    if simd {
        unsafe { return sum_two_ints32_simd(arr_1, arr_2) }
    }

    let mut result: Vec<i64> = vec![0; arr_1.len()];
    for ((arr_3_val, arr_1_val), arr_2_val) in result.iter_mut().zip(&arr_1).zip(&arr_2) {
        *arr_3_val = (arr_1_val + arr_2_val) as i64;
    }

    result
}
