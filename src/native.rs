use std::arch::x86_64::*;

#[inline]
pub fn sum_arr_int32(arr: Vec<i32>, simd: bool) -> i64 {
    if simd {
        unsafe {
            let mut sum = _mm256_setzero_si256();
            let chunks = arr.chunks_exact(8);

            let remainder = chunks.remainder();

            for chunk in chunks {
                let ptr = chunk.as_ptr() as *const __m256i;
                let vec = _mm256_loadu_si256(ptr);
                sum = _mm256_add_epi32(sum, vec);
            }

            let mut temp = [0i32; 8];
            _mm256_storeu_si256(temp.as_mut_ptr() as *mut __m256i, sum);
            let mut total = temp.iter().sum::<i32>();

            total += remainder.iter().sum::<i32>();

            return total as i64;
        }
    }

    let sum: i32 = arr.iter().sum();

    sum as i64
}
