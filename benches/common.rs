use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use rem_math::native::multiply_two_ints32;
use rem_math::native::sum_arr_int32;
use rem_math::native::sum_two_floats32;
use rem_math::native::sum_two_ints32;

pub const NUM_ITERATIONS: usize = 16_000_000;

fn sum_arr_int32_benchmark(c: &mut Criterion) {
    let arr = black_box((1..NUM_ITERATIONS as i32).collect::<Vec<i32>>());
    c.bench_function("Array accumulation", |b| {
        b.iter_batched(
            || arr.clone(),
            |input| sum_arr_int32(&input, false),
            BatchSize::SmallInput,
        )
    });
}

fn sum_arr_int32_with_simd_benchmark(c: &mut Criterion) {
    let arr = black_box((1..NUM_ITERATIONS as i32).collect::<Vec<i32>>());
    c.bench_function("Array accumulation with SIMD instructions", |b| {
        b.iter_batched(
            || arr.clone(),
            |input| sum_arr_int32(&input, true),
            BatchSize::SmallInput,
        )
    });
}

fn sum_two_floats32_benchmark(c: &mut Criterion) {
    let arr = black_box(vec![1.0; NUM_ITERATIONS]);
    c.bench_function("Array accumulation of two float arrays", |b| {
        b.iter(|| sum_two_floats32(&arr, &arr, ""))
    });
}

fn sum_two_floats32_with_simd_benchmark(c: &mut Criterion) {
    let arr = black_box(vec![1.0; NUM_ITERATIONS]);
    c.bench_function("Array accumulation of two float arrays with SIMD", |b| {
        b.iter(|| sum_two_floats32(&arr, &arr, "simd"))
    });
}

fn sum_two_floats32_with_mthreaded_benchmark(c: &mut Criterion) {
    let arr = black_box(vec![1.0; NUM_ITERATIONS]);
    c.bench_function(
        "Array accumulation of two float arrays with multithreading",
        |b| b.iter(|| sum_two_floats32(&arr, &arr, "threading")),
    );
}

fn sum_two_ints32_with_benchmark(c: &mut Criterion) {
    let arr = black_box(vec![1; NUM_ITERATIONS]);
    c.bench_function("Array accumulation of two integer arrays", |b| {
        b.iter(|| sum_two_ints32(&arr, &arr, false))
    });
}

fn sum_two_ints32_with_simd_benchmark(c: &mut Criterion) {
    let arr = black_box(vec![1; NUM_ITERATIONS]);
    c.bench_function("Array accumulation of two integer arrays with SIMD", |b| {
        b.iter(|| sum_two_ints32(&arr, &arr, true))
    });
}

fn mul_two_ints32_benchmark(c: &mut Criterion) {
    let arr = black_box(vec![1; NUM_ITERATIONS]);
    c.bench_function("Array multiply of two integer arrays", |b| {
        b.iter(|| multiply_two_ints32(&arr, &arr, ""))
    });
}

fn mul_two_ints32_with_simd_benchmark(c: &mut Criterion) {
    let arr = black_box(vec![1; NUM_ITERATIONS]);
    c.bench_function("Array multiply of two integer arrays with simd", |b| {
        b.iter(|| multiply_two_ints32(&arr, &arr, "simd"))
    });
}

fn mul_two_ints32_with_mthreaded_benchmark(c: &mut Criterion) {
    let arr = black_box(vec![1; NUM_ITERATIONS]);
    c.bench_function(
        "Array multiply of two integer arrays with multithreading",
        |b| b.iter(|| multiply_two_ints32(&arr, &arr, "threading")),
    );
}

criterion_group!(
    benches,
    sum_arr_int32_benchmark,
    sum_arr_int32_with_simd_benchmark,
    sum_two_floats32_benchmark,
    sum_two_floats32_with_simd_benchmark,
    sum_two_floats32_with_mthreaded_benchmark,
    sum_two_ints32_with_benchmark,
    sum_two_ints32_with_simd_benchmark,
    mul_two_ints32_benchmark,
    mul_two_ints32_with_simd_benchmark,
    mul_two_ints32_with_mthreaded_benchmark,
);
criterion_main!(benches);
