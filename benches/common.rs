use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use rmath::native::sum_arr_int32;
use rmath::native::sum_two_floats32;

fn sum_arr_int32_benchmark(c: &mut Criterion) {
    let arr = black_box((1..1_000_000).collect::<Vec<i32>>());
    c.bench_function("Array accumulation", |b| {
        b.iter_batched(
            || arr.clone(),
            |input| sum_arr_int32(input, false),
            BatchSize::SmallInput,
        )
    });
}

fn sum_arr_int32_with_simd_benchmark(c: &mut Criterion) {
    let arr = black_box((1..1_000_000).collect::<Vec<i32>>());
    c.bench_function("Array accumulation with SIMD instructions", |b| {
        b.iter_batched(
            || arr.clone(),
            |input| sum_arr_int32(input, true),
            BatchSize::SmallInput,
        )
    });
}

fn sum_two_floats32_benchmark(c: &mut Criterion) {
    let arr = black_box(vec![1.0; 800]);
    c.bench_function("Array accumulation of two float arrays", |b| {
        b.iter(|| sum_two_floats32(arr.clone(), arr.clone(), false))
    });
}

fn sum_two_floats32_with_simd_benchmark(c: &mut Criterion) {
    let arr = black_box(vec![1.0; 800]);
    c.bench_function("Array accumulation of two float arrays with SIMD", |b| {
        b.iter(|| sum_two_floats32(arr.clone(), arr.clone(), true))
    });
}

criterion_group!(
    benches,
    sum_arr_int32_benchmark,
    sum_arr_int32_with_simd_benchmark,
    sum_two_floats32_benchmark,
    sum_two_floats32_with_simd_benchmark,
);
criterion_main!(benches);
