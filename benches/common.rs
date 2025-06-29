use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rmath::native::sum_arr_int32;

fn sum_arr_i32_benchmark(c: &mut Criterion) {
    let arr = black_box((1..100_000).collect::<Vec<i32>>());
    c.bench_function("Array accumulation", |b| {
        b.iter(|| sum_arr_int32(arr.clone(), false))
    });
}

fn sum_arr_i32_benchmark_with_simd(c: &mut Criterion) {
    let arr = black_box((1..15).collect::<Vec<i32>>());
    c.bench_function("Array accumulation with SIMD instructions", |b| {
        b.iter(|| sum_arr_int32(arr.clone(), true))
    });
}

criterion_group!(
    benches,
    sum_arr_i32_benchmark,
    sum_arr_i32_benchmark_with_simd
);
criterion_main!(benches);
