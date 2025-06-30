import rem_math as rm
import numpy as np
import pytest
import time

NUM_ITERATIONS = 10_000_000

@pytest.mark.benchmark(
    group="sum_floatsf32",
    min_time=0.1,
    max_time=0.5,
    min_rounds=5,
    timer=time.time,
    disable_gc=True,
    warmup=False,
)
def test_sum_floatsf32(benchmark):
    @benchmark
    def result():
        array = [i for i in range(NUM_ITERATIONS)]
        sum_two_f32_result = rm.sum_two_floats32(array, array)

        return sum_two_f32_result

    assert result is not None


@pytest.mark.benchmark(
    group="sum_floatsf32_simd",
    min_time=0.1,
    max_time=0.5,
    min_rounds=5,
    timer=time.time,
    disable_gc=True,
    warmup=False,
)
def test_sum_floatsf32_simd(benchmark):
    @benchmark
    def result():
        array = [i for i in range(NUM_ITERATIONS)]
        sum_two_f32_simd_result = rm.sum_two_floats32(array, array, simd=True)

        return sum_two_f32_simd_result

    assert result is not None


@pytest.mark.benchmark(
    group="numpy_arr_sum",
    min_time=0.1,
    max_time=0.5,
    min_rounds=5,
    timer=time.time,
    disable_gc=True,
    warmup=False,
)
def test_numpy_arr_sum(benchmark):
    @benchmark
    def result():
        array = [i for i in range(NUM_ITERATIONS)]
        np_result = np.add(array, array)

        return np_result

    assert result is not None


@pytest.mark.benchmark(
    group="sum_ints32",
    min_time=0.1,
    max_time=0.5,
    min_rounds=5,
    timer=time.time,
    disable_gc=True,
    warmup=False,
)
def test_sum_ints32(benchmark):
    @benchmark
    def result():
        array = [i for i in range(NUM_ITERATIONS)]
        sum_two_i32_result = rm.sum_two_ints32(array, array)

        return sum_two_i32_result

    assert result is not None

@pytest.mark.benchmark(
    group="sum_ints32_simd",
    min_time=0.1,
    max_time=0.5,
    min_rounds=5,
    timer=time.time,
    disable_gc=True,
    warmup=False,
)
def test_sum_ints32_simd(benchmark):
    @benchmark
    def result():
        array = [i for i in range(NUM_ITERATIONS)]
        sum_two_i32_result = rm.sum_two_ints32(array, array, simd=True)

        return sum_two_i32_result

    assert result is not None
