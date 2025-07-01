import rem_math as rm
import numpy as np
import pytest
import time

NUM_ITERATIONS = 16_000_000


@pytest.fixture(scope="module")
def large_array():
    return [i for i in range(NUM_ITERATIONS)]


@pytest.fixture(scope="module")
def large_float_array():
    return [float(i) for i in range(NUM_ITERATIONS)]


@pytest.mark.benchmark(
    group="sum_floatsf32",
    min_time=0.1,
    max_time=0.5,
    min_rounds=5,
    timer=time.time,
    disable_gc=True,
    warmup=False,
)
def test_sum_floatsf32(benchmark, large_float_array):
    @benchmark
    def result():
        return rm.sum_two_floats32(large_float_array, large_float_array)

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
def test_sum_floatsf32_simd(benchmark, large_float_array):
    @benchmark
    def result():
        return rm.sum_two_floats32(large_float_array, large_float_array, "simd")

    assert result is not None


@pytest.mark.benchmark(
    group="sum_floatsf32_multithreaded",
    min_time=0.1,
    max_time=0.5,
    min_rounds=5,
    timer=time.time,
    disable_gc=True,
    warmup=False,
)
def test_sum_floatsf32_multithreaded(benchmark, large_float_array):
    @benchmark
    def result():
        return rm.sum_two_floats32(large_float_array, large_float_array, "threading")

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
def test_numpy_arr_sum(benchmark, large_array):
    @benchmark
    def result():
        return np.add(large_array, large_array)

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
def test_sum_ints32(benchmark, large_array):
    @benchmark
    def result():
        return rm.sum_two_ints32(large_array, large_array)

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
def test_sum_ints32_simd(benchmark, large_array):
    @benchmark
    def result():
        return rm.sum_two_ints32(large_array, large_array, simd=True)

    assert result is not None
