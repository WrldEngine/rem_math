import rem_math as rm
import numpy as np
import pytest
import time

NUM_ITERATIONS = 16_000_000


@pytest.fixture(scope="module")
def large_array():
    return [i for i in range(NUM_ITERATIONS)]


@pytest.mark.benchmark(
    group="mul_ints32",
    min_time=0.1,
    max_time=0.5,
    min_rounds=5,
    timer=time.time,
    disable_gc=True,
    warmup=False,
)
def test_mul_ints32_scalar(benchmark, large_array):

    @benchmark
    def result():
        return rm.multiply_two_ints32(large_array, large_array)

    assert result is not None


@pytest.mark.benchmark(
    group="mul_ints32_simd",
    min_time=0.1,
    max_time=0.5,
    min_rounds=5,
    timer=time.time,
    disable_gc=True,
    warmup=False,
)
def test_mul_ints32_simd(benchmark, large_array):

    @benchmark
    def result():
        return rm.multiply_two_ints32(large_array, large_array, "simd")

    assert result is not None


@pytest.mark.benchmark(
    group="mul_ints32_threading",
    min_time=0.1,
    max_time=0.5,
    min_rounds=5,
    timer=time.time,
    disable_gc=True,
    warmup=False,
)
def test_mul_ints32_threading(benchmark, large_array):

    @benchmark
    def result():
        return rm.multiply_two_ints32(large_array, large_array, "threading")

    assert result is not None


@pytest.mark.benchmark(
    group="mul_ints32_numpy",
    min_time=0.1,
    max_time=0.5,
    min_rounds=5,
    timer=time.time,
    disable_gc=True,
    warmup=False,
)
def test_mul_ints32_numpy(benchmark, large_array):

    @benchmark
    def result():
        return np.multiply(large_array, large_array)

    assert result is not None
