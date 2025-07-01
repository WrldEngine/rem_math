import rem_math as rm
import numpy as np
import pytest
import time

NUM_ITERATIONS = 10_000_000


@pytest.fixture(scope="module")
def large_array():
    return np.array([i for i in range(NUM_ITERATIONS)], dtype=np.int32)


@pytest.mark.benchmark(
    group="numpy_sum",
    min_time=0.1,
    max_time=0.5,
    min_rounds=5,
    timer=time.time,
    disable_gc=True,
    warmup=False,
)
def test_numpy_sum(benchmark, large_array):
    @benchmark
    def result():
        return np.sum(large_array)

    assert result is not None


@pytest.mark.benchmark(
    group="rm_sum",
    min_time=0.1,
    max_time=0.5,
    min_rounds=5,
    timer=time.time,
    disable_gc=True,
    warmup=False,
)
def test_rm_sum(benchmark, large_array):
    @benchmark
    def result():
        return rm.sum_nparr_int32(large_array)

    assert result is not None


@pytest.mark.benchmark(
    group="numpy_mul",
    min_time=0.1,
    max_time=0.5,
    min_rounds=5,
    timer=time.time,
    disable_gc=True,
    warmup=False,
)
def test_np_mul(benchmark, large_array):
    @benchmark
    def result():
        return np.multiply(large_array, large_array)

    assert result is not None


@pytest.mark.benchmark(
    group="rm_mul",
    min_time=0.1,
    max_time=0.5,
    min_rounds=5,
    timer=time.time,
    disable_gc=True,
    warmup=False,
)
def test_rm_mul(benchmark, large_array):
    @benchmark
    def result():
        return rm.multiply_two_nparr_ints32(large_array, large_array, method="simd")

    assert result is not None


@pytest.mark.benchmark(
    group="numpy_sum_two",
    min_time=0.1,
    max_time=0.5,
    min_rounds=5,
    timer=time.time,
    disable_gc=True,
    warmup=False,
)
def test_numpy_sum_two(benchmark, large_array):
    @benchmark
    def result():
        return np.add(large_array, large_array)

    assert result is not None


@pytest.mark.benchmark(
    group="rm_sum_two(i32)",
    min_time=0.1,
    max_time=0.5,
    min_rounds=5,
    timer=time.time,
    disable_gc=True,
    warmup=False,
)
def test_rm_sum_two_ints32(benchmark, large_array):
    @benchmark
    def result():
        return rm.sum_two_nparr_ints32(large_array, large_array)

    assert result is not None
