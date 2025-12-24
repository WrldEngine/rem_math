import time

import numpy as np
import pytest
import rem_math as rm

NUM_ITERATIONS = 100_000_000


@pytest.fixture(scope="module")
def large_array():
    return np.array([i for i in range(NUM_ITERATIONS)], dtype=np.int32)


@pytest.fixture(scope="module")
def large_naive_array():
    return [i for i in range(NUM_ITERATIONS)]


@pytest.fixture(scope="module")
def large_float_array():
    return np.array([i for i in range(NUM_ITERATIONS)], dtype=np.float32)


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


""" @pytest.mark.benchmark(
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
        return rm.multiply_two_nparr_ints32(large_array, large_array, "threading")

    assert result is not None """


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
        return rm.sum_two_nparr_ints32(large_array, large_array, "threading")

    assert result is not None


@pytest.mark.benchmark(
    group="rm_sum_two(gpu)",
    min_time=0.1,
    max_time=0.5,
    min_rounds=5,
    timer=time.time,
    disable_gc=True,
    warmup=False,
)
def test_rm_sum_two_ints32_gpu(benchmark, large_array):
    @benchmark
    def result():
        return rm.sum_two_nparr_ints32(large_array, large_array, "gpu")

    assert result is not None


@pytest.mark.benchmark(
    group="numpy_dot",
    min_time=0.1,
    max_time=0.5,
    min_rounds=5,
    timer=time.time,
    disable_gc=True,
    warmup=False,
)
def test_numpy_dot(benchmark, large_float_array):
    @benchmark
    def result():
        return np.dot(large_float_array, large_float_array)

    assert result is not None


@pytest.mark.benchmark(
    group="rm_dot(f32)",
    min_time=0.1,
    max_time=0.5,
    min_rounds=5,
    timer=time.time,
    disable_gc=True,
    warmup=False,
)
def test_rm_dot(benchmark, large_float_array):
    @benchmark
    def result():
        return rm.dot_two_nparr_floats32(large_float_array, large_float_array, "gpu")

    assert result is not None
