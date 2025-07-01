import rem_math as rm
import numpy as np
import pytest
import time

NUM_ITERATIONS = 10_000_000


@pytest.mark.benchmark(
    group="arr_i32",
    min_time=0.1,
    max_time=0.5,
    min_rounds=5,
    timer=time.time,
    disable_gc=True,
    warmup=False,
)
def test_sum_arr_i32(benchmark):
    @benchmark
    def result():
        array = [i for i in range(NUM_ITERATIONS)]
        return rm.sum_arr_int32(array)

    assert result is not None


@pytest.mark.benchmark(
    group="arr_i32_simd",
    min_time=0.1,
    max_time=0.5,
    min_rounds=5,
    timer=time.time,
    disable_gc=True,
    warmup=False,
)
def test_sum_arr_i32_simd(benchmark):
    @benchmark
    def result():
        array = [i for i in range(NUM_ITERATIONS)]
        return rm.sum_arr_int32(array, simd=True)

    assert result is not None


@pytest.mark.benchmark(
    group="numpy",
    min_time=0.1,
    max_time=0.5,
    min_rounds=5,
    timer=time.time,
    disable_gc=True,
    warmup=False,
)
def test_numpy_sum(benchmark):
    @benchmark
    def result():
        array = [i for i in range(NUM_ITERATIONS)]
        return np.sum(array)

    assert result is not None
