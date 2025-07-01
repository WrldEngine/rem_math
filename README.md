# Python math library written in Rust

Work in progress

## Installation
`pip install rem-math`

## Examples
Sum of two 32-bit integer array
```py
from rem_math import rem_math as rm
import numpy as np

array = [i for i in range(100_000_000)]
np_array = np.array([i for i in range(100_000_000)], dtype=np.int32)

sum_two_i32_result = rm.sum_two_ints32(array, array, simd=True)
sum_of_array = rm.sum_arr_int32(array)
sum_of_np_array = rm.sum_nparr_int32(np_array)

print(sum_two_i32_result)
```

## Benchmarks (Python)

### Accamulate array values of integer32
  ```
  --------------------------------------- benchmark 'arr_i32': 1 tests ---------------------------------------
  Name (time in ms)        Min     Max    Mean  StdDev  Median     IQR  Outliers       OPS  Rounds  Iterations
  ------------------------------------------------------------------------------------------------------------
  test_sum_arr_i32      4.2611  4.4810  4.3450  0.0814  4.3310  0.0625       2;1  230.1495       5         100
  ------------------------------------------------------------------------------------------------------------

  ---------------------------------------- benchmark 'arr_i32_simd': 1 tests -----------------------------------------
  Name (time in us)            Min     Max    Mean  StdDev  Median     IQR  Outliers  OPS (Kops/s)  Rounds  Iterations
  --------------------------------------------------------------------------------------------------------------------
  test_sum_arr_i32_simd     1.1802  1.2003  1.1903  0.0071  1.1903  0.0051       2;0      840.1472       5      100000
  --------------------------------------------------------------------------------------------------------------------

  ---------------------------------------- benchmark 'numpy': 1 tests ----------------------------------------
  Name (time in ms)        Min     Max    Mean  StdDev  Median     IQR  Outliers       OPS  Rounds  Iterations
  ------------------------------------------------------------------------------------------------------------
  test_numpy_sum        7.6939  7.8477  7.7710  0.0544  7.7710  0.0387       2;0  128.6840       5          13
  ------------------------------------------------------------------------------------------------------------

  Legend:
    Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
    OPS: Operations Per Second, computed as 1 / Mean
  ```
### Accumulate values of indexes of two arrays (float32 & int32)
    ```
    -------------------------------------- benchmark 'numpy_arr_sum': 1 tests --------------------------------------
    Name (time in ms)          Min      Max     Mean  StdDev   Median     IQR  Outliers      OPS  Rounds  Iterations
    ----------------------------------------------------------------------------------------------------------------
    test_numpy_arr_sum     70.3064  78.1331  75.1148  4.1352  78.1298  7.3991       2;0  13.3129       5           2
    ----------------------------------------------------------------------------------------------------------------
    
    -------------------------------------- benchmark 'sum_floatsf32': 1 tests --------------------------------------
    Name (time in ms)          Min      Max     Mean  StdDev   Median     IQR  Outliers      OPS  Rounds  Iterations
    ----------------------------------------------------------------------------------------------------------------
    test_sum_floatsf32     54.6912  54.8023  54.7604  0.0562  54.8004  0.0981       1;0  18.2614       5          10
    ----------------------------------------------------------------------------------------------------------------
    
    -------------------------------------- benchmark 'sum_floatsf32_simd': 1 tests --------------------------------------
    Name (time in ms)               Min      Max     Mean  StdDev   Median     IQR  Outliers      OPS  Rounds  Iterations
    ---------------------------------------------------------------------------------------------------------------------
    test_sum_floatsf32_simd     52.0826  52.5085  52.1729  0.1877  52.0914  0.1119       1;1  19.1670       5           3
    ---------------------------------------------------------------------------------------------------------------------
    
    --------------------------------------- benchmark 'sum_ints32': 1 tests ---------------------------------------
    Name (time in ms)         Min      Max     Mean  StdDev   Median     IQR  Outliers      OPS  Rounds  Iterations
    ---------------------------------------------------------------------------------------------------------------
    test_sum_ints32       32.8143  40.4775  36.1822  3.0797  34.9533  4.6386       2;0  27.6379       5          10
    ---------------------------------------------------------------------------------------------------------------
    
    -------------------------------------- benchmark 'sum_ints32_simd': 1 tests --------------------------------------
    Name (time in ms)            Min      Max     Mean  StdDev   Median     IQR  Outliers      OPS  Rounds  Iterations
    ------------------------------------------------------------------------------------------------------------------
    test_sum_ints32_simd     32.8149  34.9717  33.8741  0.9905  34.3772  1.7000       3;0  29.5211       5          10
    ------------------------------------------------------------------------------------------------------------------
    ```

### Compare with NumPy
    -------------------------------------- benchmark 'numpy_sum': 1 tests --------------------------------------
    Name (time in ms)        Min     Max    Mean  StdDev  Median     IQR  Outliers       OPS  Rounds  Iterations
    ------------------------------------------------------------------------------------------------------------
    test_numpy_sum        7.8130  8.9292  8.4827  0.6113  8.9290  1.1160       2;0  117.8870       5          14
    ------------------------------------------------------------------------------------------------------------

    --------------------------------------- benchmark 'rm_sum': 1 tests ----------------------------------------
    Name (time in ms)        Min     Max    Mean  StdDev  Median     IQR  Outliers       OPS  Rounds  Iterations
    ------------------------------------------------------------------------------------------------------------
    test_rm_sum           1.7189  1.8860  1.7523  0.0747  1.7189  0.0419       1;1  570.6719       5         100
    ------------------------------------------------------------------------------------------------------------

## Benchmarks (Rust)

### Accamulate array values of integer32
  ```
  Array accumulation      time:   [15.509 µs 15.532 µs 15.575 µs]
  Found 11 outliers among 100 measurements (11.00%)
    1 (1.00%) low mild
    3 (3.00%) high mild
    7 (7.00%) high severe

  Array accumulation with SIMD instructions
                          time:   [77.083 ns 77.499 ns 78.264 ns]
  Found 9 outliers among 100 measurements (9.00%)
    2 (2.00%) high mild
    7 (7.00%) high severe
  ```
### Accumulate values of indexes of two arrays (float32 & int32)
```angular2html
Array accumulation      time:   [4.9527 ms 4.9757 ms 5.0013 ms]
Found 11 outliers among 100 measurements (11.00%)
  1 (1.00%) high mild
  10 (10.00%) high severe

Array accumulation with SIMD instructions
                        time:   [4.8752 ms 4.8976 ms 4.9250 ms]
Found 8 outliers among 100 measurements (8.00%)
  2 (2.00%) high mild
  6 (6.00%) high severe

Benchmarking Array accumulation of two float arrays: Warming up for 3.0000 s
Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 5.6s, or reduce sample count to 80.
Array accumulation of two float arrays
                        time:   [53.260 ms 53.448 ms 53.653 ms]
Found 13 outliers among 100 measurements (13.00%)
  9 (9.00%) high mild
  4 (4.00%) high severe

Benchmarking Array accumulation of two float arrays with SIMD: Warming up for 3.0000 s
Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 5.8s, or reduce sample count to 80.
Array accumulation of two float arrays with SIMD
                        time:   [55.241 ms 55.470 ms 55.718 ms]
Found 10 outliers among 100 measurements (10.00%)
  8 (8.00%) high mild
  2 (2.00%) high severe

Benchmarking Array accumulation of two integer arrays: Warming up for 3.0000 s
Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 5.6s, or reduce sample count to 80.
Array accumulation of two integer arrays
                        time:   [53.543 ms 53.783 ms 54.043 ms]
Found 10 outliers among 100 measurements (10.00%)
  7 (7.00%) high mild
  3 (3.00%) high severe

Benchmarking Array accumulation of two integer arrays with SIMD: Warming up for 3.0000 s
Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 5.8s, or reduce sample count to 80.
Array accumulation of two integer arrays with SIMD
                        time:   [55.783 ms 56.053 ms 56.360 ms]
Found 15 outliers among 100 measurements (15.00%)
  7 (7.00%) high mild
  8 (8.00%) high severe
```

## Roadmap

- Add GPU-accelerated operations for improved performance.
- Implement own custom type objects for best performance from ecosystem.
- Expand mathematical functionality with additional features and algorithms.

Stay tuned for updates as the library evolves!