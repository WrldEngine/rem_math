# Python math library written in Rust

Work in progress

## Examples
Sum of two 32-bit integer array
```py
import rmath as rm

array = [i for i in range(100_000_000)]
sum_two_i32_result = rm.sum_two_ints32(array, array, simd=True)

print(sum_two_i32_result)
```

## Benchmarks (Python)
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

## Benchmarks (Rust)
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

## Roadmap

- Add GPU-accelerated operations for improved performance.
- Implement own custom type objects for best performance from ecosystem.
- Expand mathematical functionality with additional features and algorithms.

Stay tuned for updates as the library evolves!