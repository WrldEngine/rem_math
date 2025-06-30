from rmath import rmath as rm

arr = [i for i in range(10_000_000)]
print(rm.sum_two_ints32(arr, arr, simd=True))