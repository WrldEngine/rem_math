__kernel void sum_ints(__global const int* buffer_1, __global const int* buffer_2, __global long* result) {
	int idx = get_global_id(0);
	result[idx] = (long)buffer_1[idx] + (long)buffer_2[idx];
}