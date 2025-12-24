__kernel void sum_floats(__global const float* buffer_1, __global const float* buffer_2, __global float* result) {
	int idx = get_global_id(0);
	result[idx] = buffer_1[idx] + buffer_2[idx];
}