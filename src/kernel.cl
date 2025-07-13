__kernel void add_f(__global const float* buffer_1, __global const float* buffer_2, __global float* result) {
	int idx = get_global_id(0);
	result[idx] = buffer_1[idx] + buffer_2[idx];
}

__kernel void add_i(__global const int* buffer_1, __global const int* buffer_2, __global long* result) {
	int idx = get_global_id(0);
	result[idx] = (long)buffer_1[idx] + (long)buffer_2[idx];
}

__kernel void mul_i(__global const int* buffer_1, __global const int* buffer_2, __global long* result) {
	int idx = get_global_id(0);
	result[idx] = (long)buffer_1[idx] + (long)buffer_2[idx];
}

__kernel void dot_f(__global const float4* buffer_1, __global const float4* buffer_2, __global float* result) {
	int idx = get_global_id(0);
	result[idx] = dot(buffer_1[idx], buffer_2[idx]);
}
