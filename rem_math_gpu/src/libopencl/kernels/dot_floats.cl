#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable

__kernel void dot_f(
    __global const float* a,
    __global const float* b,
    __global float* result,
	__local float* shared_arr,
	const int arr_sz
) {
    int gid = get_global_id(0);
	int lid = get_local_id(0);

	shared_arr[lid] = a[gid] * b[gid];
	barrier(CLK_LOCAL_MEM_FENCE);

	float sum = 0.0f;
	if (lid == 0) {
		for (int i = 0; i < arr_sz; i++) {
			sum += shared_arr[i];
		};
		result[0] += sum;
	};
}