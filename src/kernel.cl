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

__kernel void dot_f(
    __global const float* a,
    __global const float* b,
    __global float* partial_sums,
    __local float* local_sums
) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_id = get_group_id(0);
    int group_size = get_local_size(0);

    float val = a[gid] * b[gid];
    local_sums[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_sums[lid] += local_sums[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        partial_sums[group_id] = local_sums[0];
    }
}