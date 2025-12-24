__kernel void mul_mat_floats(
    const int M, const int N, const int K,
    const __global float* A,
    const __global float* B,
    __global float* C)
{
    const int global_row = get_global_id(0);
    const int global_col = get_global_id(1);

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += A[k * M + global_row] * B[global_col * K + k];
    }

    C[global_col * M + global_row] = acc;
}