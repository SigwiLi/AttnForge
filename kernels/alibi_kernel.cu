#include <cuda_runtime.h>
#include <cmath>


// Kernel code
__global__ void alibi_forward_kernel(
    const float* scores, 
    const float* slopes, 
    float* output, 
    int B, // B = batch
    int H, // H = head
    int N  // N = sequence length
) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long total_elements = static_cast<long long>(B) * H * N * N;

    // bounds check
    if (idx >= total_elements){
        return;
    }

    int j = idx % N;
    int i = (idx / N) % N;
    int h = (idx/(static cast<long long>(N) * N)) % H;
    int b = idx / (H * N * N);

    if (j > i){
        output[idx] = -INFINITY;
    } else {
        int distance = i - j;
        output[idx] = scores[idx] - slopes[h] * distance;
    }
}

// Launcher
void launch_alibi_forward(
    const float* scores,
    const float* slopes,
    float* output,
    int B,
    int H,
    int N, 
    cudaStream_t stream
){
    int total_elements = B * H * N * N;

    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    alibi_forward_kernel <<<num_blocks, threads_per_block, 0, stream>>>(scores, slopes, output, B, H, N);
}

