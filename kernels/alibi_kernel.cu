#include <cuda_runtime.h>
#include <cmath>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

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
    long long elements_per_head = static_cast<long long>(N) * N;
    int h = static_cast<int>((idx / elements_per_head) % H);
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
torch::Tensor alibi_forward(
    torch::Tensor scores,
    torch::Tensor slopes
) {
    TORCH_CHECK(
        scores.is_cuda(),
        "scores must be a CUDA tensor"
    );

    TORCH_CHECK(
        slopes.is_cuda(),
        "slopes must be a CUDA tensor"
    );

    TORCH_CHECK(
        scores.device() == slopes.device(),
        "scores and slopes must be on the same GPU"
    );

    TORCH_CHECK(
        scores.scalar_type() == torch::kFloat32,
        "scores must use float32"
    );

    TORCH_CHECK(
        slopes.scalar_type() == torch::kFloat32,
        "slopes must use float32"
    );

    TORCH_CHECK(
        scores.dim() == 4,
        "scores must have shape [B, H, N, N]"
    );

    TORCH_CHECK(
        slopes.dim() == 1,
        "slopes must have shape [H]"
    );

    int B = scores.size(0);
    int H = scores.size(1);
    int N = scores.size(2);

    TORCH_CHECK(
        scores.size(3) == N,
        "attention matrices must be square"
    );

    TORCH_CHECK(
        slopes.size(0) == H,
        "there must be one slope per head"
    );

    auto contiguous_scores = scores.contiguous();
    auto contiguous_slopes = slopes.contiguous();
    auto output = torch::empty_like(contiguous_scores);

    const c10::cuda::CUDAGuard device_guard(
        contiguous_scores.device()
    );

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(
            contiguous_scores.get_device()
        );

    launch_alibi_forward(
        contiguous_scores.data_ptr<float>(),
        contiguous_slopes.data_ptr<float>(),
        output.data_ptr<float>(),
        B,
        H,
        N,
        stream
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "forward",
        &alibi_forward,
        "ALiBi bias and causal-mask forward pass"
    );
}

