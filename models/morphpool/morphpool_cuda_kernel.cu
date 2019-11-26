#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <math.h>

#include <vector>

#define CUDA_NUM_THREADS 1024
#define THREADS_PER_BLOCK 32

#define EPS 1e-8

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* a, double b) { return b; }
#endif

namespace {

template <typename scalar_t>
__global__ void morphpool_cuda_forward_kernel(
    const scalar_t* __restrict__ Input,
    const scalar_t* __restrict__ Mask,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ output_idx_x,
    scalar_t* __restrict__ output_idx_y,
    size_t batch_size,
    size_t input_channels,
    size_t height,
    size_t width,
    int num_morph,
    int kernel_size) {

    const int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    const int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    const int mid = kernel_size / 2;
    const int n = index / height / width;
    const int c = n % input_channels;
    const int h = (index / width) % height;
    const int w = index % width;
    const int batch = n / input_channels;

    if (n < batch_size && h < height && w < width) {
        for (int m=0; m<num_morph; m++)
        {
            const int output_index = batch * input_channels * num_morph * height * width
                     + c * num_morph * height * width
                     + m * height * width
                     + h * width
                     + w;
            scalar_t max_val = 0.;
            int max_y = 0;
            int max_x = 0;
            bool first = true;

            for (int i=0; i<kernel_size; i++) {
                const int y = h + i - mid;
                if (y >= 0 && y < height) {
                    for (int j=0; j<kernel_size; j++) {
                        const int x = w + j - mid;
                        if (x >= 0 && x < width) {
                            const int mask_index = m * kernel_size * kernel_size + i * kernel_size + j;
                            const int offset = n * width * height + y * width + x;

                            if (Mask[mask_index] == 1) {
                                if (Input[offset] > max_val || first) {
                                    max_val = Input[offset];
                                    max_y = y;
                                    max_x = x;
                                    first = false;
                                }
                            }
                        }
                    }
                }
            }

            output[output_index] = max_val;
            output_idx_y[output_index] = max_y;
            output_idx_x[output_index] = max_x;
        }
    }
}

template <typename scalar_t>
__global__ void morphpool_cuda_backward_kernel(
    const scalar_t* __restrict__ Grad,
    const scalar_t* __restrict__ Input,
    const scalar_t* __restrict__ Mask,
    const scalar_t* __restrict__ Indices_x,
    const scalar_t* __restrict__ Indices_y,
    const scalar_t* __restrict__ Output_fwd,
    scalar_t* __restrict__ output,
    size_t batch_size,
    size_t input_channels,
    size_t height,
    size_t width,
    int num_morph,
    int kernel_size) {

    const int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    const int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    scalar_t value = 0.;
    const int mid = kernel_size / 2;
    const int n = index / height / width;
    const int c = n % input_channels;
    const int h = (index / width) % height;
    const int w = index % width;
    const int batch = n / input_channels;


    if (n < batch_size && h < height && w < width) {
        for (int m=0; m<num_morph; m++)
        {
            for (int i=0; i<kernel_size; i++) {
                const int y = h + i - mid;
                if (y >= 0 && y < height) {
                    for (int j=0; j<kernel_size; j++) {
                        const int x = w + j - mid;
                        if (x >= 0 && x < width) {
                            const int i_mask =  kernel_size - i - 1;
                            const int j_mask =  kernel_size - j - 1;
                            const int mask_index = m * kernel_size * kernel_size + i_mask * kernel_size + j_mask;

                            const int output_fwd_index = batch * input_channels * num_morph * height * width
                                     + c * num_morph * height * width
                                     + m * height * width
                                     + y * width
                                     + x;

                            const int max_y = Indices_y[output_fwd_index];
                            const int max_x = Indices_x[output_fwd_index];

                            const int grad_index = batch * input_channels * num_morph * height * width
                                     + c * num_morph * height * width
                                     + m * height * width
                                     + max_y * width
                                     + max_x;
                            if (Mask[mask_index] == 1) {
                                if (w == max_x && h == max_y) {
                                    value = value + 1 * Grad[output_fwd_index];
                                }
                            }
                        }
                    }
                }
            }
        }
        output[index] = value;
    }
}



template <typename scalar_t>
__global__ void morphpool3d_cuda_forward_kernel(
    const scalar_t* __restrict__ Input,
    const scalar_t* __restrict__ Mask,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ output_idx_x,
    scalar_t* __restrict__ output_idx_y,
    scalar_t* __restrict__ output_idx_z,
    size_t batch_size,
    size_t input_channels,
    size_t depth,
    size_t height,
    size_t width,
    int num_morph,
    int kernel_size) {

    const int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    const int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    const int mid = kernel_size / 2;
    const int n = index / depth / height / width;
    const int c = n % input_channels;
    const int d = (index / height / width) % depth;
    const int h = (index / width) % height;
    const int w = index % width;
    const int batch = n / input_channels;

    if (n < batch_size && d < depth && h < height && w < width) {
        for (int m=0; m<num_morph; m++)
        {
            const int output_index = batch * input_channels * num_morph * depth * height * width
                     + c * num_morph * depth * height * width
                     + m * depth * height * width
                     + d * height * width
                     + h * width
                     + w;
            scalar_t max_val = 0.;
            int max_z = 0;
            int max_y = 0;
            int max_x = 0;
            bool first = true;

            for (int k=0; k<kernel_size; k++) {
                const int z = d + k - mid;
                if (z >=0 && z < depth) {
                    for (int i=0; i<kernel_size; i++) {
                        const int y = h + i - mid;
                        if (y >= 0 && y < height) {
                            for (int j=0; j<kernel_size; j++) {
                                const int x = w + j - mid;
                                if (x >= 0 && x < width) {
                                    const int mask_index = m * kernel_size * kernel_size * kernel_size + k * kernel_size * kernel_size + i * kernel_size + j;
                                    const int offset = n * depth * width * height + z * height * width + y * width + x;

                                    if (Mask[mask_index] == 1) {
                                        if (Input[offset] > max_val || first) {
                                            max_val = Input[offset];
                                            max_z = z;
                                            max_y = y;
                                            max_x = x;
                                            first = false;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            output[output_index] = max_val;
            output_idx_x[output_index] = max_x;
            output_idx_y[output_index] = max_y;
            output_idx_z[output_index] = max_z;
        }
    }
}

template <typename scalar_t>
__global__ void morphpool3d_cuda_backward_kernel(
    const scalar_t* __restrict__ Grad,
    const scalar_t* __restrict__ Input,
    const scalar_t* __restrict__ Mask,
    const scalar_t* __restrict__ Indices_x,
    const scalar_t* __restrict__ Indices_y,
    const scalar_t* __restrict__ Indices_z,
    const scalar_t* __restrict__ Output_fwd,
    scalar_t* __restrict__ output,
    size_t batch_size,
    size_t input_channels,
    size_t depth,
    size_t height,
    size_t width,
    int num_morph,
    int kernel_size) {

    const int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    const int index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    scalar_t value = 0.;
    const int mid = kernel_size / 2;
    const int n = index / depth / height / width;
    const int c = n % input_channels;
    const int d = (index / height / width) % depth;
    const int h = (index / width) % height;
    const int w = index % width;
    const int batch = n / input_channels;


    if (n < batch_size && d < depth && h < height && w < width) {
        for (int m=0; m<num_morph; m++)
        {
            for (int k=0; k<kernel_size; k++) {
                const int z = d + k - mid;
                if (z >=0 && z < depth) {
                    for (int i=0; i<kernel_size; i++) {
                        const int y = h + i - mid;
                        if (y >= 0 && y < height) {
                            for (int j=0; j<kernel_size; j++) {
                                const int x = w + j - mid;
                                if (x >= 0 && x < width) {
                                    const int k_mask = kernel_size - k - 1;
                                    const int i_mask = kernel_size - i - 1;
                                    const int j_mask = kernel_size - j - 1;
                                    const int mask_index = m * kernel_size * kernel_size * kernel_size + k_mask * kernel_size * kernel_size + i_mask * kernel_size + j_mask;

                                    const int output_fwd_index = batch * input_channels * num_morph * depth * height * width
                                             + c * num_morph * depth * height * width
                                             + m * depth * height * width
                                             + z * height * width
                                             + y * width
                                             + x;
                                    const int max_x = Indices_x[output_fwd_index];
                                    const int max_y = Indices_y[output_fwd_index];
                                    const int max_z = Indices_z[output_fwd_index];

                                    const int grad_index = batch * input_channels * num_morph * depth * height * width
                                             + c * num_morph * depth * height * width
                                             + m * depth * height * width
                                             + max_z * height * width
                                             + max_y * width
                                             + max_x;
                                    if (Mask[mask_index] == 1) {
                                        if (w == max_x && h == max_y && d == max_z) {
                                            value = value + 1 * Grad[output_fwd_index];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        output[index] = value;
    }
}

} // namespace

std::vector<at::Tensor> morphpool_cuda_forward(
    at::Tensor input,
    at::Tensor mask,
    int num_morph,
    int kernel_size) {

    auto output = at::zeros_like(input);
    auto output_idx_x = at::zeros_like(input);
    auto output_idx_y = at::zeros_like(input);
    auto output_idx_z = at::zeros_like(input);

    if (mask.dim() == 3) {
        const auto batch = input.size(0);
        const auto channel = input.size(1);
        const auto height = input.size(2);
        const auto width = input.size(3);

        auto vInput = input.view({-1, height, width});
        const auto batch_size = vInput.size(0);

        output.resize_({batch, channel, num_morph, height, width});
        output_idx_x.resize_({batch, channel, num_morph, height, width});
        output_idx_y.resize_({batch, channel, num_morph, height, width});
        output.fill_(0);
        output_idx_x.fill_(0);
        output_idx_y.fill_(0);

        const int threads = CUDA_NUM_THREADS;
        const dim3 blocks((height * width + threads - 1) / threads, batch_size);

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        AT_DISPATCH_FLOATING_TYPES(input.type(), "morphpool_cuda_forward_cuda", ([&] {
            morphpool_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                vInput.data<scalar_t>(),
                mask.data<scalar_t>(),
                output.data<scalar_t>(),
                output_idx_x.data<scalar_t>(),
                output_idx_y.data<scalar_t>(),
                batch_size,
                channel,
                height,
                width,
                num_morph,
                kernel_size);
        }));

    }
    else {
        if (mask.dim() == 4) {
            const auto batch = input.size(0);
            const auto channel = input.size(1);
            const auto depth = input.size(2);
            const auto height = input.size(3);
            const auto width = input.size(4);

            auto vInput = input.view({-1, depth, height, width});
            const auto batch_size = vInput.size(0);

            output.resize_({batch, channel, num_morph, depth, height, width});
            output_idx_x.resize_({batch, channel, num_morph, depth, height, width});
            output_idx_y.resize_({batch, channel, num_morph, depth, height, width});
            output_idx_z.resize_({batch, channel, num_morph, depth, height, width});
            output.fill_(0);
            output_idx_x.fill_(0);
            output_idx_y.fill_(0);
            output_idx_z.fill_(0);

            const int threads = CUDA_NUM_THREADS;
            const dim3 blocks((depth * height * width + threads - 1) / threads, batch_size);

            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            AT_DISPATCH_FLOATING_TYPES(input.type(), "morphpool3d_cuda_forward_cuda", ([&] {
                morphpool3d_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                    vInput.data<scalar_t>(),
                    mask.data<scalar_t>(),
                    output.data<scalar_t>(),
                    output_idx_x.data<scalar_t>(),
                    output_idx_y.data<scalar_t>(),
                    output_idx_z.data<scalar_t>(),
                    batch_size,
                    channel,
                    depth,
                    height,
                    width,
                    num_morph,
                    kernel_size);
            }));

        }
    }
    return {output, output_idx_x, output_idx_y, output_idx_z};
}

std::vector<at::Tensor> morphpool_cuda_backward(
    at::Tensor grad,
    at::Tensor input,
    at::Tensor mask,
    at::Tensor input_indices_x,
    at::Tensor input_indices_y,
    at::Tensor input_indices_z,
    at::Tensor output_fwd,
    int num_morph,
    int kernel_size) {

    auto output = at::zeros_like(input);

    if (mask.dim() == 3) {
        const auto batch = input.size(0);
        const auto channel = input.size(1);
        const auto height = input.size(2);
        const auto width = input.size(3);

        auto vInput = input.view({-1, height, width});
        const auto batch_size = vInput.size(0);

        const int threads = CUDA_NUM_THREADS;

        const dim3 blocks((height * width + threads - 1) / threads, batch_size);

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        AT_DISPATCH_FLOATING_TYPES(vInput.type(), "morphpool_backward_cuda", ([&] {
            morphpool_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
                grad.data<scalar_t>(),
                vInput.data<scalar_t>(),
                mask.data<scalar_t>(),
                input_indices_x.data<scalar_t>(),
                input_indices_y.data<scalar_t>(),
                output_fwd.data<scalar_t>(),
                output.data<scalar_t>(),
                batch_size,
                channel,
                height,
                width,
                num_morph,
                kernel_size);
        }));

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

    }
    else {
        if (mask.dim() == 4) {
            const auto batch = input.size(0);
            const auto channel = input.size(1);
            const auto depth = input.size(2);
            const auto height = input.size(3);
            const auto width = input.size(4);

            auto vInput = input.view({-1, depth, height, width});
            const auto batch_size = vInput.size(0);

            const int threads = CUDA_NUM_THREADS;

            const dim3 blocks((depth * height * width + threads - 1) / threads, batch_size);

            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            AT_DISPATCH_FLOATING_TYPES(vInput.type(), "morphpool3d_backward_cuda", ([&] {
                morphpool3d_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
                    grad.data<scalar_t>(),
                    vInput.data<scalar_t>(),
                    mask.data<scalar_t>(),
                    input_indices_x.data<scalar_t>(),
                    input_indices_y.data<scalar_t>(),
                    input_indices_z.data<scalar_t>(),
                    output_fwd.data<scalar_t>(),
                    output.data<scalar_t>(),
                    batch_size,
                    channel,
                    depth,
                    height,
                    width,
                    num_morph,
                    kernel_size);
            }));

            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
        }
    }

    return {output};
}