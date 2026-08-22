#include <torch/extension.h>
#include <ATen/record_function.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/linear_combination.h>

#include <cstdint>
#include <limits>
#include <type_traits>

// ============================================================================
// CUTLASS GEMM configuration
//
// A: row-major [M, K]
// B: column-major [K, N]
//
// PyTorch stores a Linear weight as contiguous row-major [N, K].
// That same memory can be interpreted as column-major [K, N], which gives
// the desired multiplication:
//
//     input [M, K] @ weight.T [K, N]
//
// without physically transposing the weight.
// ============================================================================

using CutlassElement = cutlass::half_t;
using AccumulatorElement = float;
using ComputeElement = float;

static constexpr int kElementsPerAccess =
    128 / cutlass::sizeof_bits<CutlassElement>::value;

// FP16 Tensor Core GEMM targeting Ampere SM80.
//
// The explicit tile sizes make this a complete kernel configuration rather
// than relying on every default selected by device::Gemm.
using LinearCombination = cutlass::epilogue::thread::LinearCombination<
    CutlassElement,
    kElementsPerAccess,
    AccumulatorElement,
    ComputeElement>;

using LinearGemm = cutlass::gemm::device::Gemm<
    CutlassElement,                         // Element A
    cutlass::layout::RowMajor,              // Layout A
    CutlassElement,                         // Element B
    cutlass::layout::ColumnMajor,           // Layout B
    CutlassElement,                         // Element C/D
    cutlass::layout::RowMajor,              // Layout C/D
    AccumulatorElement,                     // Accumulator type
    cutlass::arch::OpClassTensorOp,         // Tensor Cores
    cutlass::arch::Sm80,                    // Minimum architecture
    cutlass::gemm::GemmShape<128, 128, 32>, // Threadblock tile
    cutlass::gemm::GemmShape<64, 64, 32>,   // Warp tile
    cutlass::gemm::GemmShape<16, 8, 16>,    // MMA instruction
    LinearCombination,                      // Epilogue
    cutlass::gemm::threadblock::
        GemmIdentityThreadblockSwizzle<>,
    3, // Pipeline stages
    8, // Alignment A, in elements
    8  // Alignment B, in elements
    >;

// ============================================================================
// CUTLASS status helper
// ============================================================================

inline void check_cutlass_status(
    cutlass::Status status,
    const char *operation)
{
    TORCH_CHECK(
        status == cutlass::Status::kSuccess,
        operation,
        " failed: ",
        cutlassGetStatusString(status));
}

// ============================================================================
// CUTLASS linear launch with optional fused column bias
//
// Computes:
//
//     output[M, N] = input[M, K] @ weight[N, K].T + (bias ? bias[N] : 0)
//
// Bias is fused directly into the GEMM epilogue via C tensor with row stride 0.
// ============================================================================

cutlass::Status launch_linear_cutlass(
    const at::Half *input,
    const at::Half *weight,
    const at::Half *bias,
    at::Half *output,
    int M,
    int N,
    int K,
    cudaStream_t stream)
{
    const auto *ptr_A =
        reinterpret_cast<const CutlassElement *>(input);

    const auto *ptr_B =
        reinterpret_cast<const CutlassElement *>(weight);

    auto *ptr_D =
        reinterpret_cast<CutlassElement *>(output);

    // If bias is provided, C is the broadcast bias vector [N] with row stride 0.
    // Otherwise, ptr_D is passed with beta = 0.0f to avoid reading C.
    const CutlassElement *ptr_C =
        (bias != nullptr) ? reinterpret_cast<const CutlassElement *>(bias) : ptr_D;

    // A is row-major [M, K].
    const int64_t lda = K;

    // PyTorch weight is row-major [N, K], reinterpreted as column-major
    // [K, N]. Its leading dimension is K.
    const int64_t ldb = K;

    // C is broadcast column bias (ldc = 0) if bias != nullptr, else ldc = N.
    const int64_t ldc = (bias != nullptr) ? 0 : N;
    const int64_t ldd = N;

    typename LinearGemm::Arguments arguments(
        cutlass::gemm::GemmCoord(M, N, K),
        typename LinearGemm::TensorRefA(ptr_A, lda),
        typename LinearGemm::TensorRefB(ptr_B, ldb),
        typename LinearGemm::TensorRefC(ptr_C, ldc),
        typename LinearGemm::TensorRefD(ptr_D, ldd),
        typename LinearGemm::EpilogueOutputOp::Params(
            ComputeElement(1.0f),
            bias != nullptr ? ComputeElement(1.0f) : ComputeElement(0.0f)));

    LinearGemm gemm;

    cutlass::Status status = gemm.can_implement(arguments);

    if (status != cutlass::Status::kSuccess)
    {
        return status;
    }

    const size_t workspace_size =
        LinearGemm::get_workspace_size(arguments);

    // This non-split-K kernel should not require workspace.
    if (workspace_size != 0)
    {
        return cutlass::Status::kErrorWorkspaceNull;
    }

    status = gemm.initialize(
        arguments,
        nullptr,
        stream);

    if (status != cutlass::Status::kSuccess)
    {
        return status;
    }

    return gemm.run(stream);
}

// ============================================================================
// Convolutional gate kernel
//
// projected_input layout:
//     [B, T, 3D]
//
// conv_weight_kd layout:
//     [K, D] (transposed for contiguous warp loads along D)
//
// For each [b, t, d]:
//
// Linear 1 bias is already fused into the input-projection GEMM epilogue, so:
//
//     b_gate = projected_input[b, t, d]
//     c_gate = projected_input[b, t, D + d]
//     x_tilde = projected_input[b, t, 2D + d]
//
//     y[b, t, d] = b_gate * x_tilde
//
//     z[b, t, d] = conv_bias[d] + sum_k conv_weight_kd[k, d] * y[b, t-k, d]
//
//     gate_output[b, t, d] = c_gate * z[b, t, d]
//
// 2D Grid launch:
//     grid.x = ceil_div(model_dim, 256)
//     grid.y = batch_size * num_tokens
// ============================================================================

template <typename scalar_t, int K>
__global__ void convolution_gate_kernel(
    const scalar_t *__restrict__ projected_input,
    const scalar_t *__restrict__ conv_weight_kd,
    const scalar_t *__restrict__ conv_bias,
    scalar_t *__restrict__ gate_output,
    int num_tokens,
    int model_dim)
{
    const int d = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int bt = static_cast<int>(blockIdx.y);
    if (d >= model_dim)
    {
        return;
    }

    const int t = bt % num_tokens;
    const int64_t projected_dim = static_cast<int64_t>(3) * model_dim;
    const int64_t current_base = static_cast<int64_t>(bt) * projected_dim;
    const int64_t output_idx = static_cast<int64_t>(bt) * model_dim + d;

    const float c_gate =
        static_cast<float>(projected_input[current_base + model_dim + d]);
    float convolution = static_cast<float>(conv_bias[d]);

    // k=0 pointers. Each subsequent tap moves one token backward in projected
    // input and one row forward in the prepacked [K, D] convolution weight.
    const scalar_t *source = projected_input + current_base + d;
    const scalar_t *weight_ptr = conv_weight_kd + d;

    if (t >= K - 1)
    {
// Interior path: every tap is valid, so the unrolled loop has no
// per-tap causal-boundary predicate.
#pragma unroll
        for (int k = 0; k < K; ++k)
        {
            const float b_gate = static_cast<float>(source[0]);
            const float x_tilde =
                static_cast<float>(source[2LL * model_dim]);
            const float weight = static_cast<float>(*weight_ptr);

            convolution = fmaf(weight, b_gate * x_tilde, convolution);

            // Avoid forming a pointer before the allocation after the last tap.
            if (k + 1 < K)
            {
                source -= projected_dim;
                weight_ptr += model_dim;
            }
        }
    }
    else
    {
// Boundary path: only the first t+1 taps are valid. K is compile-time
// constant, so this remains unrolled for the specialized kernels.
#pragma unroll
        for (int k = 0; k < K; ++k)
        {
            if (k <= t)
            {
                const float b_gate = static_cast<float>(source[0]);
                const float x_tilde =
                    static_cast<float>(source[2LL * model_dim]);
                const float weight = static_cast<float>(*weight_ptr);

                convolution = fmaf(weight, b_gate * x_tilde, convolution);

                // Advance only when another valid tap exists. This keeps source
                // within the projected-input allocation when t < K - 1.
                if (k < t)
                {
                    source -= projected_dim;
                    weight_ptr += model_dim;
                }
            }
        }
    }

    gate_output[output_idx] = static_cast<scalar_t>(c_gate * convolution);
}
// Fallback generic kernel for non-templated kernel sizes
template <typename scalar_t>
__global__ void convolution_gate_kernel_generic(
    const scalar_t *__restrict__ projected_input,
    const scalar_t *__restrict__ conv_weight_kd,
    const scalar_t *__restrict__ conv_bias,
    scalar_t *__restrict__ gate_output,
    int num_tokens,
    int model_dim,
    int kernel_size)
{
    const int d = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int bt = static_cast<int>(blockIdx.y);
    if (d >= model_dim)
    {
        return;
    }

    const int t = bt % num_tokens;
    const int64_t projected_dim = static_cast<int64_t>(3) * model_dim;
    const int64_t current_base = static_cast<int64_t>(bt) * projected_dim;
    const int64_t output_idx = static_cast<int64_t>(bt) * model_dim + d;

    const float c_gate =
        static_cast<float>(projected_input[current_base + model_dim + d]);
    float convolution = static_cast<float>(conv_bias[d]);

    const scalar_t *source = projected_input + current_base + d;
    const scalar_t *weight_ptr = conv_weight_kd + d;

    if (t >= kernel_size - 1)
    {
        // Interior path: all runtime-sized taps are valid.
        for (int k = 0; k < kernel_size; ++k)
        {
            const float b_gate = static_cast<float>(source[0]);
            const float x_tilde =
                static_cast<float>(source[2LL * model_dim]);
            const float weight = static_cast<float>(*weight_ptr);

            convolution = fmaf(weight, b_gate * x_tilde, convolution);

            if (k + 1 < kernel_size)
            {
                source -= projected_dim;
                weight_ptr += model_dim;
            }
        }
    }
    else
    {
        // Boundary path: execute exactly the valid causal taps. Expressing the
        // bound directly avoids testing k <= t on every iteration.
        const int valid_taps = t + 1;
        for (int k = 0; k < valid_taps; ++k)
        {
            const float b_gate = static_cast<float>(source[0]);
            const float x_tilde =
                static_cast<float>(source[2LL * model_dim]);
            const float weight = static_cast<float>(*weight_ptr);

            convolution = fmaf(weight, b_gate * x_tilde, convolution);

            if (k + 1 < valid_taps)
            {
                source -= projected_dim;
                weight_ptr += model_dim;
            }
        }
    }

    gate_output[output_idx] = static_cast<scalar_t>(c_gate * convolution);
}
// ============================================================================
// Tensor validation helpers
// ============================================================================

void check_cuda_half_contiguous(
    const torch::Tensor &tensor,
    const char *name,
    const torch::Device &expected_device)
{
    TORCH_CHECK(
        tensor.is_cuda(),
        name,
        " must be a CUDA tensor");

    TORCH_CHECK(
        tensor.device() == expected_device,
        name,
        " must be on the same device as input");

    TORCH_CHECK(
        tensor.scalar_type() == at::kHalf,
        name,
        " must have dtype torch.float16");

    TORCH_CHECK(
        tensor.is_contiguous(),
        name,
        " must be contiguous");
}

// ============================================================================
// CUDA entry point
// ============================================================================

torch::Tensor convolution_gate_cuda(
    const torch::Tensor &input,
    const torch::Tensor &linear1_weight,
    const torch::Tensor &linear1_bias,
    const torch::Tensor &conv_weight,
    const torch::Tensor &conv_bias,
    const torch::Tensor &linear2_weight,
    const torch::Tensor &linear2_bias)
{
    // ------------------------------------------------------------------------
    // Validate input rank
    //
    // Supported:
    //   [D]
    //   [T, D]
    //   [B, T, D]
    // ------------------------------------------------------------------------

    TORCH_CHECK(
        input.dim() >= 1 && input.dim() <= 3,
        "input must have shape [D], [T, D], or [B, T, D]");

    check_cuda_half_contiguous(
        input,
        "input",
        input.device());

    check_cuda_half_contiguous(
        linear1_weight,
        "linear1_weight",
        input.device());

    check_cuda_half_contiguous(
        linear1_bias,
        "linear1_bias",
        input.device());

    check_cuda_half_contiguous(
        conv_weight,
        "conv_weight",
        input.device());
    check_cuda_half_contiguous(
        conv_bias,
        "conv_bias",
        input.device());

    check_cuda_half_contiguous(
        linear2_weight,
        "linear2_weight",
        input.device());

    check_cuda_half_contiguous(
        linear2_bias,
        "linear2_bias",
        input.device());

    // ------------------------------------------------------------------------
    // Normalize input shape to [B, T, D].
    //
    // Input [D] is the one-token form B=1, T=1.
    // Input [T, D] is the unbatched sequence form B=1.
    // ------------------------------------------------------------------------

    int64_t B64;
    int64_t T64;
    int64_t D64;

    if (input.dim() == 1)
    {
        B64 = 1;
        T64 = 1;
        D64 = input.size(0);
    }
    else if (input.dim() == 2)
    {
        B64 = 1;
        T64 = input.size(0);
        D64 = input.size(1);
    }
    else
    {
        B64 = input.size(0);
        T64 = input.size(1);
        D64 = input.size(2);
    }

    TORCH_CHECK(
        D64 > 0,
        "model dimension D must be greater than zero");

    TORCH_CHECK(
        D64 <= std::numeric_limits<int>::max(),
        "model dimension D exceeds the supported int range");

    TORCH_CHECK(
        B64 <= std::numeric_limits<int>::max() &&
            T64 <= std::numeric_limits<int>::max(),
        "batch or token dimension exceeds the supported int range");

    TORCH_CHECK(
        B64 == 0 ||
            T64 <= std::numeric_limits<int>::max() / B64,
        "B * T exceeds the supported int range");

    TORCH_CHECK(
        D64 <= std::numeric_limits<int64_t>::max() / 3,
        "3 * D overflows int64");

    const int64_t projected_dim64 = 3 * D64;

    TORCH_CHECK(
        projected_dim64 <= std::numeric_limits<int>::max(),
        "3 * D exceeds the supported int range");

    const int B = static_cast<int>(B64);
    const int T = static_cast<int>(T64);
    const int D = static_cast<int>(D64);
    const int projected_dim = static_cast<int>(projected_dim64);
    const int M = static_cast<int>(B64 * T64);

    // ------------------------------------------------------------------------
    // Validate linear projection shapes.
    //
    // PyTorch Linear weights:
    //
    //   linear1_weight: [3D, D]
    //   linear1_bias:   [3D]
    //
    //   linear2_weight: [D, D]
    //   linear2_bias:   [D]
    // ------------------------------------------------------------------------

    TORCH_CHECK(
        linear1_weight.dim() == 2 &&
            linear1_weight.size(0) == projected_dim64 &&
            linear1_weight.size(1) == D64,
        "linear1_weight must have shape [3D, D]; expected [",
        projected_dim64,
        ", ",
        D64,
        "], got ",
        linear1_weight.sizes());

    TORCH_CHECK(
        linear1_bias.dim() == 1 &&
            linear1_bias.numel() == projected_dim64,
        "linear1_bias must have shape [3D]; expected [",
        projected_dim64,
        "], got ",
        linear1_bias.sizes());

    TORCH_CHECK(
        linear2_weight.dim() == 2 &&
            linear2_weight.size(0) == D64 &&
            linear2_weight.size(1) == D64,
        "linear2_weight must have shape [D, D]; expected [",
        D64,
        ", ",
        D64,
        "], got ",
        linear2_weight.sizes());

    TORCH_CHECK(
        linear2_bias.dim() == 1 &&
            linear2_bias.numel() == D64,
        "linear2_bias must have shape [D]; expected [",
        D64,
        "], got ",
        linear2_bias.sizes());

    // ------------------------------------------------------------------------
    // Validate and prepare convolution weight in [K, D] layout.
    //
    // Accepted layouts:
    //   [K, D]    prepacked fast path; used directly without allocation/copy
    //   [D, K]    compatibility path; transposed to contiguous [K, D]
    //   [D, 1, K] PyTorch depthwise Conv1d layout; converted to [K, D]
    //
    // Callers that repeatedly reuse the same weight should prepack it once as:
    //
    //   conv_weight_kd = conv_weight.squeeze(1).transpose(0, 1).contiguous()
    // ------------------------------------------------------------------------
    int64_t kernel_size64 = 0;
    torch::Tensor conv_weight_kd;

    if (conv_weight.dim() == 2 && conv_weight.size(1) == D64)
    {
        // Fast path: already contiguous [K, D]. The earlier tensor validation
        // guarantees CUDA, FP16, same-device, and contiguous storage.
        kernel_size64 = conv_weight.size(0);
        conv_weight_kd = conv_weight;
    }
    else if (conv_weight.dim() == 2 && conv_weight.size(0) == D64)
    {
        // Compatibility path: [D, K] -> [K, D].
        kernel_size64 = conv_weight.size(1);
        conv_weight_kd = conv_weight.transpose(0, 1).contiguous();
    }
    else if (conv_weight.dim() == 3)
    {
        TORCH_CHECK(
            conv_weight.size(0) == D64 && conv_weight.size(1) == 1,
            "3D conv_weight must have shape [D, 1, K]; got ",
            conv_weight.sizes());
        // Compatibility path: [D, 1, K] -> [K, D].
        kernel_size64 = conv_weight.size(2);
        conv_weight_kd =
            conv_weight.squeeze(1).transpose(0, 1).contiguous();
    }
    else
    {
        TORCH_CHECK(
            false,
            "conv_weight must have shape [K, D], [D, K], or [D, 1, K]; got ",
            conv_weight.sizes());
    }

    TORCH_CHECK(
        kernel_size64 > 0,
        "convolution kernel size must be greater than zero");
    TORCH_CHECK(
        kernel_size64 <= std::numeric_limits<int>::max(),
        "convolution kernel size exceeds supported int range");
    TORCH_CHECK(
        conv_bias.dim() == 1 &&
            conv_bias.numel() == D64,
        "conv_bias must have shape [D]");
    const int kernel_size = static_cast<int>(kernel_size64);

    // ------------------------------------------------------------------------
    // Return correctly shaped empty output for empty B/T.
    // ------------------------------------------------------------------------

    if (M == 0)
    {
        return torch::empty_like(input);
    }

    TORCH_CHECK(
        D % 8 == 0,
        "this CUTLASS FP16 Tensor Core configuration requires D to be "
        "a multiple of 8; got D=",
        D);

    c10::cuda::CUDAGuard device_guard(input.device());

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(input.get_device());

    auto projected = torch::empty(
        {B64, T64, projected_dim64},
        input.options());

    auto gate_output = torch::empty(
        {B64, T64, D64},
        input.options());

    auto output_3d = torch::empty(
        {B64, T64, D64},
        input.options());

    // ------------------------------------------------------------------------
    // GEMM 1: Input Projection
    //
    // [M, D] @ transpose([3D, D]) + linear1_bias[3D] -> [M, 3D]
    // ------------------------------------------------------------------------

    cutlass::Status status = launch_linear_cutlass(
        input.data_ptr<at::Half>(),
        linear1_weight.data_ptr<at::Half>(),
        linear1_bias.data_ptr<at::Half>(),
        projected.data_ptr<at::Half>(),
        M,
        projected_dim,
        D,
        stream);

    check_cutlass_status(
        status,
        "CUTLASS input projection (with fused bias)");

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // ------------------------------------------------------------------------
    // Convolutional gate
    //
    // [B, T, 3D] -> [B, T, D]
    // 2D grid: (ceil_div(D, 256), B * T)
    // ------------------------------------------------------------------------

    constexpr int threads = 256;
    dim3 block(threads);
    dim3 grid(
        (D + threads - 1) / threads,
        B * T);

    switch (kernel_size)
    {
    case 2:
        convolution_gate_kernel<at::Half, 2><<<grid, block, 0, stream>>>(
            projected.data_ptr<at::Half>(),
            conv_weight_kd.data_ptr<at::Half>(),
            conv_bias.data_ptr<at::Half>(),
            gate_output.data_ptr<at::Half>(),
            T,
            D);
        break;
    case 3:
        convolution_gate_kernel<at::Half, 3><<<grid, block, 0, stream>>>(
            projected.data_ptr<at::Half>(),
            conv_weight_kd.data_ptr<at::Half>(),
            conv_bias.data_ptr<at::Half>(),
            gate_output.data_ptr<at::Half>(),
            T,
            D);
        break;
    case 4:
        convolution_gate_kernel<at::Half, 4><<<grid, block, 0, stream>>>(
            projected.data_ptr<at::Half>(),
            conv_weight_kd.data_ptr<at::Half>(),
            conv_bias.data_ptr<at::Half>(),
            gate_output.data_ptr<at::Half>(),
            T,
            D);
        break;
    case 8:
        convolution_gate_kernel<at::Half, 8><<<grid, block, 0, stream>>>(
            projected.data_ptr<at::Half>(),
            conv_weight_kd.data_ptr<at::Half>(),
            conv_bias.data_ptr<at::Half>(),
            gate_output.data_ptr<at::Half>(),
            T,
            D);
        break;
    default:
        convolution_gate_kernel_generic<at::Half><<<grid, block, 0, stream>>>(
            projected.data_ptr<at::Half>(),
            conv_weight_kd.data_ptr<at::Half>(),
            conv_bias.data_ptr<at::Half>(),
            gate_output.data_ptr<at::Half>(),
            T,
            D,
            kernel_size);
        break;
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // ------------------------------------------------------------------------
    // GEMM 2: Output Projection with fused linear2_bias epilogue
    //
    // [M, D] @ transpose([D, D]) + linear2_bias[D] -> [M, D]
    // ------------------------------------------------------------------------

    status = launch_linear_cutlass(
        gate_output.data_ptr<at::Half>(),
        linear2_weight.data_ptr<at::Half>(),
        linear2_bias.data_ptr<at::Half>(),
        output_3d.data_ptr<at::Half>(),
        M,
        D,
        D,
        stream);

    check_cutlass_status(
        status,
        "CUTLASS output projection (with fused bias)");

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // ------------------------------------------------------------------------
    // Restore original input rank.
    // ------------------------------------------------------------------------

    if (input.dim() == 1)
    {
        return output_3d.view({D64});
    }

    if (input.dim() == 2)
    {
        return output_3d.view({T64, D64});
    }

    return output_3d;
}