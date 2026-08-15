#include <torch/extension.h>

torch::Tensor convolution_gate_cuda(
    const torch::Tensor &input,
    const torch::Tensor &linear1_weight,
    const torch::Tensor &linear1_bias,
    const torch::Tensor &conv_weight,
    const torch::Tensor &conv_bias,
    const torch::Tensor &linear2_weight,
    const torch::Tensor &linear2_bias);

torch::Tensor convolution_gate(
    const torch::Tensor &input,
    const torch::Tensor &linear1_weight,
    const torch::Tensor &linear1_bias,
    const torch::Tensor &conv_weight,
    const torch::Tensor &conv_bias,
    const torch::Tensor &linear2_weight,
    const torch::Tensor &linear2_bias)
{
    TORCH_CHECK(
        input.is_cuda(),
        "input must be a CUDA tensor");

    return convolution_gate_cuda(
        input,
        linear1_weight,
        linear1_bias,
        conv_weight,
        conv_bias,
        linear2_weight,
        linear2_bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "forward",
        &convolution_gate,
        "LFM convolutional gate forward");
}