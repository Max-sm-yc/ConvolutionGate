# Convolution Gate Kernel
### Built with Gemini 3.7 Flash, Cursor Composer 2.5, Microsoft 365 Copilot

This repo contains a CUDA kernel for Convolution Gates that might be found in models such as Liquid AI's LFM 2.5. The Convolution Gate in this repo works by running across each channel of the model's dimension for a sequence of token embeddings. I utilized CUTLASS for the matrix multiplication steps in the 2 linear layers and created a kernel for the convolution itself.

The custom CUDA kernel outperforms PyTorch eager and is fractionally faster than Torch.compile (ran on RTX 3080).

     pytorch_eager: mean=  22.587 ms  median=  23.056 ms  std= 0.980 ms  throughput=   44.27 iter/s
       cuda_kernel: mean=  18.116 ms  median=  18.686 ms  std= 1.254 ms  throughput=   55.20 iter/s
     torch.compile: mean=  18.291 ms  median=  18.759 ms  std= 0.992 ms  throughput=   54.67 iter/s

### Why
My motivation for building this was to learn more about the process of creating performant kernels for AI architectures that efficiently utilize hardware. When building out this repo I learned about the architecture I was implementing and decomposed it into various sub operations (matmul, dot product, element multiplication).
I also profiled the kernel in order to optimize parts of the operation (ex. removing a separate add bias kernel that was contributing latency).

### Process
I used AI to research the architecture of gated convolutions used in LFM which allowed me to decompose the underlying math ops. I decided to use CUTLASS for the 2 matmuls in the 2 linear layers due to it being highly optimized and likely better than what I could come up with.
The kernel itself computes the convolution between token wise channels and gates the results. Memory is accessed via striding, allowing a single contiguous section of memory to be used as b_gate, c_gate, and x_tilde.

### Findings
The largest bottleneck for the combined kernels is memory bandwidth (particularly for the 2 matmuls). With increased batch size / sequence size this becomes more manageable and results in the kernel becoming more performant than PyTorch eager. Validation was conducted via comparison with PyTorch implementation, errors fell within expected floating point bounds.
I expect that the kernel is faster than eager implementation due to benefits of compilation and the fused convolution allowing more efficient memory movement.
![alt text](/assets/kernel_3/benchmark_dashboard.png)
*Speedups most noticeable as total tokens increase*

The kernel is only slightly faster than Torch compile likely due to the matmuls taking up the majority of runtime and being harder to optimize.
![alt text](/assets/kernel_3/image.png)
*Nsight Compute profile shows the 2 CUTLASS kernels taking up the vast majority of runtime*
### AI Usage
I used AI to research the convolution gate architecture, enabling me to make better design choices. Once I could distinguish the parts I needed to implement, I specified the structure of the kernel and used Antigravity / Cursor for implementation.