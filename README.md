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
