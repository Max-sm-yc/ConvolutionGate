#!/usr/bin/env python3
"""Benchmark convolution gate over a matrix of input shapes.

Examples:
    python benchmark.py
    python benchmark.py --preset quick
    python benchmark.py --batch-sizes 1,4,8 --seq-lens 1,128,2048 --dims 768,1024,2048
    python benchmark.py --kernel-sizes 3,4 --output-json results.json --output-csv results.csv
    python benchmark.py --correctness-only --reference-dtype float64

The benchmark compares:
  1. The complete custom CUDA extension forward captured in a CUDA Graph.
  2. torch.compile applied to the idiomatic PyTorch module.

The uncaptured custom kernel and PyTorch eager path are not timed. A separate
high-precision PyTorch implementation remains the correctness oracle.

Convolution convention (matching the original CUDA benchmark):
    z[t, d] = conv_bias[d] + sum_k y[t-k, d] * conv_weight[d, 0, k]
with out-of-range y indices treated as zero.

The timed PyTorch reference uses tensor shifts and elementwise operations rather
than F.conv1d, avoiding a dependency on cuDNN while preserving the CUDA kernel's
coefficient convention exactly.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

PRESETS = {
    "quick": {
        "batch_sizes": [1, 8],
        "seq_lens": [1, 128, 2048],
        "dims": [768, 1024, 2048],
        "kernel_sizes": [3, 4],
    },
    "lfm": {
        "batch_sizes": [1, 4, 8],
        "seq_lens": [1, 16, 128, 512, 2048],
        "dims": [768, 1024, 1536, 2048],
        "kernel_sizes": [3],
    },
    "full": {
        "batch_sizes": [1, 2, 4, 8, 16],
        "seq_lens": [1, 16, 128, 512, 2048, 8192],
        "dims": [512, 768, 1024, 1536, 2048],
        "kernel_sizes": [3, 4],
    },
}


def find_cutlass_include() -> str | None:
    for candidate in (
        Path(torch.__file__).resolve().parent / "include",
        Path("/usr/local/cutlass/include"),
    ):
        if (candidate / "cutlass" / "cutlass.h").is_file():
            return str(candidate)
    try:
        header = subprocess.check_output(
            [
                "find", "/home", "/usr/local",
                "-path", "*/include/cutlass/cutlass.h",
                "-print", "-quit",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return str(Path(header).parent.parent) if header else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def load_extension():
    cutlass_include = find_cutlass_include()
    if cutlass_include is None:
        raise RuntimeError("CUTLASS headers not found")
    return load(
        name="convolution_gate",
        sources=[str(ROOT / "interface.cpp"), str(ROOT / "convolution_gate.cu")],
        extra_include_paths=[cutlass_include],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=False,
    )


def parse_int_list(value: str) -> list[int]:
    values = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        number = int(item)
        if number < 1:
            raise argparse.ArgumentTypeError("all values must be positive")
        values.append(number)
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list of integers")
    return values


def causal_depthwise_conv1d(
    y: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch causal depthwise convolution without cuDNN.

    Computes z[t, d] = bias[d] + sum_k y[t-k, d] * weight[d, k].
    This independently matches the CUDA kernel's coefficient convention and
    avoids F.conv1d/cuDNN, which is useful on installations with mismatched
    cuDNN sublibraries. The loop is only over the short kernel width.
    """
    weight = conv_weight[:, 0, :] if conv_weight.ndim == 3 else conv_weight
    if weight.ndim != 2:
        raise ValueError("conv_weight must have shape [D, 1, K] or [D, K]")
    batch, seq_len, dim = y.shape
    if weight.shape[0] != dim:
        raise ValueError("conv_weight channel count must equal input dimension")
    z = conv_bias.view(1, 1, dim).expand(batch, seq_len, dim)
    terms = [z]
    for k in range(weight.shape[-1]):
        if k >= seq_len:
            break
        term = y[:, : seq_len - k, :] * weight[:, k].view(1, 1, dim)
        if k:
            term = F.pad(term, (0, 0, k, 0))
        terms.append(term)
    return torch.stack(terms, dim=0).sum(dim=0)


def explicit_causal_conv_reference(
    y: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    accumulation_dtype: torch.dtype,
) -> torch.Tensor:
    """Simple independent oracle; intentionally not used for performance."""
    weight = conv_weight[:, 0, :] if conv_weight.ndim == 3 else conv_weight
    y_acc = y.to(accumulation_dtype)
    weight_acc = weight.to(accumulation_dtype)
    z = conv_bias.to(accumulation_dtype).view(1, 1, -1).expand_as(y_acc).clone()
    seq_len = y.shape[1]
    for k in range(weight.shape[-1]):
        if k < seq_len:
            z[:, k:, :].add_(y_acc[:, : seq_len - k, :] * weight_acc[:, k].view(1, 1, -1))
    return z


def pytorch_convolution_gate(
    x: torch.Tensor,
    linear1_weight: torch.Tensor,
    linear1_bias: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    linear2_weight: torch.Tensor,
    linear2_bias: torch.Tensor,
) -> torch.Tensor:
    projected = F.linear(x, linear1_weight, linear1_bias)
    b_gate, c_gate, x_tilde = projected.chunk(3, dim=-1)
    y = b_gate * x_tilde
    z = causal_depthwise_conv1d(y, conv_weight, conv_bias)
    return F.linear(c_gate * z, linear2_weight, linear2_bias)


def high_precision_reference(tensors: tuple[torch.Tensor, ...], dtype: torch.dtype) -> torch.Tensor:
    """Independent reference with explicit convolution and high-precision math."""
    x, w1, b1, wc, bc, w2, b2 = tensors
    values = [tensor.to(dtype) for tensor in tensors]
    x_ref, w1_ref, b1_ref, wc_ref, bc_ref, w2_ref, b2_ref = values
    projected = F.linear(x_ref, w1_ref, b1_ref)
    b_gate, c_gate, x_tilde = projected.chunk(3, dim=-1)
    y = b_gate * x_tilde
    z = explicit_causal_conv_reference(y, wc_ref, bc_ref, dtype)
    return F.linear(c_gate * z, w2_ref, b2_ref)


class ConvolutionGateModule(torch.nn.Module):
    def __init__(self, tensors: tuple[torch.Tensor, ...]):
        super().__init__()
        names = ("linear1_weight", "linear1_bias", "conv_weight", "conv_bias",
                 "linear2_weight", "linear2_bias")
        for name, value in zip(names, tensors[1:]):
            self.register_buffer(name, value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pytorch_convolution_gate(
            x, self.linear1_weight, self.linear1_bias,
            self.conv_weight, self.conv_bias,
            self.linear2_weight, self.linear2_bias,
        )


def make_tensors(batch: int, seq: int, dim: int, kernel: int,
                 device: torch.device, dtype: torch.dtype, scale: float,
                 seed: int) -> tuple[torch.Tensor, ...]:
    if dim % 8 != 0:
        raise ValueError("dim must be a multiple of 8 for the CUTLASS kernel")
    generator = torch.Generator(device=device).manual_seed(seed)
    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device=device, dtype=dtype, generator=generator) * scale
    return (
        randn(batch, seq, dim), randn(3 * dim, dim), randn(3 * dim),
        randn(dim, 1, kernel), randn(dim), randn(dim, dim), randn(dim),
    )


@dataclass(frozen=True)
class Shape:
    batch: int
    seq: int
    dim: int
    kernel: int


@dataclass
class Result:
    batch: int
    seq: int
    dim: int
    kernel: int
    dtype: str
    implementation: str
    mean_ms: float | None = None
    median_ms: float | None = None
    p05_ms: float | None = None
    p95_ms: float | None = None
    std_ms: float | None = None
    iterations_per_second: float | None = None
    tokens_per_second: float | None = None
    speedup_vs_compile: float | None = None
    max_abs_error: float | None = None
    mean_abs_error: float | None = None
    max_rel_error: float | None = None
    passed: bool = False
    status: str = "ok"


def error_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.float()
    cand = candidate.float()
    diff = (ref - cand).abs()
    denominator = ref.abs().clamp_min(1e-7)
    return {
        "max_abs_error": diff.max().item(),
        "mean_abs_error": diff.mean().item(),
        "max_rel_error": (diff / denominator).max().item(),
    }


def validate(reference: torch.Tensor, candidate: torch.Tensor,
             rtol: float, atol: float, name: str) -> dict[str, float | bool]:
    metrics = error_metrics(reference, candidate)
    finite = torch.isfinite(candidate).all().item()
    passed = bool(finite and torch.allclose(reference, candidate.float(), rtol=rtol, atol=atol))
    if not passed:
        raise RuntimeError(
            f"{name} failed correctness: finite={finite}, "
            f"max_abs={metrics['max_abs_error']:.6e}, "
            f"mean_abs={metrics['mean_abs_error']:.6e}, "
            f"max_rel={metrics['max_rel_error']:.6e}, rtol={rtol}, atol={atol}"
        )
    return {**metrics, "passed": passed}


@dataclass
class CapturedCudaGraph:
    """Fixed-shape CUDA Graph and the static tensors whose addresses it uses."""

    graph: torch.cuda.CUDAGraph
    static_input: torch.Tensor
    static_output: torch.Tensor

    def replay(self) -> torch.Tensor:
        self.graph.replay()
        return self.static_output


@torch.no_grad()
def capture_custom_graph(
    extension,
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    wc_kd: torch.Tensor,
    bc: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    warmup: int = 3,
) -> CapturedCudaGraph:
    """Capture the extension's GEMM -> gate -> GEMM block for one shape."""
    static_input = torch.empty_like(x)
    static_input.copy_(x)

    def forward() -> torch.Tensor:
        return extension.forward(static_input, w1, b1, wc_kd, bc, w2, b2)

    current_stream = torch.cuda.current_stream(x.device)
    warmup_stream = torch.cuda.Stream(device=x.device)
    warmup_stream.wait_stream(current_stream)
    with torch.cuda.stream(warmup_stream):
        for _ in range(max(1, warmup)):
            forward()
    current_stream.wait_stream(warmup_stream)
    current_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = forward()

    return CapturedCudaGraph(graph, static_input, static_output)


@torch.no_grad()
def benchmark(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    timings = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))

    ordered = sorted(timings)
    def percentile(frac: float) -> float:
        index = min(len(ordered) - 1, max(0, math.ceil(frac * len(ordered)) - 1))
        return ordered[index]

    mean = statistics.mean(timings)
    return {
        "mean_ms": mean,
        "median_ms": statistics.median(timings),
        "p05_ms": percentile(0.05),
        "p95_ms": percentile(0.95),
        "std_ms": statistics.pstdev(timings) if len(timings) > 1 else 0.0,
        "iterations_per_second": 1000.0 / mean,
    }


def resolve_shapes(args: argparse.Namespace) -> list[Shape]:
    if args.preset:
        spec = PRESETS[args.preset]
        batches, seqs, dims, kernels = (
            spec["batch_sizes"], spec["seq_lens"], spec["dims"], spec["kernel_sizes"]
        )
    else:
        batches, seqs, dims, kernels = (
            args.batch_sizes, args.seq_lens, args.dims, args.kernel_sizes
        )
    return [Shape(b, s, d, k) for b in batches for s in seqs for d in dims for k in kernels]


def print_result(result: Result) -> None:
    if result.status != "ok":
        print(f"  {result.implementation:14s} {result.status}")
        return
    timing = "correctness only" if result.mean_ms is None else (
        f"{result.mean_ms:9.3f} ms  p95={result.p95_ms:9.3f}  "
        f"{result.tokens_per_second:12,.0f} tok/s"
    )
    speedup = ("" if result.speedup_vs_compile is None else
               f"  {result.speedup_vs_compile:6.3f}x vs compile")
    print(
        f"  {result.implementation:14s} {timing}{speedup}  "
        f"max_abs={result.max_abs_error:.3e}  {'PASS' if result.passed else 'FAIL'}"
    )


def run_shape(shape: Shape, args: argparse.Namespace, extension,
              device: torch.device, dtype: torch.dtype) -> list[Result]:
    print(f"\nB={shape.batch:<3} T={shape.seq:<5} D={shape.dim:<5} K={shape.kernel}")
    tensors = make_tensors(shape.batch, shape.seq, shape.dim, shape.kernel,
                           device, dtype, args.scale, args.seed)
    x, w1, b1, wc, bc, w2, b2 = tensors

    # Prepack once per shape. The packed weight and all other captured tensors
    # remain alive until timing for this shape finishes.
    wc_kd = wc.squeeze(1).transpose(0, 1).contiguous()
    module = ConvolutionGateModule(tensors).eval()

    # Float64 is the strongest oracle, but float32 is usually sufficient for
    # validating FP16/BF16 inference.
    reference_dtype = DTYPES[args.reference_dtype]
    reference = high_precision_reference(tensors, reference_dtype).float()

    captured = capture_custom_graph(
        extension, x, w1, b1, wc_kd, bc, w2, b2,
        warmup=min(args.warmup, 3),
    )

    def cuda_graph() -> torch.Tensor:
        # Replay-only benchmark. Input-copy cost is excluded because every
        # range-benchmark shape reuses the same input values.
        return captured.replay()

    implementations: list[tuple[str, Callable[[], torch.Tensor]]] = [
        ("cuda_graph", cuda_graph),
    ]

    compiled_module = None
    if not args.skip_compile:
        compiled_module = torch.compile(module, mode=args.compile_mode)

        def compiled() -> torch.Tensor:
            return compiled_module(x)

        # Trigger compilation/autotuning before correctness and timing.
        compiled()
        implementations.append(("torch_compile", compiled))

    results: list[Result] = []
    for name, fn in implementations:
        output = fn()
        metrics = validate(reference, output, args.rtol, args.atol, name)
        timings = {} if args.correctness_only else benchmark(fn, args.warmup, args.iters)
        result = Result(
            batch=shape.batch, seq=shape.seq, dim=shape.dim, kernel=shape.kernel,
            dtype=args.dtype, implementation=name,
            tokens_per_second=(shape.batch * shape.seq * 1000.0 / timings["mean_ms"])
                if timings else None,
            **timings, **metrics,
        )
        results.append(result)

    # Report graph speedup relative to torch.compile. A value above 1 means
    # CUDA Graph replay is faster; below 1 means torch.compile is faster.
    by_name = {result.implementation: result for result in results}
    graph_result = by_name["cuda_graph"]
    compile_result = by_name.get("torch_compile")
    if (graph_result.mean_ms is not None and compile_result is not None and
            compile_result.mean_ms is not None):
        graph_result.speedup_vs_compile = compile_result.mean_ms / graph_result.mean_ms
        compile_result.speedup_vs_compile = 1.0

    # Print only after calculating the cross-implementation comparison.
    for result in results:
        print_result(result)

    del captured, tensors, module, reference, compiled_module
    torch.cuda.empty_cache()
    return results


def write_results(results: list[Result], json_path: Path | None, csv_path: Path | None) -> None:
    rows = [asdict(result) for result in results]
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(rows, indent=2))
        print(f"\nJSON: {json_path}")
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)
        print(f"CSV:  {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark convolution gate across shape ranges")
    parser.add_argument("--preset", choices=tuple(PRESETS), default=None)
    parser.add_argument("--batch-sizes", type=parse_int_list, default=[1, 8])
    parser.add_argument("--seq-lens", type=parse_int_list, default=[1, 128, 2048])
    parser.add_argument("--dims", type=parse_int_list, default=[768, 1024, 2048])
    parser.add_argument("--kernel-sizes", type=parse_int_list, default=[3])
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float16")
    parser.add_argument("--reference-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--scale", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"),
                        default="reduce-overhead")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--output-json", type=Path, default=ROOT / "benchmark_results.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "benchmark_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.warmup < 0 or args.iters < 1:
        raise ValueError("warmup must be >= 0 and iters must be >= 1")

    device = torch.device("cuda")
    dtype = DTYPES[args.dtype]
    shapes = resolve_shapes(args)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
    print(f"Shapes: {len(shapes)}; dtype={args.dtype}; reference={args.reference_dtype}")
    print("Loading CUDA extension...")
    started = time.perf_counter()
    extension = load_extension()
    print(f"Extension loaded in {time.perf_counter() - started:.2f}s")

    results: list[Result] = []
    for shape in shapes:
        try:
            results.extend(run_shape(shape, args, extension, device, dtype))
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            torch.cuda.empty_cache()
            message = f"{type(exc).__name__}: {exc}"
            print(f"  ERROR: {message}")
            if not args.continue_on_error:
                raise
            results.append(Result(
                batch=shape.batch, seq=shape.seq, dim=shape.dim, kernel=shape.kernel,
                dtype=args.dtype, implementation="shape", status=message,
            ))

    write_results(results, args.output_json, args.output_csv)

    successful = [
        r for r in results
        if r.status == "ok" and r.implementation == "cuda_graph"
        and r.speedup_vs_compile is not None
    ]
    if successful and not args.correctness_only:
        speedups = [r.speedup_vs_compile for r in successful]
        print(f"\nCUDA Graph vs torch.compile across {len(speedups)} shapes: "
              f"geomean={statistics.geometric_mean(speedups):.3f}x, "
              f"min={min(speedups):.3f}x, max={max(speedups):.3f}x")


if __name__ == "__main__":
    main()
