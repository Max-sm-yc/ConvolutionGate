#!/usr/bin/env python3
"""Benchmark convolution gate: custom CUDA kernel, CUDA Graph, PyTorch eager, and torch.compile.

Usage:
    source /home/maxsun/atlas-local/venv/bin/activate
    python benchmark.py
    python benchmark.py --profile
    python benchmark.py --profile-only --profile-dir profile_output
    # Nsight Systems (steady-state custom CUDA kernel only):
    # nsys profile --trace=cuda,nvtx,osrt \
    #   --capture-range=cudaProfilerApi --capture-range-end=stop \
    #   -o nsight_cuda python benchmark_nsight_cuda.py \
    #   --nsight --nsight-target cuda --nsight-iters 10
    # Nsight Compute (filter to the custom CUDA extension NVTX range):
    # sudo -E ncu --nvtx --nvtx-include "custom_cuda_extension/" \
    #   --profile-from-start off --set full -o ncu_cuda \
    #   python benchmark_nsight_cuda.py \
    #   --nsight --nsight-target cuda --nsight-iters 1
    # Capture all implementations in separate NVTX ranges with Nsight Systems:
    # nsys profile --trace=cuda,nvtx,osrt \
    #   --capture-range=cudaProfilerApi --capture-range-end=stop \
    #   -o nsight_all python benchmark_nsight_cuda.py \
    #   --nsight --nsight-target all --nsight-iters 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent


def find_cutlass_include() -> str | None:
    try:
        header = subprocess.check_output(
            [
                "find",
                "/home/maxsun",
                "/usr/local",
                "-path",
                "*/include/cutlass/cutlass.h",
                "-print",
                "-quit",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if header:
            return str(Path(header).parent.parent)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    for candidate in (
        Path(torch.__file__).resolve().parent / "include",
        Path("/usr/local/cutlass/include"),
    ):
        if (candidate / "cutlass" / "cutlass.h").is_file():
            return str(candidate)

    return None


def load_extension():
    cutlass_include = find_cutlass_include()
    if cutlass_include is None:
        raise RuntimeError(
            "CUTLASS headers not found. Install CUTLASS or a package that "
            "bundles it (for example flashinfer or tilelang)."
        )

    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        "-lineinfo",
    ]

    return load(
        name="convolution_gate",
        sources=[
            str(ROOT / "interface.cpp"),
            str(ROOT / "convolution_gate.cu"),
        ],
        extra_include_paths=[cutlass_include],
        extra_cflags=["-O3"],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False,
    )


def causal_depthwise_conv1d(
    y: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
) -> torch.Tensor:
    """Depthwise causal conv matching convolution_gate_kernel."""
    if conv_weight.dim() == 3:
        weight = conv_weight.squeeze(1)
    else:
        weight = conv_weight

    batch_size, seq_len, dim = y.shape
    kernel_size = weight.shape[-1]
    z = conv_bias.view(1, 1, dim).expand(batch_size, seq_len, dim).clone()
    for k in range(kernel_size):
        source = y[:, : seq_len - k, :] if k > 0 else y
        target = z[:, k:, :]
        target.addcmul_(source, weight[:, k].view(1, 1, dim))
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
    projected = F.linear(x, linear1_weight)
    dim = x.shape[-1]
    b_gate = projected[..., :dim] + linear1_bias[:dim]
    c_gate = projected[..., dim : 2 * dim] + linear1_bias[dim : 2 * dim]
    x_tilde = projected[..., 2 * dim :] + linear1_bias[2 * dim :]
    y = b_gate * x_tilde
    z = causal_depthwise_conv1d(y, conv_weight, conv_bias)
    gate_output = c_gate * z
    return F.linear(gate_output, linear2_weight, linear2_bias)


class ConvolutionGateModule(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int,
        *,
        linear1_weight: torch.Tensor,
        linear1_bias: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor,
        linear2_weight: torch.Tensor,
        linear2_bias: torch.Tensor,
    ):
        super().__init__()
        self.register_buffer("linear1_weight", linear1_weight)
        self.register_buffer("linear1_bias", linear1_bias)
        self.register_buffer("conv_weight", conv_weight)
        self.register_buffer("conv_bias", conv_bias)
        self.register_buffer("linear2_weight", linear2_weight)
        self.register_buffer("linear2_bias", linear2_bias)
        self.dim = dim
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pytorch_convolution_gate(
            x,
            self.linear1_weight,
            self.linear1_bias,
            self.conv_weight,
            self.conv_bias,
            self.linear2_weight,
            self.linear2_bias,
        )


def make_tensors(
    batch_size: int,
    seq_len: int,
    dim: int,
    kernel_size: int,
    device: torch.device,
    dtype: torch.dtype,
    scale: float = 0.1,
):
    if dim % 8 != 0:
        raise ValueError("dim must be a multiple of 8 for the CUTLASS kernel")

    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype) * scale
    linear1_weight = torch.randn(3 * dim, dim, device=device, dtype=dtype) * scale
    linear1_bias = torch.randn(3 * dim, device=device, dtype=dtype) * scale
    conv_weight = torch.randn(dim, 1, kernel_size, device=device, dtype=dtype) * scale
    conv_bias = torch.randn(dim, device=device, dtype=dtype) * scale
    linear2_weight = torch.randn(dim, dim, device=device, dtype=dtype) * scale
    linear2_bias = torch.randn(dim, device=device, dtype=dtype) * scale
    return (
        x,
        linear1_weight,
        linear1_bias,
        conv_weight,
        conv_bias,
        linear2_weight,
        linear2_bias,
    )


@dataclass
class CapturedCudaGraph:
    """Captured fixed-shape custom-extension forward pass.

    Replay uses the tensor storage addresses recorded during capture.
    ``static_output`` is overwritten by every replay.
    """

    graph: torch.cuda.CUDAGraph
    static_input: torch.Tensor
    static_output: torch.Tensor

    def replay(self) -> torch.Tensor:
        self.graph.replay()
        return self.static_output

    def copy_and_replay(self, value: torch.Tensor) -> torch.Tensor:
        """Copy a new value into fixed input storage, then replay."""
        self.static_input.copy_(value)
        self.graph.replay()
        return self.static_output


@torch.no_grad()
def capture_cuda_extension_graph(
    extension,
    x: torch.Tensor,
    linear1_weight: torch.Tensor,
    linear1_bias: torch.Tensor,
    conv_weight_kd: torch.Tensor,
    conv_bias: torch.Tensor,
    linear2_weight: torch.Tensor,
    linear2_bias: torch.Tensor,
    warmup: int = 3,
) -> CapturedCudaGraph:
    """Warm up and capture the complete extension forward pass."""
    static_input = torch.empty_like(x)
    static_input.copy_(x)

    def forward() -> torch.Tensor:
        return extension.forward(
            static_input,
            linear1_weight,
            linear1_bias,
            conv_weight_kd,
            conv_bias,
            linear2_weight,
            linear2_bias,
        )

    # Initialize CUTLASS and allocator state before capture on a side stream.
    current_stream = torch.cuda.current_stream(x.device)
    warmup_stream = torch.cuda.Stream(device=x.device)
    warmup_stream.wait_stream(current_stream)
    with torch.cuda.stream(warmup_stream):
        for _ in range(max(warmup, 1)):
            forward()
    current_stream.wait_stream(warmup_stream)
    current_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = forward()

    return CapturedCudaGraph(graph, static_input, static_output)


@torch.no_grad()
def benchmark(fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    timings_ms: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings_ms.append(start.elapsed_time(end))
    return timings_ms


@torch.no_grad()
def run_nsight_capture(
    targets: list[tuple[str, object]], warmup: int, iters: int
) -> None:
    """Capture steady-state kernels with profiler API and nested NVTX ranges."""
    # Keep compilation/autotuning outside the capture.
    for name, fn in targets:
        print(f"Nsight warmup: {name} ({warmup} iteration(s))")
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()

    names = ", ".join(name for name, _ in targets)
    print(f"Starting Nsight capture: {names}; {iters} iteration(s) each")
    torch.cuda.cudart().cudaProfilerStart()
    try:
        with torch.cuda.nvtx.range("convolution_gate_nsight_capture"):
            for name, fn in targets:
                with torch.cuda.nvtx.range(name):
                    for iteration in range(iters):
                        with torch.cuda.nvtx.range(f"{name}/iteration_{iteration}"):
                            fn()
        torch.cuda.synchronize()
    finally:
        torch.cuda.cudart().cudaProfilerStop()
    print("Nsight capture complete")


def summarize(name: str, timings_ms: list[float]) -> dict[str, float]:
    mean_ms = statistics.mean(timings_ms)
    median_ms = statistics.median(timings_ms)
    std_ms = statistics.pstdev(timings_ms) if len(timings_ms) > 1 else 0.0
    print(
        f"{name:>18}: "
        f"mean={mean_ms:8.3f} ms  "
        f"median={median_ms:8.3f} ms  "
        f"std={std_ms:6.3f} ms  "
        f"throughput={1000.0 / mean_ms:8.2f} iter/s"
    )
    return {"mean_ms": mean_ms, "median_ms": median_ms, "std_ms": std_ms}


def check_correctness(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    name: str,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> None:
    if not torch.allclose(reference, candidate, rtol=rtol, atol=atol):
        diff = (reference - candidate).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        raise RuntimeError(
            f"{name} mismatch: max_abs_diff={max_diff:.6e}, mean_abs_diff={mean_diff:.6e}"
        )
    print(f"{name} correctness check passed")


@dataclass(frozen=True)
class WorkloadStats:
    batch_size: int
    seq_len: int
    dim: int
    kernel_size: int

    @property
    def tokens(self) -> int:
        return self.batch_size * self.seq_len

    def gemm1_flops(self) -> int:
        # [M, D] @ [D, 3D]
        return 2 * self.tokens * self.dim * (3 * self.dim)

    def gemm2_flops(self) -> int:
        # [M, D] @ [D, D]
        return 2 * self.tokens * self.dim * self.dim

    def gate_flops(self) -> int:
        # b*x multiply, causal conv (K mul-adds), c*z multiply.
        return self.tokens * self.dim * (4 * self.kernel_size + 2)

    @property
    def total_flops(self) -> int:
        return self.gemm1_flops() + self.gemm2_flops() + self.gate_flops()

    def approx_bytes(self, dtype: torch.dtype) -> int:
        elem = torch.tensor([], dtype=dtype).element_size()
        d = self.dim
        t = self.tokens
        k = self.kernel_size
        # Dominant activations + weights touched once per forward.
        activations = elem * t * (d + 3 * d + d + d)  # input, projected, y/z, output
        weights = elem * (3 * d * d + d * k + d + d * d + d)
        return activations + weights


@torch.no_grad()
def cuda_event_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


@torch.no_grad()
def profile_pytorch_stages(
    x: torch.Tensor,
    linear1_weight: torch.Tensor,
    linear1_bias: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    linear2_weight: torch.Tensor,
    linear2_bias: torch.Tensor,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    dim = x.shape[-1]

    def run_stage(stage: str) -> float:
        timings: list[float] = []
        for _ in range(warmup + iters):
            if stage == "linear1":
                timings.append(
                    cuda_event_ms(lambda: F.linear(x, linear1_weight))
                )
            elif stage == "gate":
                projected = F.linear(x, linear1_weight)
                timings.append(
                    cuda_event_ms(
                        lambda projected=projected: _pytorch_gate_stage(
                            projected,
                            linear1_bias,
                            conv_weight,
                            conv_bias,
                            dim,
                        )
                    )
                )
            elif stage == "linear2":
                projected = F.linear(x, linear1_weight)
                gate_output = _pytorch_gate_stage(
                    projected,
                    linear1_bias,
                    conv_weight,
                    conv_bias,
                    dim,
                )
                timings.append(
                    cuda_event_ms(
                        lambda gate_output=gate_output: F.linear(
                            gate_output, linear2_weight, linear2_bias
                        )
                    )
                )
            else:
                raise ValueError(stage)

        return statistics.mean(timings[warmup:])

    return {
        "linear1": run_stage("linear1"),
        "gate": run_stage("gate"),
        "linear2": run_stage("linear2"),
    }


def _pytorch_gate_stage(
    projected: torch.Tensor,
    linear1_bias: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    b_gate = projected[..., :dim] + linear1_bias[:dim]
    c_gate = projected[..., dim : 2 * dim] + linear1_bias[dim : 2 * dim]
    x_tilde = projected[..., 2 * dim :] + linear1_bias[2 * dim :]
    y = b_gate * x_tilde
    z = causal_depthwise_conv1d(y, conv_weight, conv_bias)
    return c_gate * z


def _format_bytes(num_bytes: int) -> str:
    if num_bytes >= 1024**3:
        return f"{num_bytes / 1024**3:.2f} GiB"
    if num_bytes >= 1024**2:
        return f"{num_bytes / 1024**2:.2f} MiB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.2f} KiB"
    return f"{num_bytes} B"


def print_workload_analysis(
    workload: WorkloadStats,
    dtype: torch.dtype,
    total_ms: float | None,
) -> None:
    gemm1 = workload.gemm1_flops()
    gemm2 = workload.gemm2_flops()
    gate = workload.gate_flops()
    total = workload.total_flops
    approx_bytes = workload.approx_bytes(dtype)

    print("\nWorkload analysis")
    print(
        f"  tokens (B*T): {workload.tokens:,}  "
        f"params touched: ~{_format_bytes(approx_bytes)}"
    )
    print("  estimated FLOPs:")
    print(f"    linear1 GEMM: {gemm1 / 1e9:8.2f} GFLOP  ({100 * gemm1 / total:5.1f}%)")
    print(f"    gate + conv:  {gate / 1e9:8.2f} GFLOP  ({100 * gate / total:5.1f}%)")
    print(f"    linear2 GEMM: {gemm2 / 1e9:8.2f} GFLOP  ({100 * gemm2 / total:5.1f}%)")
    print(f"    total:        {total / 1e9:8.2f} GFLOP")

    if total_ms is not None and total_ms > 0:
        tflops = total / (total_ms * 1e-3) / 1e12
        gbps = approx_bytes / (total_ms * 1e-3) / 1e9
        print(f"  effective throughput @ {total_ms:.3f} ms:")
        print(f"    {tflops:6.2f} TFLOP/s (includes gate estimate)")
        print(f"    {gbps:6.2f} GB/s   (rough memory traffic estimate)")


def print_stage_breakdown(stages: dict[str, float], total_ms: float | None = None) -> None:
    stage_total = sum(stages.values())
    print("\nPyTorch eager stage breakdown (CUDA events)")
    for name, ms in stages.items():
        pct = 100.0 * ms / stage_total
        print(f"  {name:>8}: {ms:8.3f} ms  ({pct:5.1f}% of staged total)")
    print(f"  {'sum':>8}: {stage_total:8.3f} ms")
    if total_ms is not None:
        overhead = total_ms - stage_total
        print(
            f"  {'overhead':>8}: {overhead:8.3f} ms  "
            f"(full forward {total_ms:.3f} ms - staged sum; includes recomputation)"
        )


def _event_device_us(event) -> int:
    return int(
        getattr(event, "device_time_total", None)
        or getattr(event, "cuda_time_total", 0)
        or 0
    )


def _event_self_device_us(event) -> int:
    return int(
        getattr(event, "self_device_time_total", None)
        or getattr(event, "self_cuda_time_total", 0)
        or 0
    )


def _profiler_rows(prof: profile, top_k: int) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for event in prof.key_averages():
        device_us = _event_device_us(event)
        cpu_us = int(getattr(event, "cpu_time_total", 0) or 0)
        if device_us <= 0 and cpu_us <= 0:
            continue
        rows.append(
            {
                "name": event.key,
                "device_us": device_us,
                "self_device_us": _event_self_device_us(event),
                "cpu_us": cpu_us,
                "count": event.count,
                "cuda_mem_mb": getattr(event, "cuda_memory_usage", 0) / (1024 * 1024),
            }
        )
    rows.sort(key=lambda row: (row["device_us"], row["cpu_us"]), reverse=True)
    return rows[:top_k]


def _kernel_rows(prof: profile, top_k: int) -> list[dict[str, float | str | int]]:
    totals: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"device_us": 0.0, "count": 0}
    )
    for event in prof.events():
        if event.device_type != torch.profiler.DeviceType.CUDA:
            continue
        name = event.name or "unknown"
        totals[name]["device_us"] += _event_device_us(event)
        totals[name]["count"] += 1

    rows = [
        {"name": name, "device_us": data["device_us"], "count": int(data["count"])}
        for name, data in totals.items()
    ]
    rows.sort(key=lambda row: row["device_us"], reverse=True)
    return rows[:top_k]


def _print_profiler_table(
    title: str,
    rows: list[dict[str, float | str | int]],
    *,
    time_key: str = "device_us",
) -> None:
    if not rows:
        print(f"\n{title}: no profiler events collected")
        return

    total_device_us = sum(float(row[time_key]) for row in rows)
    print(f"\n{title}")
    print(f"  {'name':<52} {'device_ms':>10} {'share':>7} {'count':>7}")
    for row in rows:
        device_ms = float(row[time_key]) / 1000.0
        share = 100.0 * float(row[time_key]) / total_device_us if total_device_us else 0.0
        print(
            f"  {str(row['name'])[:52]:<52} "
            f"{device_ms:10.3f} "
            f"{share:6.1f}% "
            f"{int(row['count']):7d}"
        )


@torch.no_grad()
def run_profiler(
    name: str,
    fn,
    profile_dir: Path,
    warmup: int,
    iters: int,
    top_k: int,
) -> profile:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    profile_dir.mkdir(parents=True, exist_ok=True)
    trace_path = profile_dir / f"{name}.json"

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(iters):
            fn()

    print(f"\n=== Profiler: {name} ===")
    print(prof.key_averages().table(sort_by="device_time_total", row_limit=top_k))
    _print_profiler_table("Top CUDA kernels", _kernel_rows(prof, top_k))
    _print_profiler_table(
        "Top ops (inclusive device time)",
        _profiler_rows(prof, top_k),
    )

    prof.export_chrome_trace(str(trace_path))
    print(f"Chrome trace: {trace_path}")
    print("Open chrome://tracing or https://ui.perfetto.dev to inspect timelines.")

    summary_path = profile_dir / f"{name}_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "name": name,
                "top_ops": _profiler_rows(prof, top_k),
                "top_kernels": _kernel_rows(prof, top_k),
            },
            indent=2,
        )
    )
    print(f"Summary JSON: {summary_path}")
    return prof


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark convolution gate CUDA kernel vs PyTorch"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--kernel-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=("default", "reduce-overhead", "max-autotune"),
    )
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--dtype", type=str, default="float16", choices=("float16",))
    parser.add_argument(
        "--scale",
        type=float,
        default=0.1,
        help="Random tensor scale (keeps fp16 values in range for correctness checks)",
    )
    parser.add_argument(
        "--nsight",
        action="store_true",
        help="Delimit selected steady-state kernels for Nsight Systems/Compute",
    )
    parser.add_argument(
        "--nsight-target",
        default="cuda",
        choices=("compile", "eager", "cuda", "graph", "all"),
        help=(
            "Target to capture (default: cuda); 'graph' replays the captured custom "
            "extension and 'all' gives each implementation its own NVTX range"
        ),
    )
    parser.add_argument("--nsight-warmup", type=int, default=10)
    parser.add_argument("--nsight-iters", type=int, default=1)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Collect torch profiler traces and stage-level CUDA timings",
    )
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Skip end-to-end benchmark timing and only run profiling",
    )
    parser.add_argument("--profile-warmup", type=int, default=3)
    parser.add_argument("--profile-iters", type=int, default=10)
    parser.add_argument("--profile-top-k", type=int, default=15)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=ROOT / "profile_output",
        help="Directory for chrome traces and profiler summaries",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.nsight_warmup < 0:
        raise ValueError("--nsight-warmup must be non-negative")
    if args.nsight_iters < 1:
        raise ValueError("--nsight-iters must be at least 1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    device = torch.device("cuda")
    dtype = torch.float16

    print("Loading CUDA extension...")
    t0 = time.perf_counter()
    extension = load_extension()
    print(f"Extension loaded in {time.perf_counter() - t0:.2f}s")

    tensors = make_tensors(
        args.batch_size,
        args.seq_len,
        args.dim,
        args.kernel_size,
        device,
        dtype,
        scale=args.scale,
    )
    x, linear1_weight, linear1_bias, conv_weight, conv_bias, linear2_weight, linear2_bias = (
        tensors
    )

    # Prepack once for the custom CUDA extension: [D, 1, K] -> [K, D].
    # Keep conv_weight unchanged for the PyTorch reference implementation.
    conv_weight_kd = conv_weight.squeeze(1).transpose(0, 1).contiguous()

    module = ConvolutionGateModule(
        args.dim,
        args.kernel_size,
        linear1_weight=linear1_weight,
        linear1_bias=linear1_bias,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        linear2_weight=linear2_weight,
        linear2_bias=linear2_bias,
    ).eval()

    def run_pytorch() -> torch.Tensor:
        return module(x)

    def run_cuda() -> torch.Tensor:
        return extension.forward(
            x,
            linear1_weight,
            linear1_bias,
            conv_weight_kd,
            conv_bias,
            linear2_weight,
            linear2_bias,
        )

    print(
        f"\nConfig: B={args.batch_size}, T={args.seq_len}, D={args.dim}, "
        f"K={args.kernel_size}, dtype={args.dtype}"
    )
    print(f"GPU: {torch.cuda.get_device_name(device)}\n")

    ref = run_pytorch()
    cuda_out = run_cuda()
    check_correctness(ref, cuda_out, "CUDA kernel")

    print("Capturing custom CUDA extension with torch.cuda.CUDAGraph...")
    captured_cuda = capture_cuda_extension_graph(
        extension,
        x,
        linear1_weight,
        linear1_bias,
        conv_weight_kd,
        conv_bias,
        linear2_weight,
        linear2_bias,
    )

    def run_cuda_graph() -> torch.Tensor:
        # Replay-only timing: static_input already contains x. For changing
        # inputs, call captured_cuda.copy_and_replay(new_x). The input copy is
        # excluded here to isolate the reduction in CPU launch overhead.
        return captured_cuda.replay()

    graph_out = run_cuda_graph()
    check_correctness(ref, graph_out, "CUDA Graph")

    compiled = None
    needs_compiled = (
        not args.skip_compile
        and (not args.nsight or args.nsight_target in ("compile", "all"))
    )
    if needs_compiled:
        compiled = torch.compile(module, mode=args.compile_mode)
        # Trigger Dynamo/Inductor and autotuning before starting Nsight capture.
        compile_out = compiled(x)
        check_correctness(ref, compile_out, f"torch.compile({args.compile_mode})")

    def run_compiled() -> torch.Tensor:
        if compiled is None:
            raise RuntimeError("torch.compile target requested but compilation is disabled")
        return compiled(x)

    if args.nsight:
        if args.nsight_target in ("compile", "all") and compiled is None:
            raise ValueError("compile/all cannot be combined with --skip-compile")
        target_map = {
            "eager": ("pytorch_eager", run_pytorch),
            "compile": (f"torch_compile_{args.compile_mode}", run_compiled),
            "cuda": ("custom_cuda_extension", run_cuda),
            "graph": ("custom_cuda_cudagraph", run_cuda_graph),
        }
        targets = (
            [target_map[name] for name in ("eager", "compile", "cuda", "graph")]
            if args.nsight_target == "all"
            else [target_map[args.nsight_target]]
        )
        run_nsight_capture(targets, args.nsight_warmup, args.nsight_iters)
        return

    workload = WorkloadStats(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        dim=args.dim,
        kernel_size=args.kernel_size,
    )

    pytorch_stats = None
    cuda_stats = None
    graph_stats = None
    compile_stats = None

    if not args.profile_only:
        print("\nBenchmarking...")
        pytorch_stats = summarize(
            "pytorch_eager",
            benchmark(run_pytorch, args.warmup, args.iters),
        )
        cuda_stats = summarize(
            "cuda_kernel",
            benchmark(run_cuda, args.warmup, args.iters),
        )
        graph_stats = summarize(
            "cuda_graph",
            benchmark(run_cuda_graph, args.warmup, args.iters),
        )

        if compiled is not None:
            compile_stats = summarize(
                "torch.compile",
                benchmark(run_compiled, args.warmup, args.iters),
            )

        print("\nSpeedup vs PyTorch eager:")
        print(f"  cuda_kernel: {pytorch_stats['mean_ms'] / cuda_stats['mean_ms']:.2f}x")
        print(f"  cuda_graph:  {pytorch_stats['mean_ms'] / graph_stats['mean_ms']:.2f}x")
        print(
            f"  graph vs uncaptured CUDA: "
            f"{cuda_stats['mean_ms'] / graph_stats['mean_ms']:.2f}x"
        )
        if compile_stats is not None:
            print(
                f"  torch.compile: {pytorch_stats['mean_ms'] / compile_stats['mean_ms']:.2f}x"
            )
            print(
                f"  cuda_kernel vs torch.compile: "
                f"{compile_stats['mean_ms'] / cuda_stats['mean_ms']:.2f}x"
            )

    if args.profile or args.profile_only:
        print("\n" + "=" * 72)
        print("Profiling")
        print("=" * 72)

        print_workload_analysis(
            workload,
            dtype,
            pytorch_stats["mean_ms"] if pytorch_stats else None,
        )

        stages = profile_pytorch_stages(
            x,
            linear1_weight,
            linear1_bias,
            conv_weight,
            conv_bias,
            linear2_weight,
            linear2_bias,
            args.profile_warmup,
            args.profile_iters,
        )
        print_stage_breakdown(
            stages,
            pytorch_stats["mean_ms"] if pytorch_stats else None,
        )

        run_profiler(
            "pytorch_eager",
            run_pytorch,
            args.profile_dir,
            args.profile_warmup,
            args.profile_iters,
            args.profile_top_k,
        )
        run_profiler(
            "cuda_kernel",
            run_cuda,
            args.profile_dir,
            args.profile_warmup,
            args.profile_iters,
            args.profile_top_k,
        )
        run_profiler(
            "cuda_graph",
            run_cuda_graph,
            args.profile_dir,
            args.profile_warmup,
            args.profile_iters,
            args.profile_top_k,
        )
        if compiled is not None:
            run_profiler(
                "torch_compile",
                lambda: compiled(x),
                args.profile_dir,
                args.profile_warmup,
                args.profile_iters,
                args.profile_top_k,
            )

        print("\nProfiling tips:")
        print("  - If linear1/linear2 dominate, optimize CUTLASS tile sizes or fuse bias/epilogue.")
        print("  - If gate kernels are visible but small, GEMM is the bottleneck.")
        print("  - Compare chrome traces side-by-side to spot extra allocations/memcpys.")
        print("  - In cuda_kernel trace, look for CUTLASS, convolution_gate_kernel, add_bias_kernel.")


if __name__ == "__main__":
    main()
