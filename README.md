# CUDA Kernel Profiling Studies
This repository collects profiling notes, benchmark summaries, and optimization studies from CUDA kernels implemented from scratch: FP32 GEMM, WMMA Tensor Core GEMM, reductions, fused softmax, and FlashAttention-lite.

The goal is to study how kernel structure, memory hierarchy, Tensor Core usage, and Nsight Compute metrics explain performance relative to vendor baselines.

## Benchmark Summary

| Kernel | Main optimization focus | Baseline | Result | Hardware | Profiling |
|---|---|---:|---:|---|---|
| FP32 GEMM | Shared-memory tiling, coalescing, register blocking, bank-conflict reduction | cuBLAS | 73% of cuBLAS | RTX 3050 | Nsight Compute |
| WMMA Tensor Core GEMM | FP16 input, FP32 accumulate, Tensor Cores, shared-memory tiling, pipelining | cuBLAS | 82% of cuBLAS | RTX 3050 | Nsight Compute |
| Parallel Reduction | Grid-stride loading, warp shuffles, memory-bandwidth tuning | Peak DRAM bandwidth | 97% of peak DRAM bandwidth | RTX 3050 | Nsight Compute |
| Fused Softmax | Row-wise fused softmax, warp/shared-memory reductions, online softmax | cuDNN | 90% of cuDNN | RTX 3050 | Nsight Compute |
| FlashAttention-lite | Single-head fused attention path, fused softmax, occupancy tuning | cuDNN SDPA | 85% of cuDNN SDPA | RTX 3050 | Nsight Compute |
| Sparse Binary 2D FFT | Streaming tiles, memory-footprint reduction, Hermitian symmetry | Dense cuFFT workflow | Memory-efficient sparse FFT study | RTX 3050 / A4000 | Nsight Compute |
