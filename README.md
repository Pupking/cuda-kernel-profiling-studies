# CUDA Kernel Profiling Studies
This repository collects profiling notes, benchmark summaries, and optimization studies from CUDA kernels implemented from scratch: FP32 GEMM, WMMA Tensor Core GEMM, reductions, fused softmax, and FlashAttention-lite.

The goal is to study how kernel structure, memory hierarchy, Tensor Core usage, and Nsight Compute metrics explain performance relative to vendor baselines.
