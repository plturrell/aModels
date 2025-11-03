# Gemma/Vault Inference Roadmap

This document tracks the remaining work required to turn VaultGemma into a fully functioning inference backend.

## Milestone 1 – Loader Performance (✅)
- Stream tensor weights in large chunks to remove per-element reads.
- Add regression tests/benchmarks for the loader.

## Milestone 2 – Rotary Embeddings & Attention (🚧)
- [ ] Implement rotary positional embeddings for Q/K projections.
- [ ] Port the tensor/attention primitives from `agenticAiETH_layer1_Blockchain/infrastructure/maths` and integrate them with the model layers.
- [ ] Validate attention output against the reference PyTorch implementation for a fixed prompt.

## Milestone 3 – Feed-forward & Activation Fidelity
- [ ] Ensure SwiGLU matches the exact Gemma activation function.
- [ ] Audit RMSNorm implementation for numercial parity with HF.

## Milestone 4 – Sampling & Generation
- [ ] Implement temperature / top-p / top-k sampling controls.
- [ ] Support KV cache reuse for autoregressive decoding.
- [ ] Add benchmarks and golden tests covering end-to-end token generation.

## Milestone 5 – Optimisation & Instrumentation
- [ ] Add profiling hooks and lightweight metrics.
- [ ] Investigate SIMD-backed matmul hot paths and caching opportunities.

Each milestone should land with its own tests and benchmarking notes so we can track accuracy and performance regressions.
