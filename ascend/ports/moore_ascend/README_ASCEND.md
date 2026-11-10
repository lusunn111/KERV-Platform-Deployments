# Moore runtime port for Ascend

This directory preserves the Moore implementation as an isolated starting point
and adds NPU device support and an explicit seven-token action protocol.
Use the host scripts in the project root:

```bash
KERV_NPU_ID=4 KERV_MOORE_PROFILE=optimized scripts/run_kerv_moore_ascend.sh
KERV_NPU_ID=4 scripts/benchmark_kerv_moore_ascend.sh
```

The default host launcher selects `KERV_ACTION_PROTOCOL=seven`; `moore` retains
the archive's original KF slicing behavior for diagnostics. The validated profile
uses original F.rms_norm, shared embedding, full verification paths, stable ties,
C++ trees, batched Kalman/KV commits, and persistent buffers. Graphs and numerical
fusion candidates are off. `--rope` in the benchmark is experimental.

Latest evidence, environment and reproduction: [README_ASCEND.md](../../README_ASCEND.md).
Archive comparison: [MOORE_ASCEND_COMPARISON.md](../../docs/MOORE_ASCEND_COMPARISON.md).
The inherited Moore README describes MUSA measurements, not new Ascend results.
