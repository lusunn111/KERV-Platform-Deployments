# KERV Runtime Operators

The package registers 19 KERV-specific interfaces (including Ascend verifier
`kerv_rms_norm`) in
`torch.ops.flagos_embodied`. Fifteen interfaces are enabled by the default
include list; value-cache store is optional and three experimental interfaces
remain opt-in. The source
of truth for names, categories and selection policy is
`operator_manifest.json`.

Every interface has a native ATen fallback. On Ascend use `backend=native`;
experimental CANN RMSNorm in `auto` has not passed action-token parity.
On CUDA, `backend=auto` selects a Triton or
fused path only for supported layouts and dtypes; unsupported inputs preserve
the reference implementation.
