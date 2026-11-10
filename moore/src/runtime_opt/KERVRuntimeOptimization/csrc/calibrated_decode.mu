#include <musa_runtime.h>
#include <musa_bf16.h>
#include <stdint.h>

// Eight adjacent weights share a group scale. Protected groups bypass the
// integer path and retain their original BF16 bit patterns.
template<bool Identity>
__global__ void decode8(const int8_t* __restrict__ q,
                        const float* __restrict__ scales,
                        const float* __restrict__ inverse,
                        const int* __restrict__ slots,
                        const uint16_t* __restrict__ protected_values,
                        uint16_t* __restrict__ out,
                        int k, int group, int ng, int np) {
  int col = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  int row = blockIdx.y;
  if (col >= k) return;
  int g = col / group;
  int slot = slots[g];
  uint4 result;
  if (slot >= 0) {
    result = *reinterpret_cast<const uint4*>(protected_values +
        (int64_t(row) * np + slot) * group + col % group);
  } else {
    union { uint2 packed; int8_t values[8]; } bytes;
    union { uint4 packed; uint16_t values[8]; } bits;
    bytes.packed = *reinterpret_cast<const uint2*>(q + int64_t(row) * k + col);
    float scale = scales[int64_t(row) * ng + g];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      float value = float(bytes.values[i]) * scale;
      if (!Identity) value = value * inverse[col + i];
      bits.values[i] = __bfloat16_as_ushort(__float2bfloat16_rn(value));
    }
    result = bits.packed;
  }
  *reinterpret_cast<uint4*>(out + int64_t(row) * k + col) = result;
}

extern "C" int kerv_decode8(const void* q, const void* scales,
                            const void* inverse, const void* slots,
                            const void* values, void* out,
                            int n, int k, int group, int np, int identity,
                            int threads, void* stream) {
  if (n <= 0 || k <= 0 || k % 8 || group <= 0 || group % 8 ||
      (threads != 64 && threads != 128 && threads != 256))
    return int(musaErrorInvalidValue);
  dim3 grid((k / 8 + threads - 1) / threads, n);
  int ng = (k + group - 1) / group;
  #define LAUNCH(ID) decode8<ID><<<grid, threads, 0, (musaStream_t)stream>>>( \
    (const int8_t*)q, (const float*)scales, (const float*)inverse, \
    (const int*)slots, (const uint16_t*)values, (uint16_t*)out, k, group, ng, np)
  if (identity) { LAUNCH(true); } else { LAUNCH(false); }
  #undef LAUNCH
  return int(musaGetLastError());
}
