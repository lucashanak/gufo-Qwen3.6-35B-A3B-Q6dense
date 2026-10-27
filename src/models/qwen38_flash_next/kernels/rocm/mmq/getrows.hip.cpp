// SPDX-License-Identifier: MIT
// Embedding rows of any GGML block format, dequantized element for element
// as ggml-quants.c's dequantize_row_* (and gufo::quant::DequantizeRow): the
// same products in the same order. This file is built without fast-math so
// no product is fused or reassociated. See VENDOR.md.
#include "qfn_mmq_prelude.h"
namespace qfn_mmq {
#include "qfn_mmq.h"
#include "common.hpp"

namespace {

__device__ __forceinline__ float F16(ggml_half h) {
  return __half2float(*reinterpret_cast<const __half*>(&h));
}

__device__ __forceinline__ float Sign(uint8_t signs, int j) {
  return (signs & (1u << j)) != 0 ? -1.0f : 1.0f;
}

/// Element j of iq1s_grid entry `index` (iq1s_grid_gpu stores value + 1).
__device__ __forceinline__ float Iq1sGrid(uint32_t index, int j) {
  const int nibble = 2 * (j % 4) + j / 4;
  return float(int((iq1s_grid_gpu[index] >> (4 * nibble)) & 0x0Fu) - 1);
}

__device__ __forceinline__ void Q4KScaleMin(int j, const uint8_t* q,
                                            uint8_t& d, uint8_t& m) {
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

/// Element i (0 <= i < the format's block size) of one block.
template<ggml_type type>
__device__ float Element(const void* vblock, int i);

template<>
__device__ float Element<GGML_TYPE_Q4_0>(const void* v, int i) {
  const auto* b = static_cast<const block_q4_0*>(v);
  const int q = i < 16 ? (b->qs[i] & 0x0F) : (b->qs[i - 16] >> 4);
  return float(q - 8) * F16(b->d);
}

template<>
__device__ float Element<GGML_TYPE_Q4_1>(const void* v, int i) {
  const auto* b = static_cast<const block_q4_1*>(v);
  const int q = i < 16 ? (b->qs[i] & 0x0F) : (b->qs[i - 16] >> 4);
  const half2 dm = b->dm;
  return float(q) * __low2float(dm) + __high2float(dm);
}

template<>
__device__ float Element<GGML_TYPE_Q5_0>(const void* v, int i) {
  const auto* b = static_cast<const block_q5_0*>(v);
  uint32_t qh;
  memcpy(&qh, b->qh, sizeof(qh));
  const int j = i % 16;
  const int q = i < 16 ? ((b->qs[j] & 0x0F) | int(((qh >> j) << 4) & 0x10))
                       : ((b->qs[j] >> 4) | int((qh >> (j + 12)) & 0x10));
  return float(q - 16) * F16(b->d);
}

template<>
__device__ float Element<GGML_TYPE_Q5_1>(const void* v, int i) {
  const auto* b = static_cast<const block_q5_1*>(v);
  uint32_t qh;
  memcpy(&qh, b->qh, sizeof(qh));
  const int j = i % 16;
  const int q = i < 16 ? ((b->qs[j] & 0x0F) | int(((qh >> j) & 1u) << 4))
                       : ((b->qs[j] >> 4) | int(((qh >> (j + 16)) & 1u) << 4));
  const half2 dm = b->dm;
  return float(q) * __low2float(dm) + __high2float(dm);
}

template<>
__device__ float Element<GGML_TYPE_Q8_0>(const void* v, int i) {
  const auto* b = static_cast<const block_q8_0*>(v);
  return F16(b->d) * float(b->qs[i]);
}

template<>
__device__ float Element<GGML_TYPE_Q2_K>(const void* v, int i) {
  const auto* b = static_cast<const block_q2_K*>(v);
  // 128-element halves; within one, eight runs of 16 alternate between the
  // half's two 16-byte rows and step the shift every two runs.
  const int n = i / 128, r = (i % 128) / 16, l = i % 16;
  const int shift = 2 * (r / 2);
  const uint8_t sc = b->scales[8 * n + r];
  const float d = __low2float(b->dm), min = __high2float(b->dm);
  const float dl = d * float(sc & 0xF);
  const float ml = min * float(sc >> 4);
  const uint8_t q = b->qs[32 * n + 16 * (r % 2) + l];
  return dl * float((q >> shift) & 3) - ml;
}

template<>
__device__ float Element<GGML_TYPE_Q3_K>(const void* v, int i) {
  const auto* b = static_cast<const block_q3_K*>(v);
  const int half = i / 128, within = i % 128;
  const int pair = within / 32, lane = within % 32;
  const int low = (b->qs[32 * half + lane] >> (2 * pair)) & 3;
  const int quant =
      low - ((b->hmask[lane] & (1 << (4 * half + pair))) != 0 ? 0 : 4);
  const int index = 8 * half + 2 * pair + lane / 16;
  const int lo = index < 8 ? b->scales[index] & 0xF
                           : (b->scales[index - 8] >> 4) & 0xF;
  const int hi = (b->scales[8 + index % 4] >> (2 * (index / 4))) & 3;
  const int scale = int(int8_t(lo | (hi << 4))) - 32;
  return F16(b->d) * float(scale) * float(quant);
}

template<>
__device__ float Element<GGML_TYPE_Q4_K>(const void* v, int i) {
  const auto* b = static_cast<const block_q4_K*>(v);
  const int group = i / 32;
  const uint8_t packed = b->qs[32 * (group / 2) + i % 32];
  const int quant = (group & 1) ? packed >> 4 : packed & 0x0F;
  uint8_t sc, m;
  Q4KScaleMin(group, b->scales, sc, m);
  const float d = __low2float(b->dm), dmin = __high2float(b->dm);
  return d * float(sc) * float(quant) - dmin * float(m);
}

template<>
__device__ float Element<GGML_TYPE_Q5_K>(const void* v, int i) {
  const auto* b = static_cast<const block_q5_K*>(v);
  const int gg = i / 64, wv = i % 64, lane = wv % 32;
  const bool lo = wv < 32;
  const uint8_t qb = b->qs[32 * gg + lane];
  const int bit = 2 * gg + (lo ? 0 : 1);
  const int quant = (lo ? (qb & 0x0F) : (qb >> 4)) +
                    (((b->qh[lane] >> bit) & 1) ? 16 : 0);
  uint8_t sc, m;
  Q4KScaleMin(bit, b->scales, sc, m);
  const float d = __low2float(b->dm), dmin = __high2float(b->dm);
  return d * float(sc) * float(quant) - dmin * float(m);
}

template<>
__device__ float Element<GGML_TYPE_Q6_K>(const void* v, int i) {
  const auto* b = static_cast<const block_q6_K*>(v);
  const int half = i / 128, within = i % 128;
  const int segment = within / 32, lane = within % 32;
  const uint8_t qh = b->qh[32 * half + lane];
  const uint8_t ql = b->ql[64 * half + 32 * (segment & 1) + lane];
  const int low = segment < 2 ? (ql & 0x0F) : (ql >> 4);
  const int high = (qh >> (2 * segment)) & 3;
  const int quant = ((high << 4) | low) - 32;
  return F16(b->d) * float(b->scales[8 * half + lane / 16 + 2 * segment]) *
         float(quant);
}

template<>
__device__ float Element<GGML_TYPE_IQ2_XXS>(const void* v, int i) {
  const auto* b = static_cast<const block_iq2_xxs*>(v);
  const int ib32 = i / 32, l = (i % 32) / 8, j = i % 8;
  uint32_t aux[2];
  memcpy(aux, b->qs + 4 * ib32, sizeof(aux));
  const float db = F16(b->d) * (0.5f + float(aux[1] >> 28)) * 0.25f;
  const auto* grid = reinterpret_cast<const uint8_t*>(
      iq2xxs_grid + ((aux[0] >> (8 * l)) & 0xFF));
  const uint8_t signs = ksigns_iq2xs[(aux[1] >> (7 * l)) & 127];
  return db * float(grid[j]) * Sign(signs, j);
}

template<>
__device__ float Element<GGML_TYPE_IQ2_XS>(const void* v, int i) {
  const auto* b = static_cast<const block_iq2_xs*>(v);
  const int ib32 = i / 32, l = (i % 32) / 8, j = i % 8;
  const uint8_t s = b->scales[ib32];
  const float db = F16(b->d) * (0.5f + float(l < 2 ? s & 0xF : s >> 4)) * 0.25f;
  const uint16_t qs = b->qs[4 * ib32 + l];
  const auto* grid = reinterpret_cast<const uint8_t*>(iq2xs_grid + (qs & 511));
  return db * float(grid[j]) * Sign(ksigns_iq2xs[qs >> 9], j);
}

template<>
__device__ float Element<GGML_TYPE_IQ2_S>(const void* v, int i) {
  const auto* b = static_cast<const block_iq2_s*>(v);
  const int ib32 = i / 32, l = (i % 32) / 8, j = i % 8;
  const uint8_t s = b->scales[ib32];
  const float db = F16(b->d) * (0.5f + float(l < 2 ? s & 0xF : s >> 4)) * 0.25f;
  const uint32_t index =
      b->qs[4 * ib32 + l] | ((uint32_t(b->qh[ib32]) << (8 - 2 * l)) & 0x300);
  const auto* grid = reinterpret_cast<const uint8_t*>(iq2s_grid + index);
  return db * float(grid[j]) * Sign(b->qs[QK_K / 8 + 4 * ib32 + l], j);
}

template<>
__device__ float Element<GGML_TYPE_IQ3_XXS>(const void* v, int i) {
  const auto* b = static_cast<const block_iq3_xxs*>(v);
  const int ib32 = i / 32, l = (i % 32) / 8, j = i % 8;
  uint32_t aux;
  memcpy(&aux, b->qs + QK_K / 4 + 4 * ib32, sizeof(aux));
  const float db = F16(b->d) * (0.5f + float(aux >> 28)) * 0.5f;
  const uint8_t signs = ksigns_iq2xs[(aux >> (7 * l)) & 127];
  const auto* grid = reinterpret_cast<const uint8_t*>(
      iq3xxs_grid + b->qs[8 * ib32 + 2 * l + j / 4]);
  return db * float(grid[j % 4]) * Sign(signs, j);
}

template<>
__device__ float Element<GGML_TYPE_IQ3_S>(const void* v, int i) {
  const auto* b = static_cast<const block_iq3_s*>(v);
  const int ib32 = i / 32, l = (i % 32) / 8, j = i % 8;
  const int half = j / 4;
  const uint32_t index =
      b->qs[8 * ib32 + 2 * l + half] |
      ((uint32_t(b->qh[ib32]) << (8 - 2 * l - half)) & 256);
  const auto* grid = reinterpret_cast<const uint8_t*>(iq3s_grid + index);
  const uint8_t s = b->scales[ib32 / 2];
  const float db =
      F16(b->d) * float(1 + 2 * ((ib32 % 2) == 0 ? (s & 0xF) : (s >> 4)));
  const float magnitude = float(grid[j % 4]);
  return (b->signs[4 * ib32 + l] & (1u << j)) != 0 ? -db * magnitude
                                                    : db * magnitude;
}

template<>
__device__ float Element<GGML_TYPE_IQ1_S>(const void* v, int i) {
  const auto* b = static_cast<const block_iq1_s*>(v);
  const int ib = i / 32, l = (i % 32) / 8, j = i % 8;
  const uint16_t qh = b->qh[ib];
  const float dl = F16(b->d) * float(2 * ((qh >> 12) & 7) + 1);
  const float delta = (qh & 0x8000) != 0 ? -IQ1S_DELTA : IQ1S_DELTA;
  const uint32_t index = b->qs[4 * ib + l] | (((qh >> (3 * l)) & 7u) << 8);
  return dl * (Iq1sGrid(index, j) + delta);
}

template<>
__device__ float Element<GGML_TYPE_IQ1_M>(const void* v, int i) {
  const auto* b = static_cast<const block_iq1_m*>(v);
  const int ib = i / 32, l = (i % 32) / 8, j = i % 8;
  uint16_t sc[4];
  memcpy(sc, b->scales, sizeof(sc));
  const uint16_t packed = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) |
                          ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
  const float d = __half2float(*reinterpret_cast<const __half*>(&packed));
  const int shift = 6 * (ib % 2) + (l < 2 ? 0 : 3);
  const float dl = d * float(2 * ((sc[ib / 2] >> shift) & 0x7) + 1);
  const uint8_t qh = b->qh[2 * ib + l / 2];
  const uint32_t index =
      b->qs[4 * ib + l] | ((uint32_t(qh) << (l % 2 == 0 ? 8 : 4)) & 0x700);
  const float delta =
      (qh & (l % 2 == 0 ? 0x08 : 0x80)) != 0 ? -IQ1M_DELTA : IQ1M_DELTA;
  return dl * (Iq1sGrid(index, j) + delta);
}

template<>
__device__ float Element<GGML_TYPE_IQ4_NL>(const void* v, int i) {
  const auto* b = static_cast<const block_iq4_nl*>(v);
  const int q = i < 16 ? (b->qs[i] & 0x0F) : (b->qs[i - 16] >> 4);
  return F16(b->d) * float(kvalues_iq4nl[q]);
}

template<>
__device__ float Element<GGML_TYPE_IQ4_XS>(const void* v, int i) {
  const auto* b = static_cast<const block_iq4_xs*>(v);
  const int ib = i / 32, within = i % 32;
  const uint8_t packed = b->qs[16 * ib + within % 16];
  const int q = within < 16 ? (packed & 0x0F) : (packed >> 4);
  const int ls = ((b->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) |
                 (((b->scales_h >> (2 * ib)) & 3) << 4);
  return F16(b->d) * float(ls - 32) * float(kvalues_iq4nl[q]);
}

/// res[t][s][k] = table[tokens[t]][k] for every stream s.
template<ggml_type type>
__global__ void GetRowsKernel(const char* __restrict__ table,
                              size_t row_bytes, int qk, int block_bytes,
                              const int32_t* __restrict__ tokens,
                              float* __restrict__ res, int k, int streams) {
  const int t = blockIdx.x;
  const char* row = table + row_bytes * size_t(tokens[t]);
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    const float v =
        Element<type>(row + size_t(i / qk) * block_bytes, i % qk);
    for (int s = 0; s < streams; ++s)
      res[(size_t(t) * streams + s) * k + i] = v;
  }
}

template<ggml_type type>
void Launch(const void* table, const int32_t* tokens, float* res, int n,
            int k, int streams, hipStream_t stream) {
  const int qk = int(ggml_blck_size(type));
  const int block_bytes = int(ggml_type_size(type));
  const size_t row_bytes = size_t(k / qk) * block_bytes;
  GetRowsKernel<type><<<n, 256, 0, stream>>>(static_cast<const char*>(table),
                                             row_bytes, qk, block_bytes,
                                             tokens, res, k, streams);
}

}  // namespace

extern "C" int qfn_mmq_get_rows(int weight_type, const void* table,
                                const int32_t* tokens, float* res,
                                int n_tokens, int k, int streams,
                                hipStream_t stream) {
  const auto type = static_cast<ggml_type>(weight_type);
  if (!table || !tokens || !res || n_tokens <= 0 || k <= 0 || streams <= 0)
    return -1;
  switch (type) {
#define QFN_GET_ROWS(t)                                         \
  case t:                                                       \
    if (k % ggml_blck_size(t) != 0)                             \
      return -1;                                                \
    Launch<t>(table, tokens, res, n_tokens, k, streams, stream); \
    break;
    QFN_GET_ROWS(GGML_TYPE_Q4_0)
    QFN_GET_ROWS(GGML_TYPE_Q4_1)
    QFN_GET_ROWS(GGML_TYPE_Q5_0)
    QFN_GET_ROWS(GGML_TYPE_Q5_1)
    QFN_GET_ROWS(GGML_TYPE_Q8_0)
    QFN_GET_ROWS(GGML_TYPE_Q2_K)
    QFN_GET_ROWS(GGML_TYPE_Q3_K)
    QFN_GET_ROWS(GGML_TYPE_Q4_K)
    QFN_GET_ROWS(GGML_TYPE_Q5_K)
    QFN_GET_ROWS(GGML_TYPE_Q6_K)
    QFN_GET_ROWS(GGML_TYPE_IQ2_XXS)
    QFN_GET_ROWS(GGML_TYPE_IQ2_XS)
    QFN_GET_ROWS(GGML_TYPE_IQ2_S)
    QFN_GET_ROWS(GGML_TYPE_IQ3_XXS)
    QFN_GET_ROWS(GGML_TYPE_IQ3_S)
    QFN_GET_ROWS(GGML_TYPE_IQ1_S)
    QFN_GET_ROWS(GGML_TYPE_IQ1_M)
    QFN_GET_ROWS(GGML_TYPE_IQ4_NL)
    QFN_GET_ROWS(GGML_TYPE_IQ4_XS)
#undef QFN_GET_ROWS
    default:
      return -1;
  }
  return hipGetLastError() == hipSuccess ? 0 : -2;
}

}  // namespace qfn_mmq
