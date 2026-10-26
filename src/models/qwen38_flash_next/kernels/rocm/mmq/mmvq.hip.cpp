#include "qfn_mmq_prelude.h"
namespace qfn_mmq {
#include "mmvq.hpp"
#include "unary.hpp"
#include "vecdotq.hpp"

// Each wave handles up to eight dense inputs for one weight row on gfx1151.
// `scatter` reduces the inputs' sums together and stores them from one lane
// each (same bits as reducing each sum on every lane and storing from lane 0).
template<int ncols_dst, bool has_gate, int token_waves = 1, bool ragged = false,
         bool scatter = false>
__launch_bounds__(32 * token_waves, 1) static __global__
    void mul_mat_vec_q8(const void* __restrict__ weights,
                        const void* __restrict__ gate,
                        const block_q8_1* __restrict__ input,
                        float* __restrict__ output, const uint32_t ncols_x,
                        const uint32_t nrows_x, const uint32_t stride_col_y,
                        const uint32_t valid_tokens = ncols_dst) {
  constexpr int qi = QI8_0;
  constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
  constexpr int blocks_per_iter = vdr * 32 / qi;
  const int lane = token_waves > 1 ? threadIdx.x % 32 : threadIdx.x;
  const int first_token = token_waves > 1 ? threadIdx.x / 32 * ncols_dst : 0;
  const int row = blockIdx.x;
  const int blocks_per_row = ncols_x / QK8_0;
  const int row_offset = row * blocks_per_row;
  const int kqs = vdr * (lane % (qi / vdr));
  float sum[ncols_dst] = {};
  float gate_sum[ncols_dst] = {};
  for (int kbx = lane / (qi / vdr); kbx < blocks_per_row;
       kbx += blocks_per_iter) {
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
      // The last wave may contain fewer than eight inputs. Its inactive
      // columns reuse a valid row; their results are never stored.
      const int token =
          ragged ? min(first_token + j, int(valid_tokens) - 1) : first_token + j;
      sum[j] += vec_dot_q8_0_q8_1(
          weights, input + token * stride_col_y + kbx,
          row_offset + kbx, kqs);
      if constexpr (has_gate) {
        gate_sum[j] += vec_dot_q8_0_q8_1(gate, input + j * stride_col_y + kbx,
                                         row_offset + kbx, kqs);
      }
    }
  }
  if constexpr (scatter) {
    static_assert(token_waves == 1 && !ragged);
    constexpr int shift = 5 - ScatterLevels(ncols_dst);
    // Both reductions run on the full wave; only the stores are per lane.
    float value = scatter_reduce_sum(sum, lane);
    float gate_value = 0.0f;
    if constexpr (has_gate)
      gate_value = scatter_reduce_sum(gate_sum, lane);
    const int token = lane >> shift;
    if ((lane & ((1 << shift) - 1)) == 0 && token < ncols_dst) {
      if constexpr (has_gate)
        value *= ggml_hip_op_silu_single(gate_value);
      output[token * nrows_x + row] = value;
    }
    return;
  }
#pragma unroll
  for (int j = 0; j < ncols_dst; ++j) {
    sum[j] = warp_reduce_sum<32>(sum[j]);
    if constexpr (has_gate)
      gate_sum[j] = warp_reduce_sum<32>(gate_sum[j]);
    if (lane == 0 && (token_waves == 1 || first_token + j < valid_tokens)) {
      float value = sum[j];
      if constexpr (has_gate)
        value *= ggml_hip_op_silu_single(gate_sum[j]);
      output[(first_token + j) * nrows_x + row] = value;
    }
  }
}

__device__ __forceinline__ float Q8Materialize(float v) {
  asm volatile("" : "+v"(v));
  return v;
}

// mul_mat_vec_q8 for one input over a few long rows (the draft block's
// 4-row HC inject, K 10240): a wave is a row's whole latency chain, so each
// lane loads `steps` of its blocks before the first product. The lane's
// blocks, products and order are the one-row kernel's, spelled out as it
// compiles them: d0 * sumi, then one fma with d1 into the sum.
template<int steps>
__launch_bounds__(32, 1) static __global__
    void mul_mat_vec_q8_ahead(const void* __restrict__ weights,
                              const block_q8_1* __restrict__ input,
                              float* __restrict__ output,
                              const uint32_t ncols_x) {
  constexpr int qi = QI8_0;
  constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
  constexpr int blocks_per_iter = vdr * 32 / qi;
  const int lane = threadIdx.x;
  const int row = blockIdx.x;
  const int blocks_per_row = ncols_x / QK8_0;
  const block_q8_0* bx =
      static_cast<const block_q8_0*>(weights) + row * blocks_per_row;
  const int kqs = vdr * (lane % (qi / vdr));
  float sum = 0.0f;
  for (int kb = lane / (qi / vdr); kb < blocks_per_row;
       kb += steps * blocks_per_iter) {
    int v[steps][vdr];
    int u[steps][vdr];
    half d0[steps];
    half d1[steps];
#pragma unroll
    for (int s = 0; s < steps; ++s) {
      const block_q8_0* w = bx + kb + s * blocks_per_iter;
      const block_q8_1* y = input + kb + s * blocks_per_iter;
#pragma unroll
      for (int i = 0; i < vdr; ++i) {
        v[s][i] = get_int_b2(w->qs, kqs + i);
        u[s][i] = get_int_b4(y->qs, kqs + i);
      }
      d0[s] = w->d;
      d1[s] = __low2half(y->ds);
    }
#pragma unroll
    for (int s = 0; s < steps; ++s) {
      int sumi = 0;
#pragma unroll
      for (int i = 0; i < vdr; ++i)
        sumi = ggml_hip_dp4a(v[s][i], u[s][i], sumi);
      const float p =
          Q8Materialize(__half2float(d0[s]) * static_cast<float>(sumi));
      sum = Q8Materialize(__builtin_fmaf(p, __half2float(d1[s]), sum));
    }
  }
  sum = warp_reduce_sum<32>(sum);
  if (lane == 0)
    output[row] = sum;
}

// Integer matrix products reuse each Q8 weight across up to 48 inputs.
// Keep four separate K8 sums per wave: merging them into a K32 integer sum
// would change the scalar kernel's rounded products, FMA chain and reduction.
template<int token_tiles, bool ragged, bool grouped = false>
__launch_bounds__(256) static __global__ void mul_mat_q8_decode_batch(
    const block_q8_0* __restrict__ weights,
    const block_q8_1* __restrict__ input, float* __restrict__ output,
    int k, int rows, int input_stride, int tokens) {
  if constexpr (grouped) {
    input += std::size_t{blockIdx.y} * token_tiles * 16 * input_stride;
    output += std::size_t{blockIdx.y} * token_tiles * 16 * rows;
  }
  using Int4 = __attribute__((ext_vector_type(4))) int;
  using Int8 = __attribute__((ext_vector_type(8))) int;
  __shared__ float partial[16 * 16 * 32];
  const int wave = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int sub = lane % 16;
  const int half = lane / 16;
  const int first_row = blockIdx.x * 16;
  const int blocks = k / QK8_0;
  float acc[token_tiles][4][8]{};
  for (int kb = wave; kb < blocks; kb += 8) {
    const auto* w = weights + min(first_row + sub, rows - 1) * blocks + kb;
    const float scale = __half2float(w->d);
    float scales[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
      scales[i] = __shfl(scale, 2 * i + half, 32);
    Int4 codes[4];
#pragma unroll
    for (int part = 0; part < 4; ++part)
      codes[part] = {get_int_b2(w->qs, 2 * part),
                     get_int_b2(w->qs, 2 * part + 1), 0, 0};
#pragma unroll
    for (int tile = 0; tile < token_tiles; ++tile) {
      const int token = ragged && tile == token_tiles - 1
                            ? min(tile * 16 + sub, tokens - 1)
                            : tile * 16 + sub;
      const auto* x = input + token * input_stride + kb;
      const float x_scale = __low2float(x->ds);
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        const Int4 values = {get_int_b4(x->qs, 2 * part),
                              get_int_b4(x->qs, 2 * part + 1), 0, 0};
        Int8 dots{};
        dots = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
            true, codes[part], true, values, dots, false);
#pragma unroll
        for (int i = 0; i < 8; ++i)
          acc[tile][part][i] =
              __fmaf_rn(x_scale, __fmul_rn(scales[i], float(dots[i])),
                        acc[tile][part][i]);
      }
    }
  }
  // Reuse the 32 KiB transpose for each token tile. One thread reduces each
  // output in the original lane-zero tree, avoiding duplicate wave sums.
#pragma unroll
  for (int tile = 0; tile < token_tiles; ++tile) {
#pragma unroll
    for (int part = 0; part < 4; ++part) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int out = sub * 16 + 2 * i + half;
        const int swizzle = (out ^ (out >> 4)) & 31;
        partial[out * 32 + ((wave * 4 + part) ^ swizzle)] = acc[tile][part][i];
      }
    }
    __syncthreads();
    const int out = threadIdx.x;
    const int swizzle = (out ^ (out >> 4)) & 31;
    float sums[32];
#pragma unroll
    for (int i = 0; i < 32; ++i)
      sums[i] = partial[out * 32 + (i ^ swizzle)];
#pragma unroll
    for (int delta = 16; delta; delta /= 2) {
#pragma unroll
      for (int i = 0; i < delta; ++i)
        sums[i] = __fadd_rn(sums[i], sums[i + delta]);
    }
    const int row = first_row + out % 16;
    const int token = tile * 16 + out / 16;
    if (row < rows && (!ragged || token < tokens))
      output[token * rows + row] = sums[0];
    __syncthreads();
  }
}

typedef float (*vec_dot_q_hip_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_hip_t get_vec_dot_q_hip(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        default:                return 1;
    }
}

// A lane's decoded operands of one gate/up weight block at the routed vector
// kernels' (block, iqs) position, reused by every input routed to the
// expert. Dot performs the format's vec_dot for one input: the same
// activation loads, integer sums and floating expression, so each input's
// result is the scalar kernel's.
template<ggml_type type> struct MoeFragment;

template<> struct MoeFragment<GGML_TYPE_Q3_K> {
  int vi[QR3_K];
  int sc[QR3_K];
  float d;

  MoeFragment() = default;
  __device__ __forceinline__ MoeFragment(const void* weights, int kbx,
                                         int iqs) {
    const block_q3_K* b = static_cast<const block_q3_K*>(weights) + kbx;
    const int bq8_offset = QR3_K * (iqs / (QI3_K / 2));
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);
    d = b->d;
    const int vl = get_int_b2(b->qs, iqs);
    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
    const int vh = ~get_int_b2(b->hmask, iqs % (QI3_K / 2)) >> bq8_offset;
#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
      const int isc = scale_offset + 2 * i;
      const int isc_low = isc % (QK_K / 32);
      const int sc_shift_low = 4 * (isc / (QK_K / 32));
      const int sc_low = (b->scales[isc_low] >> sc_shift_low) & 0xF;
      const int isc_high = isc % (QK_K / 64);
      const int sc_shift_high = 2 * (isc / (QK_K / 64));
      const int sc_high =
          ((b->scales[(QK_K / 32) + isc_high] >> sc_shift_high) & 3) << 4;
      sc[i] = (sc_low | sc_high) - 32;
      const int vil = (vl >> (2 * i)) & 0x03030303;
      const int vih = ((vh >> i) << 2) & 0x04040404;
      vi[i] = __vsubss4(vil, vih);
    }
  }

  __device__ __forceinline__ float Dot(const block_q8_1* y, int iqs) const {
    const int bq8_offset = QR3_K * (iqs / (QI3_K / 2));
    int u[QR3_K];
    float d8[QR3_K];
#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
      u[i] = get_int_b4(y[bq8_offset + i].qs, iqs % QI8_1);
      d8[i] = __low2float(y[bq8_offset + i].ds);
    }
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR3_K; ++i)
      sumf += d8[i] * __mul24(ggml_hip_dp4a(vi[i], u[i], 0), sc[i]);
    return d * sumf;
  }
};

template<> struct MoeFragment<GGML_TYPE_Q4_K> {
  Q4MoeFragment w;
  MoeFragment() = default;
  __device__ __forceinline__ MoeFragment(const void* weights, int kbx, int iqs)
      : w(weights, kbx, iqs) {}
  __device__ __forceinline__ float Dot(const block_q8_1* y, int iqs) const {
    return w.Dot(y, iqs);
  }
};

template<> struct MoeFragment<GGML_TYPE_Q5_K> {
  int vl[2];
  int vh[2];
  uint16_t aux[2];
  half2 dm;

  MoeFragment() = default;
  __device__ __forceinline__ MoeFragment(const void* weights, int kbx,
                                         int iqs) {
    const block_q5_K* b = static_cast<const block_q5_K*>(weights) + kbx;
    const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
    const int* ql = (const int*)(b->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int* qh = (const int*)(b->qh + 4 * ((iqs / 2) % 4));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;
    const uint16_t* scales = (const uint16_t*)b->scales;
    const int j = bq8_offset / 2;
    if (j < 2) {
      aux[0] = scales[j + 0] & 0x3f3f;
      aux[1] = scales[j + 2] & 0x3f3f;
    } else {
      aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
      aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
    }
    dm = b->dm;
  }

  __device__ __forceinline__ float Dot(const block_q8_1* y, int iqs) const {
    const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
    int u[2 * QR5_K];
    float d8[QR5_K];
#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
      const block_q8_1* bq8i = y + bq8_offset + i;
      d8[i] = __low2float(bq8i->ds);
      const int* q8 = (const int*)bq8i->qs + ((iqs / 2) % 4);
      u[2 * i + 0] = q8[0];
      u[2 * i + 1] = q8[4];
    }
    const uint8_t* sc = (const uint8_t*)aux;
    return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, sc + 2, dm, d8);
  }
};

/// warp_reduce_sum<32> over up to 32 sums at once (scatter_reduce_sum's
/// pairs and operands, five levels deep): lane l finishes sum
/// l >> (5 - levels), and every lane of that group holds the same bits.
template<int n>
static __device__ __forceinline__ float scatter_reduce_sum32(
    const float (&v)[n], int lane) {
  static_assert(n <= 32);
  constexpr int levels = n <= 1   ? 0
                         : n <= 2 ? 1
                         : n <= 4 ? 2
                         : n <= 8 ? 3
                         : n <= 16 ? 4
                                   : 5;
  float s[1 << levels];
#pragma unroll
  for (int j = 0; j < (1 << levels); ++j)
    s[j] = j < n ? v[j] : 0.0f;
  if constexpr (levels >= 1) scatter_level<16, (1 << levels)>(s, lane);
  else s[0] += xor_lane<16>(s[0]);
  if constexpr (levels >= 2) scatter_level<8, (1 << levels) / 2>(s, lane);
  else s[0] += xor_lane<8>(s[0]);
  if constexpr (levels >= 3) scatter_level<4, (1 << levels) / 4>(s, lane);
  else s[0] += xor_lane<4>(s[0]);
  if constexpr (levels >= 4) scatter_level<2, (1 << levels) / 8>(s, lane);
  else s[0] += xor_lane<2>(s[0]);
  if constexpr (levels >= 5) scatter_level<1, 2>(s, lane);
  else s[0] += xor_lane<1>(s[0]);
  return s[0];
}

template<ggml_type type, int c_rows_per_block, bool gated = false>
__launch_bounds__(mmvq_moe_max_batch(type) * (gated ? 64 : 32),
                  1) static __global__
    void mul_mat_vec_q_moe(const void* __restrict__ weights,
                           const void* __restrict__ vy,
                           const int32_t* __restrict__ ids,
                           float* __restrict__ dst, const uint32_t ncols_x,
                           const uint32_t nrows_x, const uint32_t stride_row_x,
                           const uint32_t stride_col_y,
                           const uint32_t stride_col_dst,
                           const uint32_t stride_channel_x,
                           const uint32_t stride_channel_dst,
                           const uint32_t ncols_dst, const uint32_t ids_stride,
                           const void* __restrict__ up_weights = nullptr,
                           const int32_t* __restrict__ dst_slots = nullptr) {
  constexpr int qk = ggml_hip_type_traits<type>::qk;
  constexpr int qi = ggml_hip_type_traits<type>::qi;
  constexpr int vdr = get_vdr_mmvq(type);
  constexpr int warp_size = 32;

  constexpr vec_dot_q_hip_t vec_dot_q_hip = get_vec_dot_q_hip(type);

  const bool is_up = gated && threadIdx.y != 0;
  const uint32_t token_idx =
      gated ? 0 : blockIdx.z * blockDim.y + threadIdx.y;
  const void* vx = is_up ? up_weights : weights;
  const int row0 = c_rows_per_block * blockIdx.x;
  const int blocks_per_row_x = ncols_x / qk;
  constexpr int blocks_per_iter = vdr * warp_size / qi;

  const uint32_t channel_dst = blockIdx.y;

  if (token_idx >= ncols_dst) {
    return;
  }

  // Inactive experts still write zero to every output row. The expert ID
  // and its validity are uniform within a wave.
  const int32_t id_raw = ids[channel_dst + token_idx * ids_stride];
  const bool invalid_id = id_raw < 0;
  // Expert-ordered slot rows store to their source slot; the load is issued
  // with the expert's, not after the dot product.
  const uint32_t dst_token = dst_slots ? dst_slots[token_idx] : token_idx;
  const uint32_t channel_x = invalid_id ? 0u : (uint32_t)id_raw;

  const block_q8_1* y = ((const block_q8_1*)vy) + token_idx * stride_col_y;
  uint32_t row_offsets[c_rows_per_block];
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        // A ragged tile reads the last valid row again for its unused lane.
        // Keep the bounds check out of the quantized dot-product loop.
        const uint32_t row = min(uint32_t(row0 + i), nrows_x - 1);
        row_offsets[i] = channel_x*stride_channel_x + row*stride_row_x;
    }

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); !invalid_id && kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += vec_dot_q_hip(vx, &y[kby], row_offsets[i] + kbx, kqs);
        }
    }

    if constexpr (!gated && c_rows_per_block > 1) {
      // One scatter butterfly for the wave's rows (scatter_reduce_sum32:
      // the same bits as reducing each row on every lane).
      constexpr int shift = 5 - (c_rows_per_block <= 2   ? 1
                                 : c_rows_per_block <= 4 ? 2
                                 : c_rows_per_block <= 8 ? 3
                                 : c_rows_per_block <= 16 ? 4
                                                          : 5);
      const float value = scatter_reduce_sum32(tmp, threadIdx.x);
      const int i = threadIdx.x >> shift;
      if ((threadIdx.x & ((1 << shift) - 1)) == 0 && i < c_rows_per_block &&
          uint32_t(row0 + i) < nrows_x)
        dst[channel_dst * stride_channel_dst + dst_token * stride_col_dst +
            row0 + i] = isfinite(value) ? value : 0.0f;
      return;
    }

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    if constexpr (gated) {
      // Separate waves preserve each projection's original dot-product
      // grouping. Combining two accumulators in one wave lets fast-math
      // reassociate their shared input scales.
      __shared__ float values[2][c_rows_per_block];
      if (threadIdx.x < c_rows_per_block) {
        const float value = tmp[threadIdx.x];
        values[is_up][threadIdx.x] = isfinite(value) ? value : 0.0f;
      }
      __syncthreads();
      if (!is_up && threadIdx.x < c_rows_per_block &&
          uint32_t(row0 + threadIdx.x) < nrows_x) {
        const float g = values[0][threadIdx.x];
        const float u = values[1][threadIdx.x];
        dst[channel_dst * stride_channel_dst + row0 + threadIdx.x] =
            (g * (1.0f / (1.0f + __expf(-g)))) * u;
      }
      return;
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        const float value = tmp[threadIdx.x];
        dst[channel_dst*stride_channel_dst + dst_token*stride_col_dst + row0 + threadIdx.x] =
            isfinite(value) ? value : 0.0f;
    }
}


/// A dense K-quant projection for 1-16 tokens: every wave computes its rows for
/// all tokens, so the weights are read from memory once and from cache for
/// the later tokens. Each token's sum follows mul_mat_vec_q_moe's lane and
/// block order, so a row's result does not depend on the batch width.
template<ggml_type type, int c_rows_per_block, int tokens>
__launch_bounds__(32, 1) static __global__
    void mul_mat_vec_dense_kq(const void* __restrict__ vx,
                              const void* __restrict__ vy,
                              float* __restrict__ dst, const uint32_t ncols_x,
                              const uint32_t nrows_x,
                              const uint32_t stride_row_x,
                              const uint32_t stride_col_y,
                              const uint32_t stride_col_dst) {
  constexpr int qk = ggml_hip_type_traits<type>::qk;
  constexpr int qi = ggml_hip_type_traits<type>::qi;
  constexpr int vdr = get_vdr_mmvq(type);
  constexpr int warp_size = 32;
  constexpr vec_dot_q_hip_t vec_dot_q_hip = get_vec_dot_q_hip(type);
  const int row0 = c_rows_per_block * blockIdx.x;
  const int blocks_per_row_x = ncols_x / qk;
  constexpr int blocks_per_iter = vdr * warp_size / qi;
  const block_q8_1* y = (const block_q8_1*)vy;
  uint32_t row_offsets[c_rows_per_block];
#pragma unroll
  for (int i = 0; i < c_rows_per_block; ++i)
    row_offsets[i] = min(uint32_t(row0 + i), nrows_x - 1) * stride_row_x;
  float tmp[tokens][c_rows_per_block] = {};
  for (int kbx = threadIdx.x / (qi / vdr); kbx < blocks_per_row_x;
       kbx += blocks_per_iter) {
    const int kby = kbx * (qk / QK8_1);
    const int kqs = vdr * (threadIdx.x % (qi / vdr));
#pragma unroll
    for (int j = 0; j < tokens; ++j) {
#pragma unroll
      for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[j][i] += vec_dot_q_hip(vx, &y[j * stride_col_y + kby],
                                   row_offsets[i] + kbx, kqs);
      }
    }
  }
  if constexpr (tokens * c_rows_per_block > 1 &&
                tokens * c_rows_per_block <= 32) {
    // One scatter butterfly for every sum (scatter_reduce_sum32: the same
    // bits as reducing each sum on every lane).
    constexpr int n = tokens * c_rows_per_block;
    float sums[n];
#pragma unroll
    for (int j = 0; j < tokens; ++j)
#pragma unroll
      for (int i = 0; i < c_rows_per_block; ++i)
        sums[j * c_rows_per_block + i] = tmp[j][i];
    constexpr int shift = 5 - (n <= 2    ? 1
                               : n <= 4  ? 2
                               : n <= 8  ? 3
                               : n <= 16 ? 4
                                         : 5);
    const int lane = threadIdx.x;
    const float value = scatter_reduce_sum32(sums, lane);
    const int index = lane >> shift;
    const int j = index / c_rows_per_block;
    const int i = index % c_rows_per_block;
    if ((lane & ((1 << shift) - 1)) == 0 && index < n &&
        uint32_t(row0 + i) < nrows_x)
      dst[j * stride_col_dst + row0 + i] = isfinite(value) ? value : 0.0f;
    return;
  }
#pragma unroll
  for (int j = 0; j < tokens; ++j) {
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
      const float value = warp_reduce_sum<warp_size>(tmp[j][i]);
      if (threadIdx.x == 0 && uint32_t(row0 + i) < nrows_x)
        dst[j * stride_col_dst + row0 + i] = isfinite(value) ? value : 0.0f;
    }
  }
}

/// One lane's Q8_1 activations for a Q4_K/Q5_K block quarter, as
/// vec_dot_q{4,5}_K_q8_1 read them. Load issues the reads; Finish forms the
/// scales and the code sums the minimum term multiplies (shared by every
/// row).
struct KHeadInput {
  int u[4];
  half2 ds[2];
  float d8[2];
  float sum[2];
  // sum * d8 exactly: a code sum below 2^10 times an F16 scale fits the
  // F32 significand, so m * t rounds as (sum * m) * d8 does.
  float t[2];

  __device__ __forceinline__ void Load(const block_q8_1* y, int iqs) {
    const int offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const block_q8_1* b = y + offset + i;
      ds[i] = b->ds;
      const int* q8 = (const int*)b->qs + ((iqs / 2) % 4);
      u[2 * i + 0] = q8[0];
      u[2 * i + 1] = q8[4];
    }
  }

  __device__ __forceinline__ void Finish() {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      d8[i] = __low2float(ds[i]);
      // Converted once per token: every row's product is then a plain FP32
      // FMA instead of a mixed-precision one (the same single rounding).
      asm("" : "+v"(d8[i]));
      sum[i] = ggml_hip_dp4a(0x01010101, u[2 * i + 1],
                             ggml_hip_dp4a(0x01010101, u[2 * i + 0], 0));
      t[i] = __fmul_rn(sum[i], d8[i]);
    }
  }
};

/// One lane's codes of a Q4_K/Q5_K block quarter, unpacked once for every
/// token. Dot gives vec_dot_q{4,5}_K_q8_1_impl_vmmq's integer sums and
/// float expression.
template<ggml_type type>
struct KHeadFragment {
  static_assert(type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K);
  // Raw words: the low codes, the high bits (Q5_K) and the scale words.
  int ql[2];
  int qh[2];
  uint32_t words[3];
  half2 dmh;
  // Decoded: v0i, v1i for i = 0, then for i = 1; scales and minimums.
  int v[4];
  float sc[2];
  float m[2];
  float2 dm;

  __device__ __forceinline__ void Load(const void* vx, int block, int iqs) {
    const int offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
    const block_q4_K* scales;
    if constexpr (type == GGML_TYPE_Q4_K) {
      const block_q4_K* b = (const block_q4_K*)vx + block;
      const int* q4 = (const int*)(b->qs + 16 * offset + 4 * ((iqs / 2) % 4));
      ql[0] = q4[0];
      ql[1] = q4[4];
      dmh = b->dm;
      scales = b;
    } else {
      const block_q5_K* b = (const block_q5_K*)vx + block;
      const int* q5 = (const int*)(b->qs + 16 * offset + 4 * ((iqs / 2) % 4));
      const int* h = (const int*)(b->qh + 4 * ((iqs / 2) % 4));
      ql[0] = q5[0];
      ql[1] = q5[4];
      qh[0] = h[0];
      qh[1] = h[4];
      dmh = b->dm;
      // Q5_K packs its scales and minimums as Q4_K does, at the same offset.
      scales = (const block_q4_K*)b;
    }
    const uint32_t* w = (const uint32_t*)scales->scales;
#pragma unroll
    for (int i = 0; i < 3; ++i)
      words[i] = w[i];
  }

  __device__ __forceinline__ void Decode(int iqs) {
    const int offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
    if constexpr (type == GGML_TYPE_Q4_K) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        v[2 * i + 0] = (ql[0] >> (4 * i)) & 0x0F0F0F0F;
        v[2 * i + 1] = (ql[1] >> (4 * i)) & 0x0F0F0F0F;
      }
    } else {
      const int vh0 = qh[0] >> offset;
      const int vh1 = qh[1] >> offset;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        v[2 * i + 0] = ((ql[0] >> (4 * i)) & 0x0F0F0F0F) |
                       (((vh0 >> i) << 4) & 0x10101010);
        v[2 * i + 1] = ((ql[1] >> (4 * i)) & 0x0F0F0F0F) |
                       (((vh1 >> i) << 4) & 0x10101010);
      }
    }
    dm = __half22float2(dmh);
    asm("" : "+v"(dm.x));
    asm("" : "+v"(dm.y));
    // q4_K_scale_pair on the loaded words.
    const int j = offset / 2;
    const int shift = 16 * (j & 1);
    const uint32_t lo = (words[0] >> shift) & 0xffff;
    const uint32_t mid = (words[1] >> shift) & 0xffff;
    const uint32_t hi = (words[2] >> shift) & 0xffff;
    const uint32_t low0 = lo & 0x3f3f;
    const uint32_t low1 = mid & 0x3f3f;
    const uint32_t high0 = ((hi >> 0) & 0x0f0f) | ((lo & 0xc0c0) >> 2);
    const uint32_t high1 = ((hi >> 4) & 0x0f0f) | ((mid & 0xc0c0) >> 2);
    const uint32_t aux0 = j < 2 ? low0 : high0;
    const uint32_t aux1 = j < 2 ? low1 : high1;
    sc[0] = aux0 & 0xFF;
    sc[1] = aux0 >> 8;
    m[0] = aux1 & 0xFF;
    m[1] = aux1 >> 8;
  }

  __device__ __forceinline__ float Dot(const KHeadInput& x) const {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int dot1 =
          type == GGML_TYPE_Q4_K
              ? ggml_hip_dp4a(v[2 * i + 1], x.u[2 * i + 1],
                              ggml_hip_dp4a(v[2 * i], x.u[2 * i], 0))
              : ggml_hip_dp4a(v[2 * i], x.u[2 * i],
                              ggml_hip_dp4a(v[2 * i + 1], x.u[2 * i + 1], 0));
      // Both products are integers below 2^24, so exact: the values
      // float(dot1 * sc) and float(sum * m) the original converts. The
      // fences keep fast-math from regrouping them with d8.
      float d = float(dot1) * sc[i];
      float mm = x.sum[i] * m[i];
      asm("" : "+v"(d));
      asm("" : "+v"(mm));
      sumf_d += x.d8[i] * d;
      sumf_m += x.d8[i] * mm;
    }
    return dm.x * sumf_d - dm.y * sumf_m;
  }

  /// acc + Dot(x) as the single-token kernel compiles it, each step pinned:
  /// sd = c1 d8_1 + c0 d8_0 (c = dot1 * sc, exact), the same for the
  /// minimum terms (m * t, see KHeadInput::t), then acc + sd dm.x and
  /// acc - sm dm.y as two FMAs. Explicit roundings instead of Dot's fences
  /// leave the scheduler free to interleave independent rows and tokens.
  __device__ __forceinline__ float DotAcc(const KHeadInput& x,
                                          float acc) const {
    float c[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int dot1 =
          type == GGML_TYPE_Q4_K
              ? ggml_hip_dp4a(v[2 * i + 1], x.u[2 * i + 1],
                              ggml_hip_dp4a(v[2 * i], x.u[2 * i], 0))
              : ggml_hip_dp4a(v[2 * i], x.u[2 * i],
                              ggml_hip_dp4a(v[2 * i + 1], x.u[2 * i + 1], 0));
      c[i] = __fmul_rn(float(dot1), sc[i]);
    }
    const float sd = __fmaf_rn(c[1], x.d8[1], __fmul_rn(c[0], x.d8[0]));
    const float sm = __fmaf_rn(m[1], x.t[1], __fmul_rn(m[0], x.t[0]));
    acc = __fmaf_rn(sd, dm.x, acc);
    return __fmaf_rn(-sm, dm.y, acc);
  }
};


/// mul_mat_vec_dense_kq for Q4_K/Q5_K rows and 2-16 tokens (the output
/// head's verification and batched draft rows, batched decode). Each wave decodes `c_rows` rows'
/// codes once and loads each token's activations once for all of them, so
/// a token adds only its dot products; per row and token the lane order,
/// block order, products and butterfly are mul_mat_vec_dense_kq's.
template<ggml_type type, int c_rows, int tokens>
__launch_bounds__(32, 1) static __global__
    void mul_mat_vec_dense_kq_rows(const void* __restrict__ vx,
                                   const void* __restrict__ vy,
                                   float* __restrict__ dst,
                                   const uint32_t ncols_x,
                                   const uint32_t nrows_x,
                                   const uint32_t stride_row_x,
                                   const uint32_t stride_col_y,
                                   const uint32_t stride_col_dst) {
  static_assert(tokens * c_rows <= 32);
  constexpr int qk = QK_K;
  constexpr int qi = QI4_K;
  constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
  static_assert(qi == QI5_K && vdr == VDR_Q5_K_Q8_1_MMVQ);
  constexpr int blocks_per_iter = vdr * 32 / qi;
  const int lane = threadIdx.x;
  const int row0 = c_rows * blockIdx.x;
  const int blocks_per_row_x = ncols_x / qk;
  const block_q8_1* y = (const block_q8_1*)vy;
  uint32_t row_offsets[c_rows];
#pragma unroll
  for (int i = 0; i < c_rows; ++i)
    row_offsets[i] = min(uint32_t(row0 + i), nrows_x - 1) * stride_row_x;
  const int kqs = vdr * (lane % (qi / vdr));
  float tmp[tokens * c_rows] = {};
  for (int kbx = lane / (qi / vdr); kbx < blocks_per_row_x;
       kbx += blocks_per_iter) {
    const int kby = kbx * (qk / QK8_1);
    KHeadFragment<type> w[c_rows];
    KHeadInput x[tokens];
#pragma unroll
    for (int i = 0; i < c_rows; ++i)
      w[i].Load(vx, row_offsets[i] + kbx, kqs);
#pragma unroll
    for (int j = 0; j < tokens; ++j)
      x[j].Load(y + j * stride_col_y + kby, kqs);
    // Every read of the iteration is in flight before the first product.
    // Left to itself the scheduler interleaves loads and products from six
    // tokens on, and the exposed latency made 8 rows 30% slower.
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int i = 0; i < c_rows; ++i)
      w[i].Decode(kqs);
#pragma unroll
    for (int j = 0; j < tokens; ++j) {
      x[j].Finish();
#pragma unroll
      for (int i = 0; i < c_rows; ++i)
        tmp[j * c_rows + i] = w[i].DotAcc(x[j], tmp[j * c_rows + i]);
    }
  }
  constexpr int n = tokens * c_rows;
  constexpr int shift = 5 - (n <= 1    ? 0
                             : n <= 2  ? 1
                             : n <= 4  ? 2
                             : n <= 8  ? 3
                             : n <= 16 ? 4
                                       : 5);
  const float value = scatter_reduce_sum32(tmp, lane);
  const int index = lane >> shift;
  const int j = index / c_rows;
  const int i = index % c_rows;
  if ((lane & ((1 << shift) - 1)) == 0 && index < n &&
      uint32_t(row0 + i) < nrows_x)
    dst[j * stride_col_dst + row0 + i] = isfinite(value) ? value : 0.0f;
}

/// Dense K-quant rows for many tokens (a batched decode step's 9-64 rows):
/// block b takes token pass b % passes of `waves` row groups, one per wave,
/// so the waves of a block read the same activations (from the CU's cache
/// after the first) and the passes over a row group run next to each other
/// and share its compact rows through L2. A wave computes `c_rows` rows for
/// the `tokens` tokens of its pass; tokens past `n_tokens` repeat the last
/// one and are not stored. For every row and
/// token the lane, block order, products and butterfly are
/// mul_mat_vec_dense_kq's, so a row's result is the single-token launch's.
///
/// Q4_K/Q5_K decode each row's codes once per block quarter for all tokens
/// of the pass (KHeadFragment) and finish each token's activations once for
/// all rows (KHeadInput), `kGroup` tokens at a time.
template<ggml_type type, int c_rows, int tokens, int waves>
__launch_bounds__(32 * waves, 1) static __global__
    void mul_mat_vec_dense_kq_many(const void* __restrict__ vx,
                                   const void* __restrict__ vy,
                                   float* __restrict__ dst,
                                   const uint32_t ncols_x,
                                   const uint32_t nrows_x,
                                   const uint32_t stride_row_x,
                                   const uint32_t stride_col_y,
                                   const uint32_t stride_col_dst,
                                   const uint32_t n_tokens) {
  static_assert(type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K);
  constexpr int n = tokens * c_rows;
  static_assert(n % 32 == 0, "whole 32-sum butterflies");
  constexpr int kGroup = 4;
  static_assert(tokens % kGroup == 0);
  constexpr int qk = QK_K;
  constexpr int qi = QI4_K;
  constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
  static_assert(qi == QI5_K && vdr == VDR_Q5_K_Q8_1_MMVQ);
  constexpr int blocks_per_iter = vdr * 32 / qi;
  const int lane = threadIdx.x & 31;
  const uint32_t passes = (n_tokens + tokens - 1) / tokens;
  const int t0 = tokens * int(blockIdx.x % passes);
  const int row0 =
      c_rows * (int(blockIdx.x / passes) * waves + int(threadIdx.x >> 5));
  const int last = int(n_tokens) - 1;
  const int blocks_per_row_x = ncols_x / qk;
  const block_q8_1* y = (const block_q8_1*)vy;
  uint32_t row_offsets[c_rows];
#pragma unroll
  for (int i = 0; i < c_rows; ++i)
    row_offsets[i] = min(uint32_t(row0 + i), nrows_x - 1) * stride_row_x;
  uint32_t token_offsets[tokens];
#pragma unroll
  for (int j = 0; j < tokens; ++j)
    token_offsets[j] = uint32_t(min(t0 + j, last)) * stride_col_y;
  const int kqs = vdr * (lane % (qi / vdr));
  float tmp[n] = {};
  for (int kbx = lane / (qi / vdr); kbx < blocks_per_row_x;
       kbx += blocks_per_iter) {
    const int kby = kbx * (qk / QK8_1);
    KHeadFragment<type> w[c_rows];
#pragma unroll
    for (int i = 0; i < c_rows; ++i)
      w[i].Load(vx, row_offsets[i] + kbx, kqs);
#pragma unroll
    for (int i = 0; i < c_rows; ++i)
      w[i].Decode(kqs);
#pragma unroll
    for (int g = 0; g < tokens; g += kGroup) {
      KHeadInput x[kGroup];
#pragma unroll
      for (int j = 0; j < kGroup; ++j)
        x[j].Load(y + token_offsets[g + j] + kby, kqs);
#pragma unroll
      for (int j = 0; j < kGroup; ++j) {
        x[j].Finish();
#pragma unroll
        for (int i = 0; i < c_rows; ++i)
          tmp[(g + j) * c_rows + i] =
              w[i].DotAcc(x[j], tmp[(g + j) * c_rows + i]);
      }
    }
  }
#pragma unroll
  for (int c = 0; c < n; c += 32) {
    float part[32];
#pragma unroll
    for (int s = 0; s < 32; ++s)
      part[s] = tmp[c + s];
    const float value = scatter_reduce_sum32(part, lane);
    // 32 sums: lane l finishes sum l.
    const int j = (c + lane) / c_rows;
    const int i = (c + lane) % c_rows;
    if (t0 + j <= last && uint32_t(row0 + i) < nrows_x)
      dst[(t0 + j) * stride_col_dst + row0 + i] =
          isfinite(value) ? value : 0.0f;
  }
}

/// mul_mat_vec_dense_kq over passes of `tokens` tokens, blocks ordered as
/// mul_mat_vec_dense_kq_many's: the formats
/// without a decoded fragment reread their codes from cache per token, as
/// the single-token launch does.
template<ggml_type type, int c_rows_per_block, int tokens, int waves>
__launch_bounds__(32 * waves, 1) static __global__
    void mul_mat_vec_dense_kq_pass(const void* __restrict__ vx,
                                   const void* __restrict__ vy,
                                   float* __restrict__ dst,
                                   const uint32_t ncols_x,
                                   const uint32_t nrows_x,
                                   const uint32_t stride_row_x,
                                   const uint32_t stride_col_y,
                                   const uint32_t stride_col_dst,
                                   const uint32_t n_tokens) {
  constexpr int qk = ggml_hip_type_traits<type>::qk;
  constexpr int qi = ggml_hip_type_traits<type>::qi;
  constexpr int vdr = get_vdr_mmvq(type);
  constexpr int warp_size = 32;
  constexpr vec_dot_q_hip_t vec_dot_q_hip = get_vec_dot_q_hip(type);
  const uint32_t passes = (n_tokens + tokens - 1) / tokens;
  const int t0 = tokens * int(blockIdx.x % passes);
  const int row0 = c_rows_per_block *
                   (int(blockIdx.x / passes) * waves + int(threadIdx.x >> 5));
  const int lane = threadIdx.x & 31;
  const int last = int(n_tokens) - 1;
  const int blocks_per_row_x = ncols_x / qk;
  constexpr int blocks_per_iter = vdr * warp_size / qi;
  const block_q8_1* y = (const block_q8_1*)vy;
  uint32_t row_offsets[c_rows_per_block];
#pragma unroll
  for (int i = 0; i < c_rows_per_block; ++i)
    row_offsets[i] = min(uint32_t(row0 + i), nrows_x - 1) * stride_row_x;
  uint32_t token_offsets[tokens];
#pragma unroll
  for (int j = 0; j < tokens; ++j)
    token_offsets[j] = uint32_t(min(t0 + j, last)) * stride_col_y;
  float tmp[tokens][c_rows_per_block] = {};
  for (int kbx = lane / (qi / vdr); kbx < blocks_per_row_x;
       kbx += blocks_per_iter) {
    const int kby = kbx * (qk / QK8_1);
    const int kqs = vdr * (lane % (qi / vdr));
#pragma unroll
    for (int j = 0; j < tokens; ++j) {
#pragma unroll
      for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[j][i] += vec_dot_q_hip(vx, &y[token_offsets[j] + kby],
                                   row_offsets[i] + kbx, kqs);
      }
    }
  }
  if constexpr (tokens * c_rows_per_block <= 32) {
    // One scatter butterfly for every sum (scatter_reduce_sum32: the same
    // bits as reducing each sum on every lane).
    constexpr int n = tokens * c_rows_per_block;
    float sums[n];
#pragma unroll
    for (int j = 0; j < tokens; ++j)
#pragma unroll
      for (int i = 0; i < c_rows_per_block; ++i)
        sums[j * c_rows_per_block + i] = tmp[j][i];
    constexpr int shift = 5 - (n <= 1    ? 0
                               : n <= 2  ? 1
                               : n <= 4  ? 2
                               : n <= 8  ? 3
                               : n <= 16 ? 4
                                         : 5);
    const float value = scatter_reduce_sum32(sums, lane);
    const int index = lane >> shift;
    const int j = index / c_rows_per_block;
    const int i = index % c_rows_per_block;
    if ((lane & ((1 << shift) - 1)) == 0 && index < n && t0 + j <= last &&
        uint32_t(row0 + i) < nrows_x)
      dst[(t0 + j) * stride_col_dst + row0 + i] =
          isfinite(value) ? value : 0.0f;
    return;
  }
#pragma unroll
  for (int j = 0; j < tokens; ++j) {
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
      const float value = warp_reduce_sum<warp_size>(tmp[j][i]);
      if (lane == 0 && t0 + j <= last && uint32_t(row0 + i) < nrows_x)
        dst[(t0 + j) * stride_col_dst + row0 + i] =
            isfinite(value) ? value : 0.0f;
    }
  }
}

template<ggml_type type, int c_rows, int tokens, int waves, typename Kernel>
static void launch_dense_kq_passes(Kernel kernel, const void* weights,
                                   const block_q8_1* input, float* output,
                                   int k, int rows, int n_tokens,
                                   int input_stride, hipStream_t stream) {
  const unsigned groups = (rows + c_rows * waves - 1) / (c_rows * waves);
  const unsigned passes = (n_tokens + tokens - 1) / tokens;
  kernel<<<dim3(passes * groups), dim3(32 * waves), 0, stream>>>(
      weights, input, output, k, rows, k / ggml_blck_size(type), input_stride,
      rows, n_tokens);
}

/// Tokens per pass of launch_dense_kq_many: the dispatch hands it whole
/// passes and the remainder to the decode kernels, so no token is padded.
constexpr int dense_kq_pass_tokens(ggml_type type) {
  return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ? 16 : 8;
}

/// Whole passes of dense_kq_pass_tokens tokens. Q4_K/Q5_K: four rows per
/// wave (route_exact_bench: other shapes of rows, tokens, token groups and
/// waves were no faster).
template<ggml_type type>
static void launch_dense_kq_many(const void* weights, const block_q8_1* input,
                                 float* output, int k, int rows, int n_tokens,
                                 int input_stride, hipStream_t stream) {
  constexpr int kWaves = 8;
  constexpr int kTokens = dense_kq_pass_tokens(type);
  if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)
    launch_dense_kq_passes<type, 4, kTokens, kWaves>(
        mul_mat_vec_dense_kq_many<type, 4, kTokens, kWaves>, weights, input,
        output, k, rows, n_tokens, input_stride, stream);
  else
    launch_dense_kq_passes<type, 2, kTokens, kWaves>(
        mul_mat_vec_dense_kq_pass<type, 2, kTokens, kWaves>, weights, input,
        output, k, rows, n_tokens, input_stride, stream);
}

template<ggml_type type, int tokens = 1>
static void launch_dense_kq(const void* weights, const block_q8_1* input,
                            float* output, int k, int rows, int n_tokens,
                            int input_stride, hipStream_t stream) {
  if constexpr (tokens > 1 &&
                (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
    if (n_tokens == tokens) {
      // tokens * kRows sums fit the 32-lane butterfly.
      constexpr int kRows = tokens <= 3 || tokens > 8 ? 2 : 4;
      mul_mat_vec_dense_kq_rows<type, kRows, tokens>
          <<<dim3((rows + kRows - 1) / kRows), dim3(32), 0, stream>>>(
              weights, input, output, k, rows, k / ggml_blck_size(type),
              input_stride, rows);
      return;
    }
  }
  if (n_tokens == tokens) {
    constexpr int kRows = 2;
    mul_mat_vec_dense_kq<type, kRows, tokens>
        <<<dim3((rows + kRows - 1) / kRows), dim3(32), 0, stream>>>(
            weights, input, output, k, rows, k / ggml_blck_size(type),
            input_stride, rows);
  } else if constexpr (tokens < MMVQ_MAX_BATCH_SIZE) {
    launch_dense_kq<type, tokens + 1>(weights, input, output, k, rows,
                                      n_tokens, input_stride, stream);
  } else {
    GGML_ABORT("invalid dense vector batch width");
  }
}

void mul_mat_vec_dense_kq_dispatch(const void* weights, ggml_type type,
                                   const block_q8_1* input, float* output,
                                   int k, int rows, int tokens,
                                   int input_stride, hipStream_t stream) {
  GGML_ASSERT(tokens > 0 && tokens <= MMVQ_DENSE_KQ_MAX_BATCH &&
              k % ggml_blck_size(type) == 0 && rows > 0);
  if (tokens > MMVQ_MAX_BATCH_SIZE) {
    // Whole passes in one launch, then the remainder (and a batch narrower
    // than one pass) in launches of at most eight tokens.
    const int bulk = tokens / dense_kq_pass_tokens(type) *
                     dense_kq_pass_tokens(type);
    for (int r = bulk; r < tokens; r += MMVQ_MAX_BATCH_SIZE)
      mul_mat_vec_dense_kq_dispatch(
          weights, type, input + std::size_t(r) * input_stride,
          output + std::size_t(r) * rows, k, rows,
          std::min(MMVQ_MAX_BATCH_SIZE, tokens - r), input_stride, stream);
    if (bulk == 0)
      return;
    switch (type) {
#define QFN_DENSE_KQ_MANY(t)                                              \
  case t:                                                                 \
    launch_dense_kq_many<t>(weights, input, output, k, rows, bulk,        \
                            input_stride, stream);                        \
    return;
      QFN_DENSE_KQ_MANY(GGML_TYPE_Q3_K)
      QFN_DENSE_KQ_MANY(GGML_TYPE_Q4_K)
      QFN_DENSE_KQ_MANY(GGML_TYPE_Q5_K)
      QFN_DENSE_KQ_MANY(GGML_TYPE_Q6_K)
      QFN_DENSE_KQ_MANY(GGML_TYPE_Q5_1)
      QFN_DENSE_KQ_MANY(GGML_TYPE_IQ4_NL)
#undef QFN_DENSE_KQ_MANY
      default:
        GGML_ABORT("unsupported dense vector weight format");
    }
  }
  switch (type) {
    case GGML_TYPE_Q3_K:
      launch_dense_kq<GGML_TYPE_Q3_K>(weights, input, output, k, rows, tokens,
                                      input_stride, stream);
      break;
    case GGML_TYPE_Q4_K:
      launch_dense_kq<GGML_TYPE_Q4_K>(weights, input, output, k, rows, tokens,
                                      input_stride, stream);
      break;
    case GGML_TYPE_Q5_K:
      launch_dense_kq<GGML_TYPE_Q5_K>(weights, input, output, k, rows, tokens,
                                      input_stride, stream);
      break;
    case GGML_TYPE_Q6_K:
      launch_dense_kq<GGML_TYPE_Q6_K>(weights, input, output, k, rows, tokens,
                                      input_stride, stream);
      break;
    case GGML_TYPE_Q5_1:
      launch_dense_kq<GGML_TYPE_Q5_1>(weights, input, output, k, rows, tokens,
                                      input_stride, stream);
      break;
    case GGML_TYPE_IQ4_NL:
      launch_dense_kq<GGML_TYPE_IQ4_NL>(weights, input, output, k, rows,
                                        tokens, input_stride, stream);
      break;
    default:
      GGML_ABORT("unsupported dense vector weight format");
  }
}

// Keep one anchor per expert and a slot mask for each token. Duplicate
// experts share their weights; inactive slots still get explicit zeroes.
// Two anchor lists follow the groups, each a count and then its anchors in
// ascending order: list 0 holds the inactive slots and the groups the first
// launch computes, list 1 (split launches only) the groups with one active
// token. One block of at most kMoeGroupAnchors threads covers every anchor.
constexpr int kMoeGroupAnchors = 256;

static __global__ void group_moe_slots(const int32_t* ids, int32_t* groups,
                                       int tokens, int experts_used,
                                       bool split) {
  const int anchors = tokens * experts_used;
  const int anchor = threadIdx.x;
  // The scans below read the routing from LDS, not one dependent global
  // load per step.
  __shared__ int32_t routed[kMoeGroupAnchors];
  if (anchor < anchors)
    routed[anchor] = ids[anchor];
  __syncthreads();
  int list = -1;
  if (anchor < anchors) {
    const int expert = routed[anchor];
    int32_t* group = groups + anchor * (tokens + 1);
    group[0] = -1;
    if (expert < 0) {
      group[0] = -2;
      list = 0;
    } else {
      bool leader = true;
#pragma unroll 8
      for (int i = 0; i < anchor; ++i)
        leader &= routed[i] != expert;
      if (leader) {
        group[0] = expert;
        int active = 0;
        for (int t = 0; t < tokens; ++t) {
          uint32_t slots = 0;
          for (int j = 0; j < experts_used; ++j)
            if (routed[t * experts_used + j] == expert)
              slots |= uint32_t{1} << j;
          group[t + 1] = static_cast<int32_t>(slots);
          active += slots != 0;
        }
        list = split && active == 1 ? 1 : 0;
      }
    }
  }
  // Each list keeps its anchors in ascending order: a rank is the members
  // in lower lanes of this wave plus the members of lower waves.
  __shared__ int wave_counts[2][kMoeGroupAnchors / 32];
  const int lane = anchor % 32;
  const int wave = anchor / 32;
  const uint32_t member[2] = {static_cast<uint32_t>(__ballot(list == 0)),
                              static_cast<uint32_t>(__ballot(list == 1))};
  if (lane == 0) {
    wave_counts[0][wave] = __popc(member[0]);
    wave_counts[1][wave] = __popc(member[1]);
  }
  __syncthreads();
  int32_t* lists = groups + anchors * (tokens + 1);
  const int waves = blockDim.x / 32;
  if (anchor == 0) {
    int counts[2] = {};
    for (int w = 0; w < waves; ++w) {
      counts[0] += wave_counts[0][w];
      counts[1] += wave_counts[1][w];
    }
    lists[0] = counts[0];
    lists[anchors + 1] = counts[1];
  }
  if (list < 0)
    return;
  int rank = __popc(member[list] & ((uint32_t{1} << lane) - 1));
  for (int w = 0; w < wave; ++w)
    rank += wave_counts[list][w];
  lists[list * (anchors + 1) + 1 + rank] = anchor;
}


// Whether groups with one active input take their own launch. The shared
// groups decode each weight block once for all their inputs; a single input
// keeps a compact kernel with the scalar kernel's register footprint. A dense
// projection (one expert per input) has a single group. Q8_0's dot product
// is cheap enough that its inputs keep the original combined kernel.
static bool moe_grouped_split(ggml_type type, int tokens, int experts_used) {
  return type == GGML_TYPE_Q4_K   ? tokens >= 3
         : type == GGML_TYPE_Q8_0 ? false
                                  : tokens >= 2 && experts_used > 1;
}

// `batched` groups are group_moe_batch's MoeBatchGroup list (count first):
// up to eight rows of a larger batch per expert. Otherwise group_moe_slots'
// per-anchor masks over the `tokens` rows.
template<ggml_type type, int tokens, bool split, bool single_request = false,
         bool batched = false, int block_rows = 2>
__launch_bounds__(64) static __global__
    void mul_mat_vec_moe_grouped(const void* __restrict__ gate,
                                 const void* __restrict__ up,
                                 const block_q8_1* __restrict__ input,
                                 const int32_t* __restrict__ groups,
                                 const int32_t* __restrict__ list,
                                 float* __restrict__ output, int k, int rows,
                                 int experts_used, int input_stride) {
  constexpr int qk = ggml_hip_type_traits<type>::qk;
  constexpr int qi = ggml_hip_type_traits<type>::qi;
  constexpr int vdr = get_vdr_mmvq(type);
  constexpr int blocks_per_iter = vdr * 32 / qi;
  constexpr auto dot = get_vec_dot_q_hip(type);
  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const bool is_up = tid >= 32;
  const int row0 = blockIdx.x * block_rows;
  const int listed = batched           ? groups[0]
                     : list != nullptr ? list[0]
                                       : static_cast<int>(gridDim.y);
  for (int entry = blockIdx.y; entry < listed; entry += gridDim.y) {
  const auto* batch_group =
      reinterpret_cast<const MoeBatchGroup*>(groups + 1) + entry;
  const int anchor = batched           ? batch_group->anchor
                     : list != nullptr ? list[1 + entry]
                                       : entry;
  const int32_t* group = groups + anchor * (tokens + 1);
  // Row t of the group and its slot mask.
  const auto token_row = [&](int t) {
    return batched ? batch_group->token[t] : t;
  };
  const auto token_slots = [&](int t) {
    return batched ? static_cast<int32_t>(batch_group->slots[t])
                   : group[t + 1];
  };
  const int expert = batched ? batch_group->expert : group[0];
  if (expert < 0) {
    if (!single_request && expert == -2 && tid < block_rows &&
        row0 + tid < rows)
      output[anchor * rows + row0 + tid] = 0.0f;
    continue;
  }

  constexpr int columns = single_request ? 1 : tokens;
  int token = 0;
  if constexpr (split) {
    int active = 0;
#pragma unroll
    for (int t = 0; t < tokens; ++t) {
      if (token_slots(t) != 0) {
        ++active;
        token = t;
      }
    }
    if (single_request ? active != 1 : active <= 1)
      continue;
  }
  const void* weights = is_up ? up : gate;
  const int blocks_per_row = k / qk;
  uint32_t slots[columns];
  int row_of[columns];
#pragma unroll
  for (int t = 0; t < columns; ++t) {
    slots[t] = static_cast<uint32_t>(token_slots(single_request ? token : t));
    row_of[t] = token_row(single_request ? token : t);
  }
  float sum[columns][block_rows] = {};
  for (int kb = lane / (qi / vdr); kb < blocks_per_row; kb += blocks_per_iter) {
    const int kqs = vdr * (lane % (qi / vdr));
    if constexpr (single_request || type == GGML_TYPE_Q8_0 ||
                  (type == GGML_TYPE_Q4_K && !split)) {
#pragma unroll
      for (int t = 0; t < columns; ++t) {
        if (slots[t] == 0)
          continue;
#pragma unroll
        for (int r = 0; r < block_rows; ++r)
          sum[t][r] += dot(
              weights, input + row_of[t] * input_stride + kb * (qk / QK8_1),
              (expert * rows + min(row0 + r, rows - 1)) * blocks_per_row + kb,
              kqs);
      }
    } else {
      // Decode each row's block once for every input routed to the expert.
      MoeFragment<type> w[block_rows];
#pragma unroll
      for (int r = 0; r < block_rows; ++r)
        w[r] = MoeFragment<type>(
            weights,
            (expert * rows + min(row0 + r, rows - 1)) * blocks_per_row + kb,
            kqs);
#pragma unroll
      for (int t = 0; t < columns; ++t) {
        if (slots[t] == 0)
          continue;
        const auto* x = input + row_of[t] * input_stride + kb * (qk / QK8_1);
#pragma unroll
        for (int r = 0; r < block_rows; ++r)
          sum[t][r] += w[r].Dot(x, kqs);
        // Keep one input's activations live at a time (Q4_K keeps its tuned
        // schedule).
        if constexpr (type != GGML_TYPE_Q4_K && columns > 2)
          __builtin_amdgcn_sched_barrier(0);
      }
    }
  }

  // Gate and up stay in separate waves to retain the scalar projection's
  // sum order, including the treatment of nonfinite quantization scales.
  __shared__ float values[columns][2][block_rows];
#pragma unroll
  for (int t = 0; t < columns; ++t) {
    if (slots[t] == 0)
      continue;
#pragma unroll
    for (int r = 0; r < block_rows; ++r) {
      sum[t][r] = warp_reduce_sum<32>(sum[t][r]);
      if (lane == r)
        values[t][is_up][r] = isfinite(sum[t][r]) ? sum[t][r] : 0.0f;
    }
  }
  __syncthreads();
  if (!is_up && lane < block_rows && row0 + lane < rows) {
#pragma unroll
    for (int t = 0; t < columns; ++t) {
      if (slots[t] == 0)
        continue;
      const float g = values[t][0][lane];
      const float u = values[t][1][lane];
      uint32_t bits = slots[t];
      while (bits) {
        const int slot = __ffs(static_cast<int>(bits)) - 1;
        output[(row_of[t] * experts_used + slot) * rows + row0 + lane] =
            (g * (1.0f / (1.0f + __expf(-g)))) * u;
        bits &= bits - 1;
      }
    }
  }
  // The next listed group rewrites values.
  __syncthreads();
  }
}

// Group equal experts across the complete request batch. Each distinct token
// appears once; its mask retains duplicate output slots. The append order
// affects scheduling only: no two groups write the same output element.
static __global__ void group_moe_batch(const int32_t* ids, int32_t* storage,
                                       int tokens, int experts_used) {
  const int anchor = blockIdx.x * blockDim.x + threadIdx.x;
  if (anchor >= tokens * experts_used)
    return;
  const int expert = ids[anchor];
  const int first = anchor / experts_used;
  auto* groups = reinterpret_cast<MoeBatchGroup*>(storage + 1);
  if (expert < 0) {
    auto& group = groups[atomicAdd(storage, 1)];
    group.expert = -2;
    group.anchor = anchor;
    return;
  }
  for (int j = 0; j < anchor % experts_used; ++j)
    if (ids[first * experts_used + j] == expert)
      return;
  int before = 0;
  for (int t = 0; t < first; ++t) {
    bool found = false;
    for (int j = 0; j < experts_used; ++j)
      found |= ids[t * experts_used + j] == expert;
    before += found;
  }
  if (before % MMVQ_MAX_BATCH_SIZE != 0)
    return;
  auto& group = groups[atomicAdd(storage, 1)];
  group.expert = expert;
  group.anchor = anchor;
  int count = 0;
  for (int t = first; t < tokens && count < MMVQ_MAX_BATCH_SIZE; ++t) {
    uint32_t mask = 0;
    for (int j = 0; j < experts_used; ++j)
      if (ids[t * experts_used + j] == expert)
        mask |= uint32_t{1} << j;
    if (mask) {
      group.token[count] = t;
      group.slots[count++] = mask;
    }
  }
  for (; count < MMVQ_MAX_BATCH_SIZE; ++count) {
    group.token[count] = 0;
    group.slots[count] = 0;
  }
}


template<ggml_type type, int tokens = 2>
static void launch_moe_grouped(const void* gate, const void* up,
                               const block_q8_1* input, const int32_t* groups,
                               float* output, int k, int rows, int n_tokens,
                               int experts_used, int input_stride,
                               hipStream_t stream) {
  if (n_tokens == tokens) {
    // Split groups take two launches (see moe_grouped_split): shared groups,
    // then groups with one active input. Otherwise one block row per slot.
    const bool split = moe_grouped_split(type, tokens, experts_used);
    const int anchors = tokens * experts_used;
    const int32_t* lists = split ? groups + anchors * (tokens + 1) : nullptr;
    if (!split) {
      mul_mat_vec_moe_grouped<type, tokens, false>
          <<<dim3((rows + 1) / 2, anchors), 64, 0, stream>>>(
              gate, up, input, groups, lists, output, k, rows, experts_used,
              input_stride);
      return;
    }
    // Q4_K walks each list with two block rows. The other formats keep a
    // block row per single-input group, which hides their longer dot-product
    // latency, and one per two anchors for the shared groups (each has at
    // least two slots; inactive slots, if any, are walked as well).
    const int shared_rows =
        type == GGML_TYPE_Q4_K ? std::min(2, anchors) : (anchors + 1) / 2;
    const int single_rows = type == GGML_TYPE_Q4_K ? std::min(2, anchors)
                                                   : anchors;
    // Q3_K/Q5_K: four output rows per wave (see launch_moe_batch_groups).
    constexpr int shared = type == GGML_TYPE_Q4_K ? 2 : 4;
    mul_mat_vec_moe_grouped<type, tokens, true, false, false, shared>
        <<<dim3((rows + shared - 1) / shared, shared_rows), 64, 0, stream>>>(
            gate, up, input, groups, lists, output, k, rows, experts_used,
            input_stride);
    constexpr int single = type == GGML_TYPE_Q4_K ? 2 : 4;
    mul_mat_vec_moe_grouped<type, tokens, true, true, false, single>
        <<<dim3((rows + single - 1) / single, single_rows), 64, 0, stream>>>(
            gate, up, input, groups, lists + anchors + 1, output, k, rows,
            experts_used, input_stride);
  } else if constexpr (tokens < MMVQ_MAX_BATCH_SIZE) {
    launch_moe_grouped<type, tokens + 1>(gate, up, input, groups, output, k,
                                         rows, n_tokens, experts_used,
                                         input_stride, stream);
  } else {
    GGML_ABORT("invalid gated vector batch width");
  }
}

// Rows of a larger batch, grouped by group_moe_batch: shared groups and
// single-row groups take separate launches, one block row per possible group.
template<ggml_type type>
static void launch_moe_batch_groups(const void* gate, const void* up,
                                    const block_q8_1* input,
                                    const int32_t* groups, float* output,
                                    int k, int rows, int tokens,
                                    int experts_used, int input_stride,
                                    hipStream_t stream) {
  constexpr int width = MMVQ_MAX_BATCH_SIZE;
  const int anchors = tokens * experts_used;
  // Four output rows per wave: each input's activations serve four rows,
  // and a shared group decodes each row's block once for all its inputs.
  constexpr int shared = 4;
  mul_mat_vec_moe_grouped<type, width, true, false, true, shared>
      <<<dim3((rows + shared - 1) / shared, (anchors + 1) / 2), 64, 0,
         stream>>>(
          gate, up, input, groups, nullptr, output, k, rows, experts_used,
          input_stride);
  constexpr int single = 4;
  mul_mat_vec_moe_grouped<type, width, true, true, true, single>
      <<<dim3((rows + single - 1) / single, anchors), 64, 0, stream>>>(
          gate, up, input, groups, nullptr, output, k, rows, experts_used,
          input_stride);
}

template <int tokens>
static void launch_q8(const void* weights, const void* gate, const block_q8_1* input,
                      float* output, int k, int rows, int input_stride, hipStream_t stream) {
    if (gate && tokens > 1) {
        mul_mat_vec_q8<tokens, true, 1, false, true><<<rows, 32, 0, stream>>>(
            weights, gate, input, output, k, rows, input_stride);
    } else if (gate) {
        mul_mat_vec_q8<tokens, true><<<rows, 32, 0, stream>>>(
            weights, gate, input, output, k, rows, input_stride);
    } else if (tokens == 1 && rows <= 64 && k % (32 * 8 * 20) == 0) {
        // Few rows cannot hide a lane's load latency behind other waves.
        mul_mat_vec_q8_ahead<20><<<rows, 32, 0, stream>>>(weights, input,
                                                          output, k);
    } else if (tokens > 1) {
        mul_mat_vec_q8<tokens, false, 1, false, true><<<rows, 32, 0, stream>>>(
            weights, nullptr, input, output, k, rows, input_stride);
    } else {
        mul_mat_vec_q8<tokens, false><<<rows, 32, 0, stream>>>(
            weights, nullptr, input, output, k, rows, input_stride);
    }
}

template<int waves>
static void launch_q8_batch(const void* weights, const block_q8_1* input,
                            float* output, int k, int rows, int tokens,
                            int input_stride, hipStream_t stream) {
  if (tokens % 8 != 0) {
    mul_mat_vec_q8<8, false, waves, true><<<rows, 32 * waves, 0, stream>>>(
        weights, nullptr, input, output, k, rows, input_stride, tokens);
  } else {
    mul_mat_vec_q8<8, false, waves><<<rows, 32 * waves, 0, stream>>>(
        weights, nullptr, input, output, k, rows, input_stride, tokens);
  }
}

template<int token_tiles>
static void launch_q8_matrix(const void* weights, const block_q8_1* input,
                             float* output, int k, int rows, int tokens,
                             int input_stride, hipStream_t stream) {
  if (tokens == 16 * token_tiles)
    mul_mat_q8_decode_batch<token_tiles, false>
        <<<(rows + 15) / 16, 256, 0, stream>>>(
            static_cast<const block_q8_0*>(weights), input, output, k, rows,
            input_stride, tokens);
  else
    mul_mat_q8_decode_batch<token_tiles, true>
        <<<(rows + 15) / 16, 256, 0, stream>>>(
            static_cast<const block_q8_0*>(weights), input, output, k, rows,
            input_stride, tokens);
}

int prepare_q8_decode() {
  hipFuncAttributes attributes{};
  return hipFuncGetAttributes(
             &attributes,
             reinterpret_cast<const void*>(mul_mat_vec_q8<1, false>)) ==
                 hipSuccess
             ? 0
             : -2;
}

void mul_mat_vec_q8_dispatch(const void* weights, const void* gate,
                             const block_q8_1* input, float* output, int k,
                             int rows, int tokens, int input_stride,
                             hipStream_t stream) {
  const bool matrix_shape =
      (k == 2560 && rows >= 1024) || (k == 320 && rows == 10240);
  if (!gate && matrix_shape && tokens > 48 && tokens % 32 == 0) {
    // Independent 32-row tiles retain the cache-invariant arithmetic.
    // Put the tile loop in the grid instead of launching once per tile.
    mul_mat_q8_decode_batch<2, false, true>
        <<<dim3((rows + 15) / 16, tokens / 32), 256, 0, stream>>>(
            static_cast<const block_q8_0*>(weights), input, output, k, rows,
            input_stride, 32);
    return;
  }
  if (!gate && matrix_shape && tokens >= 9 && tokens <= 48) {
    if (tokens <= 16)
      launch_q8_matrix<1>(weights, input, output, k, rows, tokens, input_stride,
                           stream);
    else if (tokens <= 32)
      launch_q8_matrix<2>(weights, input, output, k, rows, tokens, input_stride,
                           stream);
    else
      launch_q8_matrix<3>(weights, input, output, k, rows, tokens, input_stride,
                           stream);
    return;
  }
  GGML_ASSERT(k % QK8_0 == 0 && rows > 0);
  if (tokens > 8) {
    GGML_ASSERT(tokens <= 32 && !gate);
    // Each wave retains the eight-row arithmetic. Adjacent waves work
    // on the same weight row, reusing cache lines across requests.
    if (tokens <= 16) {
      launch_q8_batch<2>(weights, input, output, k, rows, tokens, input_stride,
                          stream);
    } else if (tokens <= 24) {
      launch_q8_batch<3>(weights, input, output, k, rows, tokens, input_stride,
                          stream);
    } else {
      launch_q8_batch<4>(weights, input, output, k, rows, tokens, input_stride,
                          stream);
    }
    return;
  }
  switch (tokens) {
    case 1:
      launch_q8<1>(weights, gate, input, output, k, rows, input_stride, stream);
      break;
    case 2:
      launch_q8<2>(weights, gate, input, output, k, rows, input_stride, stream);
      break;
    case 3:
      launch_q8<3>(weights, gate, input, output, k, rows, input_stride, stream);
      break;
    case 4:
      launch_q8<4>(weights, gate, input, output, k, rows, input_stride, stream);
      break;
    case 5:
      launch_q8<5>(weights, gate, input, output, k, rows, input_stride, stream);
      break;
    case 6:
      launch_q8<6>(weights, gate, input, output, k, rows, input_stride, stream);
      break;
    case 7:
      launch_q8<7>(weights, gate, input, output, k, rows, input_stride, stream);
      break;
    case 8:
      launch_q8<8>(weights, gate, input, output, k, rows, input_stride, stream);
      break;
    default:
      GGML_ABORT("invalid vector batch width");
  }
}

template<ggml_type type, int rows_per_wave = 2>
static void launch_moe(const void* weights, const block_q8_1* input,
                       const int32_t* ids, float* output, int k, int rows,
                       int tokens, int experts_used, int input_stride,
                       hipStream_t stream,
                       const int32_t* dst_slots = nullptr) {
  GGML_ASSERT(k % ggml_blck_size(type) == 0 && rows > 0);
  GGML_ASSERT(tokens > 0);
  const int block_tokens = std::min(tokens, mmvq_moe_max_batch(type));
  const int row_stride = k / ggml_blck_size(type);
  mul_mat_vec_q_moe<type, rows_per_wave>
      <<<dim3((rows + rows_per_wave - 1) / rows_per_wave, experts_used,
              (tokens + block_tokens - 1) / block_tokens),
         dim3(32, block_tokens), 0, stream>>>(
          weights, input, ids, output, k, rows, row_stride, input_stride,
          rows * experts_used, rows * row_stride, rows, tokens, experts_used,
          nullptr, dst_slots);
}

void mul_mat_vec_moe_dispatch(const void* weights, ggml_type type,
                             const block_q8_1* input, const int32_t* ids, float* output,
                             int k, int rows, int tokens, int experts_used,
                             int input_stride, hipStream_t stream,
                             const int32_t* dst_slots) {
    // dst_slots (expert-ordered slot rows) is only used by the down view.
    GGML_ASSERT(!dst_slots || (k == 640 && experts_used == 1 && tokens > 1 &&
                               (type == GGML_TYPE_Q5_1 ||
                                type == GGML_TYPE_IQ4_NL ||
                                type == GGML_TYPE_Q8_0)));
    switch (type) {
        case GGML_TYPE_Q5_1:
          // Down projection slots have independent inputs. Wider row tiles
          // reuse each short input without changing its dot-product sum.
          if (k == 640 && experts_used == 1 && tokens > 1) {
            if (tokens > 20) {
              launch_moe<GGML_TYPE_Q5_1, 8>(
                  weights, input, ids, output, k, rows, tokens, experts_used,
                  input_stride, stream, dst_slots);
            } else {
              launch_moe<GGML_TYPE_Q5_1, 4>(
                  weights, input, ids, output, k, rows, tokens, experts_used,
                  input_stride, stream, dst_slots);
            }
            break;
          }
            launch_moe<GGML_TYPE_Q5_1>(weights, input, ids, output, k, rows, tokens,
                                          experts_used, input_stride, stream);
            break;
        case GGML_TYPE_IQ4_NL:
          if (k == 640 && experts_used == 1 && tokens > 1) {
            if (tokens > 20) {
              launch_moe<GGML_TYPE_IQ4_NL, 8>(
                  weights, input, ids, output, k, rows, tokens, experts_used,
                  input_stride, stream, dst_slots);
            } else {
              launch_moe<GGML_TYPE_IQ4_NL, 4>(
                  weights, input, ids, output, k, rows, tokens, experts_used,
                  input_stride, stream, dst_slots);
            }
            break;
          }
            launch_moe<GGML_TYPE_IQ4_NL>(weights, input, ids, output, k, rows,
                                         tokens, experts_used, input_stride,
                                         stream);
            break;
        case GGML_TYPE_Q8_0:
          // Down projection slots have independent inputs. Four rows reuse
          // each short input across more weights without changing its sum.
          if (k == 640 && experts_used == 1 && tokens > 1) {
            launch_moe<GGML_TYPE_Q8_0, 4>(weights, input, ids, output, k, rows,
                                          tokens, experts_used, input_stride,
                                          stream, dst_slots);
            break;
          }
            launch_moe<GGML_TYPE_Q8_0>(weights, input, ids, output, k, rows, tokens,
                                          experts_used, input_stride, stream);
            break;
        case GGML_TYPE_Q3_K:
            launch_moe<GGML_TYPE_Q3_K>(weights, input, ids, output, k, rows, tokens,
                                          experts_used, input_stride, stream);
            break;
        case GGML_TYPE_Q4_K:
            launch_moe<GGML_TYPE_Q4_K>(weights, input, ids, output, k, rows, tokens,
                                          experts_used, input_stride, stream);
            break;
        case GGML_TYPE_Q5_K:
            launch_moe<GGML_TYPE_Q5_K>(weights, input, ids, output, k, rows, tokens,
                                          experts_used, input_stride, stream);
            break;
        default: GGML_ABORT("unsupported vector weight format");
    }
}

void mul_mat_vec_moe_gated(const void* gate, const void* up, ggml_type type,
                           const block_q8_1* input, const int32_t* ids,
                           int32_t* groups, float* output, int k, int rows,
                           int tokens, int experts_used, int input_stride,
                           hipStream_t stream) {
  GGML_ASSERT(tokens > 0 && tokens <= MMVQ_MAX_ROUTED_BATCH);
  if (tokens > MMVQ_MAX_BATCH_SIZE) {
    GGML_ASSERT(groups && experts_used <= 32);
    group_moe_batch<<<(tokens * experts_used + 127) / 128, 128, 0, stream>>>(
        ids, groups, tokens, experts_used);
    switch (type) {
      case GGML_TYPE_Q4_K:
        mul_mat_vec_moe_batch_wave64(gate, up, input, groups, output, k, rows,
                                     experts_used, input_stride, stream);
        break;
      case GGML_TYPE_Q3_K:
        launch_moe_batch_groups<GGML_TYPE_Q3_K>(gate, up, input, groups,
                                                output, k, rows, tokens,
                                                experts_used, input_stride,
                                                stream);
        break;
      case GGML_TYPE_Q5_K:
        launch_moe_batch_groups<GGML_TYPE_Q5_K>(gate, up, input, groups,
                                                output, k, rows, tokens,
                                                experts_used, input_stride,
                                                stream);
        break;
      default:
        GGML_ASSERT(type == GGML_TYPE_Q8_0);
        launch_moe_batch_groups<GGML_TYPE_Q8_0>(gate, up, input, groups,
                                                output, k, rows, tokens,
                                                experts_used, input_stride,
                                                stream);
        break;
    }
    return;
  }
  if (tokens > 1) {
    GGML_ASSERT(groups && experts_used <= 32);
    const int anchors = tokens * experts_used;
    GGML_ASSERT(anchors <= kMoeGroupAnchors);
    group_moe_slots<<<1, (anchors + 31) / 32 * 32, 0, stream>>>(
        ids, groups, tokens, experts_used,
        moe_grouped_split(type, tokens, experts_used));
    if (type == GGML_TYPE_Q4_K) {
      launch_moe_grouped<GGML_TYPE_Q4_K>(gate, up, input, groups, output, k,
                                         rows, tokens, experts_used,
                                         input_stride, stream);
    } else if (type == GGML_TYPE_Q8_0) {
      launch_moe_grouped<GGML_TYPE_Q8_0>(gate, up, input, groups, output, k,
                                         rows, tokens, experts_used,
                                         input_stride, stream);
    } else if (type == GGML_TYPE_Q3_K) {
      launch_moe_grouped<GGML_TYPE_Q3_K>(gate, up, input, groups, output, k,
                                         rows, tokens, experts_used,
                                         input_stride, stream);
    } else {
      GGML_ASSERT(type == GGML_TYPE_Q5_K);
      launch_moe_grouped<GGML_TYPE_Q5_K>(gate, up, input, groups, output, k,
                                         rows, tokens, experts_used,
                                         input_stride, stream);
    }
    return;
  }
  const int row_stride = k / ggml_blck_size(type);
  const dim3 grid((rows + 1) / 2, experts_used);
  const dim3 block(32, 2);
  if (type == GGML_TYPE_Q4_K) {
    mul_mat_vec_q_moe<GGML_TYPE_Q4_K, 2, true><<<grid, block, 0, stream>>>(
        gate, input, ids, output, k, rows, row_stride, input_stride,
        rows * experts_used, rows * row_stride, rows, 1, experts_used, up);
  } else if (type == GGML_TYPE_Q8_0) {
    mul_mat_vec_q_moe<GGML_TYPE_Q8_0, 2, true><<<grid, block, 0, stream>>>(
        gate, input, ids, output, k, rows, row_stride, input_stride,
        rows * experts_used, rows * row_stride, rows, 1, experts_used, up);
  } else if (type == GGML_TYPE_Q3_K) {
    mul_mat_vec_q_moe<GGML_TYPE_Q3_K, 2, true><<<grid, block, 0, stream>>>(
        gate, input, ids, output, k, rows, row_stride, input_stride,
        rows * experts_used, rows * row_stride, rows, 1, experts_used, up);
  } else {
    GGML_ASSERT(type == GGML_TYPE_Q5_K);
    mul_mat_vec_q_moe<GGML_TYPE_Q5_K, 2, true><<<grid, block, 0, stream>>>(
        gate, input, ids, output, k, rows, row_stride, input_stride,
        rows * experts_used, rows * row_stride, rows, 1, experts_used, up);
  }
}

}  // namespace qfn_mmq
