#pragma once
#include "common.hpp"

constexpr int MMVQ_MAX_BATCH_SIZE = 8;
constexpr int MMVQ_MAX_ROUTED_BATCH = 64;
// Dense K-quant projections: a row's sum does not depend on the width, and
// the batched decode reads the compact rows once for up to 16 tokens.
constexpr int MMVQ_DENSE_KQ_MAX_BATCH = 16;

// Block formats of the dense decode GEMV (every one but Q8_0, which has its
// own kernels): X(type) for each.
#define QFN_DENSE_KQ_FORMATS(X)                                              \
  X(GGML_TYPE_Q4_0) X(GGML_TYPE_Q4_1) X(GGML_TYPE_Q5_0) X(GGML_TYPE_Q5_1)    \
  X(GGML_TYPE_Q2_K) X(GGML_TYPE_Q3_K) X(GGML_TYPE_Q4_K) X(GGML_TYPE_Q5_K)    \
  X(GGML_TYPE_Q6_K) X(GGML_TYPE_IQ2_XXS) X(GGML_TYPE_IQ2_XS)                 \
  X(GGML_TYPE_IQ2_S) X(GGML_TYPE_IQ3_XXS) X(GGML_TYPE_IQ3_S)                 \
  X(GGML_TYPE_IQ1_S) X(GGML_TYPE_IQ1_M) X(GGML_TYPE_IQ4_NL)                  \
  X(GGML_TYPE_IQ4_XS)

constexpr bool mmvq_dense_kq_format(ggml_type type) {
#define QFN_DENSE_KQ_IS(t) type == t ||
  return QFN_DENSE_KQ_FORMATS(QFN_DENSE_KQ_IS) false;
#undef QFN_DENSE_KQ_IS
}

// Resolve the first dense decode function before serving; launches no work.
int prepare_q8_decode();

// One expert's independent inputs, in groups of eight. The anchor also
// identifies an inactive slot that must be zeroed.
struct MoeBatchGroup {
  int expert;
  int anchor;
  int token[MMVQ_MAX_BATCH_SIZE];
  uint32_t slots[MMVQ_MAX_BATCH_SIZE];
};

// 32-element formats whose 640-wide down projection quantizes its slot rows
// in expert order (see moe_vector_projection's by_expert route).
constexpr __host__ __device__ bool mmvq_moe_down_by_expert(ggml_type type) {
  return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 ||
         type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1 ||
         type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_Q8_0;
}

// Tuned routed-vector batch limits for gfx1151.
constexpr __host__ __device__ int mmvq_moe_max_batch(ggml_type type) {
    return type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q4_K ||
                   type == GGML_TYPE_Q5_K
               ? 4
               : 8;
}

void mul_mat_vec_q8_dispatch(const void* weights, const void* gate,
                            const block_q8_1* input, float* output,
                            int k, int rows, int tokens, int input_stride,
                            hipStream_t stream);
void mul_mat_vec_dense_kq_dispatch(const void* weights, ggml_type type,
                                   const block_q8_1* input, float* output,
                                   int k, int rows, int tokens,
                                   int input_stride, hipStream_t stream);
void mul_mat_vec_moe_dispatch(const void* weights, ggml_type type,
                             const block_q8_1* input, const int32_t* ids, float* output,
                             int k, int rows, int tokens, int experts_used,
                             int input_stride, hipStream_t stream,
                             const int32_t* dst_slots = nullptr);
// Scratch ints mul_mat_vec_moe_gated needs for 2..MMVQ_MAX_BATCH_SIZE tokens:
// a group per slot and two anchor lists.
constexpr int mmvq_moe_group_ints(int tokens, int experts_used) {
  return tokens * experts_used * (tokens + 1) + 2 * (tokens * experts_used + 1);
}

void mul_mat_vec_moe_gated(const void* gate, const void* up, ggml_type type,
                           const block_q8_1* input, const int32_t* ids,
                           int32_t* groups, float* output, int k, int rows,
                           int tokens, int experts_used, int input_stride,
                           hipStream_t stream);

void mul_mat_vec_moe_batch_wave64(const void* gate, const void* up,
                                 const block_q8_1* input, const int32_t* groups,
                                 float* output, int k, int rows,
                                 int experts_used, int input_stride,
                                 hipStream_t stream);
