// SPDX-License-Identifier: MIT
// Routed-expert MMQ instantiations for further GGML block formats, split
// across files so they compile in parallel. See VENDOR.md.
#include "qfn_mmq_prelude.h"
namespace qfn_mmq {
#include "qfn_mmq.h"
#include "common.hpp"
#include "mmq.hpp"
template void mul_mat_q_case<GGML_TYPE_Q6_K>(
    ggml_backend_hip_context&, const mmq_args&, hipStream_t);
template void mul_mat_q_case<GGML_TYPE_IQ4_XS>(
    ggml_backend_hip_context&, const mmq_args&, hipStream_t);
template void mul_mat_q_case<GGML_TYPE_IQ3_S>(
    ggml_backend_hip_context&, const mmq_args&, hipStream_t);
template void mul_mat_q_case<GGML_TYPE_IQ3_XXS>(
    ggml_backend_hip_context&, const mmq_args&, hipStream_t);
}  // namespace qfn_mmq
