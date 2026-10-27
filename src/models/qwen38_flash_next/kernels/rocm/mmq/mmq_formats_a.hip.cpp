// SPDX-License-Identifier: MIT
// Routed-expert MMQ instantiations for further GGML block formats, split
// across files so they compile in parallel. See VENDOR.md.
#include "qfn_mmq_prelude.h"
namespace qfn_mmq {
#include "qfn_mmq.h"
#include "common.hpp"
#include "mmq.hpp"
template void mul_mat_q_case<GGML_TYPE_Q4_0>(
    ggml_backend_hip_context&, const mmq_args&, hipStream_t);
template void mul_mat_q_case<GGML_TYPE_Q4_1>(
    ggml_backend_hip_context&, const mmq_args&, hipStream_t);
template void mul_mat_q_case<GGML_TYPE_Q5_0>(
    ggml_backend_hip_context&, const mmq_args&, hipStream_t);
template void mul_mat_q_case<GGML_TYPE_Q2_K>(
    ggml_backend_hip_context&, const mmq_args&, hipStream_t);
}  // namespace qfn_mmq
