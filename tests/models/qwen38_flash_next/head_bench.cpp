// Times the output-head GEMV on synthetic K-quant rows (diagnostic): the
// dense K-quant decode projection over `rows` x 2560 for 1-8 tokens, as the
// verification head (248,320 rows) and the MTP draft prefix (99,328 with
// its table) read it; then the token selections over one row of logits:
// the greedy draft's argmax and confidence, sampled top-64 candidates and
// greedy verification argmax.
//
// head_bench [type=13] [rows=248320] [reps=40]
//
// type is the GGML id (12 Q4_K, 13 Q5_K). Each GEMV line prints the median
// launch time, the effective weight bandwidth and an FNV-1a hash of the
// logits, so a kernel change can be checked for identical results.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen/hip/ops/token.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace {

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
}

std::uint32_t Next(std::uint32_t* state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

std::uint64_t Hash(const std::vector<float>& v) {
  std::uint64_t h = 1469598103934665603ULL;
  for (const float f : v) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    h = (h ^ bits) * 1099511628211ULL;
  }
  return h;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const int type = argc > 1 ? std::atoi(argv[1]) : 13;
    const int rows = argc > 2 ? std::atoi(argv[2]) : 248320;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 40;
    constexpr int cols = 2560, tokens = 8;
    const std::size_t block_bytes = type == 12 ? 144 : 176;
    const std::size_t blocks = std::size_t(rows) * cols / 256;
    std::vector<std::uint8_t> w(blocks * block_bytes);
    std::uint32_t seed = 0x5eed;
    for (auto& b : w)
      b = static_cast<std::uint8_t>(Next(&seed));
    for (std::size_t b = 0; b < blocks; ++b) {
      const __half d = __float2half(1e-3F * (1 + Next(&seed) % 8));
      const __half m = __float2half(1e-3F * (1 + Next(&seed) % 8));
      std::memcpy(&w[b * block_bytes], &d, 2);
      std::memcpy(&w[b * block_bytes + 2], &m, 2);
    }
    std::vector<float> x(std::size_t(tokens) * cols);
    for (auto& v : x)
      v = static_cast<float>(static_cast<int>(Next(&seed) % 2001) - 1000) /
          500.0F;
    void* dw = nullptr;
    float* dx = nullptr;
    void* dq = nullptr;
    float* out = nullptr;
    CheckHip(hipMalloc(&dw, w.size()), "weights");
    CheckHip(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice), "w");
    CheckHip(hipMalloc(&dx, x.size() * sizeof(float)), "x");
    CheckHip(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice), "x");
    CheckHip(hipMalloc(&dq, qfn_mmq_q8_1_bytes(tokens, cols)), "xq");
    CheckHip(hipMalloc(&out, std::size_t(rows) * tokens * 4), "out");
    if (qfn_mmq_quantize_q8_1(dx, dq, tokens, cols, nullptr))
      throw std::runtime_error("quantize");
    hipEvent_t a, b;
    CheckHip(hipEventCreate(&a), "event");
    CheckHip(hipEventCreate(&b), "event");
    double ms1 = 0.0;
    for (int n = 1; n <= tokens; ++n) {
      std::vector<float> times;
      for (int r = 0; r < reps + 3; ++r) {
        CheckHip(hipEventRecord(a, nullptr), "record");
        if (qfn_mmq_dense_kq_vec_preq(type, dw, dq, out, rows, cols, n,
                                      nullptr))
          throw std::runtime_error("launch");
        CheckHip(hipEventRecord(b, nullptr), "record");
        CheckHip(hipEventSynchronize(b), "sync");
        float ms = 0.0F;
        CheckHip(hipEventElapsedTime(&ms, a, b), "elapsed");
        if (r >= 3)
          times.push_back(ms);
      }
      std::sort(times.begin(), times.end());
      const double ms = times[times.size() / 2];
      if (n == 1)
        ms1 = ms;
      std::vector<float> host(std::size_t(rows) * n);
      CheckHip(
          hipMemcpy(host.data(), out, host.size() * 4, hipMemcpyDeviceToHost),
          "download");
      std::printf(
          "type=%d rows=%d N=%d  %.3f ms  (+%.3f ms/row)  %.0f GB/s  "
          "hash=%016llx\n",
          type, rows, n, ms, n > 1 ? (ms - ms1) / (n - 1) : 0.0,
          w.size() / ms / 1e6, static_cast<unsigned long long>(Hash(host)));
    }
    // The greedy draft's selection over one row of these logits.
    namespace q = gufo::models::qwen38_flash_next::rocm;
    void* scratch = nullptr;
    std::int32_t* ids = nullptr;
    CheckHip(hipMalloc(&scratch, q::kArgmaxParts * sizeof(q::ArgmaxCandidate)),
             "argmax scratch");
    CheckHip(hipMalloc(&ids, 2 * sizeof(std::int32_t)), "argmax ids");
    const auto time = [&](const char* what, auto&& launch) {
      std::vector<float> times;
      for (int r = 0; r < reps + 3; ++r) {
        CheckHip(hipEventRecord(a, nullptr), "record");
        launch();
        CheckHip(hipEventRecord(b, nullptr), "record");
        CheckHip(hipEventSynchronize(b), "sync");
        float ms = 0.0F;
        CheckHip(hipEventElapsedTime(&ms, a, b), "elapsed");
        if (r >= 3)
          times.push_back(ms);
      }
      std::sort(times.begin(), times.end());
      std::printf("%s over %d logits: %.3f ms\n", what, rows,
                  times[times.size() / 2]);
    };
    auto* candidates = static_cast<q::ArgmaxCandidate*>(scratch);
    // Logits in a model's range, so the softmax is not one-hot.
    {
      std::vector<float> logits(rows);
      for (auto& v : logits)
        v = static_cast<float>(static_cast<int>(Next(&seed) % 16001) - 8000) /
            1000.0F;
      CheckHip(hipMemcpy(out, logits.data(), rows * sizeof(float),
                         hipMemcpyHostToDevice),
               "logits upload");
    }
    time("argmax", [&] { q::Argmax(out, candidates, ids, 1, rows, nullptr); });
    time("confidence", [&] { q::ArgmaxConfidence(out, ids, rows, nullptr); });
    std::int32_t host_ids[2];
    CheckHip(hipMemcpy(host_ids, ids, sizeof(host_ids), hipMemcpyDeviceToHost),
             "confidence download");
    std::printf("argmax %d confidence bits %08x\n", host_ids[0],
                static_cast<unsigned>(host_ids[1]));
    // Sampled drafts: the top-64 candidates.
    const auto workspace = q::MtpCandidateWorkspaceSize(rows);
    std::uint32_t* top_ids = nullptr;
    std::uint32_t* top_scratch = nullptr;
    float* top_scores = nullptr;
    CheckHip(hipMalloc(&top_ids, workspace * 4), "top ids");
    CheckHip(hipMalloc(&top_scratch, workspace * 4), "top scratch");
    CheckHip(hipMalloc(&top_scores, 64 * 4), "top scores");
    time("top-64 candidates", [&] {
      q::MtpTopCandidates(out, top_ids, top_scratch, top_scores, rows, nullptr);
    });
    // Greedy verification: argmax per row and its candidate gather.
    for (const int n : {2, 8}) {
      std::printf("verification rows %d: ", n);
      time("batched argmax + gather", [&] {
        gufo::hip::LaunchBatchedGPUArgmax(
            out, top_ids, n, rows,
            {reinterpret_cast<float*>(top_scratch), workspace}, nullptr);
        q::GatherArgmaxCandidates(out, top_ids, candidates, n, rows, nullptr);
      });
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "head_bench: %s\n", e.what());
    return 1;
  }
}
