// Output-head operators. The dense K-quant GEMV that projects verification
// rows must give every row the bytes the one-token launch gives it, at every
// width 1-8, for each weight format the head and the mixers use; the
// one-token launch is checked against F64 dots of the dequantized rows. The
// greedy draft's argmax confidence is checked against an F64 softmax.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace {

namespace q = gufo::models::qwen38_flash_next::rocm;

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

float Uniform(std::uint32_t* state, float scale) {
  return scale *
         (static_cast<float>(Next(state) & 0xFFFFFF) / 8388608.0F - 1.0F);
}

struct Format {
  int id;  ///< GGML type id
  const char* name;
  std::size_t block;
  std::size_t bytes;
  std::size_t scale_offset;  ///< F16 d (Q6_K stores it last)
  void (*dequantize)(const void*, float*, std::size_t);
};

constexpr Format kQ3_K{11, "Q3_K", 256, 110, 108, gufo::quant::DequantizeQ3_K};
constexpr Format kQ4_K{12, "Q4_K", 256, 144, 0, gufo::quant::DequantizeQ4_K};
constexpr Format kQ5_K{13, "Q5_K", 256, 176, 0, gufo::quant::DequantizeQ5_K};
constexpr Format kQ6_K{14, "Q6_K", 256, 210, 208, gufo::quant::DequantizeQ6_K};
constexpr Format kQ5_1{7, "Q5_1", 32, 24, 0, gufo::quant::DequantizeQ5_1};
constexpr Format kIQ4_NL{20, "IQ4_NL", 32,
                         18, 0,        gufo::quant::DequantizeIQ4_NL};

/// Every width 1-8 against eight one-token launches, bit for bit, with each
/// batch ending at the input allocation and a guard after the outputs.
/// `input_scale` below 1 keeps products small (unsaturated activations).
void CheckWidths(const Format& f, int rows, int cols, float input_scale,
                 std::uint32_t seed) {
  constexpr int tokens = 8;
  const std::size_t row_bytes = cols / f.block * f.bytes;
  std::vector<std::uint8_t> packed(std::size_t(rows) * row_bytes);
  for (auto& b : packed)
    b = static_cast<std::uint8_t>(Next(&seed));
  const bool k_quant = f.block == 256;
  for (std::size_t b = 0; b < packed.size() / f.bytes; ++b) {
    std::uint8_t* blk = packed.data() + b * f.bytes;
    const float d = (Uniform(&seed, 0.2F) + 1.0F) * 2e-3F;
    const __half dh = __float2half(d);
    std::memcpy(blk + f.scale_offset, &dh, 2);
    if (k_quant && f.scale_offset == 0) {  // Q4_K / Q5_K dmin
      const __half mh = __float2half(d * (Uniform(&seed, 0.5F) + 1.0F));
      std::memcpy(blk + 2, &mh, 2);
    } else if (f.id == kQ5_1.id) {
      const __half mh =
          __float2half(-16.0F * d * (Uniform(&seed, 0.2F) + 1.0F));
      std::memcpy(blk + 2, &mh, 2);
    }
  }
  std::vector<float> x(std::size_t(tokens) * cols);
  for (std::size_t i = 0; i < x.size(); ++i)
    // A few large entries per block, the rest near zero.
    x[i] = Uniform(&seed, i % 13 == 0 ? 2.0F : 0.05F) * input_scale;

  void* dw = nullptr;
  float* dx = nullptr;
  void* dq = nullptr;
  float* out = nullptr;
  const std::size_t out_count = std::size_t(rows) * tokens + 64;
  CheckHip(hipMalloc(&dw, packed.size()), "head weights");
  CheckHip(hipMalloc(&dx, x.size() * sizeof(float)), "head inputs");
  CheckHip(hipMalloc(&dq, qfn_mmq_q8_1_bytes(tokens, cols)), "head q8 inputs");
  CheckHip(hipMalloc(&out, out_count * sizeof(float)), "head outputs");
  CheckHip(hipMemcpy(dw, packed.data(), packed.size(), hipMemcpyHostToDevice),
           "head weights upload");
  CheckHip(
      hipMemcpy(dx, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice),
      "head inputs upload");
  if (qfn_mmq_quantize_q8_1(dx, dq, tokens, cols, nullptr) != 0)
    throw std::runtime_error("head input quantization failed");
  const auto* q8 = static_cast<const std::uint8_t*>(dq);
  const std::string shape = std::string(f.name) + " M=" + std::to_string(rows) +
                            " K=" + std::to_string(cols);
  const auto run = [&](int first, int n, std::vector<float>* host) {
    CheckHip(hipMemset(out, 0xA5, out_count * sizeof(float)), "head guard");
    if (qfn_mmq_dense_kq_vec_preq(f.id, dw,
                                  q8 + qfn_mmq_q8_1_bytes(first, cols), out,
                                  rows, cols, n, nullptr) != 0)
      throw std::runtime_error("head GEMV launch failed: " + shape);
    host->resize(out_count);
    CheckHip(hipMemcpy(host->data(), out, out_count * sizeof(float),
                       hipMemcpyDeviceToHost),
             "head outputs download");
    for (std::size_t i = std::size_t(n) * rows; i < out_count; ++i)
      if (std::bit_cast<std::uint32_t>((*host)[i]) != 0xA5A5A5A5U)
        throw std::runtime_error("head GEMV overwrote its guard: " + shape);
  };
  std::vector<float> single(std::size_t(tokens) * rows), host;
  for (int t = 0; t < tokens; ++t) {
    run(t, 1, &host);
    std::copy_n(host.begin(), rows, single.begin() + std::size_t(t) * rows);
  }
  for (int n = 2; n <= tokens; ++n) {
    for (const int first : {tokens - n, 0}) {
      run(first, n, &host);
      for (int t = 0; t < n; ++t)
        if (std::memcmp(host.data() + std::size_t(t) * rows,
                        single.data() + std::size_t(first + t) * rows,
                        std::size_t(rows) * sizeof(float)) != 0)
          throw std::runtime_error("head GEMV row " + std::to_string(t) +
                                   " differs from its one-token launch: " +
                                   shape + " N=" + std::to_string(n));
    }
  }
  // The one-token launch against F64 dots of the dequantized rows and the
  // activations as quantize_q8_1 rounds them.
  std::vector<float> w(std::size_t(rows) * cols);
  for (int r = 0; r < rows; ++r)
    f.dequantize(packed.data() + r * row_bytes,
                 w.data() + std::size_t(r) * cols, cols);
  std::vector<float> xq(x.size());
  for (std::size_t b = 0; b < x.size(); b += 32) {
    float amax = 0.0F;
    for (std::size_t i = b; i < b + 32; ++i)
      amax = std::max(amax, std::abs(x[i]));
    const float d = amax / 127.0F;
    const float dh = __half2float(__float2half(d));
    for (std::size_t i = b; i < b + 32; ++i)
      xq[i] = amax == 0.0F ? 0.0F : dh * std::round(x[i] / d);
  }
  double worst = 0.0;
  for (int t = 0; t < tokens; ++t)
    for (int r = 0; r < rows; ++r) {
      double acc = 0.0, mag = 0.0;
      for (int i = 0; i < cols; ++i) {
        const double p = double(w[std::size_t(r) * cols + i]) *
                         xq[std::size_t(t) * cols + i];
        acc += p;
        mag += std::abs(p);
      }
      worst =
          std::max(worst, std::abs(single[std::size_t(t) * rows + r] - acc) /
                              (mag + 1e-30));
    }
  if (!(worst < 2e-3))
    throw std::runtime_error("head GEMV error " + std::to_string(worst) + ": " +
                             shape);
  std::cout << "head GEMV " << shape << " scale=" << input_scale
            << " N=1..8 match one-token rows; worst |GEMV - F64| / sum|w x| "
            << worst << '\n';
  for (void* p : {dw, static_cast<void*>(dx), dq, static_cast<void*>(out)})
    CheckHip(hipFree(p), "head test free");
}

/// The greedy draft's argmax confidence: softmax(logits)[argmax] against
/// FP64, with non-finite logits ignored, at the draft and full head widths
/// (both end in a partial batch of the kernel's grouped loads) and below one
/// block's width. Repeated launches give identical bits.
void CheckConfidence(std::uint32_t vocab, float spread, std::uint32_t seed) {
  std::vector<float> logits(vocab);
  for (auto& v : logits)
    v = Uniform(&seed, spread);
  logits[vocab / 3] = -INFINITY;
  logits[vocab / 5] = NAN;
  logits[vocab - 1] = spread * 1.5F;  // the maximum, at the row's end
  float* d_logits = nullptr;
  void* scratch = nullptr;
  std::int32_t* row = nullptr;
  CheckHip(hipMalloc(&d_logits, vocab * sizeof(float)), "confidence logits");
  CheckHip(hipMalloc(&scratch, q::kArgmaxParts * sizeof(q::ArgmaxCandidate)),
           "confidence scratch");
  CheckHip(hipMalloc(&row, 2 * sizeof(std::int32_t)), "confidence row");
  CheckHip(hipMemcpy(d_logits, logits.data(), vocab * sizeof(float),
                     hipMemcpyHostToDevice),
           "confidence upload");
  std::int32_t first[2] = {}, again[2] = {};
  for (std::int32_t* host : {first, again}) {
    q::Argmax(d_logits, static_cast<q::ArgmaxCandidate*>(scratch), row, 1,
              vocab, nullptr);
    q::ArgmaxConfidence(d_logits, row, vocab, nullptr);
    CheckHip(
        hipMemcpy(host, row, 2 * sizeof(std::int32_t), hipMemcpyDeviceToHost),
        "confidence download");
  }
  const std::string shape = "vocab=" + std::to_string(vocab);
  if (first[0] != static_cast<std::int32_t>(vocab - 1))
    throw std::runtime_error("draft argmax missed the maximum: " + shape);
  if (std::memcmp(first, again, sizeof(first)) != 0)
    throw std::runtime_error("draft confidence is not repeatable: " + shape);
  double total = 0.0;
  for (const float l : logits)
    if (std::isfinite(l))
      total += std::exp(double(l) - double(logits[vocab - 1]));
  const double expected = 1.0 / total;
  const double got = std::bit_cast<float>(first[1]);
  if (!(std::abs(got - expected) <= 1e-4 * expected))
    throw std::runtime_error("draft confidence " + std::to_string(got) +
                             " vs " + std::to_string(expected) + ": " + shape);
  std::cout << "draft confidence " << shape << " spread=" << spread << ": "
            << got << " (F64 " << expected << ")\n";
  for (void* p :
       {static_cast<void*>(d_logits), scratch, static_cast<void*>(row)})
    CheckHip(hipFree(p), "confidence free");
}

}  // namespace

int main() {
  try {
    if (qfn_mmq_init(0) != 0)
      throw std::runtime_error("prepare quantized decode projections");
    // The output head's formats at its width, with ragged row tails.
    for (const Format* f : {&kQ4_K, &kQ5_K}) {
      CheckWidths(*f, 4103, 2560, 1.0F, 0x4EAD0000U + f->id);
      CheckWidths(*f, 1021, 2560, 1e-4F, 0x4EAD1000U + f->id);
      CheckWidths(*f, 7, 2560, 1.0F, 0x4EAD2000U + f->id);
    }
    // Every other format the dense K-quant GEMV serves.
    CheckWidths(kQ6_K, 1029, 2560, 1.0F, 0x4EAD3000U);
    CheckWidths(kQ3_K, 1027, 2560, 1.0F, 0x4EAD4000U);
    CheckWidths(kQ5_1, 1025, 320, 1.0F, 0x4EAD5000U);
    CheckWidths(kIQ4_NL, 1031, 320, 1.0F, 0x4EAD6000U);
    // The draft head (98,304 prefix + 1,024 table rows), the full head, and
    // a row shorter than one block.
    CheckConfidence(99328, 8.0F, 0xC0F1U);
    CheckConfidence(248320, 4.0F, 0xC0F2U);
    CheckConfidence(248320, 30.0F, 0xC0F3U);
    CheckConfidence(777, 2.0F, 0xC0F4U);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
