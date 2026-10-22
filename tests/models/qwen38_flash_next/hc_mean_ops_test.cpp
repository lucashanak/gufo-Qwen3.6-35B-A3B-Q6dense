// HcMeanRows: DFlash target features are the mean of the four residual
// streams, written into one tap slice of a [row][tap * hidden] buffer.
// The device result must equal the host's stream-order sum times 1/4
// bit for bit, and leave the other taps untouched.
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

constexpr std::uint32_t kHidden = 2560;
constexpr std::uint32_t kStreams = 4;
constexpr std::uint32_t kTaps = 5;
constexpr std::uint32_t kTap = 2;
constexpr float kSentinel = -12345.0F;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

/// Mixed magnitudes, so the stream-order sum rounds differently from
/// other association orders; plus signed zeros and tiny values.
std::vector<float> MakeResidual(std::size_t count, std::uint32_t seed) {
  std::vector<float> values(count);
  for (std::size_t i = 0; i < count; ++i) {
    const auto bits = NextRandom(&seed);
    const float unit =
        static_cast<float>(static_cast<int>(bits & 0xFFFFU) - 32768) / 32768.0F;
    const float scale = (bits >> 16) % 3 == 0   ? 1.0e4F
                        : (bits >> 16) % 3 == 1 ? 1.0F
                                                : 1.0e-3F;
    values[i] = unit * scale;
  }
  values[0] = -0.0F;
  values[1] = 1.0e-38F;
  return values;
}

bool CheckRows(std::uint32_t tokens) {
  const std::size_t res_count =
      static_cast<std::size_t>(tokens) * kStreams * kHidden;
  const std::size_t stride = std::size_t{kTaps} * kHidden;
  const std::size_t out_count = static_cast<std::size_t>(tokens) * stride;
  const auto res = MakeResidual(res_count, 0x9E3779B9U ^ tokens);

  std::vector<float> expected(out_count, kSentinel);
  for (std::uint32_t t = 0; t < tokens; ++t) {
    const float* row = res.data() + std::size_t{t} * kStreams * kHidden;
    for (std::uint32_t i = 0; i < kHidden; ++i) {
      float sum = row[i];
      for (std::uint32_t s = 1; s < kStreams; ++s)
        sum += row[std::size_t{s} * kHidden + i];
      expected[t * stride + std::size_t{kTap} * kHidden + i] = sum * 0.25F;
    }
  }

  float* d_res = nullptr;
  float* d_out = nullptr;
  CheckHip(hipMalloc(&d_res, res_count * sizeof(float)), "hipMalloc res");
  CheckHip(hipMalloc(&d_out, out_count * sizeof(float)), "hipMalloc out");
  std::vector<float> actual(out_count, kSentinel);
  CheckHip(hipMemcpy(d_res, res.data(), res_count * sizeof(float),
                     hipMemcpyHostToDevice),
           "upload res");
  CheckHip(hipMemcpy(d_out, actual.data(), out_count * sizeof(float),
                     hipMemcpyHostToDevice),
           "upload out");
  q::HcMeanRows(d_res, d_out + std::size_t{kTap} * kHidden, tokens, kHidden,
                kStreams, stride, nullptr);
  CheckHip(hipGetLastError(), "launch");
  CheckHip(hipMemcpy(actual.data(), d_out, out_count * sizeof(float),
                     hipMemcpyDeviceToHost),
           "download");
  (void)hipFree(d_res);
  (void)hipFree(d_out);

  for (std::size_t i = 0; i < out_count; ++i) {
    if (std::memcmp(&actual[i], &expected[i], sizeof(float)) != 0) {
      std::cerr << "tokens=" << tokens << " element " << i << ": got "
                << actual[i] << " expected " << expected[i] << "\n";
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  try {
    bool ok = true;
    for (const std::uint32_t tokens : {1U, 8U, 9U, 33U, 257U})
      ok = CheckRows(tokens) && ok;
    if (!ok)
      return 1;
    std::cout << "qwen38_flash_next hc_mean_ops: ALL TESTS PASSED\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
