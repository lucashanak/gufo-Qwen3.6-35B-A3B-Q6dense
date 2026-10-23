// Chunked (WY, 16-token blocks) prefill Gated DeltaNet against the
// sequential prefill kernel: accuracy against FP64, aligned-split exactness
// and timing. Rows per pass on the command line.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp"

namespace q = gufo::models::qwen38_flash_next::rocm;

void q::W8A8GemmWave64(const void*, const void*, float*, std::size_t,
                       std::size_t, std::size_t, hipStream_t) {
  std::abort();
}

namespace {

constexpr std::uint32_t kKHeads = 16, kVHeads = 48, kDim = 128;
constexpr std::uint32_t kChannels = 10240, kZ = 6144;
constexpr float kEps = 1e-6F;

void Check(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
    std::exit(1);
  }
}

std::uint32_t rng = 0x1234567U;
float Random(float scale, float offset) {
  rng ^= rng << 13;
  rng ^= rng >> 17;
  rng ^= rng << 5;
  return offset +
         scale * static_cast<float>(static_cast<int>(rng & 0xFFFFU) - 32768) /
             32768.0F;
}

std::vector<float> Values(std::size_t n, float scale, float offset = 0.0F) {
  std::vector<float> v(n);
  for (float& x : v)
    x = Random(scale, offset);
  return v;
}

template<typename T>
T* Device(const std::vector<T>& host) {
  T* d = nullptr;
  Check(hipMalloc(&d, std::max<std::size_t>(host.size(), 1) * sizeof(T)),
        "hipMalloc");
  Check(hipMemcpy(d, host.data(), host.size() * sizeof(T),
                  hipMemcpyHostToDevice),
        "upload");
  return d;
}

template<typename T>
std::vector<T> Host(const T* d, std::size_t n) {
  std::vector<T> h(n);
  Check(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost),
        "download");
  return h;
}

struct Problem {
  std::uint32_t rows;
  std::vector<float> conv, z, ab, a, dt, norm, state;
  float *d_conv, *d_z, *d_ab, *d_a, *d_dt, *d_norm, *d_state, *d_out;
  float *d_qn, *d_kn, *d_raw, *d_history, *d_scratch;
};

Problem Make(std::uint32_t rows) {
  Problem p{rows};
  p.conv = Values(std::size_t{rows} * kChannels, 1.0F);
  p.z = Values(std::size_t{rows} * kZ, 2.0F);
  p.ab = Values(std::size_t{rows} * 2 * kVHeads, 2.0F);
  p.a = Values(kVHeads, 1.0F, -1.5F);
  p.dt = Values(kVHeads, 1.0F);
  p.norm = Values(kDim, 0.5F, 1.0F);
  p.state = Values(std::size_t{kVHeads} * kDim * kDim, 0.1F);
  p.d_conv = Device(p.conv);
  p.d_z = Device(p.z);
  p.d_ab = Device(p.ab);
  p.d_a = Device(p.a);
  p.d_dt = Device(p.dt);
  p.d_norm = Device(p.norm);
  p.d_state = Device(p.state);
  p.d_out = Device(std::vector<float>(std::size_t{rows} * kZ));
  p.d_qn = Device(std::vector<float>(std::size_t{rows} * kKHeads * kDim));
  p.d_kn = Device(std::vector<float>(std::size_t{rows} * kKHeads * kDim));
  p.d_raw = Device(std::vector<float>(std::size_t{rows} * kZ));
  p.d_history = Device(std::vector<float>(3 * kChannels));
  p.d_scratch = Device(std::vector<float>(std::size_t{rows + 4} * kChannels));
  Check(hipMemcpy(p.d_scratch, p.d_conv, p.conv.size() * sizeof(float),
                  hipMemcpyDeviceToDevice),
        "conv rows");
  return p;
}

void Chunk(Problem& p, std::uint32_t first, std::uint32_t count,
           q::GdnCheckpoint checkpoint = {}) {
  hipLaunchKernelGGL(q::GdnChunkKernel<q::GdnOutput::kF32>, dim3(kVHeads),
                     dim3(256), 0, nullptr,
                     p.d_conv + std::size_t{first} * kChannels,
                     p.d_ab + std::size_t{first} * 2 * kVHeads, p.d_a, p.d_dt,
                     p.d_z + std::size_t{first} * kZ, kZ, p.d_norm, p.d_state,
                     p.d_out + std::size_t{first} * kZ, nullptr, nullptr,
                     nullptr, 0, count, first, kKHeads, kVHeads, kEps,
                     checkpoint);
}

void Sequential(Problem& p, std::uint32_t count) {
  q::GatedDeltaNet(p.d_conv, kChannels, p.d_z, kZ, p.d_ab, nullptr, p.d_a,
                   p.d_dt, p.d_norm, p.d_history, p.d_scratch, p.d_qn, p.d_kn,
                   p.d_raw, p.d_state, p.d_out, nullptr, {}, {}, count, kKHeads,
                   kVHeads, kDim, 4, true, true, kEps, nullptr);
}

void Reset(Problem& p) {
  Check(hipMemcpy(p.d_state, p.state.data(), p.state.size() * sizeof(float),
                  hipMemcpyHostToDevice),
        "state");
}

// FP64 recurrence and epilogue; worst error per (token, head) row and per
// head state over that row's / head's own RMS.
void Accuracy(Problem& p, const char* name,
              const std::vector<float>& out, const std::vector<float>& st) {
  const std::uint32_t n = p.rows;
  std::vector<double> S(kDim * kDim);
  double worst_row = 0.0;
  double worst_state = 0.0;
  for (std::uint32_t h = 0; h < kVHeads; ++h) {
    const std::uint32_t kh = h % kKHeads;
    for (std::size_t i = 0; i < S.size(); ++i)
      S[i] = p.state[h * S.size() + i];
    for (std::uint32_t t = 0; t < n; ++t) {
      const float* row = p.conv.data() + std::size_t{t} * kChannels;
      const float* qv = row + kh * kDim;
      const float* kv = row + (kKHeads + kh) * kDim;
      const float* vv = row + 2 * kKHeads * kDim + h * kDim;
      double qs = 0, ks = 0;
      for (std::uint32_t i = 0; i < kDim; ++i) {
        qs += double(qv[i]) * qv[i];
        ks += double(kv[i]) * kv[i];
      }
      const double qsc = 1.0 / std::sqrt(qs + kEps) / std::sqrt(128.0);
      const double ik = 1.0 / std::sqrt(ks + kEps);
      const double x = p.ab[t * 2 * kVHeads + h] + p.dt[h];
      const double sp = std::max(x, 0.0) + std::log1p(std::exp(-std::abs(x)));
      const double g = std::exp(p.a[h] * sp);
      const double b =
          1.0 / (1.0 + std::exp(-double(p.ab[t * 2 * kVHeads + kVHeads + h])));
      std::vector<double> o(kDim);
      for (std::uint32_t r = 0; r < kDim; ++r) {
        double u = 0;
        for (std::uint32_t c = 0; c < kDim; ++c) {
          S[r * kDim + c] *= g;
          u += S[r * kDim + c] * kv[c] * ik;
        }
        const double delta = (vv[r] - u) * b;
        double acc = 0;
        for (std::uint32_t c = 0; c < kDim; ++c) {
          S[r * kDim + c] += delta * kv[c] * ik;
          acc += S[r * kDim + c] * qv[c] * qsc;
        }
        o[r] = acc;
      }
      double ss = 0;
      for (double x2 : o)
        ss += x2 * x2;
      const double scale = 1.0 / std::sqrt(ss / kDim + kEps);
      double rss = 0, worst = 0;
      for (std::uint32_t r = 0; r < kDim; ++r) {
        const double gate =
            1.0 / (1.0 + std::exp(-double(p.z[std::size_t{t} * kZ + h * kDim + r])));
        const double ref = o[r] * scale * p.norm[r] * gate;
        rss += ref * ref;
        worst = std::max(worst,
                         std::abs(ref - out[std::size_t{t} * kZ + h * kDim + r]));
      }
      worst_row = std::max(worst_row, worst / std::sqrt(rss / kDim + 1e-30));
    }
    double rss = 0, worst = 0;
    for (std::size_t i = 0; i < S.size(); ++i) {
      rss += S[i] * S[i];
      worst = std::max(worst, std::abs(S[i] - st[h * S.size() + i]));
    }
    worst_state = std::max(worst_state, worst / std::sqrt(rss / S.size()));
  }
  std::printf("%s vs FP64 over own RMS: output rows %.3g, state %.3g\n", name,
              worst_row, worst_state);
}

}  // namespace

int main(int argc, char** argv) {
  {
    // Accuracy and split exactness on a short prompt.
    constexpr std::uint32_t kRows = 203;
    Problem p = Make(kRows);
    const std::size_t out_n = std::size_t{kRows} * kZ;
    const std::size_t st_n = std::size_t{kVHeads} * kDim * kDim;
    Reset(p);
    Sequential(p, kRows);
    Check(hipDeviceSynchronize(), "sequential");
    Accuracy(p, "sequential", Host(p.d_out, out_n), Host(p.d_state, st_n));
    Reset(p);
    Chunk(p, 0, kRows);
    Check(hipDeviceSynchronize(), "chunk");
    const auto out = Host(p.d_out, out_n);
    const auto st = Host(p.d_state, st_n);
    Accuracy(p, "chunked", out, st);
    for (const std::uint32_t prefix : {16U, 64U, 112U, 192U}) {
      Reset(p);
      Chunk(p, 0, prefix);
      Chunk(p, prefix, kRows - prefix);
      const bool same =
          Host(p.d_out, out_n) == out && Host(p.d_state, st_n) == st;
      std::printf("aligned split %u + %u: %s\n", prefix, kRows - prefix,
                  same ? "exact" : "DIFFERENT");
    }
    // A checkpoint at a block boundary equals the prefix pass's state.
    std::vector<float> zero(st_n, 0.0F);
    float* captured = Device(zero);
    Reset(p);
    Chunk(p, 0, kRows, q::GdnCheckpoint{captured, nullptr, 96});
    Reset(p);
    Chunk(p, 0, 96);
    std::printf("checkpoint at 96: %s\n",
                Host(captured, st_n) == Host(p.d_state, st_n) ? "exact"
                                                               : "DIFFERENT");
  }
  std::vector<std::uint32_t> sizes = {2048, 4096, 8192};
  if (argc > 1) {
    sizes.clear();
    for (int i = 1; i < argc; ++i)
      sizes.push_back(static_cast<std::uint32_t>(std::atoi(argv[i])));
  }
  hipEvent_t start, stop;
  Check(hipEventCreate(&start), "event");
  Check(hipEventCreate(&stop), "event");
  for (const std::uint32_t rows : sizes) {
    Problem p = Make(rows);
    for (int route = 0; route < 2; ++route) {
      const auto run = [&] {
        if (route == 0)
          Sequential(p, rows);
        else
          Chunk(p, 0, rows);
      };
      Reset(p);
      run();
      Check(hipDeviceSynchronize(), "warmup");
      float best = 1e30F;
      for (int r = 0; r < 5; ++r) {
        Reset(p);
        Check(hipEventRecord(start), "record");
        run();
        Check(hipEventRecord(stop), "record");
        Check(hipEventSynchronize(stop), "sync");
        float ms = 0;
        Check(hipEventElapsedTime(&ms, start, stop), "elapsed");
        best = std::min(best, ms);
      }
      std::printf("rows %u %s: %.3f us per row per layer\n", rows,
                  route == 0 ? "sequential" : "chunked",
                  best * 1000.0F / static_cast<float>(rows));
    }
    for (float* ptr : {p.d_conv, p.d_z, p.d_ab, p.d_a, p.d_dt, p.d_norm,
                       p.d_state, p.d_out, p.d_qn, p.d_kn, p.d_raw,
                       p.d_history, p.d_scratch})
      Check(hipFree(ptr), "free");
  }
  return 0;
}
