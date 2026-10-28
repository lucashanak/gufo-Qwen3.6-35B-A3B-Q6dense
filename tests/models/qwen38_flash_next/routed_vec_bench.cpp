// Times the decode/verification routed experts of Flash-Next (1-64 rows):
// gate/up with SwiGLU, the down projection and the MoE epilogue, as the
// executor runs them, over 48 layers whose routing differs, so each forward
// streams fresh experts from DRAM instead of the 32 MiB MALL. Prints the
// median forward time per stage, the unique expert bytes, the effective
// bandwidth and an FNV-1a hash of every layer's output.
//
// routed_vec_bench [reps] [gate format] [down format] [routing] [rows,...]
//   formats: q3_k q4_k q5_k q8_0 (gated kernel), q2_k q6_k q4_0 iq2_xxs
//   iq3_xxs iq3_s iq4_xs (paired vector kernel and SwiGLU) / iq4_nl q5_1
//   q5_0 q4_0 q8_0 (default q3_k iq4_nl);
//   routing: disjoint, random, overlap, shared or all (default all).
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace {

namespace q = gufo::models::qwen38_flash_next::rocm;

constexpr int kExperts = 512;
constexpr int kUsed = 10;
constexpr int kHidden = 2560;
constexpr int kFf = 640;
constexpr int kLayers = 48;

// GGML type ids.
constexpr int kQ4_0 = 2, kQ5_0 = 6, kQ5_1 = 7, kQ8_0 = 8, kQ2_K = 10,
              kQ3_K = 11, kQ4_K = 12, kQ5_K = 13, kQ6_K = 14, kIQ2_XXS = 16,
              kIQ3_XXS = 18, kIQ4_NL = 20, kIQ3_S = 21, kIQ4_XS = 23;

struct Format {
  const char* name;
  int type;
  int block_elems;
  int block_bytes;
  int scale_at;  // F16 scale offset; a second F16 follows for Q4_K/Q5_K/Q5_1
  bool two_scales;
};

const Format kFormats[] = {
    {"q3_k", kQ3_K, 256, 110, 108, false},
    {"q4_k", kQ4_K, 256, 144, 0, true},
    {"q5_k", kQ5_K, 256, 176, 0, true},
    {"iq4_nl", kIQ4_NL, 32, 18, 0, false},
    {"q5_1", kQ5_1, 32, 24, 0, true},
    {"q8_0", kQ8_0, 32, 34, 0, false},
    {"q4_0", kQ4_0, 32, 18, 0, false},
    {"q5_0", kQ5_0, 32, 22, 0, false},
    {"q2_k", kQ2_K, 256, 84, 80, true},
    {"q6_k", kQ6_K, 256, 210, 208, false},
    {"iq2_xxs", kIQ2_XXS, 256, 66, 0, false},
    {"iq3_xxs", kIQ3_XXS, 256, 98, 0, false},
    {"iq3_s", kIQ3_S, 256, 110, 0, false},
    {"iq4_xs", kIQ4_XS, 256, 136, 0, false},
};

/// Formats with the gated gate/up kernel; the others take MoeExperts'
/// paired vector projections and SwiGLU.
bool Gated(int type) {
  return type == kQ3_K || type == kQ4_K || type == kQ5_K || type == kQ8_0;
}

const Format& FindFormat(const std::string& name) {
  for (const Format& f : kFormats)
    if (name == f.name)
      return f;
  throw std::runtime_error("unknown format " + name);
}

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

template<typename T>
T* Upload(const std::vector<T>& host) {
  void* p = nullptr;
  CheckHip(hipMalloc(&p, host.size() * sizeof(T) + 4096), "hipMalloc");
  CheckHip(
      hipMemcpy(p, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice),
      "hipMemcpy");
  return static_cast<T*>(p);
}

template<typename T>
T* Alloc(std::size_t count) {
  void* p = nullptr;
  CheckHip(hipMalloc(&p, count * sizeof(T) + 4096), "hipMalloc");
  CheckHip(hipMemset(p, 0, count * sizeof(T)), "hipMemset");
  return static_cast<T*>(p);
}

std::uint16_t HalfBits(float v) {
  const __half h = __float2half(v);
  std::uint16_t bits;
  std::memcpy(&bits, &h, 2);
  return bits;
}

/// Random blocks of `f` whose F16 scales hold small finite values.
void* MakeWeights(const Format& f, std::size_t rows, std::size_t cols,
                  std::uint32_t seed, std::size_t* bytes_per_expert) {
  const std::size_t blocks = kExperts * rows * cols / f.block_elems;
  std::vector<std::uint8_t> w(blocks * f.block_bytes);
  for (std::size_t i = 0; i + 4 <= w.size(); i += 4) {
    const std::uint32_t r = Next(&seed);
    std::memcpy(&w[i], &r, 4);
  }
  for (std::size_t b = 0; b < blocks; ++b) {
    const std::uint16_t d = HalfBits(0.002F + 0.004F * (Next(&seed) % 97) / 97);
    const std::uint16_t m = HalfBits(0.001F + 0.002F * (Next(&seed) % 89) / 89);
    std::memcpy(&w[b * f.block_bytes + f.scale_at], &d, 2);
    if (f.two_scales)
      std::memcpy(&w[b * f.block_bytes + f.scale_at + 2], &m, 2);
  }
  *bytes_per_expert = w.size() / kExperts;
  return Upload(w);
}

enum class Routing { kDisjoint, kRandom, kOverlap, kShared };
const char* RoutingName(Routing r) {
  return r == Routing::kDisjoint  ? "disjoint"
         : r == Routing::kRandom  ? "random"
         : r == Routing::kOverlap ? "overlap"
                                  : "shared";
}

/// [layer][row][slot] expert ids. Overlap keeps half of the previous row's
/// experts (consecutive verification tokens); shared repeats one set.
std::vector<std::int32_t> MakeRouting(Routing routing, int rows,
                                      std::uint32_t seed) {
  std::vector<std::int32_t> ids(std::size_t{kLayers} * rows * kUsed);
  for (int l = 0; l < kLayers; ++l) {
    std::set<int> layer_used;
    for (int r = 0; r < rows; ++r) {
      std::int32_t* row = ids.data() + (std::size_t{l} * rows + r) * kUsed;
      int n = 0;
      const auto has = [&](int e) {
        return std::find(row, row + n, e) != row + n;
      };
      if (r > 0 && routing == Routing::kShared) {
        std::memcpy(row, row - kUsed, kUsed * sizeof(std::int32_t));
        n = kUsed;
      } else if (r > 0 && routing == Routing::kOverlap) {
        for (int j = 0; j < kUsed / 2; ++j)
          row[n++] = row[j - kUsed];
      }
      while (n < kUsed) {
        const int e = static_cast<int>(Next(&seed) % kExperts);
        if (has(e) || (routing == Routing::kDisjoint && layer_used.count(e)))
          continue;
        row[n++] = e;
      }
      // Router order is by weight, not by id: shuffle the slots.
      for (int j = kUsed - 1; j > 0; --j)
        std::swap(row[j], row[Next(&seed) % (j + 1)]);
      for (int j = 0; j < kUsed; ++j)
        layer_used.insert(row[j]);
    }
  }
  return ids;
}

std::uint64_t Fnv(const std::vector<float>& v, std::uint64_t h) {
  const auto* p = reinterpret_cast<const std::uint8_t*>(v.data());
  for (std::size_t i = 0; i < v.size() * sizeof(float); ++i)
    h = (h ^ p[i]) * 1099511628211ULL;
  return h;
}

struct Buffers {
  const void* gate;
  const void* up;
  const void* down;
  const Format* gf;
  const Format* df;
  float* x;  // [layer][rows][hidden]
  std::int32_t* ids;
  float* weights;  // [layer][rows][used]
  float* router;   // [rows][experts + 1]
  float* shared;   // [rows][hidden]
  float* gate_e;   // [rows][used][ff]
  float* up_e;     // [rows][used][ff] (paired formats)
  float* down_e;   // [rows][used][hidden]
  float* out;      // [layer][rows][hidden]
};

hipStream_t g_stream = nullptr;

enum Stage { kGate = 1, kDown = 2, kEpilogue = 4, kAll = 7 };

/// The executor's route for `rows` (MoeExperts, or MoeBatch above eight).
/// Before this route existed, formats other than Q4_K ran eight-row chunks
/// above eight rows; each row's arithmetic, and so the hash, is the same.
void RunLayer(const Buffers& b, int layer, int rows, int stages) {
  const std::size_t lr = std::size_t{layer} * rows;
  // MoeExperts for up to eight rows, MoeBatch's grouped route above.
  const float* x = b.x + lr * kHidden;
  const std::int32_t* ids = b.ids + lr * kUsed;
  if ((stages & kGate) && Gated(b.gf->type) &&
      qfn_mmq_moe_gated_vec(b.gf->type, b.gate, b.up, x, ids, b.gate_e, kFf,
                            kHidden, rows, kExperts, kUsed, g_stream) != 0)
    throw std::runtime_error("gated launch failed");
  if ((stages & kGate) && !Gated(b.gf->type)) {
    if (qfn_mmq_moe_vec(b.gf->type, b.gate, x, ids, b.gate_e, kFf, kHidden,
                        rows, kExperts, kUsed, g_stream, b.up, b.up_e) != 0)
      throw std::runtime_error("paired launch failed");
    q::Swiglu(b.gate_e, b.up_e, std::size_t{rows} * kUsed * kFf, g_stream);
  }
  if ((stages & kDown) &&
      qfn_mmq_moe_vec(b.df->type, b.down, b.gate_e, ids, b.down_e, kHidden, kFf,
                      rows * kUsed, kExperts, 1, g_stream) != 0)
    throw std::runtime_error("down launch failed");
  if (stages & kEpilogue)
    q::MoeEpilogue(b.down_e, b.weights + lr * kUsed, b.shared,
                   b.router + kExperts, kExperts + 1, b.out + lr * kHidden,
                   rows, kUsed, kHidden, g_stream);
}

/// Median replay time of each stage set's 48-layer forward, captured as a
/// graph like the executor's decode steps; the sets alternate every round
/// so clock and contention changes affect them alike.
std::vector<float> MedianForwards(const Buffers& b, int rows,
                                  const std::vector<int>& sets, int reps) {
  std::vector<hipGraphExec_t> graphs;
  for (const int stages : sets) {
    // Warm the allocation pool outside capture.
    for (int l = 0; l < kLayers; ++l)
      RunLayer(b, l, rows, stages);
    CheckHip(hipStreamSynchronize(g_stream), "warm");
    hipGraph_t graph;
    CheckHip(hipStreamBeginCapture(g_stream, hipStreamCaptureModeRelaxed),
             "capture");
    for (int l = 0; l < kLayers; ++l)
      RunLayer(b, l, rows, stages);
    CheckHip(hipStreamEndCapture(g_stream, &graph), "capture end");
    hipGraphExec_t exec;
    CheckHip(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
             "instantiate");
    CheckHip(hipGraphDestroy(graph), "graph");
    graphs.push_back(exec);
  }
  hipEvent_t start, stop;
  CheckHip(hipEventCreate(&start), "event");
  CheckHip(hipEventCreate(&stop), "event");
  std::vector<std::vector<float>> ms(sets.size());
  for (int rep = 0; rep < reps + 1; ++rep) {
    for (std::size_t j = 0; j < sets.size(); ++j) {
      const std::size_t i = rep % 2 ? sets.size() - 1 - j : j;
      CheckHip(hipEventRecord(start, g_stream), "record");
      CheckHip(hipGraphLaunch(graphs[i], g_stream), "launch");
      CheckHip(hipEventRecord(stop, g_stream), "record");
      CheckHip(hipEventSynchronize(stop), "sync");
      float t = 0.0F;
      CheckHip(hipEventElapsedTime(&t, start, stop), "elapsed");
      if (rep > 0)
        ms[i].push_back(t);
    }
  }
  CheckHip(hipEventDestroy(start), "event");
  CheckHip(hipEventDestroy(stop), "event");
  std::vector<float> median;
  for (std::size_t i = 0; i < sets.size(); ++i) {
    CheckHip(hipGraphExecDestroy(graphs[i]), "graph exec");
    std::sort(ms[i].begin(), ms[i].end());
    median.push_back(ms[i][ms[i].size() / 2]);
  }
  return median;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 15;
    const Format& gf = FindFormat(argc > 2 ? argv[2] : "q3_k");
    const Format& df = FindFormat(argc > 3 ? argv[3] : "iq4_nl");
    const std::string only = argc > 4 ? argv[4] : "all";
    std::vector<int> row_list = {1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 24, 32, 64};
    if (argc > 5) {
      row_list.clear();
      for (const char* p = argv[5]; *p;) {
        char* end = nullptr;
        row_list.push_back(static_cast<int>(std::strtol(p, &end, 10)));
        p = *end == ',' ? end + 1 : end;
      }
    }
    CheckHip(hipSetDevice(0), "device");
    CheckHip(hipStreamCreateWithFlags(&g_stream, hipStreamNonBlocking),
             "stream");
    if (qfn_mmq_init(0) != 0)
      throw std::runtime_error("qfn_mmq_init failed");
    std::size_t gate_bytes = 0, up_bytes = 0, down_bytes = 0;
    Buffers b{};
    b.gf = &gf;
    b.df = &df;
    b.gate = MakeWeights(gf, kFf, kHidden, 11, &gate_bytes);
    b.up = MakeWeights(gf, kFf, kHidden, 13, &up_bytes);
    b.down = MakeWeights(df, kHidden, kFf, 17, &down_bytes);
    const std::size_t expert_bytes = gate_bytes + up_bytes + down_bytes;
    constexpr int kMaxRows = 64;
    b.router = Alloc<float>(std::size_t{kMaxRows} * (kExperts + 1));
    b.gate_e = Alloc<float>(std::size_t{kMaxRows} * kUsed * kFf);
    b.up_e = Alloc<float>(std::size_t{kMaxRows} * kUsed * kFf);
    b.down_e = Alloc<float>(std::size_t{kMaxRows} * kUsed * kHidden);
    {
      std::uint32_t seed = 0x51ED;
      std::vector<float> host(std::size_t{kMaxRows} * kHidden);
      for (float& v : host)
        v = 0.25F * (static_cast<int>(Next(&seed) % 2001) - 1000) / 1000.0F;
      b.shared = Upload(host);
      std::vector<float> router(std::size_t{kMaxRows} * (kExperts + 1));
      for (float& v : router)
        v = (static_cast<int>(Next(&seed) % 2001) - 1000) / 500.0F;
      CheckHip(hipMemcpy(b.router, router.data(), router.size() * sizeof(float),
                         hipMemcpyHostToDevice),
               "router upload");
    }
    std::printf("gate/up %s, down %s: %.2f MB per expert\n", gf.name, df.name,
                expert_bytes / 1e6);
    std::printf("%-8s %4s %6s %9s %9s %9s %9s %8s %7s  %s\n", "routing", "rows",
                "uniqE", "all_ms", "gate_ms", "down_ms", "epi_ms", "MB/fwd",
                "GB/s", "hash");
    for (const Routing routing : {Routing::kDisjoint, Routing::kRandom,
                                  Routing::kOverlap, Routing::kShared}) {
      if (only != "all" && only != RoutingName(routing))
        continue;
      for (const int rows : row_list) {
        if (routing == Routing::kDisjoint && rows * kUsed > kExperts)
          continue;
        if (rows == 1 && routing != Routing::kDisjoint && only == "all")
          continue;
        const auto ids = MakeRouting(routing, rows, 0xC0FFEEU + rows);
        std::size_t unique = 0;
        for (int l = 0; l < kLayers; ++l) {
          std::set<int> s(ids.begin() + std::size_t{l} * rows * kUsed,
                          ids.begin() + std::size_t{l + 1} * rows * kUsed);
          unique += s.size();
        }
        std::uint32_t seed = 0xABCDU + rows;
        std::vector<float> x(std::size_t{kLayers} * rows * kHidden);
        for (std::size_t i = 0; i < x.size(); ++i) {
          // Ordinary and small activations alternate by row.
          const float scale = (i / kHidden) % 2 ? 1.0F / 64.0F : 2.0F;
          x[i] =
              scale * (static_cast<int>(Next(&seed) % 2001) - 1000) / 1000.0F;
        }
        std::vector<float> w(std::size_t{kLayers} * rows * kUsed);
        for (float& v : w)
          v = 0.02F + (Next(&seed) % 1000) / 6000.0F;
        b.x = Upload(x);
        b.ids = Upload(ids);
        b.weights = Upload(w);
        b.out = Alloc<float>(std::size_t{kLayers} * rows * kHidden);
        const auto t =
            MedianForwards(b, rows, {kAll, kGate, kDown, kEpilogue}, reps);
        const float all = t[0], gate = t[1], down = t[2], epi = t[3];
        // The output hash covers the complete route on every layer.
        CheckHip(
            hipMemsetAsync(
                b.out, 0, std::size_t{kLayers} * rows * kHidden * sizeof(float),
                g_stream),
            "clear");
        for (int l = 0; l < kLayers; ++l)
          RunLayer(b, l, rows, kAll);
        CheckHip(hipStreamSynchronize(g_stream), "sync");
        std::vector<float> out(std::size_t{kLayers} * rows * kHidden);
        CheckHip(hipMemcpy(out.data(), b.out, out.size() * sizeof(float),
                           hipMemcpyDeviceToHost),
                 "download");
        const double mb = static_cast<double>(unique) * expert_bytes / 1e6;
        std::printf(
            "%-8s %4d %6.1f %9.3f %9.3f %9.3f %9.3f %8.1f %7.1f  %016llx\n",
            RoutingName(routing), rows, static_cast<double>(unique) / kLayers,
            all, gate, down, epi, mb, mb / all,
            static_cast<unsigned long long>(Fnv(out, 1469598103934665603ULL)));
        std::fflush(stdout);
        for (void* p :
             {static_cast<void*>(b.x), static_cast<void*>(b.ids),
              static_cast<void*>(b.weights), static_cast<void*>(b.out)})
          CheckHip(hipFree(p), "free");
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
