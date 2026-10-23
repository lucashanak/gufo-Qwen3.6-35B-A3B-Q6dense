// Batched decode replayed from captured graphs must match eager execution
// bit for bit. A small synthetic model keeps the production layer geometry
// and weight formats (Q8_0 projections, Q4_K/Q5_1 experts, BF16 indexer,
// F16 router, PLE and the Q8_0 MTP block) with four trunk layers and a
// reduced vocabulary and expert count, so no model file is loaded.

#include <hip/hip_runtime.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

using core::GgmlType;

struct DeviceModelTestAccess {
  DeviceModel& model;
  std::uint64_t state;

  std::uint64_t Next() {
    // splitmix64
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  float Uniform() {  // [-1, 1)
    return static_cast<float>(Next() >> 40) / 8388608.0F - 1.0F;
  }

  void* Upload(const std::vector<std::uint8_t>& bytes) {
    void* p = nullptr;
    constexpr std::size_t kTail = 4096;
    if (hipMalloc(&p, bytes.size() + kTail) != hipSuccess ||
        hipMemcpy(p, bytes.data(), bytes.size(), hipMemcpyHostToDevice) !=
            hipSuccess ||
        hipMemset(static_cast<std::uint8_t*>(p) + bytes.size(), 0, kTail) !=
            hipSuccess)
      throw std::runtime_error("synthetic weight upload failed");
    model.allocations_.push_back(p);
    return p;
  }
  static void PutHalf(std::uint8_t* dst, float value) {
    const auto h = static_cast<_Float16>(value);
    std::memcpy(dst, &h, 2);
  }

  DeviceTensor Make(GgmlType type, std::uint32_t rows, std::uint32_t cols,
                    std::vector<std::uint8_t> bytes, std::uint32_t experts) {
    DeviceTensor t;
    t.data = Upload(bytes);
    t.type = type;
    t.rows = rows;
    t.cols = cols;
    t.experts = experts;
    if (type == GgmlType::kF16 || type == GgmlType::kBF16)
      model.max_half_cols_ = std::max<std::size_t>(model.max_half_cols_, cols);
    if (type == GgmlType::kQ8_0 && experts == 1)
      model.max_q8_cols_ = std::max<std::size_t>(model.max_q8_cols_, cols);
    return t;
  }
  DeviceTensor F32(std::uint32_t rows, std::uint32_t cols, float center,
                   float spread) {
    std::vector<std::uint8_t> bytes(std::size_t{rows} * cols * 4);
    for (std::size_t i = 0; i < bytes.size() / 4; ++i) {
      const float v = center + spread * Uniform();
      std::memcpy(bytes.data() + 4 * i, &v, 4);
    }
    return Make(GgmlType::kF32, rows, cols, std::move(bytes), 1);
  }
  DeviceTensor Half(GgmlType type, std::uint32_t rows, std::uint32_t cols,
                    float spread) {
    std::vector<std::uint8_t> bytes(std::size_t{rows} * cols * 2);
    for (std::size_t i = 0; i < bytes.size() / 2; ++i) {
      const float v = spread * Uniform();
      if (type == GgmlType::kF16) {
        PutHalf(bytes.data() + 2 * i, v);
      } else {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &v, 4);
        const auto bf = static_cast<std::uint16_t>(bits >> 16);
        std::memcpy(bytes.data() + 2 * i, &bf, 2);
      }
    }
    return Make(type, rows, cols, std::move(bytes), 1);
  }
  /// Uniform int8 quants have a standard deviation of about 73.
  DeviceTensor Q8(std::uint32_t rows, std::uint32_t cols,
                  std::uint32_t experts = 1, float spread = 0.0F) {
    if (spread == 0.0F)
      spread = 1.0F / std::sqrt(static_cast<float>(cols));
    const std::size_t blocks = std::size_t{experts} * rows * (cols / 32);
    std::vector<std::uint8_t> bytes(blocks * 34);
    for (std::size_t b = 0; b < blocks; ++b) {
      auto* block = bytes.data() + b * 34;
      PutHalf(block, spread / 73.0F * (1.0F + 0.25F * Uniform()));
      for (int j = 0; j < 32; j += 8) {
        auto bits = Next();
        for (int k = 0; k < 8; ++k, bits >>= 8)
          block[2 + j + k] = static_cast<std::uint8_t>(
              std::max<std::int8_t>(static_cast<std::int8_t>(bits), -127));
      }
    }
    return Make(GgmlType::kQ8_0, rows, cols, std::move(bytes), experts);
  }
  /// Random 6-bit scales and minimums; dmin = 7.5 d centres the weights.
  DeviceTensor Q4K(std::uint32_t rows, std::uint32_t cols,
                   std::uint32_t experts) {
    const float d = 1.0F / std::sqrt(static_cast<float>(cols)) / 260.0F;
    const std::size_t blocks = std::size_t{experts} * rows * (cols / 256);
    std::vector<std::uint8_t> bytes(blocks * 144);
    for (std::size_t b = 0; b < blocks; ++b) {
      auto* block = bytes.data() + b * 144;
      PutHalf(block, d);
      PutHalf(block + 2, 7.5F * d);
      for (int j = 4; j < 144; j += 8) {
        auto bits = Next();
        for (int k = 0; k < 8 && j + k < 144; ++k, bits >>= 8)
          block[j + k] = static_cast<std::uint8_t>(bits);
      }
    }
    return Make(GgmlType::kQ4_K, rows, cols, std::move(bytes), experts);
  }
  /// A K-quant trunk projection as DeviceModel loads one: compact Q4_K rows
  /// for decode GEMVs and their Q8_0 re-encoding for every Q8_0 path.
  DeviceTensor Q4KProjection(std::uint32_t rows, std::uint32_t cols) {
    const DeviceTensor compact = Q4K(rows, cols, 1);
    const std::size_t row_bytes = std::size_t{cols} / 256 * 144;
    std::vector<std::uint8_t> raw(std::size_t{rows} * row_bytes);
    if (hipMemcpy(raw.data(), compact.data, raw.size(),
                  hipMemcpyDeviceToHost) != hipSuccess)
      throw std::runtime_error("synthetic K-quant download failed");
    std::vector<std::uint8_t> q8(std::size_t{rows} * (cols / 32) * 34);
    std::vector<float> row(cols);
    for (std::size_t r = 0; r < rows; ++r) {
      gufo::quant::DequantizeQ4_K(raw.data() + r * row_bytes, row.data(), cols);
      std::uint8_t* out = q8.data() + r * (cols / 32) * 34;
      for (std::size_t b = 0; b < cols / 32; ++b) {
        float amax = 0.0F;
        for (int j = 0; j < 32; ++j)
          amax = std::max(amax, std::fabs(row[b * 32 + j]));
        const float d = amax / 127.0F;
        const float id = d != 0.0F ? 1.0F / d : 0.0F;
        PutHalf(out + b * 34, d);
        for (int j = 0; j < 32; ++j)
          out[b * 34 + 2 + j] = static_cast<std::uint8_t>(
              static_cast<std::int8_t>(std::round(row[b * 32 + j] * id)));
      }
    }
    DeviceTensor t = Make(GgmlType::kQ8_0, rows, cols, std::move(q8), 1);
    t.decode = compact.data;
    t.decode_type = GgmlType::kQ4_K;
    return t;
  }
  DeviceTensor Q51(std::uint32_t rows, std::uint32_t cols,
                   std::uint32_t experts) {
    const float d = 1.0F / std::sqrt(static_cast<float>(cols)) / 9.2F;
    const std::size_t blocks = std::size_t{experts} * rows * (cols / 32);
    std::vector<std::uint8_t> bytes(blocks * 24);
    for (std::size_t b = 0; b < blocks; ++b) {
      auto* block = bytes.data() + b * 24;
      PutHalf(block, d);
      PutHalf(block + 2, -15.5F * d);
      for (int j = 4; j < 24; j += 8) {
        auto bits = Next();
        for (int k = 0; k < 8 && j + k < 24; ++k, bits >>= 8)
          block[j + k] = static_cast<std::uint8_t>(bits);
      }
    }
    return Make(GgmlType::kQ5_1, rows, cols, std::move(bytes), experts);
  }

  DeviceMixer Mixer(const Config& c, bool inject, bool q8_inject) {
    DeviceMixer m;
    m.norm = F32(1, c.HcDim(), 1.0F, 0.1F);
    m.down = Q8(c.hc_low_rank, c.HcDim());
    m.up = Q8(c.HcDim(), c.hc_low_rank);
    if (inject)
      m.inject = q8_inject ? Q8(c.hc_count, c.HcDim())
                           : F32(c.hc_count, c.HcDim(), 0.0F, 0.01F);
    return m;
  }
  void Ffn(const Config& c, DeviceLayer& l, bool q8_experts) {
    const auto e = c.num_experts;
    l.router = Half(GgmlType::kF16, e + 1, c.hidden_size, 0.05F);
    if (q8_experts) {
      l.ffn_gate_exps = Q8(c.expert_ff, c.hidden_size, e);
      l.ffn_up_exps = Q8(c.expert_ff, c.hidden_size, e);
      l.ffn_down_exps = Q8(c.hidden_size, c.expert_ff, e);
    } else {
      l.ffn_gate_exps = Q4K(c.expert_ff, c.hidden_size, e);
      l.ffn_up_exps = Q4K(c.expert_ff, c.hidden_size, e);
      l.ffn_down_exps = Q51(c.hidden_size, c.expert_ff, e);
    }
    l.shexp_gate = Q8(c.shared_expert_ff, c.hidden_size);
    l.shexp_up = Q8(c.shared_expert_ff, c.hidden_size);
    l.shexp_down = Q8(c.hidden_size, c.shared_expert_ff);
  }
  void Attention(const Config& c, DeviceLayer& l) {
    l.attn_qkv =
        Q8(2 * c.AttentionQDim() + 2 * c.AttentionKvDim(), c.hidden_size);
    l.attn_out = Q8(c.hidden_size, c.AttentionQDim());
    l.attn_q_norm = F32(1, c.head_dim, 1.0F, 0.1F);
    l.attn_k_norm = F32(1, c.head_dim, 1.0F, 0.1F);
    l.indexer_q = Half(GgmlType::kBF16, c.indexer_heads * c.indexer_head_dim,
                       c.hidden_size, 0.05F);
    l.indexer_k =
        Half(GgmlType::kBF16, c.indexer_head_dim, c.hidden_size, 0.05F);
    l.indexer_q_norm = F32(1, c.indexer_head_dim, 1.0F, 0.1F);
    l.indexer_k_norm = F32(1, c.indexer_head_dim, 1.0F, 0.1F);
  }

  static std::unique_ptr<DeviceModel> Build(const Config& c) {
    std::unique_ptr<DeviceModel> m(new DeviceModel());
    DeviceModelTestAccess b{*m, 0x5EED};
    m->config_ = c;
    m->token_embd_ = b.Q8(c.vocab_size, c.hidden_size, 1, 0.05F);
    m->output_ = b.Q8(c.vocab_size, c.hidden_size);
    m->output_bytes_ = std::size_t{c.vocab_size} * (c.hidden_size / 32 * 34);
    m->hc_head_ = b.Mixer(c, false, false);
    for (std::uint32_t il = 0; il < c.num_layers; ++il) {
      DeviceLayer l;
      l.linear = c.IsLinearLayer(il);
      l.hc_attn = b.Mixer(c, true, false);
      l.hc_ffn = b.Mixer(c, true, false);
      if (l.linear) {
        l.ssm_in = b.Q8(c.SsmConvChannels() + c.SsmValueDim(), c.hidden_size);
        l.ssm_alpha_beta =
            b.Half(GgmlType::kF16, 2 * c.ssm_num_v_heads, c.hidden_size, 0.05F);
        l.ssm_conv1d =
            b.F32(c.SsmConvChannels(), c.ssm_conv_kernel, 0.0F, 0.5F);
        l.ssm_dt = b.F32(1, c.ssm_num_v_heads, 0.5F, 0.5F);
        l.ssm_a = b.F32(1, c.ssm_num_v_heads, -1.25F, 0.75F);
        l.ssm_norm = b.F32(1, c.ssm_head_dim, 1.0F, 0.1F);
        l.ssm_out = b.Q8(c.hidden_size, c.SsmValueDim());
      } else {
        b.Attention(c, l);
      }
      if (c.IsPleLayer(il)) {
        l.ple_key = b.Q8(c.HcDim(), c.PleEmbeddingDim());
        l.ple_value = b.Q8(c.hidden_size, c.PleEmbeddingDim());
        l.ple_norm_key = b.F32(1, c.HcDim(), 1.0F, 0.1F);
        l.ple_norm_query = b.F32(1, c.HcDim(), 1.0F, 0.1F);
        l.ple_norm_conv = b.F32(1, c.HcDim(), 1.0F, 0.1F);
        l.ple_conv1d = b.F32(c.HcDim(), c.ple_conv_kernel, 0.0F, 0.5F);
      }
      b.Ffn(c, l, il == 2);  // one Q8_0 expert layer, as the artifact has
      if (il == 0) {
        // K-quant trunk projections (compact decode rows and a Q8_0 copy),
        // as the K-quant artifacts load them.
        l.ssm_out = b.Q4KProjection(c.hidden_size, c.SsmValueDim());
        l.shexp_gate = b.Q4KProjection(c.shared_expert_ff, c.hidden_size);
        l.shexp_up = b.Q4KProjection(c.shared_expert_ff, c.hidden_size);
      }
      m->layers_.push_back(l);
    }
    DeviceLayer& mtp = m->mtp_;
    mtp.hc_attn = b.Mixer(c, true, true);
    mtp.hc_ffn = b.Mixer(c, true, true);
    b.Attention(c, mtp);
    b.Ffn(c, mtp, true);
    mtp.nextn_enorm = b.F32(1, c.hidden_size, 1.0F, 0.1F);
    mtp.nextn_hnorm = b.F32(1, c.HcDim(), 1.0F, 0.1F);
    mtp.nextn_fc_embedding = b.Q8(c.hidden_size, c.hidden_size);
    mtp.nextn_fc_hidden = b.Q8(c.hidden_size, c.hidden_size);
    mtp.nextn_head = b.Mixer(c, false, false);
    m->has_mtp_ = true;
    if (hipDeviceSynchronize() != hipSuccess)
      throw std::runtime_error("synthetic model upload failed");
    return m;
  }
};

}  // namespace gufo::models::qwen38_flash_next::rocm

namespace {

namespace qfn = gufo::models::qwen38_flash_next;
namespace rocm = qfn::rocm;
using rocm::Executor;
using rocm::Session;

qfn::Config SyntheticConfig() {
  qfn::Config c;
  c.num_layers = 4;  // three DeltaNet layers, then attention
  c.num_layers_all = 5;
  c.nextn_layers = 1;
  c.hidden_size = 2560;
  c.vocab_size = 8192;
  c.context_length = 4096;
  c.rms_eps = 1e-6F;
  c.hc_count = 4;
  c.hc_low_rank = 320;
  c.full_attention_interval = 4;
  c.num_heads = 24;
  c.num_kv_heads = 2;
  c.head_dim = 256;
  c.rotary_dim = 64;
  c.rope_theta = 1e7F;
  c.rope_sections = {11, 11, 10, 0};
  c.indexer_heads = 4;
  c.indexer_head_dim = 128;
  c.indexer_top_k = 2048;
  c.compress_ratio = 4;
  c.ssm_conv_kernel = 4;
  c.ssm_head_dim = 128;
  c.ssm_num_k_heads = 16;
  c.ssm_num_v_heads = 48;
  c.ssm_inner_size = 6144;
  c.num_experts = 16;
  c.num_experts_used = 10;
  c.expert_ff = 640;
  c.shared_expert_ff = 640;
  c.ple_layer = 1;  // after layer 0: graphs split into prefix and suffix
  c.ple_ngram_size = 3;
  c.ple_heads_per_ngram = 8;
  c.ple_heads = 16;
  c.ple_head_dim = 160;
  c.ple_conv_kernel = 4;
  c.ple_eos_token = 8191;
  c.ple_multipliers = {1000003ULL, 998244353ULL, 1000000007ULL};
  for (std::uint32_t h = 0; h < c.ple_heads; ++h) {
    c.ple_head_offsets[h] = h * 1024;
    c.ple_head_vocab[h] = 1021;
  }
  c.ple_rows = 16 * 1024;
  return c;
}

/// One request: its executor session and the host view of its history.
struct Lane {
  std::unique_ptr<Session> session;
  std::vector<std::int32_t> history;  ///< tokens at trunk positions
  std::uint32_t base{0};              ///< first trunk row kept in hidden
  std::int32_t next{0};               ///< successor of the history
};

[[noreturn]] void Throw(const std::string& what, const std::string& error) {
  throw std::runtime_error(what + ": " + error);
}

std::int32_t Argmax(std::span<const float> row) {
  return static_cast<std::int32_t>(std::max_element(row.begin(), row.end()) -
                                   row.begin());
}

/// Every output the host reads, in call order.
struct Record {
  std::vector<float> logits;
  std::vector<std::int32_t> tokens;
  std::vector<float> candidates;

  void Candidates(const qfn::MtpCandidateLogits& c) {
    tokens.push_back(static_cast<std::int32_t>(c.size));
    for (std::size_t i = 0; i < c.size; ++i) {
      tokens.push_back(c.ids[i]);
      candidates.push_back(c.logits[i]);
    }
  }
};

struct Plan {
  std::uint32_t lanes;
  std::uint32_t drafts;  ///< drafted tokens per verification
  std::uint32_t cycles;
};

/// One greedy decode cycle in the engine's order: draft catch-up over the
/// kept trunk rows, draft rounds (heads then single-token bodies), one
/// verification batch, then a deterministic acceptance with rollback.
void Cycle(const Executor& exec, std::vector<Lane>& lanes, const Plan& plan,
           std::uint32_t cycle, Record& out) {
  std::string error;
  const auto vocab = exec.config().vocab_size;
  std::vector<std::size_t> active;
  for (std::size_t i = 0; i < lanes.size(); ++i) {
    // Membership changes now and then, as requests finish and arrive.
    if (!(cycle % 7 == 6 && i == 0))
      active.push_back(i);
  }
  std::vector<std::vector<std::int32_t>> replay(lanes.size());
  std::vector<Executor::MtpBatchItem> catchup;
  for (const auto i : active) {
    auto& lane = lanes[i];
    const auto mp = exec.MtpPosition(*lane.session);
    replay[i].assign(lane.history.begin() + mp + 1, lane.history.end());
    replay[i].push_back(lane.next);
    catchup.push_back({lane.session.get(), replay[i],
                       static_cast<std::int32_t>(mp - lane.base)});
  }
  if (!exec.MtpForwardBatch(catchup, &error))
    Throw("draft catch-up", error);
  std::vector<std::vector<std::int32_t>> chains(lanes.size());
  std::vector<std::uint32_t> width(lanes.size());
  for (const auto i : active) {
    chains[i] = {lanes[i].next};
    // Short chains now and then; one request occasionally verifies alone.
    width[i] = 1 + (cycle % 4 == 3 ? 1 : plan.drafts);
    if (i == 1 && cycle % 5 == 4)
      width[i] = 1;
  }
  std::vector<std::int32_t> draft(lanes.size());
  std::vector<qfn::MtpCandidateLogits> candidates(lanes.size());
  for (;;) {
    std::vector<Executor::MtpHeadItem> heads;
    std::vector<std::size_t> owners;
    for (const auto i : active) {
      if (chains[i].size() >= width[i])
        continue;
      // Greedy full head, small head and sampled candidates side by side.
      const bool sampled = i % 3 == 1;
      heads.push_back({lanes[i].session.get(),
                       {.token = sampled ? nullptr : &draft[i],
                        .candidates = sampled ? &candidates[i] : nullptr,
                        .draft_subset = i % 2 == 0}});
      owners.push_back(i);
    }
    if (heads.empty())
      break;
    if (!exec.MtpHeads(heads, &error))
      Throw("draft heads", error);
    std::vector<Executor::MtpBatchItem> bodies;
    for (const auto i : owners) {
      if (i % 3 == 1) {
        out.Candidates(candidates[i]);
        draft[i] = candidates[i].ids[0];
      }
      out.tokens.push_back(draft[i]);
      chains[i].push_back(draft[i]);
      if (chains[i].size() < width[i])
        bodies.push_back({lanes[i].session.get(),
                          std::span<const std::int32_t>(&chains[i].back(), 1),
                          -1});
    }
    if (!bodies.empty() && !exec.MtpForwardBatch(bodies, &error))
      Throw("draft body", error);
  }
  std::vector<Executor::BatchItem> verify;
  for (const auto i : active)
    verify.push_back({lanes[i].session.get(), chains[i], chains[i].size() > 1});
  if (!exec.ForwardBatch(verify, &error))
    Throw("verification", error);
  std::uint32_t offset = 0;
  for (const auto i : active) {
    auto& lane = lanes[i];
    const auto rows = static_cast<std::uint32_t>(chains[i].size());
    std::vector<float> logits(std::size_t{rows} * vocab);
    if (!exec.SelectBatchLogits(offset, rows, logits.data(), &error))
      Throw("logits", error);
    offset += rows;
    out.logits.insert(out.logits.end(), logits.begin(), logits.end());
    const auto keep = 1 + (cycle + static_cast<std::uint32_t>(i)) % rows;
    const auto start = lane.session->position() - rows;
    if (rows > 1 && !exec.Rollback(*lane.session, keep, &error))
      Throw("rollback", error);
    lane.history.insert(lane.history.end(), chains[i].begin(),
                        chains[i].begin() + keep);
    lane.base = start;
    exec.MtpRewind(*lane.session, start);
    lane.next = Argmax(std::span<const float>(logits).subspan(
        std::size_t{keep - 1} * vocab, vocab));
  }
}

std::vector<std::uint8_t> Snapshot(const Executor& exec, const Session& s) {
  std::string error;
  std::vector<std::uint8_t> bytes(exec.SnapshotBytes(s, 0));
  if (!exec.SaveSnapshot(s, 0, bytes, &error))
    Throw("snapshot", error);
  return bytes;
}

bool Same(const char* what, const Record& eager, const Record& graph,
          std::uint32_t cycle) {
  const bool same =
      eager.tokens == graph.tokens &&
      eager.logits.size() == graph.logits.size() &&
      eager.candidates.size() == graph.candidates.size() &&
      std::memcmp(eager.logits.data(), graph.logits.data(),
                  eager.logits.size() * sizeof(float)) == 0 &&
      std::memcmp(eager.candidates.data(), graph.candidates.data(),
                  eager.candidates.size() * sizeof(float)) == 0;
  if (!same)
    std::cerr << what << " differs at cycle " << cycle << '\n';
  return same;
}

int Run(const Executor& exec, const Plan& plan) {
  constexpr std::array<std::uint32_t, 6> kPrompts{5, 300, 2041, 2100, 64, 999};
  std::string error;
  // Two identical cohorts: one eager reference, one replaying graphs.
  std::array<std::vector<Lane>, 2> cohorts;
  for (auto& cohort : cohorts) {
    for (std::uint32_t i = 0; i < plan.lanes; ++i) {
      Lane lane;
      lane.session = exec.CreateSession(gufo::core::SessionMode::kSpeculative,
                                        2304, &error);
      if (!lane.session)
        Throw("session", error);
      for (std::uint32_t t = 0; t < kPrompts[i]; ++t)
        lane.history.push_back(static_cast<std::int32_t>(
            (t * 2654435761U + i * 40503U) % (exec.config().vocab_size - 2)));
      std::vector<float> logits(exec.config().vocab_size);
      // Chunked prefill as the engine feeds it: the draft block consumes
      // each chunk's first token over the previous chunk's kept rows.
      for (std::size_t t = 0; t < lane.history.size(); t += 512) {
        const auto chunk = std::span<const std::int32_t>(lane.history)
                               .subspan(t, std::min<std::size_t>(
                                               512, lane.history.size() - t));
        if (t > 0 &&
            !exec.MtpForward(*lane.session, chunk.first(1),
                             static_cast<std::int32_t>(
                                 exec.MtpPosition(*lane.session) - lane.base),
                             {}, &error))
          Throw("prefill catch-up", error);
        if (!exec.Forward(*lane.session, chunk, 1, logits.data(),
                          Executor::ForwardMode::kPrefill, &error))
          Throw("prefill", error);
        lane.base = static_cast<std::uint32_t>(
            t + chunk.size() - std::min<std::size_t>(chunk.size(), 8));
      }
      const std::array<std::int32_t, 3> table{7, 4093, 8000};
      if (!exec.AppendDraftRows(*lane.session, 2, table, &error))
        Throw("draft table", error);
      lane.next = Argmax(logits);
      cohort.push_back(std::move(lane));
    }
  }
  const auto replays = exec.BatchGraphReplays();
  for (std::uint32_t cycle = 0; cycle < plan.cycles; ++cycle) {
    std::array<Record, 2> records;
    for (std::size_t k = 0; k < 2; ++k) {
      exec.SetBatchGraphs(k == 1);
      Cycle(exec, cohorts[k], plan, cycle, records[k]);
    }
    if (!Same("batched outputs", records[0], records[1], cycle))
      return 1;
    for (const auto& r : records[0].logits) {
      if (!std::isfinite(r)) {
        std::cerr << "non-finite logits at cycle " << cycle << '\n';
        return 1;
      }
    }
  }
  exec.SetBatchGraphs(true);
  for (std::uint32_t i = 0; i < plan.lanes; ++i) {
    if (cohorts[0][i].history != cohorts[1][i].history ||
        Snapshot(exec, *cohorts[0][i].session) !=
            Snapshot(exec, *cohorts[1][i].session)) {
      std::cerr << "session " << i << " state differs\n";
      return 1;
    }
  }
  const auto replayed = exec.BatchGraphReplays() - replays;
  std::cout << "lanes=" << plan.lanes << " drafts=" << plan.drafts
            << " cycles=" << plan.cycles << " graph replays=" << replayed
            << '\n';
  // Every repeated shape replays once warmed: trunk pieces, catch-up,
  // draft bodies and heads.
  if (replayed < plan.cycles) {
    std::cerr << "too few graph replays\n";
    return 1;
  }
  return 0;
}

/// Rows [0, rows) of the latest Forward's feature taps.
std::vector<float> Features(const Executor& exec, std::uint32_t rows) {
  std::string error;
  const std::size_t count = std::size_t{rows} * exec.FeatureWidth();
  float* device = nullptr;
  if (hipMalloc(&device, count * sizeof(float)) != hipSuccess)
    throw std::runtime_error("feature buffer");
  std::vector<float> host(count);
  const bool ok = exec.CopyFeatures(0, rows, device, &error) &&
                  exec.SyncStream(&error) &&
                  hipMemcpy(host.data(), device, count * sizeof(float),
                            hipMemcpyDeviceToHost) == hipSuccess;
  (void)hipFree(device);
  if (!ok)
    Throw("features", error);
  return host;
}

/// A prompt's logits must not depend on how its prefill splits into
/// passes: a checkpoint restore finishes the prompt with a short pass, and a
/// wide pass takes other kernels than one of 95 rows or fewer. Nor may a
/// decode step's logits depend on whether it runs alone or batched with
/// another request. Bit for bit; the inner layers' feature taps name the
/// first layer whose residual differs.
int RouteChecks(const Executor& exec) {
  constexpr std::uint32_t kPrompt = 1619;
  constexpr std::array<std::uint32_t, 5> kSplits{1618, 1612, 1610, 1524, 1523};
  const auto& c = exec.config();
  const auto vocab = c.vocab_size;
  std::string error;
  const std::array<std::uint32_t, 3> taps{0, 1, 2};
  if (!exec.ConfigureFeatureTaps(taps, &error))
    Throw("feature taps", error);
  std::vector<std::int32_t> prompt(kPrompt);
  for (std::uint32_t t = 0; t < kPrompt; ++t)
    prompt[t] =
        static_cast<std::int32_t>((t * 2246822519U + 17U) % (vocab - 2));
  const auto all = std::span<const std::int32_t>(prompt);
  int failures = 0;
  // Prefill `first` tokens, then the rest; logits of the last row.
  const auto prefill = [&](Session& s, std::uint32_t first,
                           std::vector<float>* logits) {
    exec.EnableFeatureTaps(true);
    if (first != 0) {
      if (!exec.Forward(s, all.first(first), 1, logits->data(),
                        Executor::ForwardMode::kPrefill, &error))
        Throw("route prefill", error);
      // The draft block consumes the next chunk's first token over the
      // kept trunk rows, as the engine feeds a chunked prompt.
      if (s.mtp_enabled() &&
          !exec.MtpForward(s, all.subspan(first).first(1),
                           static_cast<std::int32_t>(
                               exec.MtpPosition(s) -
                               (first - std::min<std::uint32_t>(first, 8))),
                           {.kv_only = true}, &error))
        Throw("route catch-up", error);
    }
    if (!exec.Forward(s, all.subspan(first), 1, logits->data(),
                      Executor::ForwardMode::kPrefill, &error))
      Throw("route prefill", error);
    exec.EnableFeatureTaps(false);
  };
  for (const auto mode : {gufo::core::SessionMode::kAutoregressive,
                          gufo::core::SessionMode::kSpeculative}) {
    const char* mode_name = mode == gufo::core::SessionMode::kSpeculative
                                ? "speculative"
                                : "autoregressive";
    auto reference = exec.CreateSession(mode, 4096, &error);
    if (!reference)
      Throw("route session", error);
    std::vector<float> reference_logits(vocab);
    prefill(*reference, 0, &reference_logits);
    const auto reference_features = Features(exec, kPrompt);
    for (const std::uint32_t split : kSplits) {
      auto session = exec.CreateSession(mode, 4096, &error);
      if (!session)
        Throw("route session", error);
      std::vector<float> logits(vocab);
      prefill(*session, split, &logits);
      const std::uint32_t tail = kPrompt - split;
      const auto features = Features(exec, tail);
      const bool same = std::memcmp(logits.data(), reference_logits.data(),
                                    std::size_t{vocab} * sizeof(float)) == 0;
      std::cout << mode_name << " prefill " << split << " + " << tail
                << " rows: logits " << (same ? "exact" : "DIFFER");
      for (std::size_t tap = 0; tap < taps.size(); ++tap) {
        std::uint32_t differ = 0;
        for (std::uint32_t r = 0; r < tail; ++r) {
          const std::size_t width = exec.FeatureWidth();
          differ +=
              std::memcmp(features.data() + r * width + tap * c.hidden_size,
                          reference_features.data() + (split + r) * width +
                              tap * c.hidden_size,
                          c.hidden_size * sizeof(float)) != 0;
        }
        std::cout << ", layer " << taps[tap] << " rows differing " << differ;
      }
      std::cout << '\n';
      failures += !same;
    }
  }
  // Decode after the same single-pass prefill: alone, then batched with a
  // peer's one or eight rows (past eight rows the batched projections split
  // their launches).
  const std::vector<std::int32_t> other{3, 1, 4, 1, 5};
  const std::array<std::int32_t, 8> peer_chain{7, 9, 11, 13, 15, 17, 19, 21};
  std::vector<float> alone;
  for (const std::uint32_t peer_rows : {0U, 1U, 8U}) {
    auto session =
        exec.CreateSession(gufo::core::SessionMode::kAutoregressive, 4096);
    auto peer =
        exec.CreateSession(gufo::core::SessionMode::kAutoregressive, 4096);
    std::vector<float> logits(vocab);
    prefill(*session, 0, &logits);
    if (!exec.Forward(*peer, other, 1, nullptr, Executor::ForwardMode::kPrefill,
                      &error))
      Throw("route peer prefill", error);
    const std::int32_t next = Argmax(logits);
    std::vector<float> decoded(vocab);
    if (peer_rows == 0) {
      if (!exec.Forward(*session, std::span<const std::int32_t>(&next, 1), 1,
                        decoded.data(), Executor::ForwardMode::kDecode, &error))
        Throw("route decode", error);
      alone = decoded;
      continue;
    }
    const std::array<Executor::BatchItem, 2> batch{
        Executor::BatchItem{
            peer.get(),
            std::span<const std::int32_t>(peer_chain).first(peer_rows), false},
        Executor::BatchItem{session.get(),
                            std::span<const std::int32_t>(&next, 1), false}};
    if (!exec.ForwardBatch(batch, &error) ||
        !exec.SelectBatchLogits(peer_rows, 1, decoded.data(), &error))
      Throw("route batched decode", error);
    const bool same = std::memcmp(decoded.data(), alone.data(),
                                  std::size_t{vocab} * sizeof(float)) == 0;
    std::cout << "decode batched with " << peer_rows
              << " peer rows: " << (same ? "exact" : "DIFFERS") << '\n';
    failures += !same;
  }
  return failures == 0 ? 0 : 1;
}

/// The draft block's prompt state must not depend on how the prompt splits
/// into passes either: a pass's own rows (K/V-only, in-pass) and the single
/// row each later pass consumes first take one route. Compares the draft
/// K/V and indexer rows bit for bit, then the full-head candidates of the
/// first decode catch-up, which reads them.
int MtpSplitChecks(const Executor& exec) {
  constexpr std::uint32_t kPrompt = 1619;
  const std::vector<std::vector<std::uint32_t>> plans{
      {1618, kPrompt},
      {1612, kPrompt},
      {1610, kPrompt},
      {1524, kPrompt},
      {1523, kPrompt},
      {1, kPrompt},
      {2, 3, 100, 1100, kPrompt},
      {1023, 1024, 1121, kPrompt}};
  const auto& c = exec.config();
  std::string error;
  std::vector<std::int32_t> prompt(kPrompt);
  for (std::uint32_t t = 0; t < kPrompt; ++t)
    prompt[t] =
        static_cast<std::int32_t>((t * 2246822519U + 17U) % (c.vocab_size - 2));
  const auto all = std::span<const std::int32_t>(prompt);
  const std::int32_t next = 4093;
  struct State {
    std::vector<std::uint8_t> draft;  ///< snapshot bytes of the draft block
    std::size_t bytes{0};
    std::uint32_t position{0};
    std::vector<float> candidates;
    std::vector<std::int32_t> ids;
  };
  const auto run = [&](std::span<const std::uint32_t> ends) {
    auto s =
        exec.CreateSession(gufo::core::SessionMode::kSpeculative, 4096, &error);
    if (!s)
      Throw("draft split session", error);
    std::vector<float> logits(c.vocab_size);
    std::uint32_t begin = 0;
    std::uint32_t base = 0;
    for (const auto end : ends) {
      if (begin != 0 && !exec.MtpForward(*s, all.subspan(begin).first(1),
                                         static_cast<std::int32_t>(
                                             exec.MtpPosition(*s) - base),
                                         {.kv_only = true}, &error))
        Throw("draft split catch-up", error);
      if (!exec.Forward(*s, all.subspan(begin, end - begin), 1, logits.data(),
                        Executor::ForwardMode::kPrefill, &error))
        Throw("draft split prefill", error);
      base = end - std::min<std::uint32_t>(end - begin, 8);
      begin = end;
    }
    State state;
    state.position = exec.MtpPosition(*s);
    const auto snapshot = Snapshot(exec, *s);
    state.bytes = snapshot.size();
    // The draft block's sections close the payload: unpooled indexer keys,
    // then K and V rows (no pooled blocks below the sparse budget, no
    // carried residual after K/V-only catch-up, no kept trunk rows).
    const std::size_t draft =
        std::size_t{state.position} *
        (c.indexer_head_dim * sizeof(float) + 2 * c.AttentionKvDim() * 2);
    if (draft > snapshot.size())
      Throw("draft split snapshot", "payload too short");
    state.draft.assign(snapshot.end() - static_cast<std::ptrdiff_t>(draft),
                       snapshot.end());
    // The first decode step's catch-up reads every draft row.
    std::vector<std::int32_t> replay(all.begin() + state.position + 1,
                                     all.end());
    replay.push_back(next);
    qfn::MtpCandidateLogits candidates;
    if (!exec.MtpForward(*s, replay,
                         static_cast<std::int32_t>(state.position - base),
                         {.candidates = &candidates}, &error))
      Throw("draft split decode catch-up", error);
    state.ids.assign(candidates.ids.begin(),
                     candidates.ids.begin() + candidates.size);
    state.candidates.assign(candidates.logits.begin(),
                            candidates.logits.begin() + candidates.size);
    return state;
  };
  const std::array<std::uint32_t, 1> whole{kPrompt};
  const auto reference = run(whole);
  if (reference.position != kPrompt - 1 || reference.ids.empty()) {
    std::cerr << "draft split reference is incomplete\n";
    return 1;
  }
  int failures = 0;
  for (const auto& plan : plans) {
    const auto state = run(plan);
    const bool same =
        state.position == reference.position &&
        state.bytes == reference.bytes && state.draft == reference.draft &&
        state.ids == reference.ids && state.candidates == reference.candidates;
    std::cout << "draft prefill passes";
    for (const auto end : plan)
      std::cout << ' ' << end;
    std::cout << ": draft K/V, indexer rows and candidates "
              << (same ? "exact" : "DIFFER") << '\n';
    failures += !same;
  }
  return failures == 0 ? 0 : 1;
}

/// Past the 2048-token sparse budget the indexer projections and block
/// scores decide which keys a query attends to. With passes ending on
/// four-token boundaries (the sparse attention tiles group four queries from
/// the pass start), the final logits and the draft block's candidates must
/// not depend on the split.
int SparseSplitChecks(const Executor& exec) {
  constexpr std::uint32_t kPrompt = 2600;
  const std::vector<std::vector<std::uint32_t>> plans{
      {1024, 2048, kPrompt},
      {2048, 2400, kPrompt},
      {1000, 2048, 2596, kPrompt},
      // One pass straddles the budget: its first rows are dense, the rest
      // select blocks.
      {1500, kPrompt},
      {2000, kPrompt}};
  const auto& c = exec.config();
  std::string error;
  std::vector<std::int32_t> prompt(kPrompt);
  for (std::uint32_t t = 0; t < kPrompt; ++t)
    prompt[t] =
        static_cast<std::int32_t>((t * 2654435761U + 7U) % (c.vocab_size - 2));
  const auto all = std::span<const std::int32_t>(prompt);
  const auto run = [&](gufo::core::SessionMode mode,
                       std::span<const std::uint32_t> ends) {
    auto s = exec.CreateSession(mode, 4096, &error);
    if (!s)
      Throw("sparse split session", error);
    std::vector<float> logits(c.vocab_size);
    std::uint32_t begin = 0;
    std::uint32_t base = 0;
    for (const auto end : ends) {
      if (begin != 0 && s->mtp_enabled() &&
          !exec.MtpForward(
              *s, all.subspan(begin).first(1),
              static_cast<std::int32_t>(exec.MtpPosition(*s) - base),
              {.kv_only = true}, &error))
        Throw("sparse split catch-up", error);
      if (!exec.Forward(*s, all.subspan(begin, end - begin), 1, logits.data(),
                        Executor::ForwardMode::kPrefill, &error))
        Throw("sparse split prefill", error);
      base = end - std::min<std::uint32_t>(end - begin, 8);
      begin = end;
    }
    if (s->mtp_enabled()) {
      const std::int32_t next = 4093;
      qfn::MtpCandidateLogits candidates;
      if (!exec.MtpForward(*s, std::span<const std::int32_t>(&next, 1),
                           static_cast<std::int32_t>(kPrompt - 1 - base),
                           {.candidates = &candidates}, &error))
        Throw("sparse split decode catch-up", error);
      logits.insert(logits.end(), candidates.logits.begin(),
                    candidates.logits.begin() + candidates.size);
    }
    return logits;
  };
  int failures = 0;
  for (const auto mode : {gufo::core::SessionMode::kAutoregressive,
                          gufo::core::SessionMode::kSpeculative}) {
    const std::array<std::uint32_t, 2> reference_plan{2048, kPrompt};
    const auto reference = run(mode, reference_plan);
    for (const auto& plan : plans) {
      const bool same = run(mode, plan) == reference;
      std::cout << (mode == gufo::core::SessionMode::kSpeculative
                        ? "speculative"
                        : "autoregressive")
                << " sparse prefill passes";
      for (const auto end : plan)
        std::cout << ' ' << end;
      std::cout << ": logits"
                << (mode == gufo::core::SessionMode::kSpeculative
                        ? " and draft candidates "
                        : " ")
                << (same ? "exact" : "DIFFER") << '\n';
      failures += !same;
    }
  }
  return failures == 0 ? 0 : 1;
}

}  // namespace

int main() {
  try {
    const auto config = SyntheticConfig();
    auto model = rocm::DeviceModelTestAccess::Build(config);
    std::string error;
    if (!model->SetDraftVocab(2048, {8000, 8001}, &error))
      Throw("draft vocabulary", error);
    // The n-gram rows come from a BF16 table file, as from the artifact.
    char path[] = "/tmp/qfn_batch_graph_XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0)
      throw std::runtime_error("n-gram table file");
    (void)unlink(path);
    {
      rocm::DeviceModelTestAccess rng{*model, 0xB16};
      std::vector<std::uint16_t> rows(config.ple_rows * config.ple_head_dim);
      for (auto& v : rows) {
        const float f = 0.1F * rng.Uniform();
        std::uint32_t bits = 0;
        std::memcpy(&bits, &f, 4);
        v = static_cast<std::uint16_t>(bits >> 16);
      }
      if (write(fd, rows.data(), rows.size() * 2) !=
          static_cast<ssize_t>(rows.size() * 2))
        throw std::runtime_error("n-gram table write");
    }
    auto ngram =
        qfn::NgramTable::Open(fd, 0, config.ple_rows, config.ple_head_dim,
                              gufo::core::GgmlType::kBF16, &error);
    (void)close(fd);
    if (!ngram)
      Throw("n-gram table", error);
    auto exec = Executor::Create(
        *model, ngram.get(),
        {.max_batch = 512, .max_logit_rows = 8, .max_speculative = 8}, &error);
    if (!exec)
      Throw("executor", error);
    // Two, four and six requests: shared projections below, at and above
    // the 8-row GEMV, and the 32-row tiles; one request crosses the sparse
    // attention budget, one starts past it.
    for (const Plan plan : {Plan{2, 3, 12}, Plan{4, 3, 16}, Plan{6, 7, 10}}) {
      if (Run(*exec, plan) != 0)
        return 1;
    }
    std::cout << "batched graph replay matches eager execution\n";
    exec.reset();
    // A whole prompt in one pass needs the wide scratch.
    auto wide = Executor::Create(
        *model, ngram.get(),
        {.max_batch = 2048, .max_logit_rows = 8, .max_speculative = 8}, &error);
    if (!wide)
      Throw("wide executor", error);
    if (RouteChecks(*wide) != 0 || MtpSplitChecks(*wide) != 0 ||
        SparseSplitChecks(*wide) != 0)
      return 1;
    std::cout << "prefill splits and batched decode match\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
