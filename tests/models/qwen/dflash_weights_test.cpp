// DFlash-2 GGUF loader contract: rotary dimension sources, layer-tap
// normalization and an optional check of a real export (argv[1]).
#include "src/models/qwen/dflash_weights.hpp"

#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace {

using gufo::core::GgmlType;
using gufo::core::GgufValueType;
using gufo::speculative::QwenDFlashWeights;

int failures = 0;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << "\n";
    ++failures;
  }
}

/// Minimal GGUF v3 writer: metadata, F32 tensors, 32-byte aligned data.
class GgufBuilder {
public:
  void String(std::string_view key, std::string_view value) {
    Key(key, GgufValueType::kString);
    Text(value);
  }
  void U32(std::string_view key, std::uint32_t value) {
    Key(key, GgufValueType::kUint32);
    Pod(value);
  }
  void F32(std::string_view key, float value) {
    Key(key, GgufValueType::kFloat32);
    Pod(value);
  }
  void Bool(std::string_view key, bool value) {
    Key(key, GgufValueType::kBool);
    Pod(static_cast<std::uint8_t>(value ? 1 : 0));
  }
  void I32Array(std::string_view key, const std::vector<std::int32_t>& values) {
    Key(key, GgufValueType::kArray);
    Pod(static_cast<std::uint32_t>(GgufValueType::kInt32));
    Pod(static_cast<std::uint64_t>(values.size()));
    for (const auto value : values)
      Pod(value);
  }
  void Tensor(std::string name, std::uint64_t elements) {
    tensors_.push_back({std::move(name), elements, data_bytes_});
    data_bytes_ += (elements * sizeof(float) + 31) / 32 * 32;
  }

  std::vector<std::uint8_t> Build() const {
    std::vector<std::uint8_t> out;
    const auto append = [&](const void* data, std::size_t bytes) {
      const auto* p = static_cast<const std::uint8_t*>(data);
      out.insert(out.end(), p, p + bytes);
    };
    append("GGUF", 4);
    const std::uint32_t version = 3;
    const std::uint64_t tensor_count = tensors_.size();
    append(&version, sizeof(version));
    append(&tensor_count, sizeof(tensor_count));
    append(&metadata_count_, sizeof(metadata_count_));
    out.insert(out.end(), metadata_.begin(), metadata_.end());
    for (const auto& t : tensors_) {
      const std::uint64_t length = t.name.size();
      append(&length, sizeof(length));
      append(t.name.data(), t.name.size());
      const std::uint32_t dims = 1;
      const auto type = static_cast<std::uint32_t>(GgmlType::kF32);
      append(&dims, sizeof(dims));
      append(&t.elements, sizeof(t.elements));
      append(&type, sizeof(type));
      append(&t.offset, sizeof(t.offset));
    }
    out.resize((out.size() + 31) / 32 * 32, 0);
    const std::size_t data_start = out.size();
    out.resize(data_start + data_bytes_, 0);
    // Unit weights; the loader rejects only a non-unit fc.scale.
    const float one = 1.0F;
    for (std::size_t i = data_start; i + sizeof(float) <= out.size();
         i += sizeof(float))
      std::memcpy(out.data() + i, &one, sizeof(one));
    return out;
  }

private:
  struct TensorRecord {
    std::string name;
    std::uint64_t elements;
    std::uint64_t offset;
  };
  template<class T>
  void Pod(T value) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(&value);
    metadata_.insert(metadata_.end(), p, p + sizeof(T));
  }
  void Text(std::string_view text) {
    Pod(static_cast<std::uint64_t>(text.size()));
    metadata_.insert(metadata_.end(), text.begin(), text.end());
  }
  void Key(std::string_view key, GgufValueType type) {
    Text(key);
    Pod(static_cast<std::uint32_t>(type));
    ++metadata_count_;
  }

  std::vector<std::uint8_t> metadata_;
  std::uint64_t metadata_count_{0};
  std::vector<TensorRecord> tensors_;
  std::uint64_t data_bytes_{0};
};

constexpr std::uint32_t kHidden = 32;
constexpr std::uint32_t kHeads = 2;
constexpr std::uint32_t kKvHeads = 1;
constexpr std::uint32_t kHeadDim = 16;
constexpr std::uint32_t kIntermediate = 64;
constexpr std::uint32_t kConvKernel = 2;
constexpr std::uint32_t kConvGroup = 16;
constexpr std::uint32_t kRank = 4;
constexpr std::uint32_t kVocab = 64;

/// A one-layer draft with two taps; `rope` adds the rotary metadata variant.
std::optional<QwenDFlashWeights> Load(
    const std::function<void(GgufBuilder&)>& rope, std::string* error) {
  GgufBuilder b;
  b.String("general.architecture", "dflash");
  b.U32("dflash.embedding_length", kHidden);
  b.U32("dflash.feed_forward_length", kIntermediate);
  b.U32("dflash.attention.head_count", kHeads);
  b.U32("dflash.attention.head_count_kv", kKvHeads);
  b.U32("dflash.attention.key_length", kHeadDim);
  b.U32("dflash.context_length", 4096);
  b.U32("dflash.block_count", 1);
  b.F32("dflash.rope.freq_base", 10000.0F);
  b.U32("dflash.block_size", 4);
  b.U32("dflash.conv_kernel_size", kConvKernel);
  b.U32("dflash.conv_group_size", kConvGroup);
  b.U32("dflash.selector_rank", kRank);
  b.U32("dflash.selector_top_k", 2);
  b.U32("dflash.attention.sliding_window", 8);
  b.Bool("dflash.attention.causal", false);
  b.I32Array("dflash.target_layers", {6, 16});
  b.U32("tokenizer.ggml.mask_token_id", 7);
  rope(b);

  const std::uint64_t attention = kHeads * kHeadDim;
  const std::uint64_t kv = kKvHeads * kHeadDim;
  const std::uint64_t dynamic = 2ULL * kConvKernel * (kHidden / kConvGroup);
  b.Tensor("selector_predecessor.weight", kRank * kVocab);
  b.Tensor("selector_successor.weight", kRank * kVocab);
  b.Tensor("selector_hidden.weight", kHidden * kRank);
  b.Tensor("fc.weight", kHidden * 2ULL * kHidden);
  b.Tensor("enc.output_norm.weight", kHidden);
  b.Tensor("output_norm.weight", kHidden);
  b.Tensor("blk.0.attn_norm.weight", kHidden);
  b.Tensor("blk.0.attn_q.weight", attention * kHidden);
  b.Tensor("blk.0.attn_k.weight", kv * kHidden);
  b.Tensor("blk.0.attn_v.weight", kv * kHidden);
  b.Tensor("blk.0.attn_output.weight", kHidden * attention);
  b.Tensor("blk.0.attn_q_norm.weight", kHeadDim);
  b.Tensor("blk.0.attn_k_norm.weight", kHeadDim);
  b.Tensor("blk.0.ffn_norm.weight", kHidden);
  b.Tensor("blk.0.ffn_gate.weight", kIntermediate * kHidden);
  b.Tensor("blk.0.ffn_up.weight", kIntermediate * kHidden);
  b.Tensor("blk.0.ffn_down.weight", kHidden * kIntermediate);
  b.Tensor("blk.0.attn_conv_base", kHidden * kConvKernel * 2ULL);
  b.Tensor("blk.0.attn_conv_proj.weight", dynamic * kHidden);
  b.Tensor("blk.0.ffn_conv_base", kHidden * kConvKernel * 2ULL);
  b.Tensor("blk.0.ffn_conv_proj.weight", dynamic * kHidden);

  const auto bytes = b.Build();
  const auto path =
      std::filesystem::temp_directory_path() /
      ("gufo_dflash_weights_test_" + std::to_string(getpid()) + ".gguf");
  {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  }
  auto reader = gufo::core::GgufReader::OpenFile(path, error);
  std::filesystem::remove(path);
  if (!reader)
    return std::nullopt;
  return QwenDFlashWeights::LoadFromGguf(*reader, error);
}

void CheckRotarySources() {
  std::string error;
  // Neither key: plain RoPE over the whole head.
  auto plain = Load([](GgufBuilder&) {}, &error);
  Expect(plain.has_value(), "draft without rotary metadata loads: " + error);
  if (plain) {
    Expect(plain->config.rotary_dim == kHeadDim,
           "missing rotary metadata rotates the whole head");
    Expect(plain->config.head_dim == kHeadDim, "head dimension is kept");
    Expect(plain->config.vocab_size == kVocab, "vocabulary from the selector");
    // GGUF layer-input numbering 6, 16 names zero-based outputs 5, 15.
    Expect(plain->dflash_config.target_layer_ids ==
               std::vector<std::uint32_t>{5, 15},
           "target layers normalize to zero-based layer outputs");
    Expect(plain->dflash_config.mask_token_id == 7, "mask token");
  }

  auto sections = Load(
      [](GgufBuilder& b) {
        b.I32Array("dflash.rope.dimension_sections", {4, 0, 0, 0});
      },
      &error);
  Expect(sections.has_value() && sections->config.rotary_dim == 8,
         "degenerate M-RoPE sections give twice the first section");

  auto count = Load(
      [](GgufBuilder& b) { b.U32("dflash.rope.dimension_count", 12); }, &error);
  Expect(count.has_value() && count->config.rotary_dim == 12,
         "explicit rotary dimension wins");

  error.clear();
  auto invalid = Load(
      [](GgufBuilder& b) {
        b.I32Array("dflash.rope.dimension_sections", {4, 4, 0, 0});
      },
      &error);
  Expect(!invalid.has_value() && !error.empty(),
         "non-degenerate M-RoPE sections remain an error");
}

/// Optional real-export check (CPU only, nothing is uploaded).
void CheckExport(const char* path) {
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  Expect(reader != nullptr, "export opens: " + error);
  if (!reader)
    return;
  const auto weights = QwenDFlashWeights::LoadFromGguf(*reader, &error);
  Expect(weights.has_value(), "export loads: " + error);
  if (!weights)
    return;
  const auto& c = weights->config;
  const auto& d = weights->dflash_config;
  std::cout << "export: hidden=" << c.hidden_size << " layers=" << c.num_layers
            << " heads=" << c.num_attention_heads << "/"
            << c.num_key_value_heads << " head_dim=" << c.head_dim
            << " rotary_dim=" << c.rotary_dim << " theta=" << c.rope_theta
            << " vocab=" << c.vocab_size << " block=" << d.block_size
            << " mask=" << d.mask_token_id << " window=" << d.sliding_window
            << " taps=";
  for (const auto layer : d.target_layer_ids)
    std::cout << layer << ' ';
  std::cout << "\n";
  Expect(c.rotary_dim == c.head_dim, "export rotates the whole head");
}

}  // namespace

int main(int argc, char** argv) {
  CheckRotarySources();
  if (argc > 1)
    CheckExport(argv[1]);
  if (failures != 0)
    return 1;
  std::cout << "qwen_dflash_weights_test: ALL TESTS PASSED\n";
  return 0;
}
