#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_DFLASH_CHOICE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_DFLASH_CHOICE_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include "src/models/qwen38_flash_next/mtp_costs.hpp"

namespace gufo::models::qwen38_flash_next {

/// A DFlash2 block ends after the first proposal whose draft-head
/// probability is below this gate, as an MTP chain does (mtp_policy.hpp).
inline constexpr float kDFlashGateProbability = 0.9F;

/// Chooses the neural draft source of a single-session greedy cycle from
/// committed acceptance history only, never request timings: each source's
/// recent tokens per cycle over the calibrated C1 cycle cost of its recent
/// verify width (mtp_costs.hpp). Greedy output does not depend on the choice.
class DraftSourceChooser {
public:
  /// Cycles each source runs before the comparison applies.
  static constexpr std::uint32_t kProbeCycles = 2;
  /// The losing source runs once per period so its estimate stays current.
  static constexpr std::uint32_t kReprobePeriod = 16;
  /// DFlash must commit this much faster to replace MTP: its block costs
  /// about what the MTP chain it replaces costs, and MTP is the default.
  static constexpr float kDFlashMargin = 1.05F;
  /// Weight of the newest cycle: a request switching from code to prose
  /// moves an estimate most of the way within a few cycles.
  static constexpr float kAlpha = 0.3F;

  void Reset() noexcept { *this = DraftSourceChooser{}; }

  [[nodiscard]] bool PreferDFlash(std::uint32_t context) const noexcept {
    if (dflash_.cycles < kProbeCycles)
      return true;
    if (mtp_.cycles < kProbeCycles)
      return false;
    const auto costs = MtpCycleCosts(context, 1);
    const bool dflash =
        Rate(dflash_, costs) > kDFlashMargin * Rate(mtp_, costs);
    return cycles_ % kReprobePeriod == kReprobePeriod - 1 ? !dflash : dflash;
  }

  /// One cycle of `source`: `committed` tokens from `width` verified rows
  /// (1 for an ordinary decode step).
  void Observe(bool dflash, std::uint32_t committed,
               std::uint32_t width) noexcept {
    if (committed == 0 || width == 0)
      return;
    auto& s = dflash ? dflash_ : mtp_;
    s.committed = (1.0F - kAlpha) * s.committed + kAlpha * committed;
    s.width = (1.0F - kAlpha) * s.width + kAlpha * width;
    s.weight = (1.0F - kAlpha) * s.weight + kAlpha;
    ++s.cycles;
    ++cycles_;
  }

private:
  struct Source {
    float committed{0.0F};
    float width{0.0F};
    float weight{0.0F};  ///< normalizes the zero-initialized averages
    std::uint32_t cycles{0};
  };

  /// Committed tokens per calibrated millisecond at the mean verify width.
  [[nodiscard]] static float Rate(const Source& s,
                                  const std::array<float, 8>& costs) noexcept {
    const float width = std::clamp(s.width / s.weight, 1.0F, 8.0F);
    const auto low = static_cast<std::size_t>(width) - 1;
    const auto high = std::min<std::size_t>(low + 1, costs.size() - 1);
    const float fraction = width - static_cast<float>(low + 1);
    const float cost = costs[low] + fraction * (costs[high] - costs[low]);
    return s.committed / s.weight / cost;
  }

  Source dflash_;
  Source mtp_;
  std::uint32_t cycles_{0};
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_DFLASH_CHOICE_HPP_
