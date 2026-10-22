// DraftSourceChooser: probes both sources, prefers the one committing more
// tokens per calibrated millisecond, re-probes the loser once per period and
// follows a change of content within a few cycles.
#include "src/models/qwen38_flash_next/dflash_choice.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string_view>

namespace {

using gufo::models::qwen38_flash_next::DraftSourceChooser;

int failures = 0;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << "\n";
    ++failures;
  }
}

struct Cycle {
  std::uint32_t committed;
  std::uint32_t width;
};

/// Runs `cycles` decisions, feeding each source its fixed outcome; returns
/// how many cycles chose DFlash.
std::uint32_t Run(DraftSourceChooser& chooser, std::uint32_t cycles,
                  Cycle dflash, Cycle mtp, std::uint32_t context = 0) {
  std::uint32_t chosen = 0;
  for (std::uint32_t i = 0; i < cycles; ++i) {
    const bool use = chooser.PreferDFlash(context);
    chosen += use;
    const auto cycle = use ? dflash : mtp;
    chooser.Observe(use, cycle.committed, cycle.width);
  }
  return chosen;
}

void CheckProbes() {
  DraftSourceChooser chooser;
  for (std::uint32_t i = 0; i < DraftSourceChooser::kProbeCycles; ++i) {
    Expect(chooser.PreferDFlash(0), "a fresh request probes DFlash first");
    chooser.Observe(true, 1, 8);
  }
  for (std::uint32_t i = 0; i < DraftSourceChooser::kProbeCycles; ++i) {
    Expect(!chooser.PreferDFlash(0), "then it probes MTP");
    chooser.Observe(false, 3, 4);
  }
  chooser.Reset();
  Expect(chooser.PreferDFlash(0), "Reset starts a new request's probes");
}

void CheckPreference() {
  // Code: DFlash commits 5 tokens from 6 rows, MTP 3 from 3.
  DraftSourceChooser code;
  const auto code_dflash = Run(code, 64, {5, 6}, {3, 3});
  Expect(code_dflash >= 56, "long accepted blocks keep DFlash");
  Expect(code_dflash <= 64 - 64 / DraftSourceChooser::kReprobePeriod + 1,
         "MTP is re-probed once per period");

  // Prose: DFlash commits 2 tokens from 5 rows, MTP 2 from 2.
  DraftSourceChooser prose;
  const auto prose_dflash = Run(prose, 64, {2, 5}, {2, 2});
  Expect(prose_dflash <= DraftSourceChooser::kProbeCycles + 64 / 16 + 1,
         "short DFlash blocks fall back to MTP");

  // Equal tokens at equal width: the margin keeps MTP.
  DraftSourceChooser tie;
  const auto tie_dflash = Run(tie, 64, {3, 4}, {3, 4});
  Expect(tie_dflash <= DraftSourceChooser::kProbeCycles + 64 / 16 + 1,
         "a tie keeps MTP");
}

void CheckAdaptation() {
  DraftSourceChooser chooser;
  (void)Run(chooser, 32, {6, 7}, {2, 3});
  Expect(chooser.PreferDFlash(0), "code phase prefers DFlash");
  // Prose begins: DFlash now commits 1-2 tokens per block.
  std::uint32_t switched_after = 0;
  for (std::uint32_t i = 1; i <= 16; ++i) {
    const bool use = chooser.PreferDFlash(0);
    chooser.Observe(use, use ? 1 : 2, use ? 7 : 3);
    if (!chooser.PreferDFlash(0)) {
      switched_after = i;
      break;
    }
  }
  Expect(switched_after != 0 && switched_after <= 4,
         "a content change falls back to MTP within a few cycles");
}

void CheckDeterminism() {
  DraftSourceChooser a;
  DraftSourceChooser b;
  bool same = true;
  for (std::uint32_t i = 0; i < 200; ++i) {
    const bool ua = a.PreferDFlash(i * 37);
    const bool ub = b.PreferDFlash(i * 37);
    same = same && ua == ub;
    const std::uint32_t committed = 1 + (i * 7) % 6;
    const std::uint32_t width = 1 + (i * 5) % 8;
    a.Observe(ua, committed, std::max(width, committed));
    b.Observe(ub, committed, std::max(width, committed));
  }
  Expect(same, "identical acceptance histories choose identically");
}

}  // namespace

int main() {
  CheckProbes();
  CheckPreference();
  CheckAdaptation();
  CheckDeterminism();
  if (failures != 0)
    return 1;
  std::cout << "qwen38_flash_next dflash_choice: ALL TESTS PASSED\n";
  return 0;
}
