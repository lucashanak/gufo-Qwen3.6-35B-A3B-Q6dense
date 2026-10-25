#include "q36_mmq_prelude.h"
namespace q36_mmq {
// SPDX-License-Identifier: MIT
// Implementations of the ggml-API stubs declared in q36_ggml_stubs.h plus
// the bodies of ggml_backend_hip_context / ggml_hip_info /
// new_pool_for_device that the vendored common.hpp declares without
// defining.
//

#include "common.hpp"   // pulls in q36_ggml_stubs.h via redirect headers

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <vector>

// ----------------------------------------------------------------------------
// Device info singleton.
//
// Common.hpp declares `const ggml_hip_device_info & ggml_hip_info();` -
// we provide the body. The struct layout (common.hpp:1091) is:
//   { int device_count; hip_device_info devices[GGML_HIP_MAX_DEVICES];
//     std::array<float, GGML_HIP_MAX_DEVICES> default_tensor_split; }
// where hip_device_info has { cc, nsm, smpb, smpbo, integrated, vmm,
// vmm_granularity, total_vram, warp_size, supports_cooperative_launch }.
// ----------------------------------------------------------------------------

const ggml_hip_device_info & ggml_hip_info() {
    static ggml_hip_device_info info;
    static std::once_flag once;
    std::call_once(once, []{
        int count = 0;
        hipError_t err = hipGetDeviceCount(&count);
        if (err != hipSuccess) {
            fprintf(stderr, "ggml_hip_info: hipGetDeviceCount failed: %s\n", hipGetErrorString(err));
            count = 0;
        }
        if (count > GGML_HIP_MAX_DEVICES) count = GGML_HIP_MAX_DEVICES;
        info.device_count = count;
        for (int i = 0; i < count; i++) {
            hipDeviceProp_t p;
            HIP_CHECK(hipGetDeviceProperties(&p, i));
            unsigned arch = 0;
            if (sscanf(p.gcnArchName, "gfx%x", &arch) != 1) {
                fprintf(stderr, "ggml_hip_info: unsupported HIP architecture '%s'\n", p.gcnArchName);
            }
            info.devices[i].cc                          = GGML_HIP_CC_OFFSET_AMD + (int)arch;
            info.devices[i].nsm                         = p.multiProcessorCount;
            info.devices[i].smpb                        = p.sharedMemPerBlock;
            info.devices[i].smpbo                       = p.sharedMemPerBlockOptin;
            info.devices[i].integrated                  = p.integrated != 0;
            info.devices[i].vmm                         = false;
            info.devices[i].vmm_granularity             = 0;
            info.devices[i].total_vram                  = p.totalGlobalMem;
            info.devices[i].warp_size                   = p.warpSize;
            info.devices[i].supports_cooperative_launch = p.cooperativeLaunch != 0;
        }
    });
    return info;
}

// ggml_hip_get_device / ggml_hip_set_device are declared (not defined) in
// common.hpp. We provide thin wrappers.

int ggml_hip_get_device() {
    int dev = 0;
    hipGetDevice(&dev);
    return dev;
}

void ggml_hip_set_device(int device) {
    int cur = -1;
    hipGetDevice(&cur);
    if (cur != device) {
        HIP_CHECK(hipSetDevice(device));
    }
}

int64_t ggml_time_us() {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0).count();
}

// ----------------------------------------------------------------------------
// ggml_hip_error: invoked by the HIP_CHECK / HIPBLAS_CHECK macros defined
// in common.hpp on the error path. Marked [[noreturn]] in the declaration
// (common.hpp:155) - abort() satisfies that contract.
// ----------------------------------------------------------------------------

[[noreturn]] void ggml_hip_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    fprintf(stderr, "HIP error: %s\n  call: %s\n  in: %s at %s:%d\n", msg, stmt, func, file, line);
    fflush(stderr);
    abort();
}

// ----------------------------------------------------------------------------
// Scratch pool over one device arena.
// ----------------------------------------------------------------------------

namespace {

/* qwen38: every allocation the tier makes is scoped to one entry point and
 * released in reverse order, and every kernel runs on the executor's single
 * stream, so a stack over one device arena is enough. It replaces
 * hipMallocAsync/hipFreeAsync, whose reuse of a just-freed block raced with
 * in-flight kernels on this ROCm and made prefill logits nondeterministic. */
struct q36_stack_pool : public ggml_hip_pool {
    struct chunk {
        char *base;
        size_t capacity;
        size_t top;
    };

    int device;
    std::vector<chunk> chunks;
    size_t cur = 0;
    std::vector<std::pair<char *, size_t>> live;

    explicit q36_stack_pool(int device) : device(device) {}

    ~q36_stack_pool() override {
        for (const chunk &c : chunks) (void)hipFree(c.base);
    }

    /* Captured decode graphs keep the scratch addresses they were recorded with,
     * so no chunk is ever freed or moved. A request that does not fit continues
     * in the next chunk, adding one when none is left; a drained pool starts
     * again at the first chunk, at the addresses it had before. */
    void * alloc(size_t size, size_t * actual_size) override {
        ggml_hip_set_device(device);
        const size_t aligned = (size + 255) & ~(size_t) 255;
        while (chunks.empty() || chunks[cur].top + aligned > chunks[cur].capacity) {
            if (!chunks.empty() && cur + 1 < chunks.size()) {
                ++cur;
                continue;
            }
            size_t total = 0;
            for (const chunk &c : chunks) total += c.capacity;
            size_t next = total ? total : (size_t) 256 << 20;
            while (next < aligned) next *= 2;
            char *base = nullptr;
            HIP_CHECK(hipMalloc((void **) &base, next));
            chunks.push_back({base, next, 0});
            cur = chunks.size() - 1;
            if (cur > 0) {
                fprintf(stderr, "q36_stack_pool: added a %zu MiB scratch chunk (%zu MiB in total)\n", next >> 20,
                        (total + next) >> 20);
            }
        }
        chunk &c = chunks[cur];
        char *ptr = c.base + c.top;
        c.top += aligned;
        live.emplace_back(ptr, aligned);
        if (actual_size) *actual_size = aligned;
        return ptr;
    }

    void free(void * ptr, size_t /*size*/) override {
        if (!ptr) return;
        if (live.empty() || live.back().first != ptr) {
            fprintf(stderr, "q36_stack_pool: non-LIFO free\n");
            abort();
        }
        char *p = static_cast<char *>(ptr);
        size_t i = cur;
        while (i > 0 && (p < chunks[i].base || p >= chunks[i].base + chunks[i].capacity)) --i;
        chunks[i].top -= live.back().second;
        live.pop_back();
        while (cur > 0 && chunks[cur].top == 0) --cur;
    }
};

} // anonymous namespace

std::unique_ptr<ggml_hip_pool> ggml_backend_hip_context::new_pool_for_device(int device, int /*stream_no*/) {
    return std::unique_ptr<ggml_hip_pool>(new q36_stack_pool(device));
}

ggml_backend_hip_context::~ggml_backend_hip_context() {
    if (copy_event) {
        hipEventDestroy(copy_event);
        copy_event = nullptr;
    }
    // streams[][], hipblas_handles[], and pools[][] are owned-by-value
    // (hipStream_t and hipblasHandle_t are opaque handles - destroying the
    // context "should" tear them down, but in our shim ds4 manages streams
    // externally and we leave them alone. The pools auto-destruct via
    // unique_ptr.).
}

}  // namespace q36_mmq
