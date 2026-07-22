// SPDX-License-Identifier: MIT
// Copyright (c) 2026 fewtarius

#include "host-ram.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#ifdef __linux__
#include <sys/sysinfo.h>
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/host_info.h>
#endif

namespace common {

// ---------------------------------------------------------------------------
// TTL cache — avoids repeated /proc/meminfo or host_statistics64 calls in
// hot paths (Vulkan FA scratch gate checked once per attention layer per
// prefill step; SSD cache auto-sizing at conversation-create time).
// Thread-safe via std::mutex; granularity is 5 seconds.
// ---------------------------------------------------------------------------

namespace {
struct ttl_cache {
    std::mutex                             mutex;
    std::chrono::steady_clock::time_point  last_query;
    bool                                   cached = false;
    bool                                   known  = false;
    std::size_t                            bytes  = 0;
};

ttl_cache g_ram_cache;

static const std::chrono::seconds CACHE_TTL(5);
}  // namespace
#ifdef __linux__
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return static_cast<std::size_t>(info.freeram) * info.mem_unit;
    }
    return 8ULL * 1024 * 1024 * 1024;
#elif defined(__APPLE__)
    // macOS: hw.memsize returns total physical RAM, but auto-sizing wants
    // currently free RAM. Read vm_statistics64: free_count is genuinely
    // unallocated, inactive_count is pages the page cache can evict on
    // demand (Apple's standard "reclaimable" bucket). Treat both as
    // available so callers can grow into memory they can actually reclaim.
    int64_t total = 0;
    std::size_t size = sizeof(total);
    if (sysctlbyname("hw.memsize", &total, &size, NULL, 0) != 0 || total <= 0) {
        return 8ULL * 1024 * 1024 * 1024;
    }
    mach_port_t host = mach_host_self();
    vm_size_t page_size = 0;
    if (host_page_size(host, &page_size) != KERN_SUCCESS || page_size == 0) {
        return static_cast<std::size_t>(total);
    }
    vm_statistics64_data_t vm_stats;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    kern_return_t kr = host_statistics64(
        host, HOST_VM_INFO64, (host_info64_t) &vm_stats, &count);
    if (kr != KERN_SUCCESS) {
        return static_cast<std::size_t>(total);
    }
    std::size_t free_pages = static_cast<std::size_t>(vm_stats.free_count) +
                             static_cast<std::size_t>(vm_stats.inactive_count);
    return free_pages * static_cast<std::size_t>(page_size);
#else
    return 8ULL * 1024 * 1024 * 1024;
#endif
}

bool host_available_ram_query(std::size_t * out_bytes) {
    if (!out_bytes) return false;

    // Serve from TTL cache when fresh to avoid repeated /proc reads.
    {
        std::lock_guard<std::mutex> lock(g_ram_cache.mutex);
        auto now = std::chrono::steady_clock::now();
        if (g_ram_cache.cached && (now - g_ram_cache.last_query) < CACHE_TTL) {
            *out_bytes = g_ram_cache.bytes;
            return g_ram_cache.known;
        }
    }

    bool known = host_available_ram_impl(out_bytes);

    {
        std::lock_guard<std::mutex> lock(g_ram_cache.mutex);
        g_ram_cache.last_query = std::chrono::steady_clock::now();
        g_ram_cache.cached     = true;
        g_ram_cache.known      = known;
        g_ram_cache.bytes      = *out_bytes;
    }

    return known;
}

std::size_t host_available_ram() {
    std::size_t bytes = 0;
    if (host_available_ram_query(&bytes)) {
        return bytes;
    }
    // Legacy callers (SSD cache auto-sizing) want a number to plan against even
    // when the platform can't answer reliably. 8 GiB is conservative and
    // matches the original hardcoded fallback these callers used to have.
    return 8ULL * 1024 * 1024 * 1024;
}

}  // namespace common
