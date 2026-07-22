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
// Read /proc/meminfo MemAvailable. This is the kernel-calculated estimate of
// memory available for new allocations without swapping - it includes the
// reclaimable page cache. Available since Linux 3.14 (2014).
// Returns 0 if the file is unreadable or MemAvailable line is missing.
static std::size_t read_meminfo_available() {
    FILE * f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;
    std::size_t result = 0;
    char line[256];
    while (std::fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, "MemAvailable:", 13) == 0) {
            unsigned long long kb = 0;
            if (std::sscanf(line + 13, " %llu kB", &kb) == 1) {
                result = static_cast<std::size_t>(kb) * 1024ULL;
            }
            break;
        }
    }
    std::fclose(f);
    return result;
}
#endif

static bool host_available_ram_impl(std::size_t * out_bytes) {
    if (!out_bytes) return false;
#ifdef __linux__
    std::size_t avail = read_meminfo_available();
    if (avail == 0) {
        // Kernel older than 3.14 - fall back to sysinfo.freeram (genuinely
        // free RAM only, no reclaimable cache). Conservative but real.
        struct sysinfo info;
        if (sysinfo(&info) != 0) {
            *out_bytes = 0;
            return false;
        }
        avail = static_cast<std::size_t>(info.freeram) * info.mem_unit;
    }
    *out_bytes = avail;
    return true;
#elif defined(__APPLE__)
    // free + inactive: free_count is genuinely unallocated, inactive_count is
    // pages the page cache can evict on demand (Apple's "reclaimable" bucket).
    mach_port_t host = mach_host_self();
    vm_size_t page_size = 0;
    if (host_page_size(host, &page_size) != KERN_SUCCESS || page_size == 0) {
        *out_bytes = 0;
        return false;
    }
    vm_statistics64_data_t vm_stats;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    kern_return_t kr = host_statistics64(
        host, HOST_VM_INFO64, (host_info64_t) &vm_stats, &count);
    if (kr != KERN_SUCCESS) {
        *out_bytes = 0;
        return false;
    }
    std::size_t free_pages = static_cast<std::size_t>(vm_stats.free_count) +
                             static_cast<std::size_t>(vm_stats.inactive_count);
    *out_bytes = free_pages * static_cast<std::size_t>(page_size);
    return true;
#else
    *out_bytes = 0;
    return false;
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
