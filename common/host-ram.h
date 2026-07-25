// SPDX-License-Identifier: MIT
// Copyright (c) 2026 fewtarius
//
// Cross-platform available-RAM query used by auto-sizing code paths in
// common/ and by the Vulkan FA scratch gate in ggml/src/ggml-vulkan/.
//
// Two access patterns:
//   host_available_ram_query()  - returns a real reading or signals unknown;
//                                 callers that must distinguish "real" from
//                                 "fallback guess" should use this.
//   host_available_ram()        - same but with an 8 GiB conservative
//                                 fallback so legacy auto-sizing callers
//                                 don't have to think about it.
//
// Both share a thread-safe 5-second TTL cache so repeated calls in hot
// paths (Vulkan FA scratch gate per prefill layer; SSD auto-sizing at
// conversation-create time) don't hammer /proc/meminfo or Mach host ports.

#pragma once

#include <cstddef>

namespace common {

// True and sets *out_bytes to currently available RAM (free + reclaimable) on
// supported platforms. Linux falls back to sysinfo().freeram when MemAvailable
// is unavailable. False (and *out_bytes = 0) only when neither query works or
// the platform is unsupported. Out param must be non-null.
bool host_available_ram_query(std::size_t * out_bytes);

// Currently available RAM in bytes. On Linux/macOS this is the reclaimable
// memory (free + cache / inactive). On unsupported platforms an 8 GiB fallback.
// Calls share a 5-second TTL cache; frequent callers see cached results.
std::size_t host_available_ram();

}  // namespace common
