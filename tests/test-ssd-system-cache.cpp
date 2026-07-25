// SPDX-License-Identifier: MIT
// Copyright (c) 2026 fewtarius

#undef NDEBUG

#include "kv-ssd-system-cache.h"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static fs::path make_scratch() {
    const fs::path  path = fs::temp_directory_path() / "ssd_system_cache_test";
    std::error_code ec;
    fs::remove_all(path, ec);
    fs::create_directories(path, ec);
    if (ec) {
        throw std::runtime_error("could not create scratch directory: " + ec.message());
    }
    return path;
}

static void test_exact_load_and_mismatch() {
    const fs::path path      = make_scratch();
    const uint32_t tokens[]  = { 11, 22, 33, 44 };
    const uint32_t changed[] = { 11, 22, 99, 44 };
    const uint8_t  state[]   = { 1, 2, 3, 4, 5 };

    kv_ssd_system_cache cache;
    assert(cache.init(path.string(), 0x1234));
    assert(cache.store(tokens, 4, state, sizeof(state)));

    std::vector<uint8_t> loaded;
    assert(cache.load(tokens, 4, loaded));
    assert(loaded == std::vector<uint8_t>(state, state + sizeof(state)));
    assert(!cache.load(changed, 4, loaded));

    fs::remove_all(path);
}

static void test_unverifiable_long_prompt_is_not_loaded() {
    const fs::path        path = make_scratch();
    std::vector<uint32_t> tokens(4097, 7);
    const uint8_t         state[] = { 9, 8, 7 };

    kv_ssd_system_cache cache;
    assert(cache.init(path.string(), 0));
    assert(!cache.store(tokens.data(), (uint32_t) tokens.size(), state, sizeof(state)));

    std::vector<uint8_t> loaded;
    assert(!cache.load(tokens.data(), (uint32_t) tokens.size(), loaded));

    fs::remove_all(path);
}

int main() {
    test_exact_load_and_mismatch();
    test_unverifiable_long_prompt_is_not_loaded();
    return 0;
}
