// SPDX-License-Identifier: MIT
// Copyright (c) 2026 fewtarius
//
// Regression tests for on-disk checkpoint format handling in
// common/kv-ssd-cache.cpp.
//
// Two failure modes are covered, both reachable the first time a binary with
// a bumped KV_SSD_VERSION starts against a populated cache directory:
//
//   1. Records from an older format version were accepted. Only the *index*
//      file's version was checked; per-checkpoint records were validated on
//      magic alone, so a v3 record (keyed on get_tokens(), media placeholders
//      included) would be matched against a v4 text-token query and restore
//      KV state that does not correspond to the request.
//
//   2. next_id came only from the index file. A version bump makes that file
//      unreadable, so next_id reset to 1 while ckpt-1.bin..ckpt-N.bin were
//      still on disk - the next store overwrote a live checkpoint file whose
//      index entry still pointed at the old contents.
//
// Rejected files are also removed: eviction only deletes checkpoints present
// in the in-memory index, so a file rejected during the directory scan would
// otherwise occupy disk permanently.
#undef NDEBUG

#include "kv-ssd-cache.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static fs::path make_scratch() {
    const fs::path  path = fs::temp_directory_path() / "ssd_cache_format_test";
    std::error_code ec;
    fs::remove_all(path, ec);
    fs::create_directories(path, ec);
    if (ec) {
        throw std::runtime_error("could not create scratch directory: " + ec.message());
    }
    return path;
}

// Directory kv_ssd_init derives for a conversation hash (see ckpt_path()).
static fs::path conv_dir(const fs::path & base, uint64_t conv_hash) {
    char hex[32];
    snprintf(hex, sizeof(hex), "%016lx", (unsigned long) conv_hash);
    return base / hex;
}

static size_t count_ckpt_files(const fs::path & dir) {
    size_t n = 0;
    if (!fs::exists(dir)) {
        return 0;
    }
    for (const auto & e : fs::directory_iterator(dir)) {
        const std::string name = e.path().filename().string();
        if (name.rfind("ckpt-", 0) == 0 && name.size() > 9) {
            n++;
        }
    }
    return n;
}

// Store one checkpoint, then rewrite its record's version field in place so it
// looks like it came from an older build.
static void write_checkpoint_with_version(const fs::path & base, uint64_t conv_hash, uint32_t version) {
    kv_ssd_config cfg;
    cfg.auto_size = false;

    kv_ssd_cache * cache = kv_ssd_init(base.string().c_str(), &cfg, conv_hash);
    assert(cache != nullptr);

    const std::vector<uint8_t>  payload(4096, 0xAB);
    const std::vector<uint32_t> tokens = { 1, 2, 3, 4, 5 };

    const uint64_t id = kv_ssd_store(cache, 0, payload.data(), payload.size(), 0, 4, tokens.size(), 1,
                                     tokens.data(), tokens.size());
    assert(id > 0);
    kv_ssd_free(cache);

    if (version == 0) {
        return;  // leave the record as written
    }

    // version sits immediately after the 4-byte magic in kv_ssd_record.
    const fs::path file = conv_dir(base, conv_hash) / ("ckpt-" + std::to_string(id) + ".bin");
    std::fstream    fh(file, std::ios::in | std::ios::out | std::ios::binary);
    assert(fh.is_open());
    fh.seekp(sizeof(uint32_t), std::ios::beg);
    fh.write(reinterpret_cast<const char *>(&version), sizeof(version));
    fh.close();
}

// A record written by an older format version must not be loaded, and must not
// be left behind to occupy disk.
static void test_stale_record_is_rejected_and_removed() {
    const fs::path base      = make_scratch();
    const uint64_t conv_hash = 0xABCDEF01;

    write_checkpoint_with_version(base, conv_hash, 2);
    assert(count_ckpt_files(conv_dir(base, conv_hash)) == 1);

    kv_ssd_config cfg;
    cfg.auto_size = false;

    kv_ssd_cache * cache = kv_ssd_init(base.string().c_str(), &cfg, conv_hash);
    assert(cache != nullptr);
    assert(cache->index.empty());                                 // not loaded
    assert(count_ckpt_files(conv_dir(base, conv_hash)) == 0);      // and reclaimed
    kv_ssd_free(cache);

    fs::remove_all(base);
}

// A current-version record survives a reopen untouched.
static void test_current_record_is_kept() {
    const fs::path base      = make_scratch();
    const uint64_t conv_hash = 0xABCDEF02;

    write_checkpoint_with_version(base, conv_hash, 0);
    assert(count_ckpt_files(conv_dir(base, conv_hash)) == 1);

    kv_ssd_config cfg;
    cfg.auto_size = false;

    kv_ssd_cache * cache = kv_ssd_init(base.string().c_str(), &cfg, conv_hash);
    assert(cache != nullptr);
    assert(cache->index.size() == 1);
    assert(count_ckpt_files(conv_dir(base, conv_hash)) == 1);
    kv_ssd_free(cache);

    fs::remove_all(base);
}

// Losing the index file must not make the next store reuse an id that is still
// present on disk.
static void test_next_id_survives_missing_index() {
    const fs::path base      = make_scratch();
    const uint64_t conv_hash = 0xABCDEF03;

    kv_ssd_config cfg;
    cfg.auto_size = false;

    kv_ssd_cache * cache = kv_ssd_init(base.string().c_str(), &cfg, conv_hash);
    assert(cache != nullptr);

    const std::vector<uint8_t>  payload(2048, 0x5C);
    const std::vector<uint32_t> tokens = { 9, 8, 7 };

    std::vector<uint64_t> ids;
    for (int i = 0; i < 3; i++) {
        const uint64_t id = kv_ssd_store(cache, 0, payload.data(), payload.size(), 0, 2, tokens.size(),
                                         (uint32_t) i, tokens.data(), tokens.size());
        assert(id > 0);
        ids.push_back(id);
    }
    kv_ssd_free(cache);

    const fs::path dir = conv_dir(base, conv_hash);
    assert(count_ckpt_files(dir) == 3);

    // Simulate a version bump / corruption making the index unreadable.
    std::error_code ec;
    for (const auto & e : fs::directory_iterator(dir)) {
        const std::string name = e.path().filename().string();
        if (name.rfind("ckpt-", 0) != 0) {
            fs::remove(e.path(), ec);
        }
    }

    kv_ssd_cache * reopened = kv_ssd_init(base.string().c_str(), &cfg, conv_hash);
    assert(reopened != nullptr);

    const uint64_t fresh = kv_ssd_store(reopened, 0, payload.data(), payload.size(), 0, 2, tokens.size(), 9,
                                        tokens.data(), tokens.size());
    assert(fresh > 0);
    for (uint64_t prior : ids) {
        assert(fresh != prior);  // must not clobber a file still on disk
    }
    assert(count_ckpt_files(dir) == 4);
    kv_ssd_free(reopened);

    fs::remove_all(base);
}

int main() {
    test_stale_record_is_rejected_and_removed();
    test_current_record_is_kept();
    test_next_id_survives_missing_index();

    printf("test-ssd-cache-format: OK\n");
    return 0;
}
