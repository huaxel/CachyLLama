// SPDX-License-Identifier: MIT
// Copyright (c) 2026 fewtarius
//
// Regression test for the global cold-tier cap on the turn-completion path.
//
// server_context_ssd_manager enforced --cache-ssd-cold-maxsize only after
// checkpoint stores. Checkpoints also demote hot->warm->cold during
// on_turn_complete (kv_ssd_on_turn_complete frees the warm RAM blob and
// leaves the on-disk cold entry), and that path did not re-check the cap,
// so the configured limit could stay exceeded indefinitely until an
// unrelated store happened.
//
// The test stores one checkpoint into each of three conversations, sets a
// cap below the resulting cold total, drives turns past the tier windows,
// and asserts the oldest conversation directory was evicted during the
// turn instead of waiting for the next store.
#undef NDEBUG

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "server-context-ssd-manager.h"

#include <cassert>
#include <chrono>
#include <clocale>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

static std::string hex16(uint64_t h) {
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) h);
    return buf;
}

// decode a few tokens into the given sequence so the per-seq state
// serialization is non-empty
static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, llama_seq_id seq) {
    llama_batch batch = llama_batch_init((uint32_t) tokens.size(), 0, 1);
    for (uint32_t pos = 0; pos < tokens.size(); pos++) {
        common_batch_add(batch, tokens[pos], (llama_pos) pos, { seq }, pos + 1 == tokens.size());
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.n_parallel = 4;  // context must host seq ids 1..3 for the three slots
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model *          model      = llama_init->model();
    llama_context *        ctx        = llama_init->context();
    if (!model || !ctx) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    fs::path        base = fs::temp_directory_path() / "ssd_manager_cold_cap_test";
    std::error_code ec;
    fs::remove_all(base, ec);
    fs::create_directories(base, ec);
    if (ec) {
        fprintf(stderr, "%s : could not create scratch directory: %s\n", __func__, ec.message().c_str());
        return 1;
    }

    kv_ssd_config cfg;
    cfg.auto_size            = false;
    cfg.max_cold_checkpoints = 64;
    cfg.hot_turns            = 2;
    cfg.warm_turns           = 4;

    llama::server_context_ssd_manager mgr(base.string().c_str(), &cfg, 256, 64);

    // three conversations, one checkpoint each, with distinct token prefixes
    const uint64_t                 convs[3] = { 0x1111, 0x2222, 0x3333 };
    const std::vector<llama_token> seqs[3]  = {
        { 1, 2,  3,  4  },
        { 5, 6,  7,  8  },
        { 9, 10, 11, 12 },
    };

    for (int i = 0; i < 3; i++) {
        const uint32_t slot = (uint32_t) (i + 1);
        assert(decode_tokens(ctx, seqs[i], slot));

        common_prompt_checkpoint ckpt;
        ckpt.n_tokens = (int64_t) seqs[i].size();
        ckpt.pos_min  = 0;
        ckpt.pos_max  = (llama_pos) (seqs[i].size() - 1);
        ckpt.data_tgt.assign(1, 0xAB);  // non-null gate; the blob is re-serialized from ctx

        bool ok = mgr.store_checkpoint_with_tokens(slot, ctx, nullptr, ckpt, seqs[i].data(), seqs[i].size(), 0,
                                                   convs[i], std::string());
        if (!ok) {
            fprintf(stderr, "%s : store failed for conversation %d\n", __func__, i);
            return 1;
        }

        // the eviction sort ties mtimes at 1-second granularity; sleep past a
        // second so the oldest conversation is deterministic
        std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    }

    // all checkpoints are hot right now: the hot-byte total is what the cold
    // total will become once the tier windows elapse
    size_t   hot = 0, warm = 0, cold = 0, tcp = 0, mcp = 0;
    uint64_t hits = 0, misses = 0;
    float    hit_rate = 0;
    mgr.get_stats(&hot, &warm, &cold, &tcp, &mcp, &hits, &misses, &hit_rate);
    printf("%s : after stores: hot_bytes=%zu checkpoints=%zu\n", __func__, hot, tcp);
    if (hot < 3) {
        fprintf(stderr, "%s : expected at least 3 bytes of state total, got %zu\n", __func__, hot);
        return 1;
    }

    // cap below the post-demotion cold total forces at least one eviction
    const size_t cap        = hot - 1;
    mgr.cold_max_size_bytes = cap;
    printf("%s : cold_total=%zu cap=%zu\n", __func__, hot, cap);

    // drive turns: hot->warm at turn 2, warm->cold at turn 4
    // (checkpoint turn_id 0 + hot_turns/warm_turns)
    for (uint32_t t = 1; t <= 4; t++) {
        mgr.on_turn_complete(t);
        mgr.get_stats(&hot, &warm, &cold, &tcp, &mcp, &hits, &misses, &hit_rate);
        printf("%s : turn %u: hot=%zu warm=%zu cold_count=%zu\n", __func__, t, hot, warm, cold);
    }

    // the cap must hold without any intervening store: the oldest conversation
    // directory is gone, the two newest remain
    bool pass = true;
    if (fs::exists(base / hex16(convs[0]))) {
        fprintf(stderr, "%s : FAIL: oldest conversation dir still present (cap not enforced on turn path)\n", __func__);
        pass = false;
    }
    if (!fs::exists(base / hex16(convs[1])) || !fs::exists(base / hex16(convs[2]))) {
        fprintf(stderr, "%s : FAIL: a non-oldest conversation dir was evicted\n", __func__);
        pass = false;
    }

    fs::remove_all(base, ec);
    if (!pass) {
        return 1;
    }
    printf("%s : OK\n", __func__);
    return 0;
}
