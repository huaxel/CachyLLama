// SPDX-License-Identifier: MIT
// Copyright (c) 2026 fewtarius

#undef NDEBUG

#include "ggml.h"
#include "llama-moe-coact.h"
#include "llama-moe-residency.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>

#if defined(__linux__)
#    include <sys/mman.h>
#    include <unistd.h>
#endif

static llama_moe_layer_residency_internal make_layer(int model_layer, int n_expert, size_t capacity) {
    llama_moe_layer_residency_internal layer;
    layer.model_layer = model_layer;
    layer.n_expert    = n_expert;
    layer.cache.assign(capacity, llama_moe_layer_residency_internal::cache_entry{});
    layer.slot_of.assign(n_expert, -1);
    return layer;
}

static llama_moe_residency_state make_state(int model_layer, int n_expert, size_t capacity) {
    llama_moe_residency_state state;
    state.cfg.enabled                = true;
    state.cfg.max_resident_per_layer = capacity;
    state.layers.push_back(make_layer(model_layer, n_expert, capacity));
    return state;
}

static llama_moe_coact::matrix make_matrix(int n_layer, int n_expert) {
    llama_moe_coact::matrix matrix;
    matrix.num_layers  = n_layer;
    matrix.num_experts = n_expert;
    matrix.layer_pair_counts.assign(n_layer, std::vector<uint32_t>((size_t) n_expert * n_expert, 0));
    matrix.cross_counts.assign(n_layer,
                               std::vector<std::vector<uint32_t>>(n_expert, std::vector<uint32_t>(n_expert, 0)));
    matrix.observation_counts.assign(n_layer, 0);
    return matrix;
}

#if defined(__linux__)
static void test_residency_discards_evicted_mmap_page() {
    const size_t page_size = (size_t) getpagesize();
    auto * data = (uint8_t *) mmap(nullptr, 2 * page_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(data != MAP_FAILED);

    ggml_tensor tensor = {};
    tensor.data        = data;
    tensor.nb[2]       = page_size;

    llama_moe_residency_state state = make_state(0, 2, 1);
    state.layers[0].t_gate          = &tensor;
    state.layers[0].gate_stride     = page_size;

    data[0] = 0xA5;
    llama_moe_residency_touch(&state, 0, 0, nullptr);
    llama_moe_residency_touch(&state, 0, 1, nullptr);

    assert(state.total_evicted == 1);
    assert(state.layers[0].slot_of[0] == -1);
    assert(state.layers[0].slot_of[1] == 0);
    assert(data[0] == 0);

    munmap(data, 2 * page_size);
}
#endif

static void test_residency_tracks_selection_hits_and_invalid_experts() {
    llama_moe_residency_state state      = make_state(7, 4, 2);
    const int32_t             selected[] = { 1, 3 };
    const int32_t             invalid[]  = { -1, 4 };

    llama_moe_residency_touch_layer_selection(&state, 7, selected, 2);
    llama_moe_residency_touch_layer_selection(&state, 7, selected, 2);
    llama_moe_residency_touch_layer_selection(&state, 7, invalid, 2);

    assert(state.total_misses == 2);
    assert(state.total_hits == 2);
    assert(state.layers[0].slot_of[1] >= 0);
    assert(state.layers[0].slot_of[3] >= 0);
    assert(state.layers[0].token_counter == 4);
}

static void test_coactivation_predicts_same_and_next_layer_experts() {
    llama_moe_coact::matrix matrix     = make_matrix(2, 4);
    const int32_t           pair_01[]  = { 0, 1 };
    const int32_t           pair_02[]  = { 0, 2 };
    const int32_t           observed[] = { 0 };
    const int32_t           next_3[]   = { 3 };
    const int32_t           next_2[]   = { 2 };

    llama_moe_coact::record(matrix, 0, pair_01, 2);
    llama_moe_coact::record(matrix, 0, pair_01, 2);
    llama_moe_coact::record(matrix, 0, pair_02, 2);

    std::vector<int32_t> same_layer = llama_moe_coact::predict_same_layer(matrix, 0, observed, 1, 2);
    assert(same_layer.size() == 2);
    assert(same_layer[0] == 1);
    assert(same_layer[1] == 2);

    llama_moe_coact::record_cross_layer(matrix, 0, observed, 1, next_3, 1);
    llama_moe_coact::record_cross_layer(matrix, 0, observed, 1, next_3, 1);
    llama_moe_coact::record_cross_layer(matrix, 0, observed, 1, next_2, 1);

    std::vector<int32_t> next_layer = llama_moe_coact::predict_next_layer(matrix, 0, observed, 1, 2);
    assert(next_layer.size() == 2);
    assert(next_layer[0] == 3);
    assert(next_layer[1] == 2);
}

int main() {
#if defined(__linux__)
    test_residency_discards_evicted_mmap_page();
#endif
    test_residency_tracks_selection_hits_and_invalid_experts();
    test_coactivation_predicts_same_and_next_layer_experts();

    printf("test-moe-residency: OK\n");
    return 0;
}
