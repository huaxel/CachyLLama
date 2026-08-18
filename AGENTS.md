# AGENTS.md

Technical reference for CachyLLama development.

**Version:** 2.0
**Date:** 2026-08-15
**Purpose:** Development conventions, build, testing, and architecture for the
CachyLLama fork of llama.cpp. For project methodology (the Unbroken Method,
checkpoint workflow, session handoff), see `.clio/instructions.md`.

---

## Project overview

CachyLLama is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp)
focused on performance optimization for AMD APU hardware. It tracks upstream
master through periodic merges, then carries its own divergent work on top.

- **Languages:** C11 (ggml core), C++17 (llama engine, common utilities), GLSL
  (Vulkan shaders), Python (conversion scripts, GGUF tooling)
- **Build system:** CMake 3.14+
- **Architecture:** Modular C library with multi-backend hardware acceleration
  (CPU, Vulkan, CUDA, Metal, HIP, etc.) plus CachyLLama-specific subsystems
- **License:** MIT (same as upstream, Copyright (c) 2023-2026 The ggml authors)

### CachyLLama-specific subsystems

| Subsystem | Location | Description |
|-----------|----------|-------------|
| Persistent SSD-backed KV cache | `common/kv-ssd-cache.cpp`, `common/kv-ssd-system-cache.cpp`, `common/kv_page_manager.cpp` | Three-tier (hot/warm/cold) on-disk KV cache with conversation hashing |
| MoE expert residency | `src/llama-moe-residency.cpp` | madvise-based expert paging for models larger than RAM |
| MoE expert co-activation tracking | `src/llama-moe-coact.cpp` | Persists expert co-activation matrix for prewarm ordering |
| MoE expert activation tracking | C API in `include/llama.h` | Real-time per-layer expert activation counts via `/expert-stats` |
| User isolation | `include/llama.h`, server integration | `user_id` parameter, per-user concurrency cap, slot affinity |
| System prompt cache | `common/kv-ssd-system-cache.cpp` | Cross-conversation system prompt reuse with recurrent state |
| Hybrid MoE checkpoint restore | `src/llama-kv-cache*.cpp`, `src/llama-memory-recurrent.cpp` | Attention-only KV clearing, recurrent state preservation |
| DFlash framework | `src/models/dflash.cpp` | Generic decoder contract for target-architecture-specific drafting |
| Laguna-S-2.1 | `src/models/laguna.cpp` | Sigmoid-routed MoE with shared expert and softplus attention gate |
| DSV4 KV cache | `src/llama-kv-cache-dsv4.cpp` | DeepSeek-V4 sparse attention KV cache variant |
| DSA KV cache | `src/llama-kv-cache-dsa.cpp` | DeepSeek sparse attention KV cache variant |
| Hybrid memory types | `src/llama-memory-hybrid*.cpp`, `src/llama-memory-recurrent.cpp` | Hybrid SSM/attention memory management |
| Host RAM utility | `common/host-ram.h`, `common/host-ram.cpp` | Cross-platform available-RAM query |
| Vulkan FA scratch gate | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | Quantized-KV FA dequant-once with host-RAM safety check |

### Key third-party carries (not upstreamed, CachyLLama-specific)

| Feature | Source | Files |
|---------|--------|-------|
| DeepSeek-V4 Lightning Indexer | CachyLLama original | `vulkan-shaders/lightning_indexer*.comp` |
| DSV4 hyper-connection fused ops | [ggml-org/llama.cpp#26578](https://github.com/ggml-org/llama.cpp/pull/26578) | `vulkan-shaders/dsv4_hc_*.comp` |
| Concat transpose shader | Nathanw1014 carry | `vulkan-shaders/concat_transpose.comp` |
| MMID row-lists prepass | Nathanw1014 carry | `vulkan-shaders/mmid_row_lists.comp` |
| FA dequant-once scratch | [Nathanw1014/llama.cpp#25494](https://github.com/ggml-org/llama.cpp/pull/25494) | `ggml-vulkan.cpp`, `vulkan-shaders/dequant_*.comp` |
| APU `nodes_per_submit` auto-lower | CachyLLama original | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` |
| Subgroup size pinning | Nathanw1014 carry | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` |
| ROCm RDNA3.5 Strix Halo tuning | gaetan-puleo carry | `ggml/src/ggml-cuda/mmq-config-rdna3_5.cuh` |

See the [patch-set table](#patch-set-table) below for upstream status and
CachyLLama-specific additions for each carry.

---

## Quick setup

```bash
# Clone (with submodules for ggml)
git clone --recurse-submodules https://github.com/fewtarius/CachyLLama.git
cd CachyLLama

# Build (Vulkan on Linux, Metal on macOS, default CPU)
cmake -B build
cmake --build build --config Release -j$(nproc)

# Build with Vulkan (default on Linux AMD)
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j$(nproc)

# Run the server
./build/bin/llama-server -m /path/to/model.gguf

# Run CLI chat
./build/bin/llama-cli -m /path/to/model.gguf

# Run tests
cd build && ctest --output-on-failure
```

> **Note:** For end-to-end usage (runner scripts, GPU detection, benchmark
> harness, GTT configuration), use the
> [parent project](https://github.com/fewtarius/llama-ai) which builds
> CachyLLama with the right cmake flags automatically.

---

## Architecture

```
                    include/llama.h (Public C API)
                          |
                    src/llama.cpp (API implementation)
                          |
    +----------+----------+----------+----------+----------+
    |          |          |          |          |          |
    src/       src/       src/       src/       src/       src/
    llama-     llama-     llama-     llama-     llama-     models/
    model      context    sampler    kv-cache   memory     (per-arch)
    src/                        src/llama-moe-
    llama-                      residency.cpp
    chat.cpp                    llama-moe-coact.cpp
    common/                     common/kv-ssd-*.cpp
    (utilities)                 common/kv_page_manager.cpp
    common/host-ram.{h,cpp}     common/kv-ssd-system-cache.cpp
    common/arg.cpp              common/preset.cpp
    common/sampling.cpp         common/reasoning-budget.cpp
    ggml/ (Tensor library — Git submodule)
    |
    +----+----+----+----+----+----+----+----+
    |    |    |    |    |    |    |    |    |
   CPU CUDA Metal Vulkan SYCL HIP OpenCL ...
```

### Key modules

| Module | Purpose |
|--------|---------|
| `include/llama.h` | Public C API (includes CachyLLama extensions: MoE tracking, residency, user isolation) |
| `src/llama.cpp` | Core API implementation |
| `src/llama-model.cpp` | Model loading, architecture dispatch |
| `src/llama-context.cpp` | Inference context, graph evaluation |
| `src/llama-kv-cache*.cpp` | KV cache implementations (standard, ISWA, MSA, DSV4, DSA, recurrent) |
| `src/llama-memory*.cpp` | Memory types (hybrid, recurrent, hybrid-iswa) |
| `src/llama-moe-residency.cpp` | MoE expert residency management |
| `src/llama-moe-coact.cpp` | MoE expert co-activation tracking |
| `src/models/*.cpp` | Per-model architecture implementations (130+ models) |
| `common/kv-ssd-cache.cpp` | Persistent SSD-backed KV cache |
| `common/kv-ssd-system-cache.cpp` | Cross-conversation system prompt cache |
| `common/kv_page_manager.cpp` | Page-level cache management |
| `common/host-ram.{h,cpp}` | Cross-platform available-RAM query |
| `common/arg.cpp` | CLI argument parsing (includes CachyLLama flags) |
| `common/sampling.cpp` | Token sampling strategies |
| `common/speculative.cpp` | Speculative decoding (Eagle3, DSpark, DFlash, MTP) |
| `ggml/` | Tensor library with backends (submodule) |

### Directory structure

| Path | Purpose |
|------|---------|
| `include/` | Public C API headers (`llama.h`, `llama-cpp.h`) |
| `src/` | Core llama library (model loading, inference, sampling, KV cache) |
| `src/models/` | Per-model architecture implementations |
| `src/llama-moe-residency.cpp` | MoE expert residency |
| `src/llama-moe-coact.cpp` | MoE expert co-activation tracking |
| `common/` | Shared utilities (arg parsing, sampling, chat, Jinja, PEG parser) |
| `common/kv-ssd-cache.cpp` | SSD-backed KV cache |
| `common/kv-ssd-system-cache.cpp` | System prompt cache |
| `common/kv_page_manager.cpp` | Page management |
| `common/host-ram.{h,cpp}` | Host RAM query |
| `ggml/` | ggml tensor library (submodule: backends, quantization, graph execution) |
| `tools/` | Executable tools (server, CLI, bench, quantize, perplexity) |
| `tools/server/` | OpenAI-compatible HTTP server |
| `tests/` | CTest-based C++ unit tests |
| `gguf-py/` | Python GGUF reader/writer library |
| `vendor/` | Vendored dependencies (cpp-httplib, nlohmann/json, miniaudio, stb) |
| `docs/` | Documentation (build guides, architecture, development) |
| `cmake/` | CMake modules and helpers |
| `ci/` | CI run scripts |

---

## Code style

**C/C++ conventions** (same as upstream llama.cpp — do not deviate):

- **C++17** standard, **C11** for ggml core
- **4 spaces** indentation, no tabs
- **LF line endings**, UTF-8 encoding
- **Vertical alignment** for readability
- Brackets on same line: `if (cond) {`
- Pointer/reference alignment: `void * ptr`, `int & a`
- `snake_case` for functions, variables, and types
- Naming optimizes for **longest common prefix** (e.g., `number_small`, `number_big`)
- Sized integer types in public API: `int32_t`, `uint32_t`
- Declare structs as `struct foo {}` not `typedef struct foo {} foo`
- In C++ omit `struct`/`enum` keyword when unnecessary
- Avoid templates, fancy STL constructs — use basic `for` loops
- Keep it simple, minimal dependencies

**Formatting:** Use `.clang-format` (clang-tools v15+) when in doubt. The
project has a comprehensive `.clang-format` config at the root.

**EditorConfig:** Root `.editorconfig` enforces: spaces, indent 4, LF, UTF-8,
trailing whitespace trimmed.

**CachyLLama-specific conventions:**

- CachyLLama additions are marked with `SPDX-License-Identifier: GPL-3.0-or-later`
  and `Copyright (c) 2026 fewtarius` at the top of each file
- Vulkan shader dispatch code that is env-gated should use the `GGML_VK_DISABLE_*`
  / `GGML_VK_*` naming convention
- CachyLLama C API functions in `include/llama.h` should use the `llama_` prefix
  and be grouped with other CachyLLama extensions (after the upstream API section)
- MoE-related code should reference `docs/moe-expert-residency.md` for
  architecture context
- Do **not** write `Assisted-by:` in commit messages — this is a fork, commits
  go directly to the CachyLLama git history

---

## Module naming conventions

| Prefix | Purpose | Examples |
|--------|---------|----------|
| `llama-*` | Core llama modules | `llama-model`, `llama-context`, `llama-sampler` |
| `ggml-*` | ggml backend modules | `ggml-cpu`, `ggml-cuda`, `ggml-metal`, `ggml-vulkan` |
| `test-*` | Test files | `test-backend-ops`, `test-tokenizer-0`, `test-sampling` |
| `llama-moe-*` | CachyLLama MoE subsystems | `llama-moe-residency`, `llama-moe-coact` |

Source files follow the pattern: `src/llama-{module}.cpp` / `src/llama-{module}.h`

---

## Testing

### C++ unit tests (CTest)

```bash
# Build with tests enabled (default for standalone builds)
cmake -B build -DLLAMA_BUILD_TESTS=ON
cmake --build build -j$(nproc)

# Run all tests
cd build && ctest --output-on-failure

# Run specific test binaries
./build/bin/test-backend-ops
./build/bin/test-sampling
./build/bin/test-tokenizer-0
```

### CachyLLama-specific test coverage

| Test | What it covers |
|------|----------------|
| `test-backend-ops` | ggml operator consistency across backends — **must pass for Lightning Indexer, DSV4_HC, concat_transpose, and mul_mat_id changes** |
| `test-sampling` | Token sampling correctness |
| `test-tokenizer-0` | Tokenizer roundtrip tests |
| `test-chat-template` | Chat template rendering |
| `test-grammar-parser` | GBNF grammar parsing |
| `test-quantize-fns` | Quantization function correctness |

### Vulkan shader testing

The Lightning Indexer shader is validated by `test-backend-ops` on Strix Halo
(gfx1151, RDNA3.5, Vulkan 1.4, RADV Mesa 26.2). Before changing
`lightning_indexer.comp` or its dispatch in `ggml-vulkan.cpp`:

1. Run `test-backend-ops` and verify 108/108 pass (was 0/108 before the init-order
   fix — see [init-order note](#vulkan-init-order-critical-lightning-indexer-and-dsv4-hc))
2. Test with DeepSeek-V4-Flash IQ3_XXS on a RADV APU to confirm no
   "Lightning Indexer not supported" warning
3. The shader's `required_subgroup_size=32` must be passed via
   `ggml_vk_create_pipeline` on RDNA3 wave64 devices

### Python tests

```bash
# GGUF Python library tests
cd gguf-py && python -m pytest tests/

# Tokenizer tests
python tests/test-tokenizer-0.py
```

### Benchmarking

Use `llama-bench` for parameter sweeps. For end-to-end cache performance
benchmarks, use the parent project's `scripts/benchmark.sh` which drives
`llama-server` via HTTP and measures cold/warm TTFT.

---

## Commit format

CachyLLama uses its own git history as the canonical source of truth (no PR
workflow — this is a fork). Commits are squash-merged into the main branch
with descriptive messages:

```
vulkan: fix Lightning Indexer init order (108/108 on Strix Halo)
```

```
common: extract host_available_ram() from kv-ssd-cache.cpp
```

```
models: add Laguna-S-2.1 support (decoder_arch = "laguna")
```

---

## Common commands

```bash
# Build
cmake -B build && cmake --build build -j$(nproc)

# Test
cd build && ctest --output-on-failure

# Server
./build/bin/llama-server -m model.gguf

# CLI
./build/bin/llama-cli -m model.gguf

# Quantize
./build/bin/llama-quantize input.gguf output.gguf Q4_K_M

# Convert
python convert_hf_to_gguf.py /path/to/hf-model

# CI locally
./ci/run.sh

# Format check
git diff --name-only | grep -E '\.(c|cpp|h|hpp)$' | xargs clang-format --dry-run -Werror

# Search code
grep -rn "pattern" src/ common/ include/
```

---

## Context checkpoint system (`tools/server/server-context.cpp`)

The server maintains a per-slot ring buffer of in-memory KV cache snapshots
("context checkpoints") used to skip prompt reprocessing on cached turns
(LCP / f_keep optimization). Four interlocking pieces -- change one, re-verify
the others.

### 1. Two producer paths feeding one `std::list<common_prompt_checkpoint>`

- **`create_checkpoint(slot, n_tokens_cur, pos_min, pos_max)`** -- mid-prompt
  snapshots fired during prefill when a batch starts a user message (or
  `--checkpoint-near-end` is set and we're near the prompt end).  Uses the
  live KV cache's `pos_min` / `pos_max` from `llama_memory_seq_pos_*`,
  skipping if checkpoints are too close (`params_base.checkpoint_min_step`,
  default 32768).
- **`deferred_create_final_checkpoint(slot)`** -- after the first generation
  token lands, captures a full prompt snapshot at `pos_min=0`,
  `pos_max = prompt_n_tokens - 1`.  Runs asynchronously so the SSD write
  doesn't block decode.

Both add with `emplace_back`, so list position == insertion order.

### 2. Insertion-order ring buffer eviction (BOTH paths)

When the list reaches `params_base.n_ctx_checkpoints` (typically 8-32), the
ring buffer pops the FRONT (oldest) and appends the new entry at the back:

```cpp
if (size >= n_ctx_checkpoints) {
    auto & victim = slot.prompt.checkpoints.front();
    SLT_TRC(slot, "recycling oldest ... (pos_min=%d, pos_max=%d, ...)");
    slot.prompt.checkpoints.pop_front();
}
slot.prompt.checkpoints.emplace_back();
```

Why insertion-order, not "highest pos_min": deferred finals all carry
`pos_min == 0`, so the old strict-greater-than comparator always picked
`begin()` and recycled slot 1 every time -- a single-slot FIFO with 15 dead
entries.  Insertion order breaks the tie and gives a true round-robin across
conversation snapshots.  The acceptance predicate filters by `pos_min` /
`pos_max` regardless of list position, so cycling doesn't affect matching.

Cold-start mid-prompts (created on a fresh slot where `pos_min_thold == 0`
makes the first batch's `pos_min` equal 0 too) share the same `pos_min == 0`
signature as deferred finals.  They're valid LCP snapshots either way -- they
just consume one of the N ring buffer slots.

### 3. SWA-skipped entries persist across turns

In `get_available()` (`server-context.cpp` ~line 4314) there's an SWA
invalidation block:

```cpp
for (auto it = slot.prompt.checkpoints.begin(); ...;) {
    const auto & cur = *it;
    const bool deferred_final_snapshot = (cur.pos_min == 0);
    if (!deferred_final_snapshot && cur.pos_max > pos_next) {
        // erase
        it = slot.prompt.checkpoints.erase(it);
    } else { ++it; }
}
```

Deferred-final snapshots (`pos_min == 0`) are preserved across turns.  Without
this guard, the SWA step would erase every prior deferred final and the ring
buffer would never accumulate past 2 entries.

Genuine SWA invalidation still happens inside the LCP acceptance predicate
(`server-context.cpp` ~line 4198):

```cpp
if (n_swa > 0 && cur.pos_max > pos_next) return false;
```

That predicate is the right place to filter by SWA coverage; the
`get_available()` guard exists only to keep the buffer populated, not to make
SWA-correctness decisions.

**Don't add a redundant erase here based on SWA coverage -- that path will
self-conflict.**

### 4. Memory budget (`_ckpt_memory_budget()`)

The size-based eviction runs BEFORE count-based eviction in both
`create_checkpoint()` and `deferred_create_final_checkpoint()`:

```cpp
size_t _ckpt_memory_budget() const {
    const size_t default_limit = (size_t)2 * 1024 * 1024 * 1024;  // 2 GiB floor
    if (params_base.n_ctx_checkpoints <= 0) return default_limit;
    // 400 MiB per configured checkpoint = 200 MiB working set * 2 headroom.
    const size_t per = (size_t)params_base.n_ctx_checkpoints * 400 * 1024 * 1024;
    return std::max(default_limit, per);
}
```

The budget scales with `n_ctx_checkpoints` so the auto-scaled count from
llama-ai (`scripts/optimize.sh`, 8 base + 1 per 8K above 65K context, capped
at 32) actually fires.  An earlier version coupled it to `cache_ram` (1%) and
unconditionally capped checkpoints at 3-4 -- the auto-scaling was a no-op.

Worst case at 32 checkpoints: 32 * 400 MiB = 12.8 GiB (the `std::max`
floors at 2 GiB so small checkpoint counts still get a 2 GiB working
set).  For the default 16 checkpoints at typical q8_0 KV (~50 MiB each
~= 800 MiB total), nothing changes.

### 5. LCP acceptance predicate

The walk in `get_available() / receive/decode-input` (`server-context.cpp`
~line 4264):

```cpp
const auto it = std::find_if(
    slot.prompt.checkpoints.rbegin(),
    slot.prompt.checkpoints.rend(),
    [&](const auto & cur) {
        if (n_swa > 0 && cur.pos_max > pos_next) {
            return false;  // SWA filter
        }
        return cur.pos_min < pos_min_thold || cur.pos_min == 0;
    });
```

Reverse iteration (newest first).  The first qualifying entry wins.  Both
deferred finals (`pos_min=0`) and early-mid-prompts (`pos_min < pos_min_thold`)
are accepted; mid-prompts whose `pos_min` falls past `pos_min_thold` but
haven't been SWA-shifted are also accepted.

### Auto-scaling (from `llama-ai/scripts/optimize.sh`)

```bash
base_ctx=65536 base_cp=8 scale_per=8192 max_cp=32
if [[ $ctx -gt $base_ctx ]]; then
    extra=$(( (ctx - base_ctx) / scale_per ))
    SOLVER_CHECKPOINTS=$(( base_cp + extra ))
fi
[[ $SOLVER_CHECKPOINTS -gt $max_cp ]] && SOLVER_CHECKPOINTS=$max_cp
```

| Context | Checkpoints |
| ------- | ----------- |
| <= 65 K | 8           |
| 98 K    | 12          |
| 131 K   | 16          |
| 196 K   | 24          |
| 262 K   | 32 (capped) |

Verified on Nimo (Strix Halo) with Laguna-S-2.1 Q5_K_XL at 131K context:
ring buffer saturates at 16, then cycles oldest-first across turns 1..16,
1..16, ... -- true round-robin.  f_keep climbs monotonically (0.488 ->
0.949 across 20 turns), 18 ckpt-restored events after the first cold turn.

---

## CachyLLama development conventions

### Upstream tracking

- CachyLLama merges upstream `llama.cpp` master periodically via
  `git merge upstream/master`
- Before rebasing or squashing in `CachyLLama/`, push to a backup branch:
  `git branch backup-before-rebase`
- After a merge, re-check all CachyLLama carries for conflicts — especially
  `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (which accumulates shader dispatch
  additions) and `src/llama-arch.cpp` / `src/llama-model.cpp` (which gain
  per-model architecture entries)
- The `patches/` directory in the parent project is deprecated — CachyLLama
  maintains its changes directly in git history

### Adding a new model

1. Add architecture entries in `src/llama-arch.cpp` and `src/llama-arch.h`
2. Implement `src/models/{model}.cpp` following the pattern in
   `docs/development/HOWTO-add-model.md`
3. Add GGUF metadata keys in `src/llama-model.cpp` if the model has custom
   hparams
4. Update `convert_hf_to_gguf.py` if the model needs conversion support
5. Run `test-backend-ops` to verify operator consistency

### Adding a new Vulkan shader

1. Add the `.comp` file in `ggml/src/ggml-vulkan/vulkan-shaders/`
2. Register it in `ggml/src/ggml-vulkan/vulkan-shaders/CMakeLists.txt`
3. Add dispatch logic in `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
4. Gate behind an env var (`GGML_VK_DISABLE_*` or `GGML_VK_*`)
5. Run `test-backend-ops` to verify correctness

### CachyLLama-specific API additions

New public C API functions go in `include/llama.h` (after the upstream API
section) and `src/llama.cpp`. Follow the existing `LLAMA_API` visibility
convention. Document with inline comments that describe what, not why —
git history handles why.

### Environment variable conventions

| Prefix | Purpose |
|--------|---------|
| `GGML_VK_*` | Vulkan backend tuning (shaders, scratch, nodes_per_submit) |
| `GGML_VK_DISABLE_*` | Opt-out flags for individual Vulkan features |
| `LLAMA_ARG_*` | CLI flag equivalents for MoE residency offload |
| `LLAMA_SSD_*` | SSD cache configuration (defaults, not overrides) |

User overrides (that win over the solver in `llama-run.sh`) use `*_OVERRIDE`
suffix or are passed via CLI flags.

---

## Vulkan init-order critical: Lightning Indexer and DSV4_HC

On devices that enable `VK_EXT_subgroup_size_control` (Strix Halo, RDNA3
Phoenix, modern Mesa), the four Vulkan feature flags — `lightning_indexer`,
`dsv4_hc_comb`, `dsv4_hc_pre`, `dsv4_hc_post` — were previously assigned
**before** `subgroup_require_full_support` was finalized. The flag was set
from `subgroup_size_control_features.computeFullSubgroups` several lines
later, so all four evaluated to `false` at init time and the corresponding
pipelines were never built. `supports_op` then returned `false` at runtime,
causing fused operations to silently fall back to CPU with the warning:

```
layer N is assigned to device Vulkan0 but Lightning Indexer is assigned to
device CPU (usually due to missing support)
```

**Fix:** The four flag assignments were moved to **after** the
`subgroup_size_control_features.computeFullSubgroups` probe. Verified on
Nimo (Strix Halo, gfx1151, Vulkan 1.4, RADV Mesa 26.2) with DeepSeek-V4-Flash
IQ3_XXS — the warning is gone and the 15k-token prefill runs at 135-150 t/s.

---

## Patch-set table

CachyLLama diverges from `upstream/master` by carrying these third-party works.
Re-evaluate when upstream merges or upstream PRs close.

| Carry | Source | Upstream status | CachyLLama-specific additions |
|-------|--------|-----------------|------------------------------|
| Quantized-KV FA prefill dequant (Vulkan) | [Nathanw1014/llama.cpp#25494](https://github.com/ggml-org/llama.cpp/pull/25494) | PR open, under review by `jeffbolznv` (active July 2026) | Host-RAM gate via `common::host_available_ram()`, three env vars (`GGML_VK_NO_FA_SCRATCH_TRANSPOSE`, `GGML_VK_FA_SCRATCH_SAFETY_MB`, `GGML_VK_FA_SCRATCH_FORCE`), printf-style warning. Files: `ggml/src/ggml-vulkan/ggml-vulkan.cpp`, `vulkan-shaders/dequant_q8_0.comp`, `vulkan-shaders/dequant_q4_0.comp`, `vulkan-shaders/vulkan-shaders-gen.cpp`. |
| FA dequant-once to q4_0/q4_1/q5_0/q5_1 KV | Nathanw1014 carry | Not upstreamed | Extends the dequant-once path to four additional quant types. Env-gated via `GGML_VK_FA_DEQUANT_ALL=1`. |
| FA contiguize strided f16 KV | Nathanw1014 carry | Not upstreamed | Env-gated via `GGML_VK_FA_KV_CONTIG=1` (default off). Falls back to native path if `required_scratch + safety > device-local capacity`. |
| Coopmat1 FA P-fragment hoist | Nathanw1014 carry | Not upstreamed | Hoists the P-fragment load out of the hsv_tile loop. Measured +5% on Qwen3.6-35B-A3B prefill, Strix Halo. |
| Coopmat1 FA Psh query-major | Nathanw1014 carry | Not upstreamed | Stores Psh query-major so the GEMM2 A load vectorizes. |
| 32-wide subgroup pinning (coopmat1 FA) | Nathanw1014 carry | Not upstreamed | Pins `required_subgroup_size=32` where narrowing is free on RDNA3 wave64. |
| Bound command buffers by memory traffic | Nathanw1014 carry | Not upstreamed | Replaces flops-based ceiling with memory-traffic-based ceiling for UMA fairness. |
| Concat transpose shader | Nathanw1014 carry | Not upstreamed | `concat_transpose.comp` for delta-net dim-0 concat. Env-gated via `GGML_VK_CONCAT_TRANSPOSE` (default ON). |
| MMID row-list prepass | Nathanw1014 carry | Not upstreamed | `mmid_row_lists.comp` for grouped-GEMM redesign. Stage 1 of 2. |
| MMID f16-B probe | Nathanw1014 carry | Not upstreamed | Env-gated via `GGML_VK_MMID_F16B=1` (default off). |
| MMID wave32 probe | Nathanw1014 carry | Not upstreamed | Env-gated via `GGML_VK_MMID_WAVE32=1` (default off). |
| MMID scale cache (q5_K, q4_K, superblock-amortized) | Nathanw1014 carry | Not upstreamed | Shared-memory scale cache for mul_mat_id. |
| FA MMQ dot product fp32 scaling | Nathanw1014 carry | Not upstreamed | Scales the MMQ dot product in fp32 before narrowing for numerical stability. |
| FA split-K reduce shader | Nathanw1014 carry | Not upstreamed | `flash_attn_split_k_reduce.comp` for split-K FA on large prompts. |
| FA top-K selection shader | Nathanw1014 carry | Not upstreamed | `flash_attn_top_k.comp` for DeepSeek sparse FA. |
| GATED_LINEAR_ATTN | Nathanw1014 carry | Not upstreamed | `f2ef602a7` implements `GGML_OP_GATED_LINEAR_ATTN` for gated linear attention. |
| DeepSeek-V4 hyper-connection fused ops | [ggml-org/llama.cpp#26578](https://github.com/ggml-org/llama.cpp/pull/26578) | **Merged upstream** (commit `ccbc17862`) | No CachyLLama changes — picked up cleanly in merge. Three shaders: `dsv4_hc_{pre,comb,post}.comp`. HC hardcoded to 4. Tunable with `GGML_VK_DISABLE_DSV4_HC[_COMB|_PRE|_POST]=1`. Measured: prefill +16.4%, decode +41.1% on DSV4-Flash IQ3_XXS, Nimo. |
| DeepSeek-V4 Lightning Indexer | CachyLLama original | Not upstreamed | `lightning_indexer.comp`, `lightning_indexer_cm.comp`, `lightning_indexer_decode_cm.comp`. 108/108 on test-backend-ops, Strix Halo. See [init-order note](#vulkan-init-order-critical-lightning-indexer-and-dsv4-hc). |
| DSV4 sparse FA gather-to-compact | CachyLLama original | Not upstreamed | Sparse top-k FA for DeepSeek V4 CSA shape. `flash_attn_top_k.comp`. Test coverage in `8a8e05712`. |
| FA flash-attn mask optimization | Nathanw1014 carry | Not upstreamed | `flash_attn_mask_opt.comp` for optimized attention mask handling. |
| FA MMQ funcs shader | Nathanw1014 carry | Not upstreamed | `flash_attn_mmq_funcs.glsl` shared code for MMQ-based FA. |
| Keep DeepSeek lightning-indexer K cache f16 | Nathanw1014 carry | Not upstreamed | Forces f16 key cache under quantized `-ctk` for Lightning Indexer correctness. |
| Vulkan APU `nodes_per_submit` auto-lower | CachyLLama original | Not upstreamed (`ggml-vulkan.cpp` still hardcodes 100) | `1c19480da`: defaults to 8 on UMA, 100 on discrete. `GGML_VK_NODES_PER_SUBMIT=N` override. |
| Strix Halo RDNA3.5 tuning (ROCm/HIP) | gaetan-puleo carry | Upstream added `mmq-config-rdna3-5.cuh` but CachyLLama's `mmq-config-rdna3_5.cuh` has Strix Halo-specific tuning | `71d1e8f2f` bumps I from 48 to 64 in all 232 MMQ CASE entries for upstream #24127 `static_assert((I_) % 32 == 0)`. |
| `common::host_available_ram()` | CachyLLama original (refactor) | None | Extracted from duplicate implementations in `kv-ssd-cache.cpp` and `kv_page_manager.cpp`. New files `common/host-ram.{h,cpp}`. |
| DFlash framework | CachyLLama original | Not upstreamed | `src/models/dflash.cpp`. Generic decoder contract via `dflash.decoder_arch` metadata. Currently supports `"laguna"`. |
| Laguna-S-2.1 | CachyLLama original | Not upstreamed | `src/models/laguna.cpp`. Sigmoid-routed MoE, shared expert, softplus attention gate, QK-norm, per-layer-type RoPE. |

**CachyLLama focus downstream:** If a third-party carry lands upstream cleanly,
the CachyLLama copy can be dropped on the next `merge upstream/master` and the
local additions (memory gate, env overrides, follow-up fixes) rebased onto the
upstream version. When a carry does not get upstreamed, CachyLLama carries it
indefinitely — re-check upstream status each merge.

**Watch upstream #24127** (CUDA MMQ refactor) when bumping: it added
`static_assert((I_) % 32 == 0)` to the CASE macro, so any new `rdna3_5` config
must keep I as a multiple of 32.

---

## Anti-patterns

| Anti-pattern | Why it's wrong | What to do |
|--------------|----------------|------------|
| Adding third-party dependencies | Project minimizes deps intentionally | Use vendored libs in `vendor/` or implement inline |
| Using `typedef struct foo {} foo` | Project convention is `struct foo {}` | Declare as `struct foo {}` |
| Fancy template metaprogramming | Codebase avoids complex STL constructs | Use basic loops and simple patterns |
| Mixing unrelated changes in one commit | History should be scannable | One logical change per commit |
| Ignoring clang-format | Project has strict formatting rules | Run `clang-format`, respect `.editorconfig` |
| Committing handoff files | Session notes are internal | Keep `ai-assisted/` out of git |
| Writing `Assisted-by:` in commits | Fork history is our own | Use descriptive commit messages |

---

## Key documentation

| File | Purpose |
|------|---------|
| `docs/build.md` | Build instructions for all platforms/backends |
| `docs/development/HOWTO-add-model.md` | Adding new model support |
| `docs/development/parsing.md` | PEG parser for model output |
| `docs/development/user-isolation-design.md` | User isolation architecture |
| `docs/moe-expert-residency.md` | MoE expert residency mechanism, hit rates, C API |
| `docs/autoparser.md` | Auto-detecting model features |
| `docs/ops.md` | ggml operator reference |
| `docs/ops/Vulkan.csv` | Vulkan op support matrix |
| `STRIX_HALO_NOTES.md` | Strix Halo / RDNA3.5 development notes |
| `RDNA3_NOTES.md` | RDNA3 Vulkan development notes |
| `CEZANNE_NOTES.md` | Cezanne platform notes |
| `README.md` | This project's high-level overview |
| `.clio/instructions.md` | Project methodology (Unbroken Method, workflow) |

---

*For the llama-ai parent project's development conventions, see the parent
project's AGENTS.md.*
