# Patch-set status

CachyLLama diverges from `upstream/master` by carrying third-party work. Re-evaluate this table when upstream merges or upstream PRs close — row status changes weekly.

**Convention:**
- "Merged upstream" — drop our copy on the next upstream merge.
- "Not upstreamed" — keep carrying; re-check upstream status each merge.
- "Upstream added" — upstream has the same feature; ours differs in tuning/gating. Keep both until the differences can be rebased onto the upstream version.

| Carry | Source | Upstream status | CachyLLama-specific additions |
|-------|--------|-----------------|------------------------------|
| Quantized-KV FA prefill dequant (Vulkan) | [Nathanw1014/llama.cpp#25494](https://github.com/ggml-org/llama.cpp/pull/25494) | **Merged upstream** as `dc72703fc` | Retains the CachyLLama host-RAM gate via `common::host_available_ram()`, env controls (`GGML_VK_NO_FA_SCRATCH_TRANSPOSE`, `GGML_VK_FA_SCRATCH_SAFETY_MB`, `GGML_VK_FA_SCRATCH_FORCE`), and printf-style warning. The q4/q5 and contiguize variants below are dormant in the canonical build |
| FA dequant-once to q4_0/q4_1/q5_0/q5_1 KV | Nathanw1014 carry | **Dormant / not wired** | Shader `DEQUANT_TRANSPOSE` branches remain from `33cc3c520`, but only the upstream q8_0 transpose pipeline is registered; `GGML_VK_FA_DEQUANT_ALL` is not active |
| FA contiguize strided f16 KV | Nathanw1014 carry | **Dormant / not wired** | `dequant_f16_transpose.comp` remains from `404732f8c`, but the canonical build does not register it and `GGML_VK_FA_KV_CONTIG` is not active |
| Coopmat1 FA P-fragment hoist | Nathanw1014 carry | Not upstreamed | Hoists the P-fragment load out of the `hsv_tile` loop. Measured +5% on Qwen3.6-35B-A3B prefill, Strix Halo |
| Coopmat1 FA Psh query-major | Nathanw1014 carry | Not upstreamed | Stores `Psh` query-major so the GEMM2 A load vectorizes |
| 32-wide subgroup pinning (coopmat1 FA) | Nathanw1014 carry | Not upstreamed | Pins `required_subgroup_size=32` where narrowing is free on RDNA3 wave64 |
| Bound command buffers by memory traffic | Nathanw1014 carry | **Dormant / not wired** | The canonical build still uses a fixed 100-node default; no memory-traffic bound is active |
| Concat transpose shader | Nathanw1014 carry | **Dormant / not wired** | `concat_transpose.comp` remains in the tree, but has no generator, pipeline, dispatch, or active `GGML_VK_CONCAT_TRANSPOSE` control |
| MMID row-list prepass | Nathanw1014 carry | **Dormant / not wired** | `mmid_row_lists.comp` and its loader helper remain in the tree, but no prepass pipeline or dispatch is registered |
| MMID f16-B probe | Nathanw1014 carry | **Dormant / not wired** | No `GGML_VK_MMID_F16B` control or active probe remains in the canonical build |
| MMID wave32 probe | Nathanw1014 carry | **Dormant / not wired** | No `GGML_VK_MMID_WAVE32` control or active probe remains in the canonical build |
| MMID scale cache (q5_K, q4_K, superblock-amortized) | Nathanw1014 carry | **Dormant / not wired** | No active shared-memory scale-cache implementation remains in the canonical MMID path |
| FA MMQ dot product fp32 scaling | Nathanw1014 carry | Not upstreamed | Scales the MMQ dot product in fp32 before narrowing for numerical stability |
| FA split-K reduce shader | Nathanw1014 carry | **Upstream added** | `flash_attn_split_k_reduce.comp` is present upstream; no remaining CachyLLama-only shader delta |
| FA top-K selection shader | Nathanw1014 carry | **Dormant / not wired** | `flash_attn_top_k.comp` remains in the tree, but the active sparse path uses upstream compacting and has no top-K pipeline registration |
| GATED_LINEAR_ATTN | Nathanw1014 carry | **Upstream added** as `f26efa02a` | Vulkan `GGML_OP_GATED_LINEAR_ATTN` support is now upstream in `gla.comp`; no remaining CachyLLama-only shader delta |
| DeepSeek-V4 hyper-connection fused ops | [ggml-org/llama.cpp#26578](https://github.com/ggml-org/llama.cpp/pull/26578) | **Merged upstream** | Three shaders: `dsv4_hc_{pre,comb,post}.comp`. HC hardcoded to 4. Tunable with `GGML_VK_DISABLE_DSV4_HC[_COMB\|_PRE\|_POST]=1`. Measured: prefill +16.4%, decode +41.1% on DSV4-Flash IQ3_XXS, Nimo |
| DeepSeek-V4 Lightning Indexer | CachyLLama original plus upstream #27453 | **Upstream added** as `cb300598d`; local extensions are dormant | Upstream provides the active base `lightning_indexer.comp`; `lightning_indexer_cm.comp` and `lightning_indexer_decode_cm.comp` remain unregistered in the canonical build. See [vulkan-init-order.md](vulkan-init-order.md) |
| DSV4 sparse FA gather-to-compact | CachyLLama original plus upstream #28105 | **Upstream added** generic sparse FA as `fc82583e6`; local top-K variant is dormant | The active sparse path uses upstream compaction; `flash_attn_top_k.comp` remains unregistered |
| FA flash-attn mask optimization | Nathanw1014 carry | **Upstream added** | `flash_attn_mask_opt.comp` is present upstream; no remaining CachyLLama-only shader delta |
| FA MMQ funcs shader | Nathanw1014 carry | **Upstream added** | `flash_attn_mmq_funcs.glsl` is present upstream; no remaining CachyLLama-only shader delta |
| Keep DeepSeek lightning-indexer K cache f16 | Nathanw1014 carry | Not upstreamed | Forces f16 key cache under quantized `-ctk` for Lightning Indexer correctness |
| Vulkan APU `nodes_per_submit` auto-lower | CachyLLama original | Not upstreamed (`ggml-vulkan.cpp` still hardcodes 100) | Defaults to 8 on UMA, 100 on discrete. `GGML_VK_NODES_PER_SUBMIT=N` override |
| Strix Halo RDNA3.5 tuning (ROCm/HIP) | gaetan-puleo carry | Upstream added `mmq-config-rdna3-5.cuh`; CachyLLama's has Strix Halo-specific tuning | `I = 64` in all MMQ CASE entries for upstream `#24127` `static_assert((I_) % 32 == 0)` |
| `common::host_available_ram()` | CachyLLama original | None | Extracted from duplicate implementations in `kv-ssd-cache.cpp` and `kv_page_manager.cpp` |
| DFlash framework | CachyLLama original | Not upstreamed | `src/models/dflash.cpp`. Generic decoder contract via `dflash.decoder_arch` metadata. Currently supports `"laguna"` |
| Laguna-S-2.1 | CachyLLama original | Not upstreamed | `src/models/laguna.cpp`. Sigmoid-routed MoE, shared expert, softplus attention gate, QK-norm, per-layer-type RoPE |

**CachyLLama focus downstream:** If a third-party carry lands upstream cleanly, the CachyLLama copy can be dropped on the next `merge upstream/master` and the local additions (memory gate, env overrides, follow-up fixes) rebased onto the upstream version. When a carry does not get upstreamed, CachyLLama carries it indefinitely — re-check upstream status each merge.

**Watch upstream #24127** (CUDA MMQ refactor): it added `static_assert((I_) % 32 == 0)` to the CASE macro, so any new `rdna3_5` config must keep `I` as a multiple of 32.
