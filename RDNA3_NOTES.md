# RDNA3 (Phoenix) Vulkan Notes

Last updated: 2026-07-20

## Goal

Document Vulkan-backend tuning findings for AMD RDNA3 APUs (Phoenix1/Phoenix,
gfx1103, e.g. 7840U/7840HS with Radeon 780M). Companion to STRIX_HALO_NOTES.md
which targets gfx1151 (RDNA3.5) ROCm.

Unlike STRIX_HALO_NOTES, this is for the **Vulkan** backend on RDNA3, not
ROCm/HIP. ROCm-on-RDNA3 has stability issues on Mesa (GLM-4.7-Flash and
DeepSeek2 MLA produce zero generation tokens), so Vulkan is the default
backend for the 7840U.

## Environment

- CPU/GPU: AMD Ryzen 7 7840U (Ayaneo Flip KB) with Radeon 780M
- GPU arch: gfx1103, RDNA3.0
- VRAM carveout: 6 GB (`amdgpu.vis_vramlimit=6144`)
- GTT: 16 GB (`amdgpu.gttsize=18432`) - bumped from default 12 GB to fit 35B Q4
- System RAM: 32 GB total, 25 GB available to OS after carveout
- Mesa driver: RADV (open source)
- Build: CachyLLama `71d1e8f2f` Vulkan backend
- Wave size: 64 (RDNA, not 32 like CDNA)
- Subgroup: KHR_coopmat supported, KHR_coopmat2 not (Phoenix is RDNA3.0)

## Vulkan nodes_per_submit sweep (nps)

Commit `1c19480da` lowered the default `max_nodes_per_submit` from 100 to 8
on UMA devices to avoid hitting the kernel's 2s `amdgpu.lockup_timeout`.
That commit was defensive, not measured. On the 35B Q4 workload we tested
on the 7840U, nps=100 completes without lockup and gives a reproducible
generation-speed win.

**Model:** Qwen3.6-35B-A3B-UD-Q4_K_XL (20.81 GiB)
**Settings:** `-ngl 99 --flash-attn 1 --mmap 0 --direct-io 1 -b 2048 -ub 2048`
**Repetitions:** 3 per run, 4 independent runs at nps=64

| nps | pp1024 median | pp2048 median | tg64 median | notes                       |
|----:|--------------:|--------------:|------------:|-----------------------------|
|   1 | --            | 212           | 19.57       | too low, gen hurts          |
|   4 | --            | 212           | 21.27       | gen improving               |
|   8 | 268           | 268           | 21.65       | current default on UMA      |
|  16 | --            | 210           | 21.89       |                             |
|  32 | --            | 211           | 22.12       | stable, low variance        |
|  64 | 253           | 251           | 22.62       | chosen - see below          |
| 100 | 234           | 211           | 22.65       | max tg, but pp unstable     |

**Selected value: 64.**

Reasoning:
- tg64 at nps=64 is +4.5% over nps=8 (22.62 vs 21.65) with stddev <0.1 t/s
  across 4 independent runs - very reproducible
- The +0.03 t/s gain from 64 -> 100 is in the noise
- nps=64 is 8x the conservative default (vs 12.5x for nps=100), giving more
  headroom against the kernel lockup_timeout on slower workloads
- pp1024/pp2048 medians at nps=64 are slightly below nps=8 (-5/-6%), but
  pp variance on this hardware is 15-25% run-to-run so the regression is
  not statistically significant. The tg64 improvement dominates for the
  7840U use case (chat / agentic / interactive)

**Where this is wired in:** `llama-run.sh` `setup_vulkan_env()` exports
`GGML_VK_NODES_PER_SUBMIT=64` when `LLAMA_GFX_ARCH=gfx1103`. Other RDNA3
silicon (gfx1100/1101/1102) keeps the conservative nps=8 default.

**Manual override:** set `GGML_VK_NODES_PER_SUBMIT=N` in the environment
before running `llama-server` to override the auto-detection.

## FA scratch transpose: neutral on 35B Q4

Commit `37d2a1a21` adds a fused dequant+transpose shader that materialises
the FA KV cache as per-head-contiguous f16 once, so coopmat1 doesn't
re-dequant the whole KV cache inside every Q workgroup. This is the Strix
Halo optimization, ported from upstream #25494.

On the 7840U + 35B Q4 this path appears neutral:

| Setting                          | pp2048 | tg64 |
|----------------------------------|-------:|-----:|
| default (path active if possible) | 261   | 21.8 |
| GGML_VK_NO_FA_SCRATCH_TRANSPOSE=1 | 264   | 21.9 |

Difference is within noise. Either:
- The conditions for `use_dequant_kv` aren't met for Q4_K_XL's KV type
- The optimization is in the noise floor on RDNA3.0 (vs RDNA3.5 where
  the larger L2 cache benefits more from per-head-contiguous layout)

Noted but not changed. If a future model or quant type exposes the path,
worth re-evaluating.

## Why we don't lift this to RDNA3.5 ROCm

The Strix Halo RDNA3.5 ROCm tuning (`5ff23cdf3`, MMVQ/MMQ/FA tables for
gfx1151) uses different parameters (mmq_x_max=64, nwarps=2 for MMVQ, etc.)
that are validated only for gfx1151. RDNA3 (gfx110x) is a different silicon
generation with different wave size handling and different memory subsystem
characteristics. Porting those numbers to gfx1103 would be guessing.

For RDNA3 ROCm, the upstream default MMQ config (rdna2's I=128, nwarps=8)
is what we'd want to compare against before declaring a tuning win.

## Open questions

- Does nps=64 hold up at longer context (pp4096, pp8192)? On a 32 GB APU
  with a 21 GB model the headroom is tight; the kernel lockup_timeout
  might trigger at higher nps once KV cache fills.
- Does nps=64 help or hurt non-MoE dense models? Tested Qwen3.6-35B-A3B
  only. The FA scratch transpose path might activate differently for
  dense models where attention is a larger fraction of compute.
- Does nps=64 transfer to Strix Halo (gfx1151)? Probably yes, but the
 16-32 GB GTT vs 96 GB VRAM difference means lockup risk is much lower.
  Worth benching.

## Post-rebase validation (2026-09-29, Strix Halo gfx1151, RADV)

After rebasing onto upstream master (680a03628), `test-backend-ops` on
Vulkan0 (Radeon 8060S) found two CachyLLama perf features producing wrong
results against the refactored upstream shaders. Both were root-caused
and fixed the same day; both are back to default-on with `=0` opt-outs.
Validate with:

```
./build-vk/bin/test-backend-ops test -b Vulkan0
```

### GGML_VK_MMID_ROWLISTS (default: on, set =0 to disable) - FIXED 2026-09-29

The MMID row-list prepass writes offsets/cursors/entries over the
`[n_as..]` tail of the expert-count buffer, which is exactly where the
hoisted row-id layout lives. Shaders with a `from_lists` reader
(`mul_mm.comp`, `mul_mmq.comp`) consume the lists fine, but the int8
coopmat shader (`mul_mmq_cm1.comp`, used for q8_1-Y MMID on this GPU) has
no such reader and keeps reading the clobbered hoisted rows (256 failures
with row lists on: all mat-path quant-A cases; f16/f32 and vec-path cases
passed). Fixed by gating the prepass + flag on a per-pipeline
`supports_row_lists` capability (threaded through pipeline creation,
including 64-bit indexing variants; false for the cm1-mmq pipelines):
unsupported pipelines transparently fall back to the hoisted path. Full
Vulkan suite passes with row lists on (19003/19003, Strix Halo RADV).

2026-09-29 follow-up: ported the `from_lists` reader to `mul_mmq_cm1.comp`
(it already included the shared reader; needed only the push fields in
matching order plus the branch), so the cm1 path uses lists too and the
fallback is currently untriggered. Kept as a safety hatch for future
shaders without list support (e.g. a coopmat2-MMID variant).

### GGML_VK_FA_KV_CONTIG (default: on, set =0 to disable) - FIXED 2026-09-29

The strided-f16-KV contiguize transpose (`dequant_f16_transpose` prepass in
`ggml_vk_flash_attn`) read its source linearly while engaging on strided
views (`nb[1] != HSK*2`), and after a first repair attempt the source index
still applied the strides to the wrong dims (`nh*s1 + kv*s2` for a
`[HS, KV, NH]` layout). Fixed by making the shader stride-aware (explicit
`s1/s2/s3` element strides, scalar f16 reads, own 7-uint push block) with
the correct `[HS, KV, NH, NS]` decomposition. Full Vulkan suite passes with
it on (19003/19003, Strix Halo RADV); revalidate on other GPUs before
assuming universal benefit.

### Binding-count note

MMID pipelines now declare 6 descriptor bindings (A, B, D, IDS, counts,
fused-scale); the upstream subgroup refactor hardcoded 5. If a new MMID
pipeline creation call is added, it must use 6 (see `mul_mat_id_param_count`
and the `create_mm_pipelines` ID call sites).

### Row-list prepass performance (2026-09-29, Strix Halo q4_K MMID, cm1)

`GGML_VK_MMID_ROWLISTS` on vs off, `test-backend-ops perf`, q4_K/f32,
k=2048, 128 experts used 8 (decode-ish) and 32 experts used 4 (prefill-ish):

| shape | off | on | delta |
|---|---|---|---|
| n_mats=128 n=1   | 21.4us | 12.4us | **-42%** |
| n_mats=128 n=4   | 44.5us | 37.7us | **-15%** |
| n_mats=128 n=32  | 1184us | 921us  | **-22%** |
| n_mats=128 n=64  | 1511us | 965us  | **-36%** |
| n_mats=128 n=128 | 1494us | 923us  | **-38%** |
| n_mats=128 n=512 | 2031us | 1156us | **-43%** |
| n_mats=32 n=512  | 1019us | 983us  | -4% |

The win shrinks as expert count drops (32-expert cases are within noise);
the prepass replaces a per-workgroup full-ids rescan, so it pays off when
n_as and the token count are large. Decode (n=1) is the big winner - this
is the MoE decode path. Correctness: both configs pass 19003/19003.
