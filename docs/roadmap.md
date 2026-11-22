# RustBlock — Roadmap phases 2–8

**Baseline:** 8 November 2026 → updated **22 November 2026**
**Phase 1:** complete
**Phase 8:** implementation complete for its declared scope; a full tensor-by-tensor AWQ comparison is still open
**Phases 2–7 and 5.2–5.4:** partly implemented; the parts present in the code are marked partial until external acceptance evidence exists.

**22 November 2026 — desktop GUI update.** The iced 0.14 GUI gained a download queue, persistent settings, additional model sources, a runtime deploy dialog and a rewritten window chrome:

- **Download list.** Downloads are queued and reported one at a time in the Discover rail with byte counts, percentage, smoothed speed, cancel and dismiss; finished items can be added to the conversion queue, opened, inspected or deployed. Finished files are optionally queued automatically.
- **Model sources.** Discover can read Hugging Face (with a configurable endpoint, e.g. `https://hf-mirror.com`) and ModelScope. ModelScope has no public search endpoint, so it accepts a model link or `owner/name` id and lists files through the official repository API.
- **Settings.** A persistent settings dialog stores the target, verification level, overwrite policy, output mode, download folder, source choice, auto-add, requantization confirmation, Hugging Face endpoint and the LM Studio/Jan/Prism model folders in `%APPDATA%\RustBlock\settings.json`.
- **Deploy to a runtime.** A finished GGUF can be registered in Ollama through a generated Modelfile and `ollama create`, or copied into the LM Studio, Jan or Prism/llama.cpp model folders; Jan also receives a `model.yml`. Copies run on a worker with progress and cancellation.
- **Window chrome.** The title bar now contains only the supplied wordmark image, a Settings button and canvas-drawn minimize/maximize/restore/close controls. The maximized state switches the icon, closing while busy asks for confirmation, and native file dialogs are owned by the app window so they no longer open behind it.
- **Inspector.** The tab accepts a pasted path as well as the native picker, explains read failures instead of silently doing nothing, and finished jobs/downloads can jump straight into it.
- **Start actions.** The empty-queue Browse/Discover buttons and the header Add files button use the primary style.

**21 November 2026 — full review.** The project (core, CLI, GUI, docs) was re-read and today's evidence recorded:

- **Improvements applied**
  - `ops`: `metadata_set` and `prune` refuse input == output (the writer can no longer replace the source); the test confirms the input stays intact.
  - `download`: HEAD probe with a `Range: bytes=0-0` fallback for servers that reject HEAD.
  - `quality`: correct CSV escaping for tensor names (commas, quotes and newlines no longer break rows).
  - CLI `inspect`: correct native target count (21).
  - GUI: completed jobs show the output file name; help gained new entries; the empty screen lists the shortcuts.
- **Checks:** `cargo test --workspace` passed (core 42, gui 17, transcode 16, end_to_end 6, advanced 8; parity ignored by design); `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets -- -D warnings` passed.
- **Six tests added:** `ops` (same-path refusal for `metadata_set` and `prune`, input preservation), `download` (range-only probe fallback), `quality` (CSV escaping).

**22 November 2026 — GUI on iced 0.14.** The GUI was rewritten on iced (Elm architecture: `Message`, `update`, pure `view`) keeping the palette, the 8 px grid and every feature: queue with targets and verification, budget planner, requantization warning, `llama-bench`, Discover with search/license/resumable downloads and pre-download fit badges, Inspector with metadata and histogram, filterable log, help and `Ctrl+O`/`Ctrl+S`/`Ctrl+L`/`F1`. The window stays undecorated, draggable from the wordmark, with resize grips on all edges and corners.

**Checks (Windows 11, Rust 1.94.1):** `cargo test -p rustblock-gui` passes with **19 tests** (views for every tab and log level, WCAG AA contrast, wordmark decoding); `cargo clippy --workspace --all-targets -- -D warnings` is clean. Real end-to-end run: `stories15M.gguf` (93.8 MiB, F16) opened in the GUI, converted from the button to `stories15M-Q4_K_M.gguf` (20,986,208 B) in the source folder; the Details panel shows the fit badge, the weights/KV/overhead breakdown and the output path.

**22 November 2026 — GUI review and cleanup.** The GUI was re-read and its views rebuilt from scratch (`tab: Tab` → Convert/Discover/**Inspector**, `selected_job`, `logs: Vec<LogEntry>` with level filters, `benchmark: Option<BenchmarkUi>`), then cleaned of development artifacts and test models:

- **Convert:** header with a live summary (`N files · X ready · Y running · Z done · W failed`), benchmark banner, batch bar with progress and Stop, job cards with fit badge, inline speed/ETA, output name and size delta, red errors; details panel on the right.
- **Inspector:** facts (version, alignment, data start, file type), metadata, tensor type histogram with counts/bytes/bars and a memory estimate.
- **Benchmark:** `llama-bench -m <model> -ngl 99` on the largest finished model; the result goes to the log.
- **Discover:** license, richer cards, fit badge before download.
- **Log:** level filters, `visible/total`, text filter, Clear, Close.
- **Tests:** view construction covers all three tabs, the Inspector, log levels, WCAG AA contrast and the wordmark.

This document records the approved plan, the real state of the implementation and the criteria that must be met before a phase can be declared complete. Week estimates are from the original plan and are orders of magnitude, not delivery commitments.

## Legend

| Symbol | Meaning |
|---|---|
| ✅ | Complete and verified |
| ◐ | Partly present, acceptance criteria not yet met |
| ☐ | Planned, not implemented |
| ⛔ | Blocked by an external dependency or a product decision |

## Status summary

| Phase | Weeks | Status | Summary |
|---|---:|---|---|
| 1 — Recipe engine | 1–8 | ✅ | Mixed presets, overrides, protections, CLI/GUI and tests |
| 2 — imatrix | 9–11 | ◐ | Legacy/GGUF parser, weighted overrides, CLI and external generator; real fixtures unavailable |
| 3 — Measurable quality | 12–15 | ◐ | RMSE/SNR/cosine, ranking, JSON/CSV reports, budget planner fallbacks and KLD via `llama-perplexity`; 7B test and 10-model comparison not run |
| 4 — GGUF operations | 16–18 | ◐ | Split/merge with manifest, SHA-256 verification, operational commands, Jinja compilation with minijinja, rope/chat-template validation and template rendering with a JSON context; rope-pair checks against the real runtime remain external |
| 5 — RAM/VRAM fit | 19–20 | ◐ | Weights, per-layer GQA KV, sliding window, MLA and SSM state, hardware, green/yellow/red badge and budget suggestion; real comparison with llmvram/hf-mem not run |
| 6 — Discover and install | 21–24 | ◐ | CLI 16-range/8-file downloads, JSON-lines events, mmproj, quota cache, Ollama/LM Studio hand-off, quick test; GUI uses the parallel downloader, explicit projector handling, LFS hashes and local cache reuse; real resume verified against the Hub hash |
| 7 — Prism family | 25–27 | ◐ | Reblock, Q2_0 auto-detect, PTQ1 rejected, `prism.*` metadata preserved byte for byte; a real 2B reblock loaded, but numerical MTP/Hadamard verification was not run |
| 8 — safetensors → GGUF | late | ◐ | External bridge; native FP8/AWQ dequant; streaming conversion for LLaMA/Mistral/Mixtral/Qwen2/Qwen3 dense and MoE with BPE or SentencePiece, verified with llama.cpp; tensor-by-tensor AWQ comparison open |
| 5.2 — In-app guides | cross-cutting | ◐ | F1 help, “?” popovers on parameters, fit badge, budget suggestion, tooltips; GUI rebuilt and ported to iced on 22 Nov 2026; 5-user test not run |
| 5.3 — Feedback and performance | cross-cutting | ◐ | Filterable log, detailed progress, non-blocking UI, `Ctrl+O`/`Ctrl+S`/`Ctrl+L`/`F1`; startup 0.22–0.43 s, 5-user interaction test not run |
| 5.4 — UI acceptance | cross-cutting | ◐ | WCAG AA verified by an automated test and startup <1 s measured; 5-user test not run |

---

## Open work (external gates)

The remaining criteria need environments, models or data that cannot be produced here. None of this is blocked by code: every section above implemented 80–100% of its area.

| Area | Open gate | Gate type |
|---|---|---|
| Phase 2 (imatrix) | Real fixtures from bartowski/unsloth (or similar) to verify that MSE/SNR improve over the same quantization without imatrix | Real models/data (Phase 3) |
| Phase 2 | IQ* targets usable with pmetal/llama.cpp | External dependency (requires a complete IQ encoder) |
| Phase 3 | 7B test and 10-model comparison; explicit external SNR threshold | Real models (7B/10 models) |
| Phase 4 | Rope pairs and chat-template types checked against the actual runtime | External runtime |
| Phase 5 | Real comparison with `llmvram` and `hf-mem` over 10 models; 70B case on a 24 GB GPU | External models and tools |
| Phase 6 | Installation verified in real Ollama/LM Studio environments (applications installed) | External dependency |
| Phase 7 | Reblock of 5 real Prism PQ2_0 files: numerical MTP/Hadamard verification on real files | External runtime + acceptance test (real fixtures) |
| Phase 8 | Tensor-by-tensor AWQ comparison (up to known numerical differences) | Real models + external tools |
| 5.2 | Usability test with 5 non-expert users completing “download and install a model” | Real user sessions |
| 5.4 | Usability test with 5 non-expert users; today's measured startup is 0.22–0.43 s | Real user sessions |

For these gates the code must preserve several properties, and every new change must preserve them:

- acceptance evidence documented per section (automated tests, CLI output, executed scenarios);
- unverified limits stated explicitly (if something is not verified, it is written here);
- external tools rate-limited (`llama-imatrix`, `llama-perplexity`, `llama-cli`, `llama-server`, `llama-quantize`) — PATH detection, never automatic download.

---

## Phase 1 — Recipe engine (complete)

### Implemented

- Recipe engine in `crates/core/src/recipe.rs` with `Q2_K`, `Q3_K_M`, `Q4_K_M`, `Q5_K_M` presets.
- llama.cpp rules for `attn_v`, `ffn_down`, `attn_output`, output/tied embeddings, GQA and `use_more_bits`.
- Regex and file overrides, always taking priority over the preset; first match wins.
- Protections for norms, MoE routers, SSM/hybrid tensors and non-quantizable tensors.
- Fallbacks for incompatible shapes: `Q4_K → Q5_0`, `Q5_K → Q5_1`, `Q6_K → Q8_0`, `Q2_K/Q3_K/TQ → Q4_0`, then `F16`.
- Uniform `Q3_K_S`, `Q4_K_S`, `Q5_K_S` targets; CLI aliases `q3_k`, `q4_k`, `q5_k` map to the `_M` presets.
- Non-silent requantization: refused by default, explicit `--allow-requantize`, confirmation in the GUI.
- CLI with `--tensor-type`, `--tensor-type-file`, `--dump-tensor-types`, `--allow-requantize`.
- GUI grid with presets visible and `Q4_K_M` as the default for float models.
- Recipes realigned to `llama-quant.cpp` b12359: `use_more_bits` uses integer division `7*n/8`, uniform `_S` presets, `ffn_down` rules based on tensor count and the `Q4_K_S` threshold `layer < n_ffn_down/8`.
- Differential harness `quantize_parity` and CI `parity` job against the pinned b12359 build.
- Golden, override, MoE/SSM, tied embedding, shape fallback, parser, requantization and end-to-end tests.
- README and screenshots updated.

### Known limits

- **Byte-for-byte** output parity is not verified: the harness compares the per-tensor type map, not headers, offsets and payload. The gate in success metric §7 therefore remains open on that point.
- **Evidence 2026-11-19:** the harness (`crates/core/tests/quantize_parity.rs`) ran against pinned llama.cpp **b12359** (commit b1b770d90) and passes for `Q2_K`, `Q3_K_S/M`, `Q4_K_S/M`, `Q5_K_S/M`, `Q6_K` and `Q8_0`: the type map is identical to `llama-quantize` on an 8-layer F16 fixture. The CI `parity` job downloads the same release and repeats the comparison.
- Later phases are not part of Phase 1.

---

## Phase 2 — imatrix (weeks 9–11)

**Status:** ◐ partial; parser, weight application, CLI and external generation present; real fixtures and IQ targets missing.
**Dependencies:** imatrix parser, weighted quantizers, IQ encoder (for the `IQ*` part), real fixtures, Phase 3 quality metric.

### Goals

- Read importance matrices in legacy `.imatrix` and `.gguf` formats.
- Apply weights to the types that support them: `IQ*`, `Q2_K`, `Q3_K`.
- Generate an imatrix only when `llama-imatrix` is available, without reimplementing inference.
- Warn about the overfitting risks of Wikipedia calibration.

### Planned work

1. Parser for both formats with version, size and tensor-name validation.
2. Mapping between imatrix names and GGUF tensor names, with an explicit error for missing or ambiguous tensors.
3. Passing weights to the supported quantizers, keeping current behaviour when no matrix is present.
4. CLI command/option to load the matrix and report which tensors received it.
5. `llama-imatrix` detection through `PATH` or an explicit variable; if absent, the feature is disabled with a clear message. No automatic download or bundling.
6. UI warning and documentation that a Wikipedia calibration may not represent the target domain.

### Acceptance criteria

- Test with real imatrix files (for example bartowski/unsloth fixtures): MSE and SNR improve over the same quantization without imatrix. → **External gate** (Phase 3), see “Open work”.
- If `llama-imatrix` is not found, generation is disabled and the message explains how to install it or provide an existing matrix.
- Reading a malformed matrix fails without modifying the output file.
- The applied weight map is reported and dumped.

### Blocks and scope notes

- RustBlock **does not write IQ targets** yet: the `IQ*` part needs a complete IQ encoder first. The phase can start with `Q2_K`/`Q3_K`, but cannot be called complete until IQ targets exist and are verified.
- “MSE/SNR improve” needs Phase 3 metrics or an equivalent harness; it must not be claimed from the presence of the matrix alone.
- Real fixtures are not in the repository: document provenance, license and hash.
- Evidence updated 2026-11-21: `ImportanceMatrix::load` reads legacy + GGUF, `type_overrides` selects top-N by total importance, `transcode_with_imatrix` applies user overrides before the preset; `generate_with_llama_imatrix` calls the external `llama-imatrix`; `OVERFITTING_WARNING` is published in the CLI.

---

## Phase 3 — Measurable quality (weeks 12–15)

**Status:** ◐ partial; native metrics, ranking, reports, budget planner and external KLD present; 7B test and 10-model comparison missing.
**Update:** the planner now chooses the smallest target when nothing fits and Q4_K_M when hardware is unknown. Evidence 2026-11-21: CSV reports escape tensor names with commas/quotes/newlines; the budget planner has an integration test (`quality_fit_and_ops_work_on_a_real_gguf`).
**Dependencies:** Phase 2 for imatrix-weighted metrics; Phase 4 for GGUF diff.

### Goals

- Compute native per-tensor metrics: RMSE, SNR and cosine similarity; weight them with the imatrix when present.
- Produce a tensor sensitivity ranking.
- Budget quantization: given “X GB at Y context”, assign bits per tensor.
- Optional KLD by calling `llama-perplexity`.
- Export a JSON and CSV report with per-tensor data.

### Planned work

1. Decode original and quantized tensors without loading the whole model in RAM, reusing the memory-bounded pipeline.
2. Per-tensor and aggregate metrics, explicitly excluding non-quantizable tensors.
3. Normalized ranking and category explanation (`attn_v`, `ffn_down`, output, embedding, rest).
4. Budget solver with multiple assignment attempts, block-size constraints and fallbacks.
5. Optional `llama-perplexity` integration for KLD; no reimplemented inference.
6. Versioned JSON/CSV schema, importable and stable for scripts.

### Acceptance criteria

- For a 7B model, budget quantization produces a file that fits the requested limit with the least measurable degradation. → **External gate** (7B model, Phase 8)
- Mean SNR exceeds the agreed threshold and is reported with the per-tensor distribution.
- The JSON/CSV report contains one row per tensor plus totals, and is reproducible for identical input and parameters.
- The sensitivity ranking is stable on a repeatable fixture.

### Notes

- “Least possible degradation” needs a measurable objective; the SNR threshold must be an explicit parameter, not a hidden number.
- The fit estimate used by the budget must be shared with Phase 5 to avoid two divergent formulas.

---

## Phase 4 — GGUF operations (weeks 16–18)

**Status:** ◐ partial; split/merge with manifest, SHA-256 verification, operational commands, Jinja compilation with minijinja, rope/chat-template validation and template rendering with a JSON context present; rope-pair checks against the actual runtime remain external.
**Already present:** GGUF parser, PQ2_0 → Q2_0 reblock, split/merge with original header, metadata, prune, diff, SHA-256/LFS, manifest-verified byte-identical round-trip, `meta --validate` with Jinja compilation and `meta --render-template` with a JSON context.
**Evidence 2026-11-21:** `metadata_set` and `prune` now refuse input == output (integrity: source file stays intact); tests added.
**Missing:** rope-pair checks against the actual runtime.

### Goals

- Shard split/merge in the `-00001-of-0000N` format, with `--keep-split` and `--delete-splits`.
- Full metadata editor: arrays, chat template with validation, rope and other KVs.
- Layer prune and diff between two GGUF files.
- SHA-256 integrity verification against Hugging Face LFS hashes.

### Planned work

1. Shard layout model with global offsets, alignment and `split.*` metadata.
2. Write a GGUF with modified metadata without recompressing tensors.
3. Merge that rebuilds header, tensor table and data in original order.
4. `split`, `merge`, `meta`, `prune`, `diff`, `hash` commands.
5. LFS hash verification handling missing files, unavailable hashes and partial downloads.
6. Chat-template and metadata-type validation before writing.

### Acceptance criteria

- Merge of 3 shards: the result is byte-identical to the original when tensors are not recompressed. → verified on real shards (evidence 2026-11-19, README).
- Editing a token array does not corrupt the GGUF and is read back correctly.
- Split followed by merge preserves offsets, alignment and types.
- A wrong LFS hash is reported clearly and not confused with a conversion error.

### Notes

- Byte-identical merge is possible only when tensor order, alignment and metadata are preserved; the test must fix those conditions.
- Shard reading is also a prerequisite for downloading sharded models in Phase 6.

---

## Phase 5 — RAM/VRAM fit (weeks 19–20)

**Status:** ◐ partial; weights, per-layer GQA KV, sliding window, MLA and SSM state present; real comparison with llmvram/hf-mem and the 70B case not run.
**Update:** the GUI shows a green/yellow/red fit badge with weights, KV cache and overhead detail; “Suggest a format” uses the budget planner on the largest queued model and keeps lossless Reblock for PQ2_0. **Discover shows a badge computed before download** from published bytes + 512 MiB overhead: the KV cache cannot be computed without the header, so it is excluded and stated in the tooltip; with unknown hardware the badge says “Fit unknown”. The 10-model comparison, the 70B/24 GB case and the full (KV-aware) badge in the repository detail remain open because they need real files.

### Goals

- Exact per-tensor byte calculation.
- Correct KV cache for GQA, sliding windows, hybrid attention and MLA.
- “Fits / does not fit” badge in Discover before download.
- Automatic hardware detection.

### Planned work

1. Weight bytes from GGUF types and dimensions, including alignment and overhead.
2. KV cache formula per layer, head, dimension, context, cache type and GQA; sliding-window/hybrid/MLA cases with explicit fallback.
3. Configurable runtime/backend overhead without pretending to unavailable precision.
4. CPU/RAM and GPU/VRAM detection for supported backends, reporting “unknown” when unverifiable.
5. Badge and suggestion in the GUI; estimate also available through CLI/JSON. *(RAM/VRAM badge and budget planner implemented)*
6. Comparison with `llmvram` and `hf-mem` over a model set.

### Acceptance criteria

- Comparison with existing tools over 10 models: error below 5%. → **External gate** (10 real models)
- The estimate for a 70B model on a 24 GB GPU is consistent with reality. → **External gate** (70B on 24 GB)
- Every estimate reports the parameters used (context, cache, overhead) and does not show a definitive badge when hardware is unknown.
- The Discover badge is computed before download. *(implemented for weights only: KV needs the header, absent before download)*

### Notes

- “Fits” depends on runtime, backend and configuration: the UI must show the basis of the estimate, not just the verdict.
- The 10 comparison models and external tool versions must be recorded in the test.

---

## Phase 6 — Discover and installation (weeks 21–24)

**Status:** ◐ partial
**Already present:** Discover, search, repository detail, parallel 16-range download from GUI and CLI, `.part` resume, cancellation, GGUF validation, mmproj CLI/GUI, LFS/SHA-256 from the GUI, detection of existing files, quota cache with selective trim (`cache` CLI and GUI), LM Studio/Ollama installation and quick test, headless download with JSON-lines events (`download --events`, `download-many --events`); the GUI also supports ModelScope and a configurable Hugging Face endpoint.
**Evidence 2026-11-21:** the download HEAD probe now falls back to `Range: bytes=0-0` for servers that reject HEAD (tests added).
**Missing:** installation verified in a real Ollama/LM Studio environment, which requires those applications installed.

### Goals

- Parallel, resumable, verified downloads: up to 16 connections per file and 8 simultaneous files.
- Automatic `mmproj` pairing for vision models.
- Installation into LM Studio/Ollama with correct Modelfile and template; copy into Jan and Prism/llama.cpp folders.
- Optional quick test with `llama-cli`/`llama-server`.
- Managed cache and headless mode with JSON-lines output.

### Planned work

1. Download scheduler with range requests, resume, concurrency limit and backoff.
2. LFS/SHA-256 verification and partial-file handling.
3. mmproj detection and pairing with the main model.
4. LM Studio/Ollama installer through official paths and CLI, without assuming they are installed.
5. Optional load test with `llama-cli`/`llama-server`, timeout and readable output.
6. Cache directory, cleanup and headless commands with JSON-lines events.

### Acceptance criteria

- Ollama installation: the model appears and answers a test prompt. → **External gate** (installed application)
- LM Studio installation: the model is imported and selectable. → **External gate** (installed application)
- Interrupted and resumed download: the final file has the correct hash. (evidence 2026-11-19, README)
- If LM Studio/Ollama/llama.cpp are absent, the feature is disabled with a clear message.

### Notes

- Installations touch external directories and processes: they need explicit confirmation and no silent modification.
- Installation tests need real environments or documented mocks; they must not be claimed from compilation alone.

---

## Phase 7 — Prism family (weeks 25–27)

**Status:** ◐ partial
**Already present:** `PQ2_0 → Q2_0` reblock, lossless verification, block-split tests, explicit PTQ1_0 refusal, “already Q2_0” detection, `prism.*` MTP/Hadamard metadata preserved byte for byte; a real Prism 2B file reblocked and loaded in llama.cpp produced the same prefix as the Prism runtime.
**Missing:** PTQ1_0 evaluation, numerical MTP/Hadamard verification, test on 5 real files.

### Goals

- Consolidate `PQ2_0 → Q2_0` reblocking and bit-exact verification.
- Evaluate `PTQ1_0 → Q2_0` (dense trits) after verifying the layout.
- Automatically detect upstream-compatible files and avoid useless conversions.
- Handle MTP/Hadamard metadata in Bonsai 2 files.

### Acceptance criteria

- Reblock of 5 real PQ2_0 files: output loads in mainline llama.cpp with unchanged perplexity. → **External gate** (5 real files)
- A file already in Q2_0 is reported and not converted automatically. (implemented in the GUI; code evidence)
- PTQ1_0 is not declared lossless until the layout is verified and documented.
- Relevant Hadamard/MTP metadata is preserved or explicitly refused.

### Notes

- Real fixtures and an upstream build are needed; synthetic tests are not enough for acceptance.
- “Same output/perplexity” verification needs an external runtime and is an integration test.

---

## Phase 8 — safetensors → GGUF (optional, late)

**Status:** ◐ partial; implementation complete for its declared scope (Python bridge, native FP8/AWQ → BF16 dequant, streaming conversion for LLaMA/Mistral/Mixtral/Qwen2/Qwen3 dense and MoE with BPE or SentencePiece) and verified with real checkpoints; a tensor-by-tensor AWQ comparison is missing. Other HF architectures are out of scope and must use the bridge or fail explicitly.
**Dependencies:** product decision, optional Python and `convert_hf_to_gguf.py`.

### Goals

- Optional bridge to `convert_hf_to_gguf.py` when present.
- Native implementation for `llama`, `mistral`, `mixtral`, `qwen2` and `qwen3` dense and MoE, with BPE or SentencePiece tokenizer.
- FP8/AWQ import with BF16 dequantization.

### Acceptance criteria

- If `convert_hf_to_gguf.py` is installed, the tool calls it and produces a valid GGUF.
- AWQ → BF16 dequantization is numerically correct.
- If Python or the script is unavailable, the feature is disabled with clear instructions.
- A real checkpoint produces a GGUF readable by the native parser, with tokenizer and architecture metadata; verified with TinyLlama, Qwen2.5 BF16/AWQ/FP8 and a tiny Mixtral, loaded by llama.cpp b12176.

### Notes

- The bridge must not bundle the Python interpreter or Hugging Face repository dependencies.
- Native conversion is a separate project and must not block phases 2–7.
- AWQ and FP8 were tried on real checkpoints: both GGUF files load and generate coherent text. AWQ packing and E4M3FN values have numeric tests, but no tensor-by-tensor AWQ comparison was retained.
- Native conversion covers LLaMA/Mistral/Mixtral/Qwen2/Qwen3 dense and MoE with BPE or SentencePiece; it does not cover every HF architecture. Unsupported cases must fail explicitly or use the bridge.

---

## 5.2 — In-app guides for newcomers

**Status:** ◐ partial

### Already present

- Format descriptions in the GUI and tooltips/previews for choices.
- “?” popovers on output format, destination and verification with an explanation and recommended behaviour.
- Requantization warning modal explaining the degradation.
- Error messages for unsupported files.
- Green/yellow/red fit badge for RAM/VRAM with a tooltip reporting weights, KV cache, overhead and both devices.
- “Suggest a format” button using the budget planner on the largest queued model; PQ2_0 files keep lossless Reblock.
- “VRAM: not detected” in the badge and a “GPU not detected” help entry; the suggestion falls back to RAM and logs it.
- Settings dialog explaining each option; deploy dialog explains what every runtime receives.
- **Evidence 2026-11-22:** help modal updated; the empty screen lists the `Ctrl+O`/`Ctrl+S`/`Ctrl+L`/`F1` shortcuts; Inspector explains a failed read.

### To do

- Usability test with 5 non-expert users.

### Acceptance

- A non-expert completes the main flow without reading the README.
- Every badge explains the estimate and does not promise unverified compatibility.
- Guides stay available offline.

---

## 5.3 — Visual feedback and performance

**Status:** ◐ partial; logs, progress and shortcuts present; startup measured at 0.22–0.43 s on this machine; 5-user interaction test not run.

### Already present

- Progress bars for conversion and verification with speed, ETA and current tensor.
- Long work in a separate worker; the UI stays responsive.
- Download list with bytes, percentage, speed and cancellation.
- Filterable bottom log, `Ctrl+O`, `Ctrl+S`, `Ctrl+L`, `F1` and sub-second startup.

### To do

- Shortcut verification with real focus and a 5-user interaction test.

### Acceptance

- The UI does not block during a large model.
- Logs do not expose tokens or sensitive paths.
- Shortcuts work with consistent focus.

---

## 5.4 — UI acceptance criteria

**Status:** ◐ partial; contrast and startup measured, 5-user test not run.

- Usability test with 5 non-expert users: “download and install a model” completed in under 3 minutes without help (not run).
- Color contrast compliant with WCAG AA (ratio ≥ 4.5:1): verified by an automated test over 12 text/status pairs.
- Application startup under 1 second: measured 0.22–0.43 s on this machine.

These criteria require real measurements and sessions. They must not be claimed from compilation or a visual check alone.

---

## 6. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Divergence from llama.cpp codecs | High | Differential tests in CI against a pinned build |
| Upstream bugs reflected in RustBlock | Medium | Explicit overrides take priority and regression tests |
| Exotic architectures not handled | Medium | Detection, clear message and “do not quantize” fallback |
| Slow downloads | Low | Parallel download with resume |
| GUI complexity for expert users | Low | Separate advanced mode and mirrored CLI |
| GGUF write errors | High | Temporary write, verification, atomic rename and round-trip tests |

## 7. Success metrics

| Area | Metric |
|---|---|
| Correctness | 100% bit-identical match with `llama-quantize` on deterministic types |
| Quality | Mean SNR of budget quantization 10% above standard `Q4_K_M` |
| Usability | 90% of new users complete the basic flow without errors |
| Performance | 7B model downloaded in under 2 minutes at 100 Mbps |

Bit-identical match must be measured against a pinned reference build, not inferred from recipe similarity. The SNR metric needs Phase 3 and an explicit threshold.

---

## 8. Work order and gates

0. **21 November 2026 — full review (done):** project review, roadmap review, improvement implementations, tests, fmt and clippy.
1. **Correctness gate:** ✅ reference build pinned to llama.cpp **b12359** (commit b1b770d90, asset `llama-b12359-bin-ubuntu-x64.tar.gz`) and differential test run successfully on 2026-11-19. The CI `parity` job downloads the same release and repeats the comparison. Only byte-for-byte output parity remains outside.
2. **Phase 2:** imatrix parser, weights for `Q2_K`/`Q3_K`, optional external generation; the IQ part stays blocked until an IQ encoder exists.
3. **Phase 4:** split/merge and metadata, because they unlock shards, hashes and part of Phase 6.
4. **Phase 3:** metrics and budget, reusing the Phase 5 fit estimate.
5. **Phase 5:** RAM/VRAM fit and Discover badge.
6. **Phase 6:** parallel download, mmproj, installers and headless mode.
7. **Phase 7:** Prism validation on real files.
8. **Phase 8:** optional safetensors bridge.

Phases may be merged or reordered only if the correctness gates stay unchanged. Every phase must update this document with evidence, executed tests and residual limits.

## 9. Definition of done

A phase is complete only when:

- acceptance criteria are verified with documented tests or runs;
- the CLI and GUI behave the same, or the difference is documented;
- external errors (runtime, Python, LM Studio, Ollama) are handled with clear messages;
- there are no regressions in existing tests;
- README and this document are updated;
- unverified limits are stated explicitly.
- **Applicable evidence today:** `cargo test --workspace` passes (22 Nov 2026), `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets -- -D warnings` pass; every change after 22 Nov 2026 adds the executed check and new tests and keeps this criterion.

## 10. Fixtures and secrets

- Real fixtures must be documented with provenance, license and hash; model files must not be committed.
- Hugging Face tokens stay in environment variables and are never written to logs or reports.
- External tools (`llama-imatrix`, `llama-perplexity`, `llama-cli`, `llama-server`, Python) are optional and detected, not downloaded automatically.
- No phase may require access to undocumented credentials or services.

## 11. Priorities and recommendation

Engine correctness remains the prerequisite: without verifiable parity with a reference `llama-quantize` build, quality and budget features are not credible. Phase 1 is complete; the next work must start from the correctness gate and the phases that unlock the most dependencies (imatrix and GGUF operations).

The GUI can become the decisive adoption factor, but only after the engine is solid. In-app guides and compatibility badges must not anticipate estimates Phase 5 cannot yet verify.

How to close the open gates: use the shipped code to produce test models for external evaluations (7B/10 models/70B), add the comparison, run Ollama/LM Studio on a dedicated machine and document the results; then publish a 2026-12-XX review.
