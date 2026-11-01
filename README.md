# RustBlock

<p align="center"><img src="crates/gui/assets/rustblock-logo.png" width="190" alt="RustBlock logo"></p>

**The native Swiss Army knife for GGUF models.** Reblock Prism ternary files, convert floats, quantize, verify, download from Hugging Face, and install into LM Studio or Ollama — from one steel-and-blocks desktop app. No Python, Node.js or terminal knowledge required.

RustBlock opens the source **read-only**, writes to a temporary file in the destination folder, verifies the result, and only then renames it into place. There is no telemetry and no upload of model data.

## What it can do

| Target | What it produces | Lossless? | Best for |
|---|---|---:|---|
| **Reblock** | Prism `PQ2_0` → standard `Q2_0` | **Yes** | Prism Ternary Bonsai files that stock llama.cpp / LM Studio cannot load |
| **F32** | 32-bit float | Yes, from float sources | Maximum compatibility and precision; very large |
| **F16** | 16-bit float | Yes, from F16 | The usual source for quantization; ~2 bytes per weight |
| **BF16** | bfloat16 | Yes, from float sources | Runtimes/training setups that prefer bfloat16 |
| **Q8_0** | 8-bit block quant | No, lossy | Near-original quality, much smaller than F16 |
| **Q5_0** | 5-bit block quant | No, lossy | Middle ground between Q8_0 and Q4_0 |
| **Q5_1** | 5-bit block quant with stored minimum | No, lossy | Older runtime compatibility |
| **Q4_0** | 4-bit block quant | No, lossy | Small and fast; classic legacy quant |
| **Q4_1** | 4-bit block quant with stored minimum | No, lossy | Slightly better than Q4_0 at a similar size |
| **Q2_K** | 2-bit K-quant | No, lossy | Very tight memory; noticeable quality loss |
| **Q3_K** | 3-bit K-quant | No, lossy | Compromise when Q4_K does not fit |
| **Q4_K** | 4-bit K-quant | No, lossy | The usual quality/size sweet spot |
| **Q5_K** | 5-bit K-quant | No, lossy | Better quality than Q4_K at a larger size |
| **Q6_K** | 6-bit K-quant | No, lossy | Near-F16 quality at a smaller size |
| **TQ1_0** | Ternary 1.69-bit | No, lossy | Tiny ternary models; experimental |
| **TQ2_0** | Ternary 2.06-bit | No, lossy | Tiny ternary models; experimental |
| **MXFP4** | 4-bit block float | No, lossy | Recent small/MoE runtimes; experimental |
| **NVFP4** | NVIDIA 4-bit block float | No, lossy | Recent NVIDIA runtimes; experimental |
| **Q1_0** | 1-bit | No, lossy | Extreme compression; experimental |

The K-quant targets are **uniform** layouts. They are not the mixed `Q4_K_M` / `Q5_K_M` presets, and RustBlock does not use an importance matrix. IQ formats can be read as sources, but writing them requires an importance matrix and is intentionally not offered.

### Lossless reblocking

Prism ML publishes Ternary Bonsai GGUFs in two layouts:

| Format | Type id | Block | Bytes / block | Support |
|---|---:|---:|---:|---|
| `PQ2_0` | 142 | 128 elems | 34 | Prism fork only |
| `Q2_0` | 42 | 64 elems | 18 | upstream llama.cpp / LM Studio |

Both use the same 2-bit ternary codec. RustBlock splits each 128-value block into two 64-value blocks with the original scale: no dequantization and no requantization, so every represented value is **bit-identical**.

## GUI

![RustBlock Convert view](docs/gui.png)

![RustBlock Discover view](docs/gui-discover.png)

A dependency-free HTML interface concept is in `docs/ui-concept.html`.

- **Convert** view: drag & drop one or many `.gguf` files, or browse with the native file picker.
- Choose any of the 19 targets with a plain-English description and hover tooltips written for beginners. The layout picker includes float, legacy block, K-quant, ternary, FP4 and 1-bit families; legacy and experimental choices sit under **More formats** so the common options stay visible.
- Batch queue, live speed/ETA, per-file progress, **stop after the current file**, remove finished jobs and exportable log.
- Output next to each input or in a custom folder, with overwrite protection and duplicate-path checks.
- Verification: `None`, `Structure` or `Full` (full decode comparison), plus **Verify only** mode that writes nothing.
- Optional automatic install after conversion:
  - **LM Studio** is detected at the usual `~/.lmstudio/models` and cache paths; the converted GGUF is copied into `RustBlock/<model>/`.
  - **Ollama** is detected from `PATH`; RustBlock runs the official `ollama create` with a generated Modelfile. It never copies a loose GGUF into Ollama's `blobs/` directory.
  - Any custom folder can be selected instead.
- **Discover** view: search Hugging Face directly from the app, paste a full `huggingface.co/author/model` link, sort by downloads, likes or update time, and inspect real card data before downloading.
- The inspector shows the README summary, architecture, context window, GGUF payload, repository size, license, base model, tags, languages and every GGUF file with its real size. Selecting a result loads its details automatically.
- Downloads show progress, speed, cancellation and resumable `.part` files; finished downloads are added to Convert automatically.
- The dark graphite and oxide-red interface uses the included RustBlock logo; square controls and hairline rules replace rounded cards and decorative colour. Inline beginner guidance explains the three conversion steps and recommends Q4_K when in doubt.

Network access is used **only** when you search or download from Hugging Face. Local conversion and installation work offline.

### Hugging Face token

Public repositories work without a token. For gated or private repositories, set a read token before launching:

```bash
export HF_TOKEN=hf_...
# Windows PowerShell: $env:HF_TOKEN = "hf_..."
```

The token is read from the environment and is never written to disk by RustBlock.

## Install / build

Requires Rust **1.92+**.

```bash
cargo build --release
```

Binaries:

- CLI: `target/release/rustblock` (`rustblock.exe` on Windows)
- GUI: `target/release/rustblock-gui` (`rustblock-gui.exe` on Windows)

Launch the GUI with one or more models:

```bash
cargo run -p rustblock-gui --release -- model-PQ2_0.gguf
```

## CLI

Inspect a GGUF (metadata, tensor histogram, reblock and native-conversion eligibility):

```bash
rustblock inspect model.gguf
```

Verify without writing anything:

```bash
rustblock verify model.gguf            # full PQ2_0 check when applicable
rustblock verify model.gguf --quick    # structure only
```

Convert with the lossless default:

```bash
rustblock convert model-PQ2_0.gguf model-Q2_0.gguf
```

Convert to another target:

```bash
rustblock convert model-F16.gguf model-Q4_K.gguf --to q4_k --verify structure
```

`--to` accepts:

```text
reblock | f32 | f16 | bf16 | q8_0 | q5_0 | q4_0 | q2_k | q3_k | q4_k | q5_k | q6_k
```

Other options:

```text
--force             overwrite an existing output file
--dry-run           validate and estimate without writing
--verify <level>    none | structure | full   (default: structure)
--json              machine-readable output
```

`reblock` is the default so existing scripts keep working. It is only valid for PQ2_0 sources; other targets work from any GGUF whose tensor types the native codec supports.

## Supported sources and limits

- GGUF **v3** files with `F32`, `F16`, `BF16`, `Q4_0`, `Q5_0`, `Q8_0`, K-quants, `Q2_0` and Prism `PQ2_0` tensors.
- `PTQ1_0` is rejected: it has no lossless upstream equivalent, and silent requantization would be misleading.
- Unknown tensor types are rejected rather than guessed.
- Raw **safetensors → GGUF is not supported**. That conversion needs model-specific architecture, tokenizer and chat-template handling, normally done by the model author's Python conversion script. RustBlock only converts existing GGUF files.
- Quantization is inherently lossy. `Full` verification catches structural errors and gross numerical damage; it is not a mathematical proof of model quality.
- The GUI runs one batch at a time. A stop request takes effect after the current file.
- Converted `Q2_0` reblocks need llama.cpp **b11555+** or LM Studio **2.56+**. Other targets need a runtime that supports the chosen GGUF type.
- No model weights are bundled.

## Tests

```bash
cargo test --workspace
```

Coverage includes lossless PQ2_0 reblocking, exact F16 conversion, Q4_K/Q8_0 layout and metadata, all target conversions, alignment preservation, source immutability on error, dry-run, overwrite protection and end-to-end metadata preservation.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

RustBlock is an independent community tool. It is **not affiliated with, endorsed by, or sponsored by** Prism ML, Alibaba Cloud, Hugging Face, LM Studio or Ollama. “Bonsai” and “Ternary Bonsai” are products of Prism ML; Qwen is by Alibaba Cloud. All trademarks belong to their respective owners.
