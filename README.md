# ternary-reblock

Lossless GGUF reblocker: **Prism PQ2_0 → upstream Q2_0**.

Ternary Bonsai 2 models from [Prism ML](https://huggingface.co/prism-ml) are published in two GGUF layouts:

| Format | Type id | Block | Bytes / block | Support |
|---|---:|---:|---:|---|
| `PQ2_0` | 142 | 128 elems | 34 | Prism fork only |
| `Q2_0` | 42 | 64 elems | 18 | upstream llama.cpp / LM Studio |

Community fine-tunes (uncensored, abliterated, heretic, …) are often released **only as PQ2_0**, so they cannot be loaded by stock llama.cpp or LM Studio. This tool rewrites them to standard `Q2_0` **without dequantizing and requantizing**: the 2-bit ternary codec is identical, only the block grouping changes.

```
PQ2_0 block (128 elems)                  Q2_0 blocks (2 × 64 elems)
┌────────┬────────────────────────┐      ┌────────┬──────────────┐
│ d (f16)│ 32 bytes of 2-bit codes│  →   │ d (f16)│ 16 bytes     │
└────────┴────────────────────────┘      ├────────┼──────────────┤
                                         │ d (f16)│ 16 bytes     │
                                         └────────┴──────────────┘
```

The same `d` is copied to both output blocks and the codes are split in order, so every represented value is **bit-identical**.

## Status

Phase 1: conversion engine + CLI + automated tests.
Phase 2 (planned): desktop GUI (Rust + egui), drag & drop, batch conversion.

## Install / build

Requires Rust 1.80+.

```bash
cargo build --release
# binary: target/release/ternary-reblock (ternary-reblock.exe on Windows)
```

## Usage

Inspect a file (metadata, tensor type histogram, eligibility):

```bash
ternary-reblock inspect model-PQ2_0.gguf
```

Convert:

```bash
ternary-reblock convert model-PQ2_0.gguf model-Q2_0.gguf
```

Options:

```text
--force             overwrite an existing output file
--dry-run           validate and estimate without writing
--verify <level>    none | structure | full   (default: structure)
--json              machine-readable output
```

Verification levels:

- `none` — output is parsed to ensure it is structurally readable.
- `structure` — additionally compares the patched header, tensor table, offsets, alignment and file length (default).
- `full` — additionally decodes every converted element from source and output and compares the `f32` bit patterns; unchanged tensors are byte-compared.

The input is opened **read-only**. The output is written to a temporary file in the destination directory and atomically renamed into place only after verification. No network access is used.

## How it works

1. Parse the GGUF v3 header and validate tensor offsets, alignment, sizes and shapes.
2. Detect `PQ2_0` tensors (type 142); reject `PTQ1_0` (type 143), which has no lossless upstream equivalent.
3. Copy the raw header verbatim and patch only fixed-width fields:
   - tensor type `142 → 42`
   - tensor offsets (recomputed, aligned to `general.alignment`, default 32)
   - `general.file_type → 41` (`LLAMA_FTYPE_MOSTLY_Q2_0`)
4. Stream each tensor: `PQ2_0` blocks are split into two `Q2_0` blocks (34 → 2 × 18 bytes); all other tensors are copied byte-for-byte.
5. Verify and atomically rename the temporary file.

All metadata is preserved, including the `prism.hadamard.*` keys used by the model (the upstream runtime already understands them).

## Tests

```bash
cargo test --workspace
```

The suite covers:

- property tests: random blocks, every decoded element compared bit-for-bit;
- end-to-end conversion of a synthetic GGUF, including metadata preservation;
- offset/alignment recalculation with different `general.alignment` values;
- rejection of PTQ1_0 and of files without PQ2_0 tensors;
- dry-run (no output) and overwrite protection;
- source file left untouched.

## Compatibility

The converted file requires an upstream runtime that supports `Q2_0`:

- llama.cpp **b11555** or newer;
- LM Studio **2.56** or newer (runtime `llama.cpp` ≥ b11555).

The generated file is ~5.8% larger than PQ2_0 (2.125 → 2.25 bits per weight).

## Limitations

- GGUF **v3** only.
- Only `PQ2_0 → Q2_0`. `PTQ1_0` is rejected: converting it would require lossy requantization.
- Files containing unknown tensor types are handled only if those tensors can be sized; conversion refuses unknown types rather than guessing.
- No GUI yet (phase 2).

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

This is an independent community tool. It is **not affiliated with, endorsed by, or sponsored by Prism ML or Alibaba Cloud**. “Bonsai” and “Ternary Bonsai” are products of Prism ML; Qwen3.8-27B is by Alibaba Cloud. No model weights are included in this repository.
