//! Native GGUF-to-GGUF conversion.
//!
//! `transcode` covers three families of operations without Python, llama.cpp or
//! any external process:
//!
//! * lossless reblocking of Prism `PQ2_0` into upstream `Q2_0` (delegated to
//!   the proven [`crate::convert`] path);
//! * float conversion between F32, F16 and BF16;
//! * dequantization/requantization to Q4_0/Q4_1, Q5_0/Q5_1, Q8_0, the
//!   K-quant family, TQ1_0/TQ2_0, MXFP4/NVFP4 and Q1_0.
//!
//! All metadata and the source file are preserved byte-for-byte except for the
//! fixed-width header fields that must change (tensor type, tensor offset,
//! `general.file_type`, and `general.quantization_version`). The output is
//! written to a temporary file in the destination directory and only persisted
//! after the requested verification succeeds.

use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::Instant;

use half::{bf16, f16};
use pmetal_gguf::{dequant, quantize, GgmlType};
use rayon::prelude::*;
use tempfile::NamedTempFile;

#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;

use crate::convert::{convert, ConvertError, ConvertOptions, Phase, Progress, Report, VerifyMode};
use crate::gguf::{align_up, patch_u32, patch_u64, Gguf, TensorInfo};
use crate::imatrix::ImportanceMatrix;
use crate::recipe::{self, RecipeContext, TensorMeta, TensorOverride};
use crate::types::{type_info, TY_PQ2_0, TY_Q2_0};

/// The conversion menu shown by the GUI and CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// Lossless Prism PQ2_0 -> upstream Q2_0 reblocking.
    Reblock,
    /// 32-bit float (largest, maximum compatibility).
    F32,
    /// 16-bit float (the usual source for quantization).
    F16,
    /// bfloat16 (same range as F32, less mantissa).
    Bf16,
    /// 8-bit block quantization (near-lossless).
    Q8_0,
    /// 5-bit block quantization.
    Q5_0,
    /// 5-bit block quantization with a stored minimum.
    Q5_1,
    /// 4-bit block quantization.
    Q4_0,
    /// 4-bit block quantization with a stored minimum.
    Q4_1,
    /// 2-bit K-quant (smallest common K format).
    Q2K,
    /// Uniform 3-bit K-quant (llama.cpp Q3_K_S).
    Q3K,
    /// Uniform 4-bit K-quant (llama.cpp Q4_K_S).
    Q4K,
    /// Uniform 5-bit K-quant (llama.cpp Q5_K_S).
    Q5K,
    /// 6-bit K-quant.
    Q6K,
    /// Mixed 3-bit K-quant preset (llama.cpp Q3_K_M).
    Q3KM,
    /// Mixed 4-bit K-quant preset (llama.cpp Q4_K_M).
    Q4KM,
    /// Mixed 5-bit K-quant preset (llama.cpp Q5_K_M).
    Q5KM,
    /// Ternary 1.69-bit format for ternary-friendly models.
    Tq1_0,
    /// Ternary 2.06-bit format for ternary-friendly models.
    Tq2_0,
    /// 4-bit block-float format used by newer small/MoE models.
    Mxfp4,
    /// NVIDIA 4-bit block-float format.
    Nvfp4,
    /// Extreme 1-bit compression (experimental).
    Q1_0,
}

impl Target {
    pub const ALL: [Target; 22] = [
        Target::Reblock,
        Target::F32,
        Target::F16,
        Target::Bf16,
        Target::Q8_0,
        Target::Q5_0,
        Target::Q5_1,
        Target::Q4_0,
        Target::Q4_1,
        Target::Q2K,
        Target::Q3K,
        Target::Q4K,
        Target::Q5K,
        Target::Q6K,
        Target::Q3KM,
        Target::Q4KM,
        Target::Q5KM,
        Target::Tq1_0,
        Target::Tq2_0,
        Target::Mxfp4,
        Target::Nvfp4,
        Target::Q1_0,
    ];

    pub fn short(self) -> &'static str {
        match self {
            Target::Reblock => "Q2_0",
            Target::F32 => "F32",
            Target::F16 => "F16",
            Target::Bf16 => "BF16",
            Target::Q8_0 => "Q8_0",
            Target::Q5_0 => "Q5_0",
            Target::Q5_1 => "Q5_1",
            Target::Q4_0 => "Q4_0",
            Target::Q4_1 => "Q4_1",
            Target::Q2K => "Q2_K",
            Target::Q3K => "Q3_K_S",
            Target::Q4K => "Q4_K_S",
            Target::Q5K => "Q5_K_S",
            Target::Q6K => "Q6_K",
            Target::Q3KM => "Q3_K_M",
            Target::Q4KM => "Q4_K_M",
            Target::Q5KM => "Q5_K_M",
            Target::Tq1_0 => "TQ1_0",
            Target::Tq2_0 => "TQ2_0",
            Target::Mxfp4 => "MXFP4",
            Target::Nvfp4 => "NVFP4",
            Target::Q1_0 => "Q1_0",
        }
    }

    pub fn title(self) -> &'static str {
        match self {
            Target::Reblock => "Reblock PQ2_0 to Q2_0 (lossless)",
            Target::F32 => "Dequantize to F32 (32-bit float)",
            Target::F16 => "Dequantize to F16 (16-bit float)",
            Target::Bf16 => "Dequantize to BF16 (bfloat16)",
            Target::Q8_0 => "Quantize to Q8_0 (8-bit)",
            Target::Q5_0 => "Quantize to Q5_0 (5-bit)",
            Target::Q5_1 => "Quantize to Q5_1 (5-bit + minimum)",
            Target::Q4_0 => "Quantize to Q4_0 (4-bit)",
            Target::Q4_1 => "Quantize to Q4_1 (4-bit + minimum)",
            Target::Q2K => "Quantize to Q2_K (mixed 2-bit K-quant)",
            Target::Q3K => "Quantize to Q3_K_S (uniform 3-bit K-quant)",
            Target::Q4K => "Quantize to Q4_K_S (uniform 4-bit K-quant)",
            Target::Q5K => "Quantize to Q5_K_S (uniform 5-bit K-quant)",
            Target::Q6K => "Quantize to Q6_K (6-bit K-quant)",
            Target::Q3KM => "Quantize to Q3_K_M (mixed 3-bit K-quant)",
            Target::Q4KM => "Quantize to Q4_K_M (mixed 4-bit K-quant)",
            Target::Q5KM => "Quantize to Q5_K_M (mixed 5-bit K-quant)",
            Target::Tq1_0 => "Quantize to TQ1_0 (ternary 1.69-bit)",
            Target::Tq2_0 => "Quantize to TQ2_0 (ternary 2.06-bit)",
            Target::Mxfp4 => "Quantize to MXFP4 (4-bit block float)",
            Target::Nvfp4 => "Quantize to NVFP4 (NVIDIA 4-bit float)",
            Target::Q1_0 => "Quantize to Q1_0 (1-bit, experimental)",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Target::Reblock => {
                "Only for Prism Ternary Bonsai PQ2_0 files. Rewrites the private 128-value blocks as standard Q2_0 64-value blocks. It is bit-exact and does not decode or requantize anything."
            }
            Target::F32 => {
                "Decodes every weight to 32-bit float. Maximum precision and compatibility, but the file becomes much larger (about 4 bytes per weight)."
            }
            Target::F16 => {
                "Decodes to 16-bit float, the usual input for quantization. Good precision, roughly 2 bytes per weight, and reads in most runtimes."
            }
            Target::Bf16 => {
                "Decodes to bfloat16: the same exponent range as F32 but fewer mantissa bits. Useful for training and runtimes that prefer BF16."
            }
            Target::Q8_0 => {
                "8-bit block quantization. Very close to the original quality and much smaller than F16. A good safe choice when quality matters more than size."
            }
            Target::Q5_0 => {
                "5-bit block quantization. A middle ground between Q8_0 and Q4_0: smaller than Q8_0 with less damage than Q4_0."
            }
            Target::Q5_1 => {
                "5-bit blocks with a stored minimum. An older format that some runtimes still prefer; Q5_K normally gives better quality for the same space."
            }
            Target::Q4_0 => {
                "Classic 4-bit block quantization. Small and fast, but more quality loss than K-quants. Use it for testing or very tight memory."
            }
            Target::Q4_1 => {
                "4-bit blocks with a stored minimum as well as a scale. Slightly better than Q4_0 at a slightly larger size; Q4_K is usually the better modern choice."
            }
            Target::Q2K => {
                "Mixed 2-bit K-quant preset. Very small, with noticeable quality loss. Attention values use Q4_K when GQA is at least 4, ffn_down and attention output use Q3_K, and output/tied embeddings use Q6_K. Best for large models that otherwise would not fit in memory."
            }
            Target::Q3K => {
                "Uniform Q3_K_S. Every quantizable weight uses Q3_K, except the adaptive protection of output, embeddings and value tensors. Prefer a mixed preset for normal use."
            }
            Target::Q4K => {
                "Uniform Q4_K_S. Every quantizable weight uses Q4_K, except the adaptive protection of output, embeddings and value tensors. The mixed Q4_K_M preset is usually the better choice."
            }
            Target::Q5K => {
                "Uniform Q5_K_S. Every quantizable weight uses Q5_K, except the adaptive protection of output, embeddings and value tensors. The mixed Q5_K_M preset is usually the better choice."
            }
            Target::Q6K => {
                "6-bit K-quant. Near-F16 quality at a smaller size; usually the highest-quality practical quant."
            }
            Target::Q3KM => {
                "Mixed Q3_K_M preset matching llama.cpp: the first two attention-value tensors use Q5_K, other attention values and most ffn_down tensors use Q4_K, early ffn_down tensors use Q5_K, attention output uses Q4_K, and output/tied embeddings use Q6_K."
            }
            Target::Q4KM => {
                "Mixed Q4_K_M preset matching llama.cpp: Q4_K base with Q6_K on the first/last eighth and every third tensor of attention values and ffn_down, plus Q6_K output/tied embeddings. The usual quality/size sweet spot."
            }
            Target::Q5KM => {
                "Mixed Q5_K_M preset matching llama.cpp: Q5_K base with Q6_K on the first/last eighth and every third tensor of attention values and ffn_down, plus Q6_K output/tied embeddings."
            }
            Target::Tq1_0 => {
                "Ternary 1.69-bit format for models trained to tolerate ternary weights. Tiny, but only useful with a compatible runtime and a model that survives it. Experimental."
            }
            Target::Tq2_0 => {
                "Ternary 2.06-bit format with a little more precision than TQ1_0. Still experimental and runtime-dependent."
            }
            Target::Mxfp4 => {
                "4-bit block-float format used by newer small and mixture-of-experts models. Needs a recent runtime; not a universal replacement for K-quants."
            }
            Target::Nvfp4 => {
                "NVIDIA 4-bit block-float format. Requires recent runtime and GPU support; experimental for general models."
            }
            Target::Q1_0 => {
                "1-bit compression. The smallest option here and the most destructive; use it only when nothing else fits. Experimental."
            }
        }
    }

    pub fn type_id(self) -> Option<u32> {
        match self {
            Target::Reblock => Some(TY_Q2_0),
            Target::F32 => Some(0),
            Target::F16 => Some(1),
            Target::Bf16 => Some(30),
            Target::Q8_0 => Some(8),
            Target::Q5_0 => Some(6),
            Target::Q5_1 => Some(7),
            Target::Q4_0 => Some(2),
            Target::Q4_1 => Some(3),
            Target::Q2K => Some(10),
            Target::Q3K => Some(11),
            Target::Q4K => Some(12),
            Target::Q5K => Some(13),
            Target::Q6K => Some(14),
            Target::Q3KM => Some(11),
            Target::Q4KM => Some(12),
            Target::Q5KM => Some(13),
            Target::Tq1_0 => Some(34),
            Target::Tq2_0 => Some(35),
            Target::Mxfp4 => Some(39),
            Target::Nvfp4 => Some(40),
            Target::Q1_0 => Some(41),
        }
    }

    pub fn is_quantized(self) -> bool {
        matches!(
            self,
            Target::Q8_0
                | Target::Q5_0
                | Target::Q5_1
                | Target::Q4_0
                | Target::Q4_1
                | Target::Q2K
                | Target::Q3K
                | Target::Q4K
                | Target::Q5K
                | Target::Q6K
                | Target::Q3KM
                | Target::Q4KM
                | Target::Q5KM
                | Target::Tq1_0
                | Target::Tq2_0
                | Target::Mxfp4
                | Target::Nvfp4
                | Target::Q1_0
        )
    }

    /// True for a preset that mixes several tensor types. Uniform targets use
    /// the legacy adaptive protection instead.
    pub fn is_mixed(self) -> bool {
        matches!(
            self,
            Target::Q2K | Target::Q3KM | Target::Q4KM | Target::Q5KM
        )
    }

    /// Ternary layouts are handled separately in the UI and are never mixed
    /// with the adaptive precision policy below.
    pub fn is_ternary(self) -> bool {
        matches!(self, Target::Tq1_0 | Target::Tq2_0)
    }

    /// The type used for tensors that are more sensitive to quantization.
    /// This is a small, model-agnostic policy: architectures that name their
    /// output projection, embeddings or value projection in the usual GGUF way
    /// get extra precision there. It is not calibration and does not guess an
    /// architecture from metadata.
    pub fn protected_target(self) -> Option<Target> {
        match self {
            Target::Q3K => Some(Target::Q5K),
            Target::Q4K | Target::Q5K => Some(Target::Q6K),
            _ => None,
        }
    }

    pub fn file_type(self) -> Option<u32> {
        match self {
            Target::Reblock => Some(41),
            Target::F32 => Some(0),
            Target::F16 => Some(1),
            Target::Bf16 => Some(32),
            Target::Q8_0 => Some(7),
            Target::Q5_0 => Some(8),
            Target::Q5_1 => Some(9),
            Target::Q4_0 => Some(2),
            Target::Q4_1 => Some(3),
            Target::Q2K => Some(10),
            Target::Q3K => Some(11),
            Target::Q4K => Some(14),
            Target::Q5K => Some(16),
            Target::Q6K => Some(18),
            Target::Q3KM => Some(12),
            Target::Q4KM => Some(15),
            Target::Q5KM => Some(17),
            Target::Tq1_0 => Some(36),
            Target::Tq2_0 => Some(37),
            Target::Mxfp4 => Some(38),
            Target::Nvfp4 => Some(39),
            Target::Q1_0 => Some(40),
        }
    }
}

/// True when at least one tensor can be decoded by the native engine.
pub fn source_supported(ty: u32) -> bool {
    ty == TY_PQ2_0 || ty == TY_Q2_0 || GgmlType::try_from(ty).is_ok_and(dequant::is_supported)
}

#[derive(Debug, Clone, Copy)]
struct Plan {
    index: usize,
    out_ty: u32,
    new_offset: u64,
    new_size: u64,
    copy_raw: bool,
}

/// Converts a GGUF file to `target` using native codecs.
///
/// The source is opened read-only. For `Target::Reblock` this delegates to the
/// specialised lossless implementation, including its full bit-exact check.
pub fn transcode(
    input: &Path,
    output: &Path,
    target: Target,
    opts: ConvertOptions,
    on_progress: Option<&mut dyn FnMut(&Progress)>,
) -> Result<Report, ConvertError> {
    transcode_with_overrides(input, output, target, opts, &[], on_progress)
}

/// Like [`transcode`], with explicit per-tensor recipe overrides.
pub fn transcode_with_overrides(
    input: &Path,
    output: &Path,
    target: Target,
    opts: ConvertOptions,
    overrides: &[TensorOverride],
    mut on_progress: Option<&mut dyn FnMut(&Progress)>,
) -> Result<Report, ConvertError> {
    if target == Target::Reblock && !overrides.is_empty() {
        return Err(ConvertError::Codec(
            "tensor-type overrides cannot be used with the lossless reblock target".into(),
        ));
    }
    if target == Target::Reblock {
        return convert(input, output, opts, on_progress);
    }

    let started = Instant::now();
    let g = Gguf::open(input)?;
    if g.has_ptq1() {
        return Err(ConvertError::Ptq1);
    }
    if g.tensors.iter().any(|t| !source_supported(t.ty)) {
        let t = g
            .tensors
            .iter()
            .find(|t| !source_supported(t.ty))
            .expect("checked above");
        return Err(ConvertError::UnsupportedTensor {
            name: t.name.clone(),
            ty: t.ty,
        });
    }

    let (context, metas) = recipe::analyze_tensors(&g.tensors);
    let plans = plan_layout(
        &g,
        target,
        &context,
        &metas,
        overrides,
        opts.allow_requantize,
    )?;
    let new_data_size = plans
        .iter()
        .map(|p| p.new_offset + p.new_size)
        .max()
        .unwrap_or(0);
    let (header, new_data_start) = build_header(&g, target, &plans)?;
    let new_file_size = new_data_start
        .checked_add(new_data_size)
        .ok_or(ConvertError::Overflow)?;
    let old_data_size = g.file_len - g.data_start;

    if opts.dry_run {
        return Ok(Report {
            input: input.to_path_buf(),
            output: output.to_path_buf(),
            tensors_total: g.tensors.len(),
            tensors_pq2: g.pq2_count(),
            old_file_size: g.file_len,
            new_file_size,
            old_data_size,
            new_data_size,
            elapsed: started.elapsed(),
            dry_run: true,
            verified: opts.verify,
        });
    }

    if paths_same(input, output)? {
        return Err(ConvertError::SameFile);
    }
    if output.exists() && !opts.force {
        return Err(ConvertError::OutputExists(output.to_path_buf()));
    }

    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.exists() {
        fs::create_dir_all(parent)?;
    }

    let mut tmp = NamedTempFile::new_in(parent)?;
    tmp.write_all(&header)?;
    tmp.as_file().set_len(new_file_size)?;

    let src = File::open(input)?;
    let threads = conversion_workers();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("rustblock-convert-{i}"))
        .build()
        .map_err(|e| ConvertError::Codec(format!("cannot start conversion workers: {e}")))?;
    let mut written = 0u64;

    // Tensors are independent, so encode and write a bounded batch in parallel.
    // The output is preallocated and each tensor is written at its final offset
    // with positional writes, so encoding no longer waits for a serial append.
    // The batch budget prevents a large model from materializing all of its
    // output at once. When a batch has fewer tensors than workers, encode_plan
    // also splits a large tensor into source/target block-aligned spans.
    let mut batch_start = 0usize;
    while batch_start < plans.len() {
        let batch_end = next_batch_end(&g, &plans, batch_start, threads);
        let batch = &plans[batch_start..batch_end];
        let parallel_inner = batch.len() < threads && threads > 1;
        let completed: Vec<Result<u64, ConvertError>> = pool.install(|| {
            batch
                .par_iter()
                .map(|plan| {
                    let bytes = encode_plan(&src, &g, plan, target, parallel_inner, threads)?;
                    if bytes.len() as u64 != plan.new_size {
                        let t = &g.tensors[plan.index];
                        return Err(ConvertError::Codec(format!(
                            "tensor '{}' encoded to {} bytes, expected {}",
                            t.name,
                            bytes.len(),
                            plan.new_size
                        )));
                    }
                    write_all_at(tmp.as_file(), new_data_start + plan.new_offset, &bytes)?;
                    Ok(plan.new_offset + plan.new_size)
                })
                .collect()
        });

        for (plan, result) in batch.iter().zip(completed) {
            let end = result?;
            if end < written {
                return Err(ConvertError::Verify(
                    "internal error: tensor layout is not monotonic".into(),
                ));
            }
            written = end;
            emit(
                &mut on_progress,
                Phase::Converting,
                plan.index,
                &g.tensors[plan.index].name,
                g.tensors.len(),
                written,
                new_data_size,
            );
        }
        batch_start = batch_end;
    }

    tmp.flush()?;
    tmp.as_file().sync_all()?;
    if tmp.as_file().metadata()?.len() != new_file_size {
        return Err(ConvertError::Verify(format!(
            "output size is {} bytes, expected {}",
            tmp.as_file().metadata()?.len(),
            new_file_size
        )));
    }

    let g_out = Gguf::open(tmp.path())?;
    if opts.verify != VerifyMode::None {
        emit(
            &mut on_progress,
            Phase::Verifying,
            0,
            "",
            g.tensors.len(),
            0,
            new_data_size,
        );
        verify_structure(&g, &g_out, &plans, target)?;
    }
    if opts.verify == VerifyMode::Full {
        verify_values(&g, &g_out, input, tmp.path(), target, &mut on_progress)?;
    }

    if output.exists() {
        if !opts.force {
            return Err(ConvertError::OutputExists(output.to_path_buf()));
        }
        fs::remove_file(output)?;
    }
    tmp.persist(output).map_err(|e| ConvertError::Io(e.error))?;

    Ok(Report {
        input: input.to_path_buf(),
        output: output.to_path_buf(),
        tensors_total: g.tensors.len(),
        tensors_pq2: g.pq2_count(),
        old_file_size: g.file_len,
        new_file_size,
        old_data_size,
        new_data_size,
        elapsed: started.elapsed(),
        dry_run: false,
        verified: opts.verify,
    })
}

/// Quantize with an importance matrix. Importance selects the highest-value
/// tensors for a higher-precision concrete type before the normal recipe is
/// evaluated; explicit user overrides still come first.
#[allow(clippy::too_many_arguments)]
pub fn transcode_with_imatrix(
    input: &Path,
    output: &Path,
    target: Target,
    opts: ConvertOptions,
    overrides: &[TensorOverride],
    imatrix: &ImportanceMatrix,
    percentile: f32,
    on_progress: Option<&mut dyn FnMut(&Progress)>,
) -> Result<Report, ConvertError> {
    let g = Gguf::open(input)?;
    let mut combined = overrides.to_vec();
    combined.extend(
        imatrix
            .type_overrides(&g.tensors, target, percentile)
            .map_err(|error| ConvertError::Codec(error.to_string()))?,
    );
    transcode_with_overrides(input, output, target, opts, &combined, on_progress)
}

/// Number of workers for one conversion. The explicit environment override is
/// useful for benchmarking and for leaving headroom on a busy machine.
pub fn conversion_workers() -> usize {
    std::env::var("RUSTBLOCK_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|value| value.get())
                .unwrap_or(1)
        })
        .min(256)
}

/// Keep a parallel batch under a memory budget. A tensor is never split just
/// to fit the budget: one oversized tensor still gets the whole pool and is
/// encoded in block-aligned chunks.
fn next_batch_end(g: &Gguf, plans: &[Plan], start: usize, threads: usize) -> usize {
    const BATCH_PEAK_BYTES: u64 = 768 * 1024 * 1024;
    let max_tensors = threads.saturating_mul(8).max(1);
    let mut end = start;
    let mut peak = 0u64;
    while end < plans.len() && end - start < max_tensors {
        let plan = &plans[end];
        let tensor = &g.tensors[plan.index];
        let source = tensor.size.unwrap_or(0);
        let cost = if plan.copy_raw {
            source
        } else {
            source
                .saturating_add(plan.new_size)
                .saturating_add(tensor.element_count().saturating_mul(4))
        };
        if end > start && peak.saturating_add(cost) > BATCH_PEAK_BYTES {
            break;
        }
        peak = peak.saturating_add(cost);
        end += 1;
        if peak >= BATCH_PEAK_BYTES {
            break;
        }
    }
    end.max(start + 1)
}

fn encode_plan(
    src: &File,
    g: &Gguf,
    plan: &Plan,
    target: Target,
    parallel_inner: bool,
    threads: usize,
) -> Result<Vec<u8>, ConvertError> {
    let tensor = &g.tensors[plan.index];
    if plan.copy_raw {
        let source_size = tensor.size.ok_or_else(|| ConvertError::UnsupportedTensor {
            name: tensor.name.clone(),
            ty: tensor.ty,
        })?;
        let mut raw = vec![0u8; source_size as usize];
        read_exact_at(src, g.data_start + tensor.offset, &mut raw)?;
        return Ok(raw);
    }

    let total = tensor.element_count();
    let (source_block, _) = source_layout(tensor)?;
    let target_block = type_info(plan.out_ty).map(|info| info.block).unwrap_or(1);
    let unit = lcm(source_block, target_block).max(1);
    // Give the pool enough independent spans even for one medium-sized
    // tensor. A fixed 1M-element chunk left most cores idle on 2M-value
    // tensors; this targets at least two spans per worker while keeping reads
    // reasonably large.
    let cap = (1_048_576u64 / unit).max(1) * unit;
    let desired = total
        .div_ceil((threads.max(1) as u64).saturating_mul(2))
        .max(unit);
    let chunk = (desired.min(cap) / unit).max(1) * unit;
    if parallel_inner && total > chunk {
        let chunks = total.div_ceil(chunk);
        let parts: Vec<Result<Vec<u8>, ConvertError>> = (0..chunks)
            .into_par_iter()
            .map(|index| {
                let start = index.saturating_mul(chunk);
                let count = chunk.min(total - start);
                encode_span(src, g, tensor, plan, target, start, count)
            })
            .collect();
        let mut out = Vec::with_capacity(plan.new_size as usize);
        for part in parts {
            out.extend_from_slice(&part?);
        }
        return Ok(out);
    }

    let source_size = tensor.size.ok_or_else(|| ConvertError::UnsupportedTensor {
        name: tensor.name.clone(),
        ty: tensor.ty,
    })?;
    let mut raw = vec![0u8; source_size as usize];
    read_exact_at(src, g.data_start + tensor.offset, &mut raw)?;
    let values = decode_tensor(&raw, tensor)?;
    encode_values(&values, target, plan.out_ty, tensor)
}

fn encode_span(
    src: &File,
    g: &Gguf,
    tensor: &TensorInfo,
    plan: &Plan,
    target: Target,
    element_start: u64,
    element_count: u64,
) -> Result<Vec<u8>, ConvertError> {
    let (source_block, source_bytes) = source_layout(tensor)?;
    let byte_start = (element_start / source_block)
        .checked_mul(source_bytes)
        .ok_or(ConvertError::Overflow)?;
    let byte_len = (element_count / source_block)
        .checked_mul(source_bytes)
        .ok_or(ConvertError::Overflow)?;
    let mut raw = vec![0u8; byte_len as usize];
    read_exact_at(src, g.data_start + tensor.offset + byte_start, &mut raw)?;
    let values = decode_chunk(&raw, tensor, element_count)?;
    encode_values(&values, target, plan.out_ty, tensor)
}

fn source_layout(tensor: &TensorInfo) -> Result<(u64, u64), ConvertError> {
    if tensor.ty == TY_PQ2_0 {
        return Ok((128, 34));
    }
    if tensor.ty == TY_Q2_0 {
        return Ok((64, 18));
    }
    type_info(tensor.ty)
        .map(|info| (info.block, info.bytes))
        .ok_or_else(|| ConvertError::UnsupportedTensor {
            name: tensor.name.clone(),
            ty: tensor.ty,
        })
}

fn decode_chunk(raw: &[u8], tensor: &TensorInfo, elements: u64) -> Result<Vec<f32>, ConvertError> {
    if tensor.ty == TY_PQ2_0 {
        return Ok(decode_ternary(raw, 128, 34));
    }
    if tensor.ty == TY_Q2_0 {
        return Ok(decode_ternary(raw, 64, 18));
    }
    let dtype = GgmlType::try_from(tensor.ty).map_err(|_| ConvertError::UnsupportedTensor {
        name: tensor.name.clone(),
        ty: tensor.ty,
    })?;
    if elements > i32::MAX as u64 {
        return Err(ConvertError::Codec(format!(
            "chunk for tensor '{}' is too large for the codec",
            tensor.name
        )));
    }
    dequant::dequantize(raw, dtype, &[elements as i32])
        .map_err(|e| ConvertError::Codec(format!("cannot decode tensor '{}': {e}", tensor.name)))
}

fn lcm(a: u64, b: u64) -> u64 {
    fn gcd(mut x: u64, mut y: u64) -> u64 {
        while y != 0 {
            let rest = x % y;
            x = y;
            y = rest;
        }
        x
    }
    if a == 0 || b == 0 {
        0
    } else {
        (a / gcd(a, b)).saturating_mul(b)
    }
}

#[cfg(unix)]
fn read_exact_at(file: &File, offset: u64, buf: &mut [u8]) -> io::Result<()> {
    FileExt::read_exact_at(file, buf, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &File, offset: u64, buf: &mut [u8]) -> io::Result<()> {
    let mut done = 0usize;
    while done < buf.len() {
        let n = file.seek_read(&mut buf[done..], offset + done as u64)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "tensor data ended early",
            ));
        }
        done += n;
    }
    Ok(())
}

#[cfg(unix)]
fn write_all_at(file: &File, offset: u64, mut buf: &[u8]) -> io::Result<()> {
    let mut done = 0u64;
    while !buf.is_empty() {
        let n = FileExt::write_at(file, buf, offset + done)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "could not write tensor data",
            ));
        }
        done += n as u64;
        buf = &buf[n..];
    }
    Ok(())
}

#[cfg(windows)]
fn write_all_at(file: &File, offset: u64, mut buf: &[u8]) -> io::Result<()> {
    let mut done = 0u64;
    while !buf.is_empty() {
        let n = file.seek_write(buf, offset + done)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "could not write tensor data",
            ));
        }
        done += n as u64;
        buf = &buf[n..];
    }
    Ok(())
}

fn plan_layout(
    g: &Gguf,
    target: Target,
    context: &RecipeContext,
    metas: &[TensorMeta],
    overrides: &[TensorOverride],
    allow_requantize: bool,
) -> Result<Vec<Plan>, ConvertError> {
    let mut order: Vec<usize> = (0..g.tensors.len()).collect();
    order.sort_by_key(|&i| g.tensors[i].offset);

    let mut cursor = 0u64;
    let mut plans = Vec::with_capacity(order.len());
    for index in order {
        let t = &g.tensors[index];
        let elems = t.element_count();
        let requested = recipe::select_type(target, t, &metas[index], context, overrides);
        let out_ty = requested.and_then(Target::type_id).unwrap_or(t.ty);

        if !allow_requantize
            && is_quantized_type(t.ty)
            && is_quantized_type(out_ty)
            && t.ty != out_ty
        {
            return Err(ConvertError::Requantize {
                name: t.name.clone(),
                from: crate::types::type_name(t.ty),
                to: crate::types::type_name(out_ty),
            });
        }

        let info = type_info(out_ty).ok_or_else(|| ConvertError::UnsupportedTensor {
            name: t.name.clone(),
            ty: out_ty,
        })?;
        if !elems.is_multiple_of(info.block) {
            return Err(ConvertError::Codec(format!(
                "tensor '{}' has {} values, not a multiple of the block size {}",
                t.name, elems, info.block
            )));
        }
        let new_size = (elems / info.block)
            .checked_mul(info.bytes)
            .ok_or(ConvertError::Overflow)?;
        let new_offset = align_up(cursor, g.alignment).ok_or(ConvertError::Overflow)?;
        cursor = new_offset
            .checked_add(new_size)
            .ok_or(ConvertError::Overflow)?;
        plans.push(Plan {
            index,
            out_ty,
            new_offset,
            new_size,
            copy_raw: t.ty == out_ty && t.size == Some(new_size),
        });
    }
    Ok(plans)
}

fn is_quantized_type(ty: u32) -> bool {
    type_info(ty).is_some() && !matches!(ty, 0 | 1 | 24..=28 | 30)
}

fn build_header(g: &Gguf, target: Target, plans: &[Plan]) -> Result<(Vec<u8>, u64), ConvertError> {
    let has_ftype = g.kv.iter().any(|kv| kv.key == "general.file_type");
    let has_qver =
        g.kv.iter()
            .any(|kv| kv.key == "general.quantization_version");
    let add_ftype = !has_ftype && target.file_type().is_some();
    let add_qver = target.is_quantized() && !has_qver;

    let mut added = Vec::new();
    if add_qver {
        push_kv_u32(&mut added, "general.quantization_version", 2);
    }
    if add_ftype {
        push_kv_u32(
            &mut added,
            "general.file_type",
            target.file_type().expect("checked above"),
        );
    }
    let added_count = add_qver as u64 + add_ftype as u64;
    let delta = added.len() as u64;

    let mut header = Vec::with_capacity(g.tensor_info_end as usize + added.len() + 32);
    header.extend_from_slice(&g.header[..g.tensor_info_start as usize]);
    header.extend_from_slice(&added);
    header.extend_from_slice(&g.header[g.tensor_info_start as usize..g.tensor_info_end as usize]);

    if added_count > 0 {
        let old = u64::from_le_bytes(
            g.header[16..24]
                .try_into()
                .expect("GGUF kv count is 8 bytes"),
        );
        let new = old.checked_add(added_count).ok_or(ConvertError::Overflow)?;
        header[16..24].copy_from_slice(&new.to_le_bytes());
    }

    for plan in plans {
        let t = &g.tensors[plan.index];
        patch_u32(&mut header, t.type_pos + delta, plan.out_ty)?;
        patch_u64(&mut header, t.offset_pos + delta, plan.new_offset)?;
    }

    if let Some(pos) = g.file_type_pos {
        patch_u32(
            &mut header,
            pos,
            target
                .file_type()
                .ok_or_else(|| ConvertError::Codec("missing file type".into()))?,
        )?;
    }
    if let Some(kv) =
        g.kv.iter()
            .find(|kv| kv.key == "general.quantization_version")
    {
        patch_u32(&mut header, kv.value_pos, 2)?;
    }

    let data_start = align_up(header.len() as u64, g.alignment).ok_or(ConvertError::Overflow)?;
    header.resize(data_start as usize, 0);
    Ok((header, data_start))
}

fn push_kv_u32(out: &mut Vec<u8>, key: &str, value: u32) {
    out.extend_from_slice(&(key.len() as u64).to_le_bytes());
    out.extend_from_slice(key.as_bytes());
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&value.to_le_bytes());
}

fn decode_tensor(raw: &[u8], t: &TensorInfo) -> Result<Vec<f32>, ConvertError> {
    if t.ty == TY_PQ2_0 {
        return Ok(decode_ternary(raw, 128, 34));
    }
    if t.ty == TY_Q2_0 {
        return Ok(decode_ternary(raw, 64, 18));
    }
    let dtype = GgmlType::try_from(t.ty).map_err(|_| ConvertError::UnsupportedTensor {
        name: t.name.clone(),
        ty: t.ty,
    })?;
    if !t.dims.iter().all(|d| *d <= i32::MAX as u64) {
        return Err(ConvertError::Codec(format!(
            "tensor '{}' has a dimension too large for the codec",
            t.name
        )));
    }
    let shape: Vec<i32> = t.dims.iter().map(|d| *d as i32).collect();
    dequant::dequantize(raw, dtype, &shape)
        .map_err(|e| ConvertError::Codec(format!("cannot decode tensor '{}': {e}", t.name)))
}

fn decode_ternary(raw: &[u8], values_per_block: usize, bytes_per_block: usize) -> Vec<f32> {
    let blocks = raw.len() / bytes_per_block;
    let mut out = Vec::with_capacity(blocks * values_per_block);
    for block in raw.chunks_exact(bytes_per_block) {
        let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        for j in 0..values_per_block {
            let byte = block[2 + j / 4];
            let q = (byte >> ((j % 4) * 2)) & 0b11;
            out.push((q as i32 - 1) as f32 * d);
        }
    }
    out
}

fn encode_values(
    values: &[f32],
    target: Target,
    out_ty: u32,
    t: &TensorInfo,
) -> Result<Vec<u8>, ConvertError> {
    match out_ty {
        0 => Ok(values.iter().flat_map(|v| v.to_le_bytes()).collect()),
        1 => Ok(values
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_bits().to_le_bytes())
            .collect()),
        30 => Ok(values
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
            .collect()),
        2 => Ok(encode_q4_0(values)),
        6 => Ok(encode_q5_0(values)),
        8 => Ok(encode_q8_0(values)),
        13 => Ok(encode_q5k(values)),
        3 | 7 | 34 | 35 | 39 | 40 | 41 => {
            let dtype = GgmlType::try_from(out_ty)
                .map_err(|_| ConvertError::Codec(format!("no codec for tensor type {out_ty}")))?;
            quantize::quantize(values, dtype).map_err(|e| {
                ConvertError::Codec(format!(
                    "cannot quantize tensor '{}' to {}: {e}",
                    t.name,
                    target.short()
                ))
            })
        }
        10..=14 => {
            let dtype = GgmlType::try_from(out_ty)
                .map_err(|_| ConvertError::Codec(format!("no codec for tensor type {out_ty}")))?;
            quantize::quantize(values, dtype).map_err(|e| {
                ConvertError::Codec(format!(
                    "cannot quantize tensor '{}' to {}: {e}",
                    t.name,
                    target.short()
                ))
            })
        }
        _ => Err(ConvertError::UnsupportedTensor {
            name: t.name.clone(),
            ty: out_ty,
        }),
    }
}

fn encode_q8_0(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() / 32 * 34);
    for block in values.chunks_exact(32) {
        let max = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = max / 127.0;
        out.extend_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        let inv = if d == 0.0 { 0.0 } else { 1.0 / d };
        for v in block {
            let q = (v * inv).round().clamp(-127.0, 127.0) as i8;
            out.push(q as u8);
        }
    }
    out
}

fn encode_q4_0(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() / 32 * 18);
    for block in values.chunks_exact(32) {
        let max = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = max / 8.0;
        out.extend_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        let inv = if d == 0.0 { 0.0 } else { 1.0 / d };
        let mut q = [0u8; 32];
        for (i, v) in block.iter().enumerate() {
            q[i] = ((v * inv).round().clamp(-8.0, 7.0) as i8 + 8) as u8;
        }
        for i in 0..16 {
            out.push(q[i] | (q[i + 16] << 4));
        }
    }
    out
}

fn encode_q5_0(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() / 32 * 22);
    for block in values.chunks_exact(32) {
        let max = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = max / 16.0;
        out.extend_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        let inv = if d == 0.0 { 0.0 } else { 1.0 / d };
        let mut q = [0u8; 32];
        let mut high = [0u8; 4];
        for (i, v) in block.iter().enumerate() {
            let value = (v * inv).round().clamp(-16.0, 15.0) as i8 + 16;
            q[i] = value as u8;
            if value & 0x10 != 0 {
                high[i / 8] |= 1 << (i % 8);
            }
        }
        out.extend_from_slice(&high);
        for i in 0..16 {
            out.push((q[i] & 0x0f) | ((q[i + 16] & 0x0f) << 4));
        }
    }
    out
}

/// Q5_K packed directly instead of relying on pmetal's Q5K quantizer, whose
/// high-bit layout is incompatible with its own decoder in 0.5.0. The layout
/// here follows the GGUF Q5_K block: d, dmin, 12 packed 6-bit scale/min pairs,
/// 32 high-bit bytes, then 128 low-nibble bytes.
fn encode_q5k(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() / 256 * 176);
    for block in values.chunks_exact(256) {
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        let mut q = [0u8; 256];
        let mut block_scale = 0f32;
        let mut block_min = 0f32;

        for chunk in block.chunks_exact(32) {
            let min = chunk.iter().fold(0f32, |m, v| m.min(*v));
            let max = chunk.iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v));
            let scale = (max - min) / 31.0;
            block_scale = block_scale.max(scale);
            block_min = block_min.max(-min);
        }

        let d = f16::from_f32(block_scale / 63.0);
        let dmin = f16::from_f32(block_min / 63.0);
        for (j, chunk) in block.chunks_exact(32).enumerate() {
            let min = chunk.iter().fold(0f32, |m, v| m.min(*v));
            let max = chunk.iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v));
            let scale = (max - min) / 31.0;
            let sc = if d.to_f32() > 0.0 {
                (scale / d.to_f32()).round().clamp(0.0, 63.0) as u8
            } else {
                0
            };
            let m = if dmin.to_f32() > 0.0 {
                (-min / dmin.to_f32()).round().clamp(0.0, 63.0) as u8
            } else {
                0
            };
            scales[j] = sc;
            mins[j] = m;
            let actual_d = d.to_f32() * sc as f32;
            let actual_min = dmin.to_f32() * m as f32;
            for (i, value) in chunk.iter().enumerate() {
                let code = if actual_d > 0.0 {
                    ((*value + actual_min) / actual_d).round().clamp(0.0, 31.0) as u8
                } else {
                    0
                };
                q[j * 32 + i] = code;
            }
        }

        let mut packed_scales = [0u8; 12];
        for j in 0..4 {
            packed_scales[j] = scales[j] & 0x3f;
            packed_scales[j + 4] = mins[j] & 0x3f;
        }
        for j in 4..8 {
            packed_scales[j + 4] = (scales[j] & 0x0f) | ((mins[j] & 0x0f) << 4);
            packed_scales[j - 4] |= (scales[j] >> 4) << 6;
            packed_scales[j] |= (mins[j] >> 4) << 6;
        }

        let mut qh = [0u8; 32];
        let mut qs = [0u8; 128];
        for n in (0..256).step_by(64) {
            let chunk = n / 64;
            for half in 0..2 {
                for l in 0..32 {
                    let code = q[n + half * 32 + l];
                    let low = code & 0x0f;
                    let idx = chunk * 32 + half * 16 + l / 2;
                    if l % 2 == 0 {
                        qs[idx] |= low;
                    } else {
                        qs[idx] |= low << 4;
                    }
                    if code & 0x10 != 0 {
                        qh[chunk * 8 + half * 4 + l / 8] |= 1 << (l % 8);
                    }
                }
            }
        }

        out.extend_from_slice(&d.to_bits().to_le_bytes());
        out.extend_from_slice(&dmin.to_bits().to_le_bytes());
        out.extend_from_slice(&packed_scales);
        out.extend_from_slice(&qh);
        out.extend_from_slice(&qs);
    }
    out
}

fn verify_structure(
    input: &Gguf,
    output: &Gguf,
    plans: &[Plan],
    target: Target,
) -> Result<(), ConvertError> {
    if output.version != input.version
        || output.alignment != input.alignment
        || output.tensors.len() != input.tensors.len()
    {
        return Err(ConvertError::Verify(
            "output GGUF parameters do not match the input".into(),
        ));
    }
    for plan in plans {
        let t_in = &input.tensors[plan.index];
        let t_out = &output.tensors[plan.index];
        if t_in.name != t_out.name || t_in.dims != t_out.dims {
            return Err(ConvertError::Verify(format!(
                "tensor name/shape mismatch for '{}'",
                t_in.name
            )));
        }
        if t_out.ty != plan.out_ty || t_out.offset != plan.new_offset {
            return Err(ConvertError::Verify(format!(
                "tensor '{}' has the wrong type or offset",
                t_in.name
            )));
        }
        if t_out.size != Some(plan.new_size) {
            return Err(ConvertError::Verify(format!(
                "tensor '{}' has the wrong size",
                t_in.name
            )));
        }
    }
    if output.has_ptq1() {
        return Err(ConvertError::Verify(
            "output still contains PTQ1_0 tensors".into(),
        ));
    }
    if output.file_type != target.file_type() {
        return Err(ConvertError::Verify(format!(
            "output general.file_type is {:?}, expected {:?}",
            output.file_type,
            target.file_type()
        )));
    }
    if output.prism_metadata_signature() != input.prism_metadata_signature() {
        return Err(ConvertError::Verify(
            "prism.* metadata (Hadamard/MTP) was not preserved byte-for-byte".into(),
        ));
    }
    Ok(())
}

fn verify_values(
    input: &Gguf,
    output: &Gguf,
    in_path: &Path,
    out_path: &Path,
    target: Target,
    cb: &mut Option<&mut dyn FnMut(&Progress)>,
) -> Result<(), ConvertError> {
    let mut src = File::open(in_path)?;
    let mut dst = File::open(out_path)?;
    let total = output
        .tensors
        .iter()
        .filter_map(|t| t.size)
        .fold(0u64, |acc, size| acc.saturating_add(size));
    let mut done = 0u64;

    for (i, t_in) in input.tensors.iter().enumerate() {
        let t_out = &output.tensors[i];
        src.seek(SeekFrom::Start(input.data_start + t_in.offset))?;
        dst.seek(SeekFrom::Start(output.data_start + t_out.offset))?;
        let mut raw_in = vec![0u8; t_in.size.expect("validated") as usize];
        let mut raw_out = vec![0u8; t_out.size.expect("validated") as usize];
        src.read_exact(&mut raw_in)?;
        dst.read_exact(&mut raw_out)?;
        let a = decode_tensor(&raw_in, t_in)?;
        let b = decode_tensor(&raw_out, t_out)?;
        if a.len() != b.len() {
            return Err(ConvertError::Verify(format!(
                "tensor '{}' changed element count",
                t_in.name
            )));
        }

        let exact = target == Target::F32 && t_out.ty == 0;
        let max_abs = a.iter().fold(0f32, |m, v| m.max(v.abs()));
        let tolerance = if exact {
            0.0
        } else if target == Target::F16 {
            0.01 * max_abs + 1e-5
        } else if target == Target::Bf16 {
            0.08 * max_abs + 1e-4
        } else if matches!(
            target,
            Target::Tq1_0 | Target::Tq2_0 | Target::Mxfp4 | Target::Nvfp4 | Target::Q1_0
        ) {
            1.0 * max_abs + 1e-4
        } else {
            0.5 * max_abs + 1e-4
        };
        for (j, (x, y)) in a.iter().zip(&b).enumerate() {
            if !y.is_finite() || (*x - *y).abs() > tolerance {
                return Err(ConvertError::Verify(format!(
                    "value mismatch in tensor '{}' at element {j}: {x} vs {y} (tolerance {tolerance})",
                    t_in.name
                )));
            }
        }
        done += t_out.size.unwrap_or(0);
        emit(
            cb,
            Phase::Verifying,
            i,
            &t_in.name,
            input.tensors.len(),
            done,
            total,
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn emit(
    cb: &mut Option<&mut dyn FnMut(&Progress)>,
    phase: Phase,
    tensor_index: usize,
    tensor_name: &str,
    tensors_total: usize,
    bytes_done: u64,
    bytes_total: u64,
) {
    if let Some(cb) = cb.as_mut() {
        cb(&Progress {
            phase,
            tensor_index,
            tensor_name: tensor_name.to_string(),
            tensors_total,
            bytes_done,
            bytes_total,
        });
    }
}

fn paths_same(input: &Path, output: &Path) -> io::Result<bool> {
    let input = fs::canonicalize(input)?;
    if output.exists() {
        return Ok(input == fs::canonicalize(output)?);
    }
    let out_abs = if output.is_absolute() {
        output.to_path_buf()
    } else {
        std::env::current_dir()?.join(output)
    };
    let parent = out_abs.parent().unwrap_or_else(|| Path::new("."));
    let Some(name) = out_abs.file_name() else {
        return Ok(false);
    };
    Ok(input == fs::canonicalize(parent)?.join(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_roundtrip_is_close() {
        let values: Vec<f32> = (0..128).map(|i| (i as f32 * 0.07).sin()).collect();
        let raw = encode_q8_0(&values);
        let decoded = dequant::dequantize(&raw, GgmlType::Q8_0, &[128]).unwrap();
        for (a, b) in values.iter().zip(decoded) {
            assert!((*a - b).abs() < 0.01, "{a} vs {b}");
        }
    }

    #[test]
    fn ternary_decode_matches_blocks() {
        let mut raw = Vec::new();
        raw.extend_from_slice(&0x3800u16.to_le_bytes());
        raw.extend(std::iter::repeat_n(0xE4u8, 32));
        let values = decode_ternary(&raw, 128, 34);
        assert_eq!(values.len(), 128);
        assert_eq!(values[0], -0.5);
        assert_eq!(values[1], 0.0);
        assert_eq!(values[2], 0.5);
    }

    #[test]
    fn q5k_roundtrip_is_close() {
        let values: Vec<f32> = (0..512)
            .map(|i| ((i as f32 * 0.017).sin() * 2.5) + (i as f32 * 0.001))
            .collect();
        let raw = encode_q5k(&values);
        assert_eq!(raw.len(), 352);
        let back = dequant::dequantize(&raw, GgmlType::Q5K, &[512]).unwrap();
        let max = values.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = values
            .iter()
            .zip(&back)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(worst < 0.1 * max + 1e-4, "Q5_K error {worst} is too high");
    }

    #[test]
    fn niche_codecs_roundtrip_with_expected_loss() {
        let values: Vec<f32> = (0..512)
            .map(|i| ((i as f32 * 0.017).sin() * 2.5) + (i as f32 * 0.001))
            .collect();
        for ty in [
            GgmlType::Tq1_0,
            GgmlType::Tq2_0,
            GgmlType::Mxfp4,
            GgmlType::Nvfp4,
            GgmlType::Q1_0,
        ] {
            let raw = quantize::quantize(&values, ty).unwrap();
            let back = dequant::dequantize(&raw, ty, &[512]).unwrap();
            let max = values.iter().fold(0f32, |m, v| m.max(v.abs()));
            let worst = values
                .iter()
                .zip(&back)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let expected = match ty {
                GgmlType::Tq1_0 | GgmlType::Tq2_0 => 0.75,
                GgmlType::Q1_0 => 0.95,
                _ => 0.35,
            };
            assert!(
                worst <= max * expected + 1e-4,
                "{ty:?} error {worst} exceeds {expected} x {max}"
            );
        }
    }
}
