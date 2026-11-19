//! Mixed quantization recipes and per-tensor overrides.
//!
//! The rules mirror the K-quant presets in llama.cpp's `llama-quant.cpp`:
//! `use_more_bits` for `attn_v` and `ffn_down`, the Q3_K_M early-layer bumps,
//! output/tied embeddings at Q6_K, and explicit per-tensor overrides that win
//! over the preset. The engine is deliberately name-based, like llama.cpp;
//! it does not guess an architecture from free-form metadata.

use std::fs;
use std::path::{Path, PathBuf};

use regex::Regex;
use thiserror::Error;

use crate::gguf::TensorInfo;
use crate::transcode::Target;
use crate::types::type_info;

/// Broad tensor categories, matching the categories used by llama.cpp for
/// mixed quantization decisions. Specific architecture names stay in the
/// `Other` bucket unless they match a known category substring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorCategory {
    TokenEmbedding,
    AttentionV,
    AttentionQkv,
    AttentionKvB,
    AttentionK,
    AttentionQ,
    AttentionOutput,
    FfnDown,
    FfnGate,
    FfnUp,
    Output,
    Other,
}

impl TensorCategory {
    pub fn is_attention_v_like(self) -> bool {
        matches!(
            self,
            Self::AttentionV | Self::AttentionQkv | Self::AttentionKvB
        )
    }
}

/// Per-tensor facts needed to choose a mixed type. Indexes are in file order
/// and are only meaningful for their category.
#[derive(Debug, Clone, Copy)]
pub struct TensorMeta {
    pub category: TensorCategory,
    pub layer: usize,
    pub attention_index: usize,
    pub ffn_down_index: usize,
}

/// Model-level facts used by the presets.
#[derive(Debug, Clone, Copy, Default)]
pub struct RecipeContext {
    pub n_layer: usize,
    pub n_attention_wv: usize,
    pub n_ffn_down: usize,
    pub n_gqa: usize,
    pub has_tied_embeddings: bool,
}

#[derive(Debug, Error)]
pub enum RecipeError {
    #[error("invalid tensor type override '{spec}': expected REGEX=TYPE")]
    MalformedOverride { spec: String },
    #[error("unknown tensor type '{ty}' in override '{spec}'")]
    UnknownType { spec: String, ty: String },
    #[error("invalid regex '{pattern}' in override '{spec}': {source}")]
    InvalidRegex {
        spec: String,
        pattern: String,
        #[source]
        source: regex::Error,
    },
    #[error("cannot read tensor type file '{path}': {source}")]
    OverrideFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("tensor type file '{path}', line {line}: {message}")]
    OverrideFileLine {
        path: PathBuf,
        line: usize,
        message: String,
    },
}

/// A llama.cpp-compatible `--tensor-type` rule. The regex is compiled once
/// and applied to the exact GGUF tensor name.
#[derive(Debug, Clone)]
pub struct TensorOverride {
    pub pattern: String,
    pub regex: Regex,
    pub target: Target,
}

impl TensorOverride {
    pub fn parse(spec: &str) -> Result<Self, RecipeError> {
        let spec = spec.trim();
        let (pattern, type_name) =
            spec.split_once('=')
                .ok_or_else(|| RecipeError::MalformedOverride {
                    spec: spec.to_string(),
                })?;
        let pattern = pattern.trim();
        let type_name = type_name.trim();
        if pattern.is_empty() || type_name.is_empty() {
            return Err(RecipeError::MalformedOverride {
                spec: spec.to_string(),
            });
        }
        let target = parse_tensor_type(type_name).ok_or_else(|| RecipeError::UnknownType {
            spec: spec.to_string(),
            ty: type_name.to_string(),
        })?;
        let regex = Regex::new(pattern).map_err(|source| RecipeError::InvalidRegex {
            spec: spec.to_string(),
            pattern: pattern.to_string(),
            source,
        })?;
        Ok(Self {
            pattern: pattern.to_string(),
            regex,
            target,
        })
    }

    /// Reads whitespace-separated `pattern=type` entries. `#` starts a
    /// comment, so recipe files can be documented without becoming invalid.
    pub fn load_file(path: &Path) -> Result<Vec<Self>, RecipeError> {
        let text = fs::read_to_string(path).map_err(|source| RecipeError::OverrideFile {
            path: path.to_path_buf(),
            source,
        })?;
        let mut overrides = Vec::new();
        for (line_index, line) in text.lines().enumerate() {
            let line_number = line_index + 1;
            for token in line.split_whitespace() {
                if token.starts_with('#') {
                    break;
                }
                let parsed = Self::parse(token).map_err(|error| RecipeError::OverrideFileLine {
                    path: path.to_path_buf(),
                    line: line_number,
                    message: error.to_string(),
                })?;
                overrides.push(parsed);
            }
        }
        Ok(overrides)
    }

    pub fn matches(&self, tensor_name: &str) -> bool {
        self.regex.is_match(tensor_name)
    }
}

/// Parses the ggml type names accepted by `--tensor-type`. Mixed presets are
/// intentionally not valid here: llama.cpp expects a concrete tensor type.
pub fn parse_tensor_type(name: &str) -> Option<Target> {
    let normalized = name.trim().to_ascii_lowercase();
    Some(match normalized.as_str() {
        "f32" => Target::F32,
        "f16" => Target::F16,
        "bf16" => Target::Bf16,
        "q8_0" => Target::Q8_0,
        "q5_0" => Target::Q5_0,
        "q5_1" => Target::Q5_1,
        "q4_0" => Target::Q4_0,
        "q4_1" => Target::Q4_1,
        "q2_k" => Target::Q2K,
        "q3_k" | "q3_k_s" => Target::Q3K,
        "q4_k" | "q4_k_s" => Target::Q4K,
        "q5_k" | "q5_k_s" => Target::Q5K,
        "q6_k" => Target::Q6K,
        "q1_0" => Target::Q1_0,
        "tq1_0" => Target::Tq1_0,
        "tq2_0" => Target::Tq2_0,
        "mxfp4" => Target::Mxfp4,
        "nvfp4" => Target::Nvfp4,
        _ => return None,
    })
}

/// Computes the context and per-tensor metadata in one pass, preserving the
/// order of the GGUF tensor table.
pub fn analyze_tensors(tensors: &[TensorInfo]) -> (RecipeContext, Vec<TensorMeta>) {
    let mut n_layer = 0usize;
    let mut attention_count = 0usize;
    let mut ffn_down_count = 0usize;
    let mut has_output = false;
    let mut q_rows = None;
    let mut v_rows = None;
    let mut metas = Vec::with_capacity(tensors.len());

    for tensor in tensors {
        let category = tensor_category(&tensor.name);
        let layer = layer_index(&tensor.name).unwrap_or(0);
        n_layer = n_layer.max(layer.saturating_add(1));
        if category == TensorCategory::Output {
            has_output = true;
        }
        if category == TensorCategory::AttentionQ && q_rows.is_none() {
            q_rows = tensor.dims.get(1).copied();
        }
        if category == TensorCategory::AttentionV && v_rows.is_none() {
            v_rows = tensor.dims.get(1).copied();
        }

        let attention_index = if category.is_attention_v_like() {
            let index = attention_count;
            attention_count += 1;
            index
        } else {
            0
        };
        let ffn_down_index = if category == TensorCategory::FfnDown {
            let index = ffn_down_count;
            ffn_down_count += 1;
            index
        } else {
            0
        };

        metas.push(TensorMeta {
            category,
            layer,
            attention_index,
            ffn_down_index,
        });
    }

    let n_gqa = match (q_rows, v_rows) {
        (Some(q), Some(v)) if q > 0 && v > 0 && q % v == 0 => (q / v) as usize,
        _ => 1,
    };

    (
        RecipeContext {
            n_layer: n_layer.max(1),
            n_attention_wv: attention_count.max(1),
            n_ffn_down: ffn_down_count.max(1),
            n_gqa: n_gqa.max(1),
            has_tied_embeddings: !has_output,
        },
        metas,
    )
}

/// Chooses the output type for one tensor. `None` means “keep the source
/// type”: norms, routers and other non-quantizable tensors are never forced
/// to F32 just because the target is quantized.
pub fn select_type(
    target: Target,
    tensor: &TensorInfo,
    meta: &TensorMeta,
    context: &RecipeContext,
    overrides: &[TensorOverride],
) -> Option<Target> {
    if let Some(overridden) = overrides.iter().find(|rule| rule.matches(&tensor.name)) {
        return Some(fallback(overridden.target, tensor));
    }

    if !target.is_quantized() {
        return Some(target);
    }

    if !tensor_can_be_quantized(tensor) {
        return None;
    }

    let chosen = match target {
        Target::Q2K => q2_k_type(meta, context),
        Target::Q3KM => q3_k_m_type(meta, context),
        Target::Q4KM => q4_k_m_type(meta, context),
        Target::Q5KM => q5_k_m_type(meta, context),
        Target::Q3K | Target::Q4K | Target::Q5K => {
            // llama.cpp keeps the uniform `_S` presets uniform: only the output
            // tensor (and a tied token embedding) moves to Q6_K, and Q4_K_S
            // promotes the first four attention-value tensors to Q5_K.
            if is_output_like(meta, context) {
                Target::Q6K
            } else if target == Target::Q4K
                && ((meta.category.is_attention_v_like() && meta.attention_index < 4)
                    || (meta.category == TensorCategory::FfnDown
                        && meta.layer < context.n_ffn_down / 8))
            {
                Target::Q5K
            } else {
                target
            }
        }
        _ => target,
    };
    Some(fallback(chosen, tensor))
}

/// llama.cpp's `use_more_bits`: first and last eighth, plus every third layer
/// starting from the second position after the first eighth.
pub fn use_more_bits(index: usize, total: usize) -> bool {
    if total == 0 {
        return false;
    }
    let eighth = total / 8;
    // Integer division matters: llama.cpp uses `7 * n_layers / 8`, which is
    // earlier than `total - eighth` when the layer count is not a multiple of 8.
    if index < eighth || index >= (7 * total) / 8 {
        return true;
    }
    (index - eighth) % 3 == 2
}

fn q2_k_type(meta: &TensorMeta, context: &RecipeContext) -> Target {
    if is_output_like(meta, context) {
        Target::Q6K
    } else if meta.category.is_attention_v_like() {
        if context.n_gqa >= 4 {
            Target::Q4K
        } else {
            Target::Q3K
        }
    } else if matches!(
        meta.category,
        TensorCategory::FfnDown | TensorCategory::AttentionOutput
    ) {
        Target::Q3K
    } else {
        Target::Q2K
    }
}

fn q3_k_m_type(meta: &TensorMeta, context: &RecipeContext) -> Target {
    if is_output_like(meta, context) {
        return Target::Q6K;
    }
    if meta.category.is_attention_v_like() {
        return if meta.attention_index < 2 {
            Target::Q5K
        } else {
            Target::Q4K
        };
    }
    if meta.category == TensorCategory::FfnDown {
        // llama.cpp counts ffn_down tensors, not layers: for MoE models the two
        // numbers differ by the expert count.
        return if meta.layer < context.n_ffn_down / 16 {
            Target::Q5K
        } else {
            Target::Q4K
        };
    }
    if meta.category == TensorCategory::AttentionOutput {
        return Target::Q4K;
    }
    Target::Q3K
}

fn q4_k_m_type(meta: &TensorMeta, context: &RecipeContext) -> Target {
    if is_output_like(meta, context) {
        return Target::Q6K;
    }
    if meta.category.is_attention_v_like()
        && use_more_bits(meta.attention_index, context.n_attention_wv)
    {
        return Target::Q6K;
    }
    if meta.category == TensorCategory::FfnDown && use_more_bits(meta.layer, context.n_ffn_down) {
        return Target::Q6K;
    }
    Target::Q4K
}

fn q5_k_m_type(meta: &TensorMeta, context: &RecipeContext) -> Target {
    if is_output_like(meta, context) {
        return Target::Q6K;
    }
    if meta.category.is_attention_v_like()
        && use_more_bits(meta.attention_index, context.n_attention_wv)
    {
        return Target::Q6K;
    }
    if meta.category == TensorCategory::FfnDown && use_more_bits(meta.layer, context.n_ffn_down) {
        return Target::Q6K;
    }
    Target::Q5K
}

fn is_output_like(meta: &TensorMeta, context: &RecipeContext) -> bool {
    meta.category == TensorCategory::Output
        || (context.has_tied_embeddings && meta.category == TensorCategory::TokenEmbedding)
}

fn fallback(chosen: Target, tensor: &TensorInfo) -> Target {
    if fits(chosen, tensor) {
        return chosen;
    }
    let fallback = match chosen {
        Target::Q2K | Target::Q3K | Target::Q3KM | Target::Tq1_0 | Target::Tq2_0 => Target::Q4_0,
        Target::Q4K | Target::Q4KM => Target::Q5_0,
        Target::Q5K | Target::Q5KM => Target::Q5_1,
        Target::Q6K => Target::Q8_0,
        _ => Target::F16,
    };
    if fits(fallback, tensor) {
        fallback
    } else {
        Target::F16
    }
}

fn fits(target: Target, tensor: &TensorInfo) -> bool {
    let Some(info) = target.type_id().and_then(type_info) else {
        return false;
    };
    !tensor.dims.is_empty()
        && tensor.dims[0].is_multiple_of(info.block)
        && tensor.element_count().is_multiple_of(info.block)
}

pub fn tensor_can_be_quantized(tensor: &TensorInfo) -> bool {
    let name = tensor.name.as_str();
    if tensor.dims.len() < 2 {
        return false;
    }
    if name.contains("_norm.weight") || name.contains(".norm.weight") {
        return false;
    }
    // Routers and tiny elementwise/gating tables stay in their source type.
    const NEVER_QUANTIZE: [&str; 12] = [
        "ffn_gate_inp.weight",
        "ffn_gate_tid2eid.weight",
        "engram_gate.weight",
        "attn_rel_b.weight",
        "a.rvq.codebook",
        "mm.a.code_embd",
        "sam.pos_embd",
        ".position_embd",
        ".patch_embd",
        ".patch_merger",
        "attn_q_a",
        "attn_q_b",
    ];
    if NEVER_QUANTIZE.iter().any(|needle| name.contains(needle)) {
        return false;
    }
    // SSM, hybrid attention and similar non-matmul state tensors are safer
    // untouched; this is intentionally conservative.
    if name.contains("ssm_") || name.contains("time_mix_") || name.contains("altup") {
        return false;
    }
    if name.contains(".rel_pos") || name.contains(".position_embd") || name.contains(".patch_embd")
    {
        return false;
    }
    true
}

/// The effective per-tensor type map, in GGUF tensor order. This is what
/// `--dump-tensor-types` writes and what `--tensor-type-file` can feed back.
pub fn effective_tensor_types(
    tensors: &[TensorInfo],
    target: Target,
    overrides: &[TensorOverride],
) -> Vec<(String, u32)> {
    let (context, metas) = analyze_tensors(tensors);
    tensors
        .iter()
        .zip(metas.iter())
        .map(|(tensor, meta)| {
            let ty = select_type(target, tensor, meta, &context, overrides)
                .and_then(Target::type_id)
                .unwrap_or(tensor.ty);
            (tensor.name.clone(), ty)
        })
        .collect()
}

/// A regex that matches exactly one tensor name, for a dump file.
pub fn exact_name_pattern(name: &str) -> String {
    format!("^{}$", regex::escape(name))
}

pub fn tensor_category(name: &str) -> TensorCategory {
    if name == "output.weight" {
        return TensorCategory::Output;
    }
    if name == "token_embd.weight" || name == "per_layer_token_embd.weight" {
        return TensorCategory::TokenEmbedding;
    }
    if name.contains("attn_qkv.weight") || name.contains("attn_qkv_gate.weight") {
        return TensorCategory::AttentionQkv;
    }
    if name.contains("attn_kv_b.weight") {
        return TensorCategory::AttentionKvB;
    }
    if name.contains("attn_v.weight") {
        return TensorCategory::AttentionV;
    }
    if name.contains("attn_k.weight") {
        return TensorCategory::AttentionK;
    }
    if name.contains("attn_q.weight") {
        return TensorCategory::AttentionQ;
    }
    if name.contains("attn_output.weight") {
        return TensorCategory::AttentionOutput;
    }
    if name.contains("ffn_down") {
        return TensorCategory::FfnDown;
    }
    if name.contains("ffn_gate") {
        return TensorCategory::FfnGate;
    }
    if name.contains("ffn_up") {
        return TensorCategory::FfnUp;
    }
    TensorCategory::Other
}

fn layer_index(name: &str) -> Option<usize> {
    let start = name.find("blk.")? + 4;
    let digits: String = name[start..]
        .chars()
        .take_while(|character| character.is_ascii_digit())
        .collect();
    if digits.is_empty() {
        None
    } else {
        digits.parse().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(name: &str, rows: u64) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            dims: vec![256, rows],
            ty: 1,
            offset: 0,
            type_pos: 0,
            offset_pos: 0,
            size: Some(512),
        }
    }

    fn dense_model() -> (RecipeContext, Vec<TensorMeta>) {
        let mut tensors = Vec::new();
        for layer in 0..32 {
            tensors.push(tensor(&format!("blk.{layer}.attn_q.weight"), 8));
            tensors.push(tensor(&format!("blk.{layer}.attn_v.weight"), 1));
            tensors.push(tensor(&format!("blk.{layer}.attn_output.weight"), 8));
            tensors.push(tensor(&format!("blk.{layer}.ffn_down.weight"), 8));
        }
        tensors.push(tensor("token_embd.weight", 8));
        tensors.push(tensor("output.weight", 8));
        analyze_tensors(&tensors)
    }

    fn selected(target: Target, name: &str) -> Option<Target> {
        let (context, _) = dense_model();
        let mut tensors = Vec::new();
        for layer in 0..32 {
            tensors.push(tensor(&format!("blk.{layer}.attn_q.weight"), 8));
            tensors.push(tensor(&format!("blk.{layer}.attn_v.weight"), 1));
            tensors.push(tensor(&format!("blk.{layer}.attn_output.weight"), 8));
            tensors.push(tensor(&format!("blk.{layer}.ffn_down.weight"), 8));
        }
        tensors.push(tensor("token_embd.weight", 8));
        tensors.push(tensor("output.weight", 8));
        let (_, metas) = analyze_tensors(&tensors);
        let index = tensors.iter().position(|t| t.name == name)?;
        select_type(target, &tensors[index], &metas[index], &context, &[])
    }

    #[test]
    fn q4_k_m_matches_llama_layer_rules() {
        assert_eq!(
            selected(Target::Q4KM, "blk.0.attn_v.weight"),
            Some(Target::Q6K)
        );
        assert_eq!(
            selected(Target::Q4KM, "blk.6.attn_v.weight"),
            Some(Target::Q6K)
        );
        assert_eq!(
            selected(Target::Q4KM, "blk.5.attn_v.weight"),
            Some(Target::Q4K)
        );
        assert_eq!(selected(Target::Q4KM, "output.weight"), Some(Target::Q6K));
        assert_eq!(
            selected(Target::Q4KM, "token_embd.weight"),
            Some(Target::Q4K)
        );
    }

    #[test]
    fn q3_k_m_uses_early_layer_bumps() {
        assert_eq!(
            selected(Target::Q3KM, "blk.0.attn_v.weight"),
            Some(Target::Q5K)
        );
        assert_eq!(
            selected(Target::Q3KM, "blk.2.attn_v.weight"),
            Some(Target::Q4K)
        );
        assert_eq!(
            selected(Target::Q3KM, "blk.0.ffn_down.weight"),
            Some(Target::Q5K)
        );
        assert_eq!(
            selected(Target::Q3KM, "blk.2.ffn_down.weight"),
            Some(Target::Q4K)
        );
    }

    #[test]
    fn q5_k_m_and_q6_k_follow_the_reference_rules() {
        assert_eq!(
            selected(Target::Q5KM, "blk.0.attn_v.weight"),
            Some(Target::Q6K)
        );
        assert_eq!(
            selected(Target::Q5KM, "blk.5.attn_v.weight"),
            Some(Target::Q5K)
        );
        assert_eq!(
            selected(Target::Q5KM, "blk.4.ffn_down.weight"),
            Some(Target::Q5K)
        );
        assert_eq!(
            selected(Target::Q5KM, "blk.6.ffn_down.weight"),
            Some(Target::Q6K)
        );
        assert_eq!(selected(Target::Q5KM, "output.weight"), Some(Target::Q6K));

        assert_eq!(
            selected(Target::Q6K, "blk.0.attn_v.weight"),
            Some(Target::Q6K)
        );
        assert_eq!(
            selected(Target::Q6K, "blk.0.ffn_down.weight"),
            Some(Target::Q6K)
        );
        assert_eq!(selected(Target::Q6K, "output.weight"), Some(Target::Q6K));
        assert_eq!(selected(Target::Q6K, "blk.0.attn_norm.weight"), None);
    }

    #[test]
    fn override_regex_can_match_a_tensor_family() {
        let (context, _) = dense_model();
        let mut tensors = Vec::new();
        for layer in 0..32 {
            tensors.push(tensor(&format!("blk.{layer}.ffn_down.weight"), 8));
        }
        tensors.push(tensor("output.weight", 8));
        let (_, metas) = analyze_tensors(&tensors);
        let rule = TensorOverride::parse(r"^blk\.\d+\.ffn_down\.weight$=q4_k").unwrap();
        assert_eq!(
            select_type(
                Target::Q4KM,
                &tensors[0],
                &metas[0],
                &context,
                std::slice::from_ref(&rule)
            ),
            Some(Target::Q4K)
        );
    }

    #[test]
    fn override_can_mix_a_quantized_tensor_into_a_float_target() {
        let (context, metas) = dense_model();
        let tensor = tensor("blk.0.attn_v.weight", 1);
        let rule = TensorOverride::parse(r"^blk\.0\.attn_v\.weight$=q4_k").unwrap();
        assert_eq!(
            select_type(Target::F16, &tensor, &metas[1], &context, &[rule]),
            Some(Target::Q4K)
        );
    }

    #[test]
    fn q2_k_uses_gqa_for_attention_v() {
        assert_eq!(
            selected(Target::Q2K, "blk.0.attn_v.weight"),
            Some(Target::Q4K)
        );
        assert_eq!(
            selected(Target::Q2K, "blk.0.ffn_down.weight"),
            Some(Target::Q3K)
        );
        assert_eq!(selected(Target::Q2K, "output.weight"), Some(Target::Q6K));
    }

    #[test]
    fn overrides_win_even_when_they_equal_the_global_type() {
        let (context, metas) = dense_model();
        let tensor = tensor("output.weight", 8);
        let meta = metas.last().copied().unwrap();
        let rule = TensorOverride::parse(r"^output\.weight$=q4_k").unwrap();
        assert_eq!(
            select_type(Target::Q4KM, &tensor, &meta, &context, &[rule]),
            Some(Target::Q4K)
        );
    }

    #[test]
    fn moe_router_and_ssm_tensors_stay_unquantized() {
        let (context, _) = dense_model();
        for name in ["blk.0.ffn_gate_inp.weight", "blk.0.ssm_conv1d.weight"] {
            let value = tensor(name, 8);
            let meta = TensorMeta {
                category: tensor_category(name),
                layer: 0,
                attention_index: 0,
                ffn_down_index: 0,
            };
            assert_eq!(
                select_type(Target::Q4KM, &value, &meta, &context, &[]),
                None
            );
        }
    }

    #[test]
    fn tied_embeddings_use_the_output_type() {
        let tensors = vec![tensor("token_embd.weight", 8)];
        let (context, metas) = analyze_tensors(&tensors);
        assert!(context.has_tied_embeddings);
        assert_eq!(
            select_type(Target::Q4KM, &tensors[0], &metas[0], &context, &[]),
            Some(Target::Q6K)
        );
    }

    #[test]
    fn incompatible_shapes_fall_back() {
        let (context, metas) = dense_model();
        let value = TensorInfo {
            dims: vec![100, 2],
            ..tensor("blk.0.attn_v.weight", 2)
        };
        let meta = metas[1];
        assert_eq!(
            select_type(Target::Q4KM, &value, &meta, &context, &[]),
            Some(Target::F16)
        );
    }

    #[test]
    fn override_file_accepts_comments_and_multiple_entries() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("types.txt");
        fs::write(
            &path,
            "# comment\nattn_v=q5_k  ffn_down=q4_k\n# another\noutput.weight=q6_k\n",
        )
        .unwrap();
        let rules = TensorOverride::load_file(&path).unwrap();
        assert_eq!(rules.len(), 3);
        assert_eq!(rules[1].target, Target::Q4K);
    }

    #[test]
    fn use_more_bits_matches_the_reference_formula() {
        assert!(use_more_bits(0, 32));
        assert!(use_more_bits(3, 32));
        assert!(!use_more_bits(4, 32));
        assert!(use_more_bits(6, 32));
        assert!(use_more_bits(28, 32));
    }
}
