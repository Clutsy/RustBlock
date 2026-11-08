//! Native conversion for LLaMA/Qwen-style safetensors checkpoints.
//!
//! The converter handles dense LLaMA/Mistral/Qwen2/Qwen3 layouts, Mixtral and
//! Qwen2/Qwen3 MoE layouts (per-expert and fused expert tensors), SentencePiece
//! `tokenizer.model` vocabularies, and direct FP8/AWQ quantized checkpoints.
//! Plain weights are streamed from the source shards; FP8 and AWQ matrices are
//! dequantized one tensor at a time, so peak resident memory stays bounded by
//! the largest single tensor rather than by the whole checkpoint.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use half::{bf16, f16};
use pmetal_gguf::{GgmlType, MetadataValue};
use serde_json::Value;
use tempfile::NamedTempFile;

use crate::gguf::{align_up, Gguf};
use crate::safetensors_native::{
    awq_dequant, decode_floats, decode_u32, element_count, Entry, NativeError, Reader,
};

const ALIGNMENT: u64 = 32;

#[derive(Debug, thiserror::Error)]
pub enum ToGgufError {
    #[error("I/O error: {0}")]
    Io(String),
    #[error("invalid config.json: {0}")]
    Config(String),
    #[error("invalid tokenizer: {0}")]
    Tokenizer(String),
    #[error("unsupported checkpoint: {0}")]
    Unsupported(String),
    #[error("tensor '{name}' has shape {actual:?}, expected {expected:?}")]
    Shape {
        name: String,
        actual: Vec<u64>,
        expected: Vec<u64>,
    },
    #[error("safetensors: {0}")]
    Safetensors(#[from] NativeError),
    #[error("output already exists: {0}")]
    OutputExists(PathBuf),
    #[error("written GGUF is invalid: {0}")]
    InvalidOutput(String),
}

#[derive(Debug, Clone)]
pub struct ToGgufReport {
    pub input: String,
    pub output: String,
    pub architecture: String,
    pub layers: u64,
    pub tensors: usize,
    pub tokenizer_tokens: usize,
    pub file_type: u32,
}

struct ModelConfig {
    architecture: String,
    name: String,
    hidden: u64,
    layers: u64,
    heads: u64,
    kv_heads: u64,
    head_dim: u64,
    intermediate: u64,
    context: u64,
    rms_eps: f64,
    rope_theta: f64,
    rope_scaling: Option<RopeScaling>,
    vocab: Option<u64>,
    tie_embeddings: bool,
    experts: u64,
    experts_used: u64,
    expert_intermediate: u64,
    shared_expert_intermediate: Option<u64>,
    attention_bias: bool,
}

struct RopeScaling {
    kind: String,
    factor: f64,
    original_context: Option<u64>,
}

struct TokenizerData {
    model: String,
    tokens: Vec<String>,
    token_types: Vec<i32>,
    scores: Vec<f32>,
    merges: Vec<String>,
    chat_template: Option<String>,
    bos: Option<u32>,
    eos: Option<u32>,
    unk: Option<u32>,
    pad: Option<u32>,
    add_bos: bool,
    add_eos: bool,
}

enum PlanData {
    Copy {
        shard: usize,
        source: String,
    },
    Cast {
        shard: usize,
        source: String,
    },
    CastF32 {
        shard: usize,
        source: String,
    },
    StackedExperts {
        parts: Vec<(usize, String)>,
    },
    FusedGateUp {
        shard: usize,
        source: String,
        experts: u64,
        intermediate: u64,
        hidden: u64,
        half: usize,
    },
    Fp8 {
        shard: usize,
        source: String,
        scale_shard: usize,
        scale: String,
        rows: u64,
        columns: u64,
    },
    Awq {
        qweight: (usize, String),
        qzeros: Option<(usize, String)>,
        scales: (usize, String),
        qweight_shape: Vec<u64>,
        qzeros_shape: Vec<u64>,
        scales_shape: Vec<u64>,
    },
}

struct TensorPlan {
    output: String,
    /// Dimensions in Hugging Face order; the GGUF header reverses these.
    shape: Vec<u64>,
    dtype: GgmlType,
    bytes: u64,
    data: PlanData,
    offset: u64,
}

type SourceIndex = BTreeMap<String, (usize, Entry)>;
type SentencePieceVocab = (Vec<String>, Vec<i32>, Vec<f32>);

fn read_json(path: &Path) -> Result<Value, ToGgufError> {
    let text = fs::read_to_string(path)
        .map_err(|error| ToGgufError::Io(format!("{}: {error}", path.display())))?;
    serde_json::from_str(&text)
        .map_err(|error| ToGgufError::Config(format!("{}: {error}", path.display())))
}

fn number(value: &Value, key: &str) -> Result<u64, ToGgufError> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| ToGgufError::Config(format!("missing positive integer '{key}'")))
}

fn optional_number(value: &Value, key: &str) -> Option<u64> {
    value.get(key).and_then(Value::as_u64)
}

fn optional_float(value: &Value, key: &str) -> Option<f64> {
    value.get(key).and_then(Value::as_f64)
}

fn read_config(dir: &Path) -> Result<ModelConfig, ToGgufError> {
    let value = read_json(&dir.join("config.json"))?;
    let model_type = value
        .get("model_type")
        .and_then(Value::as_str)
        .ok_or_else(|| ToGgufError::Config("missing model_type".into()))?;
    let architecture = match model_type {
        "llama" | "mistral" | "mixtral" => "llama",
        "qwen2" => "qwen2",
        "qwen3" => "qwen3",
        "qwen2_moe" | "qwen2moe" => "qwen2moe",
        "qwen3_moe" | "qwen3moe" => "qwen3moe",
        other => {
            return Err(ToGgufError::Unsupported(format!(
                "model_type '{other}' is not mapped; supported: llama, mistral, mixtral, qwen2, qwen3, qwen2_moe, qwen3_moe"
            )))
        }
    }
    .to_string();
    let hidden = number(&value, "hidden_size")?;
    let layers = number(&value, "num_hidden_layers")?;
    let heads = number(&value, "num_attention_heads")?;
    let kv_heads = optional_number(&value, "num_key_value_heads").unwrap_or(heads);
    let head_dim = optional_number(&value, "head_dim").unwrap_or_else(|| hidden / heads);
    if hidden == 0 || layers == 0 || heads == 0 || kv_heads == 0 || head_dim == 0 {
        return Err(ToGgufError::Config(
            "model dimensions must all be greater than zero".into(),
        ));
    }
    let intermediate = number(&value, "intermediate_size")?;
    let experts = optional_number(&value, "num_local_experts")
        .or_else(|| optional_number(&value, "num_experts"))
        .unwrap_or(0);
    let experts_used = optional_number(&value, "num_experts_per_tok").unwrap_or(0);
    if experts > 0 && (experts_used == 0 || experts_used > experts) {
        return Err(ToGgufError::Config(format!(
            "invalid MoE routing: {experts_used} of {experts} experts selected"
        )));
    }
    let expert_intermediate = optional_number(&value, "moe_intermediate_size")
        .or_else(|| optional_number(&value, "expert_intermediate_size"))
        .unwrap_or(intermediate);
    let shared_expert_intermediate = optional_number(&value, "shared_expert_intermediate_size")
        .or_else(|| optional_number(&value, "expert_shared_feed_forward_length"));
    let name = value
        .get("_name_or_path")
        .and_then(Value::as_str)
        .filter(|name| !name.trim().is_empty())
        .map(str::to_string)
        .or_else(|| {
            dir.file_name()
                .and_then(|name| name.to_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| model_type.to_string());
    Ok(ModelConfig {
        architecture,
        name,
        hidden,
        layers,
        heads,
        kv_heads,
        head_dim,
        intermediate,
        context: optional_number(&value, "max_position_embeddings").unwrap_or(4096),
        rms_eps: optional_float(&value, "rms_norm_eps").unwrap_or(1e-5),
        rope_theta: optional_float(&value, "rope_theta").unwrap_or(10_000.0),
        rope_scaling: value.get("rope_scaling").and_then(|scaling| {
            let kind = scaling
                .get("type")
                .or_else(|| scaling.get("rope_type"))
                .and_then(Value::as_str)?;
            Some(RopeScaling {
                kind: kind.to_string(),
                factor: scaling.get("factor").and_then(Value::as_f64).unwrap_or(1.0),
                original_context: scaling
                    .get("original_max_position_embeddings")
                    .and_then(Value::as_u64),
            })
        }),
        vocab: optional_number(&value, "vocab_size"),
        tie_embeddings: value
            .get("tie_word_embeddings")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        experts,
        experts_used,
        expert_intermediate,
        shared_expert_intermediate,
        attention_bias: value
            .get("attention_bias")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    })
}

fn token_content(value: &Value) -> Option<String> {
    value.as_str().map(str::to_string).or_else(|| {
        value
            .get("content")
            .and_then(Value::as_str)
            .map(str::to_string)
    })
}

fn looks_special(token: &str) -> bool {
    token == "<pad>"
        || token == "<mask>"
        || token == "<2mass>"
        || token == "[@BOS@]"
        || (token.starts_with("<|") && token.ends_with("|>"))
        || (token.starts_with("<｜") && token.ends_with("｜>"))
        || (token.starts_with("<unused") && token.ends_with(">"))
}

fn token_id(tokens: &[String], value: Option<String>) -> Option<u32> {
    value.and_then(|token| {
        tokens
            .iter()
            .position(|candidate| candidate == &token)
            .map(|id| id as u32)
    })
}

fn tokenizer_config(dir: &Path) -> Result<Value, ToGgufError> {
    let path = dir.join("tokenizer_config.json");
    if path.is_file() {
        read_json(&path)
    } else {
        Ok(Value::Null)
    }
}

fn chat_template(config: &Value, tokenizer: Option<&Value>) -> Option<String> {
    config
        .get("chat_template")
        .and_then(Value::as_str)
        .or_else(|| {
            tokenizer
                .and_then(|tokenizer| tokenizer.get("chat_template"))
                .and_then(Value::as_str)
        })
        .map(str::to_string)
}

fn read_spm_varint(data: &[u8], pos: &mut usize) -> Result<u64, ToGgufError> {
    let mut value = 0u64;
    let mut shift = 0u32;
    loop {
        let byte = *data
            .get(*pos)
            .ok_or_else(|| ToGgufError::Tokenizer("truncated protobuf varint".into()))?;
        *pos += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
        if shift >= 64 {
            return Err(ToGgufError::Tokenizer("protobuf varint is too long".into()));
        }
    }
}

fn skip_spm_field(data: &[u8], pos: &mut usize, wire: u64) -> Result<(), ToGgufError> {
    match wire {
        0 => {
            read_spm_varint(data, pos)?;
        }
        1 => *pos = pos.saturating_add(8),
        2 => {
            let len = read_spm_varint(data, pos)? as usize;
            *pos = pos.saturating_add(len);
        }
        5 => *pos = pos.saturating_add(4),
        other => {
            return Err(ToGgufError::Tokenizer(format!(
                "unsupported protobuf wire type {other}"
            )))
        }
    }
    if *pos > data.len() {
        return Err(ToGgufError::Tokenizer(
            "protobuf field extends past end of file".into(),
        ));
    }
    Ok(())
}

fn read_spm_piece(data: &[u8]) -> Result<(String, f32, i32), ToGgufError> {
    let mut pos = 0usize;
    let mut piece = None;
    let mut score = 0.0f32;
    let mut kind = 1i32;
    while pos < data.len() {
        let key = read_spm_varint(data, &mut pos)?;
        let field = key >> 3;
        let wire = key & 7;
        match (field, wire) {
            (1, 2) => {
                let len = read_spm_varint(data, &mut pos)? as usize;
                let end = pos
                    .checked_add(len)
                    .filter(|end| *end <= data.len())
                    .ok_or_else(|| ToGgufError::Tokenizer("truncated SPM piece".into()))?;
                piece = Some(
                    std::str::from_utf8(&data[pos..end])
                        .map_err(|error| {
                            ToGgufError::Tokenizer(format!("SPM piece is not valid UTF-8: {error}"))
                        })?
                        .to_string(),
                );
                pos = end;
            }
            (2, 5) => {
                let bytes = data
                    .get(pos..pos + 4)
                    .ok_or_else(|| ToGgufError::Tokenizer("truncated SPM score".into()))?;
                score = f32::from_le_bytes(bytes.try_into().expect("four bytes"));
                pos += 4;
            }
            (3, 0) => kind = read_spm_varint(data, &mut pos)? as i32,
            _ => skip_spm_field(data, &mut pos, wire)?,
        }
    }
    let piece = piece.ok_or_else(|| ToGgufError::Tokenizer("SPM piece without text".into()))?;
    if !(1..=6).contains(&kind) {
        return Err(ToGgufError::Tokenizer(format!(
            "SPM piece '{piece}' has unknown type {kind}"
        )));
    }
    Ok((piece, score, kind))
}

fn read_sentencepiece(path: &Path) -> Result<SentencePieceVocab, ToGgufError> {
    let data =
        fs::read(path).map_err(|error| ToGgufError::Io(format!("{}: {error}", path.display())))?;
    let mut pos = 0usize;
    let mut tokens = Vec::new();
    let mut types = Vec::new();
    let mut scores = Vec::new();
    while pos < data.len() {
        let key = read_spm_varint(&data, &mut pos)?;
        let field = key >> 3;
        let wire = key & 7;
        if field == 1 && wire == 2 {
            let len = read_spm_varint(&data, &mut pos)? as usize;
            let end = pos
                .checked_add(len)
                .filter(|end| *end <= data.len())
                .ok_or_else(|| ToGgufError::Tokenizer("truncated SPM model".into()))?;
            let (piece, score, kind) = read_spm_piece(&data[pos..end])?;
            tokens.push(piece);
            scores.push(score);
            types.push(kind);
            pos = end;
        } else {
            skip_spm_field(&data, &mut pos, wire)?;
        }
    }
    if tokens.is_empty() {
        return Err(ToGgufError::Tokenizer(
            "tokenizer.model contains no pieces".into(),
        ));
    }
    Ok((tokens, types, scores))
}

fn pad_tokenizer(tokenizer: &mut TokenizerData, target: u64) -> Result<(), ToGgufError> {
    if tokenizer.tokens.len() as u64 > target {
        return Err(ToGgufError::Tokenizer(format!(
            "tokenizer has {} tokens, more than config vocab_size {target}",
            tokenizer.tokens.len()
        )));
    }
    for id in tokenizer.tokens.len() as u64..target {
        tokenizer.tokens.push(format!("[PAD{id}]"));
        tokenizer.token_types.push(5);
        if tokenizer.model == "llama" {
            tokenizer.scores.push(-10000.0);
        }
    }
    Ok(())
}

fn read_tokenizer(dir: &Path) -> Result<TokenizerData, ToGgufError> {
    let config = tokenizer_config(dir)?;
    let spm = dir.join("tokenizer.model");
    if spm.is_file() {
        let (tokens, token_types, scores) = read_sentencepiece(&spm)?;
        let bos = token_id(&tokens, config.get("bos_token").and_then(token_content))
            .or_else(|| token_id(&tokens, Some("<s>".into())));
        let eos = token_id(&tokens, config.get("eos_token").and_then(token_content))
            .or_else(|| token_id(&tokens, Some("</s>".into())));
        let unk = token_id(&tokens, config.get("unk_token").and_then(token_content))
            .or_else(|| token_id(&tokens, Some("<unk>".into())));
        let pad = token_id(&tokens, config.get("pad_token").and_then(token_content));
        return Ok(TokenizerData {
            model: "llama".into(),
            tokens,
            token_types,
            scores,
            merges: Vec::new(),
            chat_template: chat_template(&config, None),
            bos,
            eos,
            unk,
            pad,
            add_bos: config
                .get("add_bos_token")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            add_eos: config
                .get("add_eos_token")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        });
    }

    let tokenizer = read_json(&dir.join("tokenizer.json"))?;
    let model = tokenizer
        .get("model")
        .and_then(Value::as_object)
        .ok_or_else(|| ToGgufError::Tokenizer("missing model object".into()))?;
    if model.get("type").and_then(Value::as_str) != Some("BPE") {
        return Err(ToGgufError::Unsupported(
            "tokenizer is neither SentencePiece tokenizer.model nor BPE tokenizer.json".into(),
        ));
    }
    let vocab = model
        .get("vocab")
        .and_then(Value::as_object)
        .ok_or_else(|| ToGgufError::Tokenizer("missing BPE vocab".into()))?;
    let mut by_id: BTreeMap<u64, String> = BTreeMap::new();
    for (token, id) in vocab {
        let id = id
            .as_u64()
            .ok_or_else(|| ToGgufError::Tokenizer(format!("token '{token}' has invalid id")))?;
        if by_id.insert(id, token.clone()).is_some() {
            return Err(ToGgufError::Tokenizer(format!("duplicate token id {id}")));
        }
    }
    let mut types = vec![1i32; by_id.len()];
    let mut specials = BTreeMap::new();
    if let Some(added) = tokenizer.get("added_tokens").and_then(Value::as_array) {
        for item in added {
            let (Some(id), Some(content)) = (
                item.get("id").and_then(Value::as_u64),
                item.get("content").and_then(Value::as_str),
            ) else {
                continue;
            };
            if let Some(existing) = by_id.get(&id) {
                if existing != content {
                    return Err(ToGgufError::Tokenizer(format!(
                        "added token id {id} conflicts with '{existing}'"
                    )));
                }
            } else {
                by_id.insert(id, content.to_string());
            }
            let special = item.get("special").and_then(Value::as_bool) == Some(true);
            if special || looks_special(content) {
                specials.insert(content.to_string(), id as u32);
            }
            if id as usize >= types.len() {
                types.resize(id as usize + 1, 1);
            }
            // llama.cpp expects added tokens to be CONTROL when they look
            // like chat/FIM markers and USER_DEFINED otherwise.
            types[id as usize] = if special || looks_special(content) {
                3
            } else {
                4
            };
        }
    }
    let Some(max_id) = by_id.keys().next_back().copied() else {
        return Err(ToGgufError::Tokenizer("empty vocabulary".into()));
    };
    if max_id as usize + 1 != by_id.len() {
        return Err(ToGgufError::Tokenizer(
            "token ids are not contiguous from zero".into(),
        ));
    }
    types.resize(max_id as usize + 1, 1);
    let tokens: Vec<String> = by_id.into_values().collect();
    let merges = model
        .get("merges")
        .and_then(Value::as_array)
        .ok_or_else(|| ToGgufError::Tokenizer("missing BPE merges".into()))?
        .iter()
        .map(|merge| match merge {
            Value::String(value) => Ok(value.clone()),
            Value::Array(parts) if parts.len() == 2 => {
                let left = parts[0]
                    .as_str()
                    .ok_or_else(|| ToGgufError::Tokenizer("merge has a non-string part".into()))?;
                let right = parts[1]
                    .as_str()
                    .ok_or_else(|| ToGgufError::Tokenizer("merge has a non-string part".into()))?;
                Ok(format!("{left} {right}"))
            }
            _ => Err(ToGgufError::Tokenizer("invalid BPE merge".into())),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let bos = token_id(&tokens, config.get("bos_token").and_then(token_content))
        .or_else(|| specials.get("<s>").copied());
    let eos = token_id(&tokens, config.get("eos_token").and_then(token_content))
        .or_else(|| specials.get("</s>").copied());
    let unk = token_id(&tokens, config.get("unk_token").and_then(token_content));
    let pad = token_id(&tokens, config.get("pad_token").and_then(token_content));
    Ok(TokenizerData {
        model: "gpt2".into(),
        tokens,
        token_types: types,
        scores: Vec::new(),
        merges,
        chat_template: chat_template(&config, Some(&tokenizer)),
        bos,
        eos,
        unk,
        pad,
        add_bos: config
            .get("add_bos_token")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        add_eos: config
            .get("add_eos_token")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    })
}

fn load_shards(dir: &Path) -> Result<(Vec<Reader>, SourceIndex), ToGgufError> {
    let entries = fs::read_dir(dir)
        .map_err(|error| ToGgufError::Io(format!("{}: {error}", dir.display())))?;
    let mut paths = Vec::new();
    for entry in entries {
        let path = entry
            .map_err(|error| ToGgufError::Io(error.to_string()))?
            .path();
        if path.extension().and_then(|extension| extension.to_str()) == Some("safetensors") {
            paths.push(path);
        }
    }
    paths.sort();
    if paths.is_empty() {
        return Err(ToGgufError::Unsupported(
            "no .safetensors shards found in the input directory".into(),
        ));
    }
    let mut readers = Vec::with_capacity(paths.len());
    let mut tensors = BTreeMap::new();
    for (index, path) in paths.iter().enumerate() {
        let reader = Reader::open(path)?;
        for (name, entry) in &reader.entries {
            if tensors
                .insert(name.clone(), (index, entry.clone()))
                .is_some()
            {
                return Err(ToGgufError::Safetensors(NativeError::Header(format!(
                    "tensor '{name}' appears in more than one shard"
                ))));
            }
        }
        readers.push(reader);
    }
    Ok((readers, tensors))
}

fn map_source_name(name: &str, layers: u64) -> Result<Option<String>, ToGgufError> {
    if name == "model.rotary_emb.inv_freq"
        || name.ends_with(".self_attn.rotary_emb.inv_freq")
        || name == "model.embed_tokens.weight.1"
    {
        return Ok(None);
    }
    let direct = match name {
        "model.embed_tokens.weight" => Some("token_embd.weight"),
        "model.norm.weight" => Some("output_norm.weight"),
        "lm_head.weight" => Some("output.weight"),
        _ => None,
    };
    if let Some(output) = direct {
        return Ok(Some(output.to_string()));
    }
    let Some(rest) = name.strip_prefix("model.layers.") else {
        return Err(ToGgufError::Unsupported(format!(
            "tensor '{name}' is not part of the supported decoder layout"
        )));
    };
    let Some((layer, suffix)) = rest.split_once('.') else {
        return Err(ToGgufError::Unsupported(format!(
            "malformed tensor '{name}'"
        )));
    };
    let layer: u64 = layer
        .parse()
        .map_err(|_| ToGgufError::Unsupported(format!("invalid layer in '{name}'")))?;
    if layer >= layers {
        return Err(ToGgufError::Unsupported(format!(
            "tensor '{name}' references layer {layer}, but config has {layers}"
        )));
    }
    let suffix = match suffix {
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.q_proj.bias" => "attn_q.bias",
        "self_attn.k_proj.bias" => "attn_k.bias",
        "self_attn.v_proj.bias" => "attn_v.bias",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        other => {
            return Err(ToGgufError::Unsupported(format!(
                "tensor '{name}' has unsupported suffix '{other}'"
            )))
        }
    };
    Ok(Some(format!("blk.{layer}.{suffix}")))
}

fn output_for_weight(name: &str, layers: u64) -> Result<String, ToGgufError> {
    if name == "lm_head.weight" {
        return Ok("output.weight".into());
    }
    map_source_name(name, layers)?
        .ok_or_else(|| ToGgufError::Unsupported(format!("auxiliary tensor '{name}'")))
}

fn expected_shape(output: &str, cfg: &ModelConfig, vocab: u64) -> Option<Vec<u64>> {
    let attn = cfg.heads * cfg.head_dim;
    let kv = cfg.kv_heads * cfg.head_dim;
    match output {
        "token_embd.weight" | "output.weight" => Some(vec![vocab, cfg.hidden]),
        "output_norm.weight" => Some(vec![cfg.hidden]),
        _ => {
            let rest = output.strip_prefix("blk.")?;
            let (_, suffix) = rest.split_once('.')?;
            Some(match suffix {
                "attn_norm.weight" | "ffn_norm.weight" => vec![cfg.hidden],
                "attn_q.weight" => vec![attn, cfg.hidden],
                "attn_k.weight" | "attn_v.weight" => vec![kv, cfg.hidden],
                "attn_output.weight" => vec![cfg.hidden, attn],
                "attn_q.bias" => vec![attn],
                "attn_k.bias" | "attn_v.bias" => vec![kv],
                "attn_q_norm.weight" | "attn_k_norm.weight" => vec![cfg.head_dim],
                "ffn_gate.weight" | "ffn_up.weight" => vec![cfg.intermediate, cfg.hidden],
                "ffn_down.weight" => vec![cfg.hidden, cfg.intermediate],
                "ffn_gate_inp.weight" => vec![cfg.experts, cfg.hidden],
                "ffn_gate_inp_shexp.weight" => vec![cfg.hidden],
                "ffn_gate_exps.weight" | "ffn_up_exps.weight" => {
                    vec![cfg.experts, cfg.expert_intermediate, cfg.hidden]
                }
                "ffn_down_exps.weight" => {
                    vec![cfg.experts, cfg.hidden, cfg.expert_intermediate]
                }
                "ffn_gate_shexp.weight" | "ffn_up_shexp.weight" => {
                    vec![cfg.shared_expert_intermediate?, cfg.hidden]
                }
                "ffn_down_shexp.weight" => vec![cfg.hidden, cfg.shared_expert_intermediate?],
                _ => return None,
            })
        }
    }
}

fn dtype_for(entry: &Entry) -> Result<(GgmlType, u64), ToGgufError> {
    let (dtype, width) = match entry.dtype.as_str() {
        "F32" => (GgmlType::F32, 4),
        "F16" => (GgmlType::F16, 2),
        "BF16" => (GgmlType::Bf16, 2),
        other => {
            return Err(ToGgufError::Unsupported(format!(
                "tensor dtype '{other}' cannot be written as a plain GGUF float"
            )))
        }
    };
    Ok((dtype, width))
}

fn cast_plan_data(shard: usize, source: &str, entry: &Entry, target: GgmlType) -> PlanData {
    match target {
        GgmlType::F32 if entry.dtype == "F32" => PlanData::Copy {
            shard,
            source: source.into(),
        },
        GgmlType::F32 => PlanData::CastF32 {
            shard,
            source: source.into(),
        },
        GgmlType::F16 if entry.dtype == "F16" => PlanData::Copy {
            shard,
            source: source.into(),
        },
        GgmlType::F16 => PlanData::Cast {
            shard,
            source: source.into(),
        },
        _ => unreachable!("only F16 and F32 are planned"),
    }
}

fn element_count_u64(shape: &[u64]) -> Result<u64, ToGgufError> {
    shape.iter().try_fold(1u64, |total, dimension| {
        total
            .checked_mul(*dimension)
            .ok_or_else(|| ToGgufError::Unsupported("tensor is too large".into()))
    })
}

fn add_plan(
    plans: &mut Vec<TensorPlan>,
    output: String,
    shape: Vec<u64>,
    dtype: GgmlType,
    width: u64,
    data: PlanData,
) -> Result<(), ToGgufError> {
    if plans.iter().any(|plan| plan.output == output) {
        return Err(ToGgufError::Unsupported(format!(
            "duplicate output tensor '{output}'"
        )));
    }
    let bytes = element_count_u64(&shape)?
        .checked_mul(width)
        .ok_or_else(|| ToGgufError::Unsupported(format!("tensor '{output}' is too large")))?;
    plans.push(TensorPlan {
        output,
        shape,
        dtype,
        bytes,
        data,
        offset: 0,
    });
    Ok(())
}

fn add_float_plan(
    plans: &mut Vec<TensorPlan>,
    shard: usize,
    source_name: &str,
    entry: &Entry,
    output: &str,
    expected: Vec<u64>,
) -> Result<(), ToGgufError> {
    if entry.shape != expected {
        return Err(ToGgufError::Shape {
            name: source_name.into(),
            actual: entry.shape.clone(),
            expected,
        });
    }
    dtype_for(entry)?;
    // llama.cpp keeps norms and biases in F32: elementwise CPU kernels do not
    // accept an F16 vector against F32 activations. Matrices stay F16.
    let (target, width) = if expected.len() == 1 {
        (GgmlType::F32, 4)
    } else {
        (GgmlType::F16, 2)
    };
    add_plan(
        plans,
        output.into(),
        entry.shape.clone(),
        target,
        width,
        cast_plan_data(shard, source_name, entry, target),
    )
}

fn plan_copy(
    plans: &mut Vec<TensorPlan>,
    source: &SourceIndex,
    consumed: &mut BTreeSet<String>,
    source_name: &str,
    output: &str,
    expected: Vec<u64>,
) -> Result<(), ToGgufError> {
    let (shard, entry) = source
        .get(source_name)
        .ok_or_else(|| ToGgufError::Unsupported(format!("missing tensor '{source_name}'")))?;
    add_float_plan(plans, *shard, source_name, entry, output, expected)?;
    consumed.insert(source_name.into());
    Ok(())
}

fn plan_squeeze(
    plans: &mut Vec<TensorPlan>,
    source: &SourceIndex,
    consumed: &mut BTreeSet<String>,
    source_name: &str,
    output: &str,
) -> Result<(), ToGgufError> {
    let (shard, entry) = source
        .get(source_name)
        .ok_or_else(|| ToGgufError::Unsupported(format!("missing tensor '{source_name}'")))?;
    if entry.shape.len() == 2 && entry.shape[0] == 1 {
        // shared_expert_gate is [1, hidden] in HF but [hidden] in GGUF.
    } else if entry.shape.len() != 1 {
        return Err(ToGgufError::Unsupported(format!(
            "tensor '{source_name}' has shape {:?}; expected [1, hidden] or [hidden]",
            entry.shape
        )));
    }
    dtype_for(entry)?;
    let plan_shape = vec![*entry.shape.last().unwrap_or(&1)];
    add_plan(
        plans,
        output.into(),
        plan_shape,
        GgmlType::F32,
        4,
        cast_plan_data(*shard, source_name, entry, GgmlType::F32),
    )?;
    consumed.insert(source_name.into());
    Ok(())
}

fn plan_stacked(
    plans: &mut Vec<TensorPlan>,
    source: &SourceIndex,
    consumed: &mut BTreeSet<String>,
    parts: &[(usize, String)],
    output: &str,
    per_expert: &[u64],
    experts: u64,
) -> Result<(), ToGgufError> {
    if parts.len() as u64 != experts {
        return Err(ToGgufError::Unsupported(format!(
            "found {} tensors for {output}, expected {experts} experts",
            parts.len()
        )));
    }
    for (_, name) in parts {
        let (_, entry) = source
            .get(name)
            .ok_or_else(|| ToGgufError::Unsupported(format!("missing tensor '{name}'")))?;
        if entry.shape != per_expert {
            return Err(ToGgufError::Shape {
                name: name.clone(),
                actual: entry.shape.clone(),
                expected: per_expert.to_vec(),
            });
        }
        dtype_for(entry)?;
    }
    let mut shape = vec![experts];
    shape.extend_from_slice(per_expert);
    add_plan(
        plans,
        output.into(),
        shape,
        GgmlType::F16,
        2,
        PlanData::StackedExperts {
            parts: parts.to_vec(),
        },
    )?;
    for (_, name) in parts {
        consumed.insert(name.clone());
    }
    Ok(())
}

fn plan_fused_gate_up(
    plans: &mut Vec<TensorPlan>,
    source: &SourceIndex,
    consumed: &mut BTreeSet<String>,
    source_name: &str,
    output: &str,
    cfg: &ModelConfig,
    half: usize,
) -> Result<(), ToGgufError> {
    let (shard, entry) = source
        .get(source_name)
        .ok_or_else(|| ToGgufError::Unsupported(format!("missing tensor '{source_name}'")))?;
    let expected = vec![cfg.experts, 2 * cfg.expert_intermediate, cfg.hidden];
    if entry.shape != expected {
        return Err(ToGgufError::Shape {
            name: source_name.into(),
            actual: entry.shape.clone(),
            expected,
        });
    }
    dtype_for(entry)?;
    add_plan(
        plans,
        output.into(),
        vec![cfg.experts, cfg.expert_intermediate, cfg.hidden],
        GgmlType::F16,
        2,
        PlanData::FusedGateUp {
            shard: *shard,
            source: source_name.into(),
            experts: cfg.experts,
            intermediate: cfg.expert_intermediate,
            hidden: cfg.hidden,
            half,
        },
    )?;
    consumed.insert(source_name.into());
    Ok(())
}

fn existing(source: &SourceIndex, candidates: &[String]) -> Option<String> {
    candidates
        .iter()
        .find(|candidate| source.contains_key(*candidate))
        .cloned()
}

fn plan_moe_layers(
    cfg: &ModelConfig,
    source: &SourceIndex,
    consumed: &mut BTreeSet<String>,
    plans: &mut Vec<TensorPlan>,
) -> Result<(), ToGgufError> {
    if cfg.experts == 0 {
        return Ok(());
    }
    for layer in 0..cfg.layers {
        let router = existing(
            source,
            &[
                format!("model.layers.{layer}.mlp.gate.weight"),
                format!("model.layers.{layer}.block_sparse_moe.gate.weight"),
            ],
        )
        .ok_or_else(|| {
            ToGgufError::Unsupported(format!(
                "MoE layer {layer} has no router (mlp.gate or block_sparse_moe.gate)"
            ))
        })?;
        plan_copy(
            plans,
            source,
            consumed,
            &router,
            &format!("blk.{layer}.ffn_gate_inp.weight"),
            vec![cfg.experts, cfg.hidden],
        )?;

        let expert_base = if source
            .keys()
            .any(|name| name.starts_with(&format!("model.layers.{layer}.mlp.experts.")))
        {
            format!("model.layers.{layer}.mlp.experts")
        } else {
            format!("model.layers.{layer}.block_sparse_moe.experts")
        };

        let per_expert_gate = format!("{expert_base}.0.gate_proj.weight");
        let mixtral_gate = format!("{expert_base}.0.w1.weight");
        if source.contains_key(&per_expert_gate) {
            let mut gate = Vec::new();
            let mut up = Vec::new();
            let mut down = Vec::new();
            for expert in 0..cfg.experts {
                gate.push((
                    source[&format!("{expert_base}.{expert}.gate_proj.weight")].0,
                    format!("{expert_base}.{expert}.gate_proj.weight"),
                ));
                up.push((
                    source[&format!("{expert_base}.{expert}.up_proj.weight")].0,
                    format!("{expert_base}.{expert}.up_proj.weight"),
                ));
                down.push((
                    source[&format!("{expert_base}.{expert}.down_proj.weight")].0,
                    format!("{expert_base}.{expert}.down_proj.weight"),
                ));
            }
            plan_stacked(
                plans,
                source,
                consumed,
                &gate,
                &format!("blk.{layer}.ffn_gate_exps.weight"),
                &[cfg.expert_intermediate, cfg.hidden],
                cfg.experts,
            )?;
            plan_stacked(
                plans,
                source,
                consumed,
                &up,
                &format!("blk.{layer}.ffn_up_exps.weight"),
                &[cfg.expert_intermediate, cfg.hidden],
                cfg.experts,
            )?;
            plan_stacked(
                plans,
                source,
                consumed,
                &down,
                &format!("blk.{layer}.ffn_down_exps.weight"),
                &[cfg.hidden, cfg.expert_intermediate],
                cfg.experts,
            )?;
        } else if source.contains_key(&mixtral_gate) {
            let mut gate = Vec::new();
            let mut up = Vec::new();
            let mut down = Vec::new();
            for expert in 0..cfg.experts {
                gate.push((
                    source[&format!("{expert_base}.{expert}.w1.weight")].0,
                    format!("{expert_base}.{expert}.w1.weight"),
                ));
                up.push((
                    source[&format!("{expert_base}.{expert}.w3.weight")].0,
                    format!("{expert_base}.{expert}.w3.weight"),
                ));
                down.push((
                    source[&format!("{expert_base}.{expert}.w2.weight")].0,
                    format!("{expert_base}.{expert}.w2.weight"),
                ));
            }
            plan_stacked(
                plans,
                source,
                consumed,
                &gate,
                &format!("blk.{layer}.ffn_gate_exps.weight"),
                &[cfg.expert_intermediate, cfg.hidden],
                cfg.experts,
            )?;
            plan_stacked(
                plans,
                source,
                consumed,
                &up,
                &format!("blk.{layer}.ffn_up_exps.weight"),
                &[cfg.expert_intermediate, cfg.hidden],
                cfg.experts,
            )?;
            plan_stacked(
                plans,
                source,
                consumed,
                &down,
                &format!("blk.{layer}.ffn_down_exps.weight"),
                &[cfg.hidden, cfg.expert_intermediate],
                cfg.experts,
            )?;
        } else if let Some(gate_up) = existing(
            source,
            &[
                format!("{expert_base}.gate_up_proj"),
                format!("{expert_base}.gate_up_proj.weight"),
            ],
        ) {
            plan_fused_gate_up(
                plans,
                source,
                consumed,
                &gate_up,
                &format!("blk.{layer}.ffn_gate_exps.weight"),
                cfg,
                0,
            )?;
            plan_fused_gate_up(
                plans,
                source,
                consumed,
                &gate_up,
                &format!("blk.{layer}.ffn_up_exps.weight"),
                cfg,
                1,
            )?;
            let down = existing(
                source,
                &[
                    format!("{expert_base}.down_proj"),
                    format!("{expert_base}.down_proj.weight"),
                ],
            )
            .ok_or_else(|| {
                ToGgufError::Unsupported(format!("MoE layer {layer} has no fused down_proj"))
            })?;
            plan_copy(
                plans,
                source,
                consumed,
                &down,
                &format!("blk.{layer}.ffn_down_exps.weight"),
                vec![cfg.experts, cfg.hidden, cfg.expert_intermediate],
            )?;
        } else {
            let gate = existing(
                source,
                &[
                    format!("{expert_base}.gate_proj"),
                    format!("{expert_base}.gate_proj.weight"),
                ],
            )
            .ok_or_else(|| {
                ToGgufError::Unsupported(format!(
                    "MoE layer {layer} has no per-expert or fused gate tensors"
                ))
            })?;
            let up = existing(
                source,
                &[
                    format!("{expert_base}.up_proj"),
                    format!("{expert_base}.up_proj.weight"),
                ],
            )
            .ok_or_else(|| {
                ToGgufError::Unsupported(format!("MoE layer {layer} has no fused up_proj"))
            })?;
            let down = existing(
                source,
                &[
                    format!("{expert_base}.down_proj"),
                    format!("{expert_base}.down_proj.weight"),
                ],
            )
            .ok_or_else(|| {
                ToGgufError::Unsupported(format!("MoE layer {layer} has no fused down_proj"))
            })?;
            plan_copy(
                plans,
                source,
                consumed,
                &gate,
                &format!("blk.{layer}.ffn_gate_exps.weight"),
                vec![cfg.experts, cfg.expert_intermediate, cfg.hidden],
            )?;
            plan_copy(
                plans,
                source,
                consumed,
                &up,
                &format!("blk.{layer}.ffn_up_exps.weight"),
                vec![cfg.experts, cfg.expert_intermediate, cfg.hidden],
            )?;
            plan_copy(
                plans,
                source,
                consumed,
                &down,
                &format!("blk.{layer}.ffn_down_exps.weight"),
                vec![cfg.experts, cfg.hidden, cfg.expert_intermediate],
            )?;
        }

        if let Some(shared) = cfg.shared_expert_intermediate {
            let base = format!("model.layers.{layer}.mlp.shared_expert");
            for (suffix, output, shape) in [
                (
                    "gate_proj.weight",
                    "ffn_gate_shexp.weight",
                    vec![shared, cfg.hidden],
                ),
                (
                    "up_proj.weight",
                    "ffn_up_shexp.weight",
                    vec![shared, cfg.hidden],
                ),
                (
                    "down_proj.weight",
                    "ffn_down_shexp.weight",
                    vec![cfg.hidden, shared],
                ),
            ] {
                let name = format!("{base}.{suffix}");
                if source.contains_key(&name) {
                    plan_copy(
                        plans,
                        source,
                        consumed,
                        &name,
                        &format!("blk.{layer}.{output}"),
                        shape,
                    )?;
                }
            }
            let gate = existing(
                source,
                &[
                    format!("model.layers.{layer}.mlp.shared_expert_gate.weight"),
                    format!("model.layers.{layer}.mlp.shared_expert_gate"),
                ],
            );
            if let Some(gate) = gate {
                plan_squeeze(
                    plans,
                    source,
                    consumed,
                    &gate,
                    &format!("blk.{layer}.ffn_gate_inp_shexp.weight"),
                )?;
            }
        }
    }
    Ok(())
}

fn plan_quantized(
    cfg: &ModelConfig,
    vocab: u64,
    source: &SourceIndex,
    consumed: &mut BTreeSet<String>,
    plans: &mut Vec<TensorPlan>,
) -> Result<(), ToGgufError> {
    // AutoAWQ GEMM: qweight/qzeros are packed I32 and scales are float.
    let prefixes = source
        .keys()
        .filter_map(|name| name.strip_suffix(".qweight").map(str::to_string))
        .collect::<Vec<_>>();
    for prefix in prefixes {
        let qweight = format!("{prefix}.qweight");
        let scales_name = format!("{prefix}.scales");
        if !source.contains_key(&scales_name) {
            return Err(ToGgufError::Unsupported(format!(
                "AWQ tensor '{prefix}' has no scales"
            )));
        }
        let qzeros = source.contains_key(&format!("{prefix}.qzeros")).then(|| {
            (
                source[&format!("{prefix}.qzeros")].0,
                format!("{prefix}.qzeros"),
            )
        });
        let output = output_for_weight(&format!("{prefix}.weight"), cfg.layers)?;
        let expected = expected_shape(&output, cfg, vocab)
            .ok_or_else(|| ToGgufError::Unsupported(format!("no expected shape for '{output}'")))?;
        let qweight_shape = source[&qweight].1.shape.clone();
        let qzeros_shape = qzeros
            .as_ref()
            .map(|(_, name)| source[name].1.shape.clone())
            .unwrap_or_default();
        let scales_shape = source[&scales_name].1.shape.clone();
        let qweight_entry = &source[&qweight].1;
        if qweight_entry.dtype != "I32" {
            return Err(ToGgufError::Unsupported(format!(
                "AWQ qweight '{qweight}' has dtype {} instead of I32",
                qweight_entry.dtype
            )));
        }
        let qzeros_name = qzeros.as_ref().map(|(_, name)| name.clone());
        add_plan(
            plans,
            output,
            expected,
            GgmlType::F16,
            2,
            PlanData::Awq {
                qweight: (source[&qweight].0, qweight.clone()),
                qzeros,
                scales: (source[&scales_name].0, scales_name.clone()),
                qweight_shape,
                qzeros_shape,
                scales_shape,
            },
        )?;
        consumed.insert(qweight);
        consumed.insert(scales_name);
        if let Some(name) = qzeros_name {
            consumed.insert(name);
        }
    }

    // compressed-tensors / llm-compressor FP8: per-row BF16 `weight_scale`.
    let fp8_names = source
        .iter()
        .filter(|(name, (_, entry))| entry.dtype.starts_with("F8_") && name.ends_with(".weight"))
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    for name in fp8_names {
        let prefix = name.strip_suffix(".weight").unwrap_or(&name);
        let scale_name = format!("{prefix}.weight_scale");
        let (scale_shard, scale_entry) = source.get(&scale_name).ok_or_else(|| {
            ToGgufError::Unsupported(format!(
                "FP8 tensor '{name}' has no {scale_name}; block-scaled FP8 is not supported"
            ))
        })?;
        let output = output_for_weight(&name, cfg.layers)?;
        let expected = expected_shape(&output, cfg, vocab)
            .ok_or_else(|| ToGgufError::Unsupported(format!("no expected shape for '{output}'")))?;
        let entry = &source[&name].1;
        if entry.shape != expected {
            return Err(ToGgufError::Shape {
                name: name.clone(),
                actual: entry.shape.clone(),
                expected,
            });
        }
        let scale_count = element_count(&scale_entry.shape)?;
        if scale_count != 1 && scale_count as u64 != expected[0] {
            return Err(ToGgufError::Unsupported(format!(
                "FP8 scale '{scale_name}' has {scale_count} values; expected 1 or {}",
                expected[0]
            )));
        }
        let shape = entry.shape.clone();
        let columns = *shape.get(1).unwrap_or(&0);
        let rows = *shape.first().unwrap_or(&0);
        add_plan(
            plans,
            output,
            shape,
            GgmlType::F16,
            2,
            PlanData::Fp8 {
                shard: source[&name].0,
                source: name.clone(),
                scale_shard: *scale_shard,
                scale: scale_name.clone(),
                rows,
                columns,
            },
        )?;
        consumed.insert(name);
        consumed.insert(scale_name);
    }
    Ok(())
}

fn auxiliary(name: &str) -> bool {
    name.ends_with(".input_scale")
        || name.ends_with(".weight_scale")
        || name.ends_with(".weight_scale_inv")
        || name.ends_with(".qzeros")
        || name.ends_with(".scales")
        || name.ends_with(".qweight")
        || name == "model.rotary_emb.inv_freq"
        || name.ends_with(".self_attn.rotary_emb.inv_freq")
}

fn plan_tensors(
    source: &SourceIndex,
    cfg: &ModelConfig,
    vocab: u64,
) -> Result<Vec<TensorPlan>, ToGgufError> {
    let mut plans = Vec::new();
    let mut consumed = BTreeSet::new();
    plan_moe_layers(cfg, source, &mut consumed, &mut plans)?;
    plan_quantized(cfg, vocab, source, &mut consumed, &mut plans)?;

    for (name, (shard, entry)) in source {
        if consumed.contains(name) || auxiliary(name) {
            continue;
        }
        let Some(output) = map_source_name(name, cfg.layers)? else {
            continue;
        };
        let expected = expected_shape(&output, cfg, vocab)
            .ok_or_else(|| ToGgufError::Unsupported(format!("no expected shape for '{output}'")))?;
        if entry.shape != expected {
            return Err(ToGgufError::Shape {
                name: name.clone(),
                actual: entry.shape.clone(),
                expected,
            });
        }
        add_float_plan(
            &mut plans,
            *shard,
            name,
            entry,
            &output,
            entry.shape.clone(),
        )?;
    }

    let present = plans
        .iter()
        .map(|plan| plan.output.as_str())
        .collect::<BTreeSet<_>>();
    for required in ["token_embd.weight", "output_norm.weight"] {
        if !present.contains(required) {
            return Err(ToGgufError::Unsupported(format!(
                "missing required tensor '{required}'"
            )));
        }
    }
    if !cfg.tie_embeddings && !present.contains("output.weight") {
        return Err(ToGgufError::Unsupported(
            "missing lm_head.weight while tie_word_embeddings is false".into(),
        ));
    }
    for layer in 0..cfg.layers {
        let mut required = vec![
            "attn_norm.weight",
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_norm.weight",
        ];
        if cfg.experts > 0 {
            required.extend([
                "ffn_gate_inp.weight",
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
                "ffn_down_exps.weight",
            ]);
        } else {
            required.extend(["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]);
        }
        if cfg.shared_expert_intermediate.is_some() {
            required.extend([
                "ffn_gate_shexp.weight",
                "ffn_up_shexp.weight",
                "ffn_down_shexp.weight",
                "ffn_gate_inp_shexp.weight",
            ]);
        }
        for suffix in required {
            let name = format!("blk.{layer}.{suffix}");
            if !present.contains(name.as_str()) {
                return Err(ToGgufError::Unsupported(format!(
                    "missing required tensor '{name}'"
                )));
            }
        }
        if cfg.attention_bias {
            let name = format!("blk.{layer}.attn_q.bias");
            if !present.contains(name.as_str()) {
                return Err(ToGgufError::Unsupported(format!(
                    "config declares attention_bias but '{name}' is missing"
                )));
            }
        }
    }
    plans.sort_by(|a, b| a.output.cmp(&b.output));
    let mut offset = 0u64;
    for plan in &mut plans {
        offset = align_up(offset, ALIGNMENT).unwrap_or(offset);
        plan.offset = offset;
        offset = offset.saturating_add(plan.bytes);
    }
    Ok(plans)
}

fn push_string(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn push_value_data(out: &mut Vec<u8>, value: &MetadataValue) -> Result<(), ToGgufError> {
    match value {
        MetadataValue::Uint8(value) => out.push(*value),
        MetadataValue::Int8(value) => out.push(*value as u8),
        MetadataValue::Uint16(value) => out.extend_from_slice(&value.to_le_bytes()),
        MetadataValue::Int16(value) => out.extend_from_slice(&value.to_le_bytes()),
        MetadataValue::Uint32(value) => out.extend_from_slice(&value.to_le_bytes()),
        MetadataValue::Int32(value) => out.extend_from_slice(&value.to_le_bytes()),
        MetadataValue::Float32(value) => out.extend_from_slice(&value.to_le_bytes()),
        MetadataValue::Bool(value) => out.push(u8::from(*value)),
        MetadataValue::String(value) => push_string(out, value),
        MetadataValue::Uint64(value) => out.extend_from_slice(&value.to_le_bytes()),
        MetadataValue::Int64(value) => out.extend_from_slice(&value.to_le_bytes()),
        MetadataValue::Float64(value) => out.extend_from_slice(&value.to_le_bytes()),
        MetadataValue::Array(_) => {
            return Err(ToGgufError::Config(
                "nested metadata arrays are not supported".into(),
            ))
        }
    }
    Ok(())
}

fn push_metadata(out: &mut Vec<u8>, key: &str, value: &MetadataValue) -> Result<(), ToGgufError> {
    push_string(out, key);
    out.extend_from_slice(&(value.value_type() as u32).to_le_bytes());
    match value {
        MetadataValue::Array(values) => {
            let element_type = values
                .first()
                .map(|value| value.value_type() as u32)
                .unwrap_or(0);
            out.extend_from_slice(&element_type.to_le_bytes());
            out.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for value in values {
                push_value_data(out, value)?;
            }
        }
        other => push_value_data(out, other)?,
    }
    Ok(())
}

fn build_metadata(
    cfg: &ModelConfig,
    tokenizer: &TokenizerData,
    vocab: u64,
    file_type: u32,
) -> BTreeMap<String, MetadataValue> {
    let mut metadata = BTreeMap::new();
    let arch = &cfg.architecture;
    metadata.insert(
        "general.architecture".into(),
        MetadataValue::String(arch.clone()),
    );
    metadata.insert(
        "general.name".into(),
        MetadataValue::String(cfg.name.clone()),
    );
    metadata.insert("general.file_type".into(), MetadataValue::Uint32(file_type));
    metadata.insert(
        "general.alignment".into(),
        MetadataValue::Uint32(ALIGNMENT as u32),
    );
    metadata.insert(
        "general.quantization_version".into(),
        MetadataValue::Uint32(2),
    );
    metadata.insert(
        format!("{arch}.context_length"),
        MetadataValue::Uint32(cfg.context as u32),
    );
    metadata.insert(
        format!("{arch}.embedding_length"),
        MetadataValue::Uint32(cfg.hidden as u32),
    );
    metadata.insert(
        format!("{arch}.block_count"),
        MetadataValue::Uint32(cfg.layers as u32),
    );
    metadata.insert(
        format!("{arch}.feed_forward_length"),
        MetadataValue::Uint32(cfg.intermediate as u32),
    );
    metadata.insert(
        format!("{arch}.attention.head_count"),
        MetadataValue::Uint32(cfg.heads as u32),
    );
    metadata.insert(
        format!("{arch}.attention.head_count_kv"),
        MetadataValue::Uint32(cfg.kv_heads as u32),
    );
    metadata.insert(
        format!("{arch}.attention.layer_norm_rms_epsilon"),
        MetadataValue::Float32(cfg.rms_eps as f32),
    );
    metadata.insert(
        format!("{arch}.rope.freq_base"),
        MetadataValue::Float32(cfg.rope_theta as f32),
    );
    metadata.insert(
        format!("{arch}.rope.dimension_count"),
        MetadataValue::Uint32(cfg.head_dim as u32),
    );
    if cfg.experts > 0 {
        metadata.insert(
            format!("{arch}.expert_count"),
            MetadataValue::Uint32(cfg.experts as u32),
        );
        metadata.insert(
            format!("{arch}.expert_used_count"),
            MetadataValue::Uint32(cfg.experts_used as u32),
        );
        metadata.insert(
            format!("{arch}.expert_feed_forward_length"),
            MetadataValue::Uint32(cfg.expert_intermediate as u32),
        );
    }
    if let Some(shared) = cfg.shared_expert_intermediate {
        metadata.insert(
            format!("{arch}.expert_shared_feed_forward_length"),
            MetadataValue::Uint32(shared as u32),
        );
    }
    if let Some(scaling) = &cfg.rope_scaling {
        metadata.insert(
            format!("{arch}.rope.scaling.type"),
            MetadataValue::String(scaling.kind.clone()),
        );
        metadata.insert(
            format!("{arch}.rope.scaling.factor"),
            MetadataValue::Float32(scaling.factor as f32),
        );
        if let Some(context) = scaling.original_context {
            metadata.insert(
                format!("{arch}.rope.scaling.original_context_length"),
                MetadataValue::Uint32(context as u32),
            );
        }
    }
    metadata.insert(
        format!("{arch}.vocab_size"),
        MetadataValue::Uint32(vocab as u32),
    );
    metadata.insert(
        "tokenizer.ggml.model".into(),
        MetadataValue::String(tokenizer.model.clone()),
    );
    if tokenizer.model == "gpt2" {
        let pre = match arch.as_str() {
            "llama" => Some("llama3"),
            "qwen2" | "qwen3" | "qwen2moe" | "qwen3moe" => Some("qwen2"),
            _ => None,
        };
        if let Some(pre) = pre {
            metadata.insert(
                "tokenizer.ggml.pre".into(),
                MetadataValue::String(pre.into()),
            );
        }
        metadata.insert(
            "tokenizer.ggml.merges".into(),
            MetadataValue::Array(
                tokenizer
                    .merges
                    .iter()
                    .cloned()
                    .map(MetadataValue::String)
                    .collect(),
            ),
        );
    } else {
        metadata.insert(
            "tokenizer.ggml.scores".into(),
            MetadataValue::Array(
                tokenizer
                    .scores
                    .iter()
                    .copied()
                    .map(MetadataValue::Float32)
                    .collect(),
            ),
        );
    }
    metadata.insert(
        "tokenizer.ggml.tokens".into(),
        MetadataValue::Array(
            tokenizer
                .tokens
                .iter()
                .cloned()
                .map(MetadataValue::String)
                .collect(),
        ),
    );
    metadata.insert(
        "tokenizer.ggml.token_type".into(),
        MetadataValue::Array(
            tokenizer
                .token_types
                .iter()
                .copied()
                .map(MetadataValue::Int32)
                .collect(),
        ),
    );
    metadata.insert(
        "tokenizer.ggml.add_bos_token".into(),
        MetadataValue::Bool(tokenizer.add_bos),
    );
    metadata.insert(
        "tokenizer.ggml.add_eos_token".into(),
        MetadataValue::Bool(tokenizer.add_eos),
    );
    if let Some(id) = tokenizer.bos {
        metadata.insert(
            "tokenizer.ggml.bos_token_id".into(),
            MetadataValue::Uint32(id),
        );
    }
    if let Some(id) = tokenizer.eos {
        metadata.insert(
            "tokenizer.ggml.eos_token_id".into(),
            MetadataValue::Uint32(id),
        );
    }
    if let Some(id) = tokenizer.unk {
        metadata.insert(
            "tokenizer.ggml.unknown_token_id".into(),
            MetadataValue::Uint32(id),
        );
    }
    if let Some(id) = tokenizer.pad {
        metadata.insert(
            "tokenizer.ggml.padding_token_id".into(),
            MetadataValue::Uint32(id),
        );
    }
    if let Some(template) = &tokenizer.chat_template {
        metadata.insert(
            "tokenizer.chat_template".into(),
            MetadataValue::String(template.clone()),
        );
    }
    metadata
}

fn write_header(
    metadata: &BTreeMap<String, MetadataValue>,
    tensors: &[TensorPlan],
) -> Result<Vec<u8>, ToGgufError> {
    let mut header = Vec::new();
    header.extend_from_slice(b"GGUF");
    header.extend_from_slice(&3u32.to_le_bytes());
    header.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    header.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (key, value) in metadata {
        push_metadata(&mut header, key, value)?;
    }
    for tensor in tensors {
        push_string(&mut header, &tensor.output);
        header.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
        for dimension in tensor.shape.iter().rev() {
            header.extend_from_slice(&dimension.to_le_bytes());
        }
        header.extend_from_slice(&(tensor.dtype as u32).to_le_bytes());
        header.extend_from_slice(&tensor.offset.to_le_bytes());
    }
    let data_start = align_up(header.len() as u64, ALIGNMENT)
        .ok_or_else(|| ToGgufError::Unsupported("header is too large".into()))?;
    header.resize(data_start as usize, 0);
    Ok(header)
}

fn source_width(dtype: &str) -> Result<usize, ToGgufError> {
    match dtype {
        "F32" => Ok(4),
        "F16" | "BF16" => Ok(2),
        other => Err(ToGgufError::Unsupported(format!(
            "tensor dtype '{other}' cannot be converted to F16"
        ))),
    }
}

fn write_f32_slice<W: Write>(
    writer: &mut W,
    name: &str,
    dtype: &str,
    raw: &[u8],
) -> Result<u64, ToGgufError> {
    let width = source_width(dtype)?;
    if !raw.len().is_multiple_of(width) {
        return Err(ToGgufError::Unsupported(format!(
            "tensor '{name}' has a truncated {dtype} payload"
        )));
    }
    if dtype == "F32" {
        writer
            .write_all(raw)
            .map_err(|error| ToGgufError::Io(error.to_string()))?;
        return Ok(raw.len() as u64);
    }
    let mut out = Vec::with_capacity(128 * 1024);
    let mut written = 0u64;
    for chunk in raw.chunks_exact(width) {
        let value = match dtype {
            "F16" => f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32(),
            "BF16" => bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32(),
            _ => unreachable!("dtype was checked above"),
        };
        out.extend_from_slice(&value.to_le_bytes());
        if out.len() >= 128 * 1024 {
            writer
                .write_all(&out)
                .map_err(|error| ToGgufError::Io(error.to_string()))?;
            written += out.len() as u64;
            out.clear();
        }
    }
    writer
        .write_all(&out)
        .map_err(|error| ToGgufError::Io(error.to_string()))?;
    written += out.len() as u64;
    Ok(written)
}

fn write_f16_slice<W: Write>(
    writer: &mut W,
    name: &str,
    dtype: &str,
    raw: &[u8],
) -> Result<u64, ToGgufError> {
    let width = source_width(dtype)?;
    if !raw.len().is_multiple_of(width) {
        return Err(ToGgufError::Unsupported(format!(
            "tensor '{name}' has a truncated {dtype} payload"
        )));
    }
    if dtype == "F16" {
        writer
            .write_all(raw)
            .map_err(|error| ToGgufError::Io(error.to_string()))?;
        return Ok(raw.len() as u64);
    }
    let mut out = Vec::with_capacity(128 * 1024);
    let mut written = 0u64;
    for chunk in raw.chunks_exact(width) {
        let value = match dtype {
            "F32" => f32::from_le_bytes(chunk.try_into().expect("four bytes")),
            "BF16" => bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32(),
            _ => unreachable!("dtype was checked above"),
        };
        out.extend_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
        if out.len() >= 128 * 1024 {
            writer
                .write_all(&out)
                .map_err(|error| ToGgufError::Io(error.to_string()))?;
            written += out.len() as u64;
            out.clear();
        }
    }
    writer
        .write_all(&out)
        .map_err(|error| ToGgufError::Io(error.to_string()))?;
    written += out.len() as u64;
    Ok(written)
}

fn read_float_source(
    readers: &mut [Reader],
    shard: usize,
    name: &str,
) -> Result<Vec<f32>, ToGgufError> {
    let entry = readers[shard]
        .entries
        .get(name)
        .ok_or_else(|| ToGgufError::Unsupported(format!("missing tensor '{name}'")))?
        .clone();
    let count = element_count(&entry.shape)?;
    Ok(decode_floats(
        name,
        &entry.dtype,
        &readers[shard].bytes(name)?,
        count,
    )?)
}

fn write_plan<W: Write>(
    writer: &mut W,
    readers: &mut [Reader],
    plan: &TensorPlan,
) -> Result<u64, ToGgufError> {
    let mut written = 0u64;
    match &plan.data {
        PlanData::Copy { shard, source } => {
            let data = readers[*shard].bytes(source)?;
            if data.len() as u64 != plan.bytes {
                return Err(ToGgufError::Safetensors(NativeError::Tensor {
                    name: source.clone(),
                    msg: "source data changed while converting".into(),
                }));
            }
            writer
                .write_all(&data)
                .map_err(|error| ToGgufError::Io(error.to_string()))?;
            written = data.len() as u64;
        }
        PlanData::Cast { shard, source } => {
            let entry = readers[*shard]
                .entries
                .get(source)
                .ok_or_else(|| ToGgufError::Unsupported(format!("missing tensor '{source}'")))?
                .clone();
            let data = readers[*shard].bytes(source)?;
            written = write_f16_slice(writer, source, &entry.dtype, &data)?;
        }
        PlanData::CastF32 { shard, source } => {
            let entry = readers[*shard]
                .entries
                .get(source)
                .ok_or_else(|| ToGgufError::Unsupported(format!("missing tensor '{source}'")))?
                .clone();
            let data = readers[*shard].bytes(source)?;
            written = write_f32_slice(writer, source, &entry.dtype, &data)?;
        }
        PlanData::StackedExperts { parts } => {
            for (shard, source) in parts {
                let entry = readers[*shard]
                    .entries
                    .get(source)
                    .ok_or_else(|| ToGgufError::Unsupported(format!("missing tensor '{source}'")))?
                    .clone();
                let data = readers[*shard].bytes(source)?;
                written += write_f16_slice(writer, source, &entry.dtype, &data)?;
            }
        }
        PlanData::FusedGateUp {
            shard,
            source,
            experts,
            intermediate,
            hidden,
            half,
        } => {
            let entry = readers[*shard]
                .entries
                .get(source)
                .ok_or_else(|| ToGgufError::Unsupported(format!("missing tensor '{source}'")))?
                .clone();
            let data = readers[*shard].bytes(source)?;
            let width = source_width(&entry.dtype)? as u64;
            let expert_stride = 2 * intermediate * hidden * width;
            let half_stride = *intermediate * hidden * width;
            for expert in 0..*experts {
                let start = (expert * expert_stride + *half as u64 * half_stride) as usize;
                let end = start + half_stride as usize;
                let part = data.get(start..end).ok_or_else(|| {
                    ToGgufError::Unsupported(format!(
                        "fused expert tensor '{source}' is shorter than declared"
                    ))
                })?;
                written += write_f16_slice(writer, source, &entry.dtype, part)?;
            }
        }
        PlanData::Fp8 {
            shard,
            source,
            scale_shard,
            scale,
            rows,
            columns,
        } => {
            let entry = readers[*shard]
                .entries
                .get(source)
                .ok_or_else(|| ToGgufError::Unsupported(format!("missing tensor '{source}'")))?
                .clone();
            let count = element_count(&entry.shape)?;
            let weights =
                decode_floats(source, &entry.dtype, &readers[*shard].bytes(source)?, count)?;
            let scales = read_float_source(readers, *scale_shard, scale)?;
            if scales.len() != 1 && scales.len() as u64 != *rows {
                return Err(ToGgufError::Unsupported(format!(
                    "FP8 scale '{scale}' has {} values for {rows} rows",
                    scales.len()
                )));
            }
            if count as u64 != rows.saturating_mul(*columns) {
                return Err(ToGgufError::Unsupported(format!(
                    "FP8 tensor '{source}' does not match its output shape"
                )));
            }
            let mut encoded = Vec::with_capacity(count * 2);
            for (index, value) in weights.into_iter().enumerate() {
                let row = index as u64 / *columns;
                let factor = if scales.len() == 1 {
                    scales[0]
                } else {
                    scales[row as usize]
                };
                encoded.extend_from_slice(&f16::from_f32(value * factor).to_bits().to_le_bytes());
            }
            if encoded.len() as u64 != plan.bytes {
                return Err(ToGgufError::Unsupported(format!(
                    "dequantized FP8 tensor '{source}' has the wrong size"
                )));
            }
            writer
                .write_all(&encoded)
                .map_err(|error| ToGgufError::Io(error.to_string()))?;
            written = encoded.len() as u64;
        }
        PlanData::Awq {
            qweight,
            qzeros,
            scales,
            qweight_shape,
            qzeros_shape,
            scales_shape,
        } => {
            let qweight_data = decode_u32(&qweight.1, &readers[qweight.0].bytes(&qweight.1)?)?;
            let qzeros_data = qzeros
                .as_ref()
                .map(|(shard, name)| -> Result<Vec<u32>, ToGgufError> {
                    Ok(decode_u32(name, &readers[*shard].bytes(name)?)?)
                })
                .transpose()?;
            let scale_data = read_float_source(readers, scales.0, &scales.1)?;
            let (shape, data) = awq_dequant(
                plan.output.as_str(),
                &qweight_data,
                qzeros_data.as_deref(),
                &scale_data,
                qweight_shape,
                qzeros_shape,
                scales_shape,
            )?;
            let expected = plan.shape.clone();
            if shape != expected {
                return Err(ToGgufError::Unsupported(format!(
                    "AWQ tensor '{}' dequantized to {shape:?}, expected {expected:?}",
                    plan.output
                )));
            }
            if data.len() as u64 != plan.bytes {
                return Err(ToGgufError::Unsupported(format!(
                    "dequantized AWQ tensor '{}' has the wrong size",
                    plan.output
                )));
            }
            written = write_f16_slice(writer, &plan.output, "BF16", &data)?;
        }
    }
    Ok(written)
}

/// Convert a directory containing `config.json`, a tokenizer (`tokenizer.model`
/// or `tokenizer.json`) and one or more safetensors shards into a GGUF file.
pub fn convert_llama_safetensors(
    input_dir: &Path,
    output: &Path,
    force: bool,
) -> Result<ToGgufReport, ToGgufError> {
    if output.exists() && !force {
        return Err(ToGgufError::OutputExists(output.to_path_buf()));
    }
    let cfg = read_config(input_dir)?;
    let mut tokenizer = read_tokenizer(input_dir)?;
    if let Some(target) = cfg.vocab {
        pad_tokenizer(&mut tokenizer, target)?;
    }
    let vocab = cfg.vocab.unwrap_or(tokenizer.tokens.len() as u64);
    if vocab != tokenizer.tokens.len() as u64 {
        return Err(ToGgufError::Tokenizer(format!(
            "config vocab_size {vocab} differs from tokenizer vocab {}",
            tokenizer.tokens.len()
        )));
    }
    if tokenizer.model == "llama" && tokenizer.scores.len() != tokenizer.tokens.len() {
        return Err(ToGgufError::Tokenizer(
            "SentencePiece vocabulary has a score/token count mismatch".into(),
        ));
    }
    let (mut readers, source) = load_shards(input_dir)?;
    let tensors = plan_tensors(&source, &cfg, vocab)?;
    let file_type = if tensors
        .iter()
        .any(|tensor| matches!(tensor.dtype, GgmlType::Bf16))
    {
        32
    } else if tensors
        .iter()
        .any(|tensor| matches!(tensor.dtype, GgmlType::F16))
    {
        1
    } else {
        0
    };
    let metadata = build_metadata(&cfg, &tokenizer, vocab, file_type);
    let header = write_header(&metadata, &tensors)?;
    let parent = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|error| ToGgufError::Io(error.to_string()))?;
    let mut temp =
        NamedTempFile::new_in(parent).map_err(|error| ToGgufError::Io(error.to_string()))?;
    {
        let mut writer = BufWriter::new(temp.as_file_mut());
        writer
            .write_all(&header)
            .map_err(|error| ToGgufError::Io(error.to_string()))?;
        let mut written = 0u64;
        for tensor in &tensors {
            if tensor.offset > written {
                let padding = (tensor.offset - written) as usize;
                writer
                    .write_all(&vec![0u8; padding])
                    .map_err(|error| ToGgufError::Io(error.to_string()))?;
                written = tensor.offset;
            }
            let bytes = write_plan(&mut writer, &mut readers, tensor)?;
            if bytes != tensor.bytes {
                return Err(ToGgufError::Unsupported(format!(
                    "tensor '{}' produced {bytes} bytes, expected {}",
                    tensor.output, tensor.bytes
                )));
            }
            written = written.saturating_add(bytes);
        }
        writer
            .flush()
            .map_err(|error| ToGgufError::Io(error.to_string()))?;
    }
    temp.as_file()
        .sync_all()
        .map_err(|error| ToGgufError::Io(error.to_string()))?;
    if output.exists() {
        fs::remove_file(output).map_err(|error| ToGgufError::Io(error.to_string()))?;
    }
    temp.persist(output)
        .map_err(|error| ToGgufError::Io(error.to_string()))?;
    if let Err(error) = Gguf::open(output) {
        let _ = fs::remove_file(output);
        return Err(ToGgufError::InvalidOutput(error.to_string()));
    }
    Ok(ToGgufReport {
        input: input_dir.display().to_string(),
        output: output.display().to_string(),
        architecture: cfg.architecture,
        layers: cfg.layers,
        tensors: tensors.len(),
        tokenizer_tokens: tokenizer.tokens.len(),
        file_type,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;

    fn write_safetensors(path: &Path, tensors: &[(&str, Vec<u64>, usize)]) {
        let mut header = serde_json::Map::new();
        let mut offset = 0u64;
        for (name, shape, elements) in tensors {
            let bytes = elements * 4;
            header.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [offset, offset + bytes as u64],
                }),
            );
            offset += bytes as u64;
        }
        let mut raw = serde_json::to_vec(&Value::Object(header)).unwrap();
        while !raw.len().is_multiple_of(8) {
            raw.push(b' ');
        }
        let mut file = File::create(path).unwrap();
        file.write_all(&(raw.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&raw).unwrap();
        file.write_all(&vec![0u8; offset as usize]).unwrap();
    }

    #[test]
    fn converts_a_tiny_llama_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("model");
        fs::create_dir(&input).unwrap();
        fs::write(
            input.join("config.json"),
            serde_json::json!({
                "model_type": "llama",
                "hidden_size": 4,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "intermediate_size": 8,
                "max_position_embeddings": 16,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "vocab_size": 5,
                "tie_word_embeddings": true
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            input.join("tokenizer.json"),
            serde_json::json!({
                "model": {
                    "type": "BPE",
                    "vocab": {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4},
                    "merges": ["a b"]
                }
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            input.join("tokenizer_config.json"),
            serde_json::json!({
                "add_bos_token": true,
                "bos_token": "a",
                "eos_token": "e",
                "chat_template": "{{ messages }}"
            })
            .to_string(),
        )
        .unwrap();
        write_safetensors(
            &input.join("model.safetensors"),
            &[
                ("model.embed_tokens.weight", vec![5, 4], 20),
                ("model.layers.0.input_layernorm.weight", vec![4], 4),
                ("model.layers.0.post_attention_layernorm.weight", vec![4], 4),
                ("model.layers.0.self_attn.q_proj.weight", vec![4, 4], 16),
                ("model.layers.0.self_attn.k_proj.weight", vec![2, 4], 8),
                ("model.layers.0.self_attn.v_proj.weight", vec![2, 4], 8),
                ("model.layers.0.self_attn.o_proj.weight", vec![4, 4], 16),
                ("model.layers.0.mlp.gate_proj.weight", vec![8, 4], 32),
                ("model.layers.0.mlp.up_proj.weight", vec![8, 4], 32),
                ("model.layers.0.mlp.down_proj.weight", vec![4, 8], 32),
                ("model.norm.weight", vec![4], 4),
            ],
        );
        let output = dir.path().join("model.gguf");
        let report = convert_llama_safetensors(&input, &output, false).unwrap();
        assert_eq!(report.tensors, 11);
        assert_eq!(report.tokenizer_tokens, 5);
        let gguf = Gguf::open(&output).unwrap();
        let embed = gguf
            .tensors
            .iter()
            .find(|tensor| tensor.name == "token_embd.weight")
            .unwrap();
        assert_eq!(embed.dims, vec![4, 5]);
        assert!(!gguf.has_mtp_metadata);
    }

    #[test]
    fn reads_a_sentencepiece_model() {
        fn varint(mut value: u64) -> Vec<u8> {
            let mut out = Vec::new();
            loop {
                let mut byte = (value & 0x7f) as u8;
                value >>= 7;
                if value != 0 {
                    byte |= 0x80;
                }
                out.push(byte);
                if value == 0 {
                    return out;
                }
            }
        }
        fn field_bytes(field: u64, bytes: &[u8]) -> Vec<u8> {
            let mut out = varint(field << 3 | 2);
            out.extend(varint(bytes.len() as u64));
            out.extend_from_slice(bytes);
            out
        }
        fn piece(text: &str, score: f32, kind: u64) -> Vec<u8> {
            let mut out = field_bytes(1, text.as_bytes());
            out.push(2 << 3 | 5);
            out.extend_from_slice(&score.to_le_bytes());
            out.extend(varint(3 << 3));
            out.extend(varint(kind));
            out
        }
        let dir = tempfile::tempdir().unwrap();
        let mut model = Vec::new();
        for (text, score, kind) in [
            ("<unk>", 0.0, 2),
            ("<s>", 0.0, 3),
            ("</s>", 0.0, 3),
            ("hello", -1.0, 1),
            ("world", -2.0, 1),
        ] {
            model.extend(field_bytes(1, &piece(text, score, kind)));
        }
        fs::write(dir.path().join("tokenizer.model"), model).unwrap();
        fs::write(
            dir.path().join("tokenizer_config.json"),
            serde_json::json!({
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "add_bos_token": true
            })
            .to_string(),
        )
        .unwrap();
        let tokenizer = read_tokenizer(dir.path()).unwrap();
        assert_eq!(tokenizer.model, "llama");
        assert_eq!(tokenizer.tokens, ["<unk>", "<s>", "</s>", "hello", "world"]);
        assert_eq!(tokenizer.token_types, [2, 3, 3, 1, 1]);
        assert_eq!(tokenizer.scores, [0.0, 0.0, 0.0, -1.0, -2.0]);
        assert_eq!(tokenizer.bos, Some(1));
        assert_eq!(tokenizer.eos, Some(2));
    }

    #[test]
    fn converts_fused_mixtral_experts() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("moe");
        fs::create_dir(&input).unwrap();
        fs::write(
            input.join("config.json"),
            serde_json::json!({
                "model_type": "mixtral",
                "hidden_size": 4,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "intermediate_size": 8,
                "num_local_experts": 2,
                "num_experts_per_tok": 1,
                "vocab_size": 5,
                "tie_word_embeddings": true,
                "max_position_embeddings": 16,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            input.join("tokenizer.json"),
            serde_json::json!({
                "model": {
                    "type": "BPE",
                    "vocab": {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4},
                    "merges": ["a b"]
                }
            })
            .to_string(),
        )
        .unwrap();
        write_safetensors(
            &input.join("model.safetensors"),
            &[
                ("model.embed_tokens.weight", vec![5, 4], 20),
                ("model.layers.0.input_layernorm.weight", vec![4], 4),
                ("model.layers.0.post_attention_layernorm.weight", vec![4], 4),
                ("model.layers.0.self_attn.q_proj.weight", vec![4, 4], 16),
                ("model.layers.0.self_attn.k_proj.weight", vec![2, 4], 8),
                ("model.layers.0.self_attn.v_proj.weight", vec![2, 4], 8),
                ("model.layers.0.self_attn.o_proj.weight", vec![4, 4], 16),
                ("model.layers.0.mlp.gate.weight", vec![2, 4], 8),
                (
                    "model.layers.0.mlp.experts.gate_up_proj",
                    vec![2, 16, 4],
                    128,
                ),
                ("model.layers.0.mlp.experts.down_proj", vec![2, 4, 8], 64),
                ("model.norm.weight", vec![4], 4),
            ],
        );
        let output = dir.path().join("moe.gguf");
        let report = convert_llama_safetensors(&input, &output, false).unwrap();
        assert_eq!(report.tensors, 12);
        assert_eq!(report.architecture, "llama");
        let gguf = Gguf::open(&output).unwrap();
        let gate = gguf
            .tensors
            .iter()
            .find(|tensor| tensor.name == "blk.0.ffn_gate_exps.weight")
            .unwrap();
        assert_eq!(gate.dims, vec![4, 8, 2]);
        let down = gguf
            .tensors
            .iter()
            .find(|tensor| tensor.name == "blk.0.ffn_down_exps.weight")
            .unwrap();
        assert_eq!(down.dims, vec![8, 4, 2]);
        let router = gguf
            .tensors
            .iter()
            .find(|tensor| tensor.name == "blk.0.ffn_gate_inp.weight")
            .unwrap();
        assert_eq!(router.dims, vec![4, 2]);
        let header = String::from_utf8_lossy(&gguf.header);
        assert!(header.contains("llama.expert_count"));
        assert!(header.contains("llama.expert_used_count"));
    }
}
