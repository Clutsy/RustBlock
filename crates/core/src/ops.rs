//! GGUF operations that do not involve recompression: split, merge, metadata
//! editing, pruning, diffing and SHA-256/LFS verification.
//!
//! The writer is built on `pmetal-gguf`'s GGUF reader/builder. It preserves
//! tensor bytes and metadata values, but rebuilding a header can change the
//! order of metadata keys and padding; byte-identical round-trips are verified
//! as tensor/data equality unless the source layout is already canonical.

use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use pmetal_gguf::{GgmlType, GgufBuilder, GgufContent, MetadataValue, TensorInfo};
use sha2::{Digest, Sha256};

use crate::gguf::Gguf;

#[derive(Debug, thiserror::Error)]
pub enum OpsError {
    #[error("cannot read GGUF: {0}")]
    Gguf(String),
    #[error("cannot write GGUF: {0}")]
    Write(String),
    #[error("tensor '{name}' uses unsupported type {ty}")]
    UnsupportedType { name: String, ty: u32 },
    #[error("invalid metadata value: {0}")]
    Metadata(String),
    #[error("no tensors selected")]
    NoTensors,
    #[error("output already exists: {0}")]
    OutputExists(PathBuf),
    #[error("invalid split shard: {0}")]
    Split(String),
    #[error("chat template: {0}")]
    Template(String),
}

#[derive(Debug, Clone)]
pub struct DiffReport {
    pub metadata_only_in_a: Vec<String>,
    pub metadata_only_in_b: Vec<String>,
    pub metadata_changed: Vec<String>,
    pub tensors_only_in_a: Vec<String>,
    pub tensors_only_in_b: Vec<String>,
    pub tensor_changes: Vec<String>,
}

impl DiffReport {
    pub fn is_empty(&self) -> bool {
        self.metadata_only_in_a.is_empty()
            && self.metadata_only_in_b.is_empty()
            && self.metadata_changed.is_empty()
            && self.tensors_only_in_a.is_empty()
            && self.tensors_only_in_b.is_empty()
            && self.tensor_changes.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationIssue {
    pub key: String,
    pub message: String,
}

#[derive(Debug, Clone, Default)]
pub struct MetadataValidation {
    pub issues: Vec<ValidationIssue>,
}

impl MetadataValidation {
    pub fn is_ok(&self) -> bool {
        self.issues.is_empty()
    }
}

fn content(path: &Path) -> Result<GgufContent, OpsError> {
    GgufContent::from_file(path)
        .map_err(|error| OpsError::Gguf(format!("{}: {error}", path.display())))
}

fn type_of(info: &TensorInfo) -> Result<GgmlType, OpsError> {
    GgmlType::try_from(info.dtype as u32).map_err(|_| OpsError::UnsupportedType {
        name: info.name.clone(),
        ty: info.dtype as u32,
    })
}

fn ordered_tensors(content: &GgufContent) -> Vec<TensorInfo> {
    let mut tensors: Vec<TensorInfo> = content.tensor_infos.values().cloned().collect();
    tensors.sort_by_key(|tensor| tensor.offset);
    tensors
}

fn read_tensor(content: &GgufContent, file: &mut File, name: &str) -> Result<Vec<u8>, OpsError> {
    content
        .read_tensor_data(file, name)
        .map_err(|error| OpsError::Gguf(format!("tensor '{name}': {error}")))
}

fn builder_from(content: &GgufContent) -> Result<GgufBuilder, OpsError> {
    let alignment = match content.get_metadata("general.alignment") {
        Some(MetadataValue::Uint32(value)) => *value,
        Some(MetadataValue::Uint64(value)) => u32::try_from(*value)
            .map_err(|_| OpsError::Metadata("alignment is too large".into()))?,
        _ => 32,
    };
    let mut builder = GgufBuilder::new().alignment(alignment);
    for (key, value) in &content.metadata {
        if key == "general.alignment" || key.starts_with("split.") {
            continue;
        }
        builder.add_metadata(key.clone(), value.clone());
    }
    Ok(builder)
}

fn add_tensor(builder: &mut GgufBuilder, info: &TensorInfo, data: Vec<u8>) -> Result<(), OpsError> {
    let dtype = type_of(info)?;
    builder.add_raw_tensor(info.name.clone(), info.dimensions.clone(), dtype, data);
    Ok(())
}

fn finish(builder: &GgufBuilder, output: &Path, force: bool) -> Result<(), OpsError> {
    if output.exists() && !force {
        return Err(OpsError::OutputExists(output.to_path_buf()));
    }
    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|error| OpsError::Write(error.to_string()))?;
    }
    let tmp = output.with_extension("tmp.gguf");
    {
        let mut file = File::create(&tmp).map_err(|error| OpsError::Write(error.to_string()))?;
        builder
            .write(&mut file)
            .map_err(|error| OpsError::Write(error.to_string()))?;
        file.flush()
            .map_err(|error| OpsError::Write(error.to_string()))?;
    }
    if output.exists() {
        fs::remove_file(output).map_err(|error| OpsError::Write(error.to_string()))?;
    }
    fs::rename(&tmp, output).map_err(|error| OpsError::Write(error.to_string()))?;
    Ok(())
}

/// Split a GGUF into numbered shards. `max_bytes` is a soft limit based on
/// tensor payload sizes; the split metadata records the shard number/count.
pub fn split(
    input: &Path,
    output_dir: &Path,
    max_bytes: u64,
    force: bool,
) -> Result<Vec<PathBuf>, OpsError> {
    let content = content(input)?;
    let mut file = File::open(input).map_err(|error| OpsError::Gguf(error.to_string()))?;
    let tensors = ordered_tensors(&content);
    if tensors.is_empty() {
        return Err(OpsError::NoTensors);
    }
    let limit = max_bytes.max(1);
    let mut groups: Vec<Vec<&TensorInfo>> = Vec::new();
    let mut current = Vec::new();
    let mut current_bytes = 0u64;
    for tensor in &tensors {
        let size = tensor
            .byte_size_checked()
            .map_err(|error| OpsError::Gguf(error.to_string()))? as u64;
        if !current.is_empty() && current_bytes.saturating_add(size) > limit {
            groups.push(std::mem::take(&mut current));
            current_bytes = 0;
        }
        current.push(tensor);
        current_bytes = current_bytes.saturating_add(size);
    }
    if !current.is_empty() {
        groups.push(current);
    }

    fs::create_dir_all(output_dir).map_err(|error| OpsError::Write(error.to_string()))?;
    let stem = input
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("model");
    let count = groups.len();
    let manifest_path = output_dir.join(format!("{stem}.rustblock-split.json"));
    let header_path = output_dir.join(format!("{stem}.rustblock-header.bin"));
    if !force && (manifest_path.exists() || header_path.exists()) {
        return Err(OpsError::OutputExists(manifest_path));
    }
    let mut paths = Vec::with_capacity(count);
    for (index, group) in groups.iter().enumerate() {
        let name = format!("{stem}-{:05}-of-{:05}.gguf", index + 1, count);
        let path = output_dir.join(name);
        let mut builder = builder_from(&content)?;
        builder.add_metadata("split.no", MetadataValue::Uint32(index as u32));
        builder.add_metadata("split.count", MetadataValue::Uint32(count as u32));
        builder.add_metadata(
            "split.tensors.count",
            MetadataValue::Uint32(group.len() as u32),
        );
        for tensor in group {
            let data = read_tensor(&content, &mut file, &tensor.name)?;
            add_tensor(&mut builder, tensor, data)?;
        }
        finish(&builder, &path, force)?;
        paths.push(path);
    }
    let original = Gguf::open(input).map_err(|error| OpsError::Gguf(error.to_string()))?;
    let manifest = serde_json::json!({
        "version": 1,
        "source_sha256": sha256_file(input)?,
        "header": header_path.file_name().and_then(|name| name.to_str()).unwrap_or("header.bin"),
        "data_start": original.data_start,
        "file_size": original.file_len,
        "tensors": original.tensors.iter().map(|tensor| serde_json::json!({
            "name": tensor.name,
            "offset": tensor.offset,
            "size": tensor.size,
            "ty": tensor.ty,
            "dims": tensor.dims,
        })).collect::<Vec<_>>(),
    });
    fs::write(&header_path, &original.header)
        .map_err(|error| OpsError::Write(error.to_string()))?;
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest)
            .map_err(|error| OpsError::Metadata(error.to_string()))?,
    )
    .map_err(|error| OpsError::Write(error.to_string()))?;
    Ok(paths)
}

/// Merge shard files. The caller is responsible for passing files of one
/// model; the function sorts them and rejects mismatched shard metadata.
pub fn merge(
    inputs: &[PathBuf],
    output: &Path,
    force: bool,
    delete_inputs: bool,
) -> Result<(), OpsError> {
    if inputs.is_empty() {
        return Err(OpsError::Split("no input shards".into()));
    }
    if let Some(manifest) = find_split_manifest(inputs) {
        return merge_exact(inputs, &manifest, output, force, delete_inputs);
    }
    let mut sorted = inputs.to_vec();
    sorted.sort_by_key(|path| shard_index(path).unwrap_or(usize::MAX));
    let first = content(&sorted[0])?;
    let mut builder = builder_from(&first)?;
    let mut seen = HashSet::new();
    let mut total = 0usize;
    let mut file = File::open(&sorted[0]).map_err(|error| OpsError::Gguf(error.to_string()))?;
    for info in ordered_tensors(&first) {
        let data = read_tensor(&first, &mut file, &info.name)?;
        add_tensor(&mut builder, &info, data)?;
        seen.insert(info.name);
        total += 1;
    }
    for path in sorted.iter().skip(1) {
        let shard = content(path)?;
        let mut file = File::open(path).map_err(|error| OpsError::Gguf(error.to_string()))?;
        for info in ordered_tensors(&shard) {
            if !seen.insert(info.name.clone()) {
                return Err(OpsError::Split(format!(
                    "tensor '{}' appears in more than one shard",
                    info.name
                )));
            }
            let data = read_tensor(&shard, &mut file, &info.name)?;
            add_tensor(&mut builder, &info, data)?;
            total += 1;
        }
    }
    if total == 0 {
        return Err(OpsError::NoTensors);
    }
    finish(&builder, output, force)?;
    if delete_inputs {
        for path in sorted {
            fs::remove_file(path).map_err(|error| OpsError::Write(error.to_string()))?;
        }
    }
    Ok(())
}

fn find_split_manifest(inputs: &[PathBuf]) -> Option<PathBuf> {
    for input in inputs {
        let Some(name) = input.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some((stem, _)) = name.split_once("-00001-of-") else {
            continue;
        };
        let candidate = input
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(format!("{stem}.rustblock-split.json"));
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    let mut directories = HashSet::new();
    for input in inputs {
        if let Some(parent) = input.parent() {
            directories.insert(parent.to_path_buf());
        }
    }
    for directory in directories {
        let Ok(entries) = fs::read_dir(&directory) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(".rustblock-split.json"))
            {
                return Some(path);
            }
        }
    }
    None
}

fn merge_exact(
    inputs: &[PathBuf],
    manifest_path: &Path,
    output: &Path,
    force: bool,
    delete_inputs: bool,
) -> Result<(), OpsError> {
    if output.exists() && !force {
        return Err(OpsError::OutputExists(output.to_path_buf()));
    }
    let manifest: serde_json::Value = serde_json::from_slice(
        &fs::read(manifest_path).map_err(|error| OpsError::Write(error.to_string()))?,
    )
    .map_err(|error| OpsError::Metadata(error.to_string()))?;
    let base = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let header_name = manifest
        .get("header")
        .and_then(|value| value.as_str())
        .ok_or_else(|| OpsError::Split("split manifest has no header".into()))?;
    let header = fs::read(base.join(header_name))
        .map_err(|error| OpsError::Write(format!("cannot read original header: {error}")))?;
    let data_start = manifest
        .get("data_start")
        .and_then(|value| value.as_u64())
        .ok_or_else(|| OpsError::Split("split manifest has no data_start".into()))?;
    let file_size = manifest
        .get("file_size")
        .and_then(|value| value.as_u64())
        .ok_or_else(|| OpsError::Split("split manifest has no file_size".into()))?;
    let tensors = manifest
        .get("tensors")
        .and_then(|value| value.as_array())
        .ok_or_else(|| OpsError::Split("split manifest has no tensors".into()))?;
    let mut shards = Vec::new();
    for path in inputs {
        let content = content(path)?;
        shards.push((path, content));
    }
    let tmp = output.with_extension("tmp.gguf");
    if tmp.exists() {
        fs::remove_file(&tmp).map_err(|error| OpsError::Write(error.to_string()))?;
    }
    let mut file = File::create(&tmp).map_err(|error| OpsError::Write(error.to_string()))?;
    file.write_all(&header)
        .map_err(|error| OpsError::Write(error.to_string()))?;
    file.set_len(file_size)
        .map_err(|error| OpsError::Write(error.to_string()))?;
    for tensor in tensors {
        let name = tensor
            .get("name")
            .and_then(|value| value.as_str())
            .ok_or_else(|| OpsError::Split("manifest tensor has no name".into()))?;
        let offset = tensor
            .get("offset")
            .and_then(|value| value.as_u64())
            .ok_or_else(|| OpsError::Split(format!("manifest tensor '{name}' has no offset")))?;
        let expected = tensor
            .get("size")
            .and_then(|value| value.as_u64())
            .ok_or_else(|| OpsError::Split(format!("manifest tensor '{name}' has no size")))?;
        let mut found = None;
        for (path, content) in &shards {
            if content.tensor_infos.contains_key(name) {
                let mut source =
                    File::open(path).map_err(|error| OpsError::Gguf(error.to_string()))?;
                let data = read_tensor(content, &mut source, name)?;
                found = Some(data);
                break;
            }
        }
        let data = found.ok_or_else(|| OpsError::Split(format!("tensor '{name}' is missing")))?;
        if data.len() as u64 != expected {
            return Err(OpsError::Split(format!(
                "tensor '{name}' has {} bytes, expected {expected}",
                data.len()
            )));
        }
        file.seek(SeekFrom::Start(data_start + offset))
            .map_err(|error| OpsError::Write(error.to_string()))?;
        file.write_all(&data)
            .map_err(|error| OpsError::Write(error.to_string()))?;
    }
    file.sync_all()
        .map_err(|error| OpsError::Write(error.to_string()))?;
    drop(file);
    let actual = sha256_file(&tmp)?;
    let expected_hash = manifest
        .get("source_sha256")
        .and_then(|value| value.as_str())
        .ok_or_else(|| OpsError::Split("split manifest has no source hash".into()))?;
    if !actual.eq_ignore_ascii_case(expected_hash) {
        let _ = fs::remove_file(&tmp);
        return Err(OpsError::Split(format!(
            "merged SHA-256 {actual} does not match original {expected_hash}"
        )));
    }
    if output.exists() {
        fs::remove_file(output).map_err(|error| OpsError::Write(error.to_string()))?;
    }
    fs::rename(&tmp, output).map_err(|error| OpsError::Write(error.to_string()))?;
    if delete_inputs {
        for path in inputs {
            fs::remove_file(path).map_err(|error| OpsError::Write(error.to_string()))?;
        }
    }
    Ok(())
}

fn shard_index(path: &Path) -> Option<usize> {
    let name = path.file_stem()?.to_str()?;
    let (left, _) = name.rsplit_once("-of-")?;
    let (_, index) = left.rsplit_once('-')?;
    index.parse().ok()
}

pub fn metadata_get(path: &Path, key: &str) -> Result<Option<String>, OpsError> {
    let content = content(path)?;
    Ok(content.get_metadata(key).map(format_value))
}

pub fn metadata_set(
    input: &Path,
    output: &Path,
    key: &str,
    value: &str,
    force: bool,
) -> Result<(), OpsError> {
    refuse_same_path(input, output)?;
    let content = content(input)?;
    let parsed = serde_json::from_str::<serde_json::Value>(value)
        .ok()
        .map(|json| json_to_metadata(&json))
        .transpose()?
        .unwrap_or_else(|| MetadataValue::String(value.to_string()));
    let mut builder = builder_from(&content)?;
    builder.add_metadata(key.to_string(), parsed);
    let mut file = File::open(input).map_err(|error| OpsError::Gguf(error.to_string()))?;
    for info in ordered_tensors(&content) {
        let data = read_tensor(&content, &mut file, &info.name)?;
        add_tensor(&mut builder, &info, data)?;
    }
    finish(&builder, output, force)
}

pub fn metadata_list(path: &Path) -> Result<BTreeMap<String, String>, OpsError> {
    let content = content(path)?;
    Ok(content
        .metadata
        .iter()
        .map(|(key, value)| (key.clone(), format_value(value)))
        .collect())
}

/// Validate the metadata that commonly breaks a model at load time: rope
/// parameters, basic architecture dimensions and the structure of the Jinja
/// chat template. This is deliberately a structural check, not a full Jinja
/// evaluator; it catches missing delimiters and unbalanced blocks without
/// executing the template.
pub fn validate_metadata(path: &Path) -> Result<MetadataValidation, OpsError> {
    let content = content(path)?;
    let mut validation = MetadataValidation::default();

    for (key, value) in &content.metadata {
        if key.ends_with(".rope.freq_base") {
            validate_positive_scalar(&mut validation, key, value, "rope frequency base");
        } else if key.ends_with(".rope.scaling.factor") {
            validate_positive_scalar(&mut validation, key, value, "rope scaling factor");
        } else if key.ends_with(".rope.scaling.original_context_length") {
            validate_positive_scalar(&mut validation, key, value, "rope original context length");
        } else if key.ends_with(".rope.dimension_count") {
            validate_positive_scalar(&mut validation, key, value, "rope dimension count");
        } else if key.ends_with(".rope.scaling.type") {
            match value {
                MetadataValue::String(kind)
                    if matches!(
                        kind.as_str(),
                        "none" | "linear" | "yarn" | "longrope" | "llama3"
                    ) => {}
                MetadataValue::String(kind) => validation.issues.push(ValidationIssue {
                    key: key.clone(),
                    message: format!("unsupported rope scaling type '{kind}'"),
                }),
                _ => validation.issues.push(ValidationIssue {
                    key: key.clone(),
                    message: "rope scaling type must be a string".into(),
                }),
            }
        } else if key.ends_with(".context_length") {
            validate_positive_scalar(&mut validation, key, value, "context length");
        } else if key.ends_with(".embedding_length") {
            validate_positive_scalar(&mut validation, key, value, "embedding length");
        } else if key.ends_with(".attention.head_count") {
            validate_positive_values(&mut validation, key, value, "attention head count");
        } else if key.ends_with(".attention.head_count_kv") {
            validate_positive_values(&mut validation, key, value, "KV head count");
        } else if key.ends_with(".attention.key_length") {
            validate_positive_values(&mut validation, key, value, "attention key length");
        } else if key.ends_with(".attention.value_length") {
            validate_positive_values(&mut validation, key, value, "attention value length");
        }
    }

    let architecture = content
        .get_metadata("general.architecture")
        .and_then(metadata_string)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    if architecture.is_none() && content.metadata.contains_key("general.architecture") {
        validation.issues.push(ValidationIssue {
            key: "general.architecture".into(),
            message: "architecture must be a non-empty string".into(),
        });
    }

    if let (Some(arch), Some(heads), Some(kv_heads)) = (
        architecture,
        content
            .get_metadata("general.architecture")
            .and_then(metadata_string)
            .and_then(|arch| content.get_metadata(&format!("{arch}.attention.head_count")))
            .and_then(metadata_number),
        content
            .get_metadata("general.architecture")
            .and_then(metadata_string)
            .and_then(|arch| content.get_metadata(&format!("{arch}.attention.head_count_kv")))
            .and_then(metadata_number),
    ) {
        if kv_heads > heads {
            validation.issues.push(ValidationIssue {
                key: format!("{arch}.attention.head_count_kv"),
                message: format!("KV head count {kv_heads} exceeds attention head count {heads}"),
            });
        }
    }

    if let Some(template) = content.get_metadata("tokenizer.chat_template") {
        match template {
            MetadataValue::String(template) => {
                for message in validate_chat_template(template) {
                    validation.issues.push(ValidationIssue {
                        key: "tokenizer.chat_template".into(),
                        message,
                    });
                }
            }
            _ => validation.issues.push(ValidationIssue {
                key: "tokenizer.chat_template".into(),
                message: "chat template must be a string".into(),
            }),
        }
    }

    Ok(validation)
}

/// Render the model's chat template with a JSON context. This executes the
/// template without tokenizing the result; callers can inspect exactly what
/// the runtime would receive.
pub fn render_chat_template(path: &Path, context: &serde_json::Value) -> Result<String, OpsError> {
    if !context.is_object() {
        return Err(OpsError::Template("context must be a JSON object".into()));
    }
    let content = content(path)?;
    let template = match content.get_metadata("tokenizer.chat_template") {
        Some(MetadataValue::String(template)) => template,
        Some(_) => {
            return Err(OpsError::Template(
                "tokenizer.chat_template must be a string".into(),
            ));
        }
        None => {
            return Err(OpsError::Template(
                "tokenizer.chat_template is missing".into(),
            ));
        }
    };
    let environment = minijinja::Environment::new();
    let compiled = environment
        .template_from_str(template)
        .map_err(|error| OpsError::Template(error.to_string()))?;
    compiled
        .render(context)
        .map_err(|error| OpsError::Template(error.to_string()))
}

fn metadata_string(value: &MetadataValue) -> Option<&str> {
    match value {
        MetadataValue::String(value) => Some(value),
        _ => None,
    }
}

fn metadata_number(value: &MetadataValue) -> Option<f64> {
    match value {
        MetadataValue::Uint8(value) => Some(f64::from(*value)),
        MetadataValue::Int8(value) => Some(f64::from(*value)),
        MetadataValue::Uint16(value) => Some(f64::from(*value)),
        MetadataValue::Int16(value) => Some(f64::from(*value)),
        MetadataValue::Uint32(value) => Some(f64::from(*value)),
        MetadataValue::Int32(value) => Some(f64::from(*value)),
        MetadataValue::Uint64(value) => Some(*value as f64),
        MetadataValue::Int64(value) => Some(*value as f64),
        MetadataValue::Float32(value) => Some(f64::from(*value)),
        MetadataValue::Float64(value) => Some(*value),
        _ => None,
    }
}

fn metadata_numbers(value: &MetadataValue) -> Option<Vec<f64>> {
    match value {
        MetadataValue::Array(values) => values.iter().map(metadata_number).collect(),
        other => metadata_number(other).map(|value| vec![value]),
    }
}

fn validate_positive_scalar(
    validation: &mut MetadataValidation,
    key: &str,
    value: &MetadataValue,
    label: &str,
) {
    match metadata_number(value) {
        Some(number) if number.is_finite() && number > 0.0 => {}
        Some(_) => validation.issues.push(ValidationIssue {
            key: key.to_string(),
            message: format!("{label} must be greater than zero"),
        }),
        None => validation.issues.push(ValidationIssue {
            key: key.to_string(),
            message: format!("{label} must be numeric"),
        }),
    }
}

fn validate_positive_values(
    validation: &mut MetadataValidation,
    key: &str,
    value: &MetadataValue,
    label: &str,
) {
    match metadata_numbers(value) {
        Some(numbers)
            if numbers
                .iter()
                .all(|value| value.is_finite() && *value > 0.0) => {}
        Some(_) => validation.issues.push(ValidationIssue {
            key: key.to_string(),
            message: format!("{label} values must be greater than zero"),
        }),
        None => validation.issues.push(ValidationIssue {
            key: key.to_string(),
            message: format!("{label} must be numeric"),
        }),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum JinjaBlock {
    If,
    For,
    Block,
    Macro,
    Filter,
    Raw,
    Call,
}

impl JinjaBlock {
    fn open(word: &str) -> Option<Self> {
        match word {
            "if" => Some(Self::If),
            "for" => Some(Self::For),
            "block" => Some(Self::Block),
            "macro" => Some(Self::Macro),
            "filter" => Some(Self::Filter),
            "raw" => Some(Self::Raw),
            "call" => Some(Self::Call),
            _ => None,
        }
    }

    fn close(word: &str) -> Option<Self> {
        match word {
            "endif" => Some(Self::If),
            "endfor" => Some(Self::For),
            "endblock" => Some(Self::Block),
            "endmacro" => Some(Self::Macro),
            "endfilter" => Some(Self::Filter),
            "endraw" => Some(Self::Raw),
            "endcall" => Some(Self::Call),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::If => "if",
            Self::For => "for",
            Self::Block => "block",
            Self::Macro => "macro",
            Self::Filter => "filter",
            Self::Raw => "raw",
            Self::Call => "call",
        }
    }
}

fn validate_chat_template(template: &str) -> Vec<String> {
    let mut issues = Vec::new();
    let mut blocks: Vec<(JinjaBlock, usize)> = Vec::new();
    let mut pos = 0usize;
    while pos < template.len() {
        let rest = &template[pos..];
        let line = template[..pos]
            .bytes()
            .filter(|byte| *byte == b'\n')
            .count()
            + 1;
        if let Some(body) = rest.strip_prefix("{{") {
            match body.find("}}") {
                Some(end) => pos += 2 + end + 2,
                None => {
                    issues.push(format!("line {line}: unclosed '{{{{' expression"));
                    break;
                }
            }
            continue;
        }
        if let Some(body) = rest.strip_prefix("{%") {
            let Some(end) = body.find("%}") else {
                issues.push(format!("line {line}: unclosed '{{%' statement"));
                break;
            };
            let statement = body[..end]
                .trim()
                .trim_start_matches('-')
                .trim_end_matches('-')
                .trim();
            let word = statement.split_whitespace().next().unwrap_or("");
            if let Some(block) = JinjaBlock::open(word) {
                blocks.push((block, line));
            } else if let Some(expected) = JinjaBlock::close(word) {
                close_jinja_block(&mut blocks, expected, line, &mut issues);
            } else if matches!(word, "else" | "elif")
                && !matches!(
                    blocks.last().map(|block| block.0),
                    Some(JinjaBlock::If | JinjaBlock::For)
                )
            {
                issues.push(format!("line {line}: '{word}' outside an if/for block"));
            }
            pos += 2 + end + 2;
            continue;
        }
        if let Some(body) = rest.strip_prefix("{#") {
            match body.find("#}") {
                Some(end) => pos += 2 + end + 2,
                None => {
                    issues.push(format!("line {line}: unclosed '{{#' comment"));
                    break;
                }
            }
            continue;
        }
        pos += rest.chars().next().map(char::len_utf8).unwrap_or(1);
    }
    for (block, line) in blocks {
        issues.push(format!("line {line}: unclosed '{}' block", block.name()));
    }
    // The structural pass above gives precise block diagnostics. Compile the
    // template as well so malformed expressions (for example a dangling pipe)
    // are rejected instead of being accepted as opaque text.
    let environment = minijinja::Environment::new();
    if let Err(error) = environment.template_from_str(template) {
        let line = error
            .line()
            .map(|line| format!("line {line}: "))
            .unwrap_or_default();
        let message = format!("{line}Jinja syntax error: {error}");
        if !issues.iter().any(|issue| issue == &message) {
            issues.push(message);
        }
    }
    issues
}

fn close_jinja_block(
    blocks: &mut Vec<(JinjaBlock, usize)>,
    expected: JinjaBlock,
    line: usize,
    issues: &mut Vec<String>,
) {
    match blocks.last().map(|block| block.0) {
        Some(open) if open == expected => {
            blocks.pop();
        }
        Some(open) => issues.push(format!(
            "line {line}: unexpected 'end{}' while '{}' is open",
            open.name(),
            expected.name()
        )),
        None => issues.push(format!(
            "line {line}: unexpected 'end{}' block",
            expected.name()
        )),
    }
}

pub fn prune(input: &Path, output: &Path, pattern: &str, force: bool) -> Result<usize, OpsError> {
    refuse_same_path(input, output)?;
    let content = content(input)?;
    let regex =
        regex::Regex::new(pattern).map_err(|error| OpsError::Metadata(error.to_string()))?;
    let mut builder = builder_from(&content)?;
    let mut file = File::open(input).map_err(|error| OpsError::Gguf(error.to_string()))?;
    let mut removed = 0usize;
    for info in ordered_tensors(&content) {
        if regex.is_match(&info.name) {
            removed += 1;
            continue;
        }
        let data = read_tensor(&content, &mut file, &info.name)?;
        add_tensor(&mut builder, &info, data)?;
    }
    if removed == 0 {
        return Err(OpsError::Metadata("pattern matched no tensors".into()));
    }
    finish(&builder, output, force)?;
    Ok(removed)
}

pub fn diff(a: &Path, b: &Path) -> Result<DiffReport, OpsError> {
    let a_content = content(a)?;
    let b_content = content(b)?;
    let mut report = DiffReport {
        metadata_only_in_a: Vec::new(),
        metadata_only_in_b: Vec::new(),
        metadata_changed: Vec::new(),
        tensors_only_in_a: Vec::new(),
        tensors_only_in_b: Vec::new(),
        tensor_changes: Vec::new(),
    };
    for (key, value) in &a_content.metadata {
        match b_content.metadata.get(key) {
            None => report.metadata_only_in_a.push(key.clone()),
            Some(other) if format_value(value) != format_value(other) => {
                report.metadata_changed.push(key.clone())
            }
            _ => {}
        }
    }
    for key in b_content.metadata.keys() {
        if !a_content.metadata.contains_key(key) {
            report.metadata_only_in_b.push(key.clone());
        }
    }
    for (name, info) in &a_content.tensor_infos {
        match b_content.tensor_infos.get(name) {
            None => report.tensors_only_in_a.push(name.clone()),
            Some(other)
                if info.dtype != other.dtype
                    || info.dimensions != other.dimensions
                    || info.offset != other.offset =>
            {
                report.tensor_changes.push(name.clone())
            }
            _ => {}
        }
    }
    for name in b_content.tensor_infos.keys() {
        if !a_content.tensor_infos.contains_key(name) {
            report.tensors_only_in_b.push(name.clone());
        }
    }
    report.metadata_only_in_a.sort();
    report.metadata_only_in_b.sort();
    report.metadata_changed.sort();
    report.tensors_only_in_a.sort();
    report.tensors_only_in_b.sort();
    report.tensor_changes.sort();
    Ok(report)
}

pub fn sha256_file(path: &Path) -> Result<String, OpsError> {
    let mut file = File::open(path).map_err(|error| OpsError::Gguf(error.to_string()))?;
    let mut hasher = Sha256::new();
    // Keep the buffer on the heap: Windows' default main-thread stack is 1 MiB.
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| OpsError::Gguf(error.to_string()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn verify_hash(path: &Path, expected: &str) -> Result<bool, OpsError> {
    Ok(sha256_file(path)?.eq_ignore_ascii_case(expected.trim()))
}

fn json_to_metadata(value: &serde_json::Value) -> Result<MetadataValue, OpsError> {
    match value {
        serde_json::Value::Bool(value) => Ok(MetadataValue::Bool(*value)),
        serde_json::Value::String(value) => Ok(MetadataValue::String(value.clone())),
        serde_json::Value::Number(number) => {
            if let Some(value) = number.as_u64() {
                Ok(MetadataValue::Uint64(value))
            } else if let Some(value) = number.as_i64() {
                Ok(MetadataValue::Int64(value))
            } else if let Some(value) = number.as_f64() {
                Ok(MetadataValue::Float64(value))
            } else {
                Err(OpsError::Metadata("unsupported number".into()))
            }
        }
        serde_json::Value::Array(values) => {
            let mut out = Vec::with_capacity(values.len());
            for value in values {
                out.push(json_to_metadata(value)?);
            }
            Ok(MetadataValue::Array(out))
        }
        serde_json::Value::Null | serde_json::Value::Object(_) => Err(OpsError::Metadata(
            "null and object values are not supported".into(),
        )),
    }
}

fn format_value(value: &MetadataValue) -> String {
    match value {
        MetadataValue::Uint8(value) => value.to_string(),
        MetadataValue::Int8(value) => value.to_string(),
        MetadataValue::Uint16(value) => value.to_string(),
        MetadataValue::Int16(value) => value.to_string(),
        MetadataValue::Uint32(value) => value.to_string(),
        MetadataValue::Int32(value) => value.to_string(),
        MetadataValue::Uint64(value) => value.to_string(),
        MetadataValue::Int64(value) => value.to_string(),
        MetadataValue::Float32(value) => value.to_string(),
        MetadataValue::Float64(value) => value.to_string(),
        MetadataValue::Bool(value) => value.to_string(),
        MetadataValue::String(value) => value.clone(),
        MetadataValue::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(format_value)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn same_path(a: &Path, b: &Path) -> bool {
    match (fs::canonicalize(a), fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

/// Writing in place would let `finish` delete the very file it is writing
/// from; that is data loss, so it is refused up front.
fn refuse_same_path(input: &Path, output: &Path) -> Result<(), OpsError> {
    if same_path(input, output) {
        return Err(OpsError::Write(format!(
            "input and output are the same file: {}",
            output.display()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_gguf::{GgufBuilder, MetadataValue};

    fn tiny_model(path: &Path) {
        let mut builder = GgufBuilder::new();
        builder.add_f32_tensor("w.weight", vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        builder.add_metadata(
            "general.architecture",
            MetadataValue::String("llama".into()),
        );
        let mut file = File::create(path).unwrap();
        builder.write(&mut file).unwrap();
        file.sync_all().unwrap();
    }

    #[test]
    fn metadata_set_refuses_same_input_and_output() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("model.gguf");
        tiny_model(&model);
        let before = sha256_file(&model).unwrap();
        let error = metadata_set(&model, &model, "tokenizer.chat_template", "\"x\"", true)
            .expect_err("same input and output must be rejected");
        assert!(error.to_string().contains("same file"), "{error}");
        assert_eq!(sha256_file(&model).unwrap(), before);
    }

    #[test]
    fn prune_refuses_same_input_and_output() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("model.gguf");
        tiny_model(&model);
        let before = sha256_file(&model).unwrap();
        let error =
            prune(&model, &model, "w.*", true).expect_err("same input and output must be rejected");
        assert!(error.to_string().contains("same file"), "{error}");
        assert_eq!(sha256_file(&model).unwrap(), before);
    }

    #[test]
    fn metadata_set_to_a_new_file_preserves_the_input() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("model.gguf");
        tiny_model(&model);
        let before = sha256_file(&model).unwrap();
        let output = dir.path().join("model-meta.gguf");
        metadata_set(
            &model,
            &output,
            "tokenizer.chat_template",
            "\"hello\"",
            true,
        )
        .unwrap();
        assert_eq!(sha256_file(&model).unwrap(), before);
        assert!(output.exists());
        assert_eq!(
            metadata_get(&output, "tokenizer.chat_template")
                .unwrap()
                .as_deref(),
            Some("hello")
        );
    }
}
