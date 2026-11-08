//! RAM/VRAM fit estimation for GGUF models.
//!
//! The estimate is deliberately explicit: weights are exact from the tensor
//! table, KV cache is computed from architecture metadata, and runtime
//! overhead is a separate configurable term. Unknown hardware is reported as
//! unknown rather than turned into a false "fits" verdict.

use std::path::Path;
use std::process::Command;

use pmetal_gguf::{GgufContent, MetadataValue};

use crate::gguf::Gguf;
use crate::transcode::Target;

pub const MIB: u64 = 1024 * 1024;

#[derive(Debug, Clone, Copy, Default)]
pub struct Hardware {
    pub ram_bytes: Option<u64>,
    pub vram_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
pub struct FitRequest {
    pub context: Option<u64>,
    pub cache_element_bytes: u64,
    pub overhead_bytes: u64,
    pub hardware: Option<Hardware>,
}

impl Default for FitRequest {
    fn default() -> Self {
        Self {
            context: None,
            cache_element_bytes: 2,
            overhead_bytes: 512 * MIB,
            hardware: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FitEstimate {
    pub path: String,
    pub architecture: String,
    pub layers: u64,
    pub context: u64,
    pub kv_heads: u64,
    pub head_bytes: u64,
    pub weights_bytes: u64,
    pub dense_kv_bytes: u64,
    pub sliding_kv_bytes: u64,
    pub mla_kv_bytes: u64,
    pub ssm_state_bytes: u64,
    pub kv_cache_bytes: u64,
    pub overhead_bytes: u64,
    pub total_bytes: u64,
    pub hardware: Hardware,
    pub fits_ram: Option<bool>,
    pub fits_vram: Option<bool>,
    pub suggested_target: Target,
}

#[derive(Debug, thiserror::Error)]
pub enum FitError {
    #[error("cannot read GGUF: {0}")]
    Gguf(String),
    #[error("invalid architecture metadata: {0}")]
    Metadata(String),
}

pub fn estimate(path: &Path, request: FitRequest) -> Result<FitEstimate, FitError> {
    let gguf = Gguf::open(path).map_err(|error| FitError::Gguf(error.to_string()))?;
    let content =
        GgufContent::from_file(path).map_err(|error| FitError::Gguf(error.to_string()))?;
    let architecture = content.architecture().unwrap_or("unknown").to_string();
    let meta = |suffix: &str| metadata_u64(&content, &format!("{architecture}.{suffix}"));
    let meta_vec = |suffix: &str| metadata_vec(&content, &format!("{architecture}.{suffix}"));

    let layers = meta("block_count").unwrap_or(0);
    let context = request
        .context
        .or_else(|| meta("context_length"))
        .unwrap_or(4096)
        .max(1);
    let n_head = meta("attention.head_count").unwrap_or(1).max(1);
    let kv_heads = meta_vec("attention.head_count_kv")
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| vec![n_head; layers as usize]);
    let embedding = meta("embedding_length").unwrap_or(0);
    let key_lengths = meta_vec("attention.key_length")
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| vec![embedding / n_head; layers as usize]);
    let value_lengths = meta_vec("attention.value_length")
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| vec![embedding / n_head; layers as usize]);
    let head_bytes = key_lengths
        .first()
        .copied()
        .unwrap_or(0)
        .saturating_add(value_lengths.first().copied().unwrap_or(0))
        .saturating_mul(request.cache_element_bytes.max(1));
    let sliding_window = meta("attention.sliding_window").unwrap_or(0);
    let sliding_pattern = meta_vec("attention.sliding_window_pattern");
    let mla_rank = meta("attention.kv_lora_rank").unwrap_or(0);
    let mla_rope = meta("attention.qk_rope_head_dim").unwrap_or(0);

    let mut dense_kv_bytes = 0u64;
    let mut sliding_kv_bytes = 0u64;
    let mut mla_kv_bytes = 0u64;
    for layer in 0..layers {
        if mla_rank > 0 {
            mla_kv_bytes = mla_kv_bytes.saturating_add(
                context
                    .saturating_mul(mla_rank.saturating_add(mla_rope))
                    .saturating_mul(request.cache_element_bytes.max(1)),
            );
            continue;
        }
        let uses_sliding = sliding_window > 0
            && sliding_pattern
                .as_ref()
                .and_then(|pattern| pattern.get(layer as usize))
                .copied()
                .map(|value| value != 0)
                .unwrap_or(true);
        let layer_context = if uses_sliding {
            context.min(sliding_window)
        } else {
            context
        };
        let heads = kv_heads
            .get(layer as usize)
            .copied()
            .unwrap_or(n_head)
            .min(n_head);
        let key = key_lengths
            .get(layer as usize)
            .copied()
            .unwrap_or(embedding / n_head);
        let value = value_lengths
            .get(layer as usize)
            .copied()
            .unwrap_or(embedding / n_head);
        let bytes = layer_context
            .saturating_mul(heads)
            .saturating_mul(key.saturating_add(value))
            .saturating_mul(request.cache_element_bytes.max(1));
        if uses_sliding {
            sliding_kv_bytes = sliding_kv_bytes.saturating_add(bytes);
        } else {
            dense_kv_bytes = dense_kv_bytes.saturating_add(bytes);
        }
    }

    // Mamba/SSM and hybrid models keep recurrent state in addition to (or
    // instead of) the attention cache. The metadata names differ between
    // architectures; support the common aliases and keep this term separate so
    // a dense model cannot accidentally receive an SSM estimate.
    let ssm_inner = meta("ssm.inner_size")
        .or_else(|| meta("ssm.d_inner"))
        .unwrap_or(0);
    let ssm_state = meta("ssm.state_size")
        .or_else(|| meta("ssm.d_state"))
        .unwrap_or(0);
    let ssm_conv = meta("ssm.conv_kernel")
        .or_else(|| meta("ssm.d_conv"))
        .unwrap_or(0);
    let ssm_dt = meta("ssm.time_step_rank")
        .or_else(|| meta("ssm.dt_rank"))
        .unwrap_or(0);
    let ssm_layers = meta_vec("ssm.layer_indices")
        .map(|values| values.len() as u64)
        .or_else(|| meta("ssm.layer_count"))
        .unwrap_or({
            if ssm_inner > 0 || ssm_state > 0 {
                layers
            } else {
                0
            }
        });
    let ssm_per_layer = ssm_inner
        .saturating_mul(ssm_state)
        .saturating_add(ssm_inner.saturating_mul(ssm_conv))
        .saturating_add(ssm_inner.saturating_mul(ssm_dt));
    let ssm_state_bytes = ssm_layers.saturating_mul(ssm_per_layer).saturating_mul(4);

    let kv_cache_bytes = dense_kv_bytes
        .saturating_add(sliding_kv_bytes)
        .saturating_add(mla_kv_bytes)
        .saturating_add(ssm_state_bytes);
    let weights_bytes = gguf
        .tensors
        .iter()
        .filter_map(|tensor| tensor.size)
        .fold(0u64, |total, size| total.saturating_add(size));
    let total_bytes = weights_bytes
        .saturating_add(kv_cache_bytes)
        .saturating_add(request.overhead_bytes);
    let hardware = request.hardware.unwrap_or_default();
    let fits_ram = hardware
        .ram_bytes
        .map(|available| hardware_fits(total_bytes, available));
    let fits_vram = hardware
        .vram_bytes
        .map(|available| hardware_fits(total_bytes, available));
    let suggested_target = suggest_target(hardware.ram_bytes.or(hardware.vram_bytes));

    Ok(FitEstimate {
        path: path.display().to_string(),
        architecture,
        layers,
        context,
        kv_heads: kv_heads.first().copied().unwrap_or(1).max(1),
        head_bytes,
        weights_bytes,
        dense_kv_bytes,
        sliding_kv_bytes,
        mla_kv_bytes,
        ssm_state_bytes,
        kv_cache_bytes,
        overhead_bytes: request.overhead_bytes,
        total_bytes,
        hardware,
        fits_ram,
        fits_vram,
        suggested_target,
    })
}

fn metadata_u64(content: &GgufContent, key: &str) -> Option<u64> {
    match content.get_metadata(key)? {
        MetadataValue::Uint8(v) => Some(*v as u64),
        MetadataValue::Int8(v) => u64::try_from(*v).ok(),
        MetadataValue::Uint16(v) => Some(*v as u64),
        MetadataValue::Int16(v) => u64::try_from(*v).ok(),
        MetadataValue::Uint32(v) => Some(*v as u64),
        MetadataValue::Int32(v) => u64::try_from(*v).ok(),
        MetadataValue::Uint64(v) => Some(*v),
        MetadataValue::Int64(v) => u64::try_from(*v).ok(),
        MetadataValue::Array(values) => values.first().and_then(|value| match value {
            MetadataValue::Uint32(v) => Some(*v as u64),
            MetadataValue::Uint64(v) => Some(*v),
            _ => None,
        }),
        _ => None,
    }
}

fn metadata_vec(content: &GgufContent, key: &str) -> Option<Vec<u64>> {
    match content.get_metadata(key)? {
        MetadataValue::Array(values) => Some(
            values
                .iter()
                .filter_map(|value| match value {
                    MetadataValue::Uint8(v) => Some(*v as u64),
                    MetadataValue::Uint32(v) => Some(*v as u64),
                    MetadataValue::Uint64(v) => Some(*v),
                    MetadataValue::Int32(v) => u64::try_from(*v).ok(),
                    MetadataValue::Int64(v) => u64::try_from(*v).ok(),
                    _ => None,
                })
                .collect(),
        ),
        _ => metadata_u64(content, key).map(|value| vec![value]),
    }
}

pub fn suggest_target(available_bytes: Option<u64>) -> Target {
    match available_bytes {
        Some(bytes) if bytes >= 8 * 1024 * MIB * 1024 => Target::Q4KM,
        Some(bytes) if bytes >= 6 * 1024 * MIB * 1024 => Target::Q3KM,
        Some(_) => Target::Q2K,
        None => Target::Q4KM,
    }
}

/// Leave a small safety margin for the runtime, allocator and driver overhead.
pub fn hardware_fits(total_bytes: u64, available_bytes: u64) -> bool {
    total_bytes <= available_bytes.saturating_mul(95) / 100
}

/// Best-effort hardware discovery. It only reads system information and
/// optional `nvidia-smi`; it never installs or changes anything.
pub fn detect_hardware() -> Hardware {
    Hardware {
        ram_bytes: detect_ram(),
        vram_bytes: detect_vram(),
    }
}

#[cfg(windows)]
fn detect_ram() -> Option<u64> {
    let output = Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout).trim().parse().ok()
}

#[cfg(target_os = "linux")]
fn detect_ram() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(target_os = "macos")]
fn detect_ram() -> Option<u64> {
    let output = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&output.stdout).trim().parse().ok()
}

#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
fn detect_ram() -> Option<u64> {
    None
}

fn detect_vram() -> Option<u64> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let mib: u64 = text.lines().next()?.trim().parse().ok()?;
    Some(mib.saturating_mul(MIB))
}

/// Human-readable bytes for CLI and UI reports.
pub fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}
