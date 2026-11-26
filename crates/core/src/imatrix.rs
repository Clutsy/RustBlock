//! Importance-matrix loading, type selection and optional external generation.
//!
//! RustBlock reads both llama.cpp's legacy `.imatrix`/`.dat` format and a
//! GGUF container whose tensors hold the importance values. It never runs
//! calibration itself: `llama-imatrix` is an optional external tool and is
//! only invoked when the user explicitly asks for it.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;

use pmetal_gguf::dequant;
use pmetal_gguf::imatrix::IMatrix;
use pmetal_gguf::{GgmlType, GgufContent};

use crate::gguf::TensorInfo;
use crate::recipe::{exact_name_pattern, TensorOverride};
use crate::transcode::Target;

#[derive(Debug, thiserror::Error)]
pub enum ImatrixError {
    #[error("cannot read importance matrix: {0}")]
    Io(String),
    #[error("invalid importance matrix: {0}")]
    Invalid(String),
    #[error("llama-imatrix was not found; set LLAMA_IMATRIX or install llama.cpp tools")]
    MissingGenerator,
    #[error("llama-imatrix failed: {0}")]
    Generator(String),
    #[error("cannot build a tensor override: {0}")]
    Override(String),
}

/// Parsed importance values. `data` is keyed by the exact GGUF tensor name.
#[derive(Debug, Clone)]
pub struct ImportanceMatrix {
    pub data: BTreeMap<String, Vec<f32>>,
    pub ncalls: BTreeMap<String, i32>,
    pub dataset: Option<String>,
}

impl ImportanceMatrix {
    pub fn load(path: &Path) -> Result<Self, ImatrixError> {
        if path
            .extension()
            .is_some_and(|ext| ext.to_string_lossy().eq_ignore_ascii_case("gguf"))
        {
            return Self::load_gguf(path);
        }
        let matrix = IMatrix::load(path).map_err(|error| ImatrixError::Io(error.to_string()))?;
        Ok(Self {
            data: matrix.data.into_iter().collect(),
            ncalls: matrix.ncalls.into_iter().collect(),
            dataset: matrix.dataset_name,
        })
    }

    fn load_gguf(path: &Path) -> Result<Self, ImatrixError> {
        let content =
            GgufContent::from_file(path).map_err(|error| ImatrixError::Io(error.to_string()))?;
        let mut file = File::open(path).map_err(|error| ImatrixError::Io(error.to_string()))?;
        let mut data = BTreeMap::new();
        for (name, info) in &content.tensor_infos {
            let raw = content
                .read_tensor_data(&mut file, name)
                .map_err(|error| ImatrixError::Invalid(error.to_string()))?;
            let shape: Vec<i32> = info
                .dimensions
                .iter()
                .map(|dim| {
                    i32::try_from(*dim)
                        .map_err(|_| ImatrixError::Invalid(format!("tensor '{name}' is too large")))
                })
                .collect::<Result<_, _>>()?;
            let values = if info.dtype == GgmlType::F32 {
                if raw.len() % 4 != 0 {
                    return Err(ImatrixError::Invalid(format!(
                        "F32 importance tensor '{name}' has an invalid byte length"
                    )));
                }
                raw.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|chunk| f32::from_le_bytes(*chunk))
                    .collect()
            } else {
                dequant::dequantize(&raw, info.dtype, &shape)
                    .map_err(|error| ImatrixError::Invalid(format!("tensor '{name}': {error}")))?
            };
            data.insert(name.clone(), values);
        }
        Ok(Self {
            data,
            ncalls: BTreeMap::new(),
            dataset: None,
        })
    }

    /// Save in the legacy format, which is the format understood by
    /// `llama-quantize` and by the rest of llama.cpp.
    pub fn save(&self, path: &Path) -> Result<(), ImatrixError> {
        let mut matrix = IMatrix::new();
        for (name, values) in &self.data {
            matrix.insert(
                name.clone(),
                values.clone(),
                self.ncalls.get(name).copied().unwrap_or(1),
            );
        }
        matrix.dataset_name = self.dataset.clone();
        matrix
            .save(path)
            .map_err(|error| ImatrixError::Io(error.to_string()))
    }

    pub fn mean_importance(&self, name: &str) -> Option<f32> {
        self.data.get(name).map(|values| {
            if values.is_empty() {
                0.0
            } else {
                values.iter().copied().sum::<f32>() / values.len() as f32
            }
        })
    }

    pub fn total_importance(&self, name: &str) -> Option<f32> {
        self.data
            .get(name)
            .map(|values| values.iter().copied().sum())
    }

    /// Select the tensors in the top `percentile` of importance and turn them
    /// into explicit high-precision overrides. This intentionally composes
    /// with (and never overrides) the user's own regex rules.
    pub fn type_overrides(
        &self,
        tensors: &[TensorInfo],
        target: Target,
        percentile: f32,
    ) -> Result<Vec<TensorOverride>, ImatrixError> {
        let Some(high) = high_precision_type(target) else {
            return Ok(Vec::new());
        };
        let percentile = percentile.clamp(0.0, 1.0);
        if percentile <= 0.0 || self.data.is_empty() {
            return Ok(Vec::new());
        }

        let mut ranked: Vec<(String, f32)> = tensors
            .iter()
            .filter_map(|tensor| {
                self.total_importance(&tensor.name)
                    .map(|score| (tensor.name.clone(), score))
            })
            .collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        let count = ((ranked.len() as f32 * percentile).ceil() as usize).clamp(1, ranked.len());
        ranked.truncate(count);

        let type_name = high.short().to_ascii_lowercase();
        let mut overrides = Vec::with_capacity(ranked.len());
        for (name, _) in ranked {
            let spec = format!("{}={}", exact_name_pattern(&name), type_name);
            overrides.push(
                TensorOverride::parse(&spec)
                    .map_err(|error| ImatrixError::Override(error.to_string()))?,
            );
        }
        Ok(overrides)
    }
}

/// The precision used for tensors selected by an importance matrix.
pub fn high_precision_type(target: Target) -> Option<Target> {
    match target {
        Target::Q2K => Some(Target::Q4K),
        Target::Q3K | Target::Q3KM => Some(Target::Q5K),
        Target::Q4K | Target::Q4KM | Target::Q5K | Target::Q5KM => Some(Target::Q6K),
        _ => None,
    }
}

/// Find an executable without shell interpolation. `RUSTBLOCK_<NAME>` and
/// `LLAMA_<NAME>` take precedence over `PATH`.
pub fn find_tool(name: &str) -> Option<PathBuf> {
    let upper = name.to_ascii_uppercase().replace('-', "_");
    for key in [
        format!("RUSTBLOCK_{upper}"),
        upper.clone(),
        format!("LLAMA_{upper}"),
    ] {
        if let Some(path) = std::env::var_os(key) {
            let path = PathBuf::from(path);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        for suffix in if cfg!(windows) {
            ["", ".exe"]
        } else {
            ["", ""]
        } {
            let candidate = dir.join(format!("{name}{suffix}"));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

/// Run `llama-imatrix` on a calibration text file. The inference itself is
/// deliberately external; RustBlock only manages the process and output file.
pub fn generate_with_llama_imatrix(
    model: &Path,
    calibration: &Path,
    output: &Path,
    chunks: usize,
) -> Result<PathBuf, ImatrixError> {
    let tool = find_tool("llama-imatrix").ok_or(ImatrixError::MissingGenerator)?;
    let status = Command::new(&tool)
        .arg("-m")
        .arg(model)
        .arg("-f")
        .arg(calibration)
        .arg("-o")
        .arg(output)
        .arg("--chunks")
        .arg(chunks.max(1).to_string())
        .status()
        .map_err(|error| ImatrixError::Generator(format!("{}: {error}", tool.display())))?;
    if !status.success() {
        return Err(ImatrixError::Generator(format!(
            "{} exited with status {status}",
            tool.display()
        )));
    }
    if !output.is_file() {
        return Err(ImatrixError::Generator(format!(
            "{} did not create {}",
            tool.display(),
            output.display()
        )));
    }
    Ok(output.to_path_buf())
}

pub const OVERFITTING_WARNING: &str =
    "An importance matrix is only as representative as its calibration text. \
     Wikipedia-like text can overfit a specific domain; use samples that match \
     the prompts and language you will use at inference time.";
