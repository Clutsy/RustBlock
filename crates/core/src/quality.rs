//! Native per-tensor quality metrics (RMSE, SNR and cosine similarity).
//!
//! Metrics are calculated from decoded weights, not from the compressed byte
//! stream. This means they work across source and target formats, including
//! mixed K-quant outputs where every tensor has a different type.

use std::fs::File;
use std::path::Path;
use std::process::Command;

use pmetal_gguf::dequant;
use pmetal_gguf::{GgmlType, GgufContent, TensorInfo};

use crate::imatrix::{find_tool, ImportanceMatrix};

#[derive(Debug, Clone)]
pub struct TensorQuality {
    pub name: String,
    pub source_type: String,
    pub output_type: String,
    pub elements: usize,
    pub rmse: f64,
    pub snr_db: f64,
    pub cosine: f64,
    pub weighted_snr_db: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct QualityReport {
    pub source: String,
    pub output: String,
    pub tensors: Vec<TensorQuality>,
    pub total_elements: u64,
    pub mean_rmse: f64,
    pub mean_snr_db: f64,
    pub mean_cosine: f64,
    pub weighted_snr_db: Option<f64>,
}

impl QualityReport {
    /// Tensors with the lowest SNR are the most sensitive in this comparison.
    pub fn sensitivity_ranking(&self) -> Vec<&TensorQuality> {
        let mut ranked: Vec<&TensorQuality> = self.tensors.iter().collect();
        ranked.sort_by(|a, b| {
            a.snr_db
                .partial_cmp(&b.snr_db)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked
    }

    /// A compact metric used by the budget planner: higher is better.
    pub fn score(&self) -> f64 {
        self.weighted_snr_db.unwrap_or(self.mean_snr_db)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum QualityError {
    #[error("cannot read GGUF: {0}")]
    Gguf(String),
    #[error("cannot read tensor '{name}': {message}")]
    Tensor { name: String, message: String },
    #[error("tensor '{0}' is missing from the output")]
    MissingTensor(String),
    #[error("no comparable tensors found")]
    NoTensors,
}

pub fn compare(
    source: &Path,
    output: &Path,
    imatrix: Option<&ImportanceMatrix>,
) -> Result<QualityReport, QualityError> {
    let source_content = GgufContent::from_file(source)
        .map_err(|error| QualityError::Gguf(format!("{}: {error}", source.display())))?;
    let output_content = GgufContent::from_file(output)
        .map_err(|error| QualityError::Gguf(format!("{}: {error}", output.display())))?;
    let mut source_file =
        File::open(source).map_err(|error| QualityError::Gguf(error.to_string()))?;
    let mut output_file =
        File::open(output).map_err(|error| QualityError::Gguf(error.to_string()))?;

    let mut names: Vec<&String> = source_content.tensor_infos.keys().collect();
    names.sort();
    let mut tensors = Vec::new();
    let mut total_elements = 0u64;
    let mut sum_rmse = 0.0;
    let mut sum_snr = 0.0;
    let mut sum_cosine = 0.0;
    let mut weighted_signal = 0.0;
    let mut weighted_noise = 0.0;

    for name in names {
        let source_info = match source_content.tensor_infos.get(name) {
            Some(info) => info,
            None => continue,
        };
        let output_info = output_content
            .tensor_infos
            .get(name)
            .ok_or_else(|| QualityError::MissingTensor(name.clone()))?;
        let source_values = decode(&source_content, &mut source_file, name, source_info)?;
        let output_values = decode(&output_content, &mut output_file, name, output_info)?;
        let len = source_values.len().min(output_values.len());
        if len == 0 {
            continue;
        }

        let mut noise = 0.0;
        let mut signal = 0.0;
        let mut dot = 0.0;
        let mut source_norm = 0.0;
        let mut output_norm = 0.0;
        let weights = imatrix
            .and_then(|matrix| matrix.data.get(name))
            .filter(|weights| weights.len() >= len);
        for index in 0..len {
            let a = source_values[index] as f64;
            let b = output_values[index] as f64;
            let diff = a - b;
            let weight = weights
                .map(|weights| weights[index] as f64)
                .unwrap_or(1.0)
                .max(1e-12);
            signal += a * a;
            noise += diff * diff;
            weighted_signal += weight * a * a;
            weighted_noise += weight * diff * diff;
            dot += a * b;
            source_norm += a * a;
            output_norm += b * b;
        }
        let rmse = (noise / len as f64).sqrt();
        let snr_db = snr(signal, noise);
        let cosine = if source_norm > 0.0 && output_norm > 0.0 {
            dot / (source_norm.sqrt() * output_norm.sqrt())
        } else {
            1.0
        };
        let weighted_snr_db = weights.map(|_| snr(weighted_signal, weighted_noise));
        total_elements += len as u64;
        sum_rmse += rmse;
        sum_snr += snr_db;
        sum_cosine += cosine;
        tensors.push(TensorQuality {
            name: name.clone(),
            source_type: format!("{:?}", source_info.dtype),
            output_type: format!("{:?}", output_info.dtype),
            elements: len,
            rmse,
            snr_db,
            cosine,
            weighted_snr_db,
        });
    }

    if tensors.is_empty() {
        return Err(QualityError::NoTensors);
    }
    let count = tensors.len() as f64;
    Ok(QualityReport {
        source: source.display().to_string(),
        output: output.display().to_string(),
        total_elements,
        mean_rmse: sum_rmse / count,
        mean_snr_db: sum_snr / count,
        mean_cosine: sum_cosine / count,
        weighted_snr_db: if weighted_signal > 0.0 {
            Some(snr(weighted_signal, weighted_noise))
        } else {
            None
        },
        tensors,
    })
}

fn snr(signal: f64, noise: f64) -> f64 {
    if noise <= f64::EPSILON {
        f64::INFINITY
    } else if signal <= 0.0 {
        f64::NEG_INFINITY
    } else {
        10.0 * (signal / noise).log10()
    }
}

fn decode(
    content: &GgufContent,
    file: &mut File,
    name: &str,
    info: &TensorInfo,
) -> Result<Vec<f32>, QualityError> {
    let raw = content
        .read_tensor_data(file, name)
        .map_err(|error| QualityError::Tensor {
            name: name.to_string(),
            message: error.to_string(),
        })?;
    if info.dtype == GgmlType::F32 {
        if raw.len() % 4 != 0 {
            return Err(QualityError::Tensor {
                name: name.to_string(),
                message: "invalid F32 byte length".to_string(),
            });
        }
        return Ok(raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect());
    }
    let shape: Vec<i32> = info
        .dimensions
        .iter()
        .map(|dim| {
            i32::try_from(*dim).map_err(|_| QualityError::Tensor {
                name: name.to_string(),
                message: "dimension does not fit in i32".to_string(),
            })
        })
        .collect::<Result<_, _>>()?;
    dequant::dequantize(&raw, info.dtype, &shape).map_err(|error| QualityError::Tensor {
        name: name.to_string(),
        message: error.to_string(),
    })
}

/// Quote and escape a text field for CSV. Tensor names come from model files
/// and may contain commas, quotes or newlines; an unescaped field would split
/// rows or corrupt the report.
fn csv_field(text: &str) -> String {
    if text.contains(',') || text.contains('"') || text.contains('\n') || text.contains('\r') {
        format!("\"{}\"", text.replace('"', "\"\""))
    } else {
        text.to_string()
    }
}

/// Write the per-tensor report as CSV without pulling a CSV dependency into
/// the core library. The CLI passes the resulting string to stdout/file.
pub fn to_csv(report: &QualityReport) -> String {
    let mut out = String::from(
        "tensor,source_type,output_type,elements,rmse,snr_db,cosine,weighted_snr_db\n",
    );
    for tensor in &report.tensors {
        out.push_str(&format!(
            "{},{},{},{},{:.8},{:.8},{:.8},{}\n",
            csv_field(&tensor.name),
            csv_field(&tensor.source_type),
            csv_field(&tensor.output_type),
            tensor.elements,
            tensor.rmse,
            tensor.snr_db,
            tensor.cosine,
            tensor
                .weighted_snr_db
                .map(|value| format!("{value:.8}"))
                .unwrap_or_default()
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_fields_escape_commas_quotes_and_newlines() {
        assert_eq!(csv_field("plain.weight"), "plain.weight");
        assert_eq!(csv_field("a,b.weight"), "\"a,b.weight\"");
        assert_eq!(csv_field("a\"b"), "\"a\"\"b\"");
        assert_eq!(csv_field("a\nb"), "\"a\nb\"");
    }

    #[test]
    fn to_csv_keeps_rows_for_tensor_names_with_commas() {
        let report = QualityReport {
            source: "a.gguf".into(),
            output: "b.gguf".into(),
            tensors: vec![TensorQuality {
                name: "blk.0.a, b.weight".into(),
                source_type: "F32".into(),
                output_type: "F16".into(),
                elements: 8,
                rmse: 0.0,
                snr_db: f64::INFINITY,
                cosine: 1.0,
                weighted_snr_db: None,
            }],
            total_elements: 8,
            mean_rmse: 0.0,
            mean_snr_db: f64::INFINITY,
            mean_cosine: 1.0,
            weighted_snr_db: None,
        };
        let csv = to_csv(&report);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[1].starts_with("\"blk.0.a, b.weight\",F32,F16,8,"));
    }
}

#[derive(Debug, Clone)]
pub struct KldReport {
    pub command: String,
    pub stdout: String,
    pub stderr: String,
}

#[derive(Debug, thiserror::Error)]
pub enum KldError {
    #[error("llama-perplexity was not found; set LLAMA_PERPLEXITY or install llama.cpp tools")]
    MissingTool,
    #[error("KLD needs either a base logits file (--base) or --record")]
    MissingMode,
    #[error("llama-perplexity failed: {0}")]
    Failed(String),
}

/// Run the external llama.cpp KL-divergence workflow.
///
/// With `record`, the command writes reference logits from the source model:
/// `llama-perplexity -m source -f data --chunks N --kl-divergence-base ref.kld`.
/// With `base`, it compares a quantized model against those reference logits by
/// adding `--kl-divergence`. RustBlock never fabricates these huge logit files.
pub fn run_kld(
    model: &Path,
    calibration: &Path,
    chunks: u64,
    base: Option<&Path>,
    record: Option<&Path>,
) -> Result<KldReport, KldError> {
    let tool = find_tool("llama-perplexity").ok_or(KldError::MissingTool)?;
    let (logits, compare) = match (base, record) {
        (Some(base), _) => (base, true),
        (None, Some(record)) => (record, false),
        (None, None) => return Err(KldError::MissingMode),
    };
    let mut command = Command::new(&tool);
    command
        .arg("-m")
        .arg(model)
        .arg("-f")
        .arg(calibration)
        .arg("--chunks")
        .arg(chunks.max(1).to_string())
        .arg("--kl-divergence-base")
        .arg(logits);
    if compare {
        command.arg("--kl-divergence");
    }
    let output = command
        .output()
        .map_err(|error| KldError::Failed(error.to_string()))?;
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    if !output.status.success() {
        return Err(KldError::Failed(format!(
            "{}: {}",
            output.status,
            stderr.trim()
        )));
    }
    let command_line = format!(
        "{} -m {} -f {} --chunks {}{} --kl-divergence-base {}",
        tool.display(),
        model.display(),
        calibration.display(),
        chunks.max(1),
        if compare { " --kl-divergence" } else { "" },
        logits.display()
    );
    Ok(KldReport {
        command: command_line,
        stdout,
        stderr,
    })
}
