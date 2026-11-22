//! UI data model, analysis helpers and filesystem utilities.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use rustblock_core::{
    estimate_fit, source_supported, type_name, FitEstimate, FitRequest, Gguf, Hardware, Phase,
    Report, Target, VerifyMode, TY_PTQ1_0, TY_Q2_0,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileKind {
    Pq2,
    Q2,
    Float,
    Quantized,
    Mixed,
    Ptq1,
    Other,
    Invalid,
}

#[derive(Debug, Clone)]
pub struct Analysis {
    pub version: u32,
    pub file_len: u64,
    pub alignment: u64,
    pub tensors: usize,
    pub pq2: usize,
    pub ptq1: usize,
    pub types: Vec<(u32, usize)>,
    pub supported: bool,
    pub kind: FileKind,
    pub fit: Option<FitEstimate>,
    pub error: Option<String>,
}

impl Analysis {
    pub fn invalid(file_len: u64, error: String) -> Self {
        Self {
            version: 0,
            file_len,
            alignment: 0,
            tensors: 0,
            pq2: 0,
            ptq1: 0,
            types: Vec::new(),
            supported: false,
            kind: FileKind::Invalid,
            fit: None,
            error: Some(error),
        }
    }

    pub fn can_target(&self, target: Target) -> bool {
        if self.error.is_some() || self.ptq1 > 0 || !self.supported {
            return false;
        }
        if target == Target::Reblock {
            self.pq2 > 0
        } else {
            true
        }
    }

    pub fn default_target(&self) -> Target {
        if self.pq2 > 0 {
            Target::Reblock
        } else if matches!(self.kind, FileKind::Float) {
            Target::Q4KM
        } else if matches!(self.kind, FileKind::Quantized | FileKind::Mixed) {
            Target::F16
        } else {
            Target::Q4KM
        }
    }

    pub fn type_summary(&self) -> String {
        let mut parts = Vec::new();
        for (ty, count) in self.types.iter().take(4) {
            parts.push(format!("{} x{}", type_name(*ty), count));
        }
        if self.types.len() > 4 {
            parts.push(format!("+{} types", self.types.len() - 4));
        }
        if parts.is_empty() {
            "no tensors".to_string()
        } else {
            parts.join(" · ")
        }
    }

    pub fn kind_label(&self) -> &'static str {
        match self.kind {
            FileKind::Pq2 => "PQ2_0 ternary",
            FileKind::Q2 => "Q2_0 ternary",
            FileKind::Float => "float GGUF",
            FileKind::Quantized => "quantized GGUF",
            FileKind::Mixed => "mixed GGUF",
            FileKind::Ptq1 => "PTQ1_0 (unsupported)",
            FileKind::Other => "other GGUF",
            FileKind::Invalid => "invalid file",
        }
    }

    /// True when the file already contains compressed weights. Requantizing
    /// such a file cannot recover information that was lost earlier.
    pub fn is_quantized(&self) -> bool {
        matches!(
            self.kind,
            FileKind::Pq2 | FileKind::Q2 | FileKind::Quantized | FileKind::Mixed | FileKind::Ptq1
        )
    }
}

/// Best-effort detection from a file name. Hugging Face exposes sibling names
/// without downloading the tensors, so this is used for the pre-download
/// warning; the actual tensor types are checked again after the file arrives.
pub fn looks_quantized_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .any(|token| {
            let bytes = token.as_bytes();
            let starts_q_digit = bytes.len() > 1 && bytes[0] == b'q' && bytes[1].is_ascii_digit();
            let starts_iq_digit =
                bytes.len() > 2 && &bytes[..2] == b"iq" && bytes[2].is_ascii_digit();
            let starts_tq_digit =
                bytes.len() > 2 && &bytes[..2] == b"tq" && bytes[2].is_ascii_digit();
            starts_q_digit
                || starts_iq_digit
                || starts_tq_digit
                || matches!(token, "mxfp4" | "nvfp4" | "pq2" | "ptq1")
        })
}

pub fn analyze(path: &Path, hardware: Hardware) -> Analysis {
    let file_len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let g = match Gguf::open(path) {
        Ok(g) => g,
        Err(e) => return Analysis::invalid(file_len, e.to_string()),
    };

    let mut counts: BTreeMap<u32, usize> = BTreeMap::new();
    for t in &g.tensors {
        *counts.entry(t.ty).or_insert(0) += 1;
    }
    let types: Vec<(u32, usize)> = counts.iter().map(|(k, v)| (*k, *v)).collect();
    let supported = g.tensors.iter().all(|t| source_supported(t.ty));
    let pq2 = g.pq2_count();
    let ptq1 = g.tensors.iter().filter(|t| t.ty == TY_PTQ1_0).count();
    let q2 = g.tensors.iter().filter(|t| t.ty == TY_Q2_0).count();
    let all_float = g.tensors.iter().all(|t| matches!(t.ty, 0 | 1 | 30));
    let any_quant = g.tensors.iter().any(|t| {
        matches!(
            t.ty,
            2 | 3
                | 6
                | 7
                | 8
                | 9
                | 10
                | 11
                | 12
                | 13
                | 14
                | 15
                | 16
                | 17
                | 18
                | 19
                | 20
                | 21
                | 22
                | 23
                | 29
                | 34
                | 35
                | 39
                | 40
                | 41
                | 42
        )
    });

    let kind = if ptq1 > 0 {
        FileKind::Ptq1
    } else if pq2 > 0 {
        FileKind::Pq2
    } else if !supported {
        FileKind::Other
    } else if q2 > 0 && types.len() == 1 {
        FileKind::Q2
    } else if q2 > 0 {
        FileKind::Mixed
    } else if all_float {
        FileKind::Float
    } else if any_quant {
        FileKind::Quantized
    } else {
        FileKind::Other
    };

    let fit = rustblock_core::estimate_fit(
        path,
        FitRequest {
            hardware: Some(hardware),
            ..FitRequest::default()
        },
    )
    .ok();

    Analysis {
        version: g.version,
        file_len: g.file_len,
        alignment: g.alignment,
        tensors: g.tensors.len(),
        pq2,
        ptq1,
        types,
        supported,
        kind,
        fit,
        error: None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyKind {
    Exact,
    Structure,
}

#[derive(Debug, Clone)]
pub struct VerifyOutcome {
    pub tensors: usize,
    pub kind: VerifyKind,
    pub label: String,
}

#[derive(Debug, Clone)]
pub struct RunState {
    pub phase: Phase,
    pub done: u64,
    pub total: u64,
    pub speed: f64,
    pub current_tensor: String,
    pub tensor_index: usize,
    pub tensors_total: usize,
    last_time: Instant,
    last_done: u64,
}

impl RunState {
    pub fn new(total: u64) -> Self {
        Self {
            phase: Phase::Converting,
            done: 0,
            total: total.max(1),
            speed: 0.0,
            current_tensor: String::new(),
            tensor_index: 0,
            tensors_total: 0,
            last_time: Instant::now(),
            last_done: 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        phase: Phase,
        done: u64,
        total: u64,
        tensor_index: usize,
        tensor_name: String,
        tensors_total: usize,
    ) {
        let now = Instant::now();
        let reset = phase != self.phase || done < self.last_done;
        if reset {
            self.speed = 0.0;
            self.last_done = done;
            self.last_time = now;
        } else {
            let dt = now.duration_since(self.last_time).as_secs_f64();
            if dt >= 0.02 {
                let inst = (done - self.last_done) as f64 / dt;
                self.speed = if self.speed <= 0.0 {
                    inst
                } else {
                    self.speed * 0.78 + inst * 0.22
                };
                self.last_done = done;
                self.last_time = now;
            }
        }
        self.phase = phase;
        self.done = done;
        self.total = total.max(1);
        self.current_tensor = tensor_name;
        self.tensor_index = tensor_index;
        self.tensors_total = tensors_total;
    }

    pub fn fraction(&self) -> f32 {
        (self.done as f64 / self.total as f64).clamp(0.0, 1.0) as f32
    }

    pub fn eta(&self) -> Option<Duration> {
        if self.speed < 1.0 || self.done >= self.total {
            return None;
        }
        let secs = (self.total - self.done) as f64 / self.speed;
        if secs.is_finite() && secs < 86_400.0 * 30.0 {
            Some(Duration::from_secs_f64(secs))
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub enum JobStatus {
    Ready,
    Running(RunState),
    Done {
        report: Report,
        target: Target,
    },
    Verified {
        outcome: VerifyOutcome,
        elapsed: Duration,
    },
    Failed(String),
}

#[derive(Debug, Clone)]
pub struct Job {
    pub id: u64,
    pub input: PathBuf,
    pub analysis: Analysis,
    pub status: JobStatus,
    pub target: Target,
}

impl Job {
    pub fn file_name(&self) -> String {
        self.input
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| self.input.display().to_string())
    }

    /// The output path the job will produce (or produced), given the mode.
    pub fn planned_output(&self, mode: &OutputMode) -> PathBuf {
        output_path(&self.input, self.target, mode)
    }

    /// The output path of a finished job, if it is done and the file exists.
    pub fn done_output(&self) -> Option<PathBuf> {
        match &self.status {
            JobStatus::Done { report, .. } => report.output.exists().then(|| report.output.clone()),
            _ => None,
        }
    }
}

/// A loaded GGUF for the inspector view: facts, metadata, tensor histogram
/// and the memory estimate. `open` returns `None` for unreadable files.
#[derive(Debug, Clone)]
pub struct InspectorData {
    pub path: PathBuf,
    pub version: u32,
    pub file_len: u64,
    pub alignment: u64,
    pub data_start: u64,
    pub tensors: usize,
    pub pq2: usize,
    pub ptq1: usize,
    pub file_type: Option<u32>,
    /// Up to 12 `(key, value)` metadata entries, in header order.
    pub metadata: Vec<(String, String)>,
    /// `(type name, count, bytes on disk)` sorted by disk size, biggest first.
    pub histogram: Vec<(String, usize, u64)>,
    pub weights_bytes: u64,
    pub fit: Option<FitEstimate>,
}

impl InspectorData {
    pub fn open(path: &Path) -> Option<Self> {
        let g = Gguf::open(path).ok()?;
        let fit = estimate_fit(
            path,
            FitRequest {
                hardware: Some(Hardware::default()),
                ..FitRequest::default()
            },
        )
        .ok();
        let mut by_type: BTreeMap<u32, (usize, u64)> = BTreeMap::new();
        for tensor in &g.tensors {
            let entry = by_type.entry(tensor.ty).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += tensor.size.unwrap_or(0);
        }
        let mut histogram: Vec<(String, usize, u64)> = by_type
            .into_iter()
            .map(|(ty, (count, bytes))| (type_name(ty).to_string(), count, bytes))
            .collect();
        histogram.sort_by_key(|(_, _, bytes)| std::cmp::Reverse(*bytes));
        Some(Self {
            path: path.to_path_buf(),
            version: g.version,
            file_len: g.file_len,
            alignment: g.alignment,
            data_start: g.data_start,
            tensors: g.tensors.len(),
            pq2: g.pq2_count(),
            ptq1: g.tensors.iter().filter(|t| t.ty == TY_PTQ1_0).count(),
            file_type: g.file_type,
            metadata: g.metadata_summary(12),
            histogram,
            weights_bytes: g.tensors.iter().filter_map(|t| t.size).sum(),
            fit,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputMode {
    NextToInput,
    Custom(PathBuf),
}

#[derive(Debug, Clone)]
pub struct Settings {
    pub target: Target,
    pub verify: VerifyMode,
    pub overwrite: bool,
    pub output_mode: OutputMode,
    pub download_dir: PathBuf,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            target: Target::Reblock,
            verify: VerifyMode::Structure,
            overwrite: false,
            output_mode: OutputMode::NextToInput,
            download_dir: default_download_dir(),
        }
    }
}

pub fn default_download_dir() -> PathBuf {
    home_dir()
        .map(|h| h.join("Downloads").join("RustBlock"))
        .unwrap_or_else(|| {
            std::env::current_dir()
                .unwrap_or_default()
                .join("RustBlock-models")
        })
}

pub fn home_dir() -> Option<PathBuf> {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
}

pub fn find_ascii_case_insensitive(haystack: &str, needle: &str) -> Option<usize> {
    let h = haystack.as_bytes();
    let n = needle.as_bytes();
    if n.is_empty() || h.len() < n.len() {
        return None;
    }
    (0..=h.len() - n.len()).find(|&i| h[i..i + n.len()].eq_ignore_ascii_case(n))
}

fn strip_quant_suffix(base: &str) -> &str {
    if let Some(pos) = base.rfind(['-', '.', '_']) {
        let tail = &base[pos + 1..];
        let upper = tail.to_ascii_uppercase();
        if upper.starts_with('Q')
            || upper.starts_with("IQ")
            || upper.starts_with("TQ")
            || upper.starts_with("BF16")
            || upper.starts_with("F16")
            || upper.starts_with("F32")
        {
            return &base[..pos];
        }
    }
    base
}

pub fn output_path(input: &Path, target: Target, mode: &OutputMode) -> PathBuf {
    let file_name = input
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model.gguf".to_string());
    let lower = file_name.to_ascii_lowercase();
    let stem = if lower.ends_with(".gguf") {
        &file_name[..file_name.len() - 5]
    } else {
        &file_name[..]
    };

    let new_name = if target == Target::Reblock {
        if let Some(pos) = find_ascii_case_insensitive(&file_name, "PQ2_0") {
            format!("{}Q2_0{}", &file_name[..pos], &file_name[pos + 5..])
        } else {
            format!("{}-Q2_0.gguf", strip_quant_suffix(stem))
        }
    } else {
        format!("{}-{}.gguf", strip_quant_suffix(stem), target.short())
    };

    match mode {
        OutputMode::NextToInput => input.with_file_name(new_name),
        OutputMode::Custom(dir) if !dir.as_os_str().is_empty() => dir.join(new_name),
        OutputMode::Custom(_) => input.with_file_name(new_name),
    }
}

#[cfg(windows)]
pub fn path_key(path: &Path) -> String {
    path.to_string_lossy().to_lowercase()
}

#[cfg(not(windows))]
pub fn path_key(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

pub fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else if value < 10.0 {
        format!("{value:.2} {}", UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

pub fn human_speed(bytes_per_second: f64) -> String {
    if !bytes_per_second.is_finite() || bytes_per_second <= 0.0 {
        "-".to_string()
    } else {
        format!("{}/s", human_bytes(bytes_per_second as u64))
    }
}

pub fn human_duration(d: Duration) -> String {
    let secs = d.as_secs();
    if secs < 60 {
        if secs == 0 {
            format!("{} ms", d.as_millis())
        } else {
            format!("{secs} s")
        }
    } else if secs < 3600 {
        format!("{}m {:02}s", secs / 60, secs % 60)
    } else {
        format!("{}h {:02}m", secs / 3600, (secs % 3600) / 60)
    }
}

/// Remove only interrupted-download sidecars from the model folder. Completed
/// `.gguf` files and unrelated files are left untouched.
pub fn clear_partial_files(dir: &Path) -> Result<usize, String> {
    let entries = std::fs::read_dir(dir)
        .map_err(|error| format!("cannot read {}: {error}", dir.display()))?;
    let mut removed = 0usize;
    for entry in entries {
        let entry = entry.map_err(|error| format!("cannot read download folder: {error}"))?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("part") {
            continue;
        }
        std::fs::remove_file(&path)
            .map_err(|error| format!("cannot remove {}: {error}", path.display()))?;
        removed += 1;
    }
    Ok(removed)
}

pub fn shorten(text: &str, max_chars: usize) -> String {
    if max_chars < 8 {
        return text.to_string();
    }
    let count = text.chars().count();
    if count <= max_chars {
        return text.to_string();
    }
    let head: String = text.chars().take(max_chars / 2 - 1).collect();
    let tail: String = text
        .chars()
        .skip(count - (max_chars - max_chars / 2 - 1))
        .collect();
    format!("{head}…{tail}")
}

#[cfg(test)]
mod tests {
    use super::{clear_partial_files, looks_quantized_name};

    #[test]
    fn recognizes_quantized_file_names_without_flagging_qwen() {
        assert!(looks_quantized_name("model-Q4_K_M.gguf"));
        assert!(looks_quantized_name("Qwen3-8B-Q8_0.gguf"));
        assert!(looks_quantized_name("model-IQ3_XS.gguf"));
        assert!(looks_quantized_name("model-TQ2_0.gguf"));
        assert!(!looks_quantized_name("Qwen3-8B-F16.gguf"));
        assert!(!looks_quantized_name("virgin-model.gguf"));
    }

    #[test]
    fn clears_only_partial_downloads() {
        let dir = tempfile::tempdir().unwrap();
        let partial = dir.path().join("model.gguf.part");
        let complete = dir.path().join("model.gguf");
        std::fs::write(&partial, b"partial").unwrap();
        std::fs::write(&complete, b"complete").unwrap();
        assert_eq!(clear_partial_files(dir.path()).unwrap(), 1);
        assert!(!partial.exists());
        assert!(complete.exists());
    }
}
