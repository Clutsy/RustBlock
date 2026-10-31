//! UI data model and filesystem helpers.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ternary_reblock_core::{
    convert, ConvertOptions, Gguf, Phase, Report, VerifyMode, TY_PTQ1_0, TY_Q2_0,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    En,
    It,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileKind {
    Pq2,
    Q2,
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
    pub q2: usize,
    pub file_type: Option<u32>,
    pub hadamard: bool,
    pub estimated_new_size: Option<u64>,
    pub kind: FileKind,
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
            q2: 0,
            file_type: None,
            hadamard: false,
            estimated_new_size: None,
            kind: FileKind::Invalid,
            error: Some(error),
        }
    }

    pub fn can_convert(&self) -> bool {
        self.kind == FileKind::Pq2 && self.error.is_none()
    }

    pub fn can_verify(&self) -> bool {
        self.error.is_none()
    }
}

/// Reads only the GGUF header, then asks the conversion engine for a dry-run
/// estimate (no data is written).
pub fn analyze(path: &Path) -> Analysis {
    let file_len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let g = match Gguf::open(path) {
        Ok(g) => g,
        Err(e) => return Analysis::invalid(file_len, e.to_string()),
    };

    let mut analysis = Analysis {
        version: g.version,
        file_len: g.file_len,
        alignment: g.alignment,
        tensors: g.tensors.len(),
        pq2: g.pq2_count(),
        ptq1: g.tensors.iter().filter(|t| t.ty == TY_PTQ1_0).count(),
        q2: g.tensors.iter().filter(|t| t.ty == TY_Q2_0).count(),
        file_type: g.file_type,
        hadamard: g.has_hadamard_version,
        estimated_new_size: None,
        kind: FileKind::Other,
        error: None,
    };

    if analysis.ptq1 > 0 {
        analysis.kind = FileKind::Ptq1;
    } else if analysis.pq2 > 0 {
        analysis.kind = FileKind::Pq2;
        let dummy = path.with_extension("estimate.gguf");
        analysis.estimated_new_size = convert(
            path,
            &dummy,
            ConvertOptions {
                dry_run: true,
                verify: VerifyMode::None,
                ..Default::default()
            },
            None,
        )
        .ok()
        .map(|r| r.new_file_size);
    } else if analysis.q2 > 0 || analysis.file_type == Some(41) {
        analysis.kind = FileKind::Q2;
    }

    analysis
}

#[derive(Debug, Clone)]
pub enum VerifyOutcome {
    Pq2 { tensors: usize },
    Native { tensors: usize },
    Other { tensors: usize },
    Ptq1 { tensors: usize },
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
    Done(Report),
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
}

impl Job {
    pub fn file_name(&self) -> String {
        self.input
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| self.input.display().to_string())
    }

    pub fn is_finished(&self) -> bool {
        matches!(
            self.status,
            JobStatus::Done(_) | JobStatus::Verified { .. } | JobStatus::Failed(_)
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputMode {
    NextToInput,
    Custom(PathBuf),
}

#[derive(Debug, Clone)]
pub struct Settings {
    pub lang: Lang,
    pub verify: VerifyMode,
    pub overwrite: bool,
    pub verify_only: bool,
    pub output_mode: OutputMode,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            lang: Lang::En,
            verify: VerifyMode::Structure,
            overwrite: false,
            verify_only: false,
            output_mode: OutputMode::NextToInput,
        }
    }
}

pub fn find_ascii_case_insensitive(haystack: &str, needle: &str) -> Option<usize> {
    let h = haystack.as_bytes();
    let n = needle.as_bytes();
    if n.is_empty() || h.len() < n.len() {
        return None;
    }
    (0..=h.len() - n.len()).find(|&i| h[i..i + n.len()].eq_ignore_ascii_case(n))
}

pub fn output_path(input: &Path, mode: &OutputMode) -> PathBuf {
    let file_name = input
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model.gguf".to_string());

    let new_name = if let Some(pos) = find_ascii_case_insensitive(&file_name, "PQ2_0") {
        format!("{}Q2_0{}", &file_name[..pos], &file_name[pos + 5..])
    } else if file_name.len() >= 5 && file_name[file_name.len() - 5..].eq_ignore_ascii_case(".gguf")
    {
        format!("{}-Q2_0.gguf", &file_name[..file_name.len() - 5])
    } else {
        format!("{file_name}-Q2_0.gguf")
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

pub fn log_clock(d: Duration) -> String {
    let secs = d.as_secs();
    format!("{:02}:{:02}", secs / 60, secs % 60)
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
