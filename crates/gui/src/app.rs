//! RustBlock — GUI: queue manager (Convert), discovery (Discover), GGUF inspector.
//!
//! Rewritten from zero (2026-11-21) with a three-tab layout:
//!
//! - **Convert**: the job queue with per-job details, inline progress,
//!   batch progress, and a benchmark banner (llama-bench integration).
//! - **Discover**: Hugging Face search with license display and richer cards.
//! - **Inspector**: facts, metadata and tensor histogram of a local GGUF.
//!
//! The activity log has per-level filters and a level/level-count header.
//! Colors, chrome and the general language follow [`crate::theme`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::hf::{self, HfEvent, HfFile, HfModel, HfSort};
use crate::model::{
    analyze, clear_partial_files, human_bytes, human_duration, human_speed, looks_quantized_name,
    output_path, path_key, shorten, FileKind, InspectorData, Job, JobStatus, OutputMode, RunState,
    Settings, VerifyKind,
};
use crate::theme::{
    self, ACCENT, BG, BORDER, ERROR, INFO, INK, MUTED, OK, PANEL, PANEL_2, PANEL_3, STEEL,
    STEEL_LIGHT, TEXT, WARN,
};
use crate::worker::{self, Event, WorkItem, WorkKind};
use eframe::egui::{
    self, Align, Align2, Color32, CornerRadius, FontId, Id, LayerId, Layout, Margin, Order,
    ProgressBar, RichText, ScrollArea, Sense, Stroke, StrokeKind, TextEdit, Ui, Vec2,
};
use rustblock_core::imatrix::find_tool;
use rustblock_core::{
    cache_stats, plan_budget, trim_cache, BudgetRequest, FitEstimate, Phase, Target, VerifyMode,
};

/// The three views of the new GUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Convert,
    Discover,
    Inspector,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LogLevel {
    Info,
    Ok,
    Warn,
    Error,
}

#[derive(Debug, Clone)]
struct LogEntry {
    level: LogLevel,
    text: String,
}

impl LogLevel {
    fn index(self) -> usize {
        match self {
            LogLevel::Info => 0,
            LogLevel::Ok => 1,
            LogLevel::Warn => 2,
            LogLevel::Error => 3,
        }
    }
    fn color(self) -> Color32 {
        match self {
            LogLevel::Info => MUTED,
            LogLevel::Ok => OK,
            LogLevel::Warn => WARN,
            LogLevel::Error => ERROR,
        }
    }
    fn short(self) -> &'static str {
        match self {
            LogLevel::Info => "I",
            LogLevel::Ok => "✓",
            LogLevel::Warn => "⚠",
            LogLevel::Error => "✗",
        }
    }
}

fn log_entry_visible(entry: &LogEntry, filter: &str, levels: &[bool; 4]) -> bool {
    levels[entry.level.index()]
        && (filter.is_empty() || entry.text.to_ascii_lowercase().contains(filter))
}

#[derive(Debug, Clone)]
struct Batch {
    kind: WorkKind,
    total: usize,
    completed: usize,
    failed: usize,
    active: Option<u64>,
    started: Instant,
}

impl Batch {
    fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }
}

#[derive(Debug, Clone)]
struct DownloadUi {
    repo: String,
    file: String,
    done: u64,
    total: u64,
    speed: f64,
    quantized: bool,
    projector: bool,
    path: Option<PathBuf>,
    error: Option<String>,
}

#[derive(Debug)]
enum BenchmarkEvent {
    Done { ok: bool, text: String },
}

#[derive(Debug)]
struct BenchmarkUi {
    model: PathBuf,
    receiver: Receiver<BenchmarkEvent>,
    handle: JoinHandle<()>,
}

#[derive(Debug, Clone)]
enum HfState {
    Idle,
    Searching,
    Results(Vec<HfModel>),
    Error(String),
}

pub struct RustBlockApp {
    tab: Tab,
    jobs: Vec<Job>,
    next_id: u64,
    selected_job: Option<u64>,
    settings: Settings,
    rx: Option<Receiver<Event>>,
    worker: Option<JoinHandle<()>>,
    cancel: Option<Arc<AtomicBool>>,
    batch: Option<Batch>,
    last_output_dir: Option<PathBuf>,
    mark: Option<egui::TextureHandle>,
    wordmark: Option<egui::TextureHandle>,
    ctx: egui::Context,
    // Benchmark (llama-bench on the largest finished model)
    benchmark: Option<BenchmarkUi>,
    // Activity log
    logs: Vec<LogEntry>,
    show_log: bool,
    log_filter: String,
    log_levels: [bool; 4],
    // Help
    show_help: bool,
    // GGUF inspector
    inspector_path: Option<PathBuf>,
    inspector: Option<InspectorData>,
    // Discover
    hf_query: String,
    hf_sort: HfSort,
    hf_state: HfState,
    hf_rx: Option<Receiver<HfEvent>>,
    hf_worker: Option<JoinHandle<()>>,
    hf_cancel: Arc<AtomicBool>,
    selected_repo: Option<String>,
    selected_sha: String,
    hf_files: Vec<HfFile>,
    hf_details_loading: bool,
    pending_details: Option<(String, String)>,
    hf_scroll_top: bool,
    hf_details_error: Option<String>,
    hf_file_filter: String,
    discover_loaded: bool,
    download: Option<DownloadUi>,
    confirm_requantize: Option<WorkKind>,
}

fn file_name_of(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

fn done_output_name(job: &Job) -> Option<String> {
    job.done_output()
        .map(|path| file_name_of(&path))
        .filter(|name| !name.is_empty())
}

fn spawn_benchmark(model: PathBuf) -> BenchmarkUi {
    let (tx, rx) = mpsc::channel();
    let model_thread = model.clone();
    let handle = thread::spawn(move || {
        let result: Result<(bool, String), String> = if let Some(tool) = find_tool("llama-bench") {
            Command::new(&tool)
                .arg("-m")
                .arg(&model_thread)
                .arg("-ngl")
                .arg("99")
                .output()
                .map(|output| {
                    let ok = output.status.success();
                    let text = if ok {
                        String::from_utf8_lossy(&output.stdout).into_owned()
                    } else {
                        String::from_utf8_lossy(&output.stderr).into_owned()
                    };
                    (ok, text)
                })
                .map_err(|error| format!("cannot run llama-bench: {error}"))
        } else {
            Err("llama-bench not found; install llama.cpp tools or set LLAMA_BENCH".to_string())
        };
        let (ok, text) = match result {
            Ok(values) => values,
            Err(message) => (false, message),
        };
        let _ = tx.send(BenchmarkEvent::Done { ok, text });
    });
    BenchmarkUi {
        model,
        receiver: rx,
        handle,
    }
}

impl RustBlockApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut app = Self::with_context(&cc.egui_ctx);
        let files: Vec<PathBuf> = std::env::args_os()
            .skip(1)
            .map(PathBuf::from)
            .filter(|p| p.exists())
            .collect();
        if !files.is_empty() {
            app.add_paths(files);
        }
        app
    }

    fn with_context(ctx: &egui::Context) -> Self {
        theme::install(ctx);
        let mut app = Self {
            tab: Tab::Convert,
            jobs: Vec::new(),
            next_id: 1,
            selected_job: None,
            settings: Settings::default(),
            rx: None,
            worker: None,
            cancel: None,
            batch: None,
            last_output_dir: None,
            mark: theme::load_gear(ctx),
            wordmark: theme::load_wordmark(ctx),
            ctx: ctx.clone(),
            benchmark: None,
            logs: Vec::new(),
            show_log: false,
            log_filter: String::new(),
            log_levels: [true, true, true, true],
            show_help: false,
            inspector_path: None,
            inspector: None,
            hf_query: String::new(),
            hf_sort: HfSort::Downloads,
            hf_state: HfState::Idle,
            hf_rx: None,
            hf_worker: None,
            hf_cancel: Arc::new(AtomicBool::new(false)),
            selected_repo: None,
            selected_sha: "main".to_string(),
            hf_files: Vec::new(),
            hf_details_loading: false,
            pending_details: None,
            hf_scroll_top: false,
            hf_details_error: None,
            hf_file_filter: String::new(),
            discover_loaded: false,
            download: None,
            confirm_requantize: None,
        };
        app.log(
            LogLevel::Info,
            "Ready. Add a GGUF file, search Hugging Face, or open the GGUF inspector.".to_string(),
        );
        app
    }

    fn is_running(&self) -> bool {
        self.rx.is_some()
    }

    fn hf_busy(&self) -> bool {
        self.hf_rx.is_some()
    }

    fn log(&mut self, level: LogLevel, text: String) {
        self.logs.push(LogEntry { level, text });
        if self.logs.len() > 500 {
            let excess = self.logs.len() - 500;
            self.logs.drain(0..excess);
        }
    }

    fn job_name(&self, id: u64) -> String {
        self.jobs
            .iter()
            .find(|j| j.id == id)
            .map(|j| j.file_name())
            .unwrap_or_else(|| format!("#{id}"))
    }

    fn batch_completed(&mut self) {
        if let Some(batch) = &mut self.batch {
            batch.completed += 1;
        }
    }

    fn finish_batch(&mut self) {
        self.rx = None;
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
        self.cancel = None;
        self.batch = None;
        for job in &mut self.jobs {
            if matches!(job.status, JobStatus::Running(_)) {
                job.status = JobStatus::Ready;
            }
        }
    }

    fn handle_dropped(&mut self, ctx: &egui::Context) {
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });
        if !dropped.is_empty() {
            self.add_paths(dropped);
        }
    }

    fn add_paths(&mut self, paths: Vec<PathBuf>) {
        let hardware = rustblock_core::detect_hardware();
        for path in paths {
            let is_gguf = path
                .extension()
                .map(|e| e.to_string_lossy().eq_ignore_ascii_case("gguf"))
                .unwrap_or(false);
            if !is_gguf {
                self.log(
                    LogLevel::Warn,
                    format!("{} - only .gguf files are supported", path.display()),
                );
                continue;
            }
            if self.jobs.iter().any(|j| same_file(&j.input, &path)) {
                self.log(
                    LogLevel::Info,
                    format!("Already in the queue: {}", path.display()),
                );
                continue;
            }
            let analysis = analyze(&path, hardware);
            if let Some(error) = &analysis.error {
                self.log(
                    LogLevel::Error,
                    format!("Cannot read {}: {error}", path.display()),
                );
            } else {
                self.log(LogLevel::Info, format!("Added: {}", path.display()));
                if analysis.kind == FileKind::Q2 {
                    self.log(
                        LogLevel::Info,
                        format!(
                            "{} is already upstream Q2_0; no reblock is needed.",
                            file_name_of(&path)
                        ),
                    );
                }
                if (analysis.is_quantized() && analysis.kind != FileKind::Pq2)
                    || path
                        .file_name()
                        .is_some_and(|name| looks_quantized_name(&name.to_string_lossy()))
                {
                    self.log(
                        LogLevel::Warn,
                        format!(
                            "{} is already quantized. Converting it again cannot recover the original precision; a virgin F16/BF16 model is the better source.",
                            file_name_of(&path)
                        ),
                    );
                }
            }
            let target = if analysis.can_target(self.settings.target) {
                self.settings.target
            } else {
                analysis.default_target()
            };
            if !analysis.can_target(self.settings.target) {
                self.settings.target = target;
            }
            let status = if let Some(error) = &analysis.error {
                JobStatus::Failed(error.clone())
            } else {
                JobStatus::Ready
            };
            let id = self.next_id;
            self.next_id += 1;
            self.jobs.push(Job {
                id,
                input: path,
                analysis,
                status,
                target,
            });
        }
    }

    fn browse(&mut self) {
        if let Some(files) = rfd::FileDialog::new()
            .add_filter("GGUF model", &["gguf"])
            .pick_files()
        {
            self.add_paths(files);
        }
    }

    fn poll_worker(&mut self, ctx: &egui::Context) {
        let mut events = Vec::new();
        let mut disconnected = false;
        if let Some(rx) = &self.rx {
            loop {
                match rx.try_recv() {
                    Ok(event) => events.push(event),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
        }
        for event in events {
            self.handle_event(event, ctx);
        }
        if disconnected && self.rx.is_some() {
            self.log(
                LogLevel::Error,
                "The conversion worker stopped unexpectedly.".to_string(),
            );
            self.finish_batch();
        }
    }

    fn handle_event(&mut self, event: Event, ctx: &egui::Context) {
        match event {
            Event::Started { id } => {
                if let Some(job) = self.jobs.iter_mut().find(|j| j.id == id) {
                    job.status = JobStatus::Running(RunState::new(job.analysis.file_len.max(1)));
                }
                if let Some(batch) = &mut self.batch {
                    batch.active = Some(id);
                }
            }
            Event::Progress {
                id,
                phase,
                done,
                total,
                tensor_index,
                tensor_name,
                tensors_total,
            } => {
                if let Some(job) = self.jobs.iter_mut().find(|j| j.id == id) {
                    match &mut job.status {
                        JobStatus::Running(state) => state.update(
                            phase,
                            done,
                            total,
                            tensor_index,
                            tensor_name,
                            tensors_total,
                        ),
                        _ => {
                            let mut state = RunState::new(total);
                            state.update(
                                phase,
                                done,
                                total,
                                tensor_index,
                                tensor_name,
                                tensors_total,
                            );
                            job.status = JobStatus::Running(state);
                        }
                    }
                }
            }
            Event::Done { id, report, target } => {
                let name = self.job_name(id);
                let old = report.old_file_size;
                let new = report.new_file_size;
                let elapsed = report.elapsed;
                if let Some(job) = self.jobs.iter_mut().find(|j| j.id == id) {
                    job.status = JobStatus::Done { report, target };
                }
                self.batch_completed();
                self.log(
                    LogLevel::Ok,
                    format!(
                        "Converted {name} to {} ({} -> {}) in {}",
                        target.short(),
                        human_bytes(old),
                        human_bytes(new),
                        human_duration(elapsed)
                    ),
                );
            }
            Event::Verified {
                id,
                outcome,
                elapsed,
            } => {
                let name = self.job_name(id);
                if let Some(job) = self.jobs.iter_mut().find(|j| j.id == id) {
                    job.status = JobStatus::Verified {
                        outcome: outcome.clone(),
                        elapsed,
                    };
                }
                self.batch_completed();
                self.log(
                    LogLevel::Ok,
                    format!(
                        "Verified {name} in {} - {}",
                        human_duration(elapsed),
                        outcome.label
                    ),
                );
            }
            Event::Failed { id, error } => {
                let name = self.job_name(id);
                if let Some(job) = self.jobs.iter_mut().find(|j| j.id == id) {
                    job.status = JobStatus::Failed(error.clone());
                }
                if let Some(batch) = &mut self.batch {
                    batch.completed += 1;
                    batch.failed += 1;
                }
                self.log(LogLevel::Error, format!("Error on {name}: {error}"));
            }
            Event::BatchDone {
                converted,
                verified,
                failed,
                elapsed,
            } => {
                let stopped = self
                    .cancel
                    .as_ref()
                    .is_some_and(|c| c.load(Ordering::Relaxed));
                let kind = self
                    .batch
                    .as_ref()
                    .map(|b| b.kind)
                    .unwrap_or(WorkKind::Convert);
                self.finish_batch();
                let ok = if kind == WorkKind::Convert {
                    converted
                } else {
                    verified
                };
                self.log(
                    if failed == 0 {
                        LogLevel::Ok
                    } else {
                        LogLevel::Warn
                    },
                    format!(
                        "{} finished: {} succeeded, {} failed, {}",
                        if kind == WorkKind::Convert {
                            "Conversion"
                        } else {
                            "Verification"
                        },
                        ok,
                        failed,
                        human_duration(elapsed)
                    ),
                );
                if stopped {
                    self.log(
                        LogLevel::Warn,
                        "Batch stopped. Remaining files are ready to start again.".to_string(),
                    );
                }
                ctx.request_repaint();
            }
        }
    }

    fn start(&mut self, kind: WorkKind) {
        if kind == WorkKind::Convert && self.requantize_needed() {
            self.confirm_requantize = Some(kind);
            return;
        }
        self.launch(kind, false);
    }

    fn requantize_needed(&self) -> bool {
        let target = self.settings.target;
        target != Target::Reblock
            && target.is_quantized()
            && self
                .jobs
                .iter()
                .any(|job| job.analysis.is_quantized() && job.analysis.can_target(target))
    }

    fn launch(&mut self, kind: WorkKind, allow_requantize: bool) {
        if self.is_running() {
            return;
        }
        let target = self.settings.target;
        let overwrite = self.settings.overwrite;
        let verify_setting = self.settings.verify;
        let output_mode = self.settings.output_mode.clone();
        let mut items = Vec::new();
        let mut used_outputs: HashMap<String, u64> = HashMap::new();
        let mut notes: Vec<(LogLevel, String)> = Vec::new();
        for job in &mut self.jobs {
            let eligible = if kind == WorkKind::Convert {
                job.analysis.can_target(target)
            } else {
                job.analysis.error.is_none()
            };
            if !eligible {
                continue;
            }
            let output = if kind == WorkKind::Convert {
                output_path(&job.input, target, &output_mode)
            } else {
                job.input.with_extension("verify.gguf")
            };
            if kind == WorkKind::Convert {
                if path_key(&output) == path_key(&job.input) {
                    job.status = JobStatus::Failed(
                        "The output path is the input file. Choose another folder.".to_string(),
                    );
                    notes.push((
                        LogLevel::Warn,
                        format!("{}: output equals input", job.file_name()),
                    ));
                    continue;
                }
                if matches!(job.status, JobStatus::Done { .. }) && output.exists() && !overwrite {
                    continue;
                }
                let key = path_key(&output);
                if let Some(previous) = used_outputs.get(&key) {
                    let text = format!(
                        "Duplicate output path for {}: {} (job {previous})",
                        job.file_name(),
                        output.display()
                    );
                    job.status = JobStatus::Failed(text.clone());
                    notes.push((LogLevel::Warn, text));
                    continue;
                }
                used_outputs.insert(key, job.id);
                if output.exists() && !overwrite {
                    job.status = JobStatus::Failed(
                        "Output already exists. Enable Overwrite or choose another folder."
                            .to_string(),
                    );
                    notes.push((
                        LogLevel::Warn,
                        format!("{}: output already exists", job.file_name()),
                    ));
                    continue;
                }
            }
            job.target = target;
            job.status = JobStatus::Running(RunState::new(job.analysis.file_len.max(1)));
            items.push(WorkItem {
                id: job.id,
                kind,
                input: job.input.clone(),
                output,
                target,
                verify: if kind == WorkKind::Verify {
                    VerifyMode::Full
                } else {
                    verify_setting
                },
                force: overwrite,
                allow_requantize,
            });
        }
        for (level, text) in notes {
            self.log(level, text);
        }
        if items.is_empty() {
            self.log(
                LogLevel::Info,
                format!(
                    "Nothing to convert to {}. Add a compatible file or choose another target.",
                    target.short()
                ),
            );
            return;
        }
        let total = items.len();
        self.batch = Some(Batch {
            kind,
            total,
            completed: 0,
            failed: 0,
            active: items.first().map(|i| i.id),
            started: Instant::now(),
        });
        self.log(
            LogLevel::Info,
            if kind == WorkKind::Convert {
                format!(
                    "Starting conversion of {total} file(s) to {}...",
                    target.short()
                )
            } else {
                format!("Starting verification of {total} file(s)...")
            },
        );
        let cancel = Arc::new(AtomicBool::new(false));
        let (rx, handle) = worker::spawn(items, cancel.clone());
        self.rx = Some(rx);
        self.worker = Some(handle);
        self.cancel = Some(cancel);
    }

    fn requantize_modal(&mut self, ctx: &egui::Context) {
        let Some(kind) = self.confirm_requantize else {
            return;
        };
        let mut cancel = false;
        let mut proceed = false;
        egui::Window::new("Requantization warning")
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .collapsible(false)
            .resizable(false)
            .order(Order::Foreground)
            .show(ctx, |ui| {
                ui.set_max_width(430.0);
                ui.label(
                    RichText::new("These files are already quantized.")
                        .size(15.0)
                        .strong()
                        .color(WARN),
                );
                ui.add_space(6.0);
                ui.label(
                    RichText::new(
                        "Converting a quantized model to another quantized format does not restore the weights lost by the first quantization. The result can only be worse and may be noticeably degraded.\n\nUse F16/F32 to decompress, or continue only if you have a copy of the original float model.",
                    )
                    .size(11.0)
                    .color(TEXT),
                );
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    if ui
                        .add(
                            egui::Button::new(RichText::new("Cancel").size(11.0).color(INK))
                                .fill(PANEL_3)
                                .min_size(Vec2::new(100.0, 32.0)),
                        )
                        .clicked()
                    {
                        cancel = true;
                    }
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new("Requantize anyway")
                                    .size(11.0)
                                    .strong()
                                    .color(INK),
                            )
                            .fill(WARN)
                            .min_size(Vec2::new(160.0, 32.0)),
                        )
                        .clicked()
                    {
                        proceed = true;
                    }
                });
            });
        if cancel {
            self.confirm_requantize = None;
        }
        if proceed {
            self.confirm_requantize = None;
            self.launch(kind, true);
        }
    }

    // ----- Benchmark ------------------------------------------------------
    fn done_job_output(&self) -> Option<PathBuf> {
        self.jobs.iter().rev().find_map(Job::done_output)
    }

    fn start_benchmark(&mut self) {
        if self.benchmark.is_some() {
            return;
        }
        let Some(model) = self.done_job_output() else {
            return;
        };
        let ui = spawn_benchmark(model.clone());
        self.log(
            LogLevel::Info,
            format!("Benchmark: llama-bench -m {}", model.display()),
        );
        self.benchmark = Some(ui);
    }

    fn poll_benchmark(&mut self) {
        let mut done = None;
        if let Some(bm) = &self.benchmark {
            while let Ok(event) = bm.receiver.try_recv() {
                match event {
                    BenchmarkEvent::Done { ok, text } => done = Some((ok, text)),
                }
            }
        }
        if let Some((ok, text)) = done {
            let text = if text.trim().is_empty() {
                "<no output>".to_string()
            } else {
                text
            };
            self.log(
                if ok { LogLevel::Ok } else { LogLevel::Error },
                format!("Benchmark result (ok={ok}):\n{text}"),
            );
            if let Some(bm) = self.benchmark.take() {
                let _ = bm.handle.join();
            }
        }
    }

    // ----- GGUF inspector ---------------------------------------------------
    fn inspector_open(&mut self) {
        if let Some(files) = rfd::FileDialog::new()
            .add_filter("GGUF model", &["gguf"])
            .pick_files()
        {
            if let Some(path) = files.into_iter().next() {
                self.inspector_path = Some(path);
                self.inspector_refresh();
            }
        }
    }

    fn inspector_refresh(&mut self) {
        self.inspector = self.inspector_path.as_deref().and_then(InspectorData::open);
    }

    fn selected_jobs(&self) -> usize {
        let target = self.settings.target;
        self.jobs
            .iter()
            .filter(|j| j.analysis.can_target(target))
            .count()
    }

    fn download_banner(&self, ui: &mut Ui) {
        let Some(download) = self.download.clone() else {
            return;
        };
        ui.add_space(10.0);
        egui::Frame::new()
            .fill(PANEL_2)
            .stroke(Stroke::new(1.0, BORDER))
            .corner_radius(CornerRadius::same(4))
            .inner_margin(Margin::symmetric(12, 10))
            .show(ui, |ui| {
                let name = download
                    .file
                    .rsplit('/')
                    .next()
                    .unwrap_or(download.file.as_str());
                let kind = if download.projector { " · projector" } else { "" };
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(format!("{}{kind}", shorten(name, 62)))
                            .size(11.0)
                            .monospace()
                            .color(STEEL_LIGHT),
                    )
                    .on_hover_text(format!("{}/{}", download.repo, download.file));
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        let (status, color) = if download.path.is_some() {
                            ("Complete", ACCENT)
                        } else if download.error.is_some() {
                            ("Failed", ERROR)
                        } else {
                            ("Downloading", MUTED)
                        };
                        ui.label(RichText::new(status).size(10.0).color(color));
                    });
                });
                let frac = if download.total > 0 {
                    (download.done as f64 / download.total as f64).clamp(0.0, 1.0) as f32
                } else {
                    0.0
                };
                let remaining = download.total.saturating_sub(download.done);
                let eta = if download.speed >= 1.0 && remaining > 0 {
                    Some(Duration::from_secs_f64(remaining as f64 / download.speed))
                } else {
                    None
                };
                ui.add_sized(
                    [ui.available_width(), 18.0],
                    ProgressBar::new(frac)
                        .fill(ACCENT)
                        .animate(download.path.is_none() && download.error.is_none())
                        .text(format!(
                            "{} / {} · {:.1}%",
                            human_bytes(download.done),
                            if download.total > 0 {
                                human_bytes(download.total)
                            } else {
                                "?".to_string()
                            },
                            frac * 100.0
                        )),
                );
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        RichText::new(human_speed(download.speed))
                            .size(9.5)
                            .monospace()
                            .color(MUTED),
                    );
                    if let Some(eta) = eta {
                        ui.label(
                            RichText::new(format!("· {} left", human_duration(eta)))
                                .size(9.5)
                                .color(MUTED),
                        );
                    }
                    if download.path.is_none()
                        && download.error.is_none()
                        && ui
                            .add(
                                egui::Button::new(
                                    RichText::new("Cancel").size(9.5).color(MUTED),
                                )
                                .frame(false),
                            )
                            .clicked()
                    {
                        self.hf_cancel.store(true, Ordering::Relaxed);
                    }
                    if let Some(path) = &download.path {
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new("Open folder").size(9.5).color(ACCENT),
                                )
                                .frame(false),
                            )
                            .clicked()
                        {
                            open_folder(path);
                        }
                    }
                });
                if download.quantized {
                    ui.label(
                        RichText::new(
                            "Already quantized · requantizing will not restore the original weights.",
                        )
                        .size(9.5)
                        .color(WARN),
                    );
                }
                if let Some(error) = &download.error {
                    ui.label(RichText::new(error).size(9.5).color(ERROR));
                }
            });
    }

    // ----- Convert views ---------------------------------------------------
    fn queue_counts(&self) -> (usize, usize, usize, usize) {
        let ready = self.selected_jobs();
        let running = self
            .jobs
            .iter()
            .filter(|j| matches!(j.status, JobStatus::Running(_)))
            .count();
        let done = self
            .jobs
            .iter()
            .filter(|j| matches!(j.status, JobStatus::Done { .. }))
            .count();
        let failed = self
            .jobs
            .iter()
            .filter(|j| matches!(j.status, JobStatus::Failed(_)))
            .count();
        (ready, running, done, failed)
    }

    fn convert_header(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.label(RichText::new("Queue").size(26.0).strong().color(TEXT));
                let (ready, running, done, failed) = self.queue_counts();
                ui.label(
                    RichText::new(format!(
                        "{} files · {} ready · {} running · {} done · {} failed",
                        self.jobs.len(),
                        ready,
                        running,
                        done,
                        failed
                    ))
                    .size(10.5)
                    .color(MUTED),
                );
            });
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let add =
                    egui::Button::new(RichText::new("Add files").size(12.0).strong().color(INK))
                        .fill(ACCENT)
                        .stroke(Stroke::new(1.0, ACCENT))
                        .corner_radius(CornerRadius::same(6))
                        .min_size(Vec2::new(108.0, 32.0));
                if ui.add(add).clicked() {
                    self.browse();
                }
                let benchmarkable = self.benchmark.is_none() && self.done_job_output().is_some();
                if ui
                    .add_enabled(
                        benchmarkable,
                        egui::Button::new(RichText::new("Benchmark").size(10.5))
                            .fill(PANEL_2)
                            .stroke(Stroke::new(1.0, BORDER))
                            .corner_radius(CornerRadius::same(6)),
                    )
                    .on_hover_text(
                        "Run llama-bench -m <model> -ngl 99 on the largest finished model",
                    )
                    .clicked()
                {
                    self.start_benchmark();
                }
                let running = self.is_running();
                let count = self.selected_jobs();
                let enabled = !running && count > 0;
                let label = if running {
                    "Converting…".to_string()
                } else if count == 0 {
                    "Start conversion".to_string()
                } else if count == 1 {
                    "Convert 1 file".to_string()
                } else {
                    format!("Convert {count} files")
                };
                let button =
                    egui::Button::new(RichText::new(label).size(12.5).strong().color(if enabled {
                        INK
                    } else {
                        MUTED
                    }))
                    .fill(if enabled { ACCENT } else { PANEL_3 })
                    .stroke(Stroke::new(1.0, if enabled { ACCENT } else { BORDER }))
                    .corner_radius(CornerRadius::same(4))
                    .min_size(Vec2::new(ui.available_width(), 38.0));
                if ui.add_enabled(enabled, button).clicked() {
                    self.start(WorkKind::Convert);
                }
                if ui
                    .add_enabled(
                        !self.is_running(),
                        egui::Button::new(RichText::new("Clear all").size(10.5).color(MUTED))
                            .frame(false),
                    )
                    .clicked()
                {
                    self.jobs.clear();
                    self.selected_job = None;
                }
            });
        });
        ui.add_space(4.0);
    }

    fn benchmark_banner(&self, ui: &mut Ui) {
        if let Some(bm) = &self.benchmark {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(
                    RichText::new(format!(
                        "Benchmarking llama-bench -m {}",
                        shorten(&file_name_of(&bm.model), 48)
                    ))
                    .size(10.5)
                    .color(STEEL_LIGHT),
                );
            });
            ui.add_space(4.0);
        }
    }

    fn batch_bar(&mut self, ui: &mut Ui) {
        if let Some(batch) = &self.batch {
            let active_fraction = self
                .jobs
                .iter()
                .find(|j| Some(j.id) == batch.active)
                .and_then(|j| match &j.status {
                    JobStatus::Running(state) => Some(state.fraction()),
                    _ => None,
                })
                .unwrap_or(0.0);
            let overall = ((batch.completed as f32 + active_fraction) / batch.total.max(1) as f32)
                .clamp(0.0, 1.0);
            egui::Frame::new()
                .fill(PANEL_2)
                .stroke(Stroke::new(1.0, BORDER))
                .corner_radius(CornerRadius::same(8))
                .inner_margin(Margin::symmetric(14, 12))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new(format!(
                                "Batch · {} of {}",
                                batch.completed.min(batch.total),
                                batch.total
                            ))
                            .size(11.0)
                            .strong()
                            .color(STEEL_LIGHT),
                        );
                        ui.label(
                            RichText::new(human_duration(batch.elapsed()))
                                .size(10.5)
                                .monospace()
                                .color(MUTED),
                        );
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            let stopping = self
                                .cancel
                                .as_ref()
                                .is_some_and(|c| c.load(Ordering::Relaxed));
                            if ui
                                .add_enabled(
                                    !stopping,
                                    egui::Button::new(
                                        RichText::new(if stopping {
                                            "Stopping…"
                                        } else {
                                            "Stop"
                                        })
                                        .size(10.5),
                                    )
                                    .fill(PANEL)
                                    .stroke(Stroke::new(1.0, BORDER))
                                    .corner_radius(CornerRadius::same(6)),
                                )
                                .clicked()
                            {
                                if let Some(cancel) = &self.cancel {
                                    cancel.store(true, Ordering::Relaxed);
                                }
                            }
                        });
                    });
                    ui.add_space(6.0);
                    ui.add(
                        ProgressBar::new(overall)
                            .fill(ACCENT)
                            .animate(true)
                            .text(format!("{:.0}%", overall * 100.0)),
                    );
                });
            ui.add_space(10.0);
        }
    }

    fn queue_scroll(&mut self, ui: &mut Ui) {
        let running = self.is_running();
        let mut remove: Option<u64> = None;
        let mut open: Option<PathBuf> = None;
        let mut toggle: Option<u64> = None;
        ScrollArea::vertical()
            .id_salt("queue_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for job in &self.jobs {
                    let selected = self.selected_job == Some(job.id);
                    job_card(
                        ui,
                        job,
                        running,
                        selected,
                        &mut remove,
                        &mut open,
                        &mut toggle,
                    );
                    ui.add_space(8.0);
                }
            });
        if let Some(id) = remove {
            self.jobs.retain(|j| j.id != id);
        }
        if let Some(path) = open {
            open_folder(&path);
        }
        if let Some(id) = toggle {
            self.selected_job = match self.selected_job {
                Some(current) if current == id => None,
                _ => Some(id),
            };
        }
        self.selected_job = self
            .selected_job
            .filter(|id| self.jobs.iter().any(|j| j.id == *id));
    }

    fn details_panel(&mut self, ui: &mut Ui) {
        ui.label(
            RichText::new("Details")
                .size(13.0)
                .strong()
                .color(STEEL_LIGHT),
        );
        ui.add_space(6.0);
        let Some(id) = self.selected_job else {
            let (_ready, _running, _done, _failed) = self.queue_counts();
            ui.label(
                RichText::new("Select a job above for details, or start the batch.")
                    .size(9.5)
                    .color(MUTED),
            );
            return;
        };
        let Some(job) = self.jobs.iter().find(|j| j.id == id).cloned() else {
            return;
        };
        if let Some(fit) = &job.analysis.fit {
            let (label, color, hint) = fit_badge(fit);
            ui.add(egui::Label::new(RichText::new(label).color(color)));
            ui.label(
                RichText::new(format!(
                    "weights {} · kv {} · overhead {} · total {}",
                    human_bytes(fit.weights_bytes),
                    human_bytes(fit.kv_cache_bytes),
                    human_bytes(fit.overhead_bytes),
                    human_bytes(fit.total_bytes)
                ))
                .size(9.5)
                .color(MUTED),
            );
            ui.add(egui::Label::new(RichText::new(hint).size(9.0).color(MUTED)).wrap());
        }
        ui.label(RichText::new(job.file_name()).size(11.0).color(TEXT));
        ui.label(
            RichText::new(format!(
                "{} · {}",
                human_bytes(job.analysis.file_len),
                job.analysis.kind_label()
            ))
            .size(9.5)
            .color(MUTED),
        );
        ui.label(
            RichText::new(format!(
                "→ {}",
                shorten(
                    &file_name_of(&job.planned_output(&self.settings.output_mode)),
                    34
                )
            ))
            .size(9.5)
            .color(STEEL_LIGHT),
        );
        if let Some(output) = job.done_output() {
            if ui.button(RichText::new("Open folder").size(10.5)).clicked() {
                open_folder(&output);
            }
        }
        if !self.is_running()
            && ui
                .add_enabled(
                    true,
                    egui::Button::new(RichText::new("Remove").size(10.5).color(MUTED)),
                )
                .clicked()
        {
            self.jobs.retain(|j| j.id != job.id);
            self.selected_job = None;
        }
    }

    fn drop_more_bar(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label(
                RichText::new("Drop more GGUF files here")
                    .size(10.5)
                    .color(MUTED),
            );
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui
                    .add(egui::Button::new(
                        RichText::new("Browse").size(10.0).color(STEEL_LIGHT),
                    ))
                    .clicked()
                {
                    self.browse();
                }
            });
        });
    }

    fn empty_queue(&mut self, ui: &mut Ui) {
        egui::Frame::new()
            .fill(Color32::from_rgba_unmultiplied(22, 25, 28, 168))
            .stroke(Stroke::new(1.0, BORDER))
            .corner_radius(CornerRadius::same(8))
            .inner_margin(Margin::symmetric(32, 56))
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                ui.vertical_centered(|ui| {
                    ui.label(
                        RichText::new("DROP ZONE")
                            .size(9.5)
                            .strong()
                            .color(STEEL_LIGHT),
                    );
                    ui.add_space(6.0);
                    ui.label(
                        RichText::new("Drop GGUF files here")
                            .size(18.0)
                            .strong()
                            .color(TEXT),
                    );
                    ui.add_space(5.0);
                    ui.label(
                        RichText::new(
                            "One or more .gguf files. Nothing is changed until you start.",
                        )
                        .size(11.5)
                        .color(MUTED),
                    );
                    ui.add_space(16.0);
                    ui.label(
                        RichText::new("Ctrl+O add files · Ctrl+S start · Ctrl+L log · F1 help")
                            .size(10.5)
                            .color(STEEL_LIGHT),
                    );
                    ui.add_space(16.0);
                    let browse = egui::Button::new(RichText::new("Browse files").size(11.5))
                        .fill(PANEL_2)
                        .stroke(Stroke::new(1.0, BORDER))
                        .corner_radius(CornerRadius::same(4))
                        .min_size(Vec2::new(120.0, 32.0));
                    if ui.add(browse).clicked() {
                        self.browse();
                    }
                });
            });
    }

    fn convert_page(&mut self, ui: &mut Ui) {
        self.convert_header(ui);
        self.benchmark_banner(ui);
        self.batch_bar(ui);
        ui.add_space(12.0);
        if self.jobs.is_empty() {
            self.empty_queue(ui);
            return;
        }
        ui.horizontal(|ui| {
            let details_width = if self.selected_job.is_some() {
                320.0
            } else {
                0.0
            };
            let queue_width = (ui.available_width() - details_width - 16.0).max(300.0);
            ui.allocate_ui_with_layout(
                Vec2::new(queue_width, ui.available_height()),
                Layout::top_down(Align::LEFT),
                |ui| self.queue_scroll(ui),
            );
            if details_width > 10.0 {
                ui.allocate_ui_with_layout(
                    Vec2::new(details_width, ui.available_height()),
                    Layout::top_down(Align::LEFT),
                    |ui| self.details_panel(ui),
                );
            }
        });
        self.drop_more_bar(ui);
    }

    fn settings_rail(&mut self, ui: &mut Ui) {
        egui::Panel::right("settings_rail")
            .exact_size(380.0)
            .resizable(true)
            .frame(
                egui::Frame::new()
                    .fill(PANEL)
                    .inner_margin(Margin::symmetric(0, 0))
                    .stroke(Stroke::new(1.0, BORDER)),
            )
            .show(ui, |ui| {
                egui::Frame::new()
                    .inner_margin(Margin::symmetric(20, 16))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Output format").size(10.5).color(MUTED));
                            help_dot(
                                ui,
                                "Choose the output precision. Q4_K_M is the balanced default; F16/BF16 are best before requantizing.",
                            );
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                if ui
                                    .add(
                                        egui::Button::new(
                                            RichText::new("Non so coda scegliere")
                                                .size(9.5)
                                                .color(STEEL_LIGHT),
                                        )
                                        .frame(false),
                                    )
                                    .on_hover_text(
                                        "Uses detected RAM/VRAM to suggest a format. You can always change it.",
                                    )
                                    .clicked()
                                {
                                    self.suggest_target();
                                }
                            });
                        });
                        ui.add_space(1.0);
                        ui.label(
                            RichText::new(self.settings.target.short())
                                .size(26.0)
                                .strong()
                                .color(TEXT),
                        )
                        .on_hover_text(target_caption(self.settings.target));
                        ui.add_space(6.0);
                        self.format_panel(ui);
                        ui.add_space(8.0);
                        ui.separator();
                        ui.add_space(8.0);
                        let max_height = (ui.available_height() - 145.0).max(46.0);
                        ScrollArea::vertical()
                            .id_salt("settings_scroll")
                            .max_height(max_height)
                            .max_width(ui.available_width())
                            .auto_shrink([false, true])
                            .show(ui, |ui| {
                                self.destination_panel(ui);
                            });
                        ui.add_space(4.0);
                        ui.separator();
                        ui.add_space(4.0);
                        self.verification_panel(ui);
                    });
            });
    }

    fn format_panel(&mut self, ui: &mut Ui) {
        let running = self.is_running();
        let mut available: Vec<Target> = Target::ALL
            .iter()
            .copied()
            .filter(|target| {
                *target != Target::Reblock
                    || self.jobs.iter().any(|j| j.analysis.can_target(*target))
            })
            .collect();
        if available.is_empty() {
            available.push(Target::Q4KM);
        }
        if !available.contains(&self.settings.target) {
            self.settings.target = available[0];
        }
        ui.add_enabled_ui(!running, |ui| {
            let standard: Vec<Target> = available
                .iter()
                .copied()
                .filter(|target| !target.is_ternary())
                .collect();
            let ternary: Vec<Target> = available
                .iter()
                .copied()
                .filter(|target| target.is_ternary())
                .collect();
            target_grid(ui, &standard, &mut self.settings.target);
            if !ternary.is_empty() {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Ternary").size(11.0).strong().color(TEXT));
                    ui.label(
                        RichText::new("only for ternary models")
                            .size(10.0)
                            .color(WARN),
                    );
                });
                ui.add_space(2.0);
                target_grid(ui, &ternary, &mut self.settings.target);
            }
            for job in &mut self.jobs {
                if job.analysis.can_target(self.settings.target) {
                    job.target = self.settings.target;
                }
            }
        });
    }

    fn destination_panel(&mut self, ui: &mut Ui) {
        let running = self.is_running();
        ui.add_enabled_ui(!running, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Destination").size(10.5).color(MUTED));
                help_dot(
                    ui,
                    "Write next to the source or choose a folder. Existing files are protected unless Overwrite is enabled.",
                );
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.checkbox(&mut self.settings.overwrite, "Overwrite")
                        .on_hover_text("Replace output files that already exist.");
                });
            });
            let custom = matches!(self.settings.output_mode, OutputMode::Custom(_));
            let label = match &self.settings.output_mode {
                OutputMode::NextToInput => "Next to source file".to_string(),
                OutputMode::Custom(dir) => shorten(&dir.display().to_string(), 30),
            };
            let button = egui::Button::new(RichText::new(label).size(10.5).color(STEEL_LIGHT))
                .fill(PANEL_2)
                .stroke(Stroke::new(1.0, BORDER))
                .corner_radius(CornerRadius::same(4))
                .min_size(Vec2::new(ui.available_width(), 32.0));
            if ui.add(button).clicked() {
                if let Some(picked) = rfd::FileDialog::new().pick_folder() {
                    self.last_output_dir = Some(picked.clone());
                    self.settings.output_mode = OutputMode::Custom(picked);
                }
            }
            if custom
                && ui
                    .add(
                        egui::Button::new(
                            RichText::new("Use source folder").size(9.5).color(MUTED),
                        )
                        .frame(false),
                    )
                    .clicked()
            {
                self.settings.output_mode = OutputMode::NextToInput;
            }
        });
    }

    fn verification_panel(&mut self, ui: &mut Ui) {
        let running = self.is_running();
        ui.add_enabled_ui(!running, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Verification").size(10.5).color(MUTED));
                help_dot(
                    ui,
                    "Structure checks the header, offsets and size. Full decodes every tensor again and is slower.",
                );
            });
            ui.add_space(4.0);
            let gap = 4.0;
            let width = ((ui.available_width() - gap * 2.0) / 3.0).max(54.0);
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = gap;
                for (mode, label, hint) in [
                    (
                        VerifyMode::None,
                        "Off",
                        "Write the file without re-reading it.",
                    ),
                    (
                        VerifyMode::Structure,
                        "Structure",
                        "Check header, offsets, alignment and size. Recommended.",
                    ),
                    (
                        VerifyMode::Full,
                        "Full",
                        "Decode every tensor again. Slowest and most thorough.",
                    ),
                ] {
                    let selected = self.settings.verify == mode;
                    let button =
                        egui::Button::new(RichText::new(label).size(10.0).color(if selected {
                            INK
                        } else {
                            STEEL_LIGHT
                        }))
                        .fill(if selected { ACCENT } else { PANEL_2 })
                        .stroke(Stroke::new(1.0, if selected { ACCENT } else { BORDER }))
                        .corner_radius(CornerRadius::same(4))
                        .min_size(Vec2::new(width, 30.0));
                    if ui.add(button).on_hover_text(hint).clicked() {
                        self.settings.verify = mode;
                    }
                }
            });
        });
    }

    fn suggest_target(&mut self) {
        let hardware = rustblock_core::detect_hardware();
        if hardware.vram_bytes.is_none() {
            self.log(
                LogLevel::Warn,
                "No NVIDIA GPU detected; the suggestion uses system RAM. Open Help (F1) for the fit estimate."
                    .to_string(),
            );
        }
        let largest = self.jobs.iter().max_by_key(|job| job.analysis.file_len);
        if let Some(job) = largest {
            if job.analysis.kind == FileKind::Pq2 {
                self.settings.target = Target::Reblock;
                self.log(
                    LogLevel::Info,
                    format!(
                        "{} is PQ2_0: the lossless Q2_0 reblock is the safe choice.",
                        job.file_name()
                    ),
                );
                return;
            }
            if hardware.ram_bytes.is_some() || hardware.vram_bytes.is_some() {
                match plan_budget(
                    &job.input,
                    BudgetRequest {
                        budget_bytes: None,
                        context: 4096,
                        overhead_bytes: 512 * 1024 * 1024,
                        hardware: Some(hardware),
                    },
                ) {
                    Ok(plan) => {
                        self.settings.target = plan.selected;
                        self.log(
                            LogLevel::Info,
                            format!(
                                "{}: suggested {} (~{} at 4096 tokens; {}).",
                                job.file_name(),
                                plan.selected.short(),
                                human_bytes(plan.selected_total_bytes),
                                plan.reason
                            ),
                        );
                        return;
                    }
                    Err(error) => {
                        self.log(
                            LogLevel::Warn,
                            format!("Could not estimate {}: {error}", job.file_name()),
                        );
                    }
                }
            }
        }
        let target = rustblock_core::suggest_target(hardware.ram_bytes.or(hardware.vram_bytes));
        self.settings.target = target;
        self.log(
            LogLevel::Info,
            format!(
                "Detected {} RAM / {} VRAM; suggested {}.",
                hardware
                    .ram_bytes
                    .map(human_bytes)
                    .unwrap_or_else(|| "unknown".to_string()),
                hardware
                    .vram_bytes
                    .map(human_bytes)
                    .unwrap_or_else(|| "unknown".to_string()),
                target.short()
            ),
        );
    }

    // ----- Hugging Face --------------------------------------------------
    fn start_search(&mut self) {
        if self.hf_busy() {
            return;
        }
        let query = self.hf_query.trim().to_string();
        if let Some(link) = hf::parse_link(&query) {
            let lower = query.to_ascii_lowercase();
            let points_at_file = lower.contains("/blob/") || lower.contains("/resolve/");
            let unsupported_file = points_at_file && link.file.is_none();
            self.open_link(link);
            if unsupported_file {
                self.hf_details_error = Some(
                    "That link does not point to a .gguf model. Only supported GGUF model files can be downloaded; choose one from the list below."
                        .to_string(),
                );
            }
            return;
        }
        if query.contains("huggingface.co") || query.contains("hf.co") {
            let error =
                "That link is not a model link. Use huggingface.co/author/model.".to_string();
            self.hf_state = HfState::Error(error.clone());
            self.log(LogLevel::Error, error);
            return;
        }
        let sort = self.hf_sort;
        let (tx, rx) = mpsc::channel();
        let ctx = self.ctx.clone();
        self.hf_state = HfState::Searching;
        self.selected_repo = None;
        self.selected_sha = "main".to_string();
        self.hf_details_loading = false;
        self.hf_details_error = None;
        self.pending_details = None;
        self.hf_files.clear();
        self.hf_file_filter.clear();
        self.hf_rx = Some(rx);
        self.hf_cancel = Arc::new(AtomicBool::new(false));
        self.hf_worker = Some(std::thread::spawn(move || {
            let _ = tx.send(HfEvent::SearchStarted);
            match hf::search(&query, sort) {
                Ok(results) => {
                    let _ = tx.send(HfEvent::SearchResults(results));
                }
                Err(error) => {
                    let _ = tx.send(HfEvent::Error(error));
                }
            }
            ctx.request_repaint();
        }));
    }

    fn open_link(&mut self, link: hf::HfLink) {
        if self.hf_busy() {
            return;
        }
        let repo = link.repo;
        let (tx, rx) = mpsc::channel();
        let ctx = self.ctx.clone();
        self.hf_query = repo.clone();
        self.hf_state = HfState::Searching;
        self.selected_repo = Some(repo.clone());
        self.selected_sha = "main".to_string();
        self.hf_details_loading = false;
        self.hf_details_error = None;
        self.pending_details = None;
        self.hf_files.clear();
        self.hf_file_filter.clear();
        self.hf_rx = Some(rx);
        self.hf_cancel = Arc::new(AtomicBool::new(false));
        self.log(
            LogLevel::Info,
            format!("Opening Hugging Face model {repo}."),
        );
        self.hf_worker = Some(std::thread::spawn(move || {
            let _ = tx.send(HfEvent::SearchStarted);
            match hf::fetch_model(&repo) {
                Ok(model) => {
                    let _ = tx.send(HfEvent::SearchResults(vec![model]));
                }
                Err(error) => {
                    let _ = tx.send(HfEvent::Error(error));
                }
            }
            ctx.request_repaint();
        }));
    }

    fn start_details(&mut self, repo: String, revision: String) {
        if self.hf_busy() {
            return;
        }
        self.selected_repo = Some(repo.clone());
        self.selected_sha = revision.clone();
        self.hf_details_loading = true;
        self.hf_details_error = None;
        self.hf_files.clear();
        self.hf_file_filter.clear();
        let (tx, rx) = mpsc::channel();
        let ctx = self.ctx.clone();
        self.hf_rx = Some(rx);
        self.hf_cancel = Arc::new(AtomicBool::new(false));
        self.hf_worker = Some(std::thread::spawn(move || {
            let _ = tx.send(HfEvent::DetailsStarted(repo.clone()));
            match hf::list_files(&repo, &revision) {
                Ok(files) => {
                    let _ = tx.send(HfEvent::DetailsLoaded { repo, files });
                }
                Err(error) => {
                    let _ = tx.send(HfEvent::DetailsError { repo, error });
                }
            }
            ctx.request_repaint();
        }));
    }

    fn start_download(&mut self, repo: String, file: HfFile) {
        let projector = file.projector;
        self.begin_download(repo, file, projector);
    }

    fn begin_download(&mut self, repo: String, file: HfFile, projector: bool) {
        if self.hf_busy() {
            return;
        }
        if !projector {
            if let Err(reason) = hf::file_support(&file.path) {
                self.log(
                    LogLevel::Warn,
                    format!("Not downloading {}: {reason}.", file.path),
                );
                return;
            }
        } else if !hf::is_projector(&file.path) {
            self.log(
                LogLevel::Warn,
                format!("Not downloading {}: not a vision projector.", file.path),
            );
            return;
        }
        let dest = self.settings.download_dir.clone();
        let revision = self.selected_sha.clone();
        let (tx, rx) = mpsc::channel();
        let ctx = self.ctx.clone();
        self.download = Some(DownloadUi {
            repo: repo.clone(),
            file: file.path.clone(),
            done: 0,
            total: file.size,
            speed: 0.0,
            quantized: !projector && looks_quantized_name(&file.path),
            projector,
            path: None,
            error: None,
        });
        self.hf_rx = Some(rx);
        self.hf_cancel = Arc::new(AtomicBool::new(false));
        let cancel = self.hf_cancel.clone();
        self.hf_worker = Some(std::thread::spawn(move || {
            let result = if projector {
                hf::download_projector(
                    &repo,
                    &revision,
                    &file.path,
                    &dest,
                    file.sha256.as_deref(),
                    &cancel,
                    &tx,
                )
            } else {
                hf::download_parallel(
                    &repo,
                    &revision,
                    &file.path,
                    &dest,
                    file.sha256.as_deref(),
                    &cancel,
                    &tx,
                )
            };
            if let Err(error) = result {
                if error != "download cancelled" {
                    let _ = tx.send(HfEvent::DownloadFailed {
                        repo,
                        file: file.path,
                        error,
                    });
                }
            }
            ctx.request_repaint();
        }));
    }

    fn poll_hf(&mut self) {
        let mut events = Vec::new();
        let mut disconnected = false;
        if let Some(rx) = &self.hf_rx {
            loop {
                match rx.try_recv() {
                    Ok(event) => events.push(event),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
        }
        for event in events {
            match event {
                HfEvent::SearchStarted => {}
                HfEvent::SearchResults(results) => {
                    self.log(
                        LogLevel::Info,
                        format!("Hugging Face: {} GGUF repositories found.", results.len()),
                    );
                    self.pending_details = results
                        .first()
                        .map(|model| (model.id.clone(), model.sha.clone()));
                    self.hf_state = HfState::Results(results);
                }
                HfEvent::DetailsStarted(repo) => {
                    if self.selected_repo.as_deref() == Some(repo.as_str()) {
                        self.hf_details_loading = true;
                    }
                }
                HfEvent::DetailsLoaded { repo, files } => {
                    if self.selected_repo.as_deref() == Some(repo.as_str()) {
                        self.hf_files = files;
                        self.hf_details_loading = false;
                        self.hf_scroll_top = true;
                    }
                }
                HfEvent::DetailsError { repo, error } => {
                    if self.selected_repo.as_deref() == Some(repo.as_str()) {
                        self.hf_details_loading = false;
                        self.hf_details_error = Some(error.clone());
                        self.log(LogLevel::Error, error);
                    }
                }
                HfEvent::Error(error) => {
                    self.hf_state = HfState::Error(error.clone());
                    self.hf_details_loading = false;
                    self.log(LogLevel::Error, error);
                }
                HfEvent::DownloadStarted { repo, file, total } => {
                    if let Some(d) = &mut self.download {
                        if d.repo == repo && d.file == file {
                            d.total = total;
                        }
                    }
                }
                HfEvent::DownloadProgress { done, total, speed } => {
                    if let Some(d) = &mut self.download {
                        d.done = done;
                        d.total = total;
                        d.speed = if d.speed <= 0.0 {
                            speed
                        } else {
                            d.speed * 0.75 + speed * 0.25
                        };
                    }
                }
                HfEvent::Downloaded {
                    repo,
                    file,
                    path,
                    projector,
                } => {
                    if let Some(d) = &mut self.download {
                        d.path = Some(path.clone());
                        d.done = d.total;
                    }
                    if projector {
                        self.log(
                            LogLevel::Ok,
                            format!(
                                "Downloaded projector {repo}/{file} to {}. Keep it next to the model for the runtime.",
                                path.display()
                            ),
                        );
                    } else {
                        self.log(
                            LogLevel::Ok,
                            format!("Downloaded {repo}/{file} to {}.", path.display()),
                        );
                        if self
                            .download
                            .as_ref()
                            .is_some_and(|download| download.quantized)
                        {
                            self.log(
                                LogLevel::Warn,
                                format!(
                                    "{file} is already quantized. Converting it again cannot recover the original precision; use a virgin F16/BF16 model if available."
                                ),
                            );
                        }
                        self.add_paths(vec![path]);
                    }
                }
                HfEvent::DownloadFailed { repo, file, error } => {
                    if let Some(d) = &mut self.download {
                        d.error = Some(error.clone());
                    }
                    self.log(
                        LogLevel::Error,
                        format!("Download failed for {repo}/{file}: {error}"),
                    );
                }
                HfEvent::DownloadCancelled => {
                    if let Some(d) = &mut self.download {
                        d.error = Some("Download cancelled.".to_string());
                    }
                    self.log(LogLevel::Warn, "Download cancelled.".to_string());
                }
            }
        }
        if disconnected {
            self.hf_rx = None;
            if let Some(handle) = self.hf_worker.take() {
                let _ = handle.join();
            }
            self.hf_details_loading = false;
        }
        if !self.hf_busy() {
            if let Some((repo, revision)) = self.pending_details.take() {
                self.start_details(repo, revision);
            }
        }
    }

    fn current_model(&self) -> Option<HfModel> {
        let HfState::Results(models) = &self.hf_state else {
            return None;
        };
        let repo = self.selected_repo.as_deref()?;
        models.iter().find(|m| m.id == repo).cloned()
    }

    // ----- Discover views --------------------------------------------------
    fn discover_page(&mut self, ui: &mut Ui) {
        if !self.discover_loaded && matches!(self.hf_state, HfState::Idle) {
            self.discover_loaded = true;
            self.start_search();
        }
        self.discover_header(ui);
        self.download_banner(ui);
        ui.add_space(10.0);
        ScrollArea::vertical()
            .id_salt("hf_results_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                self.hf_results(ui);
            });
    }

    fn discover_header(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.label(RichText::new("Discover").size(26.0).strong().color(TEXT));
                ui.label(
                    RichText::new("GGUF models on Hugging Face")
                        .size(11.5)
                        .color(MUTED),
                );
            });
        });
        ui.add_space(14.0);
        let is_link = hf::parse_link(&self.hf_query).is_some();
        ui.horizontal(|ui| {
            let field_width = (ui.available_width() - 104.0).max(220.0);
            egui::Frame::new()
                .fill(PANEL)
                .stroke(Stroke::new(1.0, BORDER))
                .corner_radius(CornerRadius::same(8))
                .inner_margin(Margin::symmetric(12, 0))
                .show(ui, |ui| {
                    let response = ui.add_sized(
                        [field_width - 24.0, 36.0],
                        TextEdit::singleline(&mut self.hf_query)
                            .hint_text("Search a model or paste a Hugging Face link")
                            .font(FontId::proportional(13.0))
                            .frame(egui::Frame::NONE)
                            .margin(Vec2::new(0.0, 8.0)),
                    );
                    if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        self.start_search();
                    }
                });
            let search = egui::Button::new(
                RichText::new(if is_link { "Open" } else { "Search" })
                    .size(11.5)
                    .strong()
                    .color(INK),
            )
            .fill(ACCENT)
            .stroke(Stroke::new(1.0, ACCENT))
            .corner_radius(CornerRadius::same(6))
            .min_size(Vec2::new(96.0, 36.0));
            if ui.add(search).clicked() {
                self.start_search();
            }
        });
        if matches!(self.hf_state, HfState::Results(_) | HfState::Searching) {
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                let count = match &self.hf_state {
                    HfState::Results(models) => format!("{} models", models.len()),
                    _ => "Searching…".to_string(),
                };
                ui.label(RichText::new(count).size(10.5).color(MUTED));
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let mut rerun = false;
                    for sort in [HfSort::Recent, HfSort::Likes, HfSort::Downloads] {
                        let active = self.hf_sort == sort;
                        let label = match sort {
                            HfSort::Downloads => "Downloads",
                            HfSort::Likes => "Likes",
                            HfSort::Recent => "Recent",
                        };
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new(label).size(10.0).color(if active {
                                        TEXT
                                    } else {
                                        MUTED
                                    }),
                                )
                                .frame(false),
                            )
                            .clicked()
                        {
                            self.hf_sort = sort;
                            rerun = true;
                        }
                    }
                    if rerun && !self.hf_query.trim().is_empty() {
                        self.start_search();
                    }
                });
            });
        }
    }

    fn hf_results(&mut self, ui: &mut Ui) {
        match self.hf_state.clone() {
            HfState::Idle => {
                egui::Frame::new()
                    .fill(Color32::from_rgba_unmultiplied(22, 25, 28, 168))
                    .stroke(Stroke::new(1.0, BORDER))
                    .corner_radius(CornerRadius::same(8))
                    .inner_margin(Margin::symmetric(32, 44))
                    .show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        ui.vertical_centered(|ui| {
                            ui.label(
                                RichText::new("Search the GGUF index")
                                    .size(18.0)
                                    .strong()
                                    .color(TEXT),
                            );
                            ui.add_space(4.0);
                            ui.label(
                                RichText::new(
                                    "Type a model name, an author or a quantization. You can also paste a model link.",
                                )
                                .size(11.5)
                                .color(MUTED),
                            );
                        });
                    });
            }
            HfState::Searching => {
                ui.add_space(18.0);
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(RichText::new("Searching…").size(11.5).color(STEEL_LIGHT));
                });
            }
            HfState::Error(error) => {
                ui.add_space(12.0);
                ui.label(RichText::new("Search failed").size(14.0).color(ERROR));
                ui.add(egui::Label::new(RichText::new(error).size(11.0).color(MUTED)).wrap());
            }
            HfState::Results(models) => {
                if models.is_empty() {
                    ui.add_space(18.0);
                    ui.label(RichText::new("No models found.").size(12.0).color(MUTED));
                    return;
                }
                let selected = self.selected_repo.clone();
                for model in &models {
                    let is_selected = selected.as_deref() == Some(model.id.as_str());
                    if hf_result_row(ui, model, is_selected) && !is_selected && !self.hf_busy() {
                        self.start_details(model.id.clone(), model.sha.clone());
                    }
                }
            }
        }
    }

    fn hf_details_panel(&mut self, ui: &mut Ui) {
        ui.set_max_width(ui.available_width());
        let Some(repo) = self.selected_repo.clone() else {
            ui.add_space(4.0);
            ui.label(
                RichText::new("Select a model")
                    .size(13.0)
                    .strong()
                    .color(TEXT),
            );
            ui.add_space(3.0);
            ui.label(
                RichText::new("Choose a result to see the GGUF files available to download.")
                    .size(11.0)
                    .color(MUTED),
            );
            return;
        };
        let model = self.current_model();
        let loading = self.hf_details_loading;
        let error = self.hf_details_error.clone();
        let repo_url = model
            .as_ref()
            .map(HfModel::web_url)
            .unwrap_or_else(|| format!("https://huggingface.co/{repo}"));

        if let Some(model) = &model {
            ui.label(RichText::new(&model.author).size(10.5).color(MUTED));
            ui.add(
                egui::Label::new(RichText::new(&model.name).size(18.0).strong().color(TEXT)).wrap(),
            );
            if let Some(license) = &model.license {
                ui.label(
                    RichText::new(format!("License: {license}"))
                        .size(10.0)
                        .color(STEEL_LIGHT),
                );
            }
        } else {
            ui.add(egui::Label::new(RichText::new(&repo).size(16.0).strong().color(TEXT)).wrap());
        }
        ui.horizontal(|ui| {
            ui.hyperlink_to("Open on Hugging Face ↗", &repo_url);
            if model.as_ref().is_some_and(|m| m.gated) {
                ui.label(RichText::new("Gated").size(10.0).color(WARN));
            }
        });
        if loading {
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(RichText::new("Loading files…").size(10.5).color(MUTED));
            });
        }
        if let Some(error) = error {
            ui.add_space(8.0);
            ui.add(egui::Label::new(RichText::new(error).size(10.5).color(ERROR)).wrap());
        }

        let all: Vec<HfFile> = self
            .hf_files
            .iter()
            .filter(|file| file.supported || file.projector)
            .cloned()
            .collect();
        let filter = self.hf_file_filter.trim().to_ascii_lowercase();
        let files: Vec<HfFile> = all
            .iter()
            .filter(|file| filter.is_empty() || file.path.to_ascii_lowercase().contains(&filter))
            .cloned()
            .collect();

        ui.add_space(14.0);
        ui.separator();
        ui.add_space(9.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new("Files").size(12.0).strong().color(TEXT));
            ui.label(
                RichText::new(all.len().to_string())
                    .size(10.5)
                    .monospace()
                    .color(MUTED),
            );
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.add_sized(
                    [132.0, 26.0],
                    TextEdit::singleline(&mut self.hf_file_filter)
                        .hint_text("Filter files")
                        .font(FontId::proportional(10.0)),
                );
            });
        });
        ui.add_space(4.0);
        if loading && all.is_empty() {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(RichText::new("Loading…").size(10.5).color(MUTED));
            });
        } else if all.is_empty() {
            ui.label(
                RichText::new("No downloadable GGUF files.")
                    .size(10.5)
                    .color(MUTED),
            );
        } else if files.is_empty() {
            ui.label(RichText::new("No files match.").size(10.5).color(MUTED));
        }
        let mut download_start: Option<(String, HfFile)> = None;
        for file in &files {
            let cached = self
                .settings
                .download_dir
                .join(hf::file_name(&file.path))
                .exists();
            hf_file_row(ui, &repo, file, self.hf_busy(), cached, &mut download_start);
        }
        if let Some((repo, file)) = download_start {
            self.start_download(repo, file);
        }
    }

    fn hf_destination(&mut self, ui: &mut Ui) {
        ui.label(RichText::new("Save downloads to").size(9.5).color(MUTED));
        let download_dir = self.settings.download_dir.clone();
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(shorten(&download_dir.display().to_string(), 30))
                    .size(9.5)
                    .monospace()
                    .color(STEEL_LIGHT),
            )
            .on_hover_text(download_dir.display().to_string());
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui
                    .add(egui::Button::new(RichText::new("Change").size(9.5)).frame(false))
                    .clicked()
                {
                    if let Some(picked) = rfd::FileDialog::new().pick_folder() {
                        self.settings.download_dir = picked;
                    }
                }
            });
        });
        let quota_bytes = self
            .settings
            .cache_quota_gib
            .saturating_mul(1024 * 1024 * 1024);
        let stats = cache_stats(&download_dir, quota_bytes);
        ui.horizontal(|ui| {
            ui.label(RichText::new("Cache limit").size(9.5).color(MUTED));
            ui.add(
                egui::DragValue::new(&mut self.settings.cache_quota_gib)
                    .speed(1.0)
                    .suffix(" GiB"),
            );
        });
        match &stats {
            Ok(stats) => {
                ui.label(
                    RichText::new(format!(
                        "{} files · {} / {}",
                        stats.entries.len(),
                        human_bytes(stats.total_bytes),
                        human_bytes(stats.quota_bytes)
                    ))
                    .size(9.0)
                    .color(if stats.over_quota() { WARN } else { MUTED }),
                );
            }
            Err(error) => {
                ui.label(RichText::new(error).size(9.0).color(ERROR));
            }
        }
        ui.horizontal(|ui| {
            if ui
                .add(egui::Button::new(RichText::new("Open folder").size(9.5)).frame(false))
                .clicked()
            {
                open_folder(&download_dir);
            }
            if ui
                .add(
                    egui::Button::new(RichText::new("Clear partial downloads").size(9.5))
                        .frame(false),
                )
                .clicked()
            {
                match clear_partial_files(&download_dir) {
                    Ok(0) => self.log(LogLevel::Info, "No partial downloads to clear.".to_string()),
                    Ok(count) => self.log(
                        LogLevel::Ok,
                        format!("Removed {count} partial download file(s)."),
                    ),
                    Err(error) => self.log(LogLevel::Error, error),
                }
            }
            if ui
                .add_enabled(
                    !self.hf_busy(),
                    egui::Button::new(RichText::new("Trim cache").size(9.5)).frame(false),
                )
                .clicked()
            {
                match trim_cache(&download_dir, quota_bytes) {
                    Ok(report) if report.removed == 0 => self.log(
                        LogLevel::Info,
                        "Cache is already within the quota.".to_string(),
                    ),
                    Ok(report) => self.log(
                        LogLevel::Ok,
                        format!(
                            "Removed {} cache file(s), freed {}; {} remain.",
                            report.removed,
                            human_bytes(report.freed_bytes),
                            human_bytes(report.remaining_bytes)
                        ),
                    ),
                    Err(error) => self.log(LogLevel::Error, error),
                }
            }
        });
    }

    // ----- GGUF inspector view ------------------------------------------------
    fn inspector_page(&mut self, ui: &mut Ui) {
        ui.label(
            RichText::new("GGUF INSPECTOR")
                .size(26.0)
                .strong()
                .color(TEXT),
        );
        ui.add_space(14.0);
        ui.horizontal(|ui| {
            if ui
                .add(egui::Button::new(
                    RichText::new("Add file").size(12.0).strong().color(INK),
                ))
                .clicked()
            {
                self.inspector_open();
            }
            if self.inspector_path.is_some() {
                if ui
                    .add(egui::Button::new(
                        RichText::new("Inspect another")
                            .size(10.5)
                            .color(STEEL_LIGHT),
                    ))
                    .clicked()
                {
                    self.inspector_open();
                }
                if let Some(path) = &self.inspector_path {
                    if ui
                        .add(egui::Button::new(
                            RichText::new("Open folder").size(10.5).color(STEEL_LIGHT),
                        ))
                        .clicked()
                    {
                        open_folder(path);
                    }
                }
            }
        });
        ui.add_space(10.0);
        if self.inspector.is_none() {
            match &self.inspector_path {
                None => {
                    ui.vertical_centered(|ui| {
                        ui.label(RichText::new("No file selected").size(14.0).color(MUTED));
                        ui.add_space(6.0);
                        ui.label(
                            RichText::new(
                                "Open a GGUF file to see its facts, metadata and tensor types.",
                            )
                            .size(10.5)
                            .color(MUTED),
                        );
                    });
                }
                Some(path) => {
                    ui.vertical_centered(|ui| {
                        ui.label(
                            RichText::new(format!(
                                "Cannot read {} — it may not be a valid GGUF.",
                                shorten(&file_name_of(path), 64)
                            ))
                            .size(10.5)
                            .color(ERROR),
                        );
                    });
                }
            }
            return;
        }
        let Some(data) = self.inspector.clone() else {
            return;
        };
        ui.horizontal(|ui| {
            egui::Frame::new()
                .fill(PANEL_2)
                .stroke(Stroke::new(1.0, BORDER))
                .corner_radius(CornerRadius::same(8))
                .inner_margin(Margin::symmetric(16, 12))
                .show(ui, |ui| {
                    ui.vertical(|ui| {
                        ui.label(RichText::new(shorten(&file_name_of(&data.path), 56)).size(13.0).strong().color(TEXT));
                        ui.label(
                            RichText::new(format!(
                                "{} · GGUF v{} · alignment {} B · {} tensors · data at {}",
                                human_bytes(data.file_len),
                                data.version,
                                data.alignment,
                                data.tensors,
                                human_bytes(data.data_start)
                            ))
                            .size(10.0)
                            .color(MUTED),
                        );
                        ui.label(
                            RichText::new(format!("file_type {:?}", data.file_type))
                                .size(9.5)
                                .color(MUTED),
                        );
                        if data.ptq1 > 0 {
                            ui.label(
                                RichText::new("Contains PTQ1_0 (Prism) — this build cannot convert it losslessly.")
                                    .size(9.5)
                                    .color(WARN),
                            );
                        }
                        if data.pq2 > 0 {
                            ui.label(
                                RichText::new(format!("{} PQ2_0 (Prism ternary) tensors.", data.pq2))
                                    .size(9.5)
                                    .color(INFO),
                            );
                        }
                        if let Some(fit) = &data.fit {
                            let (label, color, _hint) = fit_badge(fit);
                            ui.label(RichText::new(label).strong().color(color));
                        }
                    });
                });
        });
        ui.add_space(10.0);

        // Metadata.
        ui.label(
            RichText::new("Metadata")
                .size(12.0)
                .strong()
                .color(STEEL_LIGHT),
        );
        if data.metadata.is_empty() {
            ui.label(RichText::new("no metadata entries").size(9.5).color(MUTED));
        } else {
            for (key, value) in &data.metadata {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(key).size(9.5).color(STEEL_LIGHT));
                    ui.spacing_mut().item_spacing.x = 4.0;
                    ui.label(
                        RichText::new(value.as_str())
                            .size(9.5)
                            .monospace()
                            .color(STEEL),
                    );
                });
            }
        }
        ui.add_space(10.0);

        // Tensor histogram: type, count, disk bytes and a bar.
        ui.label(
            RichText::new("Tensor types")
                .size(12.0)
                .strong()
                .color(STEEL_LIGHT),
        );
        let max_count = data
            .histogram
            .iter()
            .map(|(_, count, _)| *count)
            .max()
            .unwrap_or(0)
            .max(1);
        for (name, count, bytes) in &data.histogram {
            let frac = *count as f64 / max_count as f64;
            let bar_len = (frac * 20.0).round() as usize;
            let mut bar = "█".repeat(bar_len);
            bar.push_str(&"░".repeat(20 - bar_len));
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(name)
                        .size(9.5)
                        .monospace()
                        .color(STEEL_LIGHT)
                        .strong(),
                );
                ui.spacing_mut().item_spacing.x = 4.0;
                ui.label(
                    RichText::new(format!("{count:>5} · {:>7}", human_bytes(*bytes)))
                        .size(9.0)
                        .monospace()
                        .color(MUTED),
                );
                ui.label(RichText::new(bar).size(9.5).monospace().color(STEEL));
            });
        }
        if data.histogram.is_empty() {
            ui.label(RichText::new("no tensors").size(9.5).color(MUTED));
        }
        ui.add_space(10.0);

        // Memory estimate.
        ui.label(
            RichText::new("Memory (4096 tokens)")
                .size(12.0)
                .strong()
                .color(STEEL_LIGHT),
        );
        ui.label(
            RichText::new(format!("payload {}", human_bytes(data.weights_bytes)))
                .size(9.5)
                .color(MUTED),
        );
        if let Some(fit) = &data.fit {
            ui.label(
                RichText::new(format!(
                    "weights {} · kv {} · overhead {} · total {}",
                    human_bytes(fit.weights_bytes),
                    human_bytes(fit.kv_cache_bytes),
                    human_bytes(fit.overhead_bytes),
                    human_bytes(fit.total_bytes)
                ))
                .size(9.5)
                .color(MUTED),
            );
            let (label, color, _hint) = fit_badge(fit);
            ui.label(RichText::new(label).strong().color(color));
        }
    }

    // ----- Log panel ------------------------------------------------------------
    fn log_panel(&mut self, ui: &mut Ui) {
        egui::Panel::bottom("log_panel")
            .exact_size(208.0)
            .resizable(true)
            .frame(
                egui::Frame::new()
                    .fill(PANEL)
                    .stroke(Stroke::new(1.0, BORDER))
                    .inner_margin(Margin::symmetric(16, 10)),
            )
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("ACTIVITY LOG")
                            .size(9.5)
                            .strong()
                            .color(STEEL_LIGHT),
                    );
                    ui.add_space(8.0);
                    for level in [
                        LogLevel::Info,
                        LogLevel::Ok,
                        LogLevel::Warn,
                        LogLevel::Error,
                    ] {
                        let active = self.log_levels[level.index()];
                        let color = level.color();
                        let response = ui.add(
                            egui::Button::new(
                                RichText::new(level.short()).size(8.5).color(if active {
                                    INK
                                } else {
                                    color
                                }),
                            )
                            .fill(if active { color } else { Color32::TRANSPARENT })
                            .stroke(Stroke::new(1.0, if active { color } else { BORDER }))
                            .corner_radius(CornerRadius::same(4)),
                        );
                        if response.clicked() {
                            self.log_levels[level.index()] = !active;
                        }
                    }
                    let visible = self
                        .logs
                        .iter()
                        .filter(|entry| {
                            log_entry_visible(entry, &self.log_filter, &self.log_levels)
                        })
                        .count();
                    ui.label(
                        RichText::new(format!("{visible}/{}", self.logs.len()))
                            .size(9.0)
                            .color(STEEL_LIGHT),
                    );
                    ui.add_sized(
                        [140.0, 26.0],
                        TextEdit::singleline(&mut self.log_filter)
                            .hint_text("Filter")
                            .font(FontId::proportional(9.5)),
                    );
                    if ui
                        .button(RichText::new("Clear").size(9.5).color(MUTED))
                        .clicked()
                    {
                        self.logs.clear();
                    }
                    if ui
                        .button(RichText::new("Close").size(9.5).color(MUTED))
                        .clicked()
                    {
                        self.show_log = false;
                    }
                });
                ui.separator();
                let filter = self.log_filter.to_ascii_lowercase();
                ScrollArea::vertical()
                    .id_salt("log_scroll")
                    .stick_to_bottom(true)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for entry in self
                            .logs
                            .iter()
                            .filter(|entry| log_entry_visible(entry, &filter, &self.log_levels))
                        {
                            let level = entry.level;
                            ui.label(
                                RichText::new(entry.text.as_str())
                                    .size(10.0)
                                    .monospace()
                                    .color(level.color()),
                            );
                        }
                    });
            });
    }

    // ----- Help modal ------------------------------------------------------------
    fn help_modal(&mut self, ctx: &egui::Context) {
        if !self.show_help {
            return;
        }
        let mut open = true;
        egui::Window::new("RustBlock help")
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .show(ctx, |ui| {
                ui.set_max_width(560.0);
                ui.label(RichText::new("What the buttons do").size(15.0).strong().color(TEXT));
                ui.add_space(6.0);
                for (title, text) in [
                    ("Reblock (Q2_0)", "Lossless conversion of Prism PQ2_0 files to the upstream Q2_0 layout. Use it when the badge says PQ2_0."),
                    ("Q4_K_M", "The usual balanced choice for a float model that must fit on a normal desktop. Q5_K_M keeps more quality; Q3_K_M saves more space."),
                    ("F16 / BF16", "Uncompressed formats. They are large, but they are the safest source if you plan to quantize again later."),
                    ("Structure / Full verification", "Structure checks the header and offsets. Full decodes every tensor again and is slower."),
                    ("Fit badge", "An estimate at 4096 tokens: weights plus KV cache plus a fixed runtime overhead. It is a warning, not a guarantee."),
                    ("GPU not detected", "If VRAM is unknown, the fit badge uses system RAM and marks VRAM as not detected. RustBlock never assumes a GPU; install a supported NVIDIA driver or check the GPU manually. The estimate is a warning, not a guarantee."),
                    ("Benchmark", "Run llama-bench -m <model> -ngl 99 on the largest finished model. Compare tokens/s with other files; install llama.cpp tools to enable it."),
                    ("GGUF inspector", "Inspect a local GGUF: file facts, metadata and the tensor type histogram with byte sizes. Files visualized this way are the same ones the Hugging Face Hub shows on model pages."),
                    ("Discover license", "Licenses from the Hub are shown on model cards when the API exposes them. RustBlock never modifies licenses."),
                    ("Log level filters", "The activity log colors levels; use the I/O/⚠/✗ buttons to filter by level and the field to filter by text."),
                    ("Job details", "Select a job in the queue: fit details, output path and file facts. Details update automatically as the batch runs."),
                ] {
                    ui.label(RichText::new(title).size(11.0).strong().color(STEEL_LIGHT));
                    ui.add(egui::Label::new(RichText::new(text).size(10.5).color(MUTED)).wrap());
                    ui.add_space(5.0);
                }
                ui.separator();
                ui.label(RichText::new("Shortcuts").size(11.0).strong().color(STEEL_LIGHT));
                ui.label(RichText::new("Ctrl+O add files · Ctrl+S start · Ctrl+L log · F1 help").size(10.5).monospace().color(MUTED));
            });
        if !open {
            self.show_help = false;
        }
    }

    fn run_logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let (open, save, help, log) = ctx.input_mut(|input| {
            (
                input.consume_key(egui::Modifiers::CTRL, egui::Key::O),
                input.consume_key(egui::Modifiers::CTRL, egui::Key::S),
                input.consume_key(egui::Modifiers::NONE, egui::Key::F1),
                input.consume_key(egui::Modifiers::CTRL, egui::Key::L),
            )
        });
        if open {
            self.browse();
        }
        if save && !self.is_running() {
            self.start(WorkKind::Convert);
        }
        if help {
            self.show_help = !self.show_help;
        }
        if log {
            self.show_log = !self.show_log;
        }
        if self.is_running() && ctx.input(|i| i.viewport().close_requested()) {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
        }
        self.handle_dropped(ctx);
        self.poll_worker(ctx);
        self.poll_hf();
        self.poll_benchmark();
        if self.rx.is_some() || self.hf_rx.is_some() {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }

    fn render_ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        self.top_chrome(ui);
        if self.show_log {
            self.log_panel(ui);
        }
        match self.tab {
            Tab::Convert => self.settings_rail(ui),
            Tab::Discover => self.hf_inspector_rail(ui),
            Tab::Inspector => {}
        }
        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(BG)
                    .inner_margin(Margin::symmetric(22, 16)),
            )
            .show(ui, |ui| {
                self.paint_backdrop(ui);
                match self.tab {
                    Tab::Convert => self.convert_page(ui),
                    Tab::Discover => self.discover_page(ui),
                    Tab::Inspector => self.inspector_page(ui),
                }
            });
        self.drag_overlay(ui.ctx());
        self.requantize_modal(ui.ctx());
        self.help_modal(ui.ctx());
        resize_edges(ui);
        paint_window_border(ui.ctx());
    }

    fn top_chrome(&mut self, ui: &mut Ui) {
        let mark = self.mark.clone();
        let wordmark = self.wordmark.clone();
        egui::Panel::top("chrome")
            .exact_size(56.0)
            .frame(
                egui::Frame::new()
                    .fill(PANEL)
                    .inner_margin(Margin::symmetric(14, 8))
                    .stroke(Stroke::new(1.0, BORDER)),
            )
            .show(ui, |ui| {
                let controls_width = 196.0;
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 8.0;
                    self.brand(ui, mark.as_ref(), wordmark.as_ref());
                    ui.add_space(8.0);
                    ui.separator();
                    ui.add_space(8.0);
                    self.nav_tab(ui, Tab::Convert, "Convert");
                    ui.add_space(2.0);
                    self.nav_tab(ui, Tab::Discover, "Discover");
                    ui.add_space(2.0);
                    self.nav_tab(ui, Tab::Inspector, "Inspector");
                    let drag_width = (ui.available_width() - controls_width).max(0.0);
                    if drag_width > 1.0 {
                        let (_, response) = ui.allocate_exact_size(
                            Vec2::new(drag_width, 30.0),
                            Sense::click_and_drag(),
                        );
                        if response.drag_started() {
                            ui.ctx().send_viewport_cmd(egui::ViewportCommand::StartDrag);
                        }
                        if response.double_clicked() {
                            toggle_maximized(ui.ctx());
                        }
                    }
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if window_button(ui, "×", "Close", true) {
                            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                        if window_button(ui, "—", "Minimize", false) {
                            ui.ctx()
                                .send_viewport_cmd(egui::ViewportCommand::Minimized(true));
                        }
                        ui.add_space(6.0);
                        if ui
                            .button(RichText::new("Log").size(10.0).color(MUTED))
                            .on_hover_text("Activity log · Ctrl+L")
                            .clicked()
                        {
                            self.show_log = !self.show_log;
                        }
                        if ui
                            .button(RichText::new("?").size(12.0).color(MUTED))
                            .on_hover_text("Help · F1")
                            .clicked()
                        {
                            self.show_help = true;
                        }
                    });
                });
            });
    }

    fn brand(
        &mut self,
        ui: &mut Ui,
        mark: Option<&egui::TextureHandle>,
        wordmark: Option<&egui::TextureHandle>,
    ) {
        theme::paint_logo(ui, 24.0, mark);
        ui.add_space(8.0);
        let Some(texture) = wordmark else {
            return;
        };
        let height = 18.0;
        let size = texture.size_vec2();
        let width = height * (size.x / size.y).max(0.1);
        let (rect, response) =
            ui.allocate_exact_size(Vec2::new(width, height), Sense::click_and_drag());
        theme::paint_wordmark(ui.painter(), rect, Some(texture), Color32::WHITE);
        if response.drag_started() {
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::StartDrag);
        }
        if response.double_clicked() {
            toggle_maximized(ui.ctx());
        }
        let _ = response.on_hover_text("Native GGUF toolkit");
    }

    fn nav_tab(&mut self, ui: &mut Ui, tab: Tab, label: &str) {
        let active = self.tab == tab;
        let response = ui.add_sized(
            [88.0, 30.0],
            egui::Button::new(
                RichText::new(label)
                    .size(12.5)
                    .color(if active { TEXT } else { MUTED })
                    .strong(),
            )
            .fill(if active {
                PANEL_3
            } else {
                Color32::TRANSPARENT
            })
            .stroke(Stroke::new(
                1.0,
                if active { BORDER } else { Color32::TRANSPARENT },
            ))
            .corner_radius(CornerRadius::same(6)),
        );
        if response.clicked() {
            self.tab = tab;
        }
    }

    fn paint_backdrop(&self, ui: &Ui) {
        let Some(texture) = self.mark.as_ref() else {
            return;
        };
        let area = ui.max_rect();
        let rect = backdrop_rect(area);
        ui.painter().with_clip_rect(area).image(
            texture.id(),
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            Color32::from_rgba_unmultiplied(255, 255, 255, 26),
        );
    }

    fn drag_overlay(&self, ctx: &egui::Context) {
        let hovering = ctx.input(|i| !i.raw.hovered_files.is_empty());
        if !hovering {
            return;
        }
        let rect = ctx.content_rect();
        let painter = ctx.layer_painter(LayerId::new(Order::Foreground, Id::new("drop_overlay")));
        painter.rect_filled(
            rect,
            CornerRadius::same(0),
            Color32::from_rgba_unmultiplied(BG.r(), BG.g(), BG.b(), 236),
        );
        painter.rect_stroke(
            rect.shrink(14.0),
            CornerRadius::same(0),
            Stroke::new(1.5, ACCENT),
            StrokeKind::Inside,
        );
        painter.text(
            rect.center() - Vec2::new(0.0, 10.0),
            Align2::CENTER_CENTER,
            "Drop GGUF files",
            FontId::proportional(28.0),
            ACCENT,
        );
        painter.text(
            rect.center() + Vec2::new(0.0, 28.0),
            Align2::CENTER_CENTER,
            "Release to add them to the queue.",
            FontId::proportional(13.0),
            TEXT,
        );
    }

    fn hf_inspector_rail(&mut self, ui: &mut Ui) {
        let scroll_top = std::mem::take(&mut self.hf_scroll_top);
        egui::Panel::right("hf_inspector_rail")
            .exact_size(380.0)
            .resizable(true)
            .frame(
                egui::Frame::new()
                    .fill(PANEL)
                    .inner_margin(Margin::symmetric(0, 0))
                    .stroke(Stroke::new(1.0, BORDER)),
            )
            .show(ui, |ui| {
                egui::Frame::new()
                    .inner_margin(Margin::symmetric(20, 18))
                    .show(ui, |ui| {
                        let max_height = (ui.available_height() - 78.0).max(120.0);
                        let mut scroll = ScrollArea::vertical()
                            .id_salt("hf_details_scroll")
                            .max_height(max_height)
                            .max_width(ui.available_width())
                            .auto_shrink([false, false]);
                        if scroll_top {
                            scroll = scroll.vertical_scroll_offset(0.0);
                        }
                        scroll.show(ui, |ui| {
                            self.hf_details_panel(ui);
                        });
                        ui.add_space(6.0);
                        ui.separator();
                        ui.add_space(8.0);
                        self.hf_destination(ui);
                    });
            });
    }
}

impl eframe::App for RustBlockApp {
    fn logic(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        RustBlockApp::run_logic(self, ctx, frame);
    }

    fn ui(&mut self, ui: &mut Ui, frame: &mut eframe::Frame) {
        RustBlockApp::render_ui(self, ui, frame);
    }
}

// ----- free helpers --------------------------------------------------------

fn job_card(
    ui: &mut Ui,
    job: &Job,
    running: bool,
    selected: bool,
    remove: &mut Option<u64>,
    open: &mut Option<PathBuf>,
    toggle: &mut Option<u64>,
) {
    egui::Frame::new()
        .fill(if selected { PANEL_2 } else { PANEL })
        .stroke(Stroke::new(1.0, if selected { STEEL } else { BORDER }))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(Margin::symmetric(14, 12))
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            ui.vertical(|ui| {
                // Row 1: name (selectable) + status + actions.
                ui.horizontal(|ui| {
                    if ui
                        .selectable_label(
                            selected,
                            RichText::new(job.file_name())
                                .size(13.0)
                                .strong()
                                .color(TEXT),
                        )
                        .clicked()
                    {
                        *toggle = Some(job.id);
                    }
                    if let Some(fit) = &job.analysis.fit {
                        let (label, color, hint) = fit_badge(fit);
                        ui.label(RichText::new(label).size(9.0).strong().color(color))
                            .on_hover_text(&hint);
                    }
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui
                            .add_enabled(
                                !running,
                                egui::Button::new(RichText::new("Remove").size(9.5).color(MUTED))
                                    .frame(false),
                            )
                            .clicked()
                        {
                            *remove = Some(job.id);
                        }
                        ui.label(
                            RichText::new(status_label(job))
                                .size(9.5)
                                .color(status_color(job)),
                        );
                    });
                });
                // Row 2: facts.
                ui.add(egui::Label::new(
                    RichText::new(format!(
                        "{} · {} · {}",
                        human_bytes(job.analysis.file_len),
                        job.analysis.kind_label(),
                        job.target.short()
                    ))
                    .size(10.0)
                    .color(MUTED),
                ))
                .on_hover_text(format!(
                    "GGUF v{} · alignment {} B · {} tensors · {}",
                    job.analysis.version,
                    job.analysis.alignment,
                    job.analysis.tensors,
                    job.analysis.type_summary()
                ));
                if let Some(output) = done_output_name(job) {
                    ui.add(egui::Label::new(
                        RichText::new(format!("→ {output}"))
                            .size(9.5)
                            .color(STEEL_LIGHT),
                    ));
                }
                // Row 3: progress / result / error.
                match &job.status {
                    JobStatus::Running(state) => {
                        let phase = match state.phase {
                            Phase::Converting => "Converting",
                            Phase::Verifying => "Verifying",
                        };
                        ui.add(
                            ProgressBar::new(state.fraction())
                                .fill(ACCENT)
                                .animate(true)
                                .desired_width(240.0)
                                .text(format!("{phase} · {:.0}%", state.fraction() * 100.0)),
                        );
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new(format!("{}/s", human_speed(state.speed)))
                                    .size(9.0)
                                    .color(MUTED),
                            );
                            if let Some(eta) = state.eta() {
                                ui.label(
                                    RichText::new(format!("{} left", human_duration(eta)))
                                        .size(9.0)
                                        .color(MUTED),
                                );
                            }
                        });
                    }
                    JobStatus::Done { report, target } => {
                        ui.horizontal_wrapped(|ui| {
                            ui.label(
                                RichText::new(format!(
                                    "{} → {} · {} · {}",
                                    human_bytes(report.old_file_size),
                                    human_bytes(report.new_file_size),
                                    target.short(),
                                    human_duration(report.elapsed)
                                ))
                                .size(10.0)
                                .color(STEEL_LIGHT),
                            );
                            if ui
                                .add(
                                    egui::Button::new(
                                        RichText::new("Open folder").size(9.5).color(ACCENT),
                                    )
                                    .frame(false),
                                )
                                .clicked()
                            {
                                *open = Some(report.output.clone());
                            }
                        });
                    }
                    JobStatus::Verified { outcome, elapsed } => {
                        ui.label(
                            RichText::new(format!(
                                "{} · {} tensors · {} · {}",
                                outcome.label,
                                outcome.tensors,
                                match outcome.kind {
                                    VerifyKind::Exact => "exact",
                                    VerifyKind::Structure => "structure",
                                },
                                human_duration(*elapsed)
                            ))
                            .size(9.5)
                            .color(STEEL_LIGHT),
                        );
                    }
                    JobStatus::Failed(error) => {
                        ui.add(
                            egui::Label::new(RichText::new(error).size(10.0).color(ERROR)).wrap(),
                        );
                    }
                    JobStatus::Ready => {}
                }
            });
        });
}

fn status_label(job: &Job) -> String {
    match &job.status {
        JobStatus::Ready => {
            if job.analysis.error.is_some() {
                "INVALID".to_string()
            } else if job.analysis.ptq1 > 0 {
                "UNSUPPORTED".to_string()
            } else if job.analysis.supported {
                "READY".to_string()
            } else {
                "UNSUPPORTED".to_string()
            }
        }
        JobStatus::Running(state) => match state.phase {
            Phase::Converting => "CONVERTING".to_string(),
            Phase::Verifying => "VERIFYING".to_string(),
        },
        JobStatus::Done { .. } => "DONE".to_string(),
        JobStatus::Verified { .. } => "VERIFIED".to_string(),
        JobStatus::Failed(_) => "ERROR".to_string(),
    }
}

fn status_color(job: &Job) -> Color32 {
    match &job.status {
        JobStatus::Ready => match job.analysis.kind {
            FileKind::Pq2 => INFO,
            FileKind::Q2 | FileKind::Float | FileKind::Quantized | FileKind::Mixed => MUTED,
            FileKind::Ptq1 => WARN,
            FileKind::Invalid | FileKind::Other => ERROR,
        },
        JobStatus::Running(_) | JobStatus::Done { .. } | JobStatus::Verified { .. } => ACCENT,
        JobStatus::Failed(_) => ERROR,
    }
}

fn fit_badge(fit: &FitEstimate) -> (String, Color32, String) {
    let mut hint = format!(
        "Estimated at {} tokens: {} weights + {} KV cache + {} overhead.",
        fit.context,
        human_bytes(fit.weights_bytes),
        human_bytes(fit.kv_cache_bytes),
        human_bytes(fit.overhead_bytes)
    );
    hint.push_str(&format!(
        "\nRAM: {}.",
        fit_verdict_text(fit.hardware.ram_bytes, fit.fits_ram)
    ));
    hint.push_str(&format!(
        "\nVRAM: {}.",
        fit_verdict_text(fit.hardware.vram_bytes, fit.fits_vram)
    ));
    let (label, color) = if fit.fits_vram == Some(true) {
        fit_label("VRAM", fit.hardware.vram_bytes, fit.total_bytes)
    } else if fit.fits_ram == Some(true) {
        fit_label("RAM", fit.hardware.ram_bytes, fit.total_bytes)
    } else if fit.fits_ram == Some(false) && fit.fits_vram == Some(false) {
        ("No fit".to_string(), ERROR)
    } else if fit.fits_ram == Some(false) {
        ("No RAM fit".to_string(), ERROR)
    } else if fit.fits_vram == Some(false) {
        ("No VRAM fit".to_string(), ERROR)
    } else {
        ("Fit unknown".to_string(), MUTED)
    };
    (label, color, hint)
}

fn fit_label(device: &str, available: Option<u64>, total: u64) -> (String, Color32) {
    let Some(available) = available else {
        return (format!("Fits {device}"), OK);
    };
    if available > 0 && total.saturating_mul(100) > available.saturating_mul(88) {
        (format!("Tight {device}"), WARN)
    } else {
        (format!("Fits {device}"), OK)
    }
}

fn fit_verdict_text(available: Option<u64>, fits: Option<bool>) -> String {
    match (available, fits) {
        (Some(bytes), Some(true)) => format!("fits in {}", human_bytes(bytes)),
        (Some(bytes), Some(false)) => format!("does not fit in {}", human_bytes(bytes)),
        (Some(bytes), None) => format!("{} detected; verdict unavailable", human_bytes(bytes)),
        (None, _) => "not detected".to_string(),
    }
}

/// Fit verdict available before a download. Only the published weight size is
/// known, so the KV cache is excluded and the tooltip says so instead of
/// pretending the number is a full estimate.
fn size_fit_badge(weights: u64, hardware: rustblock_core::Hardware) -> (String, Color32, String) {
    let total = weights.saturating_add(512 * 1024 * 1024);
    let hint = format!(
        "Download estimate: {} weights + {} runtime overhead. The KV cache needs the model header, so it is not included before download.",
        human_bytes(weights),
        human_bytes(512 * 1024 * 1024)
    );
    let fits_ram = hardware
        .ram_bytes
        .map(|available| rustblock_core::hardware_fits(total, available));
    let fits_vram = hardware
        .vram_bytes
        .map(|available| rustblock_core::hardware_fits(total, available));
    let (label, color) = if fits_vram == Some(true) {
        fit_label("VRAM", hardware.vram_bytes, total)
    } else if fits_ram == Some(true) {
        fit_label("RAM", hardware.ram_bytes, total)
    } else if fits_ram == Some(false) && fits_vram == Some(false) {
        ("No fit".to_string(), ERROR)
    } else if fits_ram == Some(false) {
        ("No RAM fit".to_string(), ERROR)
    } else if fits_vram == Some(false) {
        ("No VRAM fit".to_string(), ERROR)
    } else {
        ("Fit unknown".to_string(), MUTED)
    };
    (label, color, hint)
}

/// Hardware detection is slow on Windows (PowerShell and `nvidia-smi`), so the
/// Discover list reuses one detection per process.
fn detected_hardware() -> rustblock_core::Hardware {
    static HARDWARE: std::sync::OnceLock<rustblock_core::Hardware> = std::sync::OnceLock::new();
    *HARDWARE.get_or_init(rustblock_core::detect_hardware)
}

fn window_button(ui: &mut Ui, glyph: &str, tooltip: &str, danger: bool) -> bool {
    let (rect, response) = ui.allocate_exact_size(Vec2::new(34.0, 28.0), Sense::click());
    let hovered = response.hovered();
    if hovered {
        let fill = if danger {
            ERROR.gamma_multiply(0.9)
        } else {
            PANEL_3
        };
        ui.painter().rect_filled(rect, CornerRadius::same(3), fill);
    }
    let color = if danger && hovered {
        Color32::WHITE
    } else {
        MUTED
    };
    ui.painter().text(
        rect.center(),
        Align2::CENTER_CENTER,
        glyph,
        FontId::proportional(if danger { 16.0 } else { 13.0 }),
        color,
    );
    response.on_hover_text(tooltip).clicked()
}

fn help_dot(ui: &mut Ui, help: &str) {
    ui.label(RichText::new("?").size(9.0).color(MUTED))
        .on_hover_text(help);
}

fn backdrop_rect(area: egui::Rect) -> egui::Rect {
    let limit = area.width().min(area.height()).max(1.0);
    let size = (limit * 0.82).clamp(64.0, limit);
    egui::Rect::from_center_size(area.center(), Vec2::splat(size))
}

fn paint_window_border(ctx: &egui::Context) {
    let Some(screen) = ctx.input(|input| input.raw.screen_rect) else {
        return;
    };
    let painter = ctx.layer_painter(LayerId::new(Order::Foreground, Id::new("window-frame")));
    painter.rect_stroke(
        screen.shrink(0.5),
        CornerRadius::same(0),
        Stroke::new(1.0, BORDER),
        StrokeKind::Inside,
    );
}

fn open_folder(path: &Path) {
    let target = if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| path.to_path_buf())
    };
    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("explorer").arg(target).spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(target).spawn();
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let _ = std::process::Command::new("xdg-open").arg(target).spawn();
    }
    #[cfg(not(any(windows, unix)))]
    {
        let _ = target;
    }
}

fn same_file(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => path_key(a) == path_key(b),
    }
}

fn format_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn short_date(date: &str) -> String {
    if date.len() >= 10 {
        date[..10].to_string()
    } else if date.is_empty() {
        "unknown".to_string()
    } else {
        date.to_string()
    }
}

fn toggle_maximized(ctx: &egui::Context) {
    let maximized = ctx.input(|input| input.viewport().maximized.unwrap_or(false));
    ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(!maximized));
}

fn resize_edges(ui: &Ui) {
    use egui::viewport::ResizeDirection;

    let Some(screen) = ui.ctx().input(|input| input.raw.screen_rect) else {
        return;
    };
    let border = 5.0;
    let left = screen.left();
    let right = screen.right();
    let top = screen.top();
    let bottom = screen.bottom();
    let edges = [
        (
            ResizeDirection::North,
            egui::Rect::from_min_max(egui::pos2(left, top), egui::pos2(right, top + border)),
            egui::CursorIcon::ResizeVertical,
        ),
        (
            ResizeDirection::South,
            egui::Rect::from_min_max(egui::pos2(left, bottom - border), egui::pos2(right, bottom)),
            egui::CursorIcon::ResizeVertical,
        ),
        (
            ResizeDirection::West,
            egui::Rect::from_min_max(egui::pos2(left, top), egui::pos2(left + border, bottom)),
            egui::CursorIcon::ResizeHorizontal,
        ),
        (
            ResizeDirection::East,
            egui::Rect::from_min_max(egui::pos2(right - border, top), egui::pos2(right, bottom)),
            egui::CursorIcon::ResizeHorizontal,
        ),
        (
            ResizeDirection::NorthWest,
            egui::Rect::from_min_max(
                egui::pos2(left, top),
                egui::pos2(left + border, top + border),
            ),
            egui::CursorIcon::ResizeNwSe,
        ),
        (
            ResizeDirection::NorthEast,
            egui::Rect::from_min_max(
                egui::pos2(right - border, top),
                egui::pos2(right, top + border),
            ),
            egui::CursorIcon::ResizeNeSw,
        ),
        (
            ResizeDirection::SouthWest,
            egui::Rect::from_min_max(
                egui::pos2(left, bottom - border),
                egui::pos2(left + border, bottom),
            ),
            egui::CursorIcon::ResizeNeSw,
        ),
        (
            ResizeDirection::SouthEast,
            egui::Rect::from_min_max(
                egui::pos2(right - border, bottom - border),
                egui::pos2(right, bottom),
            ),
            egui::CursorIcon::ResizeNwSe,
        ),
    ];
    for (index, (direction, rect, cursor)) in edges.into_iter().enumerate() {
        let response = ui.interact(rect, Id::new(("rustblock-resize", index)), Sense::drag());
        if response.hovered() {
            ui.ctx().set_cursor_icon(cursor);
        }
        if response.drag_started() {
            ui.ctx()
                .send_viewport_cmd(egui::ViewportCommand::BeginResize(direction));
        }
    }
}

fn hf_result_row(ui: &mut Ui, model: &HfModel, selected: bool) -> bool {
    let (rect, response) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), 64.0), Sense::click());
    if selected {
        ui.painter()
            .rect_filled(rect, CornerRadius::same(8), PANEL_2);
        ui.painter().rect_filled(
            egui::Rect::from_min_size(
                rect.min + Vec2::new(1.0, 10.0),
                Vec2::new(2.0, rect.height() - 20.0),
            ),
            CornerRadius::same(2),
            ACCENT,
        );
    } else if response.hovered() {
        ui.painter()
            .rect_filled(rect, CornerRadius::same(8), PANEL_2.gamma_multiply(0.7));
    }
    ui.painter().line_segment(
        [
            egui::pos2(rect.left(), rect.bottom()),
            egui::pos2(rect.right(), rect.bottom()),
        ],
        Stroke::new(1.0, BORDER),
    );
    let clip = egui::Rect::from_min_max(
        egui::pos2(rect.left() + 16.0, rect.top()),
        egui::pos2(rect.left() + rect.width() * 0.64, rect.bottom()),
    );
    let painter = ui.painter().with_clip_rect(clip);
    painter.text(
        egui::pos2(rect.left() + 16.0, rect.top() + 21.0),
        Align2::LEFT_CENTER,
        &model.author,
        FontId::proportional(10.5),
        MUTED,
    );
    painter.text(
        egui::pos2(rect.left() + 16.0, rect.top() + 43.0),
        Align2::LEFT_CENTER,
        &model.name,
        FontId::proportional(13.0),
        TEXT,
    );
    ui.painter().text(
        egui::pos2(rect.right() - 16.0, rect.center().y),
        Align2::RIGHT_CENTER,
        format!(
            "{} files · {} downloads · {}",
            model.gguf_files,
            format_count(model.downloads),
            short_date(&model.last_modified)
        ),
        FontId::proportional(10.5),
        MUTED,
    );
    if let Some(license) = &model.license {
        ui.painter().text(
            egui::pos2(rect.right() - 16.0, rect.center().y + 16.0),
            Align2::RIGHT_CENTER,
            format!("License: {license}"),
            FontId::proportional(9.5),
            STEEL_LIGHT,
        );
    }
    if response.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    response.clicked()
}

fn hf_file_row(
    ui: &mut Ui,
    repo: &str,
    file: &HfFile,
    busy: bool,
    cached: bool,
    download_start: &mut Option<(String, HfFile)>,
) {
    let row_width = ui.available_width();
    let action_width = if file.projector { 92.0 } else { 70.0 };
    ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(
            Vec2::new((row_width - action_width - 8.0).max(120.0), 0.0),
            Layout::top_down(Align::LEFT),
            |ui| {
                ui.label(
                    RichText::new(shorten(&file.path, 38))
                        .size(10.0)
                        .monospace()
                        .color(STEEL_LIGHT),
                )
                .on_hover_text(&file.path);
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing.x = 6.0;
                    ui.label(
                        RichText::new(human_bytes(file.size))
                            .size(9.5)
                            .monospace()
                            .color(MUTED),
                    );
                    if !file.projector {
                        let (label, color, hint) = size_fit_badge(file.size, detected_hardware());
                        ui.label(RichText::new(label).size(9.0).strong().color(color))
                            .on_hover_text(hint);
                    }
                    if file.projector {
                        ui.label(RichText::new("· vision projector").size(9.5).color(INFO));
                    } else if cached {
                        ui.label(RichText::new("· cached locally").size(9.5).color(ACCENT));
                    } else if looks_quantized_name(&file.path) {
                        ui.label(RichText::new("· already quantized").size(9.5).color(WARN));
                    }
                });
            },
        );
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if ui
                .add_enabled(
                    !busy,
                    egui::Button::new(
                        RichText::new(if cached {
                            "Use cached"
                        } else if file.projector {
                            "Projector"
                        } else {
                            "Download"
                        })
                        .size(10.0)
                        .color(ACCENT),
                    )
                    .frame(false),
                )
                .clicked()
            {
                *download_start = Some((repo.to_string(), file.clone()));
            }
        });
    });
    ui.add_space(8.0);
    ui.separator();
}

fn target_grid(ui: &mut Ui, targets: &[Target], selected: &mut Target) {
    let gap = 4.0;
    let width = ((ui.available_width() - gap * 5.0) / 6.0).max(48.0);
    for chunk in targets.chunks(6) {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = gap;
            for target in chunk {
                let is_selected = *selected == *target;
                let button = egui::Button::new(
                    RichText::new(target.short())
                        .size(10.5)
                        .strong()
                        .color(if is_selected { INK } else { STEEL_LIGHT }),
                )
                .fill(if is_selected { ACCENT } else { PANEL_2 })
                .stroke(Stroke::new(1.0, if is_selected { ACCENT } else { BORDER }))
                .corner_radius(CornerRadius::same(6))
                .min_size(Vec2::new(width, 30.0));
                if ui.add(button).on_hover_text(target.description()).clicked() {
                    *selected = *target;
                }
            }
        });
    }
}

fn target_caption(target: Target) -> &'static str {
    match target {
        Target::Reblock => "Lossless PQ2_0 rewrite",
        Target::F32 => "32-bit float · largest",
        Target::F16 => "16-bit float · high quality",
        Target::Bf16 => "bfloat16 · high quality",
        Target::Q8_0 => "8-bit · near lossless",
        Target::Q5_0 => "5-bit · compact",
        Target::Q5_1 => "5-bit with minimum",
        Target::Q4_0 => "4-bit · small",
        Target::Q4_1 => "4-bit with minimum",
        Target::Q2K => "2-bit K-quant · smallest",
        Target::Q3K => "3-bit K-quant · tight fit",
        Target::Q4K => "4-bit K-quant · balanced",
        Target::Q5K => "5-bit K-quant · high quality",
        Target::Q6K => "6-bit K-quant · near F16",
        Target::Q3KM => "3-bit mixed · tight fit",
        Target::Q4KM => "4-bit mixed · balanced",
        Target::Q5KM => "5-bit mixed · high quality",
        Target::Tq1_0 | Target::Tq2_0 => "Ternary · experimental",
        Target::Mxfp4 => "4-bit block float",
        Target::Nvfp4 => "NVIDIA 4-bit float",
        Target::Q1_0 => "1-bit · experimental",
    }
}

#[cfg(test)]
mod backdrop_tests {
    use super::*;

    #[test]
    fn backdrop_is_centred_in_the_content_area() {
        let area = egui::Rect::from_min_size(egui::pos2(0.0, 56.0), Vec2::new(740.0, 520.0));
        let rect = backdrop_rect(area);
        assert_eq!(rect.center(), area.center());
        assert!(rect.width() == rect.height());
        assert!(rect.width() <= area.width() && rect.height() <= area.height());
    }

    fn textured_images(shapes: &[egui::epaint::ClippedShape]) -> Vec<egui::Rect> {
        let mut rects = Vec::new();
        for clipped in shapes {
            if let egui::Shape::Mesh(mesh) = &clipped.shape {
                if mesh.texture_id != egui::TextureId::Managed(0) {
                    rects.push(mesh.calc_bounds());
                }
            }
        }
        rects
    }

    /// Renders one real frame without a window: the title bar must stay flat
    /// and the only large mark must be the backdrop behind the content.
    #[test]
    fn backdrop_renders_once_behind_the_content() {
        let ctx = egui::Context::default();
        let mut app = RustBlockApp::with_context(&ctx);
        let screen = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), Vec2::new(1120.0, 680.0));
        let raw = egui::RawInput {
            screen_rect: Some(screen),
            ..Default::default()
        };
        let output = ctx.run_ui(raw, |ui| {
            app.top_chrome(ui);
            app.paint_backdrop(ui);
        });
        let images = textured_images(&output.shapes);
        assert!(!images.is_empty(), "the brand textures must render");
        let large: Vec<egui::Rect> = images
            .iter()
            .copied()
            .filter(|rect| rect.width() > 300.0)
            .collect();
        assert_eq!(large.len(), 1, "exactly one large mark expected: {large:?}");
        let backdrop = large[0];
        assert!((backdrop.width() - backdrop.height()).abs() < 1.0);
        assert!((backdrop.center().x - screen.center().x).abs() < 1.5);
        assert!(
            backdrop.top() >= 56.0,
            "the backdrop must sit below the title bar, got {backdrop:?}"
        );
    }

    #[test]
    fn backdrop_never_overflows_a_tiny_window() {
        let area = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), Vec2::new(120.0, 80.0));
        let rect = backdrop_rect(area);
        assert!(
            area.contains_rect(rect),
            "{rect:?} must fit inside {area:?}"
        );
    }
}

#[cfg(test)]
mod fit_badge_tests {
    use super::*;

    fn estimate(total: u64, ram: Option<u64>, vram: Option<u64>) -> FitEstimate {
        FitEstimate {
            path: "test.gguf".to_string(),
            architecture: "test".to_string(),
            layers: 1,
            context: 4096,
            kv_heads: 1,
            head_bytes: 0,
            weights_bytes: total,
            dense_kv_bytes: 0,
            sliding_kv_bytes: 0,
            mla_kv_bytes: 0,
            ssm_state_bytes: 0,
            kv_cache_bytes: 0,
            overhead_bytes: 0,
            total_bytes: total,
            hardware: rustblock_core::Hardware {
                ram_bytes: ram,
                vram_bytes: vram,
            },
            fits_ram: ram.map(|available| rustblock_core::hardware_fits(total, available)),
            fits_vram: vram.map(|available| rustblock_core::hardware_fits(total, available)),
            suggested_target: Target::Q4KM,
        }
    }

    #[test]
    fn download_badge_uses_weights_only_and_says_so() {
        let gib = 1024 * 1024 * 1024;
        let hw = |ram, vram| rustblock_core::Hardware {
            ram_bytes: ram,
            vram_bytes: vram,
        };
        let (label, _, hint) = size_fit_badge(8 * gib, hw(Some(32 * gib), Some(12 * gib)));
        assert_eq!(label, "Fits VRAM");
        assert!(hint.contains("not included"), "{hint}");
        assert_eq!(
            size_fit_badge(116 * gib, hw(Some(32 * gib), Some(12 * gib))).0,
            "No fit"
        );
        assert_eq!(size_fit_badge(8 * gib, hw(None, None)).0, "Fit unknown");
    }

    #[test]
    fn badge_distinguishes_fit_tight_and_unknown() {
        let gib = 1024 * 1024 * 1024;
        assert_eq!(
            fit_badge(&estimate(8 * gib, Some(32 * gib), Some(12 * gib))).0,
            "Fits VRAM"
        );
        assert_eq!(
            fit_badge(&estimate(11 * gib, Some(32 * gib), Some(12 * gib))).0,
            "Tight VRAM"
        );
        assert_eq!(
            fit_badge(&estimate(64 * gib, Some(8 * gib), Some(8 * gib))).0,
            "No fit"
        );
        assert_eq!(fit_badge(&estimate(8 * gib, None, None)).0, "Fit unknown");
    }
}

#[test]
fn log_level_filter_matches_only_selected_levels() {
    let entry = LogEntry {
        level: LogLevel::Warn,
        text: "something happened".into(),
    };
    assert!(log_entry_visible(&entry, "", &[true; 4]));
    assert!(!log_entry_visible(&entry, "", &[true, true, false, true]));
    assert!(!log_entry_visible(&entry, "absent", &[true; 4]));
    assert!(log_entry_visible(
        &entry,
        "some",
        &[false, false, true, false]
    ));
}

#[test]
fn every_surface_renders_without_panicking() {
    let ctx = egui::Context::default();
    let mut app = RustBlockApp::with_context(&ctx);
    app.discover_loaded = true;
    app.inspector = Some(InspectorData {
        path: "test.gguf".into(),
        version: 3,
        file_len: 1024,
        alignment: 32,
        data_start: 512,
        tensors: 2,
        pq2: 0,
        ptq1: 0,
        file_type: Some(1),
        metadata: vec![("general.name".to_string(), "test".to_string())],
        histogram: vec![("F16".to_string(), 2, 64), ("Q4_K_M".to_string(), 3, 48)],
        weights_bytes: 112,
        fit: None,
    });
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::pos2(0.0, 0.0),
            Vec2::new(1120.0, 680.0),
        )),
        ..Default::default()
    };
    for tab in [Tab::Convert, Tab::Discover, Tab::Inspector] {
        app.tab = tab;
        app.show_log = true;
        let _ = ctx.run_ui(raw.clone(), |ui| {
            app.top_chrome(ui);
            if app.show_log {
                app.log_panel(ui);
            }
            match app.tab {
                Tab::Convert => app.settings_rail(ui),
                Tab::Discover => app.hf_inspector_rail(ui),
                Tab::Inspector => {}
            }
            egui::CentralPanel::default()
                .frame(
                    egui::Frame::new()
                        .fill(BG)
                        .inner_margin(Margin::symmetric(22, 16)),
                )
                .show(ui, |ui| {
                    app.paint_backdrop(ui);
                    match app.tab {
                        Tab::Convert => app.convert_page(ui),
                        Tab::Discover => app.discover_page(ui),
                        Tab::Inspector => app.inspector_page(ui),
                    }
                });
        });
    }
}
