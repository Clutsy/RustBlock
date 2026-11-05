//! RustBlock application state and UI.
use crate::hf::{self, HfEvent, HfFile, HfModel, HfSort};
use crate::model::{
    analyze, human_bytes, human_duration, human_speed, looks_quantized_name, output_path, path_key,
    shorten, FileKind, Job, JobStatus, OutputMode, RunState, Settings, VerifyKind,
};
use crate::theme::{
    self, ACCENT, BG, BORDER, ERROR, INFO, INK, MUTED, PANEL, PANEL_2, PANEL_3, STEEL_LIGHT, TEXT,
    WARN,
};
use crate::worker::{self, Event, WorkItem, WorkKind};
use eframe::egui::{
    self, Align, Align2, Color32, CornerRadius, FontId, Id, LayerId, Layout, Margin, Order,
    ProgressBar, RichText, ScrollArea, Sense, Stroke, StrokeKind, TextEdit, Ui, Vec2,
};
use rustblock_core::{Phase, Target, VerifyMode};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LogLevel {
    Info,
    Ok,
    Warn,
    Error,
}
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Convert,
    Discover,
}
#[derive(Debug, Clone)]
enum HfState {
    Idle,
    Searching,
    Results(Vec<HfModel>),
    Error(String),
}
#[derive(Debug, Clone)]
struct DownloadUi {
    repo: String,
    file: String,
    done: u64,
    total: u64,
    speed: f64,
    quantized: bool,
    path: Option<PathBuf>,
    error: Option<String>,
}
pub struct RustBlockApp {
    tab: Tab,
    jobs: Vec<Job>,
    next_id: u64,
    settings: Settings,
    rx: Option<Receiver<Event>>,
    worker: Option<JoinHandle<()>>,
    cancel: Option<Arc<AtomicBool>>,
    batch: Option<Batch>,
    last_output_dir: Option<PathBuf>,
    centered: bool,
    logo: Option<egui::TextureHandle>,
    ctx: egui::Context,
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
}
impl RustBlockApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        theme::install(&cc.egui_ctx);
        let mut app = Self {
            tab: Tab::Convert,
            jobs: Vec::new(),
            next_id: 1,
            settings: Settings::default(),
            rx: None,
            worker: None,
            cancel: None,
            batch: None,
            last_output_dir: None,
            centered: false,
            logo: theme::load_logo(&cc.egui_ctx),
            ctx: cc.egui_ctx.clone(),
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
        };
        app.log(
            LogLevel::Info,
            "Ready. Add a GGUF file or search Hugging Face for a GGUF repository.".to_string(),
        );
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
    fn is_running(&self) -> bool {
        self.rx.is_some()
    }
    fn hf_busy(&self) -> bool {
        self.hf_rx.is_some()
    }
    fn log(&mut self, _level: LogLevel, _text: String) {}
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
            let analysis = analyze(&path);
            if let Some(error) = &analysis.error {
                self.log(
                    LogLevel::Error,
                    format!("Cannot read {}: {error}", path.display()),
                );
            } else {
                self.log(LogLevel::Info, format!("Added: {}", path.display()));
                if (analysis.is_quantized() && analysis.kind != FileKind::Pq2)
                    || path
                        .file_name()
                        .is_some_and(|name| looks_quantized_name(&name.to_string_lossy()))
                {
                    self.log(
                        LogLevel::Warn,
                        format!(
                            "{} is already quantized. Converting it again cannot recover the original precision; a virgin F16/BF16 model is the better source.",
                            path.file_name()
                                .map(|n| n.to_string_lossy().into_owned())
                                .unwrap_or_else(|| path.display().to_string())
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
    // ----- Hugging Face -------------------------------------------------
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
        if self.hf_busy() {
            return;
        }
        if let Err(reason) = hf::file_support(&file.path) {
            self.log(
                LogLevel::Warn,
                format!("Not downloading {}: {reason}.", file.path),
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
            quantized: looks_quantized_name(&file.path),
            path: None,
            error: None,
        });
        self.hf_rx = Some(rx);
        self.hf_cancel = Arc::new(AtomicBool::new(false));
        let cancel = self.hf_cancel.clone();
        self.hf_worker = Some(std::thread::spawn(move || {
            let result = hf::download(&repo, &revision, &file.path, &dest, false, &cancel, &tx);
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
                HfEvent::Downloaded { repo, file, path } => {
                    if let Some(d) = &mut self.download {
                        d.path = Some(path.clone());
                        d.done = d.total;
                    }
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

    fn selected_jobs(&self) -> usize {
        let target = self.settings.target;
        self.jobs
            .iter()
            .filter(|j| j.analysis.can_target(target))
            .count()
    }

    // ----- UI -----------------------------------------------------------
    fn top_chrome(&mut self, ui: &mut Ui) {
        let logo = self.logo.clone();
        egui::Panel::top("chrome")
            .exact_size(60.0)
            .frame(
                egui::Frame::new()
                    .fill(PANEL)
                    .inner_margin(Margin::symmetric(20, 10))
                    .stroke(Stroke::new(1.0, BORDER)),
            )
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    theme::paint_logo(ui, 28.0, logo.as_ref());
                    ui.add_space(8.0);
                    ui.label(RichText::new("RustBlock").size(15.0).strong().color(TEXT));
                    ui.add_space(26.0);
                    self.nav_tab(ui, Tab::Convert, "Convert");
                    ui.add_space(4.0);
                    self.nav_tab(ui, Tab::Discover, "Discover");
                });
            });
    }

    fn nav_tab(&mut self, ui: &mut Ui, tab: Tab, label: &str) {
        let active = self.tab == tab;
        let response = ui.add_sized(
            [96.0, 32.0],
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
            .corner_radius(CornerRadius::same(4)),
        );
        if response.clicked() {
            self.tab = tab;
        }
    }

    fn convert_page(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.label(RichText::new("Convert").size(28.0).strong().color(TEXT));
                let ready = self.selected_jobs();
                ui.label(
                    RichText::new(format!(
                        "{} · {} ready for {}",
                        if self.jobs.len() == 1 {
                            "1 file".to_string()
                        } else {
                            format!("{} files", self.jobs.len())
                        },
                        ready,
                        self.settings.target.short()
                    ))
                    .size(11.5)
                    .color(MUTED),
                );
            });
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let add =
                    egui::Button::new(RichText::new("Add files").size(12.0).strong().color(INK))
                        .fill(ACCENT)
                        .stroke(Stroke::new(1.0, ACCENT))
                        .corner_radius(CornerRadius::same(4))
                        .min_size(Vec2::new(104.0, 34.0));
                if ui.add(add).clicked() {
                    self.browse();
                }
            });
        });
        self.download_banner(ui);
        ui.add_space(16.0);
        ScrollArea::vertical()
            .id_salt("queue_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                if self.jobs.is_empty() {
                    self.empty_queue(ui);
                } else {
                    self.queue_panel(ui);
                }
            });
    }

    fn empty_queue(&mut self, ui: &mut Ui) {
        egui::Frame::new()
            .fill(PANEL)
            .stroke(Stroke::new(1.0, BORDER))
            .corner_radius(CornerRadius::same(6))
            .inner_margin(Margin::symmetric(28, 46))
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                ui.vertical_centered(|ui| {
                    ui.label(
                        RichText::new("Drop GGUF files here")
                            .size(20.0)
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
                        ui.label(RichText::new("Output format").size(10.5).color(MUTED));
                        ui.add_space(1.0);
                        ui.label(
                            RichText::new(self.settings.target.short())
                                .size(26.0)
                                .strong()
                                .color(TEXT),
                        );
                        ui.label(
                            RichText::new(target_caption(self.settings.target))
                                .size(11.5)
                                .color(MUTED),
                        );
                        ui.add_space(8.0);
                        self.format_panel(ui);
                        ui.add_space(8.0);
                        ui.separator();
                        ui.add_space(8.0);
                        let max_height = (ui.available_height() - 64.0).max(96.0);
                        ScrollArea::vertical()
                            .id_salt("settings_scroll")
                            .max_height(max_height)
                            .max_width(ui.available_width())
                            .auto_shrink([false, true])
                            .show(ui, |ui| {
                                self.options_panel(ui);
                            });
                        ui.add_space(8.0);
                        ui.separator();
                        ui.add_space(8.0);
                        self.start_controls(ui);
                    });
            });
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

    fn queue_panel(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label(RichText::new("Queue").size(13.0).strong().color(TEXT));
            ui.label(
                RichText::new(self.jobs.len().to_string())
                    .size(10.5)
                    .monospace()
                    .color(MUTED),
            );
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui
                    .add_enabled(
                        !self.is_running(),
                        egui::Button::new(RichText::new("Clear all").size(10.5).color(MUTED))
                            .frame(false),
                    )
                    .clicked()
                {
                    self.jobs.clear();
                }
            });
        });
        ui.add_space(9.0);
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
                .corner_radius(CornerRadius::same(4))
                .inner_margin(Margin::symmetric(12, 10))
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
                                    .corner_radius(CornerRadius::same(4)),
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
        let running = self.is_running();
        let mut remove = None;
        let mut open = None;
        for job in &self.jobs {
            job_card(ui, job, running, &mut remove, &mut open);
            ui.add_space(8.0);
        }
        egui::Frame::new()
            .fill(Color32::TRANSPARENT)
            .stroke(Stroke::new(1.0, BORDER))
            .corner_radius(CornerRadius::same(4))
            .inner_margin(Margin::symmetric(12, 10))
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Drop more GGUF files here")
                            .size(10.5)
                            .color(MUTED),
                    );
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new("Browse").size(10.0).color(STEEL_LIGHT),
                                )
                                .frame(false),
                            )
                            .clicked()
                        {
                            self.browse();
                        }
                    });
                });
            });
        if let Some(id) = remove {
            self.jobs.retain(|j| j.id != id);
        }
        if let Some(path) = open {
            open_folder(&path);
        }
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
            available.push(Target::Q4K);
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

    fn options_panel(&mut self, ui: &mut Ui) {
        let running = self.is_running();
        ui.add_enabled_ui(!running, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Destination").size(10.5).color(MUTED));
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

            ui.add_space(10.0);
            ui.label(RichText::new("Verification").size(10.5).color(MUTED));
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

    fn start_controls(&mut self, ui: &mut Ui) {
        let count = self.selected_jobs();
        let running = self.is_running();
        let enabled = !running && count > 0;
        if self.jobs.is_empty() {
            ui.label(
                RichText::new("Add a GGUF file to begin.")
                    .size(10.5)
                    .color(MUTED),
            );
        } else if count == 0 {
            ui.label(
                RichText::new(format!("No file can use {}.", self.settings.target.short()))
                    .size(10.5)
                    .color(WARN),
            );
        } else if count < self.jobs.len() {
            ui.label(
                RichText::new(format!("{count} of {} files ready.", self.jobs.len()))
                    .size(10.5)
                    .color(MUTED),
            );
        }
        ui.add_space(6.0);
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
            .min_size(Vec2::new(ui.available_width(), 42.0));
        if ui.add_enabled(enabled, button).clicked() {
            self.start(WorkKind::Convert);
        }
    }

    fn download_banner(&mut self, ui: &mut Ui) {
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
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(shorten(name, 62))
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
                ui.label(RichText::new("Discover").size(28.0).strong().color(TEXT));
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
                .corner_radius(CornerRadius::same(4))
                .inner_margin(Margin::symmetric(12, 0))
                .show(ui, |ui| {
                    let response = ui.add_sized(
                        [field_width - 24.0, 38.0],
                        TextEdit::singleline(&mut self.hf_query)
                            .hint_text("Search a model or paste a Hugging Face link")
                            .font(FontId::proportional(13.0))
                            .frame(egui::Frame::NONE)
                            .margin(Vec2::new(0.0, 9.0)),
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
            .corner_radius(CornerRadius::same(4))
            .min_size(Vec2::new(92.0, 38.0));
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
                    .fill(PANEL)
                    .stroke(Stroke::new(1.0, BORDER))
                    .corner_radius(CornerRadius::same(6))
                    .inner_margin(Margin::symmetric(28, 36))
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
            .filter(|file| file.supported)
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
            hf_file_row(ui, &repo, file, self.hf_busy(), &mut download_start);
        }
        if let Some((repo, file)) = download_start {
            self.start_download(repo, file);
        }
    }

    fn hf_destination(&mut self, ui: &mut Ui) {
        ui.label(RichText::new("Save downloads to").size(9.5).color(MUTED));
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(shorten(
                    &self.settings.download_dir.display().to_string(),
                    30,
                ))
                .size(9.5)
                .monospace()
                .color(STEEL_LIGHT),
            )
            .on_hover_text(self.settings.download_dir.display().to_string());
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
}

impl eframe::App for RustBlockApp {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if !self.centered {
            if let Some(cmd) = egui::ViewportCommand::center_on_screen(ctx) {
                ctx.send_viewport_cmd(cmd);
                self.centered = true;
            }
        }
        if self.is_running() && ctx.input(|i| i.viewport().close_requested()) {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
        }
        self.handle_dropped(ctx);
        self.poll_worker(ctx);
        self.poll_hf();
        if self.rx.is_some() || self.hf_rx.is_some() {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }

    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        self.top_chrome(ui);
        match self.tab {
            Tab::Convert => self.settings_rail(ui),
            Tab::Discover => self.hf_inspector_rail(ui),
        }
        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(BG)
                    .inner_margin(Margin::symmetric(20, 14)),
            )
            .show(ui, |ui| match self.tab {
                Tab::Convert => self.convert_page(ui),
                Tab::Discover => self.discover_page(ui),
            });
        self.drag_overlay(ui.ctx());
    }
}

fn hf_result_row(ui: &mut Ui, model: &HfModel, selected: bool) -> bool {
    let (rect, response) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), 60.0), Sense::click());
    if selected {
        ui.painter()
            .rect_filled(rect, CornerRadius::same(3), PANEL_2);
        ui.painter().rect_filled(
            egui::Rect::from_min_size(rect.min, Vec2::new(3.0, rect.height())),
            CornerRadius::same(0),
            ACCENT,
        );
    } else if response.hovered() {
        ui.painter()
            .rect_filled(rect, CornerRadius::same(3), PANEL_2.gamma_multiply(0.7));
    }
    ui.painter().line_segment(
        [
            egui::pos2(rect.left(), rect.bottom()),
            egui::pos2(rect.right(), rect.bottom()),
        ],
        Stroke::new(1.0, BORDER),
    );
    let clip = egui::Rect::from_min_max(
        egui::pos2(rect.left() + 12.0, rect.top()),
        egui::pos2(rect.left() + rect.width() * 0.64, rect.bottom()),
    );
    let painter = ui.painter().with_clip_rect(clip);
    painter.text(
        egui::pos2(rect.left() + 12.0, rect.top() + 15.0),
        Align2::LEFT_CENTER,
        &model.author,
        FontId::proportional(10.5),
        MUTED,
    );
    painter.text(
        egui::pos2(rect.left() + 12.0, rect.top() + 38.0),
        Align2::LEFT_CENTER,
        &model.name,
        FontId::proportional(13.0),
        TEXT,
    );
    ui.painter().text(
        egui::pos2(rect.right() - 12.0, rect.center().y),
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
    if response.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    response.clicked()
}

fn target_grid(ui: &mut Ui, targets: &[Target], selected: &mut Target) {
    let gap = 4.0;
    let width = ((ui.available_width() - gap * 3.0) / 4.0).max(48.0);
    for chunk in targets.chunks(4) {
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
                .corner_radius(CornerRadius::same(4))
                .min_size(Vec2::new(width, 28.0));
                if ui.add(button).on_hover_text(target.description()).clicked() {
                    *selected = *target;
                }
            }
        });
    }
}

fn job_card(
    ui: &mut Ui,
    job: &Job,
    running: bool,
    remove: &mut Option<u64>,
    open: &mut Option<PathBuf>,
) {
    let color = status_color(job);
    egui::Frame::new()
        .fill(PANEL)
        .stroke(Stroke::new(1.0, BORDER))
        .corner_radius(CornerRadius::same(4))
        .inner_margin(Margin::symmetric(12, 10))
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.label(
                        RichText::new(job.file_name())
                            .size(13.0)
                            .strong()
                            .color(TEXT),
                    );
                    ui.label(
                        RichText::new(format!(
                            "{} · {}",
                            human_bytes(job.analysis.file_len),
                            job.analysis.kind_label()
                        ))
                        .size(10.5)
                        .color(MUTED),
                    )
                    .on_hover_text(format!(
                        "GGUF v{} · alignment {} B · {} tensors · {}",
                        job.analysis.version,
                        job.analysis.alignment,
                        job.analysis.tensors,
                        job.analysis.type_summary()
                    ));
                });
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
                    ui.label(RichText::new(status_label(job)).size(9.5).color(color));
                });
            });
            if job.target != Target::Reblock
                && job.analysis.error.is_none()
                && (job.analysis.is_quantized() || looks_quantized_name(&job.file_name()))
            {
                ui.label(
                    RichText::new("Already quantized · use an F16/BF16 model for best quality.")
                        .size(9.5)
                        .color(WARN),
                );
            }
            if job.analysis.error.is_none() && !job.analysis.supported {
                ui.label(
                    RichText::new("Contains an unsupported tensor type.")
                        .size(9.5)
                        .color(WARN),
                );
            }
            match &job.status {
                JobStatus::Running(state) => {
                    ui.add_space(5.0);
                    let phase = match state.phase {
                        Phase::Converting => "Converting",
                        Phase::Verifying => "Verifying",
                    };
                    ui.add(
                        ProgressBar::new(state.fraction())
                            .fill(ACCENT)
                            .animate(true)
                            .text(format!("{} · {:.0}%", phase, state.fraction() * 100.0)),
                    );
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new(format!("{}/s", human_bytes(state.speed as u64)))
                                .size(9.0)
                                .monospace()
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
                    ui.add_space(2.0);
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
                                    RichText::new("Open folder").size(10.0).color(ACCENT),
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
                    ui.label(RichText::new(error).size(10.0).color(ERROR));
                }
                JobStatus::Ready => {}
            }
        });
}

fn hf_file_row(
    ui: &mut Ui,
    repo: &str,
    file: &HfFile,
    busy: bool,
    download_start: &mut Option<(String, HfFile)>,
) {
    let row_width = ui.available_width();
    let action_width = 70.0;
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
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(human_bytes(file.size))
                            .size(9.5)
                            .monospace()
                            .color(MUTED),
                    );
                    if looks_quantized_name(&file.path) {
                        ui.label(RichText::new("· already quantized").size(9.5).color(WARN));
                    }
                });
            },
        );
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if ui
                .add_enabled(
                    !busy,
                    egui::Button::new(RichText::new("Download").size(10.0).color(ACCENT))
                        .frame(false),
                )
                .clicked()
            {
                *download_start = Some((repo.to_string(), file.clone()));
            }
        });
    });
    ui.add_space(6.0);
    ui.separator();
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
        Target::Tq1_0 | Target::Tq2_0 => "Ternary · experimental",
        Target::Mxfp4 => "4-bit block float",
        Target::Nvfp4 => "NVIDIA 4-bit float",
        Target::Q1_0 => "1-bit · experimental",
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

fn same_file(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => path_key(a) == path_key(b),
    }
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
