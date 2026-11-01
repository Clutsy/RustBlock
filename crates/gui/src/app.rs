//! RustBlock application state and UI.
use crate::hf::{self, HfDetails, HfEvent, HfFile, HfModel, HfSort};
use crate::model::{
    analyze, human_bytes, human_duration, human_speed, log_clock, output_path, path_key, shorten,
    FileKind, InstallResult, InstallTarget, Job, JobStatus, OutputMode, RunState, Settings,
    VerifyKind,
};
use crate::targets::{self, Detected};
use crate::theme::{
    self, ACCENT, BG, BORDER, ERROR, INFO, INK, MUTED, PANEL, PANEL_2, PANEL_3, STEEL, STEEL_DARK,
    STEEL_LIGHT, TEXT, WARN,
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
impl LogLevel {
    fn color(self) -> Color32 {
        match self {
            Self::Info => MUTED,
            Self::Ok => ACCENT,
            Self::Warn => WARN,
            Self::Error => ERROR,
        }
    }
}
struct LogEntry {
    at: Duration,
    level: LogLevel,
    text: String,
}
struct Batch {
    kind: WorkKind,
    total: usize,
    completed: usize,
    failed: usize,
    started: Instant,
    active: Option<u64>,
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
    path: Option<PathBuf>,
    error: Option<String>,
}
pub struct RustBlockApp {
    tab: Tab,
    jobs: Vec<Job>,
    next_id: u64,
    settings: Settings,
    entries: Vec<LogEntry>,
    started: Instant,
    rx: Option<Receiver<Event>>,
    worker: Option<JoinHandle<()>>,
    cancel: Option<Arc<AtomicBool>>,
    batch: Option<Batch>,
    show_about: bool,
    last_output_dir: Option<PathBuf>,
    centered: bool,
    close_warned: bool,
    logo: Option<egui::TextureHandle>,
    ctx: egui::Context,
    detected: Vec<Detected>,
    hf_query: String,
    hf_sort: HfSort,
    hf_state: HfState,
    hf_rx: Option<Receiver<HfEvent>>,
    hf_worker: Option<JoinHandle<()>>,
    hf_cancel: Arc<AtomicBool>,
    selected_repo: Option<String>,
    selected_sha: String,
    hf_files: Vec<HfFile>,
    hf_details: Option<HfDetails>,
    hf_details_loading: bool,
    pending_details: Option<(String, String)>,
    link_file: Option<String>,
    hf_scroll_top: bool,
    hf_details_error: Option<String>,
    hf_file_filter: String,
    show_log: bool,
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
            entries: Vec::new(),
            started: Instant::now(),
            rx: None,
            worker: None,
            cancel: None,
            batch: None,
            show_about: false,
            last_output_dir: None,
            centered: false,
            close_warned: false,
            logo: theme::load_logo(&cc.egui_ctx),
            ctx: cc.egui_ctx.clone(),
            detected: targets::detect(),
            hf_query: String::new(),
            hf_sort: HfSort::Downloads,
            hf_state: HfState::Idle,
            hf_rx: None,
            hf_worker: None,
            hf_cancel: Arc::new(AtomicBool::new(false)),
            selected_repo: None,
            selected_sha: "main".to_string(),
            hf_files: Vec::new(),
            hf_details: None,
            hf_details_loading: false,
            pending_details: None,
            link_file: None,
            hf_scroll_top: false,
            hf_details_error: None,
            hf_file_filter: String::new(),
            show_log: false,
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
    fn log(&mut self, level: LogLevel, text: String) {
        let at = self.started.elapsed();
        self.entries.push(LogEntry { at, level, text });
        if self.entries.len() > 1500 {
            self.entries.remove(0);
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
            Event::Done {
                id,
                report,
                target,
                install,
            } => {
                let name = self.job_name(id);
                let old = report.old_file_size;
                let new = report.new_file_size;
                let elapsed = report.elapsed;
                if let Some(job) = self.jobs.iter_mut().find(|j| j.id == id) {
                    job.status = JobStatus::Done {
                        report,
                        target,
                        install: install.clone(),
                    };
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
                match install {
                    Some(InstallResult::Installed(path)) => {
                        self.log(LogLevel::Ok, format!("Installed: {path}"));
                    }
                    Some(InstallResult::Failed(error)) => {
                        self.log(LogLevel::Warn, format!("Install failed: {error}"));
                    }
                    None => {}
                }
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
        let verify_only = self.settings.verify_only;
        let output_mode = self.settings.output_mode.clone();
        let install = if kind == WorkKind::Convert {
            self.settings.install.clone()
        } else {
            InstallTarget::None
        };
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
                install: install.clone(),
            });
        }
        for (level, text) in notes {
            self.log(level, text);
        }
        if items.is_empty() {
            self.log(
                LogLevel::Info,
                if verify_only {
                    "Nothing to verify.".to_string()
                } else {
                    format!(
                        "Nothing to convert to {}. Add a compatible file or choose another target.",
                        target.short()
                    )
                },
            );
            return;
        }
        let total = items.len();
        self.batch = Some(Batch {
            kind,
            total,
            completed: 0,
            failed: 0,
            started: Instant::now(),
            active: items.first().map(|i| i.id),
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
    fn export_log(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .set_file_name("rustblock-log.txt")
            .add_filter("Text", &["txt", "log"])
            .save_file()
        {
            match std::fs::write(&path, self.render_log()) {
                Ok(()) => self.log(LogLevel::Ok, format!("Log saved to {}", path.display())),
                Err(error) => self.log(LogLevel::Error, format!("Could not save log: {error}")),
            }
        }
    }
    fn render_log(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "RustBlock {}\nOS: {} {}\n\n",
            env!("CARGO_PKG_VERSION"),
            std::env::consts::OS,
            std::env::consts::ARCH
        ));
        for entry in &self.entries {
            out.push_str(&format!("[{}] {}\n", log_clock(entry.at), entry.text));
        }
        out
    }
    // ----- Hugging Face -------------------------------------------------
    fn start_search(&mut self) {
        if self.hf_busy() {
            return;
        }
        let query = self.hf_query.trim().to_string();
        if let Some(link) = hf::parse_link(&query) {
            self.open_link(link);
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
        self.hf_details = None;
        self.hf_details_loading = false;
        self.hf_details_error = None;
        self.pending_details = None;
        self.link_file = None;
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
        self.hf_details = None;
        self.hf_details_loading = false;
        self.hf_details_error = None;
        self.pending_details = None;
        self.link_file = link.file;
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
        self.hf_details = None;
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
            match hf::load_details(&repo, &revision) {
                Ok((details, files)) => {
                    let _ = tx.send(HfEvent::DetailsLoaded {
                        repo,
                        details,
                        files,
                    });
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
                HfEvent::DetailsLoaded {
                    repo,
                    details,
                    files,
                } => {
                    if self.selected_repo.as_deref() == Some(repo.as_str()) {
                        self.hf_details = Some(details);
                        self.hf_files = files;
                        self.hf_details_loading = false;
                        self.hf_scroll_top = true;
                        if let Some(file) = self.link_file.take() {
                            self.hf_file_filter =
                                file.rsplit('/').next().unwrap_or(file.as_str()).to_string();
                        }
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
                        d.speed = speed;
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
            .filter(|j| {
                if self.settings.verify_only {
                    j.analysis.error.is_none()
                } else {
                    j.analysis.can_target(target)
                }
            })
            .count()
    }

    // ----- UI -----------------------------------------------------------
    fn top_chrome(&mut self, ui: &mut Ui) {
        let logo = self.logo.clone();
        let engine = self
            .detected
            .iter()
            .find(|d| !matches!(d.target, InstallTarget::None))
            .map(|d| d.label.clone());
        egui::Panel::top("chrome")
            .exact_size(54.0)
            .frame(
                egui::Frame::new()
                    .fill(PANEL)
                    .inner_margin(Margin::symmetric(18, 7))
                    .stroke(Stroke::new(1.0, BORDER)),
            )
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    theme::paint_logo(ui, 30.0, logo.as_ref());
                    ui.add_space(8.0);
                    ui.vertical(|ui| {
                        ui.label(RichText::new("RustBlock").size(14.0).strong().color(TEXT));
                        ui.label(RichText::new("GGUF workshop").size(10.5).color(MUTED));
                    });
                    ui.add_space(28.0);
                    ui.horizontal(|ui| {
                        self.nav_tab(ui, Tab::Convert, "Convert");
                        ui.add_space(4.0);
                        self.nav_tab(ui, Tab::Discover, "Discover");
                    });
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new(if self.show_log { "Hide log" } else { "Log" })
                                        .size(11.0)
                                        .color(STEEL_LIGHT),
                                )
                                .frame(false),
                            )
                            .clicked()
                        {
                            self.show_log = !self.show_log;
                        }
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new("About").size(11.0).color(STEEL_LIGHT),
                                )
                                .frame(false),
                            )
                            .clicked()
                        {
                            self.show_about = true;
                        }
                        ui.add_space(6.0);
                        ui.label(
                            RichText::new(engine.as_deref().unwrap_or("No local app"))
                                .size(10.5)
                                .color(MUTED),
                        );
                        let (dot, _) = ui.allocate_exact_size(Vec2::splat(6.0), Sense::hover());
                        ui.painter().rect_filled(
                            dot,
                            CornerRadius::same(0),
                            if engine.is_some() { ACCENT } else { BORDER },
                        );
                    });
                });
            });
    }

    fn nav_tab(&mut self, ui: &mut Ui, tab: Tab, label: &str) {
        let active = self.tab == tab;
        let response = ui.add_sized(
            [72.0, 30.0],
            egui::Button::new(
                RichText::new(label)
                    .size(12.5)
                    .color(if active { TEXT } else { MUTED })
                    .strong(),
            )
            .frame(false),
        );
        if active {
            let y = response.rect.bottom() - 1.0;
            ui.painter().rect_filled(
                egui::Rect::from_min_max(
                    egui::pos2(response.rect.left(), y),
                    egui::pos2(response.rect.right(), y + 2.0),
                ),
                CornerRadius::same(0),
                ACCENT,
            );
        }
        if response.clicked() {
            self.tab = tab;
        }
    }

    fn status_bar(&mut self, ui: &mut Ui) {
        egui::Panel::bottom("status_bar")
            .exact_size(27.0)
            .frame(
                egui::Frame::new()
                    .fill(PANEL)
                    .inner_margin(Margin::symmetric(14, 4))
                    .stroke(Stroke::new(1.0, BORDER)),
            )
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    let (dot, _) = ui.allocate_exact_size(Vec2::splat(6.0), Sense::hover());
                    ui.painter().rect_filled(dot, CornerRadius::same(0), ACCENT);
                    ui.label(
                        RichText::new(if self.is_running() {
                            "Working"
                        } else if self.hf_busy() {
                            "Fetching"
                        } else {
                            "Ready"
                        })
                        .size(10.5)
                        .color(STEEL_LIGHT),
                    );
                    ui.label(RichText::new("|").size(10.0).color(BORDER));
                    let done = self.jobs.iter().filter(|j| j.is_finished()).count();
                    ui.label(
                        RichText::new(format!(
                            "{} · {} converted",
                            format_files(self.jobs.len()),
                            done
                        ))
                        .size(10.0)
                        .color(MUTED),
                    );
                    ui.label(RichText::new("|").size(10.0).color(BORDER));
                    ui.label(RichText::new("Apache-2.0 source").size(10.0).color(MUTED));
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        ui.label(
                            RichText::new("Ctrl+Enter Start")
                                .size(9.5)
                                .monospace()
                                .color(MUTED),
                        );
                        ui.label(
                            RichText::new("Ctrl+2 Discover")
                                .size(9.5)
                                .monospace()
                                .color(MUTED),
                        );
                        ui.label(
                            RichText::new("Ctrl+1 Convert")
                                .size(9.5)
                                .monospace()
                                .color(MUTED),
                        );
                    });
                });
            });
    }

    fn activity_panel(&mut self, ui: &mut Ui) {
        egui::Panel::bottom("activity")
            .exact_size(170.0)
            .resizable(true)
            .frame(
                egui::Frame::new()
                    .fill(PANEL_2)
                    .inner_margin(Margin::symmetric(16, 8))
                    .stroke(Stroke::new(1.0, BORDER)),
            )
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Activity").size(11.5).strong().color(TEXT));
                    ui.label(
                        RichText::new(format!("{} entries", self.entries.len()))
                            .size(10.0)
                            .color(MUTED),
                    );
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui
                            .add(egui::Button::new(RichText::new("Hide").size(10.0)).frame(false))
                            .clicked()
                        {
                            self.show_log = false;
                        }
                        if ui
                            .add(egui::Button::new(RichText::new("Clear").size(10.0)).frame(false))
                            .clicked()
                        {
                            self.entries.clear();
                        }
                        if ui
                            .add(egui::Button::new(RichText::new("Export").size(10.0)).frame(false))
                            .clicked()
                        {
                            self.export_log();
                        }
                    });
                });
                ui.add_space(2.0);
                ScrollArea::vertical()
                    .id_salt("activity_scroll")
                    .auto_shrink([false, false])
                    .stick_to_bottom(true)
                    .max_height(128.0)
                    .show(ui, |ui| {
                        if self.entries.is_empty() {
                            ui.label(RichText::new("No activity yet.").size(10.0).color(MUTED));
                        }
                        for entry in &self.entries {
                            ui.horizontal_wrapped(|ui| {
                                ui.label(
                                    RichText::new(log_clock(entry.at))
                                        .monospace()
                                        .size(9.5)
                                        .color(STEEL_DARK),
                                );
                                ui.label(RichText::new(&entry.text).size(10.0).color(
                                    if matches!(entry.level, LogLevel::Info) {
                                        STEEL_LIGHT
                                    } else {
                                        entry.level.color()
                                    },
                                ));
                            });
                        }
                    });
            });
    }

    fn convert_page(&mut self, ui: &mut Ui) {
        let pipeline = format!("source GGUF  →  target {}", self.settings.target.short());
        self.page_header(
            ui,
            "Convert",
            "Turn a GGUF into another layout. The source file is never modified.",
            Some(&pipeline),
        );
        ui.add_space(12.0);
        how_it_works(ui);
        ui.add_space(14.0);
        ScrollArea::vertical()
            .id_salt("queue_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                self.drop_strip(ui);
                ui.add_space(20.0);
                self.queue_panel(ui);
            });
    }

    fn settings_rail(&mut self, ui: &mut Ui) {
        egui::Panel::right("settings_rail")
            .exact_size(340.0)
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
                        ui.label(RichText::new("Output").size(10.5).color(MUTED));
                        ui.label(
                            RichText::new(self.settings.target.short())
                                .size(17.0)
                                .strong()
                                .color(TEXT),
                        );
                        ui.add(
                            egui::Label::new(
                                RichText::new(self.settings.target.description())
                                    .size(11.0)
                                    .color(MUTED),
                            )
                            .wrap(),
                        );
                        ui.add_space(10.0);
                        let max_height = (ui.available_height() - 112.0).max(120.0);
                        ScrollArea::vertical()
                            .id_salt("settings_scroll")
                            .max_height(max_height)
                            .max_width(ui.available_width())
                            .scroll_bar_visibility(
                                egui::scroll_area::ScrollBarVisibility::AlwaysVisible,
                            )
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                self.settings_panel(ui);
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
            .exact_size(340.0)
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
                        ui.label(RichText::new("Inspector").size(10.5).color(MUTED));
                        ui.add_space(10.0);
                        let mut scroll = ScrollArea::vertical()
                            .id_salt("hf_details_scroll")
                            .max_height(ui.available_height())
                            .max_width(ui.available_width())
                            .scroll_bar_visibility(
                                egui::scroll_area::ScrollBarVisibility::AlwaysVisible,
                            )
                            .auto_shrink([false, false]);
                        if scroll_top {
                            scroll = scroll.vertical_scroll_offset(0.0);
                        }
                        scroll.show(ui, |ui| {
                            self.hf_details_panel(ui);
                        });
                    });
            });
    }

    fn page_header(&self, ui: &mut Ui, title: &str, subtitle: &str, detail: Option<&str>) {
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.label(RichText::new(title).size(26.0).strong().color(TEXT));
                ui.add(egui::Label::new(RichText::new(subtitle).size(12.0).color(MUTED)).wrap());
            });
            if let Some(detail) = detail {
                ui.with_layout(Layout::right_to_left(Align::Min), |ui| {
                    ui.label(RichText::new(detail).size(10.0).monospace().color(MUTED));
                });
            }
        });
    }

    fn drop_strip(&mut self, ui: &mut Ui) {
        let hovering = !ui.ctx().input(|i| i.raw.hovered_files.is_empty());
        let stroke = Stroke::new(
            if hovering { 1.5 } else { 1.0 },
            if hovering { ACCENT } else { BORDER },
        );
        egui::Frame::new()
            .fill(if hovering {
                ACCENT.gamma_multiply(0.06)
            } else {
                Color32::TRANSPARENT
            })
            .stroke(stroke)
            .corner_radius(CornerRadius::same(0))
            .inner_margin(Margin::symmetric(16, 13))
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                ui.horizontal(|ui| {
                    let (mark, _) = ui.allocate_exact_size(Vec2::splat(32.0), Sense::hover());
                    ui.painter().rect_stroke(
                        mark,
                        CornerRadius::same(0),
                        Stroke::new(1.0, BORDER),
                        StrokeKind::Inside,
                    );
                    let arrow = if hovering { ACCENT } else { STEEL };
                    let c = mark.center();
                    let stroke = Stroke::new(1.6, arrow);
                    ui.painter()
                        .line_segment([c + Vec2::new(0.0, -6.0), c + Vec2::new(0.0, 5.0)], stroke);
                    ui.painter()
                        .line_segment([c + Vec2::new(-4.0, 1.0), c + Vec2::new(0.0, 5.0)], stroke);
                    ui.painter()
                        .line_segment([c + Vec2::new(4.0, 1.0), c + Vec2::new(0.0, 5.0)], stroke);
                    ui.add_space(6.0);
                    ui.vertical(|ui| {
                        ui.label(
                            RichText::new(if self.jobs.is_empty() {
                                "Drop GGUF files here"
                            } else {
                                "Add more files"
                            })
                            .size(13.5)
                            .strong()
                            .color(TEXT),
                        );
                        ui.add(
                            egui::Label::new(
                                RichText::new(
                                    "Several files are fine. Each one becomes a separate job.",
                                )
                                .size(10.5)
                                .color(MUTED),
                            )
                            .wrap(),
                        );
                    });
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        let browse = egui::Button::new(RichText::new("Browse files").size(11.0))
                            .fill(PANEL)
                            .stroke(Stroke::new(1.0, STEEL))
                            .corner_radius(CornerRadius::same(0));
                        if ui.add(browse).clicked() {
                            self.browse();
                        }
                    });
                });
                if self.jobs.is_empty() {
                    ui.add_space(10.0);
                    ui.separator();
                    ui.add_space(7.0);
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new(
                                "Choose a layout on the right, then drop one or many GGUFs.",
                            )
                            .size(10.5)
                            .color(MUTED),
                        );
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            if ui
                                .add(
                                    egui::Button::new(
                                        RichText::new("Open a Hugging Face link").size(10.5),
                                    )
                                    .frame(false),
                                )
                                .clicked()
                            {
                                self.tab = Tab::Discover;
                            }
                        });
                    });
                }
            });
    }

    fn queue_panel(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label(RichText::new("Queue").size(11.5).strong().color(TEXT));
            ui.label(
                RichText::new(format!("{}", self.jobs.len()))
                    .size(10.0)
                    .monospace()
                    .color(MUTED),
            );
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui
                    .add_enabled(
                        !self.is_running() && !self.jobs.is_empty(),
                        egui::Button::new(RichText::new("Clear").size(10.5).color(MUTED))
                            .frame(false),
                    )
                    .clicked()
                {
                    self.jobs.clear();
                }
                if ui
                    .add_enabled(
                        !self.is_running() && self.jobs.iter().any(|j| j.is_finished()),
                        egui::Button::new(RichText::new("Clean").size(10.5).color(MUTED))
                            .frame(false),
                    )
                    .clicked()
                {
                    self.jobs.retain(|j| !j.is_finished());
                }
            });
        });
        ui.add_space(4.0);
        ui.separator();
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
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(match batch.kind {
                        WorkKind::Convert => "Converting",
                        WorkKind::Verify => "Verifying",
                    })
                    .size(11.0)
                    .color(ACCENT),
                );
                ui.label(
                    RichText::new(format!(
                        "{} / {} files · {} elapsed",
                        batch.completed.min(batch.total),
                        batch.total,
                        human_duration(batch.started.elapsed())
                    ))
                    .size(10.0)
                    .color(MUTED),
                );
                if batch.failed > 0 {
                    ui.label(
                        RichText::new(format!("{} failed", batch.failed))
                            .size(10.0)
                            .color(ERROR),
                    );
                }
            });
            ui.add(
                ProgressBar::new(overall)
                    .fill(ACCENT)
                    .animate(true)
                    .text(format!("{:.0}%", overall * 100.0)),
            );
            ui.add_space(6.0);
        }
        if self.jobs.is_empty() {
            ui.add_space(22.0);
            ui.label(
                RichText::new("Nothing queued yet")
                    .size(15.0)
                    .strong()
                    .color(TEXT),
            );
            ui.add(
                egui::Label::new(
                    RichText::new(
                        "Drop a GGUF above, or pull one from Discover. Nothing is written until you press start.",
                    )
                    .size(11.0)
                    .color(MUTED),
                )
                .wrap(),
            );
            ui.add_space(8.0);
            let browse = egui::Button::new(RichText::new("Browse a model").size(11.0))
                .fill(PANEL)
                .stroke(Stroke::new(1.0, BORDER))
                .corner_radius(CornerRadius::same(0));
            if ui.add(browse).clicked() {
                self.browse();
            }
            return;
        }
        let running = self.is_running();
        let output_mode = self.settings.output_mode.clone();
        let mut remove = None;
        let mut open = None;
        for job in &self.jobs {
            job_card(ui, job, running, &output_mode, &mut remove, &mut open);
        }
        if let Some(id) = remove {
            self.jobs.retain(|j| j.id != id);
        }
        if let Some(path) = open {
            open_folder(&path);
        }
    }

    fn settings_panel(&mut self, ui: &mut Ui) {
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
            section_head(
                ui,
                "Layout",
                Some(match self.settings.target {
                    Target::Reblock => "reblock",
                    Target::F32 | Target::F16 | Target::Bf16 => "float",
                    Target::Q8_0
                    | Target::Q5_0
                    | Target::Q5_1
                    | Target::Q4_0
                    | Target::Q4_1 => "block",
                    Target::Tq1_0 | Target::Tq2_0 => "ternary",
                    Target::Mxfp4 | Target::Nvfp4 => "fp4",
                    Target::Q1_0 => "1-bit",
                    _ => "k-quant",
                }),
            );
            ui.add(
                egui::Label::new(
                    RichText::new(
                        "New to this? Q4_K is the usual choice. F16 keeps the most quality; Q2_K and Q1_0 save the most space. Q2_0 appears only for Prism PQ2_0 files.",
                    )
                    .size(10.5)
                    .color(MUTED),
                )
                .wrap(),
            );
            ui.add_space(7.0);
            let (primary, extra): (Vec<Target>, Vec<Target>) = available
                .iter()
                .copied()
                .partition(|target| !is_extra_target(*target));
            target_grid(ui, &primary, &mut self.settings.target);
            if !extra.is_empty() {
                let selected_extra = extra.contains(&self.settings.target);
                egui::CollapsingHeader::new(
                    RichText::new(format!("More formats · {}", extra.len()))
                        .size(10.5)
                        .color(MUTED),
                )
                .id_salt("extra_targets")
                .default_open(selected_extra)
                .show(ui, |ui| {
                    target_grid(ui, &extra, &mut self.settings.target);
                });
            }
            for job in &mut self.jobs {
                if job.analysis.can_target(self.settings.target) {
                    job.target = self.settings.target;
                }
            }

            ui.add_space(14.0);
            ui.separator();
            ui.add_space(10.0);
            section_head(ui, "Verification", Some("after write"));
            let vgap = 1.0;
            let vwidth = ((ui.available_width() - vgap * 2.0) / 3.0).max(48.0);
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = vgap;
                for (mode, label, hint) in [
                    (
                        VerifyMode::None,
                        "None",
                        "Fastest. Writes the output without re-reading it.",
                    ),
                    (
                        VerifyMode::Structure,
                        "Structure",
                        "Header, offsets, alignment and file size. Recommended.",
                    ),
                    (
                        VerifyMode::Full,
                        "Full",
                        "Decodes every tensor again and checks for gross errors.",
                    ),
                ] {
                    let selected = self.settings.verify == mode;
                    let button =
                        egui::Button::new(RichText::new(label).size(10.5).color(if selected {
                            INK
                        } else {
                            STEEL_LIGHT
                        }))
                        .fill(if selected { TEXT } else { PANEL_2 })
                        .stroke(Stroke::new(1.0, if selected { TEXT } else { BORDER }))
                        .corner_radius(CornerRadius::same(0))
                        .min_size(Vec2::new(vwidth, 28.0));
                    if ui.add(button).on_hover_text(hint).clicked() {
                        self.settings.verify = mode;
                    }
                }
            });
            ui.add_space(6.0);
            ui.add(
                egui::Label::new(
                    RichText::new(
                        "Structure is enough for most people. Full reads every weight back and compares it, so it takes longer.",
                    )
                    .size(10.0)
                    .color(MUTED),
                )
                .wrap(),
            );
            ui.checkbox(&mut self.settings.overwrite, "Overwrite existing outputs");
            ui.checkbox(
                &mut self.settings.verify_only,
                "Verify only, write nothing",
            )
            .on_hover_text(
                "Checks files without producing output. PQ2_0 files get the full lossless check.",
            );

            ui.add_space(14.0);
            ui.separator();
            ui.add_space(10.0);
            section_head(ui, "Destination", Some("where the output lands"));
            ui.label(
                RichText::new("Your original file stays where it is. This only chooses where the new GGUF is written.")
                    .size(10.0)
                    .color(MUTED),
            );
            let custom = matches!(self.settings.output_mode, OutputMode::Custom(_));
            ui.horizontal(|ui| {
                if ui.radio(!custom, "Next to input").clicked() {
                    self.settings.output_mode = OutputMode::NextToInput;
                }
                if ui.radio(custom, "Custom folder").clicked() && !custom {
                    let dir = self
                        .last_output_dir
                        .clone()
                        .or_else(|| std::env::current_dir().ok())
                        .unwrap_or_default();
                    self.settings.output_mode = OutputMode::Custom(dir);
                }
            });
            if let OutputMode::Custom(dir) = self.settings.output_mode.clone() {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(shorten(&dir.display().to_string(), 38))
                            .size(10.0)
                            .monospace()
                            .color(MUTED),
                    );
                    if ui
                        .add(egui::Button::new(RichText::new("Choose").size(10.0)).frame(false))
                        .clicked()
                    {
                        if let Some(picked) = rfd::FileDialog::new().pick_folder() {
                            self.last_output_dir = Some(picked.clone());
                            self.settings.output_mode = OutputMode::Custom(picked);
                        }
                    }
                });
            }

            ui.add_space(12.0);
            section_head(ui, "Install after conversion", Some("optional"));
            ui.label(
                RichText::new("Optional: copy the finished GGUF into LM Studio, or import it with Ollama.")
                    .size(10.0)
                    .color(MUTED),
            );
            let mut installs = self.detected.clone();
            if let InstallTarget::Folder(path) = &self.settings.install {
                if !installs.iter().any(|d| d.target == self.settings.install) {
                    installs.push(Detected {
                        label: "Custom folder".to_string(),
                        detail: path.display().to_string(),
                        target: InstallTarget::Folder(path.clone()),
                    });
                }
            }
            egui::ComboBox::from_id_salt("install_target")
                .width(ui.available_width())
                .selected_text(self.settings.install.label())
                .show_ui(ui, |ui| {
                    for detected in &installs {
                        ui.selectable_value(
                            &mut self.settings.install,
                            detected.target.clone(),
                            &detected.label,
                        )
                        .on_hover_text(&detected.detail);
                    }
                });
            ui.horizontal(|ui| {
                if ui
                    .add(egui::Button::new(RichText::new("Custom folder").size(10.0)).frame(false))
                    .clicked()
                {
                    if let Some(picked) = rfd::FileDialog::new().pick_folder() {
                        self.settings.install = InstallTarget::Folder(picked);
                    }
                }
                if ui
                    .add(egui::Button::new(RichText::new("Rescan").size(10.0)).frame(false))
                    .clicked()
                {
                    self.detected = targets::detect();
                }
            });
            ui.label(
                RichText::new("LM Studio gets a copy. Ollama uses `ollama create`.")
                    .size(10.0)
                    .color(MUTED),
            );
        });
    }

    fn start_controls(&mut self, ui: &mut Ui) {
        let count = self.selected_jobs();
        let running = self.is_running();
        let verify_only = self.settings.verify_only;
        let target = self.settings.target;
        let label = if verify_only {
            format!("Verify {count} file(s)")
        } else {
            "Start conversion".to_string()
        };
        let enabled = !running && count > 0;
        let button =
            egui::Button::new(RichText::new(label).size(12.0).strong().color(if enabled {
                INK
            } else {
                MUTED
            }))
            .fill(if enabled { ACCENT } else { PANEL_3 })
            .stroke(Stroke::new(1.0, if enabled { ACCENT } else { BORDER }))
            .corner_radius(CornerRadius::same(0))
            .min_size(Vec2::new(ui.available_width(), 38.0));
        if ui.add_enabled(enabled, button).clicked() {
            self.start(if verify_only {
                WorkKind::Verify
            } else {
                WorkKind::Convert
            });
        }
        ui.add_space(6.0);
        if running {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(RichText::new("Working").size(10.5).color(ACCENT));
                let stopping = self
                    .cancel
                    .as_ref()
                    .is_some_and(|c| c.load(Ordering::Relaxed));
                if ui
                    .add_enabled(
                        !stopping,
                        egui::Button::new(RichText::new("Stop after this file").size(10.0)),
                    )
                    .clicked()
                {
                    if let Some(cancel) = &self.cancel {
                        cancel.store(true, Ordering::Relaxed);
                    }
                }
            });
        } else {
            ui.label(
                RichText::new(if count == 0 {
                    "Add a file to enable conversion.".to_string()
                } else {
                    format!(
                        "{count} file{} queued. Nothing is written until you start.",
                        if count == 1 { "" } else { "s" }
                    )
                })
                .size(10.0)
                .color(MUTED),
            );
        }
        let _ = target;
    }

    fn discover_page(&mut self, ui: &mut Ui) {
        self.discover_header(ui);
        ui.add_space(10.0);
        ScrollArea::vertical()
            .id_salt("hf_results_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                self.hf_results(ui);
            });
    }

    fn discover_header(&mut self, ui: &mut Ui) {
        self.page_header(
            ui,
            "Discover",
            "Search by name, or paste a full Hugging Face link. Inspect the card, then download a file.",
            Some("index huggingface.co  ·  gguf"),
        );
        ui.add_space(14.0);
        let is_link = hf::parse_link(&self.hf_query).is_some();
        egui::Frame::new()
            .fill(PANEL)
            .stroke(Stroke::new(1.0, BORDER))
            .corner_radius(CornerRadius::same(0))
            .inner_margin(Margin::same(0))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    let width = (ui.available_width() - 88.0).max(180.0);
                    let response = ui.add_sized(
                        [width, 36.0],
                        TextEdit::singleline(&mut self.hf_query)
                            .hint_text("Model name, or paste huggingface.co/author/model")
                            .font(FontId::proportional(13.0))
                            .frame(egui::Frame::NONE)
                            .margin(Vec2::new(12.0, 9.0)),
                    );
                    if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        self.start_search();
                    }
                    let search_label = if is_link { "Open link" } else { "Search" };
                    let search = egui::Button::new(
                        RichText::new(search_label).size(11.5).strong().color(INK),
                    )
                    .fill(ACCENT)
                    .stroke(Stroke::new(1.0, ACCENT))
                    .corner_radius(CornerRadius::same(0))
                    .min_size(Vec2::new(88.0, 36.0));
                    if ui.add_enabled(!self.hf_busy(), search).clicked() {
                        self.start_search();
                    }
                });
            });
        ui.add_space(5.0);
        ui.add(
            egui::Label::new(
                RichText::new(
                    "Paste a model URL and press Enter to open its files directly. Dataset and Space links are not models.",
                )
                .size(10.0)
                .color(MUTED),
            )
            .wrap(),
        );
        ui.add_space(10.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new("Try").size(10.5).color(MUTED));
            for example in ["Qwen", "Llama", "Mistral", "vision"] {
                if ui
                    .add(
                        egui::Button::new(RichText::new(example).size(10.5).color(STEEL_LIGHT))
                            .frame(false),
                    )
                    .clicked()
                {
                    self.hf_query = example.to_string();
                    self.start_search();
                }
            }
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
                            egui::Button::new(RichText::new(label).size(10.5).color(if active {
                                ACCENT
                            } else {
                                MUTED
                            }))
                            .frame(false),
                        )
                        .clicked()
                    {
                        self.hf_sort = sort;
                        rerun = true;
                    }
                }
                ui.label(RichText::new("Sort").size(10.5).color(MUTED));
                if rerun && !self.hf_query.trim().is_empty() {
                    self.start_search();
                }
            });
        });
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let count = match &self.hf_state {
                HfState::Results(models) => format!("{} results", models.len()),
                HfState::Searching => "searching…".to_string(),
                HfState::Error(_) => "request failed".to_string(),
                HfState::Idle => "ready".to_string(),
            };
            ui.label(
                RichText::new(format!(
                    "{} · sorted by {} · {}",
                    count,
                    match self.hf_sort {
                        HfSort::Downloads => "downloads",
                        HfSort::Likes => "likes",
                        HfSort::Recent => "recent",
                    },
                    if self.hf_query.trim().is_empty() {
                        "top of the index"
                    } else {
                        self.hf_query.trim()
                    }
                ))
                .size(10.0)
                .monospace()
                .color(MUTED),
            );
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui
                    .add_enabled(
                        !self.hf_busy() && !self.hf_query.is_empty(),
                        egui::Button::new(RichText::new("Clear").size(10.5).color(MUTED))
                            .frame(false),
                    )
                    .clicked()
                {
                    self.hf_query.clear();
                    self.hf_state = HfState::Idle;
                    self.selected_repo = None;
                    self.hf_details = None;
                    self.hf_files.clear();
                }
            });
        });
        ui.label(
            RichText::new(
                "Gated or private repositories need HF_TOKEN set in the environment, with the model terms accepted.",
            )
            .size(10.0)
            .color(MUTED),
        );
    }

    fn hf_results(&mut self, ui: &mut Ui) {
        match self.hf_state.clone() {
            HfState::Idle => {
                ui.add_space(10.0);
                ui.label(
                    RichText::new("Search the GGUF index")
                        .size(15.0)
                        .strong()
                        .color(TEXT),
                );
                ui.label(
                    RichText::new(
                        "Type a model family, an author or a quantization, or paste a Hugging Face model link. The inspector shows real card data and every file size.",
                    )
                    .size(11.0)
                    .color(MUTED),
                );
                ui.add_space(8.0);
                if ui
                    .add(
                        egui::Button::new(RichText::new("Show most downloaded").size(11.0))
                            .fill(PANEL)
                            .stroke(Stroke::new(1.0, BORDER)),
                    )
                    .clicked()
                {
                    self.hf_query.clear();
                    self.start_search();
                }
            }
            HfState::Searching => {
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(
                        RichText::new("Querying Hugging Face…")
                            .size(11.5)
                            .color(STEEL_LIGHT),
                    );
                });
            }
            HfState::Error(error) => {
                ui.add_space(10.0);
                ui.label(
                    RichText::new("Search failed")
                        .size(14.0)
                        .strong()
                        .color(ERROR),
                );
                ui.add(egui::Label::new(RichText::new(error).size(11.0).color(MUTED)).wrap());
            }
            HfState::Results(models) => {
                if models.is_empty() {
                    ui.add_space(10.0);
                    ui.label(
                        RichText::new("No GGUF repositories found. Try a broader name.")
                            .size(11.0)
                            .color(MUTED),
                    );
                    return;
                }
                ui.add_space(8.0);
                hf_result_header(ui);
                let selected = self.selected_repo.clone();
                for model in &models {
                    let is_selected = selected.as_deref() == Some(model.id.as_str());
                    let clicked = hf_result_row(ui, model, is_selected);
                    if clicked && !is_selected && !self.hf_busy() {
                        self.link_file = None;
                        self.start_details(model.id.clone(), model.sha.clone());
                    }
                }
            }
        }
    }

    fn hf_details_panel(&mut self, ui: &mut Ui) {
        ui.set_max_width(ui.available_width());
        let Some(repo) = self.selected_repo.clone() else {
            ui.add_space(18.0);
            ui.label(
                RichText::new("No model selected")
                    .size(14.0)
                    .strong()
                    .color(TEXT),
            );
            ui.add(
                egui::Label::new(
                    RichText::new(
                        "Pick a repository on the left to see its README, architecture, context window, license and every GGUF file with its real size.",
                    )
                    .size(11.0)
                    .color(MUTED),
                )
                .wrap(),
            );
            return;
        };
        let model = self.current_model();
        let details = self.hf_details.clone();
        let loading = self.hf_details_loading;
        let error = self.hf_details_error.clone();
        let repo_url = model
            .as_ref()
            .map(HfModel::web_url)
            .unwrap_or_else(|| format!("https://huggingface.co/{repo}"));

        if let Some(model) = &model {
            ui.label(RichText::new(&model.author).size(10.5).color(MUTED));
            ui.add(
                egui::Label::new(RichText::new(&model.name).size(16.0).strong().color(TEXT)).wrap(),
            );
        } else {
            ui.add(egui::Label::new(RichText::new(&repo).size(14.0).strong().color(TEXT)).wrap());
        }
        ui.horizontal_wrapped(|ui| {
            ui.hyperlink_to("Open ↗", &repo_url);
            if model.as_ref().is_some_and(|m| m.gated) {
                ui.label(RichText::new("Gated").size(10.5).color(WARN));
            }
        });
        let mut meta = Vec::new();
        if let Some(model) = &model {
            if !model.pipeline.is_empty() {
                meta.push(model.pipeline.clone());
            }
        }
        if let Some(license) = details
            .as_ref()
            .and_then(|d| d.license.clone())
            .or_else(|| model.as_ref().and_then(|m| m.license.clone()))
        {
            meta.push(license);
        }
        if !meta.is_empty() {
            ui.label(RichText::new(meta.join(" · ")).size(10.5).color(MUTED));
        }
        if loading {
            ui.add_space(5.0);
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(
                    RichText::new("Reading the model card and GGUF tree…")
                        .size(10.0)
                        .color(MUTED),
                );
            });
        }
        if let Some(error) = error {
            ui.add_space(5.0);
            ui.add(egui::Label::new(RichText::new(error).size(10.0).color(ERROR)).wrap());
        }

        if let Some(model) = &model {
            ui.add_space(10.0);
            ui.separator();
            ui.add_space(7.0);
            ui.columns(3, |cols| {
                metric_cell(&mut cols[0], &format_count(model.downloads), "downloads");
                metric_cell(&mut cols[1], &format_count(model.likes), "likes");
                metric_cell(&mut cols[2], &format!("{}", model.gguf_files), "GGUF files");
            });
            ui.add_space(7.0);
            ui.separator();
        }

        ui.add_space(10.0);
        let updated = model
            .as_ref()
            .map(|m| format!("updated {}", short_date(&m.last_modified)));
        section_head(ui, "About", updated.as_deref());
        if let Some(description) = details.as_ref().and_then(|d| d.description.clone()) {
            ui.add(
                egui::Label::new(RichText::new(description).size(11.5).color(STEEL_LIGHT)).wrap(),
            );
        } else if loading {
            ui.label(
                RichText::new("Fetching the first useful paragraph…")
                    .size(10.0)
                    .color(MUTED),
            );
        } else {
            ui.label(
                RichText::new("This repository has no readable README summary.")
                    .size(10.0)
                    .color(MUTED),
            );
        }

        ui.add_space(10.0);
        ui.separator();
        ui.add_space(8.0);
        section_head(ui, "Facts", Some("from the model card"));
        let d = details.as_ref();
        let context = d
            .and_then(|d| d.context_length)
            .map(|v| format!("{} tokens", format_count(v)));
        let payload = d.and_then(|d| d.total_gguf).map(human_bytes);
        let storage = d.and_then(|d| d.used_storage).map(human_bytes);
        let license = d
            .and_then(|d| d.license.clone())
            .or_else(|| model.as_ref().and_then(|m| m.license.clone()));
        let base = d
            .and_then(|d| d.base_model.clone())
            .or_else(|| model.as_ref().and_then(|m| m.base_model.clone()));
        let created = model.as_ref().map(|m| short_date(&m.created_at));
        egui::Grid::new("hf_facts")
            .num_columns(2)
            .spacing([12.0, 5.0])
            .show(ui, |ui| {
                fact(
                    ui,
                    "Architecture",
                    d.and_then(|d| d.architecture.as_deref()),
                );
                fact(
                    ui,
                    "Library",
                    model
                        .as_ref()
                        .map(|m| m.library.as_str())
                        .filter(|v| !v.is_empty()),
                );
                fact(ui, "Context", context.as_deref());
                fact(ui, "GGUF payload", payload.as_deref());
                fact(ui, "Repository size", storage.as_deref());
                fact(ui, "License", license.as_deref());
                fact(ui, "Base model", base.as_deref());
                fact(ui, "Created", created.as_deref());
                if let Some(text) = d
                    .filter(|d| !d.languages.is_empty())
                    .map(|d| d.languages.join(", "))
                {
                    fact(ui, "Languages", Some(&text));
                }
                if let Some(model) = &model {
                    let tags: Vec<&str> = model
                        .tags
                        .iter()
                        .filter(|tag| {
                            !tag.starts_with("base_model:")
                                && !tag.starts_with("license:")
                                && tag.as_str() != "gguf"
                                && tag.as_str() != "region:us"
                                && tag.as_str() != "endpoints_compatible"
                        })
                        .take(6)
                        .map(String::as_str)
                        .collect();
                    if !tags.is_empty() {
                        fact(ui, "Tags", Some(&tags.join(", ")));
                    }
                }
            });

        ui.add_space(10.0);
        ui.separator();
        ui.add_space(8.0);
        let files = self.hf_files.clone();
        let filter = self.hf_file_filter.trim().to_ascii_lowercase();
        let filtered: Vec<&HfFile> = files
            .iter()
            .filter(|f| filter.is_empty() || f.path.to_ascii_lowercase().contains(&filter))
            .collect();
        section_head(ui, "GGUF files", Some(&format!("{} files", files.len())));
        ui.add(
            egui::Label::new(
                RichText::new(
                    "Choose one file to download. Q4_K is a good default; smaller numbers mean a smaller file and more quality loss.",
                )
                .size(9.5)
                .color(MUTED),
            )
            .wrap(),
        );
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(format!("{} of {}", filtered.len(), files.len()))
                    .size(9.5)
                    .monospace()
                    .color(MUTED),
            );
            let width = (ui.available_width() - 4.0).max(80.0);
            ui.add_sized(
                [width, 24.0],
                TextEdit::singleline(&mut self.hf_file_filter)
                    .hint_text("filter by quant or folder")
                    .font(FontId::proportional(10.5)),
            );
        });
        ui.add_space(2.0);
        if self.hf_details_loading {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(RichText::new("Loading files…").size(10.0).color(MUTED));
            });
        } else if files.is_empty() {
            ui.label(
                RichText::new("This repository has no GGUF files to download.")
                    .size(10.0)
                    .color(MUTED),
            );
        } else if filtered.is_empty() {
            ui.label(
                RichText::new("No GGUF files match this filter.")
                    .size(10.0)
                    .color(MUTED),
            );
        }
        let mut download_start: Option<(String, HfFile)> = None;
        for file in filtered {
            hf_file_row(ui, &repo, file, self.hf_busy(), &mut download_start);
        }
        if let Some((repo, file)) = download_start {
            self.start_download(repo, file);
        }
        ui.add_space(3.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new("Download to").size(10.0).color(MUTED));
            ui.label(
                RichText::new(shorten(
                    &self.settings.download_dir.display().to_string(),
                    38,
                ))
                .size(9.5)
                .monospace()
                .color(STEEL_LIGHT),
            );
            if ui
                .add(egui::Button::new(RichText::new("Choose").size(10.0)).frame(false))
                .clicked()
            {
                if let Some(picked) = rfd::FileDialog::new().pick_folder() {
                    self.settings.download_dir = picked;
                }
            }
        });
        ui.add(
            egui::Label::new(
                RichText::new(
                    "The GGUF is saved here and added to the Convert queue automatically when the download finishes.",
                )
                .size(9.5)
                .color(MUTED),
            )
            .wrap(),
        );

        if let Some(download) = self.download.clone() {
            ui.add_space(10.0);
            ui.separator();
            ui.add_space(8.0);
            section_head(ui, "Transfer", None);
            ui.label(
                RichText::new(format!("{} · {}", download.repo, download.file))
                    .size(9.5)
                    .monospace()
                    .color(MUTED),
            );
            let frac = if download.total > 0 {
                download.done as f32 / download.total as f32
            } else {
                0.0
            };
            ui.add(
                ProgressBar::new(frac)
                    .fill(ACCENT)
                    .animate(true)
                    .text(format!(
                        "{} / {} · {:.0}% · {}",
                        human_bytes(download.done),
                        human_bytes(download.total),
                        frac * 100.0,
                        human_speed(download.speed)
                    )),
            );
            if self.hf_busy()
                && ui
                    .add(
                        egui::Button::new(RichText::new("Cancel download").size(10.0))
                            .fill(PANEL)
                            .stroke(Stroke::new(1.0, BORDER)),
                    )
                    .clicked()
            {
                self.hf_cancel.store(true, Ordering::Relaxed);
            }
            if let Some(path) = &download.path {
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        RichText::new(format!("Saved and added to Convert: {}", path.display()))
                            .size(9.5)
                            .color(STEEL_LIGHT),
                    );
                    if ui
                        .add(egui::Button::new(RichText::new("Open folder").size(9.5)).frame(false))
                        .clicked()
                    {
                        open_folder(path);
                    }
                });
            }
            if let Some(error) = &download.error {
                ui.label(RichText::new(error).size(9.5).color(ERROR));
            }
        }
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

    fn about_window(&mut self, ctx: &egui::Context) {
        if !self.show_about {
            return;
        }
        let logo = self.logo.clone();
        let mut open = true;
        let mut close = false;
        egui::Window::new("About RustBlock")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .default_width(520.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    theme::paint_logo(ui, 56.0, logo.as_ref());
                    ui.add_space(6.0);
                    ui.vertical(|ui| {
                        ui.label(RichText::new("RustBlock").size(22.0).strong().color(TEXT));
                        ui.label(
                            RichText::new("GGUF workshop · native codecs")
                                .size(9.0)
                                .monospace()
                                .color(STEEL),
                        );
                    });
                });
                ui.add_space(10.0);
                ui.label(
                    RichText::new(
                        "RustBlock converts GGUF models locally. It never uploads a model and never modifies the source file. Outputs are written atomically after verification.",
                    )
                    .size(12.0)
                    .color(TEXT),
                );
                ui.add_space(6.0);
                ui.label(
                    RichText::new(
                        "Lossless PQ2_0 → Q2_0 reblocking, F32/F16/BF16 conversion, and native block, K-quant, ternary, FP4 and 1-bit GGUF targets. Raw safetensors → GGUF needs a model-specific architecture converter and is intentionally not claimed as supported.",
                    )
                    .size(10.5)
                    .color(MUTED),
                );
                ui.add_space(8.0);
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(format!("Version {}", env!("CARGO_PKG_VERSION")))
                            .size(9.5)
                            .monospace()
                            .color(MUTED),
                    );
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui.button("Close").clicked() {
                            close = true;
                        }
                    });
                });
            });
        self.show_about = open && !close;
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
        if !self.is_running() {
            self.close_warned = false;
        }
        if self.is_running() && ctx.input(|i| i.viewport().close_requested()) {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            if !self.close_warned {
                self.close_warned = true;
                self.log(
                    LogLevel::Warn,
                    "A conversion is running. Stop it before closing the window.".to_string(),
                );
            }
        }
        self.handle_dropped(ctx);
        self.poll_worker(ctx);
        self.poll_hf();
        let start = ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::Enter));
        if start && !self.is_running() {
            let kind = if self.settings.verify_only {
                WorkKind::Verify
            } else {
                WorkKind::Convert
            };
            self.start(kind);
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::Num1)) {
            self.tab = Tab::Convert;
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::Num2)) {
            self.tab = Tab::Discover;
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::F)) {
            self.tab = Tab::Discover;
            ctx.memory_mut(|memory| memory.request_focus(Id::new("hf_search")));
        }
        if ctx.input(|i| i.key_pressed(egui::Key::F12)) {
            self.show_log = !self.show_log;
        }
        if self.rx.is_some() || self.hf_rx.is_some() {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }

    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        self.top_chrome(ui);
        self.status_bar(ui);
        if self.show_log {
            self.activity_panel(ui);
        }
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
        self.about_window(ui.ctx());
    }
}

fn hf_result_header(ui: &mut Ui) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 22.0), Sense::hover());
    let painter = ui.painter();
    let x = |f: f32| rect.left() + rect.width() * f;
    let y = rect.center().y;
    painter.text(
        egui::pos2(rect.left() + 10.0, y),
        Align2::LEFT_CENTER,
        "Repository",
        FontId::proportional(9.5),
        MUTED,
    );
    painter.text(
        egui::pos2(x(0.50), y),
        Align2::LEFT_CENTER,
        "Pipeline",
        FontId::proportional(9.5),
        MUTED,
    );
    painter.text(
        egui::pos2(x(0.68), y),
        Align2::LEFT_CENTER,
        "Downloads",
        FontId::proportional(9.5),
        MUTED,
    );
    painter.text(
        egui::pos2(x(0.77), y),
        Align2::LEFT_CENTER,
        "Likes",
        FontId::proportional(9.5),
        MUTED,
    );
    painter.text(
        egui::pos2(x(0.85), y),
        Align2::LEFT_CENTER,
        "GGUF",
        FontId::proportional(9.5),
        MUTED,
    );
    painter.text(
        egui::pos2(rect.right() - 8.0, y),
        Align2::RIGHT_CENTER,
        "Updated",
        FontId::proportional(9.5),
        MUTED,
    );
    painter.line_segment(
        [
            egui::pos2(rect.left(), rect.bottom()),
            egui::pos2(rect.right(), rect.bottom()),
        ],
        Stroke::new(1.0, BORDER),
    );
}

fn hf_result_row(ui: &mut Ui, model: &HfModel, selected: bool) -> bool {
    let (rect, response) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), 48.0), Sense::click());
    if selected {
        ui.painter()
            .rect_filled(rect, CornerRadius::same(0), ACCENT.gamma_multiply(0.10));
        ui.painter().rect_filled(
            egui::Rect::from_min_size(rect.min, Vec2::new(3.0, rect.height())),
            CornerRadius::same(0),
            ACCENT,
        );
    } else if response.hovered() {
        ui.painter()
            .rect_filled(rect, CornerRadius::same(0), PANEL_2);
    }
    ui.painter().line_segment(
        [
            egui::pos2(rect.left(), rect.bottom()),
            egui::pos2(rect.right(), rect.bottom()),
        ],
        Stroke::new(1.0, BORDER),
    );
    let x = |f: f32| rect.left() + rect.width() * f;
    let clip = egui::Rect::from_min_max(
        egui::pos2(rect.left() + 10.0, rect.top()),
        egui::pos2(x(0.50) - 10.0, rect.bottom()),
    );
    let painter = ui.painter().with_clip_rect(clip);
    painter.text(
        egui::pos2(rect.left() + 10.0, rect.top() + 10.0),
        Align2::LEFT_CENTER,
        &model.author,
        FontId::proportional(10.0),
        MUTED,
    );
    painter.text(
        egui::pos2(rect.left() + 10.0, rect.top() + 29.0),
        Align2::LEFT_CENTER,
        &model.name,
        FontId::proportional(12.0),
        TEXT,
    );
    let painter = ui.painter();
    painter.text(
        egui::pos2(x(0.50), rect.top() + 19.0),
        Align2::LEFT_CENTER,
        shorten(&model.pipeline, 22),
        FontId::proportional(10.5),
        STEEL_LIGHT,
    );
    let mono = FontId::monospace(10.0);
    painter.text(
        egui::pos2(x(0.68), rect.top() + 19.0),
        Align2::LEFT_CENTER,
        format_count(model.downloads),
        mono.clone(),
        STEEL_LIGHT,
    );
    painter.text(
        egui::pos2(x(0.77), rect.top() + 19.0),
        Align2::LEFT_CENTER,
        format_count(model.likes),
        mono.clone(),
        STEEL_LIGHT,
    );
    painter.text(
        egui::pos2(x(0.85), rect.top() + 19.0),
        Align2::LEFT_CENTER,
        model.gguf_files.to_string(),
        mono.clone(),
        STEEL_LIGHT,
    );
    painter.text(
        egui::pos2(rect.right() - 8.0, rect.top() + 19.0),
        Align2::RIGHT_CENTER,
        short_date(&model.last_modified),
        mono,
        STEEL_LIGHT,
    );
    if response.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    response.clicked()
}

fn section_head(ui: &mut Ui, title: &str, hint: Option<&str>) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(title).size(11.5).strong().color(TEXT));
        if let Some(hint) = hint {
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.label(RichText::new(hint).size(10.0).color(MUTED));
            });
        }
    });
    ui.add_space(6.0);
}

fn how_it_works(ui: &mut Ui) {
    egui::Frame::new()
        .fill(PANEL_2)
        .stroke(Stroke::new(1.0, BORDER))
        .corner_radius(CornerRadius::same(0))
        .inner_margin(Margin::symmetric(14, 10))
        .show(ui, |ui| {
            ui.columns(3, |cols| {
                let steps = [
                    (
                        "Add a model",
                        "Drop a .gguf here, or download one from Discover. The original is never changed.",
                    ),
                    (
                        "Choose a layout",
                        "Pick an output on the right. If you are unsure, Q4_K is the usual balance.",
                    ),
                    (
                        "Start",
                        "RustBlock writes a new file, checks it, then leaves it in the destination.",
                    ),
                ];
                for (col, (title, text)) in cols.iter_mut().zip(steps) {
                    col.label(RichText::new(title).size(11.5).strong().color(TEXT));
                    col.add(
                        egui::Label::new(RichText::new(text).size(10.0).color(MUTED)).wrap(),
                    );
                }
            });
        });
}

fn is_extra_target(target: Target) -> bool {
    matches!(
        target,
        Target::Q5_1
            | Target::Q4_1
            | Target::Tq1_0
            | Target::Tq2_0
            | Target::Mxfp4
            | Target::Nvfp4
            | Target::Q1_0
    )
}

fn target_grid(ui: &mut Ui, targets: &[Target], selected: &mut Target) {
    let gap = 1.0;
    let width = ((ui.available_width() - gap * 2.0) / 3.0).max(48.0);
    for chunk in targets.chunks(3) {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = gap;
            for target in chunk {
                let is_selected = *selected == *target;
                let button = egui::Button::new(
                    RichText::new(target.short())
                        .size(11.0)
                        .color(if is_selected { INK } else { STEEL_LIGHT }),
                )
                .fill(if is_selected { ACCENT } else { PANEL_2 })
                .stroke(Stroke::new(1.0, if is_selected { ACCENT } else { BORDER }))
                .corner_radius(CornerRadius::same(0))
                .min_size(Vec2::new(width, 32.0));
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
    output_mode: &OutputMode,
    remove: &mut Option<u64>,
    open: &mut Option<PathBuf>,
) {
    let color = status_color(job);
    let frame = egui::Frame::new()
        .fill(Color32::TRANSPARENT)
        .inner_margin(Margin::symmetric(0, 10))
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            ui.horizontal(|ui| {
                let (dot, _) = ui.allocate_exact_size(Vec2::splat(6.0), Sense::hover());
                ui.painter().rect_filled(dot, CornerRadius::same(0), color);
                ui.vertical(|ui| {
                    ui.label(
                        RichText::new(job.file_name())
                            .size(12.5)
                            .strong()
                            .color(TEXT),
                    );
                    ui.label(
                        RichText::new(format!(
                            "{} · {} · {} tensors",
                            human_bytes(job.analysis.file_len),
                            job.analysis.kind_label(),
                            job.analysis.tensors
                        ))
                        .size(10.0)
                        .color(MUTED),
                    )
                    .on_hover_text(format!(
                        "GGUF v{} · alignment {} B · {}",
                        job.analysis.version,
                        job.analysis.alignment,
                        job.analysis.type_summary()
                    ));
                });
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui
                        .add_enabled(
                            !running,
                            egui::Button::new(RichText::new("Remove").size(10.0).color(MUTED))
                                .frame(false),
                        )
                        .on_hover_text("Remove from queue")
                        .clicked()
                    {
                        *remove = Some(job.id);
                    }
                    ui.label(RichText::new(status_label(job)).size(10.0).color(color));
                    ui.label(
                        RichText::new(job.target.short())
                            .size(10.0)
                            .monospace()
                            .color(STEEL_LIGHT),
                    );
                });
            });
            if job.analysis.error.is_none() && !job.analysis.supported {
                ui.label(
                    RichText::new("Contains a tensor type RustBlock cannot decode natively.")
                        .size(10.0)
                        .color(WARN),
                );
            }
            if job.analysis.can_target(job.target) {
                let output = output_path(&job.input, job.target, output_mode);
                ui.label(
                    RichText::new(shorten(&output.display().to_string(), 78))
                        .size(9.5)
                        .monospace()
                        .color(MUTED),
                );
            }
            match &job.status {
                JobStatus::Running(state) => {
                    ui.add_space(3.0);
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
                    ui.horizontal_wrapped(|ui| {
                        ui.label(
                            RichText::new(format!("{} / s", human_bytes(state.speed as u64)))
                                .size(9.5)
                                .monospace()
                                .color(MUTED),
                        );
                        if let Some(eta) = state.eta() {
                            ui.label(
                                RichText::new(format!("ETA {}", human_duration(eta)))
                                    .size(9.5)
                                    .monospace()
                                    .color(MUTED),
                            );
                        }
                        if !state.current_tensor.is_empty() {
                            ui.label(
                                RichText::new(shorten(&state.current_tensor, 42))
                                    .size(9.5)
                                    .monospace()
                                    .color(STEEL),
                            );
                        }
                    });
                }
                JobStatus::Done {
                    report,
                    target,
                    install,
                } => {
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
                    if let Some(InstallResult::Installed(path)) = install {
                        ui.label(
                            RichText::new(format!("Installed → {}", shorten(path, 58)))
                                .size(9.5)
                                .color(INFO),
                        );
                    }
                    if let Some(InstallResult::Failed(error)) = install {
                        ui.label(RichText::new(error).size(9.5).color(WARN));
                    }
                }
                JobStatus::Verified { outcome, elapsed } => {
                    ui.label(
                        RichText::new(format!(
                            "{} · {} tensors · {} · {}",
                            outcome.label,
                            outcome.tensors,
                            human_duration(*elapsed),
                            match outcome.kind {
                                VerifyKind::Exact => "exact",
                                VerifyKind::Structure => "structure",
                            }
                        ))
                        .size(10.0)
                        .color(STEEL_LIGHT),
                    );
                }
                JobStatus::Failed(error) => {
                    ui.label(RichText::new(error).size(10.0).color(ERROR));
                }
                JobStatus::Ready => {}
            }
        });
    let rect = frame.response.rect;
    ui.painter().rect_filled(
        egui::Rect::from_min_size(rect.min, Vec2::new(2.0, rect.height())),
        CornerRadius::same(0),
        color,
    );
    ui.add_space(2.0);
    ui.separator();
}

fn hf_file_row(
    ui: &mut Ui,
    repo: &str,
    file: &HfFile,
    busy: bool,
    download_start: &mut Option<(String, HfFile)>,
) {
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.label(
                RichText::new(shorten(&file.path, 40))
                    .size(9.5)
                    .monospace()
                    .color(STEEL_LIGHT),
            )
            .on_hover_text(&file.path);
            ui.label(
                RichText::new(human_bytes(file.size))
                    .size(9.0)
                    .monospace()
                    .color(MUTED),
            );
        });
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if ui
                .add_enabled(
                    !busy,
                    egui::Button::new(RichText::new("Download").size(10.0).color(ACCENT))
                        .frame(false),
                )
                .on_hover_text(
                    "Download this GGUF to the folder chosen below, then add it to Convert",
                )
                .clicked()
            {
                *download_start = Some((repo.to_string(), file.clone()));
            }
        });
    });
    ui.separator();
}

fn metric_cell(ui: &mut Ui, value: &str, label: &str) {
    ui.label(
        RichText::new(value)
            .size(12.5)
            .monospace()
            .strong()
            .color(TEXT),
    );
    ui.label(RichText::new(label).size(9.5).color(MUTED));
}

fn fact(ui: &mut Ui, label: &str, value: Option<&str>) {
    ui.label(RichText::new(label).size(10.5).color(MUTED));
    let text = value.unwrap_or("—");
    ui.add(egui::Label::new(RichText::new(text).size(10.5).color(STEEL_LIGHT)).wrap());
    ui.end_row();
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

fn format_files(n: usize) -> String {
    if n == 1 {
        "1 file".to_string()
    } else {
        format!("{n} files")
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
