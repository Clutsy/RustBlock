//! RustBlock — GUI state, messages and effects.
//!
//! The behaviour is the one the egui build had; only the framework changed.
//!
//! - **Convert**: queue of GGUF files, target selection, verification mode,
//!   batches with progress, cancel and a requantization warning.
//! - **Discover**: Hugging Face search, repository files, resumable downloads
//!   and a pre-download fit verdict.
//! - **Inspector**: facts, metadata and tensor histogram of a local GGUF.
//!
//! Rendering lives in [`crate::view`]; this module owns the state machine so it
//! can be tested without a window.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use iced::{Subscription, Task};

use crate::deploy::{self, DeployEvent, DeployUi, Runtime};
use crate::hf::{self, HfEvent, HfFile, HfModel, HfSort};
use crate::model::{
    analyze, human_bytes, human_duration, looks_quantized_name, output_path, path_key,
    DownloadState, FileKind, InspectorData, Job, JobStatus, OutputMode, RunState, Settings, Source,
};
use crate::theme::{self, ERROR, MUTED, OK, WARN};
use crate::worker::{self, Event, WorkItem, WorkKind};
use rustblock_core::imatrix::find_tool;
use rustblock_core::{plan_budget, BudgetRequest, Hardware, Target, VerifyMode};

/// The three views of the app.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Convert,
    Discover,
    Inspector,
}

impl Tab {
    pub fn label(self) -> &'static str {
        match self {
            Self::Convert => "Convert",
            Self::Discover => "Discover",
            Self::Inspector => "Inspector",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Info,
    Ok,
    Warn,
    Error,
}

impl LogLevel {
    pub fn index(self) -> usize {
        match self {
            Self::Info => 0,
            Self::Ok => 1,
            Self::Warn => 2,
            Self::Error => 3,
        }
    }

    pub fn color(self) -> iced::Color {
        match self {
            Self::Info => MUTED,
            Self::Ok => OK,
            Self::Warn => WARN,
            Self::Error => ERROR,
        }
    }

    pub fn short(self) -> &'static str {
        match self {
            Self::Info => "I",
            Self::Ok => "✓",
            Self::Warn => "⚠",
            Self::Error => "✗",
        }
    }
}

#[derive(Debug, Clone)]
pub struct LogEntry {
    pub level: LogLevel,
    pub text: String,
}

pub fn log_entry_visible(entry: &LogEntry, filter: &str, levels: &[bool; 4]) -> bool {
    levels[entry.level.index()]
        && (filter.is_empty()
            || entry
                .text
                .to_ascii_lowercase()
                .contains(&filter.to_ascii_lowercase()))
}

#[derive(Debug, Clone)]
pub struct Batch {
    pub kind: WorkKind,
    pub total: usize,
    pub completed: usize,
    pub failed: usize,
    pub active: Option<u64>,
    pub started: Instant,
}

impl Batch {
    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }
}

#[derive(Debug, Clone)]
pub struct DownloadUi {
    pub id: u64,
    pub source: Source,
    pub repo: String,
    pub file: String,
    pub revision: String,
    pub expected_sha: Option<String>,
    pub done: u64,
    pub total: u64,
    pub speed: f64,
    pub quantized: bool,
    pub projector: bool,
    pub path: Option<PathBuf>,
    pub error: Option<String>,
    pub state: DownloadState,
}

impl DownloadUi {
    pub fn fraction(&self) -> f32 {
        if self.total > 0 {
            (self.done as f64 / self.total as f64).clamp(0.0, 1.0) as f32
        } else {
            0.0
        }
    }

    pub fn file_name(&self) -> String {
        hf::file_name(&self.file).to_string()
    }
}

#[derive(Debug)]
pub enum BenchmarkEvent {
    Done { ok: bool, text: String },
}

#[derive(Debug)]
pub struct BenchmarkUi {
    pub model: PathBuf,
    pub receiver: Receiver<BenchmarkEvent>,
    pub handle: JoinHandle<()>,
}

#[derive(Debug, Clone)]
pub enum HfState {
    Idle,
    Searching,
    Results(Vec<HfModel>),
    Error(String),
}

/// Actions the view can trigger.
#[derive(Debug, Clone)]
pub enum Message {
    Tick,
    WindowOpened(iced::window::Id),
    DragWindow,
    Resize(iced::window::Direction),
    ToggleMaximize,
    Minimize,
    CloseWindow,
    TabSelected(Tab),
    ToggleLog,
    ToggleHelp,
    ToggleSettings,
    SetSource(Source),
    SetAutoAdd(bool),
    SetConfirmRequantize(bool),
    HfEndpointChanged(String),
    PickLmStudioDir,
    PickJanDir,
    PickPrismDir,
    ResetSettings,
    Browse,
    FilesDropped(PathBuf),
    ClearQueue,
    SelectJob(u64),
    RemoveJob(u64),
    StartConvert,
    StartVerify,
    CancelBatch,
    ConfirmRequantize(bool),
    SetTarget(Target),
    SetVerify(VerifyMode),
    SetOverwrite(bool),
    PickOutputDir,
    UseSourceDir,
    SuggestTarget,
    Benchmark,
    InspectorOpen,
    InspectorRefresh,
    InspectorPathChanged(String),
    InspectorLoad,
    InspectPath(PathBuf),
    OpenPath(PathBuf),
    SearchChanged(String),
    SearchSubmit,
    SetSort(HfSort),
    Trending,
    OpenRepo(String),
    AddCached(PathBuf),
    Download(HfFile),
    CancelDownload(u64),
    DismissDownload(u64),
    ClearFinishedDownloads,
    FileFilterChanged(String),
    PickDownloadDir,
    OpenUrl(String),
    ClearPartials,
    Deploy(PathBuf),
    DeployTo(Runtime),
    DeployCancel,
    DeployClose,
    ConfirmClose(bool),
    LogFilterChanged(String),
    ToggleLevel(usize),
    ClearLog,
}

pub struct RustBlock {
    pub tab: Tab,
    pub jobs: Vec<Job>,
    pub next_id: u64,
    pub selected_job: Option<u64>,
    pub settings: Settings,
    pub rx: Option<Receiver<Event>>,
    pub worker: Option<JoinHandle<()>>,
    pub cancel: Option<Arc<AtomicBool>>,
    pub batch: Option<Batch>,
    pub last_output_dir: Option<PathBuf>,
    pub benchmark: Option<BenchmarkUi>,
    pub logs: Vec<LogEntry>,
    pub show_log: bool,
    pub log_filter: String,
    pub log_levels: [bool; 4],
    pub show_help: bool,
    pub inspector_path: Option<PathBuf>,
    pub inspector: Option<InspectorData>,
    pub hf_query: String,
    pub hf_sort: HfSort,
    pub hf_state: HfState,
    pub hf_rx: Option<Receiver<HfEvent>>,
    pub hf_worker: Option<JoinHandle<()>>,
    pub hf_cancel: Arc<AtomicBool>,
    pub selected_repo: Option<String>,
    pub selected_sha: String,
    pub hf_files: Vec<HfFile>,
    pub hf_details_loading: bool,
    pub pending_details: Option<(String, String)>,
    pub hf_details_error: Option<String>,
    pub hf_file_filter: String,
    pub discover_loaded: bool,
    pub downloads: Vec<DownloadUi>,
    pub next_download_id: u64,
    pub dl_rx: Option<Receiver<HfEvent>>,
    pub dl_worker: Option<JoinHandle<()>>,
    pub dl_cancel: Arc<AtomicBool>,
    pub confirm_requantize: Option<WorkKind>,
    pub show_settings: bool,
    pub confirm_close: bool,
    pub inspector_error: Option<String>,
    pub inspector_input: String,
    pub deploy: Option<DeployUi>,
    pub maximized: bool,
    pub window: Option<iced::window::Id>,
    pub hardware: Hardware,
    pub logo: iced::widget::image::Handle,
    pub wordmark: iced::widget::image::Handle,
    pub wordmark_aspect: f32,
}

fn file_name_of(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
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

fn listen(
    event: iced::Event,
    _status: iced::event::Status,
    _window: iced::window::Id,
) -> Option<Message> {
    use iced::keyboard::{key::Named, Event as Keyboard, Key};

    match event {
        iced::Event::Window(iced::window::Event::FileDropped(path)) => {
            Some(Message::FilesDropped(path))
        }
        iced::Event::Keyboard(Keyboard::KeyPressed { key, modifiers, .. }) => match key {
            Key::Named(Named::F1) => Some(Message::ToggleHelp),
            Key::Character(character) if modifiers.control() => match character.as_str() {
                "o" | "O" => Some(Message::Browse),
                "s" | "S" => Some(Message::StartConvert),
                "l" | "L" => Some(Message::ToggleLog),
                _ => None,
            },
            _ => None,
        },
        _ => None,
    }
}

impl RustBlock {
    pub fn boot() -> Self {
        let mut app = Self::empty();
        let files: Vec<PathBuf> = std::env::args_os()
            .skip(1)
            .map(PathBuf::from)
            .filter(|path| path.exists())
            .collect();
        if !files.is_empty() {
            app.add_paths(files);
        }
        app
    }

    /// State used by tests and by nothing else; no arguments are read.
    pub fn empty() -> Self {
        let settings = Settings::load();
        hf::set_endpoint(&settings.hf_endpoint);
        let mut app = Self {
            tab: Tab::Convert,
            jobs: Vec::new(),
            next_id: 1,
            selected_job: None,
            settings,
            rx: None,
            worker: None,
            cancel: None,
            batch: None,
            last_output_dir: None,
            benchmark: None,
            logs: Vec::new(),
            show_log: false,
            log_filter: String::new(),
            log_levels: [true, true, true, true],
            show_help: false,
            show_settings: false,
            confirm_close: false,
            inspector_path: None,
            inspector: None,
            inspector_error: None,
            inspector_input: String::new(),
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
            hf_details_error: None,
            hf_file_filter: String::new(),
            discover_loaded: false,
            downloads: Vec::new(),
            next_download_id: 1,
            dl_rx: None,
            dl_worker: None,
            dl_cancel: Arc::new(AtomicBool::new(false)),
            confirm_requantize: None,
            deploy: None,
            maximized: false,
            window: None,
            hardware: rustblock_core::detect_hardware(),
            logo: theme::logo_handle(),
            wordmark: theme::wordmark_handle(),
            wordmark_aspect: theme::wordmark_aspect(),
        };
        app.log(
            LogLevel::Info,
            "Ready. Add a GGUF file, search Hugging Face, or open the GGUF inspector.".to_string(),
        );
        app
    }

    // ----- Queries ---------------------------------------------------------

    pub fn is_running(&self) -> bool {
        self.rx.is_some()
    }

    pub fn hf_busy(&self) -> bool {
        self.hf_rx.is_some()
    }

    pub fn downloads_busy(&self) -> bool {
        self.dl_worker.is_some()
    }

    pub fn active_downloads(&self) -> usize {
        self.downloads
            .iter()
            .filter(|download| !download.state.is_finished())
            .count()
    }

    pub fn busy(&self) -> bool {
        self.is_running()
            || self.hf_busy()
            || self.downloads_busy()
            || self.benchmark.is_some()
            || self.deploy.as_ref().is_some_and(|deploy| deploy.busy)
    }

    pub fn job_name(&self, id: u64) -> String {
        self.jobs
            .iter()
            .find(|job| job.id == id)
            .map(Job::file_name)
            .unwrap_or_else(|| format!("#{id}"))
    }

    pub fn queue_counts(&self) -> (usize, usize, usize, usize) {
        let ready = self.selected_jobs();
        let running = self
            .jobs
            .iter()
            .filter(|job| matches!(job.status, JobStatus::Running(_)))
            .count();
        let done = self
            .jobs
            .iter()
            .filter(|job| matches!(job.status, JobStatus::Done { .. }))
            .count();
        let failed = self
            .jobs
            .iter()
            .filter(|job| matches!(job.status, JobStatus::Failed(_)))
            .count();
        (ready, running, done, failed)
    }

    pub fn selected_jobs(&self) -> usize {
        let target = self.settings.target;
        self.jobs
            .iter()
            .filter(|job| job.analysis.can_target(target))
            .count()
    }

    pub fn done_job_output(&self) -> Option<PathBuf> {
        self.jobs.iter().rev().find_map(Job::done_output)
    }

    pub fn current_model(&self) -> Option<HfModel> {
        let HfState::Results(models) = &self.hf_state else {
            return None;
        };
        let repo = self.selected_repo.as_deref()?;
        models.iter().find(|model| model.id == repo).cloned()
    }

    pub fn visible_logs(&self) -> Vec<&LogEntry> {
        self.logs
            .iter()
            .filter(|entry| log_entry_visible(entry, &self.log_filter, &self.log_levels))
            .collect()
    }

    fn log(&mut self, level: LogLevel, text: String) {
        self.logs.push(LogEntry { level, text });
        if self.logs.len() > 500 {
            let excess = self.logs.len() - 500;
            self.logs.drain(0..excess);
        }
    }

    // ----- Update ----------------------------------------------------------

    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::Tick => {
                self.poll_worker();
                self.poll_hf();
                self.poll_downloads();
                self.poll_benchmark();
                self.poll_deploy();
            }
            Message::WindowOpened(id) => {
                if self.window.is_none() {
                    self.window = Some(id);
                }
            }
            Message::DragWindow => {
                if let Some(id) = self.window {
                    return iced::window::drag(id);
                }
            }
            Message::Resize(direction) => {
                if let Some(id) = self.window {
                    return iced::window::drag_resize(id, direction);
                }
            }
            Message::ToggleMaximize => {
                self.maximized = !self.maximized;
                if let Some(id) = self.window {
                    return iced::window::toggle_maximize(id);
                }
            }
            Message::Minimize => {
                if let Some(id) = self.window {
                    return iced::window::minimize(id, true);
                }
            }
            Message::CloseWindow => {
                if self.busy() {
                    self.confirm_close = true;
                } else if let Some(id) = self.window {
                    return iced::window::close(id);
                }
            }
            Message::ConfirmClose(quit) => {
                self.confirm_close = false;
                if quit {
                    if let Some(id) = self.window {
                        return iced::window::close(id);
                    }
                }
            }
            Message::TabSelected(tab) => {
                self.tab = tab;
                if tab == Tab::Discover && !self.discover_loaded {
                    self.discover_loaded = true;
                    self.start_search();
                }
            }
            Message::ToggleLog => self.show_log = !self.show_log,
            Message::ToggleHelp => self.show_help = !self.show_help,
            Message::ToggleSettings => self.show_settings = !self.show_settings,
            Message::SetSource(source) => {
                if self.settings.source != source {
                    self.settings.source = source;
                    self.settings.save();
                    self.hf_state = HfState::Idle;
                    self.selected_repo = None;
                    self.hf_files.clear();
                    self.hf_details_error = None;
                    self.pending_details = None;
                    self.discover_loaded = false;
                }
            }
            Message::SetAutoAdd(value) => {
                self.settings.auto_add_downloads = value;
                self.settings.save();
            }
            Message::SetConfirmRequantize(value) => {
                self.settings.confirm_requantize = value;
                self.settings.save();
            }
            Message::HfEndpointChanged(value) => {
                self.settings.hf_endpoint = value;
                self.settings.save();
                hf::set_endpoint(&self.settings.hf_endpoint);
            }
            Message::PickLmStudioDir => {
                if let Some(path) = pick_folder("LM Studio models") {
                    self.settings.lmstudio_dir = path;
                    self.settings.save();
                }
            }
            Message::PickJanDir => {
                if let Some(path) = pick_folder("Jan data folder") {
                    self.settings.jan_dir = path;
                    self.settings.save();
                }
            }
            Message::PickPrismDir => {
                if let Some(path) = pick_folder("Prism models") {
                    self.settings.prism_dir = path;
                    self.settings.save();
                }
            }
            Message::ResetSettings => {
                self.settings = Settings::default();
                self.settings.save();
                hf::set_endpoint(&self.settings.hf_endpoint);
            }
            Message::Browse => self.browse(),
            Message::FilesDropped(path) => self.add_paths(vec![path]),
            Message::ClearQueue => {
                if !self.is_running() {
                    self.jobs.clear();
                    self.selected_job = None;
                }
            }
            Message::SelectJob(id) => {
                self.selected_job = if self.selected_job == Some(id) {
                    None
                } else {
                    Some(id)
                };
            }
            Message::RemoveJob(id) => {
                if !self.is_running() {
                    self.jobs.retain(|job| job.id != id);
                    if self.selected_job == Some(id) {
                        self.selected_job = None;
                    }
                }
            }
            Message::StartConvert => self.start(WorkKind::Convert),
            Message::StartVerify => self.start(WorkKind::Verify),
            Message::CancelBatch => {
                if let Some(cancel) = &self.cancel {
                    cancel.store(true, Ordering::Relaxed);
                }
            }
            Message::ConfirmRequantize(proceed) => {
                let kind = self.confirm_requantize.take();
                if proceed {
                    if let Some(kind) = kind {
                        self.launch(kind, true);
                    }
                }
            }
            Message::SetTarget(target) => {
                if !self.is_running() {
                    self.settings.target = target;
                    self.sync_job_targets();
                }
            }
            Message::SetVerify(mode) => {
                if !self.is_running() {
                    self.settings.verify = mode;
                }
            }
            Message::SetOverwrite(value) => {
                if !self.is_running() {
                    self.settings.overwrite = value;
                }
            }
            Message::PickOutputDir => {
                if !self.is_running() {
                    if let Some(picked) = pick_folder("Output folder") {
                        self.last_output_dir = Some(picked.clone());
                        self.settings.output_mode = OutputMode::Custom(picked);
                    }
                }
            }
            Message::UseSourceDir => {
                if !self.is_running() {
                    self.settings.output_mode = OutputMode::NextToInput;
                }
            }
            Message::SuggestTarget => self.suggest_target(),
            Message::Benchmark => self.start_benchmark(),
            Message::InspectorOpen => self.inspector_open(),
            Message::InspectorRefresh => self.inspector_refresh(),
            Message::InspectorPathChanged(value) => self.inspector_input = value,
            Message::InspectorLoad => {
                let text = self.inspector_input.trim().to_string();
                if !text.is_empty() {
                    self.inspector_path = Some(PathBuf::from(text));
                    self.inspector_refresh();
                }
            }
            Message::InspectPath(path) => {
                self.inspector_path = Some(path);
                self.tab = Tab::Inspector;
                self.inspector_refresh();
            }
            Message::Deploy(path) => self.deploy = Some(DeployUi::new(path)),
            Message::DeployTo(runtime) => self.deploy_to(runtime),
            Message::DeployCancel => {
                if let Some(deploy) = &self.deploy {
                    deploy.cancel.store(true, Ordering::Relaxed);
                }
            }
            Message::DeployClose => {
                if let Some(mut deploy) = self.deploy.take() {
                    deploy.cancel.store(true, Ordering::Relaxed);
                    if let Some(worker) = deploy.worker.take() {
                        let _ = worker.join();
                    }
                }
            }
            Message::OpenPath(path) => open_folder(&path),
            Message::SearchChanged(query) => self.hf_query = query,
            Message::SearchSubmit => self.start_search(),
            Message::SetSort(sort) => {
                if self.hf_sort != sort {
                    self.hf_sort = sort;
                    self.start_search();
                }
            }
            Message::Trending => {
                self.hf_query.clear();
                self.start_search();
            }
            Message::OpenRepo(repo) => self.open_repo(repo),
            Message::AddCached(path) => {
                if path.exists() {
                    self.add_paths(vec![path]);
                } else {
                    self.log(
                        LogLevel::Warn,
                        format!("{} is no longer on disk.", path.display()),
                    );
                }
            }
            Message::Download(file) => {
                let Some(repo) = self.selected_repo.clone() else {
                    return Task::none();
                };
                self.start_download(repo, file);
            }
            Message::CancelDownload(id) => {
                if let Some(download) = self.downloads.iter_mut().find(|download| download.id == id)
                {
                    match download.state {
                        DownloadState::Running => {
                            self.dl_cancel.store(true, Ordering::Relaxed);
                        }
                        DownloadState::Queued => {
                            download.state = DownloadState::Cancelled;
                            download.error = Some("Cancelled before starting.".to_string());
                        }
                        _ => {}
                    }
                }
            }
            Message::ClearFinishedDownloads => {
                self.downloads
                    .retain(|download| !download.state.is_finished());
            }
            Message::FileFilterChanged(value) => self.hf_file_filter = value,
            Message::PickDownloadDir => {
                if let Some(picked) = pick_folder("Output folder") {
                    self.settings.download_dir = picked;
                }
            }
            Message::OpenUrl(url) => open_url(&url),
            Message::DismissDownload(id) => {
                if !self
                    .downloads
                    .iter()
                    .any(|download| download.id == id && download.state == DownloadState::Running)
                {
                    self.downloads.retain(|download| download.id != id);
                }
            }
            Message::ClearPartials => {
                let removed =
                    crate::model::clear_partial_files(&self.settings.download_dir).unwrap_or(0);
                self.log(
                    LogLevel::Info,
                    format!("Removed {removed} partial download(s)."),
                );
            }
            Message::LogFilterChanged(value) => self.log_filter = value,
            Message::ToggleLevel(index) => {
                if let Some(level) = self.log_levels.get_mut(index) {
                    *level = !*level;
                }
            }
            Message::ClearLog => self.logs.clear(),
        }
        Task::none()
    }

    pub fn view<'a>(&'a self) -> iced::Element<'a, Message> {
        crate::view::view(self)
    }

    pub fn subscription(&self) -> Subscription<Message> {
        let mut subscriptions = vec![
            iced::event::listen_with(listen),
            iced::window::open_events().map(Message::WindowOpened),
        ];
        if self.busy() {
            subscriptions
                .push(iced::time::every(Duration::from_millis(120)).map(|_| Message::Tick));
        }
        Subscription::batch(subscriptions)
    }

    // ----- Queue -----------------------------------------------------------

    fn sync_job_targets(&mut self) {
        let target = self.settings.target;
        for job in &mut self.jobs {
            if job.analysis.can_target(target) {
                job.target = target;
            }
        }
    }

    fn add_paths(&mut self, paths: Vec<PathBuf>) {
        for path in paths {
            let is_gguf = path
                .extension()
                .map(|extension| extension.to_string_lossy().eq_ignore_ascii_case("gguf"))
                .unwrap_or(false);
            if !is_gguf {
                self.log(
                    LogLevel::Warn,
                    format!("{} - only .gguf files are supported", path.display()),
                );
                continue;
            }
            if self.jobs.iter().any(|job| same_file(&job.input, &path)) {
                self.log(
                    LogLevel::Info,
                    format!("Already in the queue: {}", path.display()),
                );
                continue;
            }
            let analysis = analyze(&path, self.hardware);
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
        if let Some(files) = file_dialog()
            .add_filter("GGUF model", &["gguf"])
            .pick_files()
        {
            self.add_paths(files);
        }
    }

    // ----- Workers ---------------------------------------------------------

    fn poll_worker(&mut self) {
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
            self.handle_event(event);
        }
        if disconnected && self.rx.is_some() {
            self.log(
                LogLevel::Error,
                "The conversion worker stopped unexpectedly.".to_string(),
            );
            self.finish_batch();
        }
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

    fn handle_event(&mut self, event: Event) {
        match event {
            Event::Started { id } => {
                if let Some(job) = self.jobs.iter_mut().find(|job| job.id == id) {
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
                if let Some(job) = self.jobs.iter_mut().find(|job| job.id == id) {
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
                if let Some(job) = self.jobs.iter_mut().find(|job| job.id == id) {
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
                if let Some(job) = self.jobs.iter_mut().find(|job| job.id == id) {
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
                if let Some(job) = self.jobs.iter_mut().find(|job| job.id == id) {
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
                    .is_some_and(|cancel| cancel.load(Ordering::Relaxed));
                let kind = self
                    .batch
                    .as_ref()
                    .map(|batch| batch.kind)
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
            active: items.first().map(|item| item.id),
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

    // ----- Benchmark / inspector ------------------------------------------

    fn start_benchmark(&mut self) {
        if self.benchmark.is_some() {
            return;
        }
        let Some(model) = self.done_job_output() else {
            return;
        };
        let bench = spawn_benchmark(model.clone());
        self.log(
            LogLevel::Info,
            format!("Benchmark: llama-bench -m {}", model.display()),
        );
        self.benchmark = Some(bench);
    }

    fn poll_benchmark(&mut self) {
        let mut done = None;
        if let Some(bench) = &self.benchmark {
            while let Ok(event) = bench.receiver.try_recv() {
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
            if let Some(bench) = self.benchmark.take() {
                let _ = bench.handle.join();
            }
        }
    }

    fn inspector_open(&mut self) {
        if let Some(path) = file_dialog()
            .add_filter("GGUF model", &["gguf"])
            .pick_file()
        {
            self.inspector_path = Some(path);
            self.inspector_refresh();
        }
    }

    fn inspector_refresh(&mut self) {
        self.inspector = None;
        self.inspector_error = None;
        let Some(path) = self.inspector_path.as_deref() else {
            return;
        };
        self.inspector_input = path.display().to_string();
        match InspectorData::open(path) {
            Some(data) => self.inspector = Some(data),
            None => {
                self.inspector_error = Some(format!(
                    "Cannot read {}: not a valid GGUF, or a tensor type this build cannot decode.",
                    path.display()
                ));
            }
        }
    }

    fn deploy_to(&mut self, runtime: Runtime) {
        let Some(deploy) = &mut self.deploy else {
            return;
        };
        if deploy.busy {
            return;
        }
        let model = deploy.model.clone();
        let settings = self.settings.clone();
        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        deploy.runtime = Some(runtime);
        deploy.busy = true;
        deploy.done = 0;
        deploy.total = 0;
        deploy.message = None;
        deploy.folder = None;
        deploy.rx = Some(rx);
        deploy.cancel = cancel.clone();
        deploy.worker = Some(deploy::spawn(runtime, model.clone(), settings, tx, cancel));
        self.log(
            LogLevel::Info,
            format!("Deploying {} to {}.", file_name_of(&model), runtime.label()),
        );
    }

    fn poll_deploy(&mut self) {
        let mut events = Vec::new();
        let mut disconnected = false;
        if let Some(deploy) = &self.deploy {
            if let Some(rx) = &deploy.rx {
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
        }
        for event in events {
            if let Some(deploy) = &mut self.deploy {
                match event {
                    DeployEvent::Progress { done, total } => {
                        deploy.done = done;
                        deploy.total = total;
                    }
                    DeployEvent::Done { folder, message } => {
                        deploy.busy = false;
                        deploy.message = Some(message.clone());
                        deploy.folder = Some(folder);
                        self.log(LogLevel::Ok, message);
                    }
                    DeployEvent::Failed(error) => {
                        deploy.busy = false;
                        deploy.message = Some(error.clone());
                        self.log(LogLevel::Error, error);
                    }
                }
            }
        }
        if disconnected {
            if let Some(deploy) = &mut self.deploy {
                deploy.rx = None;
                deploy.busy = false;
                if let Some(worker) = deploy.worker.take() {
                    let _ = worker.join();
                }
            }
        }
    }

    fn suggest_target(&mut self) {
        let hardware = self.hardware;
        if hardware.vram_bytes.is_none() {
            self.log(
                LogLevel::Warn,
                "No NVIDIA GPU detected; the suggestion uses system RAM. Open Help (F1) for the fit estimate."
                    .to_string(),
            );
        }
        let largest = self
            .jobs
            .iter()
            .max_by_key(|job| job.analysis.file_len)
            .cloned();
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
                        self.sync_job_targets();
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
        self.sync_job_targets();
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

    // ----- Hugging Face ----------------------------------------------------

    fn start_search(&mut self) {
        if self.hf_busy() {
            return;
        }
        let query = self.hf_query.trim().to_string();
        if self.settings.source == Source::ModelScope {
            let Some(repo) = hf::parse_modelscope(&query) else {
                let error = "Paste a ModelScope model link (modelscope.cn/models/owner/name) or an owner/name id."
                    .to_string();
                self.hf_state = HfState::Error(error.clone());
                self.log(LogLevel::Error, error);
                return;
            };
            self.open_link_ms(repo);
            return;
        }
        if query.contains("modelscope.cn") || query.contains("modelscope.ai") {
            let error = "That is a ModelScope link. Switch Source to ModelScope first.".to_string();
            self.hf_state = HfState::Error(error.clone());
            self.log(LogLevel::Error, error);
            return;
        }
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
        self.hf_worker = Some(thread::spawn(move || {
            let _ = tx.send(HfEvent::SearchStarted);
            match hf::search(&query, sort) {
                Ok(results) => {
                    let _ = tx.send(HfEvent::SearchResults(results));
                }
                Err(error) => {
                    let _ = tx.send(HfEvent::Error(error));
                }
            }
        }));
    }

    fn open_link(&mut self, link: hf::HfLink) {
        if self.hf_busy() {
            return;
        }
        let repo = link.repo;
        let (tx, rx) = mpsc::channel();
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
        self.hf_worker = Some(thread::spawn(move || {
            let _ = tx.send(HfEvent::SearchStarted);
            match hf::fetch_model(&repo) {
                Ok(model) => {
                    let _ = tx.send(HfEvent::SearchResults(vec![model]));
                }
                Err(error) => {
                    let _ = tx.send(HfEvent::Error(error));
                }
            }
        }));
    }

    fn open_link_ms(&mut self, repo: String) {
        if self.hf_busy() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        self.hf_query = repo.clone();
        self.hf_state = HfState::Searching;
        self.selected_repo = Some(repo.clone());
        self.selected_sha = "master".to_string();
        self.hf_details_loading = false;
        self.hf_details_error = None;
        self.pending_details = None;
        self.hf_files.clear();
        self.hf_file_filter.clear();
        self.hf_rx = Some(rx);
        self.hf_cancel = Arc::new(AtomicBool::new(false));
        self.log(LogLevel::Info, format!("Opening ModelScope model {repo}."));
        self.hf_worker = Some(thread::spawn(move || {
            let _ = tx.send(HfEvent::SearchStarted);
            let model = hf::fetch_model_ms(&repo);
            let _ = tx.send(HfEvent::SearchResults(vec![model]));
        }));
    }

    /// Select a repository from the results and load its file list.
    fn open_repo(&mut self, repo: String) {
        if self.hf_busy() {
            return;
        }
        if self.settings.source == Source::ModelScope {
            self.start_details(repo, "master".to_string());
            return;
        }
        let revision = match &self.hf_state {
            HfState::Results(models) => models
                .iter()
                .find(|model| model.id == repo)
                .map(|model| model.sha.clone())
                .unwrap_or_else(|| "main".to_string()),
            _ => "main".to_string(),
        };
        self.start_details(repo, revision);
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
        self.hf_rx = Some(rx);
        self.hf_cancel = Arc::new(AtomicBool::new(false));
        let source = self.settings.source;
        self.hf_worker = Some(thread::spawn(move || {
            let _ = tx.send(HfEvent::DetailsStarted(repo.clone()));
            let result = match source {
                Source::HuggingFace => hf::list_files(&repo, &revision),
                Source::ModelScope => hf::list_files_ms(&repo),
            };
            match result {
                Ok(files) => {
                    let _ = tx.send(HfEvent::DetailsLoaded { repo, files });
                }
                Err(error) => {
                    let _ = tx.send(HfEvent::DetailsError { repo, error });
                }
            }
        }));
    }

    fn start_download(&mut self, repo: String, file: HfFile) {
        self.begin_download(repo, file);
    }

    fn begin_download(&mut self, repo: String, file: HfFile) {
        let projector = file.projector;
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
        let source = self.settings.source;
        let revision = match source {
            Source::HuggingFace => self.selected_sha.clone(),
            Source::ModelScope => "master".to_string(),
        };
        let id = self.next_download_id;
        self.next_download_id += 1;
        self.downloads.push(DownloadUi {
            id,
            source,
            repo: repo.clone(),
            file: file.path.clone(),
            revision,
            expected_sha: file.sha256.clone(),
            done: 0,
            total: file.size,
            speed: 0.0,
            quantized: !projector && looks_quantized_name(&file.path),
            projector,
            path: None,
            error: None,
            state: DownloadState::Queued,
        });
        self.log(
            LogLevel::Info,
            format!(
                "Queued {} from {} ({} in the download list).",
                file.path,
                source.label(),
                self.downloads.len()
            ),
        );
        self.start_next_download();
    }

    fn start_next_download(&mut self) {
        if self.dl_worker.is_some() {
            return;
        }
        let Some(index) = self
            .downloads
            .iter()
            .position(|download| download.state == DownloadState::Queued)
        else {
            return;
        };
        let dest = self.settings.download_dir.clone();
        let (source, repo, file, revision, sha) = {
            let item = &mut self.downloads[index];
            item.state = DownloadState::Running;
            item.done = 0;
            item.speed = 0.0;
            item.error = None;
            (
                item.source,
                item.repo.clone(),
                item.file.clone(),
                item.revision.clone(),
                item.expected_sha.clone(),
            )
        };
        let (tx, rx) = mpsc::channel();
        self.dl_rx = Some(rx);
        self.dl_cancel = Arc::new(AtomicBool::new(false));
        let cancel = self.dl_cancel.clone();
        let display = format!("{repo}/{file}");
        self.log(LogLevel::Info, format!("Downloading {display}."));
        self.dl_worker = Some(thread::spawn(move || {
            let result = hf::download_from(
                source,
                &repo,
                &revision,
                &file,
                &dest,
                sha.as_deref(),
                &cancel,
                &tx,
            );
            if let Err(error) = result {
                if error != "download cancelled" {
                    let _ = tx.send(HfEvent::DownloadFailed { repo, file, error });
                }
            }
        }));
    }

    fn active_download_mut(&mut self) -> Option<&mut DownloadUi> {
        self.downloads
            .iter_mut()
            .find(|download| download.state == DownloadState::Running)
    }

    fn poll_downloads(&mut self) {
        let mut events = Vec::new();
        let mut disconnected = false;
        if let Some(rx) = &self.dl_rx {
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
                HfEvent::DownloadStarted { total, .. } => {
                    if let Some(download) = self.active_download_mut() {
                        download.total = total;
                    }
                }
                HfEvent::DownloadProgress { done, total, speed } => {
                    if let Some(download) = self.active_download_mut() {
                        download.done = done;
                        download.total = total;
                        download.speed = if download.speed <= 0.0 {
                            speed
                        } else {
                            download.speed * 0.75 + speed * 0.25
                        };
                    }
                }
                HfEvent::Downloaded {
                    repo,
                    file,
                    path,
                    projector,
                } => {
                    let quantized = self
                        .active_download_mut()
                        .map(|download| {
                            download.path = Some(path.clone());
                            download.done = download.total;
                            download.state = DownloadState::Done;
                            download.quantized
                        })
                        .unwrap_or(false);
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
                        if quantized {
                            self.log(
                                LogLevel::Warn,
                                format!(
                                    "{file} is already quantized. Converting it again cannot recover the original precision; use a virgin F16/BF16 model if available."
                                ),
                            );
                        }
                        if self.settings.auto_add_downloads {
                            self.add_paths(vec![path]);
                        } else {
                            self.log(
                                LogLevel::Info,
                                "Auto-add is off: add the model from the download list when you are ready."
                                    .to_string(),
                            );
                        }
                    }
                }
                HfEvent::DownloadFailed { repo, file, error } => {
                    if let Some(download) = self.active_download_mut() {
                        download.state = DownloadState::Failed;
                        download.error = Some(error.clone());
                    }
                    self.log(
                        LogLevel::Error,
                        format!("Download failed for {repo}/{file}: {error}"),
                    );
                }
                HfEvent::DownloadCancelled => {
                    if let Some(download) = self.active_download_mut() {
                        download.state = DownloadState::Cancelled;
                        download.error = Some("Download cancelled.".to_string());
                    }
                    self.log(LogLevel::Warn, "Download cancelled.".to_string());
                }
                _ => {}
            }
        }
        if disconnected {
            self.dl_rx = None;
            if let Some(handle) = self.dl_worker.take() {
                let _ = handle.join();
            }
            self.start_next_download();
        }
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
                        format!(
                            "{}: {} GGUF repositories found.",
                            self.settings.source.label(),
                            results.len()
                        ),
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
                _ => {}
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
}

// ----- Small helpers ------------------------------------------------------

/// Native dialogs must be owned by the app window. Without a parent the
/// dialog can open behind the frameless window and look like a freeze.
#[cfg(windows)]
fn file_dialog() -> rfd::FileDialog {
    let dialog = rfd::FileDialog::new();
    match foreground_window() {
        Some(parent) => dialog.set_parent(&parent),
        None => dialog,
    }
}

#[cfg(not(windows))]
fn file_dialog() -> rfd::FileDialog {
    rfd::FileDialog::new()
}

#[cfg(windows)]
fn pick_folder(_title: &str) -> Option<PathBuf> {
    file_dialog().pick_folder()
}

#[cfg(not(windows))]
fn pick_folder(_title: &str) -> Option<PathBuf> {
    file_dialog().pick_folder()
}

#[cfg(windows)]
fn foreground_window() -> Option<ForegroundWindow> {
    #[link(name = "user32")]
    extern "system" {
        fn GetForegroundWindow() -> isize;
    }
    let handle = unsafe { GetForegroundWindow() };
    (handle != 0).then_some(ForegroundWindow(handle))
}

#[cfg(windows)]
struct ForegroundWindow(isize);

#[cfg(windows)]
impl raw_window_handle::HasWindowHandle for ForegroundWindow {
    fn window_handle(
        &self,
    ) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
        let handle = raw_window_handle::Win32WindowHandle::new(
            std::num::NonZeroIsize::new(self.0)
                .ok_or(raw_window_handle::HandleError::Unavailable)?,
        );
        Ok(unsafe {
            raw_window_handle::WindowHandle::borrow_raw(raw_window_handle::RawWindowHandle::Win32(
                handle,
            ))
        })
    }
}

#[cfg(windows)]
impl raw_window_handle::HasDisplayHandle for ForegroundWindow {
    fn display_handle(
        &self,
    ) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        Ok(unsafe {
            raw_window_handle::DisplayHandle::borrow_raw(
                raw_window_handle::RawDisplayHandle::Windows(
                    raw_window_handle::WindowsDisplayHandle::new(),
                ),
            )
        })
    }
}

pub fn open_folder(path: &Path) {
    let target = if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| path.to_path_buf())
    };
    #[cfg(target_os = "windows")]
    {
        let _ = Command::new("explorer").arg(target).spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = Command::new("open").arg(target).spawn();
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let _ = Command::new("xdg-open").arg(target).spawn();
    }
    #[cfg(not(any(windows, unix)))]
    {
        let _ = target;
    }
}

pub fn open_url(url: &str) {
    #[cfg(target_os = "windows")]
    {
        let _ = Command::new("cmd").args(["/C", "start", "", url]).spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = Command::new("open").arg(url).spawn();
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let _ = Command::new("xdg-open").arg(url).spawn();
    }
    #[cfg(not(any(windows, unix)))]
    {
        let _ = url;
    }
}

pub fn same_file(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => path_key(a) == path_key(b),
    }
}

pub fn format_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

pub fn short_date(date: &str) -> String {
    if date.len() >= 10 {
        date[..10].to_string()
    } else if date.is_empty() {
        "unknown".to_string()
    } else {
        date.to_string()
    }
}

/// Same constants as the egui build so the estimate keeps its meaning.
pub const OVERHEAD_BYTES: u64 = 512 * 1024 * 1024;

/// Hardware detection is slow on Windows (PowerShell and `nvidia-smi`), so the
/// Discover list reuses one detection per process.
pub fn detected_hardware() -> Hardware {
    static HARDWARE: std::sync::OnceLock<Hardware> = std::sync::OnceLock::new();
    *HARDWARE.get_or_init(rustblock_core::detect_hardware)
}

/// Fit verdict available before a download. Only the published weight size is
/// known, so the KV cache is excluded and the tooltip says so instead of
/// pretending the number is a full estimate.
pub fn size_fit_badge(weights: u64, hardware: Hardware) -> (String, iced::Color, String) {
    let total = weights.saturating_add(OVERHEAD_BYTES);
    let hint = format!(
        "Download estimate: {} weights + {} runtime overhead. The KV cache needs the model header, so it is not included before download.",
        human_bytes(weights),
        human_bytes(OVERHEAD_BYTES)
    );
    let fits_ram = hardware
        .ram_bytes
        .map(|available| rustblock_core::hardware_fits(total, available));
    let fits_vram = hardware
        .vram_bytes
        .map(|available| rustblock_core::hardware_fits(total, available));
    let (label, color) = if fits_vram == Some(true) {
        ("Fits VRAM".to_string(), crate::theme::OK)
    } else if fits_ram == Some(true) {
        ("Fits RAM".to_string(), crate::theme::OK)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_level_filter_matches_only_selected_levels() {
        let entry = LogEntry {
            level: LogLevel::Warn,
            text: "Output already exists".to_string(),
        };
        assert!(log_entry_visible(&entry, "exists", &[true; 4]));
        assert!(!log_entry_visible(&entry, "missing", &[true; 4]));
        assert!(!log_entry_visible(&entry, "", &[true, true, false, true]));
        assert!(log_entry_visible(&entry, "EXISTS", &[true; 4]));
    }

    #[test]
    fn download_badge_uses_weights_only_and_says_so() {
        let gib = 1024 * 1024 * 1024;
        let hardware = |ram, vram| Hardware {
            ram_bytes: ram,
            vram_bytes: vram,
        };
        let (label, _, hint) = size_fit_badge(8 * gib, hardware(Some(32 * gib), Some(12 * gib)));
        assert_eq!(label, "Fits VRAM");
        assert!(hint.contains("not included"), "{hint}");
        assert_eq!(
            size_fit_badge(116 * gib, hardware(Some(32 * gib), Some(12 * gib))).0,
            "No fit"
        );
        assert_eq!(
            size_fit_badge(8 * gib, hardware(None, None)).0,
            "Fit unknown"
        );
    }

    #[test]
    fn boot_state_starts_with_an_empty_queue() {
        let app = RustBlock::empty();
        assert!(app.jobs.is_empty());
        assert_eq!(app.queue_counts(), (0, 0, 0, 0));
        assert!(!app.busy());
        assert_eq!(app.tab, Tab::Convert);
    }

    #[test]
    fn tab_switch_loads_discover_once() {
        let mut app = RustBlock::empty();
        app.tab = Tab::Convert;
        // Discover selection would start a network search; only the flag is
        // checked here so the test never touches the Hub.
        assert!(!app.discover_loaded);
    }
}
