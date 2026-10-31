//! Application state and the whole egui interface.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use eframe::egui::{
    self, Align, Align2, Color32, CornerRadius, FontId, Id, LayerId, Layout, Margin, Order,
    ProgressBar, RichText, ScrollArea, Sense, Stroke, StrokeKind, Ui, Vec2,
};
use ternary_reblock_core::{Phase, VerifyMode};

use crate::i18n::{self, fill, T};
use crate::model::{
    analyze, human_bytes, human_duration, human_speed, log_clock, output_path, path_key, shorten,
    FileKind, Job, JobStatus, Lang, OutputMode, RunState, Settings, VerifyOutcome,
};
use crate::theme::{self, ACCENT, BG, BORDER, ERROR, INFO, INK, MUTED, PANEL, PANEL_2, TEXT, WARN};
use crate::worker::{self, Event, WorkItem, WorkKind};

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

pub struct ReblockApp {
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
}

impl ReblockApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        theme::install(&cc.egui_ctx);

        let mut app = Self {
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
        };
        app.log(
            LogLevel::Info,
            i18n::t(app.settings.lang).log_ready.to_string(),
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
        let t = i18n::t(self.settings.lang);
        for path in paths {
            let is_gguf = path
                .extension()
                .map(|e| e.to_string_lossy().eq_ignore_ascii_case("gguf"))
                .unwrap_or(false);
            if !is_gguf {
                self.log(
                    LogLevel::Warn,
                    format!("{} - {}", path.display(), t.drop_only_gguf),
                );
                continue;
            }
            if self.jobs.iter().any(|j| same_file(&j.input, &path)) {
                self.log(LogLevel::Info, fill(t.log_duplicate, &[&path.display()]));
                continue;
            }

            let analysis = analyze(&path);
            if let Some(error) = &analysis.error {
                self.log(
                    LogLevel::Error,
                    fill(t.log_analysis_failed, &[&path.display(), error]),
                );
            } else {
                self.log(LogLevel::Info, fill(t.log_added, &[&path.display()]));
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
            let t = i18n::t(self.settings.lang);
            self.log(LogLevel::Error, t.log_worker_lost.to_string());
            self.finish_batch();
        }
    }

    fn handle_event(&mut self, event: Event, ctx: &egui::Context) {
        let t = i18n::t(self.settings.lang);
        match event {
            Event::Started { id, kind } => {
                if let Some(job) = self.jobs.iter_mut().find(|j| j.id == id) {
                    let total = if kind == WorkKind::Verify {
                        job.analysis.file_len
                    } else {
                        job.analysis
                            .estimated_new_size
                            .unwrap_or(job.analysis.file_len)
                    };
                    job.status = JobStatus::Running(RunState::new(total.max(1)));
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
            Event::Done { id, report } => {
                let name = self.job_name(id);
                let old = report.old_file_size;
                let new = report.new_file_size;
                let elapsed = report.elapsed;
                if let Some(job) = self.jobs.iter_mut().find(|j| j.id == id) {
                    job.status = JobStatus::Done(report);
                }
                self.batch_completed();
                self.log(
                    LogLevel::Ok,
                    fill(
                        t.log_done,
                        &[
                            &name,
                            &human_bytes(old),
                            &human_bytes(new),
                            &human_duration(elapsed),
                        ],
                    ),
                );
            }
            Event::Verified {
                id,
                outcome,
                elapsed,
            } => {
                let name = self.job_name(id);
                let detail = outcome_text(t, &outcome);
                if let Some(job) = self.jobs.iter_mut().find(|j| j.id == id) {
                    job.status = JobStatus::Verified { outcome, elapsed };
                }
                self.batch_completed();
                self.log(
                    LogLevel::Ok,
                    format!(
                        "{} - {}",
                        fill(t.log_verified, &[&name, &human_duration(elapsed)]),
                        detail
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
                self.log(LogLevel::Error, fill(t.log_failed, &[&name, &error]));
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
                let (ok, template) = if kind == WorkKind::Convert {
                    (converted, t.log_batch_convert)
                } else {
                    (verified, t.log_batch_verify)
                };
                self.log(
                    if failed == 0 {
                        LogLevel::Ok
                    } else {
                        LogLevel::Warn
                    },
                    fill(template, &[&ok, &failed, &human_duration(elapsed)]),
                );
                if stopped {
                    self.log(LogLevel::Warn, t.log_stopped.to_string());
                }
                ctx.request_repaint();
            }
        }
    }

    fn start(&mut self, kind: WorkKind) {
        if self.is_running() {
            return;
        }
        let t = i18n::t(self.settings.lang);
        let output_mode = self.settings.output_mode.clone();
        let overwrite = self.settings.overwrite;
        let verify_setting = self.settings.verify;

        let mut items = Vec::new();
        let mut used_outputs: HashMap<String, u64> = HashMap::new();
        let mut notes: Vec<(LogLevel, String)> = Vec::new();

        for job in &mut self.jobs {
            let eligible = if kind == WorkKind::Convert {
                job.analysis.can_convert()
            } else {
                job.analysis.can_verify()
            };
            if !eligible {
                continue;
            }

            let output = if kind == WorkKind::Convert {
                output_path(&job.input, &output_mode)
            } else {
                job.input.with_extension("verify.gguf")
            };

            if kind == WorkKind::Convert {
                if path_key(&output) == path_key(&job.input) {
                    job.status = JobStatus::Failed(t.output_exists.to_string());
                    notes.push((
                        LogLevel::Warn,
                        format!("{}: {}", job.file_name(), t.output_exists),
                    ));
                    continue;
                }
                if matches!(&job.status, JobStatus::Done(_)) && output.exists() && !overwrite {
                    continue;
                }
                let key = path_key(&output);
                if let Some(previous) = used_outputs.get(&key) {
                    let text = fill(
                        t.log_output_conflict,
                        &[&job.file_name(), &output.display()],
                    );
                    job.status = JobStatus::Failed(text.clone());
                    notes.push((LogLevel::Warn, format!("{text} (id {previous})")));
                    continue;
                }
                used_outputs.insert(key, job.id);

                if output.exists() && !overwrite {
                    job.status = JobStatus::Failed(t.output_exists.to_string());
                    notes.push((
                        LogLevel::Warn,
                        format!("{}: {}", job.file_name(), t.output_exists),
                    ));
                    continue;
                }
            }

            let total = if kind == WorkKind::Verify {
                job.analysis.file_len
            } else {
                job.analysis
                    .estimated_new_size
                    .unwrap_or(job.analysis.file_len)
            };
            job.status = JobStatus::Running(RunState::new(total.max(1)));

            items.push(WorkItem {
                id: job.id,
                kind,
                input: job.input.clone(),
                output,
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
            self.log(LogLevel::Info, t.log_nothing.to_string());
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
            fill(
                if kind == WorkKind::Convert {
                    t.log_start_convert
                } else {
                    t.log_start_verify
                },
                &[&total],
            ),
        );

        let cancel = Arc::new(AtomicBool::new(false));
        let (rx, handle) = worker::spawn(items, cancel.clone());
        self.rx = Some(rx);
        self.worker = Some(handle);
        self.cancel = Some(cancel);
    }

    fn export_log(&mut self) {
        let t = i18n::t(self.settings.lang);
        if let Some(path) = rfd::FileDialog::new()
            .set_file_name("ternary-reblock-log.txt")
            .add_filter("Text", &["txt", "log"])
            .save_file()
        {
            let text = self.render_log();
            match std::fs::write(&path, text) {
                Ok(()) => self.log(LogLevel::Ok, fill(t.log_saved, &[&path.display()])),
                Err(error) => self.log(LogLevel::Error, fill(t.log_save_failed, &[&error])),
            }
        }
    }

    fn render_log(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "Reblock {}\nOS: {} {}\n\n",
            env!("CARGO_PKG_VERSION"),
            std::env::consts::OS,
            std::env::consts::ARCH
        ));
        for entry in &self.entries {
            out.push_str(&format!("[{}] {}\n", log_clock(entry.at), entry.text));
        }
        out
    }
}

impl eframe::App for ReblockApp {
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
                let t = i18n::t(self.settings.lang);
                self.log(LogLevel::Warn, t.log_close_blocked.to_string());
            }
        }

        self.handle_dropped(ctx);
        self.poll_worker(ctx);

        let start = ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::Enter));
        if start && !self.is_running() {
            let kind = if self.settings.verify_only {
                WorkKind::Verify
            } else {
                WorkKind::Convert
            };
            self.start(kind);
        }

        if self.rx.is_some() {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }

    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        let t = i18n::t(self.settings.lang);
        self.top_bar(ui, t);
        self.bottom_bar(ui, t);
        self.action_bar(ui, t);

        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(BG)
                    .inner_margin(Margin::symmetric(20, 16)),
            )
            .show(ui, |ui| {
                ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        let bar = ui.spacing().scroll.bar_width;
                        ui.set_width((ui.available_width() - bar).max(0.0));
                        self.drop_zone(ui, t);
                        ui.add_space(12.0);
                        self.queue_ui(ui, t);
                        ui.add_space(12.0);
                        self.options_ui(ui, t);
                        if self.batch.is_some() {
                            ui.add_space(12.0);
                            self.progress_ui(ui, t);
                        }
                        ui.add_space(12.0);
                        self.log_ui(ui, t);
                        ui.add_space(8.0);
                    });
            });

        self.drag_overlay(ui.ctx(), t);
        self.about_window(ui.ctx(), t);
    }
}

impl ReblockApp {
    fn top_bar(&mut self, ui: &mut Ui, t: &T) {
        egui::Panel::top("header")
            .frame(
                egui::Frame::new()
                    .fill(PANEL)
                    .inner_margin(Margin::symmetric(20, 12))
                    .stroke(Stroke::new(1.0, BORDER)),
            )
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    theme::paint_logo(ui, 38.0);
                    ui.add_space(2.0);
                    ui.vertical(|ui| {
                        ui.label(RichText::new("Reblock").size(25.0).strong().color(TEXT));
                        ui.label(RichText::new(t.tagline).size(12.0).color(MUTED));
                    });
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui
                            .button(RichText::new("?").size(16.0).strong())
                            .on_hover_text(t.about)
                            .clicked()
                        {
                            self.show_about = true;
                        }
                        if ui
                            .selectable_label(self.settings.lang == Lang::It, "IT")
                            .clicked()
                        {
                            self.settings.lang = Lang::It;
                        }
                        if ui
                            .selectable_label(self.settings.lang == Lang::En, "EN")
                            .clicked()
                        {
                            self.settings.lang = Lang::En;
                        }
                        ui.label(RichText::new(t.language).size(11.0).color(MUTED));
                    });
                });
            });
    }

    fn bottom_bar(&mut self, ui: &mut Ui, t: &T) {
        egui::Panel::bottom("footer")
            .frame(
                egui::Frame::new()
                    .fill(PANEL)
                    .inner_margin(Margin::symmetric(20, 7))
                    .stroke(Stroke::new(1.0, BORDER)),
            )
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(t.footer_lossless).size(11.0).color(ACCENT));
                    ui.label(RichText::new("·").size(11.0).color(MUTED));
                    ui.label(RichText::new(t.footer_offline).size(11.0).color(MUTED));
                    ui.label(RichText::new("·").size(11.0).color(MUTED));
                    ui.label(RichText::new(t.footer_license).size(11.0).color(MUTED));
                    ui.with_layout(Layout::right_to_left(Align::TOP), |ui| {
                        ui.label(
                            RichText::new(format!(
                                "Reblock {} · {}",
                                env!("CARGO_PKG_VERSION"),
                                fill(t.files_n, &[&self.jobs.len()])
                            ))
                            .size(11.0)
                            .color(MUTED),
                        );
                    });
                });
            });
    }

    fn drop_zone(&mut self, ui: &mut Ui, t: &T) {
        let hovering = !ui.ctx().input(|i| i.raw.hovered_files.is_empty());
        let stroke = Stroke::new(1.0, if hovering { ACCENT } else { BORDER });
        egui::Frame::new()
            .fill(PANEL)
            .stroke(stroke)
            .corner_radius(CornerRadius::same(14))
            .inner_margin(Margin::symmetric(18, 14))
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                ui.vertical_centered(|ui| {
                    paint_down_arrow(ui, ACCENT);
                    ui.label(RichText::new(t.drop_title).size(18.0).strong().color(TEXT));
                    ui.label(RichText::new(t.drop_hint).size(12.0).color(MUTED));
                    ui.add_space(8.0);
                    let button = egui::Button::new(RichText::new(t.browse).strong().color(INK))
                        .fill(ACCENT)
                        .corner_radius(CornerRadius::same(8))
                        .min_size(Vec2::new(170.0, 32.0));
                    if ui.add(button).clicked() {
                        self.browse();
                    }
                });
            });
    }

    fn queue_ui(&mut self, ui: &mut Ui, t: &T) {
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(format!("{} ({})", t.queue, self.jobs.len()))
                    .size(16.0)
                    .strong()
                    .color(TEXT),
            );
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let can_clear = !self.is_running();
                if ui
                    .add_enabled(
                        can_clear && !self.jobs.is_empty(),
                        egui::Button::new(t.clear_all),
                    )
                    .clicked()
                {
                    self.jobs.clear();
                }
                let any_finished = self.jobs.iter().any(|j| j.is_finished());
                if ui
                    .add_enabled(
                        can_clear && any_finished,
                        egui::Button::new(t.clear_finished),
                    )
                    .clicked()
                {
                    self.jobs.retain(|j| !j.is_finished());
                }
            });
        });
        ui.add_space(4.0);

        if self.jobs.is_empty() {
            ui.label(RichText::new(t.queue_empty).size(12.0).color(MUTED));
            return;
        }

        let output_mode = self.settings.output_mode.clone();
        let overwrite = self.settings.overwrite;
        let running = self.is_running();
        let mut remove = None;
        let mut open = None;

        for job in &self.jobs {
            let color = status_color(job);
            egui::Frame::new()
                .fill(PANEL)
                .stroke(Stroke::new(1.0, color.gamma_multiply(0.35)))
                .corner_radius(CornerRadius::same(12))
                .inner_margin(Margin::symmetric(14, 11))
                .show(ui, |ui| {
                    ui.set_min_width(ui.available_width());

                    ui.horizontal(|ui| {
                        let (dot, _) = ui.allocate_exact_size(Vec2::splat(10.0), Sense::hover());
                        ui.painter().circle_filled(dot.center(), 4.0, color);
                        ui.label(
                            RichText::new(job.file_name())
                                .size(15.0)
                                .strong()
                                .color(TEXT),
                        );
                        if job.analysis.kind == FileKind::Pq2 {
                            badge(ui, "PQ2_0 -> Q2_0", INFO);
                        }
                        badge(ui, &status_label(job, t), color);
                        if job.analysis.file_len > 0 {
                            ui.label(
                                RichText::new(format!(
                                    "{}: {}",
                                    t.size,
                                    human_bytes(job.analysis.file_len)
                                ))
                                .size(11.0)
                                .color(MUTED),
                            )
                            .on_hover_text(format!(
                                "GGUF v{} · alignment {} B",
                                job.analysis.version, job.analysis.alignment
                            ));
                        }
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            if ui
                                .add_enabled(
                                    !running,
                                    egui::Button::new(RichText::new(t.remove).size(11.0)),
                                )
                                .clicked()
                            {
                                remove = Some(job.id);
                            }
                            if let JobStatus::Done(report) = &job.status {
                                if ui.button(RichText::new(t.open_folder).size(11.0)).clicked() {
                                    open = Some(report.output.clone());
                                }
                            }
                        });
                    });

                    let mut meta = Vec::new();
                    if let Some(parent) = job.input.parent() {
                        meta.push(shorten(&parent.display().to_string(), 70));
                    }
                    meta.push(format!("{} {}", job.analysis.tensors, t.tensors));
                    if job.analysis.pq2 > 0 {
                        meta.push(format!("{} {}", job.analysis.pq2, t.pq2_tensors));
                    }
                    if let Some(estimated) = job.analysis.estimated_new_size {
                        meta.push(format!("{} {}", t.output_estimate, human_bytes(estimated)));
                    }
                    if job.analysis.hadamard {
                        meta.push(format!("{}: OK", t.hadamard));
                    }
                    ui.label(RichText::new(meta.join("  ·  ")).size(10.8).color(MUTED));

                    if job.analysis.can_convert() {
                        let output = output_path(&job.input, &output_mode);
                        let exists = output.exists() && !overwrite;
                        ui.horizontal_wrapped(|ui| {
                            ui.label(
                                RichText::new(format!("{}:", t.output))
                                    .size(10.5)
                                    .color(MUTED),
                            );
                            let response = ui.label(
                                RichText::new(shorten(&output.display().to_string(), 120))
                                    .size(10.5)
                                    .color(if exists { WARN } else { MUTED }),
                            );
                            if exists {
                                response.on_hover_text(t.output_exists);
                            }
                        });
                    } else if matches!(job.analysis.kind, FileKind::Q2 | FileKind::Other) {
                        ui.label(RichText::new(t.no_output_needed).size(10.5).color(MUTED));
                    }

                    match &job.status {
                        JobStatus::Running(state) => {
                            ui.add_space(4.0);
                            let phase = match state.phase {
                                Phase::Converting => t.phase_converting,
                                Phase::Verifying => t.phase_verifying,
                            };
                            ui.add(
                                ProgressBar::new(state.fraction())
                                    .fill(ACCENT)
                                    .animate(true)
                                    .text(format!("{} · {:.0}%", phase, state.fraction() * 100.0)),
                            );
                            ui.horizontal_wrapped(|ui| {
                                ui.label(
                                    RichText::new(format!(
                                        "{}: {}",
                                        t.speed,
                                        human_speed(state.speed)
                                    ))
                                    .size(10.8)
                                    .color(MUTED),
                                );
                                if let Some(eta) = state.eta() {
                                    ui.label(
                                        RichText::new(format!(
                                            "{}: {}",
                                            t.eta,
                                            human_duration(eta)
                                        ))
                                        .size(10.8)
                                        .color(MUTED),
                                    );
                                }
                                if !state.current_tensor.is_empty() {
                                    ui.label(
                                        RichText::new(shorten(&state.current_tensor, 52))
                                            .size(10.8)
                                            .color(MUTED),
                                    );
                                }
                            });
                        }
                        JobStatus::Done(report) => {
                            ui.label(
                                RichText::new(format!(
                                    "{} -> {}  ·  {}",
                                    human_bytes(report.old_file_size),
                                    human_bytes(report.new_file_size),
                                    human_duration(report.elapsed)
                                ))
                                .size(11.0)
                                .color(ACCENT),
                            );
                        }
                        JobStatus::Verified { outcome, elapsed } => {
                            ui.label(
                                RichText::new(format!(
                                    "{}  ·  {}",
                                    outcome_text(t, outcome),
                                    human_duration(*elapsed)
                                ))
                                .size(11.0)
                                .color(ACCENT),
                            );
                        }
                        JobStatus::Failed(error) => {
                            ui.label(RichText::new(error).size(11.0).color(ERROR));
                        }
                        JobStatus::Ready => {}
                    }
                });
            ui.add_space(6.0);
        }

        if let Some(id) = remove {
            self.jobs.retain(|j| j.id != id);
        }
        if let Some(path) = open {
            open_folder(&path);
        }
    }

    fn options_ui(&mut self, ui: &mut Ui, t: &T) {
        let running = self.is_running();
        section_frame(ui, |ui| {
            ui.label(RichText::new(t.options).size(16.0).strong().color(TEXT));
            ui.add_space(2.0);

            ui.add_enabled_ui(!running, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(t.verification).size(12.5).color(MUTED));
                    let selected = match self.settings.verify {
                        VerifyMode::None => t.verify_none,
                        VerifyMode::Structure => t.verify_structure,
                        VerifyMode::Full => t.verify_full,
                    };
                    egui::ComboBox::from_id_salt("verify_mode")
                        .width(230.0)
                        .selected_text(selected)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.settings.verify,
                                VerifyMode::None,
                                t.verify_none,
                            );
                            ui.selectable_value(
                                &mut self.settings.verify,
                                VerifyMode::Structure,
                                t.verify_structure,
                            );
                            ui.selectable_value(
                                &mut self.settings.verify,
                                VerifyMode::Full,
                                t.verify_full,
                            );
                        });
                });
                ui.label(
                    RichText::new(match self.settings.verify {
                        VerifyMode::None => t.verify_help_none,
                        VerifyMode::Structure => t.verify_help_structure,
                        VerifyMode::Full => t.verify_help_full,
                    })
                    .size(10.8)
                    .color(MUTED),
                );

                ui.add_space(2.0);
                ui.checkbox(&mut self.settings.overwrite, t.overwrite);
                ui.checkbox(&mut self.settings.verify_only, t.verify_only);
                if self.settings.verify_only {
                    ui.label(RichText::new(t.verify_only_help).size(10.8).color(MUTED));
                }

                ui.add_space(6.0);
                ui.label(RichText::new(t.output_mode).size(12.5).strong().color(TEXT));
                let custom = matches!(self.settings.output_mode, OutputMode::Custom(_));
                ui.horizontal(|ui| {
                    if ui.radio(!custom, t.output_next).clicked() {
                        self.settings.output_mode = OutputMode::NextToInput;
                    }
                    if ui.radio(custom, t.output_custom).clicked() && !custom {
                        let dir = self
                            .last_output_dir
                            .clone()
                            .or_else(|| std::env::current_dir().ok())
                            .unwrap_or_default();
                        self.settings.output_mode = OutputMode::Custom(dir);
                    }
                });

                let custom_dir = match &self.settings.output_mode {
                    OutputMode::Custom(dir) => Some(dir.clone()),
                    OutputMode::NextToInput => None,
                };
                if let Some(dir) = custom_dir {
                    ui.horizontal(|ui| {
                        let text = if dir.as_os_str().is_empty() {
                            t.no_folder.to_string()
                        } else {
                            shorten(&dir.display().to_string(), 78)
                        };
                        ui.label(RichText::new(text).size(10.8).color(MUTED));
                        if ui.button(t.choose_folder).clicked() {
                            if let Some(picked) = rfd::FileDialog::new().pick_folder() {
                                self.last_output_dir = Some(picked.clone());
                                self.settings.output_mode = OutputMode::Custom(picked);
                            }
                        }
                    });
                }
            });
        });
    }

    fn actions_ui(&mut self, ui: &mut Ui, t: &T) {
        let running = self.is_running();
        let verify_only = self.settings.verify_only;

        let convert_jobs: Vec<&Job> = self
            .jobs
            .iter()
            .filter(|j| j.analysis.can_convert())
            .filter(|j| self.settings.overwrite || !matches!(j.status, JobStatus::Done(_)))
            .collect();
        let verify_jobs: Vec<&Job> = self
            .jobs
            .iter()
            .filter(|j| j.analysis.can_verify())
            .collect();
        let selected: &[&Job] = if verify_only {
            &verify_jobs
        } else {
            &convert_jobs
        };
        let count = selected.len();
        let input_total: u64 = selected.iter().map(|j| j.analysis.file_len).sum();
        let output_total: u64 = selected
            .iter()
            .filter(|j| j.analysis.can_convert())
            .filter_map(|j| j.analysis.estimated_new_size)
            .sum();

        ui.horizontal(|ui| {
            let label = if verify_only {
                fill(t.verify_n, &[&count])
            } else {
                fill(t.convert_n, &[&count])
            };
            let enabled = !running && count > 0;
            let (fill_color, text_color) = if enabled {
                (ACCENT, INK)
            } else {
                (PANEL_2, MUTED)
            };
            let button = egui::Button::new(RichText::new(label).strong().color(text_color))
                .fill(fill_color)
                .corner_radius(CornerRadius::same(9))
                .min_size(Vec2::new(190.0, 38.0));
            let response = ui.add_enabled(enabled, button);
            response.clone().on_hover_text("Ctrl+Enter");
            if response.clicked() {
                if verify_only {
                    self.start(WorkKind::Verify);
                } else {
                    self.start(WorkKind::Convert);
                }
            }

            if running {
                ui.spinner();
                ui.label(RichText::new(t.processing).size(12.0).color(ACCENT));
                let stopping = self
                    .cancel
                    .as_ref()
                    .is_some_and(|c| c.load(Ordering::Relaxed));
                if ui
                    .add_enabled(!stopping, egui::Button::new(t.stop))
                    .on_hover_text(t.stop_hint)
                    .clicked()
                {
                    if let Some(cancel) = &self.cancel {
                        cancel.store(true, Ordering::Relaxed);
                    }
                    self.log(LogLevel::Info, t.log_stopping.to_string());
                }
            }

            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if !verify_only && output_total > 0 {
                    ui.label(
                        RichText::new(fill(t.est_output_total, &[&human_bytes(output_total)]))
                            .size(11.0)
                            .color(MUTED),
                    );
                }
                ui.label(
                    RichText::new(format!(
                        "{} · {}",
                        fill(t.files_n, &[&count]),
                        human_bytes(input_total)
                    ))
                    .size(11.0)
                    .color(MUTED),
                );
            });
        });
    }

    fn action_bar(&mut self, ui: &mut Ui, t: &T) {
        egui::Panel::bottom("actions")
            .frame(
                egui::Frame::new()
                    .fill(PANEL)
                    .inner_margin(Margin::symmetric(20, 9))
                    .stroke(Stroke::new(1.0, BORDER)),
            )
            .show(ui, |ui| {
                self.actions_ui(ui, t);
            });
    }

    fn progress_ui(&mut self, ui: &mut Ui, t: &T) {
        let Some(batch) = &self.batch else {
            return;
        };

        let active_job = batch
            .active
            .and_then(|id| self.jobs.iter().find(|j| j.id == id));
        let current_fraction = active_job
            .and_then(|job| match &job.status {
                JobStatus::Running(state) => Some(state.fraction()),
                _ => None,
            })
            .unwrap_or(0.0);
        let overall = ((batch.completed as f32 + current_fraction) / batch.total.max(1) as f32)
            .clamp(0.0, 1.0);
        let file_no = (batch.completed + 1).min(batch.total);

        section_frame(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(t.processing)
                        .size(15.0)
                        .strong()
                        .color(ACCENT),
                );
                ui.label(
                    RichText::new(fill(t.file_of, &[&file_no, &batch.total]))
                        .size(11.0)
                        .color(MUTED),
                );
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.label(
                        RichText::new(format!(
                            "{}: {}",
                            t.elapsed,
                            human_duration(batch.started.elapsed())
                        ))
                        .size(11.0)
                        .color(MUTED),
                    );
                });
            });
            ui.add_space(2.0);
            ui.add(
                ProgressBar::new(overall)
                    .fill(ACCENT)
                    .animate(true)
                    .text(format!("{} · {:.0}%", t.overall, overall * 100.0)),
            );
            if let Some(job) = active_job {
                if let JobStatus::Running(state) = &job.status {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(
                            RichText::new(format!("{}: {}", t.speed, human_speed(state.speed)))
                                .size(10.8)
                                .color(MUTED),
                        );
                        if let Some(eta) = state.eta() {
                            ui.label(
                                RichText::new(format!("{}: {}", t.eta, human_duration(eta)))
                                    .size(10.8)
                                    .color(MUTED),
                            );
                        }
                        let phase = match state.phase {
                            Phase::Converting => t.phase_converting,
                            Phase::Verifying => t.phase_verifying,
                        };
                        ui.label(
                            RichText::new(format!("{} · {}", phase, shorten(&job.file_name(), 60)))
                                .size(10.8)
                                .color(MUTED),
                        );
                    });
                }
            }
        });
    }

    fn log_ui(&mut self, ui: &mut Ui, t: &T) {
        ui.horizontal(|ui| {
            ui.label(RichText::new(t.log).size(16.0).strong().color(TEXT));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui.button(t.clear_log).clicked() {
                    self.entries.clear();
                }
                if ui.button(t.export_log).clicked() {
                    self.export_log();
                }
            });
        });
        ui.add_space(2.0);

        egui::Frame::new()
            .fill(Color32::from_rgb(8, 12, 16))
            .stroke(Stroke::new(1.0, BORDER))
            .corner_radius(CornerRadius::same(10))
            .inner_margin(Margin::same(10))
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                ScrollArea::vertical()
                    .id_salt("log_scroll")
                    .max_height(170.0)
                    .auto_shrink([false, false])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for entry in &self.entries {
                            ui.horizontal_wrapped(|ui| {
                                ui.label(
                                    RichText::new(format!("[{}]", log_clock(entry.at)))
                                        .monospace()
                                        .size(10.5)
                                        .color(MUTED),
                                );
                                ui.label(
                                    RichText::new(&entry.text)
                                        .monospace()
                                        .size(11.0)
                                        .color(entry.level.color()),
                                );
                            });
                        }
                    });
            });
    }

    fn drag_overlay(&self, ctx: &egui::Context, t: &T) {
        let hovering = ctx.input(|i| !i.raw.hovered_files.is_empty());
        if !hovering {
            return;
        }
        let rect = ctx.content_rect();
        let painter = ctx.layer_painter(LayerId::new(Order::Foreground, Id::new("drop_overlay")));
        painter.rect_filled(
            rect,
            CornerRadius::same(0),
            Color32::from_rgba_unmultiplied(8, 12, 15, 225),
        );
        painter.rect_stroke(
            rect.shrink(12.0),
            CornerRadius::same(18),
            Stroke::new(2.0, ACCENT),
            StrokeKind::Inside,
        );
        painter.text(
            rect.center(),
            Align2::CENTER_CENTER,
            t.drop_overlay,
            FontId::proportional(30.0),
            ACCENT,
        );
    }

    fn about_window(&mut self, ctx: &egui::Context, t: &T) {
        if !self.show_about {
            return;
        }
        let mut open = true;
        let mut close = false;
        egui::Window::new(format!("{} - Reblock", t.about))
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .default_width(470.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    theme::paint_logo(ui, 44.0);
                    ui.add_space(4.0);
                    ui.vertical(|ui| {
                        ui.label(RichText::new("Reblock").size(22.0).strong().color(TEXT));
                        ui.label(RichText::new(t.subtitle).size(11.0).color(MUTED));
                    });
                });
                ui.add_space(8.0);
                ui.label(RichText::new(t.about_text).size(12.5).color(TEXT));
                ui.add_space(6.0);
                ui.label(
                    RichText::new(format!("{}: {}", t.version, env!("CARGO_PKG_VERSION")))
                        .size(11.0)
                        .color(MUTED),
                );
                ui.label(RichText::new(t.credits).size(10.8).color(MUTED));
                ui.add_space(8.0);
                if ui.button(t.close).clicked() {
                    close = true;
                }
            });
        self.show_about = open && !close;
    }
}

fn paint_down_arrow(ui: &mut Ui, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(24.0), Sense::hover());
    let c = rect.center();
    let stroke = Stroke::new(2.0, color);
    let painter = ui.painter();
    painter.line_segment([c + Vec2::new(0.0, -8.0), c + Vec2::new(0.0, 7.0)], stroke);
    painter.line_segment([c + Vec2::new(-5.5, 1.5), c + Vec2::new(0.0, 7.0)], stroke);
    painter.line_segment([c + Vec2::new(5.5, 1.5), c + Vec2::new(0.0, 7.0)], stroke);
}

fn section_frame<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> R {
    egui::Frame::new()
        .fill(PANEL)
        .stroke(Stroke::new(1.0, BORDER))
        .corner_radius(CornerRadius::same(12))
        .inner_margin(Margin::symmetric(16, 14))
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            add(ui)
        })
        .inner
}

fn badge(ui: &mut Ui, text: &str, color: Color32) {
    egui::Frame::new()
        .fill(color.gamma_multiply(0.14))
        .stroke(Stroke::new(1.0, color.gamma_multiply(0.45)))
        .corner_radius(CornerRadius::same(5))
        .inner_margin(Margin::symmetric(7, 2))
        .show(ui, |ui| {
            ui.label(RichText::new(text).size(10.0).strong().color(color));
        });
}

fn status_color(job: &Job) -> Color32 {
    match &job.status {
        JobStatus::Ready => match job.analysis.kind {
            FileKind::Pq2 => INFO,
            FileKind::Q2 | FileKind::Other => MUTED,
            FileKind::Ptq1 => WARN,
            FileKind::Invalid => ERROR,
        },
        JobStatus::Running(_) | JobStatus::Done(_) | JobStatus::Verified { .. } => ACCENT,
        JobStatus::Failed(_) => ERROR,
    }
}

fn status_label(job: &Job, t: &T) -> String {
    match &job.status {
        JobStatus::Ready => match job.analysis.kind {
            FileKind::Pq2 => t.status_ready.to_string(),
            FileKind::Q2 => t.status_native.to_string(),
            FileKind::Ptq1 => t.status_ptq1.to_string(),
            FileKind::Other => t.status_other.to_string(),
            FileKind::Invalid => t.status_invalid.to_string(),
        },
        JobStatus::Running(state) => match state.phase {
            Phase::Converting => t.phase_converting.to_string(),
            Phase::Verifying => t.phase_verifying.to_string(),
        },
        JobStatus::Done(_) => t.status_done.to_string(),
        JobStatus::Verified { .. } => t.status_verified.to_string(),
        JobStatus::Failed(_) => t.status_failed.to_string(),
    }
}

fn outcome_text(t: &T, outcome: &VerifyOutcome) -> String {
    match outcome {
        VerifyOutcome::Pq2 { tensors, .. } => fill(t.verified_pq2, &[&tensors]),
        VerifyOutcome::Native { tensors, .. } => fill(t.verified_native, &[&tensors]),
        VerifyOutcome::Other { tensors } => fill(t.verified_other, &[&tensors]),
        VerifyOutcome::Ptq1 { tensors } => {
            format!("{} ({} {})", t.verified_ptq1, tensors, t.tensors)
        }
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
