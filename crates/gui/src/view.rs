//! Everything the user sees: chrome, queue, Discover, inspector, log and modals.
//!
//! The view is a pure function of [`RustBlock`]; every interaction is a
//! [`Message`] handled in [`crate::app`].

use iced::widget::{
    button, checkbox, column, container, image, progress_bar, row, scrollable, space, stack, text,
    text_input, MouseArea,
};
use iced::{Alignment, Border, Color, ContentFit, Element, Font, Length, Padding};

use crate::app::{
    detected_hardware, format_count, short_date, size_fit_badge, DownloadUi, HfState, LogLevel,
    Message, RustBlock, Tab,
};
use crate::hf::{self, HfFile, HfModel, HfSort};
use crate::model::{
    human_bytes, human_duration, human_speed, looks_quantized_name, shorten, JobStatus, OutputMode,
    VerifyKind,
};
use crate::theme::{
    self, ACCENT, ACCENT_DARK, BORDER, ERROR, INFO, INK, MUTED, OK, PANEL, PANEL_2, PANEL_3, STEEL,
    STEEL_LIGHT, TEXT, WARN, WELL,
};
use rustblock_core::{FitEstimate, Target, VerifyMode};

fn spacer() -> iced::widget::Space {
    space().width(Length::Fill)
}

type El<'a> = Element<'a, Message>;

/// Every scroller shares the thin steel bar from the theme.
fn scroll<'a>(content: impl Into<El<'a>>) -> scrollable::Scrollable<'a, Message> {
    scrollable(content).style(theme::scrollbar)
}

const MONO: Font = Font::MONOSPACE;

// ----- Root ---------------------------------------------------------------

pub fn view<'a>(app: &'a RustBlock) -> El<'a> {
    let mut layout = column![chrome(app)].spacing(0);
    layout = layout.push(content(app));
    if app.show_log {
        layout = layout.push(log_panel(app));
    }
    layout = layout.push(status_bar(app));

    let root = container(layout)
        .width(Length::Fill)
        .height(Length::Fill)
        .style(|_| container::Style {
            border: Border {
                color: BORDER,
                width: 1.0,
                radius: 0.0.into(),
            },
            ..container::Style::default()
        });

    let mut layers: Vec<El<'_>> = vec![backdrop(app), root.into()];
    layers.extend(resize_edges());
    if app.confirm_requantize.is_some() {
        layers.push(requantize_modal());
    }
    if app.show_help {
        layers.push(help_modal(app));
    }
    stack(layers)
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

/// The window is frameless, so the resize grips are ours: five pixels on each
/// edge, nine on the corners, with the matching cursor.
fn resize_edges() -> Vec<El<'static>> {
    use iced::mouse::Interaction;
    use iced::window::Direction;

    let edge = |direction: Direction,
                cursor: Interaction,
                width: Length,
                height: Length,
                align_x: Alignment,
                align_y: Alignment| {
        let area = MouseArea::new(space().width(width).height(height))
            .on_press(Message::Resize(direction))
            .interaction(cursor);
        container(area)
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(align_x)
            .align_y(align_y)
            .into()
    };

    let fill = Length::Fill;
    let thick = Length::Fixed(8.0);
    let corner = Length::Fixed(14.0);
    vec![
        edge(
            Direction::North,
            Interaction::ResizingVertically,
            fill,
            thick,
            Alignment::Start,
            Alignment::Start,
        ),
        edge(
            Direction::South,
            Interaction::ResizingVertically,
            fill,
            thick,
            Alignment::Start,
            Alignment::End,
        ),
        edge(
            Direction::West,
            Interaction::ResizingHorizontally,
            thick,
            fill,
            Alignment::Start,
            Alignment::Start,
        ),
        edge(
            Direction::East,
            Interaction::ResizingHorizontally,
            thick,
            fill,
            Alignment::End,
            Alignment::Start,
        ),
        edge(
            Direction::NorthWest,
            Interaction::ResizingDiagonallyUp,
            corner,
            corner,
            Alignment::Start,
            Alignment::Start,
        ),
        edge(
            Direction::NorthEast,
            Interaction::ResizingDiagonallyDown,
            corner,
            corner,
            Alignment::End,
            Alignment::Start,
        ),
        edge(
            Direction::SouthWest,
            Interaction::ResizingDiagonallyDown,
            corner,
            corner,
            Alignment::Start,
            Alignment::End,
        ),
        edge(
            Direction::SouthEast,
            Interaction::ResizingDiagonallyUp,
            corner,
            corner,
            Alignment::End,
            Alignment::End,
        ),
    ]
}

fn backdrop(app: &RustBlock) -> El<'_> {
    container(
        image(app.logo.clone())
            .width(Length::Fill)
            .height(Length::Fill)
            .content_fit(ContentFit::Contain)
            .opacity(0.09),
    )
    .width(Length::Fill)
    .height(Length::Fill)
    .padding(72)
    .center(Length::Fill)
    .into()
}

fn content(app: &RustBlock) -> El<'_> {
    match app.tab {
        Tab::Convert => convert_page(app),
        Tab::Discover => discover_page(app),
        Tab::Inspector => inspector_page(app),
    }
}

// ----- Chrome -------------------------------------------------------------

fn chrome(app: &RustBlock) -> El<'_> {
    let brand = MouseArea::new(
        row![
            image(app.logo.clone())
                .width(Length::Fixed(26.0))
                .height(Length::Fixed(26.0))
                .content_fit(ContentFit::Contain),
            column![
                text("RUSTBLOCK").size(14.0).color(TEXT),
                text("NATIVE GGUF TOOLKIT")
                    .size(7.5)
                    .font(MONO)
                    .color(MUTED),
            ]
            .spacing(1),
        ]
        .spacing(10)
        .align_y(Alignment::Center),
    )
    .on_press(Message::DragWindow)
    .on_double_click(Message::ToggleMaximize);

    let tabs = row![
        tab_button(app, Tab::Convert),
        tab_button(app, Tab::Discover),
        tab_button(app, Tab::Inspector),
    ]
    .spacing(3);

    let right = row![
        text_button(
            if app.show_log { "LOG ON" } else { "LOG" },
            Message::ToggleLog,
            app.show_log,
        ),
        text_button("HELP", Message::ToggleHelp, app.show_help),
        window_button("—", Message::Minimize, false),
        window_button("□", Message::ToggleMaximize, false),
        window_button("✕", Message::CloseWindow, true),
    ]
    .spacing(2)
    .align_y(Alignment::Center);

    container(
        row![brand, spacer(), tabs, spacer(), status_pill(app), right,]
            .padding(Padding::from([0.0, 12.0]))
            .spacing(14)
            .align_y(Alignment::Center),
    )
    .width(Length::Fill)
    .height(Length::Fixed(52.0))
    .align_y(Alignment::Center)
    .style(theme::chrome())
    .into()
}

fn tab_button(app: &RustBlock, tab: Tab) -> El<'static> {
    button(text(tab.label()).size(12.0))
        .on_press(Message::TabSelected(tab))
        .padding(Padding::from([6.0, 14.0]))
        .width(Length::Fixed(92.0))
        .height(Length::Fixed(30.0))
        .style(theme::tab(app.tab == tab))
        .into()
}

fn text_button(label: &'static str, message: Message, active: bool) -> El<'static> {
    button(
        text(label)
            .size(9.5)
            .font(MONO)
            .color(if active { ACCENT } else { MUTED }),
    )
    .on_press(message)
    .padding(Padding::from([5.0, 8.0]))
    .height(Length::Fixed(26.0))
    .style(theme::ghost)
    .into()
}

fn window_button(glyph: &'static str, message: Message, danger: bool) -> El<'static> {
    button(text(glyph).size(13.0))
        .on_press(message)
        .width(Length::Fixed(34.0))
        .height(Length::Fixed(26.0))
        .padding(0)
        .style(theme::window_control(danger))
        .into()
}

fn status_pill(app: &RustBlock) -> El<'static> {
    let (label, color) = if app.is_running() {
        let progress = app
            .jobs
            .iter()
            .find_map(|job| match &job.status {
                JobStatus::Running(state) => Some(state.fraction()),
                _ => None,
            })
            .unwrap_or(0.0);
        (format!("Converting {:.0}%", progress * 100.0), ACCENT)
    } else if app.hf_busy() {
        ("Hub busy".to_string(), INFO)
    } else {
        let (ready, _running, done, failed) = app.queue_counts();
        if done > 0 {
            (format!("{done} done"), OK)
        } else if ready > 0 {
            (format!("{ready} ready"), OK)
        } else if failed > 0 {
            (format!("{failed} failed"), ERROR)
        } else {
            ("Idle".to_string(), MUTED)
        }
    };
    container(text(label).size(9.5).font(MONO).color(color))
        .padding(Padding::from([4.0, 10.0]))
        .style(move |_| theme::outlined(PANEL_2, BORDER, 20.0))
        .into()
}

fn status_bar(app: &RustBlock) -> El<'static> {
    let ram = app
        .hardware
        .ram_bytes
        .map(human_bytes)
        .unwrap_or_else(|| "RAM ?".to_string());
    let vram = app
        .hardware
        .vram_bytes
        .map(|bytes| format!("{} VRAM", human_bytes(bytes)))
        .unwrap_or_else(|| "no NVIDIA GPU".to_string());
    container(
        row![
            text(format!("{ram} RAM · {vram}"))
                .size(9.0)
                .font(MONO)
                .color(MUTED),
            spacer(),
            text(format!(
                "{} files · iced {}",
                app.jobs.len(),
                env!("CARGO_PKG_VERSION")
            ))
            .size(9.0)
            .font(MONO)
            .color(STEEL),
        ]
        .padding(Padding::from([0.0, 12.0]))
        .align_y(Alignment::Center),
    )
    .width(Length::Fill)
    .height(Length::Fixed(24.0))
    .style(|_| theme::outlined(PANEL, BORDER, 0.0))
    .into()
}

// ----- Convert ------------------------------------------------------------

fn convert_page(app: &RustBlock) -> El<'_> {
    let main = column![
        convert_header(app),
        benchmark_note(app),
        batch_card(app),
        queue_view(app),
        drop_hint(app),
    ]
    .spacing(12)
    .padding(Padding::from([18.0, 20.0]))
    .width(Length::Fill)
    .height(Length::Fill);

    row![main, settings_rail(app)]
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

fn convert_header(app: &RustBlock) -> El<'static> {
    let (ready, running, done, failed) = app.queue_counts();
    let running_batch = app.is_running();
    let count = app.selected_jobs();
    let label = if running_batch {
        "Converting…".to_string()
    } else if count == 0 {
        "Start conversion".to_string()
    } else if count == 1 {
        "Convert 1 file".to_string()
    } else {
        format!("Convert {count} files")
    };
    let can_start = !running_batch && count > 0;
    let bench_ready = app.benchmark.is_none() && app.done_job_output().is_some();

    row![
        column![
            text("Queue").size(26.0).color(TEXT),
            text(format!(
                "{} files · {ready} ready · {running} running · {done} done · {failed} failed",
                app.jobs.len()
            ))
            .size(10.5)
            .color(MUTED),
        ]
        .spacing(3),
        spacer(),
        button(text("Clear all").size(10.5).color(MUTED))
            .on_press(Message::ClearQueue)
            .padding(Padding::from([6.0, 10.0]))
            .style(theme::ghost),
        button(text("Benchmark").size(10.5))
            .on_press_maybe(bench_ready.then_some(Message::Benchmark))
            .padding(Padding::from([6.0, 12.0]))
            .height(Length::Fixed(32.0))
            .style(theme::secondary),
        button(text("Add files").size(12.0).color(INK))
            .on_press(Message::Browse)
            .padding(Padding::from([6.0, 16.0]))
            .height(Length::Fixed(32.0))
            .style(theme::secondary),
        button(text(label).size(12.5).color(INK))
            .on_press_maybe(can_start.then_some(Message::StartConvert))
            .padding(Padding::from([8.0, 18.0]))
            .height(Length::Fixed(38.0))
            .style(theme::primary),
    ]
    .spacing(8)
    .align_y(Alignment::Center)
    .into()
}

fn benchmark_note(app: &RustBlock) -> El<'static> {
    match &app.benchmark {
        Some(bench) => container(
            text(format!(
                "Running llama-bench on {}…",
                shorten(&bench.model.display().to_string(), 52)
            ))
            .size(10.5)
            .color(WARN),
        )
        .padding(Padding::from([8.0, 12.0]))
        .width(Length::Fill)
        .style(|_| theme::outlined(PANEL_2, BORDER, 6.0))
        .into(),
        None => spacer().height(Length::Fixed(0.0)).into(),
    }
}

fn batch_card(app: &RustBlock) -> El<'static> {
    let Some(batch) = &app.batch else {
        return spacer().height(Length::Fixed(0.0)).into();
    };
    let active_fraction = app
        .jobs
        .iter()
        .find(|job| Some(job.id) == batch.active)
        .and_then(|job| match &job.status {
            JobStatus::Running(state) => Some(state.fraction()),
            _ => None,
        })
        .unwrap_or(0.0);
    let overall =
        ((batch.completed as f32 + active_fraction) / batch.total.max(1) as f32).clamp(0.0, 1.0);
    let stopping = app
        .cancel
        .as_ref()
        .is_some_and(|cancel| cancel.load(std::sync::atomic::Ordering::Relaxed));

    container(
        column![
            row![
                text(format!(
                    "Batch · {} of {}",
                    batch.completed.min(batch.total),
                    batch.total
                ))
                .size(11.0)
                .color(STEEL_LIGHT),
                text(human_duration(batch.elapsed()))
                    .size(10.5)
                    .font(MONO)
                    .color(MUTED),
                spacer(),
                button(text(if stopping { "Stopping…" } else { "Stop" }).size(10.5))
                    .on_press_maybe((!stopping).then_some(Message::CancelBatch))
                    .padding(Padding::from([5.0, 12.0]))
                    .style(theme::secondary),
            ]
            .align_y(Alignment::Center)
            .spacing(10),
            progress_bar(0.0..=1.0, overall)
                .girth(6.0)
                .style(theme::progress),
        ]
        .spacing(8),
    )
    .padding(14)
    .width(Length::Fill)
    .style(|_| theme::outlined(PANEL_2, BORDER, 8.0))
    .into()
}

fn queue_view(app: &RustBlock) -> El<'_> {
    if app.jobs.is_empty() {
        return empty_queue();
    }
    let mut list = column![].spacing(8);
    for job in &app.jobs {
        list = list.push(job_card(app, job));
    }
    scroll(list).height(Length::Fill).width(Length::Fill).into()
}

fn empty_queue() -> El<'static> {
    container(
        column![
            text("DROP ZONE").size(9.5).font(MONO).color(STEEL_LIGHT),
            text("Drop GGUF files here").size(19.0).color(TEXT),
            text("One or more .gguf files. Nothing is changed until you start.")
                .size(11.5)
                .color(MUTED),
            text("Ctrl+O add files · Ctrl+S start · Ctrl+L log · F1 help")
                .size(10.0)
                .font(MONO)
                .color(STEEL_LIGHT),
            row![
                button(text("Browse files").size(11.5))
                    .on_press(Message::Browse)
                    .padding(Padding::from([7.0, 16.0]))
                    .style(theme::secondary),
                button(text("Discover models").size(11.5).color(MUTED))
                    .on_press(Message::TabSelected(Tab::Discover))
                    .padding(Padding::from([7.0, 14.0]))
                    .style(theme::ghost),
            ]
            .spacing(8),
        ]
        .spacing(10)
        .align_x(Alignment::Center),
    )
    .width(Length::Fill)
    .height(Length::Fill)
    .padding(Padding::from([32.0, 26.0]))
    .center(Length::Fill)
    .style(|_| theme::outlined(theme::translucent(PANEL, 0.66), BORDER, 8.0))
    .into()
}

fn drop_hint(app: &RustBlock) -> El<'static> {
    let total: u64 = app.jobs.iter().map(|job| job.analysis.file_len).sum();
    row![
        text("Drop more GGUF files anywhere in this window.")
            .size(10.0)
            .color(MUTED),
        spacer(),
        text(format!("queue {}", human_bytes(total)))
            .size(10.0)
            .font(MONO)
            .color(STEEL),
        button(text("Browse").size(10.0))
            .on_press(Message::Browse)
            .padding(Padding::from([4.0, 10.0]))
            .style(theme::secondary),
    ]
    .spacing(8)
    .align_y(Alignment::Center)
    .into()
}

fn job_card<'a>(app: &'a RustBlock, job: &'a crate::model::Job) -> El<'a> {
    let selected = app.selected_job == Some(job.id);
    let (status, color) = job_status(job);

    let mut body = column![
        row![
            text(job.file_name())
                .size(12.5)
                .font(MONO)
                .color(STEEL_LIGHT)
                .width(Length::Fill),
            chip(job.target.short(), PANEL_3, TEXT),
        ]
        .align_y(Alignment::Center)
        .spacing(8),
        text(format!(
            "{} · {} · v{} · align {} · {} tensors",
            human_bytes(job.analysis.file_len),
            job.analysis.kind_label(),
            job.analysis.version,
            job.analysis.alignment,
            job.analysis.tensors
        ))
        .size(9.5)
        .color(MUTED),
        text(status).size(10.0).color(color),
    ]
    .spacing(4);

    if let JobStatus::Running(state) = &job.status {
        body = body.push(
            progress_bar(0.0..=1.0, state.fraction())
                .girth(5.0)
                .style(theme::progress),
        );
    }

    if selected {
        body = body.push(details(app, job));
    }

    let mut actions = column![
        button(text(if selected { "Hide" } else { "Details" }).size(10.0))
            .on_press(Message::SelectJob(job.id))
            .padding(Padding::from([5.0, 10.0]))
            .width(Length::Fill)
            .style(theme::secondary),
    ]
    .spacing(6)
    .width(Length::Fixed(92.0));

    if let Some(path) = job.done_output() {
        actions = actions.push(
            button(text("Open").size(10.0))
                .on_press(Message::OpenPath(path))
                .padding(Padding::from([5.0, 10.0]))
                .width(Length::Fill)
                .style(theme::ghost),
        );
    }
    if !app.is_running() {
        actions = actions.push(
            button(text("Remove").size(10.0).color(MUTED))
                .on_press(Message::RemoveJob(job.id))
                .padding(Padding::from([4.0, 10.0]))
                .width(Length::Fill)
                .style(theme::ghost),
        );
    }

    container(
        row![body.width(Length::Fill), actions]
            .spacing(12)
            .align_y(Alignment::Start),
    )
    .padding(13)
    .width(Length::Fill)
    .style(theme::surface(selected))
    .into()
}

fn details<'a>(app: &'a RustBlock, job: &'a crate::model::Job) -> El<'a> {
    let mut rows = column![divider()].spacing(6);

    if let Some(fit) = &job.analysis.fit {
        let (label, color, hint) = fit_badge(fit);
        rows = rows.push(
            row![
                chip_text(label, color),
                text(fit_summary(fit)).size(9.5).font(MONO).color(MUTED),
            ]
            .spacing(10)
            .align_y(Alignment::Center),
        );
        rows = rows.push(text(hint).size(9.0).color(MUTED));
    }

    rows = rows.push(
        text(format!(
            "→ {}",
            shorten(
                &job.planned_output(&app.settings.output_mode)
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                52
            )
        ))
        .size(9.5)
        .font(MONO)
        .color(STEEL_LIGHT),
    );
    rows = rows.push(text(job.analysis.type_summary()).size(9.5).color(MUTED));

    if let JobStatus::Done { report, .. } = &job.status {
        rows = rows.push(
            text(format!(
                "{} → {} in {}",
                human_bytes(report.old_file_size),
                human_bytes(report.new_file_size),
                human_duration(report.elapsed)
            ))
            .size(9.5)
            .font(MONO)
            .color(OK),
        );
    }
    if let JobStatus::Verified { outcome, elapsed } = &job.status {
        rows = rows.push(
            text(format!(
                "{} · {} check · {} tensors · {}",
                outcome.label,
                verify_label(outcome.kind),
                outcome.tensors,
                human_duration(*elapsed)
            ))
            .size(9.5)
            .color(OK),
        );
    }
    if let JobStatus::Failed(error) = &job.status {
        rows = rows.push(text(error.as_str()).size(9.5).color(ERROR));
    }

    let mut buttons = row![].spacing(6);
    if !app.is_running() && job.analysis.error.is_none() {
        buttons = buttons.push(
            button(text("Verify").size(10.0))
                .on_press(Message::StartVerify)
                .padding(Padding::from([4.0, 10.0]))
                .style(theme::secondary),
        );
    }
    if let Some(output) = job.done_output() {
        buttons = buttons.push(
            button(text("Open folder").size(10.0))
                .on_press(Message::OpenPath(output))
                .padding(Padding::from([4.0, 10.0]))
                .style(theme::secondary),
        );
    }
    rows = rows.push(buttons);

    rows.into()
}

fn job_status(job: &crate::model::Job) -> (String, Color) {
    match &job.status {
        JobStatus::Ready => {
            if job.analysis.can_target(job.target) {
                (format!("Ready · {}", job.target.short()), MUTED)
            } else {
                ("No compatible target".to_string(), WARN)
            }
        }
        JobStatus::Running(state) => (
            format!(
                "{} · {:.0}% · {} · {}",
                phase_name(state.phase),
                state.fraction() * 100.0,
                human_bytes(state.done),
                match state.eta() {
                    Some(eta) => format!("{} left", human_duration(eta)),
                    None => "estimating".to_string(),
                }
            ),
            ACCENT,
        ),
        JobStatus::Done { report, target } => (
            format!(
                "Done · {} · {} → {}",
                target.short(),
                human_bytes(report.old_file_size),
                human_bytes(report.new_file_size)
            ),
            OK,
        ),
        JobStatus::Verified { outcome, .. } => (format!("Verified · {}", outcome.label), OK),
        JobStatus::Failed(error) => (format!("Failed · {error}"), ERROR),
    }
}

fn phase_name(phase: rustblock_core::Phase) -> &'static str {
    match phase {
        rustblock_core::Phase::Converting => "Converting",
        rustblock_core::Phase::Verifying => "Verifying",
    }
}

fn settings_rail(app: &RustBlock) -> El<'_> {
    let body = column![
        format_section(app),
        divider(),
        destination_section(app),
        divider(),
        verification_section(app),
        text("Tip: the fit badge is an estimate at 4096 tokens, not a promise.")
            .size(9.0)
            .color(MUTED),
    ]
    .spacing(10)
    .width(Length::Fill);
    container(scroll(container(body).padding(Padding::from([0.0, 10.0]))).height(Length::Fill))
        .width(Length::Fixed(372.0))
        .height(Length::Fill)
        .padding(Padding::from([16.0, 16.0]))
        .style(|_| theme::outlined(PANEL, BORDER, 0.0))
        .into()
}

fn format_section(app: &RustBlock) -> El<'static> {
    let running = app.is_running();
    let mut available: Vec<Target> = Target::ALL
        .iter()
        .copied()
        .filter(|target| {
            *target != Target::Reblock
                || app.jobs.iter().any(|job| job.analysis.can_target(*target))
        })
        .collect();
    if available.is_empty() {
        available.push(Target::Q4KM);
    }
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

    let mut section = column![
        row![
            section_label("OUTPUT FORMAT"),
            spacer(),
            button(text("Non so cosa scegliere").size(9.0).color(STEEL_LIGHT))
                .on_press_maybe((!running).then_some(Message::SuggestTarget))
                .padding(Padding::from([2.0, 4.0]))
                .style(theme::ghost),
        ]
        .align_y(Alignment::Center),
        text(app.settings.target.short()).size(24.0).color(TEXT),
        text(target_caption(app.settings.target))
            .size(9.5)
            .color(MUTED),
    ]
    .spacing(4);

    section = section.push(target_grid(&standard, app.settings.target));
    if !ternary.is_empty() {
        section = section.push(
            row![
                text("Ternary").size(11.0).color(TEXT),
                text("only for ternary models").size(9.5).color(WARN),
            ]
            .spacing(8),
        );
        section = section.push(target_grid(&ternary, app.settings.target));
    }
    section.into()
}

fn target_grid(targets: &[Target], selected: Target) -> El<'static> {
    let mut grid = column![].spacing(4);
    for chunk in targets.chunks(3) {
        let mut line = row![].spacing(4);
        for target in chunk {
            line = line.push(
                button(
                    text(target.short())
                        .size(9.5)
                        .color(if *target == selected {
                            INK
                        } else {
                            STEEL_LIGHT
                        }),
                )
                .on_press(Message::SetTarget(*target))
                .width(Length::FillPortion(1))
                .padding(Padding::from([6.0, 4.0]))
                .style(theme::segment(*target == selected)),
            );
        }
        for _ in chunk.len()..3 {
            line = line.push(spacer().width(Length::FillPortion(1)));
        }
        grid = grid.push(line);
    }
    grid.into()
}

fn destination_section(app: &RustBlock) -> El<'static> {
    let running = app.is_running();
    let custom = matches!(app.settings.output_mode, OutputMode::Custom(_));
    let label = match &app.settings.output_mode {
        OutputMode::NextToInput => "Next to source file".to_string(),
        OutputMode::Custom(dir) => shorten(&dir.display().to_string(), 32),
    };
    let mut section = column![row![
        section_label("DESTINATION"),
        spacer(),
        checkbox(app.settings.overwrite)
            .label("Overwrite")
            .on_toggle(Message::SetOverwrite)
            .size(14.0)
            .text_size(10.0)
            .style(theme::checkbox_style),
    ]
    .align_y(Alignment::Center),]
    .spacing(4);

    section = section.push(
        button(text(label).size(10.5).font(MONO).color(STEEL_LIGHT))
            .on_press_maybe((!running).then_some(Message::PickOutputDir))
            .width(Length::Fill)
            .padding(Padding::from([7.0, 10.0]))
            .style(theme::secondary),
    );
    if custom {
        section = section.push(
            button(text("Use source folder").size(9.5).color(MUTED))
                .on_press_maybe((!running).then_some(Message::UseSourceDir))
                .padding(Padding::from([2.0, 4.0]))
                .style(theme::ghost),
        );
    }
    section.into()
}

fn verification_section(app: &RustBlock) -> El<'static> {
    let running = app.is_running();
    let mut line = row![].spacing(4);
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
        let selected = app.settings.verify == mode;
        line = line.push(
            button(
                text(label)
                    .size(10.0)
                    .color(if selected { INK } else { STEEL_LIGHT }),
            )
            .on_press_maybe((!running).then_some(Message::SetVerify(mode)))
            .width(Length::FillPortion(1))
            .padding(Padding::from([6.0, 4.0]))
            .style(theme::segment(selected)),
        );
        let _ = hint;
    }
    column![section_label("VERIFICATION"), line]
        .spacing(5)
        .into()
}

// ----- Discover -----------------------------------------------------------

fn discover_page(app: &RustBlock) -> El<'_> {
    let main = column![discover_header(app), search_bar(app), discover_body(app),]
        .spacing(12)
        .padding(Padding::from([18.0, 20.0]))
        .width(Length::Fill)
        .height(Length::Fill);

    row![main, downloads_rail(app)]
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

fn discover_header(app: &RustBlock) -> El<'static> {
    row![
        column![
            text("Discover").size(26.0).color(TEXT),
            text("GGUF repositories on Hugging Face, with a fit verdict before the download.")
                .size(10.5)
                .color(MUTED),
        ]
        .spacing(3),
        spacer(),
        button(text("Trending").size(10.5))
            .on_press_maybe((!app.hf_busy()).then_some(Message::Trending))
            .padding(Padding::from([6.0, 14.0]))
            .height(Length::Fixed(32.0))
            .style(theme::secondary),
    ]
    .align_y(Alignment::Center)
    .into()
}

fn search_bar(app: &RustBlock) -> El<'_> {
    let busy = app.hf_busy();
    let mut sorts = row![].spacing(3);
    for (sort, label) in [
        (HfSort::Downloads, "Downloads"),
        (HfSort::Likes, "Likes"),
        (HfSort::Recent, "Recent"),
    ] {
        sorts = sorts.push(
            button(text(label).size(9.5))
                .on_press(Message::SetSort(sort))
                .padding(Padding::from([5.0, 10.0]))
                .style(theme::segment(app.hf_sort == sort)),
        );
    }

    row![
        text_input(
            "Search GGUF repositories or paste a huggingface.co link",
            &app.hf_query
        )
        .on_input(Message::SearchChanged)
        .on_submit(Message::SearchSubmit)
        .size(11.0)
        .padding(Padding::from([8.0, 10.0]))
        .style(theme::text_field),
        sorts,
        button(
            text(if busy { "Searching…" } else { "Search" })
                .size(11.0)
                .color(INK)
        )
        .on_press_maybe((!busy).then_some(Message::SearchSubmit))
        .padding(Padding::from([8.0, 18.0]))
        .style(theme::primary),
    ]
    .spacing(8)
    .align_y(Alignment::Center)
    .into()
}

fn discover_body(app: &RustBlock) -> El<'_> {
    if let Some(error) = &app.hf_details_error {
        let card = container(text(error.as_str()).size(10.5).color(WARN))
            .padding(12)
            .width(Length::Fill)
            .style(|_| theme::outlined(PANEL_2, BORDER, 6.0));
        return scroll(column![card, results(app)].spacing(10))
            .height(Length::Fill)
            .into();
    }
    match &app.hf_state {
        HfState::Idle => idle_discover(),
        HfState::Searching => container(
            column![
                text("Searching Hugging Face…")
                    .size(13.0)
                    .color(STEEL_LIGHT),
                text("The first request can take a few seconds.")
                    .size(10.0)
                    .color(MUTED),
            ]
            .spacing(4),
        )
        .width(Length::Fill)
        .height(Length::Fill)
        .center(Length::Fill)
        .into(),
        HfState::Error(error) => container(
            column![
                text("Search failed").size(14.0).color(ERROR),
                text(error.as_str()).size(10.5).color(MUTED),
                button(text("Try the trending list").size(10.5))
                    .on_press(Message::Trending)
                    .padding(Padding::from([6.0, 12.0]))
                    .style(theme::secondary),
            ]
            .spacing(8)
            .align_x(Alignment::Center),
        )
        .width(Length::Fill)
        .height(Length::Fill)
        .center(Length::Fill)
        .into(),
        HfState::Results(_) => results(app),
    }
}

fn idle_discover() -> El<'static> {
    let mut chips = row![].spacing(6);
    for query in ["Qwen", "Llama", "Mistral", "Gemma", "Phi", "DeepSeek"] {
        chips = chips.push(
            button(text(query).size(10.5))
                .on_press(Message::SearchChanged(query.to_string()))
                .padding(Padding::from([5.0, 12.0]))
                .style(theme::secondary),
        );
    }
    container(
        column![
            text("Search the Hub").size(17.0).color(TEXT),
            text("Type a query above, paste a model link, or start from a popular family.")
                .size(10.5)
                .color(MUTED),
            chips,
        ]
        .spacing(10)
        .align_x(Alignment::Center),
    )
    .width(Length::Fill)
    .height(Length::Fill)
    .center(Length::Fill)
    .into()
}

fn results(app: &RustBlock) -> El<'_> {
    let HfState::Results(models) = &app.hf_state else {
        return spacer().into();
    };
    let mut list = column![].spacing(6);
    for model in models {
        list = list.push(result_card(app, model));
    }
    scroll(list).height(Length::Fill).into()
}

fn result_card<'a>(app: &'a RustBlock, model: &'a HfModel) -> El<'a> {
    let selected = app.selected_repo.as_deref() == Some(model.id.as_str());
    let mut meta = row![].spacing(8).align_y(Alignment::Center);
    meta = meta.push(
        text(format_count(model.downloads))
            .size(9.5)
            .font(MONO)
            .color(MUTED),
    );
    meta = meta.push(text("downloads").size(9.0).color(MUTED));
    if let Some(license) = &model.license {
        meta = meta.push(chip(license.clone(), PANEL_3, STEEL_LIGHT));
    }
    if model.gated {
        meta = meta.push(chip("gated", PANEL_3, WARN));
    }

    button(
        column![
            row![
                text(model.name.as_str()).size(12.5).color(if selected {
                    TEXT
                } else {
                    STEEL_LIGHT
                }),
                spacer(),
                text(short_date(&model.last_modified))
                    .size(9.0)
                    .font(MONO)
                    .color(MUTED),
            ]
            .align_y(Alignment::Center),
            text(model.id.as_str()).size(9.5).font(MONO).color(MUTED),
            meta,
        ]
        .spacing(4),
    )
    .on_press(Message::OpenRepo(model.id.clone()))
    .width(Length::Fill)
    .padding(11)
    .style(card_button(selected))
    .into()
}

fn card_button(selected: bool) -> impl Fn(&iced::Theme, button::Status) -> button::Style {
    move |_, status| {
        let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
        button::Style {
            background: Some(iced::Background::Color(if selected || hovered {
                PANEL_3
            } else {
                PANEL_2
            })),
            text_color: TEXT,
            border: Border {
                color: if selected { ACCENT_DARK } else { BORDER },
                width: 1.0,
                radius: 8.0.into(),
            },
            shadow: Default::default(),
            snap: false,
        }
    }
}

fn downloads_rail(app: &RustBlock) -> El<'_> {
    let mut body = column![].spacing(10);
    if let Some(download) = &app.download {
        body = body.push(download_card(app, download));
    }
    body = body.push(repo_panel(app));

    let body = container(body.width(Length::Fill)).padding(Padding::from([0.0, 10.0]));
    container(scroll(body).height(Length::Fill))
        .width(Length::Fixed(372.0))
        .height(Length::Fill)
        .padding(Padding::from([16.0, 16.0]))
        .style(|_| theme::outlined(PANEL, BORDER, 0.0))
        .into()
}

fn repo_panel(app: &RustBlock) -> El<'_> {
    let Some(model) = app.current_model() else {
        return container(
            column![
                section_label("REPOSITORY"),
                text("Pick a repository to list its GGUF files.")
                    .size(10.5)
                    .color(MUTED),
                text(format!(
                    "Downloads are resumable and verified against the SHA-256 published by the Hub. Destination: {}",
                    shorten(&app.settings.download_dir.display().to_string(), 44)
                ))
                .size(9.0)
                .color(MUTED),
                button(text("Change folder").size(9.5))
                    .on_press(Message::PickDownloadDir)
                    .padding(Padding::from([4.0, 8.0]))
                    .style(theme::secondary),
            ]
            .spacing(8),
        )
        .into();
    };

    let license = model
        .license
        .clone()
        .map(|license| format!(" · {license}"))
        .unwrap_or_default();
    let mut panel = column![
        row![
            text(model.name.clone())
                .size(15.0)
                .color(TEXT)
                .width(Length::Fill),
            button(text("Hub ↗").size(9.5))
                .on_press(Message::OpenUrl(model.web_url()))
                .padding(Padding::from([4.0, 6.0]))
                .style(theme::ghost),
        ]
        .align_y(Alignment::Center),
        text(model.id.clone()).size(9.5).font(MONO).color(MUTED),
        text(format!(
            "by {} · {} downloads · {} GGUF files{license}",
            model.author,
            format_count(model.downloads),
            model.gguf_files
        ))
        .size(9.5)
        .color(STEEL_LIGHT),
        row![
            text_input("Filter files", &app.hf_file_filter)
                .on_input(Message::FileFilterChanged)
                .size(10.0)
                .padding(Padding::from([6.0, 8.0]))
                .width(Length::Fill)
                .style(theme::text_field),
            button(text("Folder").size(9.5))
                .on_press(Message::PickDownloadDir)
                .padding(Padding::from([5.0, 10.0]))
                .style(theme::secondary),
        ]
        .spacing(6)
        .align_y(Alignment::Center),
    ]
    .spacing(6);

    if app.hf_details_loading {
        panel = panel.push(text("Loading files…").size(10.0).color(INFO));
    }
    if let Some(error) = &app.hf_details_error {
        panel = panel.push(text(error.as_str()).size(9.5).color(ERROR));
    }

    let filter = app.hf_file_filter.to_ascii_lowercase();
    let files: Vec<&HfFile> = app
        .hf_files
        .iter()
        .filter(|file| filter.is_empty() || file.path.to_ascii_lowercase().contains(&filter))
        .collect();

    if files.is_empty() && !app.hf_details_loading {
        panel = panel.push(
            text("No downloadable GGUF files match the filter.")
                .size(9.5)
                .color(MUTED),
        );
    }

    for file in files {
        panel = panel.push(file_row(app, file));
    }

    panel = panel.push(
        row![
            text("Downloads are resumable and SHA-256 checked.")
                .size(8.5)
                .color(MUTED)
                .width(Length::Fill),
            button(text("Clean partials").size(8.5).color(MUTED))
                .on_press(Message::ClearPartials)
                .padding(Padding::from([2.0, 4.0]))
                .style(theme::ghost),
        ]
        .align_y(Alignment::Center),
    );

    panel.into()
}

fn file_row<'a>(app: &'a RustBlock, file: &'a HfFile) -> El<'a> {
    let path = app.settings.download_dir.join(hf::file_name(&file.path));
    let cached = path.exists();
    let (label, color, _hint) = size_fit_badge(file.size, detected_hardware());

    let mut meta = row![
        text(human_bytes(file.size))
            .size(9.5)
            .font(MONO)
            .color(MUTED),
        chip_text(label, color),
    ]
    .spacing(6)
    .align_y(Alignment::Center);

    if file.projector {
        meta = meta.push(text("vision projector").size(9.0).color(INFO));
    } else if cached {
        meta = meta.push(text("already downloaded").size(9.0).color(ACCENT));
    } else if looks_quantized_name(&file.path) {
        meta = meta.push(text("already quantized").size(9.0).color(WARN));
    }
    if !file.supported && !file.projector {
        meta = meta.push(text("not a standalone model").size(9.0).color(ERROR));
    }

    let action: El<'_> = if cached {
        button(text("Add").size(9.5))
            .on_press(Message::AddCached(path))
            .padding(Padding::from([5.0, 10.0]))
            .style(theme::secondary)
            .into()
    } else if file.projector {
        button(text("Get").size(9.5))
            .on_press(Message::Download(file.clone()))
            .padding(Padding::from([5.0, 10.0]))
            .style(theme::secondary)
            .into()
    } else if file.supported {
        button(text("Get").size(9.5).color(INK))
            .on_press(Message::Download(file.clone()))
            .padding(Padding::from([5.0, 12.0]))
            .style(theme::primary)
            .into()
    } else {
        button(text("—").size(9.5).color(MUTED))
            .padding(Padding::from([5.0, 12.0]))
            .style(theme::secondary)
            .into()
    };

    container(
        row![
            column![
                text(shorten(hf::file_name(&file.path), 34))
                    .size(10.0)
                    .font(MONO)
                    .color(STEEL_LIGHT),
                meta,
            ]
            .spacing(3)
            .width(Length::Fill),
            action,
        ]
        .spacing(8)
        .align_y(Alignment::Center),
    )
    .padding(Padding::from([7.0, 8.0]))
    .width(Length::Fill)
    .style(|_| theme::outlined(PANEL_2, BORDER, 6.0))
    .into()
}

fn download_card<'a>(app: &'a RustBlock, download: &'a DownloadUi) -> El<'a> {
    let frac = if download.total > 0 {
        (download.done as f64 / download.total as f64).clamp(0.0, 1.0) as f32
    } else {
        0.0
    };
    let (status, color) = if download.path.is_some() {
        ("Complete".to_string(), OK)
    } else if download.error.is_some() {
        ("Failed".to_string(), ERROR)
    } else if download.projector {
        ("Downloading projector".to_string(), ACCENT)
    } else {
        ("Downloading".to_string(), ACCENT)
    };
    let mut content = column![
        row![
            text(shorten(hf::file_name(&download.file), 34))
                .size(10.0)
                .font(MONO)
                .color(STEEL_LIGHT),
            spacer(),
            text(status).size(9.5).color(color),
        ]
        .align_y(Alignment::Center),
        progress_bar(0.0..=1.0, frac)
            .girth(6.0)
            .style(theme::progress),
        row![
            text(format!(
                "{} / {} · {:.1}% · {}",
                human_bytes(download.done),
                if download.total > 0 {
                    human_bytes(download.total)
                } else {
                    "?".to_string()
                },
                frac * 100.0,
                human_speed(download.speed)
            ))
            .size(9.0)
            .font(MONO)
            .color(MUTED),
            spacer(),
            button(text("Cancel").size(9.0).color(MUTED))
                .on_press_maybe(
                    (download.path.is_none() && download.error.is_none())
                        .then_some(Message::CancelDownload)
                )
                .padding(Padding::from([2.0, 6.0]))
                .style(theme::ghost),
            button(text("Dismiss").size(9.0).color(MUTED))
                .on_press_maybe(
                    (download.path.is_some() || download.error.is_some())
                        .then_some(Message::DismissDownload)
                )
                .padding(Padding::from([2.0, 6.0]))
                .style(theme::ghost),
        ]
        .align_y(Alignment::Center),
    ]
    .spacing(6);

    if let Some(path) = &download.path {
        content = content.push(
            button(text("Open folder").size(9.0))
                .on_press(Message::OpenPath(path.clone()))
                .padding(Padding::from([2.0, 6.0]))
                .style(theme::ghost),
        );
    }
    let _ = app;
    if download.quantized {
        content = content.push(
            text("Already quantized · requantizing will not restore the original weights.")
                .size(9.0)
                .color(WARN),
        );
    }
    if let Some(error) = &download.error {
        content = content.push(text(error.as_str()).size(9.0).color(ERROR));
    }

    container(content)
        .padding(10)
        .width(Length::Fill)
        .style(|_| theme::outlined(PANEL_2, BORDER, 6.0))
        .into()
}

// ----- Inspector ----------------------------------------------------------

fn inspector_page(app: &RustBlock) -> El<'_> {
    let mut header = row![
        column![
            text("Inspector").size(26.0).color(TEXT),
            text("Facts, metadata and tensor histogram of a local GGUF.")
                .size(10.5)
                .color(MUTED),
        ]
        .spacing(3),
        spacer(),
        button(text("Refresh").size(10.5))
            .on_press(Message::InspectorRefresh)
            .padding(Padding::from([6.0, 12.0]))
            .style(theme::secondary),
        button(text("Open file").size(11.0).color(INK))
            .on_press(Message::InspectorOpen)
            .padding(Padding::from([7.0, 14.0]))
            .height(Length::Fixed(34.0))
            .style(theme::primary),
    ]
    .align_y(Alignment::Center);

    header = header.push(spacer());

    let body: El<'_> = match &app.inspector {
        None => container(
            column![
                text("Open a GGUF").size(17.0).color(TEXT),
                text("Pick any local .gguf: header facts, metadata and the exact tensor type histogram.")
                    .size(10.5)
                    .color(MUTED),
            ]
            .spacing(6)
            .align_x(Alignment::Center),
        )
        .width(Length::Fill)
        .height(Length::Fill)
        .center(Length::Fill)
        .style(|_| theme::outlined(theme::translucent(PANEL, 0.66), BORDER, 8.0))
        .into(),
        Some(data) => {
            let mut facts = column![].spacing(4);
            for (label, value) in [
                ("file", human_bytes(data.file_len)),
                ("version", format!("v{}", data.version)),
                ("tensors", data.tensors.to_string()),
                ("alignment", data.alignment.to_string()),
                ("data start", human_bytes(data.data_start)),
                ("PQ2_0 tensors", data.pq2.to_string()),
                ("PTQ1_0 tensors", data.ptq1.to_string()),
                (
                    "file type",
                    data.file_type
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "unknown".to_string()),
                ),
            ] {
                facts = facts.push(
                    row![
                        text(label).size(9.5).font(MONO).color(MUTED).width(Length::Fixed(130.0)),
                        text(value).size(10.5).font(MONO).color(TEXT),
                    ]
                    .spacing(8),
                );
            }

            let mut metadata = column![].spacing(3);
            for (key, value) in &data.metadata {
                metadata = metadata.push(
                    row![
                        text(key.as_str())
                            .size(9.5)
                            .font(MONO)
                            .color(MUTED)
                            .width(Length::Fixed(230.0)),
                        text(value.as_str()).size(10.0).color(TEXT),
                    ]
                    .spacing(8),
                );
            }

            let max_bytes = data
                .histogram
                .iter()
                .map(|(_, _, bytes)| *bytes)
                .max()
                .unwrap_or(1);
            let mut histogram = column![].spacing(5);
            for (name, count, bytes) in &data.histogram {
                let fraction = *bytes as f32 / max_bytes as f32;
                histogram = histogram.push(
                    column![
                        row![
                            text(name.as_str()).size(10.0).font(MONO).color(TEXT),
                            spacer(),
                            text(format!("{count} × {}", human_bytes(*bytes)))
                                .size(9.5)
                                .font(MONO)
                                .color(MUTED),
                        ],
                        progress_bar(0.0..=1.0, fraction)
                            .girth(5.0)
                            .style(theme::progress),
                    ]
                    .spacing(3),
                );
            }

            let fit: El<'_> = match &data.fit {
                Some(estimate) => {
                    let (label, color, hint) = fit_badge(estimate);
                    column![
                        row![chip_text(label, color), text(fit_summary(estimate)).size(9.5).color(MUTED)]
                            .spacing(10)
                            .align_y(Alignment::Center),
                        text(hint).size(9.0).color(MUTED),
                    ]
                    .spacing(4)
                    .into()
                }
                None => text("Fit estimate unavailable.").size(9.5).color(MUTED).into(),
            };

            scroll(
                column![
                    row![
                        text(data.path.display().to_string())
                            .size(9.5)
                            .font(MONO)
                            .color(STEEL_LIGHT),
                        spacer(),
                        text(format!("{} tensors", data.tensors)).size(9.5).color(MUTED),
                    ]
                    .align_y(Alignment::Center),
                    divider(),
                    text("FACTS").size(9.0).font(MONO).color(STEEL_LIGHT),
                    facts,
                    row![
                        text("weights")
                            .size(9.5)
                            .font(MONO)
                            .color(MUTED)
                            .width(Length::Fixed(130.0)),
                        text(human_bytes(data.weights_bytes))
                            .size(10.5)
                            .font(MONO)
                            .color(TEXT),
                    ]
                    .spacing(8),
                    fit,
                    divider(),
                    text("METADATA").size(9.0).font(MONO).color(STEEL_LIGHT),
                    metadata,
                    divider(),
                    text("TENSOR TYPES").size(9.0).font(MONO).color(STEEL_LIGHT),
                    histogram,
                ]
                .spacing(8)
                .width(Length::Fill),
            )
            .height(Length::Fill)
            .into()
        }
    };

    column![header, body]
        .spacing(12)
        .padding(Padding::from([18.0, 20.0]))
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

// ----- Log ----------------------------------------------------------------

fn log_panel(app: &RustBlock) -> El<'_> {
    let visible = app.visible_logs().len();
    let mut levels = row![].spacing(3);
    for level in [
        LogLevel::Info,
        LogLevel::Ok,
        LogLevel::Warn,
        LogLevel::Error,
    ] {
        levels = levels.push(
            button(text(level.short()).size(9.0))
                .on_press(Message::ToggleLevel(level.index()))
                .padding(Padding::from([3.0, 8.0]))
                .style(theme::log_chip(
                    level.color(),
                    app.log_levels[level.index()],
                )),
        );
    }

    let header = row![
        section_label("ACTIVITY LOG"),
        levels,
        text(format!("{visible}/{}", app.logs.len()))
            .size(9.0)
            .font(MONO)
            .color(STEEL_LIGHT),
        spacer(),
        text_input("Filter", &app.log_filter)
            .on_input(Message::LogFilterChanged)
            .size(9.5)
            .padding(Padding::from([5.0, 8.0]))
            .width(Length::Fixed(160.0))
            .style(theme::text_field),
        button(text("Clear").size(9.5).color(MUTED))
            .on_press(Message::ClearLog)
            .padding(Padding::from([4.0, 8.0]))
            .style(theme::ghost),
        button(text("Close").size(9.5).color(MUTED))
            .on_press(Message::ToggleLog)
            .padding(Padding::from([4.0, 8.0]))
            .style(theme::ghost),
    ]
    .spacing(8)
    .align_y(Alignment::Center);

    let mut lines = column![].spacing(2);
    for entry in app.visible_logs() {
        lines = lines.push(
            text(format!("{}  {}", entry.level.short(), entry.text))
                .size(10.0)
                .font(MONO)
                .color(entry.level.color())
                .width(Length::Fill),
        );
    }
    if app.logs.is_empty() {
        lines = lines.push(text("No activity yet.").size(10.0).color(MUTED));
    }

    container(
        column![
            header,
            divider(),
            container(scroll(lines).height(Length::Fill).anchor_bottom())
                .height(Length::Fill)
                .style(theme::plain(WELL)),
        ]
        .spacing(6),
    )
    .width(Length::Fill)
    .height(Length::Fixed(196.0))
    .padding(Padding::from([8.0, 14.0]))
    .style(|_| theme::outlined(PANEL, BORDER, 0.0))
    .into()
}

// ----- Modals -------------------------------------------------------------

fn help_modal<'a>(app: &'a RustBlock) -> El<'a> {
    let sections = [
        ("Reblock (Q2_0)", "Lossless conversion of Prism PQ2_0 files to the upstream Q2_0 layout. Use it when the badge says PQ2_0."),
        ("Q4_K_M", "The balanced choice for a float model that must fit a normal desktop. Q5_K_M keeps more quality; Q3_K_M saves more space."),
        ("F16 / BF16", "Uncompressed formats. Large, but the safest source if you plan to quantize again later."),
        ("Structure / Full verification", "Structure checks the header and offsets. Full decodes every tensor again and is slower."),
        ("Fit badge", "An estimate at 4096 tokens: weights plus KV cache plus a fixed runtime overhead. A warning, not a guarantee."),
        ("Discover badge", "Computed before the download from the published weights plus 512 MiB. The KV cache needs the model header, so it is not included."),
        ("GPU not detected", "When VRAM is unknown the badge uses system RAM and marks VRAM as not detected. RustBlock never assumes a GPU."),
        ("Benchmark", "Run llama-bench on the newest finished model. Install llama.cpp tools to enable it."),
        ("GGUF inspector", "Facts, metadata and the tensor type histogram with byte sizes, straight from the file."),
        ("Shortcuts", "Ctrl+O add files · Ctrl+S start · Ctrl+L log · F1 help"),
    ];
    let mut content = column![
        image(app.wordmark.clone())
            .width(Length::Fixed(26.0 * app.wordmark_aspect))
            .height(Length::Fixed(26.0))
            .content_fit(ContentFit::Contain),
        text("RustBlock help").size(18.0).color(TEXT),
    ]
    .spacing(8);
    for (title, body) in sections {
        content = content.push(
            column![
                text(title).size(11.0).color(STEEL_LIGHT),
                text(body).size(10.0).color(MUTED).width(Length::Fill),
            ]
            .spacing(2),
        );
    }
    let footer = row![
        spacer(),
        button(text("Close").size(10.5))
            .on_press(Message::ToggleHelp)
            .padding(Padding::from([6.0, 16.0]))
            .style(theme::secondary),
    ];
    let card = column![
        container(scroll(content).height(Length::Fill)).height(Length::Fill),
        footer,
    ]
    .spacing(10);

    modal(
        container(card)
            .width(Length::Fixed(600.0))
            .height(Length::Fixed(470.0))
            .padding(18)
            .style(|_| theme::outlined(PANEL, BORDER, 8.0)),
    )
}

fn requantize_modal() -> El<'static> {
    modal(
        container(
            column![
                text("Requantization warning").size(17.0).color(WARN),
                text("These files are already quantized.").size(11.0).color(TEXT),
                text("Converting a quantized model to another quantized format does not restore the weights lost by the first quantization. The result can only be worse and may be noticeably degraded.")
                    .size(10.5)
                    .color(MUTED)
                    .width(Length::Fill),
                text("Use F16/F32 to decompress, or continue only if you have a copy of the original float model.")
                    .size(10.5)
                    .color(MUTED)
                    .width(Length::Fill),
                row![
                    button(text("Cancel").size(11.0).color(INK))
                        .on_press(Message::ConfirmRequantize(false))
                        .padding(Padding::from([7.0, 18.0]))
                        .style(theme::secondary),
                    button(text("Requantize anyway").size(11.0).color(INK))
                        .on_press(Message::ConfirmRequantize(true))
                        .padding(Padding::from([7.0, 18.0]))
                        .style(theme::primary),
                ]
                .spacing(8),
            ]
            .spacing(10),
        )
        .width(Length::Fixed(480.0))
        .padding(20)
        .style(|_| theme::outlined(PANEL, BORDER, 8.0)),
    )
}

fn modal(card: impl Into<El<'static>>) -> El<'static> {
    container(card.into())
        .width(Length::Fill)
        .height(Length::Fill)
        .center(Length::Fill)
        .style(theme::scrim())
        .into()
}

// ----- Small pieces -------------------------------------------------------

fn section_label(label: &'static str) -> El<'static> {
    text(label).size(9.0).font(MONO).color(STEEL_LIGHT).into()
}

fn divider() -> El<'static> {
    container(spacer())
        .width(Length::Fill)
        .height(Length::Fixed(1.0))
        .style(theme::plain(BORDER))
        .into()
}

fn chip(label: impl Into<String>, fill: Color, color: Color) -> El<'static> {
    container(text(label.into()).size(8.5).font(MONO).color(color))
        .padding(Padding::from([2.0, 7.0]))
        .style(move |_| theme::outlined(fill, fill, 4.0))
        .into()
}

fn chip_text(label: String, color: Color) -> El<'static> {
    text(label).size(9.0).font(MONO).color(color).into()
}

fn verify_label(kind: VerifyKind) -> &'static str {
    match kind {
        VerifyKind::Exact => "exact",
        VerifyKind::Structure => "structure",
    }
}

pub fn fit_badge(fit: &FitEstimate) -> (String, Color, String) {
    let hint = fit_summary(fit);
    let (label, color) = match (fit.fits_ram, fit.fits_vram) {
        (_, Some(true)) => ("Fits VRAM".to_string(), OK),
        (Some(true), _) => ("Fits RAM".to_string(), OK),
        (Some(false), Some(false)) => ("No fit".to_string(), ERROR),
        (Some(false), None) => ("No RAM fit".to_string(), ERROR),
        (None, Some(false)) => ("No VRAM fit".to_string(), ERROR),
        _ => ("Fit unknown".to_string(), MUTED),
    };
    (
        label,
        color,
        format!(
            "At {} tokens. {hint}. Estimated from the GGUF header; a warning, not a guarantee.",
            fit.context
        ),
    )
}

pub fn fit_summary(fit: &FitEstimate) -> String {
    format!(
        "weights {} · kv {} · overhead {} · total {}",
        human_bytes(fit.weights_bytes),
        human_bytes(fit.kv_cache_bytes),
        human_bytes(fit.overhead_bytes),
        human_bytes(fit.total_bytes)
    )
}

pub fn target_caption(target: Target) -> &'static str {
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
mod tests {
    use super::*;

    #[test]
    fn every_tab_builds_its_view() {
        let mut app = RustBlock::empty();
        for tab in [Tab::Convert, Tab::Discover, Tab::Inspector] {
            app.tab = tab;
            let _ = view(&app);
        }
        app.show_log = true;
        let _ = view(&app);
        app.show_help = true;
        let _ = view(&app);
        app.confirm_requantize = Some(crate::worker::WorkKind::Convert);
        let _ = view(&app);
    }

    #[test]
    fn every_log_level_state_builds() {
        let mut app = RustBlock::empty();
        app.show_log = true;
        for level in 0..4 {
            app.log_levels[level] = false;
            let _ = view(&app);
            app.log_levels[level] = true;
        }
        app.log_filter = "nothing".to_string();
        let _ = view(&app);
    }

    #[test]
    fn job_and_discover_states_build() {
        let mut app = RustBlock::empty();
        app.hf_state = HfState::Error("offline".to_string());
        let _ = view(&app);
        app.hf_state = HfState::Searching;
        let _ = view(&app);
        app.hf_state = HfState::Idle;
        let _ = view(&app);
        app.hf_details_error = Some("bad link".to_string());
        let _ = view(&app);
    }

    #[test]
    fn target_captions_cover_every_preset() {
        for target in Target::ALL {
            assert!(!target_caption(target).is_empty());
        }
    }
}
