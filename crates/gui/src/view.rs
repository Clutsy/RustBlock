//! Everything the user sees: chrome, queue, Discover, inspector, log and modals.
//!
//! The view is a pure function of [`RustBlock`]; every interaction is a
//! [`Message`] handled in [`crate::app`].

use iced::widget::canvas::{Frame, Path, Stroke};
use iced::widget::{
    button, canvas, checkbox, column, container, image, progress_bar, row, scrollable, space,
    stack, text, text_input, MouseArea,
};
use iced::{
    mouse, Alignment, Border, Color, ContentFit, Element, Font, Length, Padding, Point, Rectangle,
    Size,
};

use crate::app::{
    detected_hardware, format_count, short_date, size_fit_badge, DownloadUi, HfState, LogLevel,
    Message, RustBlock, Tab, UpdateState,
};
use crate::deploy::Runtime;
use crate::hf::{self, HfFile, HfModel, HfSort};
use crate::model::{
    human_bytes, human_duration, human_speed, looks_quantized_name, shorten, DownloadState,
    InspectorBlock, InspectorData, JobStatus, OutputMode, Settings, Source, VerifyKind,
};
use crate::theme::{
    self, ACCENT, ACCENT_DARK, BORDER, ERROR, INFO, INK, MUTED, OK, PANEL, PANEL_2, PANEL_3, STEEL,
    STEEL_LIGHT, TEXT, WARN, WELL,
};
use rustblock_core::{conversion_workers, imatrix::find_tool, FitEstimate, Target, VerifyMode};

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
    if app.show_settings {
        layers.push(settings_modal(app));
    }
    if app.deploy.is_some() {
        layers.push(deploy_modal(app));
    }
    if app.confirm_close {
        layers.push(confirm_close_modal());
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
        Tab::Downloads => downloads_page(app),
        Tab::Inspector => inspector_page(app),
    }
}

// ----- Chrome -------------------------------------------------------------

fn chrome(app: &RustBlock) -> El<'_> {
    let brand = MouseArea::new(
        image(app.wordmark.clone())
            .height(Length::Fixed(26.0))
            .content_fit(ContentFit::Contain),
    )
    .on_press(Message::DragWindow)
    .on_double_click(Message::ToggleMaximize);

    let tabs = row![
        tab_button(app, Tab::Convert),
        tab_button(app, Tab::Discover),
        tab_button(app, Tab::Downloads),
        tab_button(app, Tab::Inspector),
    ]
    .spacing(3);

    let mut right = row![].spacing(2).align_y(Alignment::Center);
    if let UpdateState::Available(info) = &app.update.state {
        right = right.push(
            button(
                text(format!("UPDATE {}", info.version))
                    .size(8.5)
                    .font(MONO)
                    .color(INK),
            )
            .on_press(Message::ToggleSettings)
            .padding(Padding::from([5.0, 9.0]))
            .style(theme::primary),
        );
    }
    right = right
        .push(chrome_icon_button(
            NavIcon::Log,
            "LOG",
            Message::ToggleLog,
            app.show_log,
        ))
        .push(chrome_icon_button(
            NavIcon::Settings,
            "SETTINGS",
            Message::ToggleSettings,
            app.show_settings,
        ))
        .push(chrome_icon_button(
            NavIcon::Help,
            "HELP",
            Message::ToggleHelp,
            app.show_help,
        ))
        .push(window_button(WindowIcon::Minimize, Message::Minimize))
        .push(window_button(
            if app.maximized {
                WindowIcon::Restore
            } else {
                WindowIcon::Maximize
            },
            Message::ToggleMaximize,
        ))
        .push(window_button(WindowIcon::Close, Message::CloseWindow));

    container(
        row![brand, spacer(), tabs, spacer(), status_pill(app), right,]
            .padding(Padding::from([0.0, 10.0]))
            .spacing(10)
            .align_y(Alignment::Center),
    )
    .width(Length::Fill)
    .height(Length::Fixed(52.0))
    .align_y(Alignment::Center)
    .style(theme::chrome())
    .into()
}

fn tab_button(app: &RustBlock, tab: Tab) -> El<'static> {
    let active = app.tab == tab;
    button(
        row![
            nav_glyph(tab_icon(tab), 13.0, if active { INK } else { MUTED }),
            text(tab.label()).size(11.0),
        ]
        .spacing(6)
        .align_y(Alignment::Center),
    )
    .on_press(Message::TabSelected(tab))
    .padding(Padding::from([6.0, 8.0]))
    .width(Length::Fixed(100.0))
    .height(Length::Fixed(30.0))
    .style(theme::tab(active))
    .into()
}

fn tab_icon(tab: Tab) -> NavIcon {
    match tab {
        Tab::Convert => NavIcon::Convert,
        Tab::Discover => NavIcon::Discover,
        Tab::Downloads => NavIcon::Downloads,
        Tab::Inspector => NavIcon::Inspector,
    }
}

/// Small chrome button with a drawn icon, used by Settings and Help.
fn chrome_icon_button(
    icon: NavIcon,
    label: &'static str,
    message: Message,
    active: bool,
) -> El<'static> {
    let color = if active { INK } else { STEEL_LIGHT };
    button(
        row![
            nav_glyph(icon, 13.0, color),
            text(label).size(9.0).font(MONO).color(color),
        ]
        .spacing(4)
        .align_y(Alignment::Center),
    )
    .on_press(message)
    .padding(Padding::from([5.0, 8.0]))
    .height(Length::Fixed(26.0))
    .style(move |style_theme, status| {
        if active {
            theme::primary(style_theme, status)
        } else {
            theme::secondary(style_theme, status)
        }
    })
    .into()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NavIcon {
    Convert,
    Discover,
    Downloads,
    Inspector,
    Settings,
    Help,
    Log,
}

fn nav_glyph(icon: NavIcon, size: f32, color: Color) -> El<'static> {
    canvas(NavGlyph { icon, color })
        .width(Length::Fixed(size))
        .height(Length::Fixed(size))
        .into()
}

struct NavGlyph {
    icon: NavIcon,
    color: Color,
}

impl canvas::Program<Message> for NavGlyph {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &iced::Renderer,
        _theme: &iced::Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        let mut frame = Frame::new(renderer, bounds.size());
        let center = frame.center();
        let color = self.color;
        let stroke = Stroke::default().with_color(color).with_width(1.2);
        let point = |x: f32, y: f32| Point::new(center.x + x, center.y + y);
        match self.icon {
            NavIcon::Convert => {
                frame.fill_rectangle(point(-5.0, -4.5), Size::new(10.0, 2.0), color);
                frame.fill_rectangle(point(-5.0, -1.5), Size::new(7.0, 2.0), color);
                frame.fill_rectangle(point(-5.0, 1.5), Size::new(4.0, 2.0), color);
            }
            NavIcon::Discover => {
                frame.stroke(&Path::circle(point(-1.0, -1.0), 4.0), stroke);
                frame.stroke(
                    &Path::line(point(2.0, 2.0), point(5.5, 5.5)),
                    stroke.with_width(1.6),
                );
            }
            NavIcon::Downloads => {
                frame.stroke(&Path::line(point(0.0, -5.0), point(0.0, 2.0)), stroke);
                frame.stroke(&Path::line(point(-3.5, -1.5), point(0.0, 2.0)), stroke);
                frame.stroke(&Path::line(point(3.5, -1.5), point(0.0, 2.0)), stroke);
                frame.stroke(
                    &Path::line(point(-5.0, 5.0), point(5.0, 5.0)),
                    stroke.with_width(1.6),
                );
            }
            NavIcon::Inspector => {
                for (x, y) in [(-5.0, -5.0), (1.0, -5.0), (-5.0, 1.0), (1.0, 1.0)] {
                    frame.stroke_rectangle(point(x, y), Size::new(4.0, 4.0), stroke);
                }
            }
            NavIcon::Settings => {
                frame.stroke(&Path::circle(center, 3.8), stroke);
                frame.stroke(&Path::circle(center, 1.5), stroke);
                for step in 0..8 {
                    let angle = step as f32 * std::f32::consts::PI / 4.0;
                    let inner = 4.6;
                    let outer = 6.2;
                    frame.stroke(
                        &Path::line(
                            point(angle.cos() * inner, angle.sin() * inner),
                            point(angle.cos() * outer, angle.sin() * outer),
                        ),
                        stroke.with_width(1.6),
                    );
                }
            }
            NavIcon::Log => {
                for (line, width) in [10.0_f32, 7.0, 10.0, 5.0].iter().enumerate() {
                    frame.fill_rectangle(
                        point(-5.0, -4.5 + line as f32 * 3.0),
                        Size::new(*width, 1.6),
                        color,
                    );
                }
            }
            NavIcon::Help => {
                frame.stroke(&Path::circle(center, 6.0), stroke);
                let arc = Path::new(|builder| {
                    let center_y = -1.5;
                    let radius = 2.2;
                    let start = 200.0_f32.to_radians();
                    builder.move_to(point(start.cos() * radius, center_y + start.sin() * radius));
                    for step in 1..=12 {
                        let angle = (200.0 + step as f32 * 11.0).to_radians();
                        builder
                            .line_to(point(angle.cos() * radius, center_y + angle.sin() * radius));
                    }
                    builder.line_to(point(0.0, 1.4));
                });
                frame.stroke(&arc, stroke);
                frame.fill(&Path::circle(point(0.0, 4.0), 0.9), color);
            }
        }
        vec![frame.into_geometry()]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WindowIcon {
    Minimize,
    Maximize,
    Restore,
    Close,
}

impl WindowIcon {
    fn danger(self) -> bool {
        self == WindowIcon::Close
    }
}

/// Crisp window controls drawn with canvas: font glyphs for minimize,
/// maximize and close never line up across fonts and render blurry.
fn window_button(icon: WindowIcon, message: Message) -> El<'static> {
    button(
        canvas(WindowGlyph {
            icon,
            color: if icon.danger() { STEEL_LIGHT } else { MUTED },
        })
        .width(Length::Fixed(34.0))
        .height(Length::Fixed(26.0)),
    )
    .on_press(message)
    .width(Length::Fixed(34.0))
    .height(Length::Fixed(26.0))
    .padding(0)
    .style(theme::window_control(icon.danger()))
    .into()
}

struct WindowGlyph {
    icon: WindowIcon,
    color: Color,
}

impl canvas::Program<Message> for WindowGlyph {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &iced::Renderer,
        _theme: &iced::Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        let mut frame = Frame::new(renderer, bounds.size());
        let center = frame.center();
        let color = self.color;
        let stroke = Stroke::default().with_color(color).with_width(1.0);
        match self.icon {
            WindowIcon::Minimize => {
                frame.fill_rectangle(
                    Point::new(center.x - 6.0, center.y + 0.5),
                    Size::new(12.0, 1.0),
                    color,
                );
            }
            WindowIcon::Maximize => {
                frame.stroke_rectangle(
                    Point::new(center.x - 5.5, center.y - 4.5),
                    Size::new(11.0, 9.0),
                    stroke,
                );
            }
            WindowIcon::Restore => {
                frame.stroke_rectangle(
                    Point::new(center.x - 6.5, center.y - 3.0),
                    Size::new(9.0, 7.0),
                    stroke,
                );
                frame.fill_rectangle(
                    Point::new(center.x - 4.0, center.y - 5.0),
                    Size::new(9.0, 1.0),
                    color,
                );
                frame.fill_rectangle(
                    Point::new(center.x + 4.0, center.y - 5.0),
                    Size::new(1.0, 7.0),
                    color,
                );
            }
            WindowIcon::Close => {
                for step in 0..10 {
                    let offset = step as f32;
                    frame.fill_rectangle(
                        Point::new(center.x - 5.0 + offset, center.y - 5.0 + offset),
                        Size::new(1.0, 1.0),
                        color,
                    );
                    frame.fill_rectangle(
                        Point::new(center.x + 4.0 - offset, center.y - 5.0 + offset),
                        Size::new(1.0, 1.0),
                        color,
                    );
                }
            }
        }
        vec![frame.into_geometry()]
    }
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
            .on_press_maybe(app.benchmark.is_none().then_some(Message::Benchmark))
            .padding(Padding::from([6.0, 12.0]))
            .height(Length::Fixed(32.0))
            .style(theme::secondary),
        button(text("Add files").size(12.0).color(INK))
            .on_press(Message::Browse)
            .padding(Padding::from([6.0, 16.0]))
            .height(Length::Fixed(34.0))
            .style(theme::primary),
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
    if let Some(bench) = &app.benchmark {
        return container(
            row![
                text(format!(
                    "Running llama-bench on {} · {} CPU threads",
                    shorten(&bench.model.display().to_string(), 52),
                    conversion_workers()
                ))
                .size(10.0)
                .color(WARN),
                spacer(),
                text("The result will appear here.").size(9.0).color(MUTED),
            ]
            .spacing(8)
            .align_y(Alignment::Center),
        )
        .padding(Padding::from([8.0, 12.0]))
        .width(Length::Fill)
        .style(|_| theme::outlined(PANEL_2, BORDER, 6.0))
        .into();
    }
    let Some((ok, output)) = &app.benchmark_result else {
        return spacer().height(Length::Fixed(0.0)).into();
    };
    let preview = output.lines().take(12).collect::<Vec<_>>().join("\n");
    let border = if *ok { BORDER } else { ERROR };
    container(
        column![
            row![
                text(if *ok {
                    "Benchmark complete"
                } else {
                    "Benchmark failed"
                })
                .size(10.5)
                .color(if *ok { OK } else { ERROR }),
                spacer(),
                button(text("Show log").size(9.5))
                    .on_press(Message::ToggleLog)
                    .padding(Padding::from([4.0, 10.0]))
                    .style(theme::secondary),
                button(text("Dismiss").size(9.5).color(MUTED))
                    .on_press(Message::DismissBenchmark)
                    .padding(Padding::from([4.0, 8.0]))
                    .style(theme::ghost),
            ]
            .spacing(6)
            .align_y(Alignment::Center),
            text(preview)
                .size(9.0)
                .font(MONO)
                .color(if *ok { STEEL_LIGHT } else { ERROR }),
        ]
        .spacing(6),
    )
    .padding(10)
    .width(Length::Fill)
    .style(move |_| theme::outlined(PANEL_2, border, 6.0))
    .into()
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
                text(format!("{} threads", conversion_workers()))
                    .size(9.0)
                    .font(MONO)
                    .color(STEEL_LIGHT),
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
                button(text("Browse files").size(12.5).color(INK))
                    .on_press(Message::Browse)
                    .padding(Padding::from([9.0, 20.0]))
                    .height(Length::Fixed(38.0))
                    .style(theme::primary),
                button(text("Discover models").size(12.5).color(ACCENT))
                    .on_press(Message::TabSelected(Tab::Discover))
                    .padding(Padding::from([9.0, 18.0]))
                    .height(Length::Fixed(38.0))
                    .style(theme::accent_outline),
            ]
            .spacing(10),
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

    if let Some(output) = job.done_output() {
        let mut targets = row![].spacing(4).align_y(Alignment::Center);
        let mut detected = 0;
        for runtime in Runtime::ALL {
            let (ready, _) = runtime_ready(runtime, &app.settings);
            if ready {
                detected += 1;
                targets = targets.push(
                    button(
                        text(format!("Copy to {}", runtime.label()))
                            .size(9.5)
                            .color(INK),
                    )
                    .on_press(Message::DeployNow(output.clone(), runtime))
                    .padding(Padding::from([4.0, 10.0]))
                    .style(theme::primary),
                );
            }
        }
        if detected > 0 {
            body = body.push(
                row![text("COPY TO").size(8.5).font(MONO).color(MUTED), targets,]
                    .spacing(6)
                    .align_y(Alignment::Center),
            );
        } else {
            body = body.push(
                text("No runtime folder detected; set one in Settings to copy the finished model.")
                    .size(9.0)
                    .color(MUTED),
            );
        }
    }

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
                .on_press(Message::OpenPath(output.clone()))
                .padding(Padding::from([4.0, 10.0]))
                .style(theme::secondary),
        );
        buttons = buttons.push(
            button(text("Inspect").size(10.0))
                .on_press(Message::InspectPath(output.clone()))
                .padding(Padding::from([4.0, 10.0]))
                .style(theme::ghost),
        );
        buttons = buttons.push(
            button(text("Test in runtime").size(10.0).color(INK))
                .on_press(Message::Deploy(output))
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
            button(text("Suggest a format").size(9.0).color(STEEL_LIGHT))
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

    section = section.push(target_grid(
        &standard,
        app.settings.target,
        Message::SetTarget,
    ));
    if !ternary.is_empty() {
        section = section.push(
            row![
                text("Ternary").size(11.0).color(TEXT),
                text("only for ternary models").size(9.5).color(WARN),
            ]
            .spacing(8),
        );
        section = section.push(target_grid(
            &ternary,
            app.settings.target,
            Message::SetTarget,
        ));
    }
    section.into()
}

fn target_grid(
    targets: &[Target],
    selected: Target,
    message: impl Fn(Target) -> Message + 'static,
) -> El<'static> {
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
                .on_press(message(*target))
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

    row![main, repository_rail(app)]
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

fn discover_header(app: &RustBlock) -> El<'static> {
    let mut sources = row![].spacing(3);
    for (source, label) in [
        (Source::HuggingFace, "Hugging Face"),
        (Source::ModelScope, "ModelScope"),
    ] {
        sources = sources.push(
            button(text(label).size(10.0))
                .on_press(Message::SetSource(source))
                .padding(Padding::from([5.0, 10.0]))
                .width(Length::Fixed(118.0))
                .height(Length::Fixed(28.0))
                .style(theme::segment(app.settings.source == source)),
        );
    }

    column![
        row![
            text("Discover").size(26.0).color(TEXT),
            spacer(),
            button(text("Trending").size(10.5).color(ACCENT))
                .on_press_maybe((!app.hf_busy()).then_some(Message::Trending))
                .padding(Padding::from([6.0, 14.0]))
                .height(Length::Fixed(30.0))
                .style(theme::accent_outline),
        ]
        .align_y(Alignment::Center),
        row![
            text("GGUF repositories from Hugging Face or ModelScope, with a fit verdict before the download.")
                .size(10.5)
                .color(MUTED)
                .width(Length::Fill),
            sources,
        ]
        .spacing(12)
        .align_y(Alignment::Center),
    ]
    .spacing(6)
    .into()
}

fn search_bar(app: &RustBlock) -> El<'_> {
    let busy = app.hf_busy();
    let mut sorts = row![].spacing(3);
    if app.settings.source == Source::HuggingFace {
        for (sort, label) in [
            (HfSort::Downloads, "Downloads"),
            (HfSort::Likes, "Likes"),
            (HfSort::Recent, "Recent"),
        ] {
            sorts = sorts.push(
                button(text(label).size(9.5))
                    .on_press(Message::SetSort(sort))
                    .padding(Padding::from([5.0, 10.0]))
                    .height(Length::Fixed(26.0))
                    .style(theme::segment(app.hf_sort == sort)),
            );
        }
    }

    let placeholder = match app.settings.source {
        Source::HuggingFace => "Search GGUF repositories or paste a huggingface.co link",
        Source::ModelScope => "Paste a modelscope.cn/models/owner/name link",
    };

    row![
        text_input(placeholder, &app.hf_query)
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
        .height(Length::Fixed(38.0))
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
            button(text(query).size(10.5).color(STEEL_LIGHT))
                .on_press(Message::QuickSearch(query.to_string()))
                .padding(Padding::from([5.0, 12.0]))
                .style(theme::accent_outline),
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

fn repository_rail(app: &RustBlock) -> El<'_> {
    let body = column![section_label("REPOSITORY"), repo_panel(app)].spacing(8);
    let body = container(body.width(Length::Fill)).padding(Padding::from([0.0, 10.0]));
    container(scroll(body).height(Length::Fill))
        .width(Length::Fixed(372.0))
        .height(Length::Fill)
        .padding(Padding::from([16.0, 16.0]))
        .style(|_| theme::outlined(PANEL, BORDER, 0.0))
        .into()
}

fn downloads_page(app: &RustBlock) -> El<'_> {
    let active = app.active_downloads();
    let finished = app
        .downloads
        .iter()
        .filter(|download| download.state.is_finished())
        .count();
    let header = row![
        column![
            row![
                nav_glyph(NavIcon::Downloads, 22.0, ACCENT),
                text("Downloads").size(26.0).color(TEXT),
            ]
            .spacing(9)
            .align_y(Alignment::Center),
            text(format!(
                "{active} active · {finished} finished · {}",
                shorten(&app.settings.download_dir.display().to_string(), 58)
            ))
            .size(10.5)
            .color(MUTED),
        ]
        .spacing(3),
        spacer(),
        button(text("Clean partials").size(9.5).color(MUTED))
            .on_press(Message::ClearPartials)
            .padding(Padding::from([5.0, 10.0]))
            .style(theme::ghost),
        button(text("Open folder").size(10.5))
            .on_press(Message::OpenPath(app.settings.download_dir.clone()))
            .padding(Padding::from([6.0, 12.0]))
            .style(theme::secondary),
        button(text("Clear finished").size(10.0).color(MUTED))
            .on_press_maybe((!app.downloads.is_empty()).then_some(Message::ClearFinishedDownloads))
            .padding(Padding::from([6.0, 10.0]))
            .style(theme::ghost),
    ]
    .spacing(8)
    .align_y(Alignment::Center);

    let body: El<'_> = if app.downloads.is_empty() {
        container(
            column![
                nav_glyph(NavIcon::Downloads, 34.0, STEEL),
                text("No downloads yet").size(18.0).color(TEXT),
                text("Search the Hub and press Get on a file. Transfers keep running while you convert or inspect a model.")
                    .size(10.5)
                    .color(MUTED),
                row![
                    button(text("Browse models").size(11.5).color(INK))
                        .on_press(Message::TabSelected(Tab::Discover))
                        .padding(Padding::from([8.0, 18.0]))
                        .style(theme::primary),
                    button(text("Open folder").size(11.0))
                        .on_press(Message::OpenPath(app.settings.download_dir.clone()))
                        .padding(Padding::from([8.0, 16.0]))
                        .style(theme::secondary),
                ]
                .spacing(8),
            ]
            .spacing(9)
            .align_x(Alignment::Center),
        )
        .width(Length::Fill)
        .height(Length::Fill)
        .center(Length::Fill)
        .style(|_| theme::outlined(theme::translucent(PANEL, 0.66), BORDER, 8.0))
        .into()
    } else {
        let mut list = column![].spacing(8);
        for download in &app.downloads {
            list = list.push(download_card(download));
        }
        if app.settings.verify_downloads {
            list = list.push(
                text("SHA-256 verification reads the whole file once, so a large model can pause at 100% while it is checked. Turn it off in Settings → Downloads.")
                    .size(9.0)
                    .color(MUTED),
            );
        }
        scroll(list).height(Length::Fill).into()
    };

    column![header, body]
        .spacing(12)
        .padding(Padding::from([18.0, 20.0]))
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

fn repo_panel(app: &RustBlock) -> El<'_> {
    let Some(model) = app.current_model() else {
        return container(
            column![
                text("Pick a repository to list its GGUF files.")
                    .size(10.5)
                    .color(MUTED),
                text(format!(
                    "Downloads are resumable and SHA-256 checked when the source publishes a hash. Destination: {}",
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
            button(text("Open ↗").size(9.5))
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
        button(text("Add").size(9.5).color(ACCENT))
            .on_press(Message::AddCached(path))
            .padding(Padding::from([5.0, 10.0]))
            .style(theme::accent_outline)
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

fn download_card(download: &DownloadUi) -> El<'_> {
    let frac = download.fraction();
    let (status, color) = match download.state {
        DownloadState::Queued => ("Queued".to_string(), MUTED),
        DownloadState::Running => {
            let label = if download.verifying {
                "Verifying SHA-256"
            } else if download.projector {
                "Vision projector"
            } else {
                "Downloading"
            };
            (
                label.to_string(),
                if download.verifying { INFO } else { ACCENT },
            )
        }
        DownloadState::Done => ("Complete".to_string(), OK),
        DownloadState::Failed => ("Failed".to_string(), ERROR),
        DownloadState::Cancelled => ("Cancelled".to_string(), WARN),
    };

    let detail = match download.state {
        DownloadState::Queued => format!(
            "{} · waiting for the current download",
            human_bytes(download.total)
        ),
        DownloadState::Running if download.verifying => format!(
            "{} / {} · {:.0}% · one full read",
            human_bytes(download.done),
            human_bytes(download.total),
            frac * 100.0
        ),
        DownloadState::Running => format!(
            "{} / {} · {:.1}% · {}",
            human_bytes(download.done),
            if download.total > 0 {
                human_bytes(download.total)
            } else {
                "?".to_string()
            },
            frac * 100.0,
            human_speed(download.speed)
        ),
        DownloadState::Done => format!(
            "{} · {}",
            human_bytes(download.total),
            shorten(
                &download
                    .path
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_default(),
                38
            )
        ),
        DownloadState::Failed | DownloadState::Cancelled => download
            .error
            .clone()
            .unwrap_or_else(|| "Stopped.".to_string()),
    };

    let mut content = column![
        row![
            text(shorten(&download.file_name(), 30))
                .size(10.0)
                .font(MONO)
                .color(STEEL_LIGHT)
                .width(Length::Fill),
            text(status).size(9.5).color(color),
        ]
        .align_y(Alignment::Center),
        row![
            chip(download.source.label(), PANEL_3, MUTED),
            text(shorten(&download.repo, 30))
                .size(8.5)
                .font(MONO)
                .color(MUTED),
        ]
        .spacing(6)
        .align_y(Alignment::Center),
        progress_bar(0.0..=1.0, frac)
            .girth(5.0)
            .style(theme::progress),
        text(detail)
            .size(8.5)
            .font(MONO)
            .color(if download.state == DownloadState::Failed {
                ERROR
            } else {
                MUTED
            }),
    ]
    .spacing(5);

    let mut actions = row![].spacing(4);
    match download.state {
        DownloadState::Queued | DownloadState::Running => {
            actions = actions.push(
                button(text("Cancel").size(9.0).color(MUTED))
                    .on_press(Message::CancelDownload(download.id))
                    .padding(Padding::from([2.0, 6.0]))
                    .style(theme::ghost),
            );
        }
        DownloadState::Done => {
            if let Some(path) = &download.path {
                if !download.projector {
                    actions = actions.push(
                        button(text("Add").size(9.0))
                            .on_press(Message::AddCached(path.clone()))
                            .padding(Padding::from([2.0, 8.0]))
                            .style(theme::secondary),
                    );
                    actions = actions.push(
                        button(text("Deploy").size(9.0))
                            .on_press(Message::Deploy(path.clone()))
                            .padding(Padding::from([2.0, 8.0]))
                            .style(theme::secondary),
                    );
                }
                actions = actions.push(
                    button(text("Inspect").size(9.0).color(MUTED))
                        .on_press(Message::InspectPath(path.clone()))
                        .padding(Padding::from([2.0, 6.0]))
                        .style(theme::ghost),
                );
                actions = actions.push(
                    button(text("Open").size(9.0).color(MUTED))
                        .on_press(Message::OpenPath(path.clone()))
                        .padding(Padding::from([2.0, 6.0]))
                        .style(theme::ghost),
                );
            }
        }
        DownloadState::Failed | DownloadState::Cancelled => {
            actions = actions.push(
                button(text("Dismiss").size(9.0).color(MUTED))
                    .on_press(Message::DismissDownload(download.id))
                    .padding(Padding::from([2.0, 6.0]))
                    .style(theme::ghost),
            );
        }
    }
    content = content.push(actions);

    if download.quantized {
        content = content.push(
            text("Already quantized · converting it again will not restore the original weights.")
                .size(8.5)
                .color(WARN),
        );
    }

    container(content)
        .padding(9)
        .width(Length::Fill)
        .style(|_| theme::outlined(PANEL_2, BORDER, 6.0))
        .into()
}

// ----- Inspector ----------------------------------------------------------

fn inspector_page(app: &RustBlock) -> El<'_> {
    let mut header = row![
        column![
            text("Inspector").size(26.0).color(TEXT),
            text("Every block in the file, one at a time, with a per-block type editor.")
                .size(10.5)
                .color(MUTED),
        ]
        .spacing(3),
        spacer(),
        text_input("Path to a .gguf file", &app.inspector_input)
            .on_input(Message::InspectorPathChanged)
            .on_submit(Message::InspectorLoad)
            .size(10.0)
            .padding(Padding::from([7.0, 9.0]))
            .width(Length::Fixed(270.0))
            .style(theme::text_field),
        button(text("Inspect").size(10.5))
            .on_press(Message::InspectorLoad)
            .padding(Padding::from([6.0, 12.0]))
            .style(theme::secondary),
        button(text("Refresh").size(10.5))
            .on_press(Message::InspectorRefresh)
            .padding(Padding::from([6.0, 12.0]))
            .style(theme::ghost),
        button(text("Open file").size(11.0).color(INK))
            .on_press(Message::InspectorOpen)
            .padding(Padding::from([7.0, 14.0]))
            .height(Length::Fixed(34.0))
            .style(theme::primary),
    ]
    .align_y(Alignment::Center);

    header = header.push(spacer());

    let body: El<'_> = match &app.inspector {
        None => inspector_empty(app),
        Some(data) => inspector_loaded(app, data),
    };

    column![header, body]
        .spacing(12)
        .padding(Padding::from([18.0, 20.0]))
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

fn inspector_empty(app: &RustBlock) -> El<'_> {
    container(
        column![
            text("Open a GGUF").size(17.0).color(TEXT),
            text("Drop a .gguf anywhere in this window, pick one, or paste its path: every block is listed with its type, shape and size, and each one can be re-encoded on its own.")
                .size(10.5)
                .color(MUTED),
            match &app.inspector_error {
                Some(error) => text(error.as_str()).size(10.0).color(ERROR),
                None => text("Dropping a model while this tab is open inspects it instead of queueing it.")
                    .size(9.5)
                    .color(STEEL_LIGHT),
            },
        ]
        .spacing(6)
        .align_x(Alignment::Center),
    )
    .width(Length::Fill)
    .height(Length::Fill)
    .center(Length::Fill)
    .style(|_| theme::outlined(theme::translucent(PANEL, 0.66), BORDER, 8.0))
    .into()
}

fn inspector_loaded<'a>(app: &'a RustBlock, data: &'a InspectorData) -> El<'a> {
    let visible: Vec<usize> = data
        .blocks
        .iter()
        .enumerate()
        .filter(|(_, block)| block.matches(&app.inspector_filter))
        .map(|(index, _)| index)
        .collect();

    let list: El<'a> = if visible.is_empty() {
        container(
            text("No block matches this filter.")
                .size(10.0)
                .color(MUTED),
        )
        .padding(12)
        .width(Length::Fill)
        .into()
    } else {
        let mut rows = column![].spacing(2);
        for &index in &visible {
            rows = rows.push(block_row(app, data, index));
        }
        scroll(rows).height(Length::Fill).into()
    };

    let left = container(
        column![
            row![
                section_label("BLOCKS"),
                spacer(),
                text(format!("{} / {}", visible.len(), data.blocks.len()))
                    .size(9.0)
                    .font(MONO)
                    .color(MUTED),
            ]
            .align_y(Alignment::Center),
            text_input("Filter by name or type", &app.inspector_filter)
                .on_input(Message::InspectorFilterChanged)
                .size(9.5)
                .padding(Padding::from([6.0, 8.0]))
                .style(theme::text_field),
            list,
        ]
        .spacing(8)
        .height(Length::Fill),
    )
    .width(Length::Fill)
    .height(Length::Fill)
    .padding(10)
    .style(|_| theme::outlined(PANEL, BORDER, 8.0));

    let right = container(scroll(inspector_details(app, data)).height(Length::Fill))
        .width(Length::Fixed(444.0))
        .height(Length::Fill)
        .padding(12)
        .style(|_| theme::outlined(PANEL, BORDER, 8.0));

    row![left, right].spacing(12).height(Length::Fill).into()
}

fn block_row<'a>(app: &'a RustBlock, data: &'a InspectorData, index: usize) -> El<'a> {
    let block = &data.blocks[index];
    let selected = app.inspector_selected == Some(index);
    button(
        row![
            text(format!("{index:>4}"))
                .size(8.5)
                .font(MONO)
                .color(STEEL),
            text(shorten(&block.name, 54))
                .size(9.5)
                .font(MONO)
                .color(if selected { TEXT } else { STEEL_LIGHT })
                .width(Length::Fill),
            text(block.type_name).size(8.5).font(MONO).color(MUTED),
            text(human_bytes(block.bytes))
                .size(8.5)
                .font(MONO)
                .color(MUTED)
                .width(Length::Fixed(62.0)),
        ]
        .spacing(8)
        .align_y(Alignment::Center),
    )
    .on_press(Message::InspectorSelect(index))
    .width(Length::Fill)
    .padding(Padding::from([5.0, 7.0]))
    .style(theme::list_row(selected))
    .into()
}

fn inspector_details<'a>(app: &'a RustBlock, data: &'a InspectorData) -> El<'a> {
    let mut body = column![].spacing(8).width(Length::Fill);

    let selected = app
        .inspector_selected
        .and_then(|index| data.blocks.get(index).map(|block| (index, block)));
    match selected {
        Some((index, block)) => body = body.push(block_editor(app, block, index)),
        None => {
            body = body.push(
                column![
                    text("No block selected").size(13.0).color(TEXT),
                    text("Pick a block from the list to see its layout and re-encode it.")
                        .size(9.5)
                        .color(MUTED),
                ]
                .spacing(4),
            );
        }
    }

    body = body.push(divider());
    body = body.push(file_summary(data));
    body.into()
}

fn block_editor<'a>(app: &'a RustBlock, block: &'a InspectorBlock, index: usize) -> El<'a> {
    let target = app.inspector_target;
    let mut body = column![
        row![
            text(block.name.as_str())
                .size(10.5)
                .font(MONO)
                .color(TEXT)
                .width(Length::Fill),
            chip(block.type_name, PANEL_3, TEXT),
        ]
        .spacing(6)
        .align_y(Alignment::Center),
        text(format!(
            "block {index} · type id {} · {} values/block · {} bytes/block",
            block.ty, block.block, block.block_bytes
        ))
        .size(8.5)
        .font(MONO)
        .color(MUTED),
    ]
    .spacing(4);

    body = body.push(divider());
    body = body.push(block_facts(block));
    body = body.push(divider());
    body = body.push(section_label("CHANGE TYPE"));
    let targets: Vec<Target> = Target::ALL
        .iter()
        .copied()
        .filter(|target| *target != Target::Reblock)
        .collect();
    body = body.push(target_grid(&targets, target, Message::InspectorEditTarget));
    body = body.push(text(target_caption(target)).size(9.0).color(MUTED));
    body = body.push(
        text(format!("{} → {}", block.type_name, target.short()))
            .size(10.0)
            .font(MONO)
            .color(STEEL_LIGHT),
    );
    body = body.push(
        text_input("Path for the edited copy", &app.inspector_output)
            .on_input(Message::InspectorEditOutput)
            .size(9.5)
            .padding(Padding::from([6.0, 8.0]))
            .style(theme::text_field),
    );
    let encodable = rustblock_core::source_supported(block.ty);
    if !encodable {
        body = body.push(
            text("This block's source type has no native decoder in this build, so it cannot be re-encoded. It is still listed and copied by other conversions.")
                .size(9.0)
                .color(ERROR),
        );
    }
    let unchanged = target.type_id() == Some(block.ty);
    body = body.push(
        text(if unchanged {
            "This is already the block's type: the copy will be byte-identical."
        } else {
            "Only this block is re-encoded. Every other tensor is copied byte-for-byte; metadata, names and shapes are untouched. Re-quantizing an already quantized block loses precision."
        })
        .size(9.0)
        .color(if unchanged { MUTED } else { WARN }),
    );

    if let Some((done, total, label)) = &app.edit_progress {
        let fraction = if *total > 0 {
            *done as f32 / *total as f32
        } else {
            0.0
        };
        body = body.push(
            column![
                text(label.as_str()).size(9.5).font(MONO).color(STEEL_LIGHT),
                progress_bar(0.0..=1.0, fraction.clamp(0.0, 1.0))
                    .girth(5.0)
                    .style(theme::progress),
                text(format!("{} / {}", human_bytes(*done), human_bytes(*total)))
                    .size(8.5)
                    .font(MONO)
                    .color(MUTED),
            ]
            .spacing(4),
        );
    }
    if let Some(error) = &app.edit_error {
        body = body.push(text(error.as_str()).size(9.5).color(ERROR));
    }
    if let Some(path) = &app.edit_done {
        body = body.push(
            row![
                text(format!(
                    "Edited copy: {}",
                    shorten(&path.display().to_string(), 42)
                ))
                .size(9.0)
                .font(MONO)
                .color(OK)
                .width(Length::Fill),
                button(text("Inspect it").size(9.0))
                    .on_press(Message::InspectPath(path.clone()))
                    .padding(Padding::from([4.0, 8.0]))
                    .style(theme::secondary),
            ]
            .spacing(6)
            .align_y(Alignment::Center),
        );
    }
    body = body.push(
        button(
            text(if app.edit_busy() {
                "Writing…"
            } else {
                "Write edited copy"
            })
            .size(11.0)
            .color(INK),
        )
        .on_press_maybe((!app.edit_busy() && encodable).then_some(Message::InspectorWriteBlock))
        .width(Length::Fill)
        .padding(Padding::from([8.0, 14.0]))
        .style(theme::primary),
    );
    body.into()
}

fn block_facts(block: &InspectorBlock) -> El<'static> {
    let shape = if block.dims.is_empty() {
        "scalar".to_string()
    } else {
        block
            .dims
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(" × ")
    };
    let mut facts = column![].spacing(3);
    for (label, value) in [
        ("shape", shape),
        ("elements", format_count(block.elements)),
        ("on disk", human_bytes(block.bytes)),
        ("bits / weight", format!("{:.3}", block.bits_per_weight)),
        (
            "native codec",
            if rustblock_core::source_supported(block.ty) {
                "yes".to_string()
            } else {
                "no".to_string()
            },
        ),
        ("data offset", format!("0x{:x}", block.offset)),
        ("share", format!("{:.2}%", block.share * 100.0)),
    ] {
        facts = facts.push(
            row![
                text(label)
                    .size(9.0)
                    .font(MONO)
                    .color(MUTED)
                    .width(Length::Fixed(96.0)),
                text(value)
                    .size(9.5)
                    .font(MONO)
                    .color(TEXT)
                    .width(Length::Fill),
            ]
            .spacing(8),
        );
    }
    facts.into()
}

fn file_summary<'a>(data: &'a InspectorData) -> El<'a> {
    let mut body = column![].spacing(8).width(Length::Fill);
    body = body.push(section_label("FILE"));
    body = body.push(
        text(data.path.display().to_string())
            .size(8.5)
            .font(MONO)
            .color(STEEL_LIGHT),
    );

    let mut facts = column![].spacing(3);
    for (label, value) in [
        ("file", human_bytes(data.file_len)),
        ("version", format!("v{}", data.version)),
        ("tensors", data.tensors.to_string()),
        ("alignment", data.alignment.to_string()),
        ("data start", human_bytes(data.data_start)),
        ("weights", human_bytes(data.weights_bytes)),
        (
            "file type",
            data.file_type
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
        ),
        ("PQ2_0 / PTQ1_0", format!("{} / {}", data.pq2, data.ptq1)),
    ] {
        facts = facts.push(
            row![
                text(label)
                    .size(9.0)
                    .font(MONO)
                    .color(MUTED)
                    .width(Length::Fixed(110.0)),
                text(value)
                    .size(9.5)
                    .font(MONO)
                    .color(TEXT)
                    .width(Length::Fill),
            ]
            .spacing(8),
        );
    }
    body = body.push(facts);

    match &data.fit {
        Some(estimate) => {
            let (label, color, hint) = fit_badge(estimate);
            body = body.push(
                row![
                    chip_text(label, color),
                    text(fit_summary(estimate)).size(9.5).color(MUTED),
                ]
                .spacing(10)
                .align_y(Alignment::Center),
            );
            body = body.push(text(hint).size(9.0).color(MUTED));
        }
        None => body = body.push(text("Fit estimate unavailable.").size(9.5).color(MUTED)),
    }

    let max_bytes = data
        .histogram
        .iter()
        .map(|(_, _, bytes)| *bytes)
        .max()
        .unwrap_or(1);
    body = body.push(divider());
    body = body.push(section_label("TENSOR TYPES"));
    for (name, count, bytes) in &data.histogram {
        let fraction = *bytes as f32 / max_bytes as f32;
        body = body.push(
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

    body = body.push(divider());
    body = body.push(section_label("METADATA"));
    for (key, value) in &data.metadata {
        body = body.push(
            row![
                text(key.as_str())
                    .size(9.0)
                    .font(MONO)
                    .color(MUTED)
                    .width(Length::Fixed(190.0)),
                text(value.as_str())
                    .size(9.5)
                    .color(TEXT)
                    .width(Length::Fill),
            ]
            .spacing(8),
        );
    }
    body.into()
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
        button(text("Copy log").size(9.5))
            .on_press(Message::CopyLog)
            .padding(Padding::from([4.0, 10.0]))
            .style(theme::secondary),
        button(text("Clear").size(9.5))
            .on_press(Message::ClearLog)
            .padding(Padding::from([4.0, 10.0]))
            .style(theme::secondary),
        button(text("Close").size(9.5).color(INK))
            .on_press(Message::ToggleLog)
            .padding(Padding::from([4.0, 12.0]))
            .style(theme::primary),
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
        ("Benchmark", "Run llama-bench on the newest finished model, or on the first queued file. The result appears above the queue; install llama.cpp tools or set LLAMA_BENCH."),
        ("Updates", "RustBlock checks GitHub Releases on launch. When a newer version exists, Settings shows Install update: it downloads the release installer and restarts the app."),
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

fn settings_modal(app: &RustBlock) -> El<'_> {
    let settings = &app.settings;

    let conversion = column![
        section_label("CONVERSION"),
        setting_toggle(
            "Confirm requantization",
            "Ask before converting a model that is already quantized.",
            settings.confirm_requantize,
            Message::SetConfirmRequantize,
        ),
        setting_toggle(
            "Auto-add downloads",
            "Put a finished download straight into the conversion queue.",
            settings.auto_add_downloads,
            Message::SetAutoAdd,
        ),
    ]
    .spacing(8);

    let downloads = column![
        section_label("DOWNLOADS"),
        folder_setting(
            "Download folder",
            &settings.download_dir,
            Message::PickDownloadDir,
            settings.download_dir.exists(),
        ),
        setting_toggle(
            "Verify SHA-256 after download",
            "Reads the whole file once after the transfer. Turn it off to stop at 100% on slow disks: the published size is still checked, the hash pass is skipped.",
            settings.verify_downloads,
            Message::SetVerifyDownloads,
        ),
    ]
    .spacing(8);

    let network = column![
        section_label("NETWORK"),
        text("Hugging Face mirror").size(10.0).color(STEEL_LIGHT),
        text_input("https://huggingface.co", &settings.hf_endpoint)
            .on_input(Message::HfEndpointChanged)
            .size(10.0)
            .padding(Padding::from([7.0, 9.0]))
            .style(theme::text_field),
        text("Use https://hf-mirror.com when huggingface.co is blocked; it applies to search, file lists and downloads. Hugging Face or ModelScope is chosen in Discover.")
            .size(9.0)
            .color(MUTED),
    ]
    .spacing(6);

    let ollama = match find_tool("ollama") {
        Some(path) => format!(
            "Ollama found at {}",
            shorten(&path.display().to_string(), 56)
        ),
        None => "Ollama not found on PATH; install it to import models with a generated Modelfile."
            .to_string(),
    };
    let runtimes = column![
        section_label("RUNTIME FOLDERS"),
        text("Detected folders appear as one-click copy targets on finished models.")
            .size(9.5)
            .color(MUTED),
        folder_setting(
            "LM Studio models",
            &settings.lmstudio_dir,
            Message::PickLmStudioDir,
            settings.lmstudio_dir.exists(),
        ),
        folder_setting(
            "Jan data folder",
            &settings.jan_dir,
            Message::PickJanDir,
            settings.jan_dir.exists(),
        ),
        folder_setting(
            "Prism / llama.cpp",
            &settings.prism_dir,
            Message::PickPrismDir,
            settings.prism_dir.exists(),
        ),
        text(ollama).size(9.0).color(MUTED),
    ]
    .spacing(8);

    let (update_status, update_color, install_ready) = match &app.update.state {
        UpdateState::Idle => ("Not checked yet.".to_string(), MUTED, false),
        UpdateState::Checking => ("Checking GitHub...".to_string(), INFO, false),
        UpdateState::UpToDate => ("You are on the latest release.".to_string(), OK, false),
        UpdateState::Available(info) => (
            format!(
                "RustBlock {} is available (installed v{}).",
                info.version,
                env!("CARGO_PKG_VERSION")
            ),
            ACCENT,
            info.installer_url.is_some(),
        ),
        UpdateState::Installing => (
            "Downloading and preparing the installer...".to_string(),
            WARN,
            false,
        ),
        UpdateState::Error(error) => (error.clone(), ERROR, false),
    };
    let mut update_buttons = row![].spacing(6);
    if let UpdateState::Available(info) = &app.update.state {
        update_buttons = update_buttons.push(
            button(text("Release notes").size(9.5))
                .on_press(Message::OpenUrl(info.page_url.clone()))
                .padding(Padding::from([5.0, 10.0]))
                .style(theme::secondary),
        );
    }
    if install_ready {
        update_buttons = update_buttons.push(
            button(text("Install update").size(9.5).color(INK))
                .on_press(Message::InstallUpdate)
                .padding(Padding::from([5.0, 12.0]))
                .style(theme::primary),
        );
    }
    if matches!(
        app.update.state,
        UpdateState::Idle | UpdateState::UpToDate | UpdateState::Error(_)
    ) {
        update_buttons = update_buttons.push(
            button(text("Check now").size(9.5))
                .on_press(Message::CheckUpdates)
                .padding(Padding::from([5.0, 10.0]))
                .style(theme::secondary),
        );
    }
    let update_notes: El<'_> = match &app.update.state {
        UpdateState::Available(info) if !info.notes.trim().is_empty() => text(
            info.notes
                .lines()
                .find(|line| !line.trim().is_empty())
                .unwrap_or_default()
                .chars()
                .take(110)
                .collect::<String>(),
        )
        .size(8.5)
        .color(MUTED)
        .into(),
        _ => space().height(Length::Fixed(0.0)).into(),
    };
    let updates = column![
        section_label("UPDATES"),
        row![
            column![
                text(format!("Installed v{}", env!("CARGO_PKG_VERSION")))
                    .size(10.5)
                    .color(TEXT),
                text(update_status).size(9.0).color(update_color),
            ]
            .spacing(2)
            .width(Length::Fill),
            update_buttons,
        ]
        .align_y(Alignment::Center)
        .spacing(8),
        update_notes,
    ]
    .spacing(6);

    let ram = app
        .hardware
        .ram_bytes
        .map(human_bytes)
        .unwrap_or_else(|| "not detected".to_string());
    let vram = app
        .hardware
        .vram_bytes
        .map(|bytes| format!("{} VRAM", human_bytes(bytes)))
        .unwrap_or_else(|| "no NVIDIA GPU detected".to_string());
    let system = column![
        section_label("SYSTEM"),
        row![
            text("CPU threads").size(10.5).color(TEXT),
            spacer(),
            text(format!("{} · all logical processors", conversion_workers()))
                .size(9.5)
                .font(MONO)
                .color(STEEL_LIGHT),
        ]
        .align_y(Alignment::Center),
        row![
            text("System RAM").size(10.5).color(TEXT),
            spacer(),
            text(ram).size(9.5).font(MONO).color(STEEL_LIGHT),
        ]
        .align_y(Alignment::Center),
        row![
            text("GPU VRAM").size(10.5).color(TEXT),
            spacer(),
            text(vram).size(9.5).font(MONO).color(STEEL_LIGHT),
        ]
        .align_y(Alignment::Center),
    ]
    .spacing(7);

    let content = column![
        conversion,
        divider(),
        downloads,
        divider(),
        network,
        divider(),
        updates,
        divider(),
        system,
        divider(),
        runtimes
    ]
    .spacing(12)
    .width(Length::Fill);

    let footer = column![
        divider(),
        row![
            column![
                text(format!("RustBlock v{}", env!("CARGO_PKG_VERSION")))
                    .size(9.0)
                    .font(MONO)
                    .color(STEEL_LIGHT),
                text("Saved to %APPDATA%\\RustBlock\\settings.json")
                    .size(8.0)
                    .font(MONO)
                    .color(MUTED),
            ]
            .spacing(1),
            spacer(),
            button(text("Reset defaults").size(10.0))
                .on_press(Message::ResetSettings)
                .padding(Padding::from([7.0, 14.0]))
                .style(theme::secondary),
            button(text("Done").size(11.0).color(INK))
                .on_press(Message::ToggleSettings)
                .padding(Padding::from([7.0, 22.0]))
                .style(theme::primary),
        ]
        .spacing(8)
        .align_y(Alignment::Center),
    ]
    .spacing(8);

    let card = column![
        row![
            text("Settings").size(18.0).color(TEXT),
            spacer(),
            text("Everything is saved as you change it.")
                .size(9.0)
                .color(MUTED),
        ]
        .align_y(Alignment::Center),
        divider(),
        container(
            scroll(container(content).padding(Padding::from([0.0, 12.0]))).height(Length::Fill)
        )
        .height(Length::Fill),
        footer,
    ]
    .spacing(12);

    modal(
        container(card)
            .width(Length::Fixed(700.0))
            .height(Length::Fixed(560.0))
            .padding(18)
            .style(|_| theme::outlined(PANEL, BORDER, 8.0)),
    )
}

fn setting_toggle<'a>(
    label: &'a str,
    hint: &'a str,
    value: bool,
    message: fn(bool) -> Message,
) -> El<'a> {
    row![
        column![
            text(label).size(11.0).color(TEXT),
            text(hint).size(9.0).color(MUTED),
        ]
        .spacing(2)
        .width(Length::Fill),
        checkbox(value)
            .on_toggle(message)
            .style(theme::checkbox_style),
    ]
    .spacing(10)
    .align_y(Alignment::Center)
    .into()
}

fn folder_setting(label: &str, path: &std::path::Path, pick: Message, ready: bool) -> El<'static> {
    row![
        column![
            row![
                text(label.to_string()).size(10.5).color(TEXT),
                chip(
                    if ready { "found" } else { "missing" },
                    PANEL_3,
                    if ready { OK } else { MUTED },
                ),
            ]
            .spacing(6)
            .align_y(Alignment::Center),
            text(shorten(&path.display().to_string(), 58))
                .size(9.0)
                .font(MONO)
                .color(MUTED),
        ]
        .spacing(2)
        .width(Length::Fill),
        button(text("Change").size(9.5))
            .on_press(pick)
            .padding(Padding::from([4.0, 10.0]))
            .style(theme::secondary),
    ]
    .spacing(8)
    .align_y(Alignment::Center)
    .into()
}

fn runtime_ready(runtime: Runtime, settings: &Settings) -> (bool, String) {
    match runtime {
        Runtime::Ollama => match find_tool("ollama") {
            Some(path) => (true, shorten(&path.display().to_string(), 48)),
            None => (false, "ollama not found on PATH".to_string()),
        },
        Runtime::LmStudio => (
            settings.lmstudio_dir.exists(),
            shorten(&settings.lmstudio_dir.display().to_string(), 48),
        ),
        Runtime::Jan => (
            settings.jan_dir.exists(),
            shorten(&settings.jan_dir.display().to_string(), 48),
        ),
        Runtime::Prism => (
            settings.prism_dir.exists(),
            shorten(&settings.prism_dir.display().to_string(), 48),
        ),
    }
}

fn deploy_modal(app: &RustBlock) -> El<'_> {
    let Some(deploy) = &app.deploy else {
        return space().height(Length::Fixed(0.0)).into();
    };
    let name = deploy
        .model
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let folder = deploy.folder.clone();

    let mut runtimes = column![].spacing(8);
    for runtime in Runtime::ALL {
        let (ready, detail) = runtime_ready(runtime, &app.settings);
        let selected = deploy.runtime == Some(runtime);
        let state = if deploy.busy && selected {
            "Working…"
        } else if !ready {
            "Folder missing"
        } else {
            "Deploy"
        };
        runtimes = runtimes.push(
            container(
                row![
                    column![
                        row![
                            text(runtime.label()).size(11.5).color(TEXT),
                            if ready {
                                chip("ready", PANEL_3, OK)
                            } else {
                                chip("not found", PANEL_3, WARN)
                            },
                        ]
                        .spacing(6)
                        .align_y(Alignment::Center),
                        text(runtime.hint()).size(9.0).color(MUTED),
                        text(detail).size(8.5).font(MONO).color(STEEL_LIGHT),
                    ]
                    .spacing(3)
                    .width(Length::Fill),
                    button(text(state).size(9.5).color(INK))
                        .on_press_maybe(
                            (!deploy.busy && ready).then_some(Message::DeployTo(runtime))
                        )
                        .padding(Padding::from([5.0, 12.0]))
                        .style(theme::secondary),
                ]
                .spacing(10)
                .align_y(Alignment::Center),
            )
            .padding(10)
            .width(Length::Fill)
            .style({
                let (fill, line) = if selected {
                    (PANEL_3, ACCENT_DARK)
                } else {
                    (PANEL_2, BORDER)
                };
                move |_| theme::outlined(fill, line, 6.0)
            }),
        );
    }

    let mut status = column![].spacing(6);
    if deploy.busy || deploy.done > 0 {
        status = status.push(
            progress_bar(0.0..=1.0, deploy.fraction())
                .girth(6.0)
                .style(theme::progress),
        );
        status = status.push(
            text(format!(
                "{} / {}",
                human_bytes(deploy.done),
                if deploy.total > 0 {
                    human_bytes(deploy.total)
                } else {
                    "?".to_string()
                }
            ))
            .size(9.0)
            .font(MONO)
            .color(MUTED),
        );
    }
    if let Some(message) = &deploy.message {
        status = status.push(
            text(message.as_str())
                .size(9.5)
                .color(if deploy.folder.is_some() { OK } else { ERROR }),
        );
    }

    let mut footer = row![spacer()].spacing(8);
    if let Some(folder) = folder {
        footer = footer.push(
            button(text("Open folder").size(10.0))
                .on_press(Message::OpenPath(folder))
                .padding(Padding::from([6.0, 12.0]))
                .style(theme::secondary),
        );
    }
    if deploy.busy {
        footer = footer.push(
            button(text("Cancel").size(10.0).color(MUTED))
                .on_press(Message::DeployCancel)
                .padding(Padding::from([6.0, 14.0]))
                .style(theme::ghost),
        );
    } else {
        footer = footer.push(
            button(text("Close").size(10.5))
                .on_press(Message::DeployClose)
                .padding(Padding::from([6.0, 16.0]))
                .style(theme::secondary),
        );
    }

    let card = column![
        text("Test in a runtime").size(18.0).color(TEXT),
        text(shorten(&name, 62))
            .size(10.0)
            .font(MONO)
            .color(STEEL_LIGHT),
        text("Pick a runtime. Ollama registers the model with its own command; the other targets copy the GGUF into their model folder.")
            .size(9.5)
            .color(MUTED),
        divider(),
        container(scroll(container(runtimes).padding(Padding::from([0.0, 12.0]))).height(Length::Fill))
            .height(Length::Fill),
        status,
        footer,
    ]
    .spacing(10);

    modal(
        container(card)
            .width(Length::Fixed(620.0))
            .height(Length::Fixed(520.0))
            .padding(18)
            .style(|_| theme::outlined(PANEL, BORDER, 8.0)),
    )
}

fn confirm_close_modal() -> El<'static> {
    modal(
        container(
            column![
                text("Quit while RustBlock is busy?").size(17.0).color(WARN),
                text("A conversion, download or deploy is still running. Closing now stops it; partial files stay on disk.")
                    .size(10.5)
                    .color(MUTED)
                    .width(Length::Fill),
                row![
                    button(text("Keep working").size(11.0))
                        .on_press(Message::ConfirmClose(false))
                        .padding(Padding::from([7.0, 16.0]))
                        .style(theme::secondary),
                    button(text("Quit anyway").size(11.0).color(INK))
                        .on_press(Message::ConfirmClose(true))
                        .padding(Padding::from([7.0, 16.0]))
                        .style(theme::primary),
                ]
                .spacing(8),
            ]
            .spacing(10),
        )
        .width(Length::Fixed(460.0))
        .padding(20)
        .style(|_| theme::outlined(PANEL, BORDER, 8.0)),
    )
}

fn modal<'a>(card: impl Into<El<'a>>) -> El<'a> {
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
        for tab in [Tab::Convert, Tab::Discover, Tab::Downloads, Tab::Inspector] {
            app.tab = tab;
            let _ = view(&app);
        }
        app.show_log = true;
        let _ = view(&app);
        app.show_help = true;
        let _ = view(&app);
        app.confirm_requantize = Some(crate::worker::WorkKind::Convert);
        let _ = view(&app);
        app.confirm_requantize = None;
        app.show_settings = true;
        let _ = view(&app);
        app.show_settings = false;
        app.deploy = Some(crate::deploy::DeployUi::new(std::path::PathBuf::from(
            "model.gguf",
        )));
        let _ = view(&app);
        app.deploy = None;
        app.confirm_close = true;
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
    fn inspector_lists_blocks_and_builds_the_editor() {
        let mut app = RustBlock::empty();
        app.tab = Tab::Inspector;
        app.inspector_path = Some(std::path::PathBuf::from("model.gguf"));
        app.inspector_input = "model.gguf".to_string();
        app.inspector_output = "model.edited.gguf".to_string();
        app.inspector_selected = Some(1);
        app.inspector = Some(InspectorData {
            path: std::path::PathBuf::from("model.gguf"),
            version: 3,
            file_len: 16384,
            alignment: 32,
            data_start: 1024,
            tensors: 3,
            blocks: vec![
                InspectorBlock {
                    name: "blk.0.attn_norm.weight".to_string(),
                    ty: 0,
                    type_name: "F32",
                    dims: vec![64],
                    elements: 64,
                    bytes: 256,
                    offset: 1024,
                    block: 1,
                    block_bytes: 4,
                    bits_per_weight: 32.0,
                    share: 0.02,
                },
                InspectorBlock {
                    name: "blk.0.attn_q.weight".to_string(),
                    ty: 12,
                    type_name: "Q4_K",
                    dims: vec![64, 64],
                    elements: 4096,
                    bytes: 2304,
                    offset: 1280,
                    block: 256,
                    block_bytes: 144,
                    bits_per_weight: 4.5,
                    share: 0.8,
                },
                InspectorBlock {
                    name: "output.weight".to_string(),
                    ty: 1,
                    type_name: "F16",
                    dims: vec![64, 64],
                    elements: 4096,
                    bytes: 8192,
                    offset: 5120,
                    block: 1,
                    block_bytes: 2,
                    bits_per_weight: 16.0,
                    share: 0.18,
                },
            ],
            pq2: 0,
            ptq1: 0,
            file_type: Some(14),
            metadata: vec![("general.architecture".to_string(), "llama".to_string())],
            histogram: vec![("Q4_K".to_string(), 1, 2304)],
            weights_bytes: 10752,
            fit: None,
        });
        let _ = view(&app);
        app.inspector_filter = "q4_k".to_string();
        let _ = view(&app);
        app.inspector_filter = "nothing".to_string();
        let _ = view(&app);
    }

    #[test]
    fn target_captions_cover_every_preset() {
        for target in Target::ALL {
            assert!(!target_caption(target).is_empty());
        }
    }
}
