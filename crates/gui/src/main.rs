#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

//! RustBlock — native desktop GGUF conversion toolkit.
//!
//! The window is frameless: the app draws its own chrome and owns dragging,
//! minimize, maximize and close. Everything else is plain iced.

mod app;
mod deploy;
mod edit;
mod hf;
mod model;
mod theme;
mod view;
mod worker;

use iced::window;

fn main() -> iced::Result {
    iced::application(
        app::RustBlock::boot,
        app::RustBlock::update,
        app::RustBlock::view,
    )
    .title(|_: &app::RustBlock| "RustBlock — Native GGUF converter".to_string())
    .theme(|_: &app::RustBlock| theme::theme())
    .subscription(app::RustBlock::subscription)
    .antialiasing(true)
    .window(window::Settings {
        size: iced::Size::new(1120.0, 700.0),
        min_size: Some(iced::Size::new(1000.0, 640.0)),
        position: window::Position::Specific(iced::Point::new(20.0, 0.0)),
        decorations: false,
        resizable: true,
        icon: Some(theme::app_icon()),
        ..window::Settings::default()
    })
    .run()
}
