#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

//! RustBlock — native desktop GGUF conversion toolkit.

mod app;
mod hf;
mod model;
mod targets;
mod theme;
mod worker;

use eframe::egui;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("RustBlock — Native GGUF converter")
            .with_inner_size([1120.0, 720.0])
            .with_min_inner_size([880.0, 560.0])
            .with_clamp_size_to_monitor_size(true)
            .with_icon(theme::app_icon()),
        ..Default::default()
    };

    eframe::run_native(
        "RustBlock",
        options,
        Box::new(|cc| Ok(Box::new(app::RustBlockApp::new(cc)))),
    )
}
