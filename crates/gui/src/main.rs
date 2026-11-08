#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

//! RustBlock — native desktop GGUF conversion toolkit.

mod app;
mod hf;
mod model;
mod theme;
mod worker;

use eframe::egui;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("RustBlock — Native GGUF converter")
            // The window chrome is drawn inside the app, so the client area is
            // the whole window and the custom title bar owns dragging and
            // minimize/close.
            .with_decorations(false)
            .with_resizable(true)
            .with_inner_size([1120.0, 680.0])
            .with_min_inner_size([880.0, 560.0])
            .with_position(egui::pos2(20.0, 0.0))
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
