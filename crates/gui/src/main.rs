#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

//! Reblock — desktop GUI for lossless PQ2_0 → Q2_0 GGUF conversion.

mod app;
mod i18n;
mod model;
mod theme;
mod worker;

use eframe::egui;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Reblock — Ternary Bonsai converter")
            .with_inner_size([1080.0, 660.0])
            .with_min_inner_size([840.0, 540.0])
            .with_clamp_size_to_monitor_size(true)
            .with_icon(theme::app_icon()),
        ..Default::default()
    };

    eframe::run_native(
        "Reblock",
        options,
        Box::new(|cc| Ok(Box::new(app::ReblockApp::new(cc)))),
    )
}
