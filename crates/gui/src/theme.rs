//! Colors, global widget style and the programmatically drawn app icon.

use eframe::egui::{self, Color32, CornerRadius, FontId, IconData, Stroke, TextStyle, Vec2};

pub const BG: Color32 = Color32::from_rgb(11, 15, 19);
pub const PANEL: Color32 = Color32::from_rgb(19, 25, 31);
pub const PANEL_2: Color32 = Color32::from_rgb(26, 34, 41);
pub const PANEL_3: Color32 = Color32::from_rgb(33, 43, 51);
pub const BORDER: Color32 = Color32::from_rgb(43, 56, 66);
pub const TEXT: Color32 = Color32::from_rgb(228, 236, 241);
pub const MUTED: Color32 = Color32::from_rgb(139, 156, 168);
pub const ACCENT: Color32 = Color32::from_rgb(111, 207, 151);
pub const ACCENT_DARK: Color32 = Color32::from_rgb(52, 145, 102);
pub const WARN: Color32 = Color32::from_rgb(240, 184, 83);
pub const ERROR: Color32 = Color32::from_rgb(240, 104, 104);
pub const INFO: Color32 = Color32::from_rgb(110, 180, 240);
pub const INK: Color32 = Color32::from_rgb(8, 20, 14);

pub fn install(ctx: &egui::Context) {
    ctx.set_theme(egui::ThemePreference::Dark);

    ctx.all_styles_mut(|style| {
        style.visuals = visuals();

        style.spacing.item_spacing = Vec2::new(10.0, 8.0);
        style.spacing.button_padding = Vec2::new(14.0, 7.0);
        style.spacing.interact_size = Vec2::new(42.0, 30.0);
        style.spacing.menu_margin = egui::Margin::same(8);
        style.animation_time = 0.12;

        style
            .text_styles
            .insert(TextStyle::Heading, FontId::proportional(22.0));
        style
            .text_styles
            .insert(TextStyle::Body, FontId::proportional(14.0));
        style
            .text_styles
            .insert(TextStyle::Button, FontId::proportional(14.0));
        style
            .text_styles
            .insert(TextStyle::Small, FontId::proportional(11.5));
        style
            .text_styles
            .insert(TextStyle::Monospace, FontId::monospace(12.0));
    });
}

fn visuals() -> egui::Visuals {
    let mut v = egui::Visuals::dark();
    v.panel_fill = BG;
    v.window_fill = PANEL;
    v.extreme_bg_color = Color32::from_rgb(9, 12, 16);
    v.faint_bg_color = PANEL_2;
    v.code_bg_color = Color32::from_rgb(9, 12, 16);
    v.window_stroke = Stroke::new(1.0, BORDER);
    v.window_corner_radius = CornerRadius::same(12);
    v.menu_corner_radius = CornerRadius::same(8);
    v.selection.bg_fill = ACCENT.gamma_multiply(0.28);
    v.selection.stroke = Stroke::new(1.0, ACCENT);
    v.hyperlink_color = ACCENT;
    v.warn_fg_color = WARN;
    v.error_fg_color = ERROR;

    let corner = CornerRadius::same(7);
    v.widgets.noninteractive.bg_fill = PANEL;
    v.widgets.noninteractive.weak_bg_fill = PANEL;
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, BORDER);
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT);
    v.widgets.noninteractive.corner_radius = corner;

    v.widgets.inactive.bg_fill = PANEL_2;
    v.widgets.inactive.weak_bg_fill = PANEL_2;
    v.widgets.inactive.bg_stroke = Stroke::new(1.0, BORDER);
    v.widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT);
    v.widgets.inactive.corner_radius = corner;

    v.widgets.hovered.bg_fill = PANEL_3;
    v.widgets.hovered.weak_bg_fill = PANEL_3;
    v.widgets.hovered.bg_stroke = Stroke::new(1.0, ACCENT.gamma_multiply(0.7));
    v.widgets.hovered.fg_stroke = Stroke::new(1.5, Color32::WHITE);
    v.widgets.hovered.corner_radius = corner;

    v.widgets.active.bg_fill = ACCENT_DARK;
    v.widgets.active.weak_bg_fill = ACCENT_DARK;
    v.widgets.active.bg_stroke = Stroke::new(1.0, ACCENT);
    v.widgets.active.fg_stroke = Stroke::new(1.5, Color32::WHITE);
    v.widgets.active.corner_radius = corner;

    v.widgets.open.bg_fill = PANEL_3;
    v.widgets.open.weak_bg_fill = PANEL_3;
    v.widgets.open.bg_stroke = Stroke::new(1.0, BORDER);
    v.widgets.open.fg_stroke = Stroke::new(1.0, TEXT);
    v.widgets.open.corner_radius = corner;

    v
}

/// Draws a small bonsai mark with the painter (28×28 by default).
pub fn paint_logo(ui: &mut egui::Ui, size: f32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(size), egui::Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, CornerRadius::same((size * 0.22) as u8), PANEL_2);

    let c = rect.center();
    let s = size / 28.0;
    painter.line_segment(
        [c + Vec2::new(0.0, 1.0 * s), c + Vec2::new(0.0, 10.0 * s)],
        Stroke::new(2.4 * s, Color32::from_rgb(158, 118, 78)),
    );
    painter.circle_filled(c + Vec2::new(0.0, -6.0 * s), 7.0 * s, ACCENT);
    painter.circle_filled(c + Vec2::new(-7.0, 0.5 * s), 4.8 * s, ACCENT_DARK);
    painter.circle_filled(c + Vec2::new(7.0, 0.5 * s), 4.8 * s, ACCENT_DARK);
    painter.circle_filled(
        c + Vec2::new(0.0, -9.0 * s),
        4.2 * s,
        Color32::from_rgb(138, 226, 172),
    );
}

/// 64×64 window icon: a stylised bonsai on a dark rounded tile.
pub fn app_icon() -> IconData {
    const SIZE: usize = 64;
    const SS: usize = 4;
    let mut rgba = vec![0u8; SIZE * SIZE * 4];

    for y in 0..SIZE {
        for x in 0..SIZE {
            let mut acc = [0f32; 4];
            for sy in 0..SS {
                for sx in 0..SS {
                    let px = x as f32 + (sx as f32 + 0.5) / SS as f32;
                    let py = y as f32 + (sy as f32 + 0.5) / SS as f32;
                    let c = icon_sample(px, py);
                    for k in 0..4 {
                        acc[k] += c[k] as f32;
                    }
                }
            }
            let n = (SS * SS) as f32;
            let i = (y * SIZE + x) * 4;
            for k in 0..4 {
                rgba[i + k] = (acc[k] / n).round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    IconData {
        rgba,
        width: SIZE as u32,
        height: SIZE as u32,
    }
}

fn icon_sample(x: f32, y: f32) -> [u8; 4] {
    let d = sd_round_rect(x, y, 3.0, 3.0, 61.0, 61.0, 14.0);
    if d > 0.0 {
        return [0, 0, 0, 0];
    }
    if d > -1.4 {
        return [49, 66, 57, 255];
    }

    // Trunk and a small branch.
    if (30.0..=33.5).contains(&x) && (31.0..=52.0).contains(&y) {
        return [165, 122, 80, 255];
    }
    if (24.5..=39.5).contains(&x) && (37.0..=40.0).contains(&y) {
        return [165, 122, 80, 255];
    }

    // Canopy.
    if dist(x, y, 32.0, 22.0) <= 13.0 {
        return [111, 207, 151, 255];
    }
    if dist(x, y, 20.0, 30.0) <= 8.0 {
        return [78, 173, 120, 255];
    }
    if dist(x, y, 44.0, 30.0) <= 8.0 {
        return [78, 173, 120, 255];
    }
    if dist(x, y, 25.0, 18.0) <= 6.5 {
        return [139, 226, 173, 255];
    }
    if dist(x, y, 39.0, 18.0) <= 6.5 {
        return [139, 226, 173, 255];
    }

    [19, 27, 23, 255]
}

fn dist(x: f32, y: f32, cx: f32, cy: f32) -> f32 {
    ((x - cx).powi(2) + (y - cy).powi(2)).sqrt()
}

fn sd_round_rect(x: f32, y: f32, x0: f32, y0: f32, x1: f32, y1: f32, r: f32) -> f32 {
    let cx = (x0 + x1) * 0.5;
    let cy = (y0 + y1) * 0.5;
    let hx = (x1 - x0) * 0.5 - r;
    let hy = (y1 - y0) * 0.5 - r;
    let qx = (x - cx).abs() - hx;
    let qy = (y - cy).abs() - hy;
    let outside = (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt();
    outside + qx.max(qy).min(0.0) - r
}
