//! Steel-and-blocks visual language: dark graphite, brushed steel, a single
//! neutral steel accent and a gear/cube/arrows mark drawn entirely with egui shapes.

use eframe::egui::{
    self, Color32, CornerRadius, FontId, IconData, Pos2, Shape, Stroke, StrokeKind, TextStyle, Vec2,
};
use std::collections::VecDeque;
use std::f32::consts::TAU;

pub const BG: Color32 = Color32::from_rgb(14, 16, 18);
pub const PANEL: Color32 = Color32::from_rgb(22, 25, 28);
pub const PANEL_2: Color32 = Color32::from_rgb(29, 33, 37);
pub const PANEL_3: Color32 = Color32::from_rgb(39, 44, 49);
pub const BORDER: Color32 = Color32::from_rgb(51, 58, 64);
pub const TEXT: Color32 = Color32::from_rgb(232, 229, 222);
pub const MUTED: Color32 = Color32::from_rgb(143, 151, 157);
pub const ACCENT: Color32 = Color32::from_rgb(166, 173, 178);
pub const ACCENT_DARK: Color32 = Color32::from_rgb(118, 126, 132);
pub const STEEL: Color32 = Color32::from_rgb(125, 134, 141);
pub const STEEL_LIGHT: Color32 = Color32::from_rgb(189, 195, 199);
pub const STEEL_DARK: Color32 = Color32::from_rgb(104, 112, 118);
pub const WARN: Color32 = Color32::from_rgb(190, 184, 168);
pub const ERROR: Color32 = Color32::from_rgb(222, 105, 92);
pub const INFO: Color32 = Color32::from_rgb(132, 153, 158);
pub const OK: Color32 = Color32::from_rgb(126, 174, 138);
pub const INK: Color32 = Color32::from_rgb(14, 16, 18);

pub fn install(ctx: &egui::Context) {
    ctx.set_theme(egui::ThemePreference::Dark);

    ctx.all_styles_mut(|style| {
        style.visuals = visuals();
        style.spacing.item_spacing = Vec2::new(8.0, 8.0);
        style.spacing.button_padding = Vec2::new(10.0, 6.0);
        style.spacing.interact_size = Vec2::new(36.0, 28.0);
        style.spacing.menu_margin = egui::Margin::same(6);
        style.animation_time = 0.12;
        style
            .text_styles
            .insert(TextStyle::Heading, FontId::proportional(24.0));
        style
            .text_styles
            .insert(TextStyle::Body, FontId::proportional(13.5));
        style
            .text_styles
            .insert(TextStyle::Button, FontId::proportional(12.0));
        style
            .text_styles
            .insert(TextStyle::Small, FontId::proportional(11.0));
        style
            .text_styles
            .insert(TextStyle::Monospace, FontId::monospace(11.0));
    });
}

fn visuals() -> egui::Visuals {
    let mut v = egui::Visuals::dark();
    v.panel_fill = BG;
    v.window_fill = PANEL;
    v.extreme_bg_color = Color32::from_rgb(10, 12, 13);
    v.faint_bg_color = PANEL_2;
    v.code_bg_color = Color32::from_rgb(26, 31, 35);
    v.window_stroke = Stroke::new(1.0, BORDER);
    v.window_corner_radius = CornerRadius::same(8);
    v.menu_corner_radius = CornerRadius::same(6);
    v.window_shadow = egui::epaint::Shadow::NONE;
    v.popup_shadow = egui::epaint::Shadow::NONE;
    v.selection.bg_fill = ACCENT.gamma_multiply(0.35);
    v.selection.stroke = Stroke::new(1.0, ACCENT);
    v.hyperlink_color = ACCENT;
    v.warn_fg_color = WARN;
    v.error_fg_color = ERROR;
    v.override_text_color = Some(TEXT);

    let corner = CornerRadius::same(6);
    v.widgets.noninteractive.bg_fill = PANEL;
    v.widgets.noninteractive.weak_bg_fill = PANEL;
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, BORDER);
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, STEEL_LIGHT);
    v.widgets.noninteractive.corner_radius = corner;

    v.widgets.inactive.bg_fill = PANEL;
    v.widgets.inactive.weak_bg_fill = PANEL;
    v.widgets.inactive.bg_stroke = Stroke::new(1.0, BORDER);
    v.widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT);
    v.widgets.inactive.corner_radius = corner;

    v.widgets.hovered.bg_fill = PANEL_2;
    v.widgets.hovered.weak_bg_fill = PANEL_2;
    v.widgets.hovered.bg_stroke = Stroke::new(1.0, STEEL);
    v.widgets.hovered.fg_stroke = Stroke::new(1.0, TEXT);
    v.widgets.hovered.corner_radius = corner;

    v.widgets.active.bg_fill = ACCENT;
    v.widgets.active.weak_bg_fill = ACCENT;
    v.widgets.active.bg_stroke = Stroke::new(1.0, ACCENT_DARK);
    v.widgets.active.fg_stroke = Stroke::new(1.0, INK);
    v.widgets.active.corner_radius = corner;

    v.widgets.open.bg_fill = PANEL_2;
    v.widgets.open.weak_bg_fill = PANEL_2;
    v.widgets.open.bg_stroke = Stroke::new(1.0, BORDER);
    v.widgets.open.fg_stroke = Stroke::new(1.0, TEXT);
    v.widgets.open.corner_radius = corner;
    v
}

pub const LOGO_PNG: &[u8] = include_bytes!("../assets/rustblock-logo.png");
pub const WORDMARK_PNG: &[u8] = include_bytes!("../assets/rustblock-wordmark.png");

pub fn load_wordmark(ctx: &egui::Context) -> Option<egui::TextureHandle> {
    let mut decoded = image::load_from_memory(WORDMARK_PNG).ok()?.to_rgba8();
    clear_connected_border(&mut decoded);
    let (width, height) = decoded.dimensions();
    let mut bounds = (width, height, 0u32, 0u32);
    for (x, y, pixel) in decoded.enumerate_pixels() {
        if pixel[3] > 8 {
            bounds.0 = bounds.0.min(x);
            bounds.1 = bounds.1.min(y);
            bounds.2 = bounds.2.max(x);
            bounds.3 = bounds.3.max(y);
        }
    }
    if bounds.2 >= bounds.0 && bounds.3 >= bounds.1 {
        decoded = image::imageops::crop_imm(
            &decoded,
            bounds.0,
            bounds.1,
            bounds.2 - bounds.0 + 1,
            bounds.3 - bounds.1 + 1,
        )
        .to_image();
    }
    let size = [decoded.width() as usize, decoded.height() as usize];
    let image = egui::ColorImage::from_rgba_unmultiplied(size, decoded.as_raw());
    Some(ctx.load_texture("rustblock-wordmark", image, egui::TextureOptions::LINEAR))
}

/// Remove a flat canvas connected to the image border without punching holes
/// inside the artwork. The supplied wordmark is black-on-black with a dark
/// cube; a global key would hollow the cube and the white letters, so only the
/// background reachable from the edges is cleared.
fn clear_connected_border(image: &mut image::RgbaImage) {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return;
    }
    let background = *image.get_pixel(0, 0);
    if background[3] == 0 {
        return;
    }
    let matches = |pixel: &image::Rgba<u8>| {
        pixel[3] > 0
            && pixel[0].abs_diff(background[0]) <= 6
            && pixel[1].abs_diff(background[1]) <= 6
            && pixel[2].abs_diff(background[2]) <= 6
    };
    let mut visited = vec![false; (width as usize) * (height as usize)];
    let mut queue = VecDeque::new();
    for x in 0..width {
        queue.push_back((x, 0));
        queue.push_back((x, height - 1));
    }
    for y in 1..height - 1 {
        queue.push_back((0, y));
        queue.push_back((width - 1, y));
    }
    while let Some((x, y)) = queue.pop_front() {
        let index = y as usize * width as usize + x as usize;
        if visited[index] {
            continue;
        }
        visited[index] = true;
        if !matches(image.get_pixel(x, y)) {
            continue;
        }
        image.put_pixel(x, y, image::Rgba([0, 0, 0, 0]));
        if x > 0 {
            queue.push_back((x - 1, y));
        }
        if x + 1 < width {
            queue.push_back((x + 1, y));
        }
        if y > 0 {
            queue.push_back((x, y - 1));
        }
        if y + 1 < height {
            queue.push_back((x, y + 1));
        }
    }
}

pub fn load_gear(ctx: &egui::Context) -> Option<egui::TextureHandle> {
    let decoded = image::load_from_memory(LOGO_PNG).ok()?.to_rgba8();
    let size = [decoded.width() as usize, decoded.height() as usize];
    let image = egui::ColorImage::from_rgba_unmultiplied(size, decoded.as_raw());
    Some(ctx.load_texture("rustblock-gear", image, egui::TextureOptions::LINEAR))
}

pub fn paint_wordmark(
    painter: &egui::Painter,
    rect: egui::Rect,
    texture: Option<&egui::TextureHandle>,
    tint: Color32,
) {
    let Some(texture) = texture else {
        return;
    };
    painter.image(
        texture.id(),
        rect,
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        tint,
    );
}

pub fn paint_logo(ui: &mut egui::Ui, size: f32, logo: Option<&egui::TextureHandle>) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(size), egui::Sense::hover());
    if let Some(texture) = logo {
        ui.painter().image(
            texture.id(),
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            Color32::WHITE,
        );
        ui.painter().rect_stroke(
            rect,
            CornerRadius::same(0),
            Stroke::new(1.0, BORDER),
            StrokeKind::Inside,
        );
        return;
    }
    let painter = ui.painter();
    let c = rect.center();
    let r = size * 0.46;

    painter.rect_filled(
        rect,
        CornerRadius::same((size * 0.16) as u8),
        Color32::from_rgb(9, 13, 18),
    );

    // Gear body and teeth.
    for i in 0..14 {
        let a = i as f32 / 14.0 * TAU;
        let p0 = c + Vec2::angled(a - 0.11) * (r * 0.82);
        let p1 = c + Vec2::angled(a - 0.055) * r;
        let p2 = c + Vec2::angled(a + 0.055) * r;
        let p3 = c + Vec2::angled(a + 0.11) * (r * 0.82);
        painter.add(Shape::convex_polygon(
            vec![p0, p1, p2, p3],
            STEEL_DARK,
            Stroke::NONE,
        ));
    }
    painter.circle_filled(c, r * 0.82, STEEL);
    painter.circle_filled(c, r * 0.68, Color32::from_rgb(17, 21, 26));
    painter.circle_stroke(c, r * 0.68, Stroke::new(size * 0.018, STEEL_LIGHT));

    // Isometric block cube.
    let s = size * 0.135;
    let top = c + Vec2::new(0.0, -s * 0.95);
    let left = c + Vec2::new(-s, 0.0);
    let right = c + Vec2::new(s, 0.0);
    painter.add(Shape::convex_polygon(
        vec![
            top,
            c + Vec2::new(0.0, -s * 0.1),
            c + Vec2::new(-s, -s * 0.55),
            left,
        ],
        STEEL_LIGHT,
        Stroke::new(1.0, Color32::from_rgb(30, 38, 46)),
    ));
    painter.add(Shape::convex_polygon(
        vec![
            left,
            c + Vec2::new(-s, -s * 0.55),
            c + Vec2::new(0.0, -s * 0.1),
            c,
        ],
        Color32::from_rgb(142, 154, 166),
        Stroke::new(1.0, Color32::from_rgb(30, 38, 46)),
    ));
    painter.add(Shape::convex_polygon(
        vec![
            c,
            c + Vec2::new(0.0, -s * 0.1),
            c + Vec2::new(s, -s * 0.55),
            right,
        ],
        Color32::from_rgb(96, 108, 120),
        Stroke::new(1.0, Color32::from_rgb(30, 38, 46)),
    ));
    for k in 1..3 {
        let f = k as f32 / 3.0;
        painter.line_segment(
            [
                c + Vec2::new(-s * f, s * 0.95 * f),
                c + Vec2::new(s * f, s * 0.95 * f),
            ],
            Stroke::new(1.0, Color32::from_rgb(45, 56, 66)),
        );
        painter.line_segment(
            [
                c + Vec2::new(-s * f, -s * 0.55 * f),
                c + Vec2::new(-s * f, s * 0.4 * f),
            ],
            Stroke::new(1.0, Color32::from_rgb(45, 56, 66)),
        );
    }

    // Two orbital arrows.
    paint_arc_arrow(painter, c, r * 0.9, 0.35, 2.5, STEEL_LIGHT);
    paint_arc_arrow(painter, c, r * 0.9, 3.5, 5.55, STEEL_LIGHT);
}

fn paint_arc_arrow(
    painter: &egui::Painter,
    c: Pos2,
    radius: f32,
    start: f32,
    end: f32,
    color: Color32,
) {
    let mut points = Vec::with_capacity(28);
    let steps = 24;
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let a = start + (end - start) * t;
        points.push(c + Vec2::angled(a) * radius);
    }
    painter.add(Shape::line(points.clone(), Stroke::new(2.2, color)));
    if let (Some(&tip), Some(&prev)) = (points.last(), points.get(points.len() - 3)) {
        let dir = (tip - prev).normalized();
        let normal = Vec2::new(-dir.y, dir.x);
        let head = 5.0;
        painter.add(Shape::convex_polygon(
            vec![
                tip,
                tip - dir * head + normal * (head * 0.55),
                tip - dir * head - normal * (head * 0.55),
            ],
            color,
            Stroke::NONE,
        ));
    }
}

pub fn app_icon() -> IconData {
    let decoded = image::load_from_memory(LOGO_PNG)
        .expect("embedded RustBlock logo must be a valid PNG")
        .to_rgba8();
    let resized =
        image::imageops::resize(&decoded, 128, 128, image::imageops::FilterType::Lanczos3);
    IconData {
        rgba: resized.into_raw(),
        width: 128,
        height: 128,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn channel(value: u8) -> f32 {
        let value = value as f32 / 255.0;
        if value <= 0.04045 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    }

    fn luminance(color: Color32) -> f32 {
        0.2126 * channel(color.r()) + 0.7152 * channel(color.g()) + 0.0722 * channel(color.b())
    }

    fn contrast(a: Color32, b: Color32) -> f32 {
        let (lighter, darker) = if luminance(a) >= luminance(b) {
            (luminance(a), luminance(b))
        } else {
            (luminance(b), luminance(a))
        };
        (lighter + 0.05) / (darker + 0.05)
    }

    #[test]
    fn text_and_status_colors_meet_wcag_aa() {
        for (foreground, background) in [
            (TEXT, BG),
            (TEXT, PANEL),
            (TEXT, PANEL_2),
            (MUTED, BG),
            (MUTED, PANEL),
            (MUTED, PANEL_2),
            (MUTED, PANEL_3),
            (WARN, PANEL),
            (ERROR, PANEL),
            (OK, PANEL),
            (INFO, PANEL),
            (INK, ACCENT),
            (STEEL_LIGHT, PANEL),
        ] {
            let ratio = contrast(foreground, background);
            assert!(
                ratio >= 4.5,
                "contrast {ratio:.2}:1 is below WCAG AA for {foreground:?} on {background:?}"
            );
        }
    }

    #[test]
    fn border_keying_keeps_enclosed_dark_pixels() {
        let mut image = image::RgbaImage::from_pixel(9, 9, image::Rgba([0, 0, 0, 255]));
        for y in 2..7 {
            for x in 2..7 {
                image.put_pixel(x, y, image::Rgba([240, 240, 240, 255]));
            }
        }
        image.put_pixel(4, 4, image::Rgba([0, 0, 0, 255]));
        clear_connected_border(&mut image);
        assert_eq!(image.get_pixel(0, 0)[3], 0);
        assert_eq!(image.get_pixel(2, 2)[3], 255);
        assert_eq!(image.get_pixel(4, 4)[3], 255);
    }
}
