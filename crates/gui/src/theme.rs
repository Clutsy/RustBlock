//! Steel-and-blocks visual language: dark graphite, brushed steel, a single
//! neutral steel accent and a gear/cube/arrows mark.
//!
//! The palette is the same one the egui build used; only the framework types
//! changed. Every surface is flat (no shadows), borders are one pixel and the
//! radii follow the 6 px widget / 8 px surface grid.

use std::collections::VecDeque;

use iced::widget::{button, checkbox, container, progress_bar, scrollable, text_input};
use iced::{Background, Border, Color, Shadow, Theme};

pub const BG: Color = Color::from_rgb(14.0 / 255.0, 16.0 / 255.0, 18.0 / 255.0);
pub const PANEL: Color = Color::from_rgb(22.0 / 255.0, 25.0 / 255.0, 28.0 / 255.0);
pub const PANEL_2: Color = Color::from_rgb(29.0 / 255.0, 33.0 / 255.0, 37.0 / 255.0);
pub const PANEL_3: Color = Color::from_rgb(39.0 / 255.0, 44.0 / 255.0, 49.0 / 255.0);
pub const BORDER: Color = Color::from_rgb(51.0 / 255.0, 58.0 / 255.0, 64.0 / 255.0);
pub const TEXT: Color = Color::from_rgb(232.0 / 255.0, 229.0 / 255.0, 222.0 / 255.0);
pub const MUTED: Color = Color::from_rgb(143.0 / 255.0, 151.0 / 255.0, 157.0 / 255.0);
pub const ACCENT: Color = Color::from_rgb(166.0 / 255.0, 173.0 / 255.0, 178.0 / 255.0);
pub const ACCENT_DARK: Color = Color::from_rgb(118.0 / 255.0, 126.0 / 255.0, 132.0 / 255.0);
pub const STEEL: Color = Color::from_rgb(125.0 / 255.0, 134.0 / 255.0, 141.0 / 255.0);
pub const STEEL_LIGHT: Color = Color::from_rgb(189.0 / 255.0, 195.0 / 255.0, 199.0 / 255.0);
pub const WARN: Color = Color::from_rgb(190.0 / 255.0, 184.0 / 255.0, 168.0 / 255.0);
pub const ERROR: Color = Color::from_rgb(222.0 / 255.0, 105.0 / 255.0, 92.0 / 255.0);
pub const INFO: Color = Color::from_rgb(132.0 / 255.0, 153.0 / 255.0, 158.0 / 255.0);
pub const OK: Color = Color::from_rgb(126.0 / 255.0, 174.0 / 255.0, 138.0 / 255.0);
pub const INK: Color = Color::from_rgb(14.0 / 255.0, 16.0 / 255.0, 18.0 / 255.0);

/// The very dark field used by inputs and the log scroller.
pub const WELL: Color = Color::from_rgb(10.0 / 255.0, 12.0 / 255.0, 13.0 / 255.0);

pub const LOGO_PNG: &[u8] = include_bytes!("../assets/rustblock-logo.png");
pub const WORDMARK_PNG: &[u8] = include_bytes!("../assets/rustblock-wordmark.png");

/// Translucent surface used by empty states so the watermark stays visible.
pub fn translucent(color: Color, alpha: f32) -> Color {
    Color { a: alpha, ..color }
}

pub fn theme() -> Theme {
    Theme::custom(
        "RustBlock",
        iced::theme::Palette {
            background: BG,
            text: TEXT,
            primary: ACCENT,
            success: OK,
            danger: ERROR,
            warning: WARN,
        },
    )
}

fn border(color: Color, radius: f32) -> Border {
    Border {
        color,
        width: 1.0,
        radius: radius.into(),
    }
}

// ----- Containers ------------------------------------------------------

pub fn plain(fill: Color) -> impl Fn(&Theme) -> container::Style {
    move |_| container::Style {
        background: Some(Background::Color(fill)),
        ..container::Style::default()
    }
}

pub fn surface(selected: bool) -> impl Fn(&Theme) -> container::Style {
    move |_| container::Style {
        background: Some(Background::Color(if selected { PANEL_2 } else { PANEL })),
        border: border(if selected { ACCENT_DARK } else { BORDER }, 8.0),
        ..container::Style::default()
    }
}

pub fn outlined(fill: Color, line: Color, radius: f32) -> container::Style {
    container::Style {
        background: Some(Background::Color(fill)),
        border: border(line, radius),
        ..container::Style::default()
    }
}

pub fn chrome() -> impl Fn(&Theme) -> container::Style {
    |_| container::Style {
        background: Some(Background::Color(PANEL)),
        border: Border {
            color: BORDER,
            width: 0.0,
            radius: 0.0.into(),
        },
        ..container::Style::default()
    }
}

pub fn scrim() -> impl Fn(&Theme) -> container::Style {
    |_| container::Style {
        background: Some(Background::Color(Color { a: 0.72, ..INK })),
        ..container::Style::default()
    }
}

// ----- Buttons ---------------------------------------------------------

fn button_style(
    fill: Color,
    _hover: Color,
    _pressed: Color,
    text: Color,
    line: Color,
) -> button::Style {
    button::Style {
        background: Some(Background::Color(fill)),
        text_color: text,
        border: border(line, 6.0),
        shadow: Shadow::default(),
        snap: false,
    }
}

pub fn primary(_: &Theme, status: button::Status) -> button::Style {
    match status {
        button::Status::Hovered => {
            button_style(STEEL_LIGHT, STEEL_LIGHT, STEEL_LIGHT, INK, STEEL_LIGHT)
        }
        button::Status::Pressed => {
            button_style(ACCENT_DARK, ACCENT_DARK, ACCENT_DARK, INK, ACCENT_DARK)
        }
        button::Status::Disabled => button_style(PANEL_3, PANEL_3, PANEL_3, MUTED, BORDER),
        _ => button_style(ACCENT, ACCENT, ACCENT, INK, ACCENT),
    }
}

pub fn secondary(_: &Theme, status: button::Status) -> button::Style {
    match status {
        button::Status::Hovered => button_style(PANEL_2, PANEL_2, PANEL_2, TEXT, STEEL),
        button::Status::Pressed => button_style(PANEL_3, PANEL_3, PANEL_3, TEXT, ACCENT_DARK),
        button::Status::Disabled => button_style(PANEL, PANEL, PANEL, MUTED, BORDER),
        _ => button_style(PANEL_2, PANEL_2, PANEL_2, STEEL_LIGHT, BORDER),
    }
}

/// Outlined accent button: quieter than primary, stronger than secondary.
pub fn accent_outline(_: &Theme, status: button::Status) -> button::Style {
    match status {
        button::Status::Hovered => button_style(PANEL_3, PANEL_3, PANEL_3, ACCENT, ACCENT),
        button::Status::Pressed => {
            button_style(ACCENT_DARK, ACCENT_DARK, ACCENT_DARK, INK, ACCENT_DARK)
        }
        button::Status::Disabled => button_style(PANEL, PANEL, PANEL, MUTED, BORDER),
        _ => button_style(PANEL_2, PANEL_2, PANEL_2, ACCENT, ACCENT_DARK),
    }
}

pub fn ghost(_: &Theme, status: button::Status) -> button::Style {
    match status {
        button::Status::Hovered => {
            let mut style = button_style(PANEL_2, PANEL_2, PANEL_2, TEXT, PANEL_2);
            style.border.width = 0.0;
            style
        }
        button::Status::Pressed => {
            let mut style = button_style(PANEL_3, PANEL_3, PANEL_3, TEXT, PANEL_3);
            style.border.width = 0.0;
            style
        }
        button::Status::Disabled => {
            let mut style = button_style(PANEL, PANEL, PANEL, MUTED, PANEL);
            style.border.width = 0.0;
            style
        }
        _ => {
            let mut style = button_style(BG, BG, BG, MUTED, BG);
            style.background = None;
            style.border.width = 0.0;
            style
        }
    }
}

/// Navigation tab: filled when active, quiet otherwise.
pub fn tab(active: bool) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |_, status| {
        let (fill, text, line) = if active {
            (ACCENT, INK, ACCENT)
        } else {
            match status {
                button::Status::Hovered => (PANEL_2, TEXT, BORDER),
                button::Status::Pressed => (PANEL_3, TEXT, BORDER),
                _ => (BG, MUTED, BG),
            }
        };
        let mut style = button_style(fill, fill, fill, text, line);
        if !active && !matches!(status, button::Status::Hovered | button::Status::Pressed) {
            style.background = None;
        }
        style
    }
}

/// Small square window control.
pub fn window_control(danger: bool) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |_, status| {
        let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
        let fill = if hovered {
            if danger {
                ERROR
            } else {
                PANEL_3
            }
        } else {
            PANEL
        };
        let text = if hovered && danger {
            INK
        } else if hovered {
            TEXT
        } else {
            MUTED
        };
        let mut style = button_style(fill, fill, fill, text, fill);
        style.border.width = 0.0;
        style
    }
}

/// Level chips in the activity log.
pub fn log_chip(
    level_color: Color,
    active: bool,
) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |_, status| {
        let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
        if active {
            button_style(level_color, level_color, level_color, INK, level_color)
        } else if hovered {
            button_style(PANEL_2, PANEL_2, PANEL_2, level_color, BORDER)
        } else {
            let mut style = button_style(PANEL, PANEL, PANEL, level_color, BORDER);
            style.background = None;
            style
        }
    }
}

/// Segmented control used for sort/verification.
pub fn segment(selected: bool) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |_, status| {
        let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
        if selected {
            button_style(ACCENT, ACCENT, ACCENT, INK, ACCENT)
        } else if hovered {
            button_style(PANEL_2, PANEL_2, PANEL_2, TEXT, BORDER)
        } else {
            button_style(BG, BG, BG, MUTED, BORDER)
        }
    }
}

/// Rows in the inspector's block list.
pub fn list_row(selected: bool) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |_, status| {
        let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
        if selected {
            button_style(PANEL_2, PANEL_2, PANEL_2, TEXT, ACCENT_DARK)
        } else if hovered {
            button_style(PANEL_2, PANEL_2, PANEL_2, TEXT, BORDER)
        } else {
            let mut style = button_style(BG, BG, BG, STEEL_LIGHT, BG);
            style.border.width = 0.0;
            style
        }
    }
}

// ----- Scrollbars ------------------------------------------------------

/// Thin steel scrollbar on a transparent rail: visible when needed, never a
/// bright band between two panels.
pub fn scrollbar(_: &Theme, _: scrollable::Status) -> scrollable::Style {
    let rail = scrollable::Rail {
        background: None,
        border: Border {
            color: Color::TRANSPARENT,
            width: 0.0,
            radius: 3.0.into(),
        },
        scroller: scrollable::Scroller {
            background: Background::Color(Color { a: 0.6, ..STEEL }),
            border: Border {
                color: Color::TRANSPARENT,
                width: 0.0,
                radius: 3.0.into(),
            },
        },
    };

    scrollable::Style {
        container: container::Style::default(),
        vertical_rail: rail,
        horizontal_rail: rail,
        gap: None,
        auto_scroll: scrollable::AutoScroll {
            background: Background::Color(PANEL_2),
            border: border(BORDER, 6.0),
            shadow: Shadow::default(),
            icon: MUTED,
        },
    }
}

// ----- Fields ----------------------------------------------------------

pub fn text_field(_: &Theme, status: text_input::Status) -> text_input::Style {
    let line = match status {
        text_input::Status::Focused { .. } => ACCENT,
        text_input::Status::Hovered => STEEL,
        _ => BORDER,
    };
    text_input::Style {
        background: Background::Color(WELL),
        border: border(line, 6.0),
        icon: MUTED,
        placeholder: MUTED,
        value: TEXT,
        selection: Color { a: 0.45, ..ACCENT },
    }
}

pub fn checkbox_style(_: &Theme, status: checkbox::Status) -> checkbox::Style {
    let checked = match status {
        checkbox::Status::Active { is_checked }
        | checkbox::Status::Hovered { is_checked }
        | checkbox::Status::Disabled { is_checked } => is_checked,
    };
    checkbox::Style {
        background: Background::Color(if checked { ACCENT } else { PANEL_2 }),
        icon_color: INK,
        border: border(if checked { ACCENT } else { BORDER }, 4.0),
        text_color: Some(STEEL_LIGHT),
    }
}

pub fn progress(_: &Theme) -> progress_bar::Style {
    progress_bar::Style {
        background: Background::Color(PANEL_3),
        bar: Background::Color(ACCENT),
        border: Border {
            color: BORDER,
            width: 0.0,
            radius: 3.0.into(),
        },
    }
}

// ----- Images ----------------------------------------------------------

/// The square mark, decoded once and handed to iced as raw RGBA.
pub fn logo_handle() -> iced::widget::image::Handle {
    let decoded = image::load_from_memory(LOGO_PNG)
        .expect("embedded RustBlock logo must be a valid PNG")
        .to_rgba8();
    let (width, height) = decoded.dimensions();
    iced::widget::image::Handle::from_rgba(width, height, decoded.into_raw())
}

/// Aspect ratio of the wordmark, used to keep it undistorted.
pub fn wordmark_aspect() -> f32 {
    match image::load_from_memory(WORDMARK_PNG) {
        Ok(decoded) => decoded.width().max(1) as f32 / decoded.height().max(1) as f32,
        Err(_) => 4.0,
    }
}

/// The wordmark with its flat canvas removed, cropped to the artwork bounds.
pub fn wordmark_handle() -> iced::widget::image::Handle {
    let Some((width, height, pixels)) = wordmark_rgba() else {
        return logo_handle();
    };
    iced::widget::image::Handle::from_rgba(width, height, pixels)
}

fn wordmark_rgba() -> Option<(u32, u32, Vec<u8>)> {
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
    Some((decoded.width(), decoded.height(), decoded.into_raw()))
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

pub fn app_icon() -> iced::window::Icon {
    let decoded = image::load_from_memory(LOGO_PNG)
        .expect("embedded RustBlock logo must be a valid PNG")
        .to_rgba8();
    let resized =
        image::imageops::resize(&decoded, 128, 128, image::imageops::FilterType::Lanczos3);
    iced::window::icon::from_rgba(resized.into_raw(), 128, 128).expect("128x128 RGBA icon is valid")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn channel(value: f32) -> f32 {
        if value <= 0.04045 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    }

    fn luminance(color: Color) -> f32 {
        0.2126 * channel(color.r) + 0.7152 * channel(color.g) + 0.0722 * channel(color.b)
    }

    fn contrast(a: Color, b: Color) -> f32 {
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

    #[test]
    fn wordmark_decodes_with_visible_pixels() {
        let (width, height, pixels) = wordmark_rgba().expect("wordmark must decode");
        assert!(width > 10 && height > 5);
        assert!(pixels.chunks_exact(4).any(|pixel| pixel[3] > 200));
    }
}
