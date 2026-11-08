//! board.rs — отрисовка доски 14×14 для 4-игрочных шахмат и обработка кликов.

use std::collections::HashMap;
use std::path::Path;

use egui::{Color32, Pos2, Rect, Sense, Stroke, Ui, Vec2};

use chess4p_game::{
    is_promoted, piece_color, piece_type, sq_index, Color, PieceType, GameState,
};

pub const CELL: f32 = 50.0;
pub const BOARD_PX: f32 = CELL * 14.0;

// Цвета игроков (как в bot/config.py COLOR_ORDER: red, blue, yellow, green)
pub const PLAYER_COLORS: [Color32; 4] = [
    Color32::from_rgb(196, 40, 40),   // red
    Color32::from_rgb(40, 90, 196),   // blue
    Color32::from_rgb(210, 190, 30),  // yellow
    Color32::from_rgb(40, 160, 60),   // green
];

pub const PLAYER_NAMES: [&str; 4] = ["Красный", "Синий", "Жёлтый", "Зелёный"];

const LIGHT: Color32 = Color32::from_rgb(235, 236, 208);
const DARK: Color32 = Color32::from_rgb(115, 149, 82);
const VOID: Color32 = Color32::from_rgb(38, 40, 42);

/// Пустые 3×3 углы доски (в них ничего не может стоять)
pub fn is_void(x: usize, y: usize) -> bool {
    (x <= 3 || x >= 12) && (y <= 3 || y >= 12)
}

// ============================================================
// Спрайты фигур (PNG из assets/pieces)
// ============================================================

pub struct PieceTextures {
    map: HashMap<String, egui::TextureHandle>,
}

fn color_dir_name(c: Color) -> &'static str {
    match c {
        Color::Red => "red",
        Color::Blue => "blue",
        Color::Yellow => "yellow",
        Color::Green => "green",
    }
}

fn piece_file_name(pt: PieceType) -> &'static str {
    match pt {
        PieceType::Pawn => "pawn",
        PieceType::Knight => "knight",
        PieceType::Bishop => "bishop",
        PieceType::Rook => "rook",
        PieceType::Queen | PieceType::King => "",
    }
}

fn piece_file_name_promoted() -> &'static str {
    "queen"
}

impl PieceTextures {
    /// Загружает {color}-{piece}.png из каталога. None — каталог не найден,
    /// тогда используется fallback-отрисовка буквами.
    pub fn load(ctx: &egui::Context, dir: &Path) -> Option<Self> {
        let mut map = HashMap::new();
        let mut loaded = 0;

        for ci in 0..4 {
            let color = Color::ALL[ci];
            let cname = color_dir_name(color);
            for pname in ["pawn", "knight", "bishop", "rook", "queen", "king"] {
                let path = dir.join(format!("{cname}-{pname}.png"));
                if !path.exists() {
                    continue;
                }
                let bytes = match std::fs::read(&path) {
                    Ok(b) => b,
                    Err(_) => continue,
                };
                let img = image::load_from_memory(&bytes).ok()?;
                let rgba = img.to_rgba8();
                let size = [rgba.width() as usize, rgba.height() as usize];
                let color_image = egui::ColorImage::from_rgba_unmultiplied(
                    size,
                    rgba.as_raw(),
                );
                let tex = ctx.load_texture(
                    format!("piece-{cname}-{pname}"),
                    color_image,
                    egui::TextureOptions::LINEAR,
                );
                map.insert(format!("{cname}-{pname}"), tex);
                loaded += 1;
            }
        }

        if loaded == 0 {
            return None;
        }
        Some(Self { map })
    }

    fn get(&self, c: Color, pt: PieceType, promoted: bool) -> Option<&egui::TextureHandle> {
        let cname = color_dir_name(c);
        let pname = if promoted {
            piece_file_name_promoted()
        } else {
            match piece_file_name(pt) {
                "" => match pt {
                    PieceType::Queen => "queen",
                    _ => "king",
                },
                s => s,
            }
        };
        self.map.get(&format!("{cname}-{pname}"))
    }
}

/// Fallback-глиф (если спрайты не найдены)
fn piece_glyph(pt: PieceType, promoted: bool) -> &'static str {
    if promoted {
        return "♕";
    }
    match pt {
        PieceType::Pawn => "♟",
        PieceType::Knight => "♞",
        PieceType::Bishop => "♝",
        PieceType::Rook => "♜",
        PieceType::Queen => "♛",
        PieceType::King => "♚",
    }
}

fn piece_letter(pt: PieceType, promoted: bool) -> &'static str {
    if promoted {
        return "Q";
    }
    match pt {
        PieceType::Pawn => "P",
        PieceType::Knight => "N",
        PieceType::Bishop => "B",
        PieceType::Rook => "R",
        PieceType::Queen => "Q",
        PieceType::King => "K",
    }
}

pub struct BoardResponse {
    pub clicked: Option<(usize, usize)>,
}

/// Отрисовка доски. y=1 — низ (красные), y=14 — верх (жёлтые).
pub fn draw_board(
    ui: &mut Ui,
    state: &GameState,
    selected: Option<(usize, usize)>,
    legal_targets: &[(usize, usize)],
    last_move: Option<((usize, usize), (usize, usize))>,
    textures: Option<&PieceTextures>,
    font_ok: bool,
) -> BoardResponse {
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(BOARD_PX), Sense::click());
    let painter = ui.painter();
    let origin = rect.min;

    // Фон пустых углов
    painter.rect_filled(rect, 2.0, VOID);

    // Клетки — бело-зелёная шахматная раскраска, углы пустые, без обводок
    for y in 1..=14usize {
        for x in 1..=14usize {
            if is_void(x, y) {
                continue;
            }
            let p0 = origin + Vec2::new((x - 1) as f32 * CELL, (14 - y) as f32 * CELL);
            let r = Rect::from_min_size(p0, Vec2::splat(CELL));
            let checker = if (x + y) % 2 == 1 { DARK } else { LIGHT };
            painter.rect_filled(r, 0.0, checker);
        }
    }

    // Подсветка последнего хода
    if let Some((from, to)) = last_move {
        for sq in [from, to] {
            if !is_void(sq.0, sq.1) {
                highlight_cell(painter, origin, sq, Color32::from_rgba_unmultiplied(255, 220, 60, 110));
            }
        }
    }

    // Подсветка выбранной фигуры
    if let Some(sel) = selected {
        highlight_cell(painter, origin, sel, Color32::from_rgba_unmultiplied(90, 170, 255, 130));
    }

    // Подсветка легальных ходов
    for &(tx, ty) in legal_targets {
        if is_void(tx, ty) {
            continue;
        }
        let p0 = origin + Vec2::new((tx - 1) as f32 * CELL, (14 - ty) as f32 * CELL);
        let center = p0 + Vec2::splat(CELL) * 0.5;
        let target_idx = sq_index(tx, ty);
        let occupied = state.board[target_idx] != chess4p_game::EMPTY;
        if occupied {
            // взятие — кольцо
            painter.circle_stroke(
                center,
                CELL * 0.46,
                Stroke::new(4.0_f32, Color32::from_rgba_unmultiplied(40, 200, 90, 200)),
            );
        } else {
            painter.circle_filled(
                center,
                CELL * 0.16,
                Color32::from_rgba_unmultiplied(40, 200, 90, 160),
            );
        }
    }

    // Фигуры
    for y in 1..=14usize {
        for x in 1..=14usize {
            if is_void(x, y) {
                continue;
            }
            let idx = sq_index(x, y);
            let val = state.board[idx];
            if val == chess4p_game::EMPTY {
                continue;
            }
            let Some(color) = piece_color(val) else { continue };
            let Some(pt) = piece_type(val) else { continue };
            let promoted = is_promoted(val);

            let p0 = origin + Vec2::new((x - 1) as f32 * CELL, (14 - y) as f32 * CELL);
            let cell_rect = Rect::from_min_size(p0, Vec2::splat(CELL));

            // Мёртвые/сдавшиеся — полупрозрачные
            let alpha = if state.dead_colors[color.index()] || state.resigned[color.index()] {
                100u8
            } else {
                255u8
            };

            if let Some(texs) = textures {
                if let Some(tex) = texs.get(color, pt, promoted) {
                    let pad = CELL * 0.06;
                    let img_rect = Rect::from_min_size(
                        cell_rect.min + Vec2::splat(pad),
                        Vec2::splat(CELL - pad * 2.0),
                    );
                    painter.image(
                        tex.id(),
                        img_rect,
                        Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                        Color32::from_rgba_unmultiplied(255, 255, 255, alpha),
                    );
                    continue;
                }
            }

            // Fallback: буква цветом игрока
            let center = cell_rect.center();
            let glyph = if font_ok {
                piece_glyph(pt, promoted)
            } else {
                piece_letter(pt, promoted)
            };
            let font_size = if font_ok { CELL * 0.85 } else { CELL * 0.55 };
            painter.text(
                center,
                egui::Align2::CENTER_CENTER,
                glyph,
                egui::FontId::proportional(font_size),
                Color32::from_rgba_unmultiplied(
                    PLAYER_COLORS[color.index()].r(),
                    PLAYER_COLORS[color.index()].g(),
                    PLAYER_COLORS[color.index()].b(),
                    alpha,
                ),
            );
        }
    }

    // Координаты
    for x in 1..=14usize {
        let p = origin + Vec2::new((x - 1) as f32 * CELL + CELL * 0.5, BOARD_PX + 10.0);
        painter.text(
            p,
            egui::Align2::CENTER_CENTER,
            &((b'a' + (x as u8 - 1)) as char).to_string(),
            egui::FontId::proportional(12.0),
            Color32::GRAY,
        );
    }
    for y in 1..=14usize {
        let p = origin + Vec2::new(-11.0, (14 - y) as f32 * CELL + CELL * 0.5);
        painter.text(
            p,
            egui::Align2::CENTER_CENTER,
            &y.to_string(),
            egui::FontId::proportional(12.0),
            Color32::GRAY,
        );
    }

    // Обработка клика (клики по пустым углам игнорируются)
    let clicked = response.interact_pointer_pos().and_then(|pos: Pos2| {
        let dx = (pos.x - origin.x) / CELL;
        let dy = (pos.y - origin.y) / CELL;
        if !(0.0..14.0).contains(&dx) || !(0.0..14.0).contains(&dy) {
            return None;
        }
        let x = dx.floor() as usize + 1;
        let y = (14.0 - dy).floor() as usize + 1;
        let y = y.clamp(1, 14);
        if is_void(x, y) {
            return None;
        }
        Some((x, y))
    });

    BoardResponse { clicked }
}

fn highlight_cell(painter: &egui::Painter, origin: Pos2, (x, y): (usize, usize), color: Color32) {
    let p0 = origin + Vec2::new((x - 1) as f32 * CELL, (14 - y) as f32 * CELL);
    let r = Rect::from_min_size(p0, Vec2::splat(CELL));
    painter.rect_filled(r, 0.0, color);
}
