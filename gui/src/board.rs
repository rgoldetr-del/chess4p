//! board.rs — отрисовка доски 14×14 для 4-игрочных шахмат и обработка кликов.

use egui::{Color32, Pos2, Rect, Sense, Stroke, Ui, Vec2};

use chess4p_game::{
    is_promoted, piece_color, piece_type, sq_index,
    PieceType, GameState,
};

pub const CELL: f32 = 38.0;
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

/// Unicode-глиф фигуры (нужен шрифт с шахматными символами, напр. Segoe UI Symbol)
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

/// Квадрант игрока для подсветки границ:
/// Red — низ-право, Blue — низ-лево, Yellow — верх-лево, Green — верх-право
fn quadrant_color(x: usize, y: usize) -> Color32 {
    let top = y > 7;
    let left = x <= 7;
    match (top, left) {
        (false, false) => PLAYER_COLORS[0], // red
        (false, true) => PLAYER_COLORS[1],  // blue
        (true, true) => PLAYER_COLORS[2],   // yellow
        (true, false) => PLAYER_COLORS[3],  // green
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
    font_ok: bool,
) -> BoardResponse {
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(BOARD_PX), Sense::click());
    let painter = ui.painter();
    let origin = rect.min;

    // Фон и клетки
    for y in 1..=14usize {
        for x in 1..=14usize {
            let p0 = origin + Vec2::new((x - 1) as f32 * CELL, (14 - y) as f32 * CELL);
            let r = Rect::from_min_size(p0, Vec2::splat(CELL));
            let checker = if (x + y) % 2 == 1 { DARK } else { LIGHT };
            painter.rect_filled(r, 0.0, checker);
        }
    }

    // Цветные границы квадрантов
    for y in 1..=14usize {
        for x in 1..=14usize {
            // верхняя клетка квадрантного ряда/столбца — тонкая рамка цвета игрока
            let p0 = origin + Vec2::new((x - 1) as f32 * CELL, (14 - y) as f32 * CELL);
            let r = Rect::from_min_size(p0, Vec2::splat(CELL));
            let qc = quadrant_color(x, y);
            let mut stroke = Stroke::NONE;
            if x == 1 || x == 14 || y == 1 || y == 14 {
                stroke = Stroke::new(3.0_f32, qc);
            } else if (x == 7 && y == 7) || (x == 8 && y == 7) || (x == 7 && y == 8) || (x == 8 && y == 8) {
                stroke = Stroke::new(2.0_f32, qc);
            }
            if stroke.width > 0.0 {
                painter.rect_stroke(r, 0.0, stroke, egui::StrokeKind::Inside);
            }
        }
    }

    // Подсветка последнего хода
    if let Some((from, to)) = last_move {
        for sq in [from, to] {
            highlight_cell(painter, origin, sq, Color32::from_rgba_unmultiplied(255, 220, 60, 110));
        }
    }

    // Подсветка выбранной фигуры
    if let Some(sel) = selected {
        highlight_cell(painter, origin, sel, Color32::from_rgba_unmultiplied(90, 170, 255, 130));
    }

    // Подсветка легальных ходов
    for &(tx, ty) in legal_targets {
        let p0 = origin + Vec2::new((tx - 1) as f32 * CELL, (14 - ty) as f32 * CELL);
        let center = p0 + Vec2::splat(CELL) * 0.5;
        let target_idx = sq_index(tx, ty);
        let occupied = state.board[target_idx] != chess4p_game::board::EMPTY;
        if occupied {
            // взятие — кольцо
            painter.circle_stroke(center, CELL * 0.46, Stroke::new(4.0_f32, Color32::from_rgba_unmultiplied(40, 200, 90, 200)));
        } else {
            painter.circle_filled(center, CELL * 0.16, Color32::from_rgba_unmultiplied(40, 200, 90, 160));
        }
    }

    // Фигуры
    for y in 1..=14usize {
        for x in 1..=14usize {
            let idx = sq_index(x, y);
            let val = state.board[idx];
            if val == chess4p_game::board::EMPTY {
                continue;
            }
            let Some(color) = piece_color(val) else { continue };
            let Some(pt) = piece_type(val) else { continue };
            let promoted = is_promoted(val);

            let p0 = origin + Vec2::new((x - 1) as f32 * CELL, (14 - y) as f32 * CELL);
            let center = p0 + Vec2::splat(CELL * 0.5);

            // Мёртвые/сдавшиеся — полупрозрачные
            let alpha = if state.dead_colors[color.index()] || state.resigned[color.index()] {
                90
            } else {
                255
            };

            let text = if font_ok {
                piece_glyph(pt, promoted)
            } else {
                piece_letter(pt, promoted)
            };
            let font_size = if font_ok { CELL * 0.85 } else { CELL * 0.55 };
            painter.text(
                center,
                egui::Align2::CENTER_CENTER,
                text,
                egui::FontId::proportional(font_size),
                Color32::from_rgba_unmultiplied(PLAYER_COLORS[color.index()].r(), PLAYER_COLORS[color.index()].g(), PLAYER_COLORS[color.index()].b(), alpha),
            );
        }
    }

    // Координатная сетка-подписи
    for x in 1..=14usize {
        let p = origin + Vec2::new((x - 1) as f32 * CELL + CELL * 0.5, BOARD_PX + 9.0);
        painter.text(
            p,
            egui::Align2::CENTER_CENTER,
            &((b'a' + (x as u8 - 1)) as char).to_string(),
            egui::FontId::proportional(11.0),
            Color32::GRAY,
        );
    }
    for y in 1..=14usize {
        let p = origin + Vec2::new(-9.0, (14 - y) as f32 * CELL + CELL * 0.5);
        painter.text(
            p,
            egui::Align2::CENTER_CENTER,
            &y.to_string(),
            egui::FontId::proportional(11.0),
            Color32::GRAY,
        );
    }

    // Обработка клика
    let clicked = response.interact_pointer_pos().map(|pos: Pos2| {
        let dx = (pos.x - origin.x) / CELL;
        let dy = (pos.y - origin.y) / CELL;
        let x = (dx.floor() as i64 + 1).clamp(1, 14) as usize;
        let y = (14.0 - dy).floor() as i64 + 1;
        let y = y.clamp(1, 14) as usize;
        (x, y)
    });

    BoardResponse { clicked }
}

fn highlight_cell(painter: &egui::Painter, origin: Pos2, (x, y): (usize, usize), color: Color32) {
    let p0 = origin + Vec2::new((x - 1) as f32 * CELL, (14 - y) as f32 * CELL);
    let r = Rect::from_min_size(p0, Vec2::splat(CELL));
    painter.rect_filled(r, 0.0, color);
}