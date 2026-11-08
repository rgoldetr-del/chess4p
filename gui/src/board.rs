//! board.rs — отрисовка доски 14×14 для 4-игрочных шахмат и обработка кликов.

use std::collections::HashMap;
use std::path::Path;

use egui::{Color32, Pos2, Rect, Sense, Stroke, Ui, Vec2};

use chess4p_game::{
    is_promoted, piece_color, piece_type, sq_index, Color, PieceType, GameState,
};

pub const DEFAULT_CELL: f32 = 50.0;
pub const MIN_CELL: f32 = 24.0;
pub const MAX_CELL: f32 = 96.0;

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

/// Цвета разметки: зелёный (по умолчанию), красный (Shift), синий (Ctrl), жёлтый (Alt)
/// Цвет пометки по индексу: 0 зелёный, 1 красный (Shift), 2 синий (Ctrl), 3 жёлтый (Alt)
#[inline]
pub fn mark_color(ci: usize) -> Color32 {
    const COLORS: [(u8, u8, u8); 4] = [
        (0, 160, 70),    // зелёный
        (200, 40, 40),   // красный (Shift)
        (50, 110, 220),  // синий (Ctrl)
        (230, 200, 30),  // жёлтый (Alt)
    ];
    let (r, g, b) = COLORS[ci.min(3)];
    Color32::from_rgba_unmultiplied(r, g, b, 170)
}

/// Пользовательские пометки на доске: стрелки и круги (правая кнопка мыши)
#[derive(Default)]
pub struct BoardMarks {
    /// (fx, fy, tx, ty, color_idx)
    pub arrows: Vec<(usize, usize, usize, usize, usize)>,
    /// ((x, y), color_idx)
    pub circles: Vec<((usize, usize), usize)>,
    /// Текущий drag правой кнопкой (откуда начал тянуть)
    drag_from: Option<(usize, usize)>,
}

impl BoardMarks {
    pub fn clear(&mut self) {
        self.arrows.clear();
        self.circles.clear();
        self.drag_from = None;
    }

    fn toggle_arrow(&mut self, from: (usize, usize), to: (usize, usize), ci: usize) {
        let pos = self
            .arrows
            .iter()
            .position(|&(a, b, c, d, _)| (a, b) == from && (c, d) == to);
        match pos {
            Some(i) => {
                self.arrows.remove(i);
            }
            None => self.arrows.push((from.0, from.1, to.0, to.1, ci)),
        }
    }

    fn toggle_circle(&mut self, sq: (usize, usize), ci: usize) {
        let pos = self.circles.iter().position(|&(s, _)| s == sq);
        match pos {
            Some(i) => {
                self.circles.remove(i);
            }
            None => self.circles.push((sq, ci)),
        }
    }
}

pub struct BoardResponse {
    pub clicked: Option<(usize, usize)>,
    pub hovered: bool,
}

/// Отрисовка доски. y=1 — низ (красные), y=14 — верх (жёлтые).
#[allow(clippy::too_many_arguments)]
pub fn draw_board(
    ui: &mut Ui,
    cell: f32,
    void_color: Color32,
    state: &GameState,
    selected: Option<(usize, usize)>,
    legal_targets: &[(usize, usize)],
    last_move: Option<((usize, usize), (usize, usize))>,
    textures: Option<&PieceTextures>,
    font_ok: bool,
    marks: &mut BoardMarks,
) -> BoardResponse {
    let board_px = cell * 14.0;
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(board_px), Sense::click());
    let painter = ui.painter();
    let origin = rect.min;

    // Фон пустых углов
    painter.rect_filled(rect, 2.0, void_color);

    // Клетки — бело-зелёная шахматная раскраска, углы пустые, без обводок
    for y in 1..=14usize {
        for x in 1..=14usize {
            if is_void(x, y) {
                continue;
            }
            let p0 = origin + Vec2::new((x - 1) as f32 * cell, (14 - y) as f32 * cell);
            let r = Rect::from_min_size(p0, Vec2::splat(cell));
            let checker = if (x + y) % 2 == 1 { DARK } else { LIGHT };
            painter.rect_filled(r, 0.0, checker);
        }
    }

    // Подсветка последнего хода
    if let Some((from, to)) = last_move {
        for sq in [from, to] {
            if !is_void(sq.0, sq.1) {
                highlight_cell(painter, origin, cell, sq, Color32::from_rgba_unmultiplied(255, 220, 60, 110));
            }
        }
    }

    // Подсветка выбранной фигуры
    if let Some(sel) = selected {
        highlight_cell(painter, origin, cell, sel, Color32::from_rgba_unmultiplied(90, 170, 255, 130));
    }

    // Подсветка легальных ходов
    for &(tx, ty) in legal_targets {
        if is_void(tx, ty) {
            continue;
        }
        let p0 = origin + Vec2::new((tx - 1) as f32 * cell, (14 - ty) as f32 * cell);
        let center = p0 + Vec2::splat(cell) * 0.5;
        let target_idx = sq_index(tx, ty);
        let occupied = state.board[target_idx] != chess4p_game::EMPTY;
        if occupied {
            // взятие — кольцо
            painter.circle_stroke(
                center,
                cell * 0.46,
                Stroke::new(4.0_f32, Color32::from_rgba_unmultiplied(40, 200, 90, 200)),
            );
        } else {
            painter.circle_filled(
                center,
                cell * 0.16,
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

            let p0 = origin + Vec2::new((x - 1) as f32 * cell, (14 - y) as f32 * cell);
            let cell_rect = Rect::from_min_size(p0, Vec2::splat(cell));

            // Мёртвые/сдавшиеся — полупрозрачные
            let alpha = if state.dead_colors[color.index()] || state.resigned[color.index()] {
                100u8
            } else {
                255u8
            };

            if let Some(texs) = textures {
                if let Some(tex) = texs.get(color, pt, promoted) {
                    let pad = cell * 0.06;
                    let img_rect = Rect::from_min_size(
                        cell_rect.min + Vec2::splat(pad),
                        Vec2::splat(cell - pad * 2.0),
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
            let font_size = if font_ok { cell * 0.85 } else { cell * 0.55 };
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
        let p = origin + Vec2::new((x - 1) as f32 * cell + cell * 0.5, board_px + 10.0);
        painter.text(
            p,
            egui::Align2::CENTER_CENTER,
            &((b'a' + (x as u8 - 1)) as char).to_string(),
            egui::FontId::proportional(12.0),
            Color32::GRAY,
        );
    }
    for y in 1..=14usize {
        let p = origin + Vec2::new(-11.0, (14 - y) as f32 * cell + cell * 0.5);
        painter.text(
            p,
            egui::Align2::CENTER_CENTER,
            &y.to_string(),
            egui::FontId::proportional(12.0),
            Color32::GRAY,
        );
    }

    // --- Правая кнопка: круги и стрелки (как на chess.com) ---
    let (mods, sec_pressed, sec_released, sec_down, pointer_pos) = ui.input(|i| {
        (
            i.modifiers,
            i.pointer.button_pressed(egui::PointerButton::Secondary),
            i.pointer.button_released(egui::PointerButton::Secondary),
            i.pointer.button_down(egui::PointerButton::Secondary),
            i.pointer.interact_pos(),
        )
    });
    let mark_color_idx = if mods.shift {
        1
    } else if mods.ctrl {
        2
    } else if mods.alt {
        3
    } else {
        0
    };

    let square_at = |pos: Option<Pos2>| -> Option<(usize, usize)> {
        let pos = pos?;
        let dx = (pos.x - origin.x) / cell;
        let dy = (pos.y - origin.y) / cell;
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
    };

    // ЛЕВАЯ кнопка — очистить всё нарисованное
    if response.clicked() {
        marks.clear();
    }

    if sec_pressed {
        marks.drag_from = square_at(pointer_pos);
    }
    if sec_released {
        if let Some(from) = marks.drag_from.take() {
            match square_at(pointer_pos) {
                Some(to) if to == from => marks.toggle_circle(from, mark_color_idx),
                Some(to) => marks.toggle_arrow(from, to, mark_color_idx),
                None => {}
            }
        }
    }

    // --- Рисуем круги ---
    for &((x, y), ci) in &marks.circles {
        let p0 = origin + Vec2::new((x - 1) as f32 * cell, (14 - y) as f32 * cell);
        let center = p0 + Vec2::splat(cell) * 0.5;
        painter.circle_stroke(
            center,
            cell * 0.44,
            Stroke::new(cell * 0.09, mark_color(ci)),
        );
    }

    // --- Рисуем стрелки (поверх фигур) ---
    for &(fx, fy, tx, ty, ci) in &marks.arrows {
        draw_arrow(painter, origin, cell, (fx, fy), (tx, ty), mark_color(ci));
    }

    // --- Превью стрелки во время перетаскивания ---
    if sec_down && marks.drag_from.is_some() {
        let from = marks.drag_from.unwrap();
        if let Some(to) = square_at(pointer_pos) {
            let preview = Color32::from_rgba_unmultiplied(
                mark_color(mark_color_idx).r(),
                mark_color(mark_color_idx).g(),
                mark_color(mark_color_idx).b(),
                90,
            );
            if to != from {
                draw_arrow(painter, origin, cell, from, to, preview);
            } else {
                // превью круга
                let p0 = origin + Vec2::new((from.0 - 1) as f32 * cell, (14 - from.1) as f32 * cell);
                painter.circle_stroke(
                    p0 + Vec2::splat(cell) * 0.5,
                    cell * 0.44,
                    Stroke::new(cell * 0.09, preview),
                );
            }
        }
    }

    // Обработка клика ЛЕВОЙ кнопкой (клики по пустым углам игнорируются)
    let clicked = if response.clicked() {
        response.interact_pointer_pos().and_then(|pos: Pos2| {
        let dx = (pos.x - origin.x) / cell;
        let dy = (pos.y - origin.y) / cell;
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
        })
    } else {
        None
    };

    BoardResponse {
        clicked,
        hovered: response.hovered(),
    }
}

/// Стрелка в стиле chess.com: полупрозрачная линия с наконечником
fn draw_arrow(
    painter: &egui::Painter,
    origin: Pos2,
    cell: f32,
    from: (usize, usize),
    to: (usize, usize),
    color: Color32,
) {
    let c = |(x, y): (usize, usize)| -> Pos2 {
        origin + Vec2::new((x - 1) as f32 * cell + cell * 0.5, (14 - y) as f32 * cell + cell * 0.5)
    };
    let a = c(from);
    let b = c(to);
    let dir = Vec2::new(b.x - a.x, b.y - a.y);
    let len = dir.length();
    if len < 1.0 {
        return;
    }
    let unit = dir / len;

    // Хвост отступает от центра на 0.35 клетки, наконечник — треугольник 0.42 клетки
    let head = 0.42 * cell;
    let tail = 0.35 * cell;
    let start = a + unit * tail;
    let end = b - unit * head;

    if (end - start).length() > 2.0 {
        painter.line_segment([start, end], Stroke::new(cell * 0.14, color));
    }

    // Наконечник (треугольник)
    let perp = Vec2::new(-unit.y, unit.x);
    let tip = b - unit * cell * 0.08;
    let p1 = tip - unit * head + perp * head * 0.6;
    let p2 = tip - unit * head - perp * head * 0.6;
    let shape = egui::Shape::convex_polygon(
        vec![tip, p1, p2],
        color,
        Stroke::NONE,
    );
    painter.add(shape);
}

fn highlight_cell(
    painter: &egui::Painter,
    origin: Pos2,
    cell: f32,
    (x, y): (usize, usize),
    color: Color32,
) {
    let p0 = origin + Vec2::new((x - 1) as f32 * cell, (14 - y) as f32 * cell);
    let r = Rect::from_min_size(p0, Vec2::splat(cell));
    painter.rect_filled(r, 0.0, color);
}
