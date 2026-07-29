/// Базовые типы и кодирование фигур.

pub type Board = [u8; 196];
pub const EMPTY: u8 = 0;
pub const PROMOTED_FLAG: u8 = 0x80;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Color {
    Red,
    Blue,
    Yellow,
    Green,
}

impl Color {
    #[inline]
    pub fn next(&self) -> Color {
        match self {
            Color::Red => Color::Blue,
            Color::Blue => Color::Yellow,
            Color::Yellow => Color::Green,
            Color::Green => Color::Red,
        }
    }

    #[inline]
    pub fn index(&self) -> usize {
        match self {
            Color::Red => 0,
            Color::Blue => 1,
            Color::Yellow => 2,
            Color::Green => 3,
        }
    }

    pub const ALL: [Color; 4] = [Color::Red, Color::Blue, Color::Yellow, Color::Green];
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum PieceType {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

impl PieceType {
    /// Очки за взятие (по правилам FFA Chess.com).
    #[inline]
    pub fn capture_points(self, promoted: bool) -> i32 {
        match self {
            PieceType::Pawn => 1,
            PieceType::Knight => 3,
            PieceType::Bishop => 5,
            PieceType::Rook => 5,
            PieceType::Queen => if promoted { 1 } else { 9 },
            PieceType::King => 0, // За короля очки даются через мат (+20)
        }
    }
}

#[inline]
pub fn make_piece_val(c: Color, p: PieceType, promoted: bool) -> u8 {
    let c_idx = c.index();
    let p_idx = match p {
        PieceType::Pawn => 1,
        PieceType::Knight => 2,
        PieceType::Bishop => 3,
        PieceType::Rook => 4,
        PieceType::Queen => 5,
        PieceType::King => 6,
    };
    let mut val = (c_idx * 6 + p_idx) as u8;
    if promoted {
        val |= PROMOTED_FLAG;
    }
    val
}

#[inline]
pub fn piece_color(val: u8) -> Option<Color> {
    if val == EMPTY {
        return None;
    }
    let c_idx = ((val & 0x7F) - 1) / 6;
    match c_idx {
        0 => Some(Color::Red),
        1 => Some(Color::Blue),
        2 => Some(Color::Yellow),
        3 => Some(Color::Green),
        _ => None,
    }
}

#[inline]
pub fn piece_type(val: u8) -> Option<PieceType> {
    if val == EMPTY {
        return None;
    }
    let p_idx = ((val & 0x7F) - 1) % 6;
    match p_idx {
        0 => Some(PieceType::Pawn),
        1 => Some(PieceType::Knight),
        2 => Some(PieceType::Bishop),
        3 => Some(PieceType::Rook),
        4 => Some(PieceType::Queen),
        5 => Some(PieceType::King),
        _ => None,
    }
}

#[inline]
pub fn is_promoted(val: u8) -> bool {
    val & PROMOTED_FLAG != 0
}

/// Проверка валидности клетки на 14×14 доске с вырезанными углами.
#[inline]
pub fn is_valid_square(x: i32, y: i32) -> bool {
    if x < 1 || x > 14 || y < 1 || y > 14 {
        return false;
    }
    let in_x = x <= 3 || x >= 12;
    let in_y = y <= 3 || y >= 12;
    !(in_x && in_y)
}

#[inline]
pub fn sq_index(x: usize, y: usize) -> usize {
    (y - 1) * 14 + (x - 1)
}

#[inline]
pub fn sq_from_index(idx: usize) -> (usize, usize) {
    ((idx % 14) + 1, (idx / 14) + 1)
}