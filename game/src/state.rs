use crate::board::*;
use crate::movegen;
use crate::zobrist::get_zobrist;
use std::collections::HashMap;

// ============================================================
// Структуры
// ============================================================

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Move {
    pub from: (usize, usize),
    pub to: (usize, usize),
    pub piece: PieceType,
    pub color: Color,
    pub captured: Option<u8>,
    pub promotion: Option<PieceType>,
    pub castle: u8,
    pub en_passant: bool,
    pub check: bool,
    pub checkmate: bool,
    pub skipped: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct EnPassantInfo {
    pub target_sq: (usize, usize),
    pub victim_sq: (usize, usize),
    pub creator: Color,
}

#[derive(Clone)]
pub struct BoardState {
    pub board: Board,
    pub turn: Color,
    pub en_passants: [Option<EnPassantInfo>; 4],
    pub castling_rights: [[bool; 4]; 4],
    pub dead_colors: [bool; 4],
    pub resigned: [bool; 4],
    pub scores: [i32; 4],
    pub zobrist_hash: u64,
    pub king_positions: [(usize, usize); 4],
    pub piece_positions: [Vec<usize>; 4],
    pub halfmove_clock: usize,

    // Новые поля для совместимости истории
    pub last_move: Option<Move>,
    pub last_active_mover: Option<Color>,
    pub ply_count: usize,
}

/// Undo info для MCTS make/unmake.
#[derive(Clone)]
pub struct UndoInfo {
    pub from_idx: usize,
    pub to_idx: usize,
    pub piece_val: u8,
    pub captured_val: u8,
    pub captured_idx: usize,

    pub en_passant: bool,
    pub ep_captured_idx: Option<usize>,
    pub ep_captured_val: u8,
    pub ep_creator_idx: Option<usize>,

    pub castle: u8,
    pub rook_from_idx: Option<usize>,
    pub rook_to_idx: Option<usize>,
    pub rook_val: u8,

    pub promotion: Option<PieceType>,

    // Старые полные состояния
    pub old_en_passants: [Option<EnPassantInfo>; 4],
    pub old_castling_rights: [[bool; 4]; 4],
    pub old_zobrist: u64,
    pub old_halfmove: usize,
    pub old_king_pos: (usize, usize),
    pub old_scores: [i32; 4],
    pub old_dead_colors: [bool; 4],
    pub old_resigned: [bool; 4],
    pub old_turn: Color,

    // Новые поля
    pub old_last_move: Option<Move>,
    pub old_last_active_mover: Option<Color>,
    pub old_ply_count: usize,
}

#[derive(Clone)]
pub struct GameState {
    pub board: Board,
    pub turn: Color,

    pub history: Vec<BoardState>,
    pub move_history: Vec<Move>,

    pub en_passants: [Option<EnPassantInfo>; 4],
    pub castling_rights: [[bool; 4]; 4],
    pub dead_colors: [bool; 4],
    pub resigned: [bool; 4],
    pub scores: [i32; 4],

    pub current_move_idx: usize,
    pub zobrist_hash: u64,
    pub king_positions: [(usize, usize); 4],
    pub halfmove_clock: usize,

    pub hash_history: Vec<u64>,
    pub hash_counts: HashMap<u64, u8>,
    pub piece_positions: [Vec<usize>; 4],
    pub castle_king_start: [(usize, usize); 4],

    // Новые поля фундамента
    pub last_move: Option<Move>,
    pub last_active_mover: Option<Color>,
    pub ply_count: usize,
    pub rng_state: u64,
    pub recent_moves: Vec<Move>,
    pub deterministic_dead_king: bool,

}

impl GameState {
    #[inline]
    pub fn color_index(color: Color) -> usize {
        color.index()
    }

    // ========================================================
    // Конструктор
    // ========================================================

    pub fn new() -> Self {
        Self::new_with_variant(0)
    }

    pub fn new_with_variant(variant: u8) -> Self {
        let mut board: Board = [EMPTY; 196];

        let king_pos: [(usize, usize); 4] = match variant {
            0 => [(8, 1), (1, 7), (7, 14), (14, 8)],
            1 => [(8, 1), (1, 8), (7, 14), (14, 7)],
            2 => [(8, 1), (1, 7), (8, 14), (14, 7)],
            3 => [(7, 1), (1, 8), (8, 14), (14, 7)],
            4 => [(7, 1), (1, 8), (7, 14), (14, 8)],
            _ => [(8, 1), (1, 7), (7, 14), (14, 8)],
        };

        let queen_pos: [(usize, usize); 4] = match variant {
            0 => [(7, 1), (1, 8), (8, 14), (14, 7)],
            1 => [(7, 1), (1, 7), (8, 14), (14, 8)],
            2 => [(7, 1), (1, 8), (7, 14), (14, 8)],
            3 => [(8, 1), (1, 7), (7, 14), (14, 8)],
            4 => [(8, 1), (1, 7), (8, 14), (14, 7)],
            _ => [(7, 1), (1, 8), (8, 14), (14, 7)],
        };

        let place = |b: &mut Board, x: usize, y: usize, c: Color, p: PieceType| {
            b[sq_index(x, y)] = make_piece_val(c, p, false);
        };

        // Red (y=1..2)
        place(&mut board, 4, 1, Color::Red, PieceType::Rook);
        place(&mut board, 5, 1, Color::Red, PieceType::Knight);
        place(&mut board, 6, 1, Color::Red, PieceType::Bishop);
        place(&mut board, 9, 1, Color::Red, PieceType::Bishop);
        place(&mut board, 10, 1, Color::Red, PieceType::Knight);
        place(&mut board, 11, 1, Color::Red, PieceType::Rook);
        for x in 4..=11 {
            place(&mut board, x, 2, Color::Red, PieceType::Pawn);
        }

        // Blue (x=1..2)
        place(&mut board, 1, 4, Color::Blue, PieceType::Rook);
        place(&mut board, 1, 5, Color::Blue, PieceType::Knight);
        place(&mut board, 1, 6, Color::Blue, PieceType::Bishop);
        place(&mut board, 1, 9, Color::Blue, PieceType::Bishop);
        place(&mut board, 1, 10, Color::Blue, PieceType::Knight);
        place(&mut board, 1, 11, Color::Blue, PieceType::Rook);
        for y in 4..=11 {
            place(&mut board, 2, y, Color::Blue, PieceType::Pawn);
        }

        // Yellow (y=13..14)
        place(&mut board, 4, 14, Color::Yellow, PieceType::Rook);
        place(&mut board, 5, 14, Color::Yellow, PieceType::Knight);
        place(&mut board, 6, 14, Color::Yellow, PieceType::Bishop);
        place(&mut board, 9, 14, Color::Yellow, PieceType::Bishop);
        place(&mut board, 10, 14, Color::Yellow, PieceType::Knight);
        place(&mut board, 11, 14, Color::Yellow, PieceType::Rook);
        for x in 4..=11 {
            place(&mut board, x, 13, Color::Yellow, PieceType::Pawn);
        }

        // Green (x=13..14)
        place(&mut board, 14, 4, Color::Green, PieceType::Rook);
        place(&mut board, 14, 5, Color::Green, PieceType::Knight);
        place(&mut board, 14, 6, Color::Green, PieceType::Bishop);
        place(&mut board, 14, 9, Color::Green, PieceType::Bishop);
        place(&mut board, 14, 10, Color::Green, PieceType::Knight);
        place(&mut board, 14, 11, Color::Green, PieceType::Rook);
        for y in 4..=11 {
            place(&mut board, 13, y, Color::Green, PieceType::Pawn);
        }

        let colors = Color::ALL;

        for i in 0..4 {
            place(&mut board, king_pos[i].0, king_pos[i].1, colors[i], PieceType::King);
            place(&mut board, queen_pos[i].0, queen_pos[i].1, colors[i], PieceType::Queen);
        }

        let mut piece_positions: [Vec<usize>; 4] = Default::default();

        for i in 0..196 {
            if board[i] != EMPTY {
                if let Some(c) = piece_color(board[i]) {
                    piece_positions[c.index()].push(i);
                }
            }
        }

        let mut state = Self {
            board,
            turn: Color::Red,
            history: Vec::new(),
            move_history: Vec::new(),
            en_passants: [None; 4],
            castling_rights: [[true, true, true, false]; 4],
            dead_colors: [false; 4],
            resigned: [false; 4],
            scores: [0; 4],
            current_move_idx: 0,
            zobrist_hash: 0,
            king_positions: king_pos,
            halfmove_clock: 0,
            hash_history: Vec::new(),
            hash_counts: HashMap::new(),
            piece_positions,
            castle_king_start: king_pos,

            recent_moves: Vec::with_capacity(32),
            last_move: None,
            last_active_mover: None,
            ply_count: 0,
            rng_state: 0,
            deterministic_dead_king: false,
        };

        let z = get_zobrist();
        let mut hash = 0u64;

        for i in 0..196 {
            if state.board[i] != EMPTY {
                hash ^= z.pieces[i][state.board[i] as usize];
            }
        }

        hash ^= z.turn[0];

        for c in 0..4 {
            for r in 0..4 {
                if state.castling_rights[c][r] {
                    hash ^= z.castling[c][r];
                }
            }
        }

        state.zobrist_hash = hash;
        state.hash_history.push(hash);
        state.hash_counts.insert(hash, 1);
        state.history.push(state.snapshot());

        state
    }

    // ========================================================
    // Snapshot / clone
    // ========================================================

    pub fn snapshot(&self) -> BoardState {
        BoardState {
            board: self.board,
            turn: self.turn,
            en_passants: self.en_passants,
            castling_rights: self.castling_rights,
            dead_colors: self.dead_colors,
            resigned: self.resigned,
            scores: self.scores,
            zobrist_hash: self.zobrist_hash,
            king_positions: self.king_positions,
            piece_positions: self.piece_positions.clone(),
            halfmove_clock: self.halfmove_clock,

            last_move: self.last_move,
            last_active_mover: self.last_active_mover,
            ply_count: self.ply_count,
        }
    }

    pub fn clone_light(&self) -> Self {
        Self {
            board: self.board,
            turn: self.turn,
            en_passants: self.en_passants,
            castling_rights: self.castling_rights,
            dead_colors: self.dead_colors,
            resigned: self.resigned,
            scores: self.scores,
            zobrist_hash: self.zobrist_hash,
            king_positions: self.king_positions,
            halfmove_clock: self.halfmove_clock,
            hash_counts: self.hash_counts.clone(),

            history: Vec::new(),
            move_history: Vec::new(),
            hash_history: Vec::new(),
            current_move_idx: 0,

            piece_positions: self.piece_positions.clone(),
            castle_king_start: self.castle_king_start,

            recent_moves: self.recent_moves.clone(),
            last_move: self.last_move,
            last_active_mover: self.last_active_mover,
            ply_count: self.ply_count,
            rng_state: self.rng_state,
            deterministic_dead_king: self.deterministic_dead_king,
        }
    }

    // ========================================================
    // RNG / settings
    // ========================================================

    pub fn set_seed(&mut self, seed: u64) {
        self.rng_state = seed;
    }

    pub fn set_deterministic_dead_king(&mut self, deterministic: bool) {
        self.deterministic_dead_king = deterministic;
    }

    pub(crate) fn next_rand(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }

        let mut s = self.rng_state;

        if s == 0 {
            s = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x9E3779B97F4A7C15);
        }

        // xorshift64
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;

        self.rng_state = s;

        (s as usize) % max
    }

    // ========================================================
    // Count helpers
    // ========================================================

    pub fn active_count(&self) -> usize {
        (0..4)
            .filter(|&i| !self.dead_colors[i] && !self.resigned[i])
            .count()
    }

    pub fn alive_count(&self) -> usize {
        (0..4)
            .filter(|&i| !self.dead_colors[i])
            .count()
    }

    // ========================================================
    // Направления
    // ========================================================

    #[inline]
    pub fn forward_vector(color: Color) -> (i32, i32) {
        match color {
            Color::Red => (0, 1),
            Color::Yellow => (0, -1),
            Color::Blue => (1, 0),
            Color::Green => (-1, 0),
        }
    }

    #[inline]
    pub fn is_promotion_square(x: i32, y: i32, color: Color) -> bool {
        match color {
            Color::Red => y == 8,
            Color::Yellow => y == 7,
            Color::Blue => x == 8,
            Color::Green => x == 7,
        }
    }

    #[inline]
    pub fn pawn_diagonals(color: Color) -> [(i32, i32); 2] {
        let (fx, fy) = Self::forward_vector(color);

        if fx == 0 {
            [(1, fy), (-1, fy)]
        } else {
            [(fx, 1), (fx, -1)]
        }
    }

    // ========================================================
    // Атаки
    // ========================================================

    pub fn is_square_attacked(
        board: &Board,
        dead_colors: &[bool; 4],
        resigned: &[bool; 4],
        x: i32,
        y: i32,
        attacker_color: Color,
    ) -> bool {
        let att_c_idx = attacker_color.index();

        if dead_colors[att_c_idx] {
            return false;
        }

        // Сдавшийся: только король атакует соседние клетки.
        if resigned[att_c_idx] {
            for (dx, dy) in &[
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1),
            ] {
                let nx = x + dx;
                let ny = y + dy;

                if is_valid_square(nx, ny) {
                    let p = board[sq_index(nx as usize, ny as usize)];

                    if p != EMPTY
                        && piece_color(p) == Some(attacker_color)
                        && piece_type(p) == Some(PieceType::King)
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        // Пешки
        let (fx, fy) = Self::forward_vector(attacker_color);

        let pawn_offsets: [(i32, i32); 2] = if fx == 0 {
            [(1, fy), (-1, fy)]
        } else {
            [(fx, 1), (fx, -1)]
        };

        for (dx, dy) in &pawn_offsets {
            let nx = x - dx;
            let ny = y - dy;

            if is_valid_square(nx, ny) {
                let p = board[sq_index(nx as usize, ny as usize)];

                if p != EMPTY
                    && piece_color(p) == Some(attacker_color)
                    && piece_type(p) == Some(PieceType::Pawn)
                    && !is_promoted(p)
                {
                    return true;
                }
            }
        }

        // Кони
        for (dx, dy) in &[
            (1, 2), (1, -2), (-1, 2), (-1, -2),
            (2, 1), (2, -1), (-2, 1), (-2, -1),
        ] {
            let nx = x + dx;
            let ny = y + dy;

            if is_valid_square(nx, ny) {
                let p = board[sq_index(nx as usize, ny as usize)];

                if p != EMPTY
                    && piece_color(p) == Some(attacker_color)
                    && piece_type(p) == Some(PieceType::Knight)
                {
                    return true;
                }
            }
        }

        // Король
        for (dx, dy) in &[
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ] {
            let nx = x + dx;
            let ny = y + dy;

            if is_valid_square(nx, ny) {
                let p = board[sq_index(nx as usize, ny as usize)];

                if p != EMPTY
                    && piece_color(p) == Some(attacker_color)
                    && piece_type(p) == Some(PieceType::King)
                {
                    return true;
                }
            }
        }

        // Диагонали: слон, ферзь, promoted pawn
        for (dx, dy) in &[(1, 1), (1, -1), (-1, 1), (-1, -1)] {
            let mut nx = x + dx;
            let mut ny = y + dy;

            while is_valid_square(nx, ny) {
                let p = board[sq_index(nx as usize, ny as usize)];

                if p != EMPTY {
                    if piece_color(p) == Some(attacker_color)
                        && (piece_type(p) == Some(PieceType::Bishop)
                            || piece_type(p) == Some(PieceType::Queen)
                            || (piece_type(p) == Some(PieceType::Pawn) && is_promoted(p)))
                    {
                        return true;
                    }

                    break;
                }

                nx += dx;
                ny += dy;
            }
        }

        // Линии: ладья, ферзь, promoted pawn
        for (dx, dy) in &[(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let mut nx = x + dx;
            let mut ny = y + dy;

            while is_valid_square(nx, ny) {
                let p = board[sq_index(nx as usize, ny as usize)];

                if p != EMPTY {
                    if piece_color(p) == Some(attacker_color)
                        && (piece_type(p) == Some(PieceType::Rook)
                            || piece_type(p) == Some(PieceType::Queen)
                            || (piece_type(p) == Some(PieceType::Pawn) && is_promoted(p)))
                    {
                        return true;
                    }

                    break;
                }

                nx += dx;
                ny += dy;
            }
        }

        false
    }

    pub fn get_attacked_squares(
        board: &Board,
        dead_colors: &[bool; 4],
        resigned: &[bool; 4],
        attacker_color: Color,
    ) -> [bool; 196] {
        let mut attacked = [false; 196];
        let att_c_idx = attacker_color.index();

        if dead_colors[att_c_idx] {
            return attacked;
        }

        if resigned[att_c_idx] {
            for y in 1..=14 {
                for x in 1..=14 {
                    let p_val = board[sq_index(x, y)];

                    if p_val != EMPTY
                        && piece_color(p_val) == Some(attacker_color)
                        && piece_type(p_val) == Some(PieceType::King)
                    {
                        for (dx, dy) in &[
                            (-1, -1), (-1, 0), (-1, 1),
                            (0, -1),           (0, 1),
                            (1, -1),  (1, 0),  (1, 1),
                        ] {
                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;

                            if is_valid_square(nx, ny) {
                                attacked[sq_index(nx as usize, ny as usize)] = true;
                            }
                        }
                    }
                }
            }

            return attacked;
        }

        let pawn_diags = Self::pawn_diagonals(attacker_color);

        for y in 1..=14 {
            for x in 1..=14 {
                let p_val = board[sq_index(x, y)];

                if p_val == EMPTY || piece_color(p_val) != Some(attacker_color) {
                    continue;
                }

                let p_type = piece_type(p_val).unwrap();

                match p_type {
                    PieceType::Pawn => {
                        if !is_promoted(p_val) {
                            for (dx, dy) in &pawn_diags {
                                let nx = x as i32 + dx;
                                let ny = y as i32 + dy;

                                if is_valid_square(nx, ny) {
                                    attacked[sq_index(nx as usize, ny as usize)] = true;
                                }
                            }
                        } else {
                            // Promoted pawn = queen
                            for (dx, dy) in &[
                                (1, 1), (1, -1), (-1, 1), (-1, -1),
                                (1, 0), (-1, 0), (0, 1), (0, -1),
                            ] {
                                let mut nx = x as i32 + dx;
                                let mut ny = y as i32 + dy;

                                while is_valid_square(nx, ny) {
                                    attacked[sq_index(nx as usize, ny as usize)] = true;

                                    if board[sq_index(nx as usize, ny as usize)] != EMPTY {
                                        break;
                                    }

                                    nx += dx;
                                    ny += dy;
                                }
                            }
                        }
                    }

                    PieceType::Knight => {
                        for (dx, dy) in &[
                            (1, 2), (1, -2), (-1, 2), (-1, -2),
                            (2, 1), (2, -1), (-2, 1), (-2, -1),
                        ] {
                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;

                            if is_valid_square(nx, ny) {
                                attacked[sq_index(nx as usize, ny as usize)] = true;
                            }
                        }
                    }

                    PieceType::King => {
                        for (dx, dy) in &[
                            (-1, -1), (-1, 0), (-1, 1),
                            (0, -1),           (0, 1),
                            (1, -1),  (1, 0),  (1, 1),
                        ] {
                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;

                            if is_valid_square(nx, ny) {
                                attacked[sq_index(nx as usize, ny as usize)] = true;
                            }
                        }
                    }

                    PieceType::Bishop => {
                        for (dx, dy) in &[(1, 1), (1, -1), (-1, 1), (-1, -1)] {
                            let mut nx = x as i32 + dx;
                            let mut ny = y as i32 + dy;

                            while is_valid_square(nx, ny) {
                                attacked[sq_index(nx as usize, ny as usize)] = true;

                                if board[sq_index(nx as usize, ny as usize)] != EMPTY {
                                    break;
                                }

                                nx += dx;
                                ny += dy;
                            }
                        }
                    }

                    PieceType::Rook => {
                        for (dx, dy) in &[(1, 0), (-1, 0), (0, 1), (0, -1)] {
                            let mut nx = x as i32 + dx;
                            let mut ny = y as i32 + dy;

                            while is_valid_square(nx, ny) {
                                attacked[sq_index(nx as usize, ny as usize)] = true;

                                if board[sq_index(nx as usize, ny as usize)] != EMPTY {
                                    break;
                                }

                                nx += dx;
                                ny += dy;
                            }
                        }
                    }

                    PieceType::Queen => {
                        for (dx, dy) in &[
                            (1, 1), (1, -1), (-1, 1), (-1, -1),
                            (1, 0), (-1, 0), (0, 1), (0, -1),
                        ] {
                            let mut nx = x as i32 + dx;
                            let mut ny = y as i32 + dy;

                            while is_valid_square(nx, ny) {
                                attacked[sq_index(nx as usize, ny as usize)] = true;

                                if board[sq_index(nx as usize, ny as usize)] != EMPTY {
                                    break;
                                }

                                nx += dx;
                                ny += dy;
                            }
                        }
                    }
                }
            }
        }

        attacked
    }
}

// ============================================================
// AppliedMove
// ============================================================

#[derive(Clone)]
pub struct AppliedMove {
    pub mv: Move,
    pub undo: UndoInfo,
}

impl GameState {
    // ========================================================
    // Индексы ходов
    // ========================================================

    #[inline]
    pub fn get_move_index(from: (usize, usize), to: (usize, usize)) -> usize {
        (from.0 - 1)
            + (from.1 - 1) * 14
            + (to.0 - 1) * 196
            + (to.1 - 1) * 2744
    }

    #[inline]
    pub fn idx_to_move(idx: usize) -> ((usize, usize), (usize, usize)) {
        let fx = (idx % 14) + 1;
        let fy = ((idx / 14) % 14) + 1;
        let tx = ((idx / 196) % 14) + 1;
        let ty = ((idx / 2744) % 14) + 1;

        ((fx, fy), (tx, ty))
    }

    pub fn legal_move_indices(&mut self) -> Vec<usize> {
        movegen::generate_legal_moves(self)
            .iter()
            .map(|m| Self::get_move_index(m.from, m.to))
            .collect()
    }

    pub fn get_legal_move_indices(moves: &[Move]) -> Vec<usize> {
        moves
            .iter()
            .map(|m| Self::get_move_index(m.from, m.to))
            .collect()
    }

    // ========================================================
    // Шах / мат / пат
    // ========================================================

    pub fn is_king_in_check(&self, color: Color) -> bool {
        let ci = color.index();

        if self.dead_colors[ci] {
            return false;
        }

        let (kx, ky) = self.king_positions[ci];

        if kx == 0 && ky == 0 {
            return false;
        }

        for &oc in &Color::ALL {
            if oc != color && !self.dead_colors[oc.index()] {
                if Self::is_square_attacked(
                    &self.board,
                    &self.dead_colors,
                    &self.resigned,
                    kx as i32,
                    ky as i32,
                    oc,
                ) {
                    return true;
                }
            }
        }

        false
    }

    pub fn is_checkmate(&self, color: Color) -> bool {
        if self.dead_colors[color.index()] {
            return true;
        }

        let mut temp_state = self.clone_light();
        temp_state.turn = color;

        let has_moves = temp_state.has_any_legal_move();
        let in_check = temp_state.is_king_in_check(color);

        !has_moves && in_check
    }

    pub fn is_stalemate(&self, color: Color) -> bool {
        if self.dead_colors[color.index()] {
            return false;
        }

        let mut temp_state = self.clone_light();
        temp_state.turn = color;

        let has_moves = temp_state.has_any_legal_move();
        let in_check = temp_state.is_king_in_check(color);

        !has_moves && !in_check
    }

    // ========================================================
    // Конец игры
    // ========================================================

    pub fn is_insufficient_material(&self) -> bool {
        for i in 0..196 {
            let p = self.board[i];

            if p != EMPTY {
                if let Some(c) = piece_color(p) {
                    let c_idx = c.index();

                    if !self.dead_colors[c_idx]
                        && !self.resigned[c_idx]
                        && piece_type(p) != Some(PieceType::King)
                    {
                        return false;
                    }
                }
            }
        }

        true
    }

    pub fn is_game_over(&self) -> bool {
        let active_count = self.active_count();

        // Игра заканчивается, когда остаётся один активный игрок.
        // Случайные короли сдавшихся игроков не считаются активными.
        if active_count <= 1 {
            return true;
        }

        // 50-move rule
        if self.halfmove_clock >= 200 {
            return true;
        }

        // Threefold repetition
        if *self.hash_counts.get(&self.zobrist_hash).unwrap_or(&0) >= 3 {
            return true;
        }

        // Insufficient material
        if self.is_insufficient_material() {
            return true;
        }

        // Правило 21 очка:
        // если осталось 2 активных игрока и разница >= 21,
        // лидер может заявить победу.
        // Для self-play считаем это автоматическим окончанием игры.
        if active_count == 2 {
            let mut active_scores: Vec<i32> = (0..4)
                .filter(|&i| !self.dead_colors[i] && !self.resigned[i])
                .map(|i| self.scores[i])
                .collect();

            if active_scores.len() == 2 {
                active_scores.sort_unstable();

                if active_scores[1] - active_scores[0] >= 21 {
                    return true;
                }
            }
        }

        false
    }

    pub fn is_turn_dead(&self) -> bool {
        let c_idx = self.turn.index();
        self.dead_colors[c_idx] || self.resigned[c_idx]
    }

    pub fn is_turn_resigned(&self) -> bool {
        let c_idx = self.turn.index();
        self.resigned[c_idx] && !self.dead_colors[c_idx]
    }

    // ========================================================
    // Генерация ходов
    // ========================================================

    pub fn generate_legal_moves(&mut self) -> Vec<Move> {
        movegen::generate_legal_moves(self)
    }

    pub fn generate_pseudo_legal_moves(&self) -> Vec<Move> {
        movegen::generate_pseudo_legal_moves(self)
    }

    pub fn has_any_legal_move(&self) -> bool {
        movegen::has_any_legal_move(self)
    }

    // ========================================================
    // Награда за мат
    // ========================================================

    /// +20 получает последний активный игрок, участвовавший в матовой атаке.
    ///
    /// Логика:
    /// 1. Если preferred-игрок активен — даём ему.
    /// 2. Иначе если last_active_mover активен — даём ему.
    /// 3. Иначе пытаемся найти активного атакующего короля.
    ///    Если он один — даём ему.
    ///    Если их несколько — делим 20 между ними.
    fn award_mate_bonus(&mut self, preferred: Option<Color>, mated: Color) {
        let mated_idx = mated.index();

        if let Some(c) = preferred {
            let ci = c.index();

            if c != mated && !self.dead_colors[ci] && !self.resigned[ci] {
                self.scores[ci] += 20;
                return;
            }
        }

        if let Some(c) = self.last_active_mover {
            let ci = c.index();

            if c != mated && !self.dead_colors[ci] && !self.resigned[ci] {
                self.scores[ci] += 20;
                return;
            }
        }

        // Fallback: ищем активных игроков, которые атакуют короля.
        let (kx, ky) = self.king_positions[mated_idx];

        if kx == 0 || ky == 0 {
            return;
        }

        let mut attackers: Vec<Color> = Vec::new();

        for &c in &Color::ALL {
            let ci = c.index();

            if c == mated {
                continue;
            }

            if self.dead_colors[ci] || self.resigned[ci] {
                continue;
            }

            if Self::is_square_attacked(
                &self.board,
                &self.dead_colors,
                &self.resigned,
                kx as i32,
                ky as i32,
                c,
            ) {
                attackers.push(c);
            }
        }

        if attackers.is_empty() {
            return;
        }

        if attackers.len() == 1 {
            self.scores[attackers[0].index()] += 20;
        } else {
            // Редкий fallback: матовая атака несколькими игроками.
            // Делим 20 очков между активными атакующими.
            let n = attackers.len() as i32;
            let share = 20 / n;
            let rem = 20 % n;

            for (i, c) in attackers.iter().enumerate() {
                let mut add = share;

                if i == 0 {
                    add += rem;
                }

                self.scores[c.index()] += add;
            }
        }
    }

    // ========================================================
    // Terminal detection after move
    // ========================================================

    fn detect_terminal_next(&mut self, preferred_mover: Option<Color>) {
        let next_color = self.turn;
        let next_c_idx = next_color.index();

        if self.dead_colors[next_c_idx] {
            return;
        }

        let has_moves = self.has_any_legal_move();

        if has_moves {
            return;
        }

        // Игрок только что получил мат или пат.
        self.dead_colors[next_c_idx] = true;

        let in_check = self.is_king_in_check(next_color);

        if in_check {
            // Мат:
            // +20 последнему активному участнику матовой атаки.
            self.award_mate_bonus(preferred_mover, next_color);
        } else {
            // Пат:
            // Если запатован активный игрок — он получает +20.
            // Если запатован сдавшийся король — никто не получает очков.
            if !self.resigned[next_c_idx] {
                self.scores[next_c_idx] += 20;
            }
        }

        // Передаём ход, пропуская dead-игроков.
        self.turn = self.turn.next();

        let mut safety = 0;
        while self.dead_colors[self.turn.index()] && safety < 5 {
            self.turn = self.turn.next();
            safety += 1;
        }
    }

    // ========================================================
    // Основной внутренний apply_move
    // ========================================================

    pub fn apply_move_internal(
        &mut self,
        from: (usize, usize),
        to: (usize, usize),
        detect_end: bool,
        promotion_piece: Option<PieceType>,
    ) -> Option<AppliedMove> {
        let from_idx = sq_index(from.0, from.1);
        let to_idx = sq_index(to.0, to.1);

        let piece_val = self.board[from_idx];

        if piece_val == EMPTY {
            return None;
        }

        let color = piece_color(piece_val).unwrap();
        let p_type = piece_type(piece_val).unwrap();
        let color_idx = color.index();

        if self.turn != color {
            return None;
        }

        let mover_is_active = !self.dead_colors[color_idx] && !self.resigned[color_idx];

        // ====================================================
        // Undo info
        // ====================================================

        let old_en_passants = self.en_passants;
        let old_resigned = self.resigned;

        let mut undo = UndoInfo {
            from_idx,
            to_idx,
            piece_val,

            captured_val: EMPTY,
            captured_idx: to_idx,

            en_passant: false,
            ep_captured_idx: None,
            ep_captured_val: EMPTY,
            ep_creator_idx: None,

            castle: 0,
            rook_from_idx: None,
            rook_to_idx: None,
            rook_val: EMPTY,

            promotion: None,

            old_en_passants,
            old_castling_rights: self.castling_rights,
            old_zobrist: self.zobrist_hash,
            old_halfmove: self.halfmove_clock,
            old_king_pos: self.king_positions[color_idx],
            old_scores: self.scores,
            old_dead_colors: self.dead_colors,
            old_resigned,
            old_turn: self.turn,

            old_last_move: self.last_move,
            old_last_active_mover: self.last_active_mover,
            old_ply_count: self.ply_count,
        };

        // ====================================================
        // En passant detection
        // ====================================================

        let mut ep_creator_idx = None;

        if p_type == PieceType::Pawn {
            for i in 0..4 {
                if i == color_idx {
                    continue;
                }

                if let Some(ep) = self.en_passants[i] {
                    if ep.target_sq == to {
                        let creator_idx = ep.creator.index();

                        if self.dead_colors[creator_idx] || self.resigned[creator_idx] {
                            continue;
                        }

                        let v_idx = sq_index(ep.victim_sq.0, ep.victim_sq.1);

                        undo.captured_val = self.board[v_idx];
                        undo.en_passant = true;

                        ep_creator_idx = Some(i);

                        undo.ep_captured_idx = Some(v_idx);
                        undo.ep_captured_val = self.board[v_idx];
                        undo.ep_creator_idx = Some(i);

                        break;
                    }
                }
            }
        }

        if undo.captured_val == EMPTY {
            undo.captured_val = self.board[to_idx];
        }

        // ====================================================
        // Castle detection
        // ====================================================

        undo.castle = if p_type == PieceType::King {
            let dx = to.0 as i32 - from.0 as i32;
            let dy = to.1 as i32 - from.1 as i32;

            if dx.abs() == 2 || dy.abs() == 2 {
                let ci = color.index();
                let (kx, ky) = (
                    self.castle_king_start[ci].0 as i32,
                    self.castle_king_start[ci].1 as i32,
                );

                let (rook1, rook2): ((i32, i32), (i32, i32)) = match color {
                    Color::Red => ((4, 1), (11, 1)),
                    Color::Blue => ((1, 4), (1, 11)),
                    Color::Yellow => ((4, 14), (11, 14)),
                    Color::Green => ((14, 4), (14, 11)),
                };

                let d1 = ((rook1.0 - kx).abs() + (rook1.1 - ky).abs()) as usize;
                let d2 = ((rook2.0 - kx).abs() + (rook2.1 - ky).abs()) as usize;

                let short_rook = if d1 < d2 { rook1 } else { rook2 };

                let step_x = (short_rook.0 - kx).signum();
                let step_y = (short_rook.1 - ky).signum();

                if dx == step_x * 2 && dy == step_y * 2 {
                    1
                } else {
                    2
                }
            } else {
                0
            }
        } else {
            0
        };

        // ====================================================
        // Promotion
        // ====================================================

        undo.promotion = if p_type == PieceType::Pawn
            && Self::is_promotion_square(to.0 as i32, to.1 as i32, color)
        {
            Some(promotion_piece.unwrap_or(PieceType::Queen))
        } else {
            None
        };

        let final_piece_val = if let Some(prom) = undo.promotion {
            make_piece_val(color, prom, true)
        } else {
            piece_val
        };

        // ====================================================
        // Zobrist update
        // ====================================================

        let z = get_zobrist();

        self.zobrist_hash ^= z.pieces[from_idx][piece_val as usize];
        self.zobrist_hash ^= z.turn[color_idx];

        if let Some(old_ep) = self.en_passants[color_idx] {
            let ep_idx = sq_index(old_ep.target_sq.0, old_ep.target_sq.1);
            self.zobrist_hash ^= z.ep_squares[ep_idx];
        }

        if let Some(ci) = ep_creator_idx {
            if let Some(ep) = self.en_passants[ci] {
                let ep_idx = sq_index(ep.target_sq.0, ep.target_sq.1);
                self.zobrist_hash ^= z.ep_squares[ep_idx];
                self.en_passants[ci] = None;
            }
        }

        // ====================================================
        // Scoring
        // ====================================================

        if undo.captured_val != EMPTY {
            let cap_c = piece_color(undo.captured_val).unwrap().index();
            let cap_p = piece_type(undo.captured_val).unwrap();
            let target_was_dead = self.dead_colors[cap_c];

            if undo.en_passant {
                let v_idx = undo.ep_captured_idx.unwrap();
                self.zobrist_hash ^= z.pieces[v_idx][undo.captured_val as usize];
            } else {
                self.zobrist_hash ^= z.pieces[to_idx][undo.captured_val as usize];
            }

            // Очки только за живые фигуры.
            if !target_was_dead
                && !self.resigned[cap_c]
                && mover_is_active
            {
                self.scores[color_idx] +=
                    cap_p.capture_points(is_promoted(undo.captured_val));
            }

            // Захват короля трактуем как мат.
            if cap_p == PieceType::King && !target_was_dead {
                self.dead_colors[cap_c] = true;

                if mover_is_active {
                    self.scores[color_idx] += 20;
                }
            }
        }

        // ====================================================
        // Board update
        // ====================================================

        self.board[from_idx] = EMPTY;
        self.board[to_idx] = final_piece_val;

        if undo.en_passant {
            if let Some(v_idx) = undo.ep_captured_idx {
                self.board[v_idx] = EMPTY;
            }
        }

        if piece_type(final_piece_val) == Some(PieceType::King) {
            self.king_positions[color_idx] = to;
        }

        // ====================================================
        // piece_positions update
        // ====================================================

        if let Some(pos) = self.piece_positions[color_idx]
            .iter()
            .position(|&x| x == from_idx)
        {
            self.piece_positions[color_idx][pos] = to_idx;
        }

        if undo.captured_val != EMPTY {
            let cap_c = piece_color(undo.captured_val).unwrap().index();
            let cap_idx = if undo.en_passant {
                undo.ep_captured_idx.unwrap()
            } else {
                to_idx
            };

            if let Some(pos) = self.piece_positions[cap_c]
                .iter()
                .position(|&x| x == cap_idx)
            {
                self.piece_positions[cap_c].swap_remove(pos);
            }
        }

        self.zobrist_hash ^= z.pieces[to_idx][final_piece_val as usize];

        // ====================================================
        // Castle rook move
        // ====================================================

        if undo.castle != 0 {
            let (rook_from, rook_to) = self.get_rook_positions(color, undo.castle);

            let r_from_idx = sq_index(rook_from.0, rook_from.1);
            let r_to_idx = sq_index(rook_to.0, rook_to.1);

            let rook_val = self.board[r_from_idx];

            undo.rook_from_idx = Some(r_from_idx);
            undo.rook_to_idx = Some(r_to_idx);
            undo.rook_val = rook_val;

            if rook_val != EMPTY {
                self.board[r_to_idx] = rook_val;
                self.board[r_from_idx] = EMPTY;

                if let Some(pos) = self.piece_positions[color_idx]
                    .iter()
                    .position(|&x| x == r_from_idx)
                {
                    self.piece_positions[color_idx][pos] = r_to_idx;
                }

                self.zobrist_hash ^= z.pieces[r_from_idx][rook_val as usize];
                self.zobrist_hash ^= z.pieces[r_to_idx][rook_val as usize];
            }
        }

        // ====================================================
        // Castling rights
        // ====================================================

        self.update_castling_rights(color, from, to);

        // ====================================================
        // En passant creation
        // ====================================================

        self.en_passants[color_idx] = None;

        if p_type == PieceType::Pawn {
            let (fx, fy) = Self::forward_vector(color);

            let dx = to.0 as i32 - from.0 as i32;
            let dy = to.1 as i32 - from.1 as i32;

            if dx == fx * 2 && dy == fy * 2 {
                let mid_x = to.0 as i32 - fx;
                let mid_y = to.1 as i32 - fy;

                self.en_passants[color_idx] = Some(EnPassantInfo {
                    target_sq: (mid_x as usize, mid_y as usize),
                    victim_sq: to,
                    creator: color,
                });

                let ep_idx = sq_index(mid_x as usize, mid_y as usize);
                self.zobrist_hash ^= z.ep_squares[ep_idx];
            }
        }

        // ====================================================
        // Halfmove clock
        // ====================================================

        if p_type == PieceType::Pawn || undo.captured_val != EMPTY {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock += 1;
        }

        // ====================================================
        // Last move / last active mover / ply count
        // ====================================================

        let mut mv = Move {
            from,
            to,
            piece: p_type,
            color,
            captured: if undo.captured_val != EMPTY {
                Some(undo.captured_val)
            } else {
                None
            },
            promotion: undo.promotion,
            castle: undo.castle,
            en_passant: undo.en_passant,
            check: false,
            checkmate: false,
            skipped: false,
        };

        self.last_move = Some(mv);

        if mover_is_active {
            self.last_active_mover = Some(color);
        }

        self.ply_count += 1;

        // ====================================================
        // Turn advance
        // ====================================================

        self.turn = self.turn.next();

        let mut safety = 0;
        while self.dead_colors[self.turn.index()] && safety < 5 {
            self.turn = self.turn.next();
            safety += 1;
        }

        // ====================================================
        // Multi-check bonus
        // ====================================================

        let mut check_count = 0;

        for &c in &Color::ALL {
            if c != color && !self.dead_colors[c.index()] {
                let (kx, ky) = self.king_positions[c.index()];

                if kx != 0
                    && Self::is_square_attacked(
                        &self.board,
                        &self.dead_colors,
                        &self.resigned,
                        kx as i32,
                        ky as i32,
                        color,
                    )
                {
                    check_count += 1;
                }
            }
        }

        if check_count >= 2 && mover_is_active {
            let bonus = match piece_type(final_piece_val) {
                Some(PieceType::Queen) => {
                    if check_count == 3 {
                        5
                    } else {
                        1
                    }
                }
                _ => {
                    if check_count == 3 {
                        20
                    } else {
                        5
                    }
                }
            };

            self.scores[color_idx] += bonus;
        }

        mv.check = check_count > 0;
        self.last_move = Some(mv);

        self.recent_moves.push(mv);
        if self.recent_moves.len() > 32 {
            self.recent_moves.remove(0);
        }
        // ====================================================
        // End detection
        // ====================================================

        if detect_end {
            self.detect_terminal_next(Some(color));
        }

        // ====================================================
        // Final zobrist turn + hash history
        // ====================================================

        self.zobrist_hash ^= z.turn[self.turn.index()];
        self.hash_history.push(self.zobrist_hash);

        let cnt = self.hash_counts.entry(self.zobrist_hash).or_insert(0);
        *cnt = cnt.saturating_add(1);

        Some(AppliedMove { mv, undo })
    }

    // ========================================================
    // unmake_move
    // ========================================================

    pub fn unmake_move(&mut self, undo: UndoInfo) {
        let _z = get_zobrist();

        // ====================================================
        // Board restore
        // ====================================================

        self.board[undo.from_idx] = undo.piece_val;
        self.board[undo.to_idx] = undo.captured_val;

        if undo.en_passant {
            if let Some(v_idx) = undo.ep_captured_idx {
                self.board[v_idx] = undo.ep_captured_val;
            }
        }

        if undo.castle != 0 {
            if let (Some(r_from), Some(r_to)) = (undo.rook_from_idx, undo.rook_to_idx) {
                self.board[r_from] = undo.rook_val;
                self.board[r_to] = EMPTY;
            }
        }

        // ====================================================
        // King position restore
        // ====================================================

        if piece_type(undo.piece_val) == Some(PieceType::King) {
            let color = piece_color(undo.piece_val).unwrap();
            self.king_positions[color.index()] = undo.old_king_pos;
        }

        // ====================================================
        // piece_positions restore
        // ====================================================

        let color = piece_color(undo.piece_val).unwrap();
        let color_idx = color.index();

        if let Some(pos) = self.piece_positions[color_idx]
            .iter()
            .position(|&x| x == undo.to_idx)
        {
            self.piece_positions[color_idx][pos] = undo.from_idx;
        } else if !self.piece_positions[color_idx].contains(&undo.from_idx) {
            self.piece_positions[color_idx].push(undo.from_idx);
        }

        if undo.captured_val != EMPTY {
            let cap_c = piece_color(undo.captured_val).unwrap().index();
            let cap_idx = if undo.en_passant {
                undo.ep_captured_idx.unwrap()
            } else {
                undo.to_idx
            };

            if !self.piece_positions[cap_c].contains(&cap_idx) {
                self.piece_positions[cap_c].push(cap_idx);
            }
        }

        if undo.castle != 0 {
            if let (Some(r_from), Some(r_to)) = (undo.rook_from_idx, undo.rook_to_idx) {
                if let Some(pos) = self.piece_positions[color_idx]
                    .iter()
                    .position(|&x| x == r_to)
                {
                    self.piece_positions[color_idx][pos] = r_from;
                } else if !self.piece_positions[color_idx].contains(&r_from) {
                    self.piece_positions[color_idx].push(r_from);
                }
            }
        }

        // ====================================================
        // Full state restore
        // ====================================================

        self.castling_rights = undo.old_castling_rights;
        self.en_passants = undo.old_en_passants;
        self.resigned = undo.old_resigned;
        self.scores = undo.old_scores;
        self.dead_colors = undo.old_dead_colors;
        self.halfmove_clock = undo.old_halfmove;
        self.turn = undo.old_turn;
        self.zobrist_hash = undo.old_zobrist;

        self.recent_moves.pop();
        self.last_move = undo.old_last_move;
        self.last_active_mover = undo.old_last_active_mover;
        self.ply_count = undo.old_ply_count;

        // ====================================================
        // Hash history restore
        // ====================================================

        if let Some(last_hash) = self.hash_history.pop() {
            if let Some(count) = self.hash_counts.get_mut(&last_hash) {
                if *count > 0 {
                    *count -= 1;
                }

                if *count == 0 {
                    self.hash_counts.remove(&last_hash);
                }
            }
        }
    }

    // ========================================================
    // Public move helpers
    // ========================================================

    pub fn make_move_unchecked(
        &mut self,
        from: (usize, usize),
        to: (usize, usize),
        detect_end: bool,
        promotion_piece: Option<PieceType>,
    ) -> bool {
        self.apply_move_internal(from, to, detect_end, promotion_piece)
            .is_some()
    }

    pub fn make_move_fast_idx(&mut self, idx: usize) -> bool {
        let (from, to) = Self::idx_to_move(idx);
        self.apply_move_internal(from, to, true, None).is_some()
    }

    pub fn mcts_push_idx(&mut self, idx: usize) -> Option<UndoInfo> {
        let (from, to) = Self::idx_to_move(idx);
        self.apply_move_internal(from, to, false, None)
            .map(|applied| applied.undo)
    }

    pub fn get_turn_after_idx(&mut self, idx: usize) -> Option<usize> {
        if let Some(undo) = self.mcts_push_idx(idx) {
            let turn = self.turn.index();
            self.unmake_move(undo);
            Some(turn)
        } else {
            None
        }
    }

    // ========================================================
    // Strict make_move for PGN / old API
    // ========================================================

    pub fn make_move(
        &mut self,
        from: (usize, usize),
        to: (usize, usize),
        strict: bool,
    ) -> bool {
        if self.dead_colors[self.turn.index()] {
            return false;
        }

        let from_idx = sq_index(from.0, from.1);
        let p_val = self.board[from_idx];

        if p_val == EMPTY {
            return false;
        }

        let color = piece_color(p_val).unwrap();

        if self.turn != color {
            return false;
        }

        let legal_moves = self.generate_legal_moves();

        let mv_opt = legal_moves
            .iter()
            .find(|m| m.from == from && m.to == to)
            .copied();

        if strict && mv_opt.is_none() {
            return false;
        }

        let mv = mv_opt.unwrap_or(Move {
            from,
            to,
            piece: piece_type(p_val).unwrap(),
            color,
            captured: None,
            promotion: None,
            castle: 0,
            en_passant: false,
            check: false,
            checkmate: false,
            skipped: false,
        });

        if self.current_move_idx < self.move_history.len() {
            self.move_history.truncate(self.current_move_idx);

            let keep = self.current_move_idx + 1;

            if self.history.len() > keep {
                self.history.truncate(keep);
            }
        }

        let colors = Color::ALL;
        let color_idx = color.index();

        while self.move_history.len() % 4 != color_idx {
            let skip_color = colors[self.move_history.len() % 4];
            self.record_alignment_skip(skip_color);
        }

        let promotion = mv.promotion;

        let applied = self.apply_move_internal(from, to, true, promotion);

        if applied.is_none() {
            return false;
        }

        let mut recorded = mv;
        recorded.check = self.is_king_in_check(self.turn);
        recorded.checkmate = false;

        self.move_history.push(recorded);
        self.current_move_idx += 1;
        self.history.push(self.snapshot());

        while self.move_history.len() % 4 != self.turn.index() {
            let skip_color = colors[self.move_history.len() % 4];
            self.record_alignment_skip(skip_color);
        }

        true
    }

    fn record_alignment_skip(&mut self, color: Color) {
        let mv = Move {
            from: (0, 0),
            to: (0, 0),
            piece: PieceType::King,
            color,
            captured: None,
            promotion: None,
            castle: 0,
            en_passant: false,
            check: false,
            checkmate: false,
            skipped: true,
        };

        self.move_history.push(mv);
        self.recent_moves.push(mv);
        if self.recent_moves.len() > 32 {
            self.recent_moves.remove(0);
        }
        self.history.push(self.snapshot());
        self.current_move_idx += 1;
        self.ply_count += 1;
        self.last_move = Some(mv);
    }

    // ========================================================
    // Castling helpers
    // ========================================================

    pub fn get_rook_positions(
        &self,
        color: Color,
        castle: u8,
    ) -> ((usize, usize), (usize, usize)) {
        let ci = color.index();

        let (kx, ky) = (
            self.castle_king_start[ci].0 as i32,
            self.castle_king_start[ci].1 as i32,
        );

        let (rook1, rook2): ((usize, usize), (usize, usize)) = match color {
            Color::Red => ((4, 1), (11, 1)),
            Color::Blue => ((1, 4), (1, 11)),
            Color::Yellow => ((4, 14), (11, 14)),
            Color::Green => ((14, 4), (14, 11)),
        };

        let d1 = ((rook1.0 as i32 - kx).abs() + (rook1.1 as i32 - ky).abs()) as usize;
        let d2 = ((rook2.0 as i32 - kx).abs() + (rook2.1 as i32 - ky).abs()) as usize;

        let (short_rook, long_rook) = if d1 < d2 {
            (rook1, rook2)
        } else {
            (rook2, rook1)
        };

        let rook_from = if castle == 1 { short_rook } else { long_rook };

        let step_x = ((rook_from.0 as i32) - kx).signum();
        let step_y = ((rook_from.1 as i32) - ky).signum();

        let rook_to = ((kx + step_x) as usize, (ky + step_y) as usize);

        (rook_from, rook_to)
    }

    fn update_castling_rights(
        &mut self,
        color: Color,
        from: (usize, usize),
        to: (usize, usize),
    ) {
        let z = get_zobrist();
        let ci = color.index();
        let ks = self.castle_king_start[ci];

        if from == ks && self.castling_rights[ci][0] {
            self.zobrist_hash ^= z.castling[ci][0];
            self.castling_rights[ci][0] = false;
        }

        let (rook1, rook2): ((usize, usize), (usize, usize)) = match color {
            Color::Red => ((4, 1), (11, 1)),
            Color::Blue => ((1, 4), (1, 11)),
            Color::Yellow => ((4, 14), (11, 14)),
            Color::Green => ((14, 4), (14, 11)),
        };

        let (kx, ky) = (ks.0 as i32, ks.1 as i32);

        let d1 = ((rook1.0 as i32 - kx).abs() + (rook1.1 as i32 - ky).abs()) as usize;
        let d2 = ((rook2.0 as i32 - kx).abs() + (rook2.1 as i32 - ky).abs()) as usize;

        let (short_rook, long_rook) = if d1 < d2 {
            (rook1, rook2)
        } else {
            (rook2, rook1)
        };

        if from == short_rook && self.castling_rights[ci][1] {
            self.zobrist_hash ^= z.castling[ci][1];
            self.castling_rights[ci][1] = false;
        }

        if from == long_rook && self.castling_rights[ci][2] {
            self.zobrist_hash ^= z.castling[ci][2];
            self.castling_rights[ci][2] = false;
        }

        // Если захватываем ладью соперника.
        for &opp in &Color::ALL {
            if opp == color {
                continue;
            }

            let oci = opp.index();

            let (or1, or2): ((usize, usize), (usize, usize)) = match opp {
                Color::Red => ((4, 1), (11, 1)),
                Color::Blue => ((1, 4), (1, 11)),
                Color::Yellow => ((4, 14), (11, 14)),
                Color::Green => ((14, 4), (14, 11)),
            };

            let (okx, oky) = (
                self.castle_king_start[oci].0 as i32,
                self.castle_king_start[oci].1 as i32,
            );

            let od1 = ((or1.0 as i32 - okx).abs() + (or1.1 as i32 - oky).abs()) as usize;
            let od2 = ((or2.0 as i32 - okx).abs() + (or2.1 as i32 - oky).abs()) as usize;

            let (os, ol) = if od1 < od2 { (or1, or2) } else { (or2, or1) };

            if to == os && self.castling_rights[oci][1] {
                self.zobrist_hash ^= z.castling[oci][1];
                self.castling_rights[oci][1] = false;
            }

            if to == ol && self.castling_rights[oci][2] {
                self.zobrist_hash ^= z.castling[oci][2];
                self.castling_rights[oci][2] = false;
            }
        }
    }
}

impl GameState {
    // ========================================================
    // Pass turn helpers
    // ========================================================

    /// Простой pass turn без записи в move_history.
    /// Используется для MCTS / fast dead-turn processing.
    pub fn pass_turn(&mut self) {
        let z = get_zobrist();

        self.zobrist_hash ^= z.turn[self.turn.index()];

        self.turn = self.turn.next();

        let mut safety = 0;
        while self.dead_colors[self.turn.index()] && safety < 5 {
            self.turn = self.turn.next();
            safety += 1;
        }

        self.zobrist_hash ^= z.turn[self.turn.index()];

        self.hash_history.push(self.zobrist_hash);

        let cnt = self.hash_counts.entry(self.zobrist_hash).or_insert(0);
        *cnt = cnt.saturating_add(1);

        self.ply_count += 1;
    }

    /// Pass turn с записью skip-хода в историю.
    /// Нужно для PGN / navigation / совместимости.
    pub fn pass_turn_and_record_skip(&mut self, is_explicit_skip: bool) {
        let skip_color = self.turn;

        let mv = Move {
            from: (0, 0),
            to: (0, 0),
            piece: PieceType::King,
            color: skip_color,
            captured: None,
            promotion: None,
            castle: if is_explicit_skip { 255 } else { 0 },
            en_passant: false,
            check: false,
            checkmate: false,
            skipped: true,
        };

        self.move_history.push(mv);

        self.recent_moves.push(mv);
        if self.recent_moves.len() > 32 {
            self.recent_moves.remove(0);
        }
        let z = get_zobrist();

        self.zobrist_hash ^= z.turn[self.turn.index()];

        self.turn = self.turn.next();

        let mut safety = 0;
        while self.dead_colors[self.turn.index()] && safety < 5 {
            self.turn = self.turn.next();
            safety += 1;
        }

        self.zobrist_hash ^= z.turn[self.turn.index()];

        self.hash_history.push(self.zobrist_hash);

        let cnt = self.hash_counts.entry(self.zobrist_hash).or_insert(0);
        *cnt = cnt.saturating_add(1);

        self.history.push(self.snapshot());
        self.current_move_idx += 1;
        self.ply_count += 1;
        self.last_move = Some(mv);
    }

    // ========================================================
    // Resign
    // ========================================================

    pub fn resign_player(&mut self, color: Color) {
        let c_idx = color.index();

        if !self.dead_colors[c_idx] {
            self.resigned[c_idx] = true;
        }
    }

    // ========================================================
    // Dead king walking
    // ========================================================

    /// Быстрый auto-play для dead/resigned.
    ///
    /// dead:
    ///   просто пропускаем ход.
    ///
    /// resigned:
    ///   король делает случайный или детерминированный legal-ход.
    ///
    /// Если legal-ходов нет:
    ///   - если это мат: +20 последнему активному участнику;
    ///   - если это пат сдавшемуся королю: никто не получает очков;
    ///   - король умирает.
    pub fn auto_play_dead_fast(&mut self) -> bool {
        let c_idx = self.turn.index();

        if self.dead_colors[c_idx] {
            self.pass_turn();
            return true;
        }

        if !self.resigned[c_idx] {
            return false;
        }

        let resigned_color = self.turn;

        let legal = self.legal_king_moves_for_resigned();

        if legal.is_empty() {
            let in_check = self.is_king_in_check(resigned_color);

            self.dead_colors[c_idx] = true;

            if in_check {
                // Мат сдавшемуся королю:
                // +20 получает последний активный игрок, участвовавший в атаке.
                let preferred = self.last_active_mover;
                self.award_mate_bonus(preferred, resigned_color);
            } else {
                // Пат сдавшемуся королю:
                // никто не получает очков.
            }

            self.pass_turn();
            return true;
        }

        let idx = if self.deterministic_dead_king {
            0
        } else {
            self.next_rand(legal.len())
        };

        let mv = legal[idx];

        self.apply_move_internal(mv.from, mv.to, true, None)
            .is_some()
    }

    /// Старый API.
    /// Теперь использует fast-версию.
    pub fn play_random_dead_king_move(&mut self) -> bool {
        self.auto_play_dead_fast()
    }

    /// Legal-ходы короля для сдавшегося игрока.
    /// Король не может ходить под шах.
    fn legal_king_moves_for_resigned(&self) -> Vec<Move> {
        let c_idx = self.turn.index();
        let kp = self.king_positions[c_idx];

        if kp == (0, 0) {
            return Vec::new();
        }

        let mut king_moves = Vec::new();

        movegen::generate_king_moves(self, kp, self.turn, &mut king_moves);

        let color = self.turn;
        let mut legal = Vec::new();

        for mv in king_moves {
            let from_idx = sq_index(mv.from.0, mv.from.1);
            let to_idx = sq_index(mv.to.0, mv.to.1);

            let piece_val = self.board[from_idx];

            let mut temp_board = self.board;
            temp_board[from_idx] = EMPTY;
            temp_board[to_idx] = piece_val;

            let mut is_attacked = false;

            for &oc in &Color::ALL {
                if oc == color {
                    continue;
                }

                if self.dead_colors[oc.index()] {
                    continue;
                }

                if Self::is_square_attacked(
                    &temp_board,
                    &self.dead_colors,
                    &self.resigned,
                    mv.to.0 as i32,
                    mv.to.1 as i32,
                    oc,
                ) {
                    is_attacked = true;
                    break;
                }
            }

            if !is_attacked {
                legal.push(mv);
            }
        }

        legal
    }

    // ========================================================
    // Scores / rewards
    // ========================================================

    /// Возвращает скорректированные очки:
    /// - draw bonus +10 активным;
    /// - winner bonus +20 за каждого живого короля;
    /// - 21-point bonus.
    fn adjusted_scores(&self) -> [i32; 4] {
        let mut scores = self.scores;

        let is_draw = self.halfmove_clock >= 200
            || *self.hash_counts.get(&self.zobrist_hash).unwrap_or(&0) >= 3
            || self.is_insufficient_material();

        if is_draw {
            for i in 0..4 {
                if !self.dead_colors[i] && !self.resigned[i] {
                    scores[i] += 10;
                }
            }
        }

        let active_count = self.active_count();

        if active_count == 1 {
            let winner_idx = (0..4)
                .find(|&i| !self.dead_colors[i] && !self.resigned[i])
                .unwrap();

            let alive_kings = (0..4)
                .filter(|&i| !self.dead_colors[i] && i != winner_idx)
                .count();

            scores[winner_idx] += 20 * alive_kings as i32;
        } else if active_count == 2 {
            let active_players: Vec<usize> = (0..4)
                .filter(|&i| !self.dead_colors[i] && !self.resigned[i])
                .collect();

            if active_players.len() == 2 {
                let p1 = active_players[0];
                let p2 = active_players[1];

                if (scores[p1] - scores[p2]).abs() >= 21 {
                    if scores[p1] > scores[p2] {
                        scores[p2] += 20;
                    } else {
                        scores[p1] += 20;
                    }
                }
            }
        }

        scores
    }

    /// Старый совместимый hard-reward.
    /// При равных очках tie-break по индексу игрока.
    pub fn final_rewards(&self) -> [f32; 4] {
        let scores = self.adjusted_scores();

        let mut players = [
            (0usize, scores[0]),
            (1usize, scores[1]),
            (2usize, scores[2]),
            (3usize, scores[3]),
        ];

        players.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let rank_rewards = [1.0f32, 1.0 / 3.0, -1.0 / 3.0, -1.0];

        let mut rewards = [0.0f32; 4];

        for (place, &(player_idx, _score)) in players.iter().enumerate() {
            rewards[player_idx] = rank_rewards[place];
        }

        rewards
    }

    /// Expected rewards с усреднением при равных очках.
    ///
    /// Если игроки делят места, они получают среднее от соответствующих rank rewards.
    pub fn final_rewards_expected(&self) -> [f32; 4] {
        let scores = self.adjusted_scores();

        let mut players = [
            (0usize, scores[0]),
            (1usize, scores[1]),
            (2usize, scores[2]),
            (3usize, scores[3]),
        ];

        players.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let rank_rewards = [1.0f32, 1.0 / 3.0, -1.0 / 3.0, -1.0];

        let mut rewards = [0.0f32; 4];

        let mut i = 0;

        while i < 4 {
            let mut j = i + 1;

            while j < 4 && players[j].1 == players[i].1 {
                j += 1;
            }

            let group_len = (j - i) as f32;
            let group_sum: f32 = rank_rewards[i..j].iter().sum();
            let avg = group_sum / group_len;

            for k in i..j {
                rewards[players[k].0] = avg;
            }

            i = j;
        }

        rewards
    }

    /// Soft placement matrix:
    ///
    /// [player][place] = probability
    ///
    /// При тайе игрок получает равномерное распределение по разделённым местам.
    pub fn final_placement_soft(&self) -> [[f32; 4]; 4] {
        let scores = self.adjusted_scores();

        let mut players = [
            (0usize, scores[0]),
            (1usize, scores[1]),
            (2usize, scores[2]),
            (3usize, scores[3]),
        ];

        players.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let mut soft = [[0.0f32; 4]; 4];

        let mut i = 0;

        while i < 4 {
            let mut j = i + 1;

            while j < 4 && players[j].1 == players[i].1 {
                j += 1;
            }

            let group_len = (j - i) as f32;
            let prob = 1.0 / group_len;

            for k in i..j {
                let player_idx = players[k].0;

                for place in i..j {
                    soft[player_idx][place] = prob;
                }
            }

            i = j;
        }

        soft
    }

    /// Flat version:
    /// 16 float:
    /// player0_place0, player0_place1, ...
    pub fn final_placement_soft_flat(&self) -> Vec<f32> {
        let soft = self.final_placement_soft();

        let mut out = Vec::with_capacity(16);

        for row in soft.iter() {
            out.extend_from_slice(row);
        }

        out
    }

    // ========================================================
    // Terminal value without mutating state
    // ========================================================

    /// Если у текущего игрока нет legal moves, возвращает terminal expected rewards.
    ///
    /// Если legal moves есть — None.
    pub fn terminal_value_if_no_moves(&self) -> Option<[f32; 4]> {
        if self.is_game_over() {
            return Some(self.final_rewards_expected());
        }

        let legal = movegen::generate_legal_moves(self);

        if !legal.is_empty() {
            return None;
        }

        let mut temp = self.clone_light();

        let c = temp.turn;
        let ci = c.index();

        if temp.dead_colors[ci] {
            temp.pass_turn();
            return Some(temp.final_rewards_expected());
        }

        let in_check = temp.is_king_in_check(c);

        temp.dead_colors[ci] = true;

        if in_check {
            let preferred = temp.last_active_mover;
            temp.award_mate_bonus(preferred, c);
        } else {
            // Пат активному игроку:
            // +20 запатованному.
            //
            // Пат сдавшемуся королю:
            // никто не получает очков.
            if !temp.resigned[ci] {
                temp.scores[ci] += 20;
            }
        }

        Some(temp.final_rewards_expected())
    }

    // ========================================================
    // Navigation
    // ========================================================

    pub fn go_to_move(&mut self, move_idx: usize) -> bool {
        if move_idx >= self.history.len() {
            return false;
        }

        let state = &self.history[move_idx];

        self.board = state.board;
        self.turn = state.turn;
        self.en_passants = state.en_passants;
        self.castling_rights = state.castling_rights;
        self.dead_colors = state.dead_colors;
        self.resigned = state.resigned;
        self.scores = state.scores;
        self.zobrist_hash = state.zobrist_hash;
        self.king_positions = state.king_positions;
        self.piece_positions = state.piece_positions.clone();
        self.halfmove_clock = state.halfmove_clock;

        self.last_move = state.last_move;
        self.last_active_mover = state.last_active_mover;
        self.ply_count = state.ply_count;

        self.current_move_idx = move_idx;

        true
    }

    // ========================================================
    // Small helpers / debug / training adjudication
    // ========================================================

    pub fn repetition_count(&self) -> u8 {
        *self.hash_counts.get(&self.zobrist_hash).unwrap_or(&0)
    }

    pub fn last_move_index(&self) -> Option<usize> {
        self.last_move.and_then(|mv| {
            if mv.skipped {
                None
            } else {
                Some(Self::get_move_index(mv.from, mv.to))
            }
        })
    }

    pub fn get_ply_count(&self) -> usize {
        self.ply_count
    }

    pub fn any_active_in_check(&self) -> bool {
        for &c in &Color::ALL {
            let ci = c.index();

            if !self.dead_colors[ci] && !self.resigned[ci] {
                if self.is_king_in_check(c) {
                    return true;
                }
            }
        }

        false
    }

    pub fn active_score_spread(&self) -> i32 {
        let active_scores: Vec<i32> = (0..4)
            .filter(|&i| !self.dead_colors[i] && !self.resigned[i])
            .map(|i| self.scores[i])
            .collect();

        if active_scores.len() < 2 {
            return 0;
        }

        let max = *active_scores.iter().max().unwrap();
        let min = *active_scores.iter().min().unwrap();

        max - min
    }

    /// Training-only adjudication.
    ///
    /// Это НЕ официальное правило.
    /// Нужно, чтобы партии не шли бесконечно в мёртвых равных позициях.
    ///
    /// Например:
    ///   if state.is_training_adjudication_draw(350, 5) { ... }
    pub fn is_training_adjudication_draw(
        &self,
        min_ply: usize,
        max_score_spread: i32,
    ) -> bool {
        if self.ply_count < min_ply {
            return false;
        }

        if self.active_count() < 2 {
            return false;
        }

        if self.active_score_spread() > max_score_spread {
            return false;
        }

        if self.any_active_in_check() {
            return false;
        }

        true
    }
}