1use crate::board::*;
use crate::movegen;
use crate::zobrist::get_zobrist;
use std::collections::HashMap;

// ============================================================
// RNG для dead king walking (thread-local, без race condition)
// ============================================================
thread_local! {
    static RNG_STATE: std::cell::Cell<u64> = std::cell::Cell::new(0);
}

fn fast_random(max: usize) -> usize {
    if max == 0 {
        return 0;
    }
    RNG_STATE.with(|cell| {
        let mut s = cell.get();
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
        cell.set(s);
        (s as usize) % max
    })
}

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
}

/// Сохранённое состояние для unmake_move (MCTS без клонирования).
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
    pub old_en_passant: Option<EnPassantInfo>,
    pub old_castling_rights: [[bool; 4]; 4],
    pub old_zobrist: u64,
    pub old_halfmove: usize,
    pub old_king_pos: (usize, usize),
    pub old_scores: [i32; 4],
    pub old_dead_colors: [bool; 4],
    pub old_turn: Color,
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
        for x in 4..=11 { place(&mut board, x, 2, Color::Red, PieceType::Pawn); }

        // Blue (x=1..2)
        place(&mut board, 1, 4, Color::Blue, PieceType::Rook);
        place(&mut board, 1, 5, Color::Blue, PieceType::Knight);
        place(&mut board, 1, 6, Color::Blue, PieceType::Bishop);
        place(&mut board, 1, 9, Color::Blue, PieceType::Bishop);
        place(&mut board, 1, 10, Color::Blue, PieceType::Knight);
        place(&mut board, 1, 11, Color::Blue, PieceType::Rook);
        for y in 4..=11 { place(&mut board, 2, y, Color::Blue, PieceType::Pawn); }

        // Yellow (y=13..14)
        place(&mut board, 4, 14, Color::Yellow, PieceType::Rook);
        place(&mut board, 5, 14, Color::Yellow, PieceType::Knight);
        place(&mut board, 6, 14, Color::Yellow, PieceType::Bishop);
        place(&mut board, 9, 14, Color::Yellow, PieceType::Bishop);
        place(&mut board, 10, 14, Color::Yellow, PieceType::Knight);
        place(&mut board, 11, 14, Color::Yellow, PieceType::Rook);
        for x in 4..=11 { place(&mut board, x, 13, Color::Yellow, PieceType::Pawn); }

        // Green (x=13..14)
        place(&mut board, 14, 4, Color::Green, PieceType::Rook);
        place(&mut board, 14, 5, Color::Green, PieceType::Knight);
        place(&mut board, 14, 6, Color::Green, PieceType::Bishop);
        place(&mut board, 14, 9, Color::Green, PieceType::Bishop);
        place(&mut board, 14, 10, Color::Green, PieceType::Knight);
        place(&mut board, 14, 11, Color::Green, PieceType::Rook);
        for y in 4..=11 { place(&mut board, 13, y, Color::Green, PieceType::Pawn); }

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
        }
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
        if fx == 0 { [(1, fy), (-1, fy)] } else { [(fx, 1), (fx, -1)] }
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
            for (dx, dy) in &[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)] {
                let nx = x + dx;
                let ny = y + dy;
                if is_valid_square(nx, ny) {
                    let p = board[sq_index(nx as usize, ny as usize)];
                    if p != EMPTY && piece_color(p) == Some(attacker_color)
                        && piece_type(p) == Some(PieceType::King) {
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
                if p != EMPTY && piece_color(p) == Some(attacker_color)
                    && piece_type(p) == Some(PieceType::Pawn) && !is_promoted(p) {
                    return true;
                }
            }
        }

        // Кони
        for (dx, dy) in &[(1,2),(1,-2),(-1,2),(-1,-2),(2,1),(2,-1),(-2,1),(-2,-1)] {
            let nx = x + dx;
            let ny = y + dy;
            if is_valid_square(nx, ny) {
                let p = board[sq_index(nx as usize, ny as usize)];
                if p != EMPTY && piece_color(p) == Some(attacker_color)
                    && piece_type(p) == Some(PieceType::Knight) {
                    return true;
                }
            }
        }

        // Король
        for (dx, dy) in &[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)] {
            let nx = x + dx;
            let ny = y + dy;
            if is_valid_square(nx, ny) {
                let p = board[sq_index(nx as usize, ny as usize)];
                if p != EMPTY && piece_color(p) == Some(attacker_color)
                    && piece_type(p) == Some(PieceType::King) {
                    return true;
                }
            }
        }

        // Диагонали (слон, ферзь, превращённая пешка)
        for (dx, dy) in &[(1,1),(1,-1),(-1,1),(-1,-1)] {
            let mut nx = x + dx;
            let mut ny = y + dy;
            while is_valid_square(nx, ny) {
                let p = board[sq_index(nx as usize, ny as usize)];
                if p != EMPTY {
                    if piece_color(p) == Some(attacker_color)
                        && (piece_type(p) == Some(PieceType::Bishop)
                            || piece_type(p) == Some(PieceType::Queen)
                            || (piece_type(p) == Some(PieceType::Pawn) && is_promoted(p))) {
                        return true;
                    }
                    break;
                }
                nx += dx;
                ny += dy;
            }
        }

        // Линии (ладья, ферзь, превращённая пешка)
        for (dx, dy) in &[(1,0),(-1,0),(0,1),(0,-1)] {
            let mut nx = x + dx;
            let mut ny = y + dy;
            while is_valid_square(nx, ny) {
                let p = board[sq_index(nx as usize, ny as usize)];
                if p != EMPTY {
                    if piece_color(p) == Some(attacker_color)
                        && (piece_type(p) == Some(PieceType::Rook)
                            || piece_type(p) == Some(PieceType::Queen)
                            || (piece_type(p) == Some(PieceType::Pawn) && is_promoted(p))) {
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
                    if p_val != EMPTY && piece_color(p_val) == Some(attacker_color)
                        && piece_type(p_val) == Some(PieceType::King) {
                        for (dx, dy) in &[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)] {
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
                            // Превращённая пешка = ферзь
                            for (dx, dy) in &[(1,1),(1,-1),(-1,1),(-1,-1),(1,0),(-1,0),(0,1),(0,-1)] {
                                let mut nx = x as i32 + dx;
                                let mut ny = y as i32 + dy;
                                while is_valid_square(nx, ny) {
                                    attacked[sq_index(nx as usize, ny as usize)] = true;
                                    if board[sq_index(nx as usize, ny as usize)] != EMPTY { break; }
                                    nx += dx;
                                    ny += dy;
                                }
                            }
                        }
                    }
                    PieceType::Knight => {
                        for (dx, dy) in &[(1,2),(1,-2),(-1,2),(-1,-2),(2,1),(2,-1),(-2,1),(-2,-1)] {
                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;
                            if is_valid_square(nx, ny) {
                                attacked[sq_index(nx as usize, ny as usize)] = true;
                            }
                        }
                    }
                    PieceType::King => {
                        for (dx, dy) in &[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)] {
                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;
                            if is_valid_square(nx, ny) {
                                attacked[sq_index(nx as usize, ny as usize)] = true;
                            }
                        }
                    }
                    PieceType::Bishop => {
                        for (dx, dy) in &[(1,1),(1,-1),(-1,1),(-1,-1)] {
                            let mut nx = x as i32 + dx;
                            let mut ny = y as i32 + dy;
                            while is_valid_square(nx, ny) {
                                attacked[sq_index(nx as usize, ny as usize)] = true;
                                if board[sq_index(nx as usize, ny as usize)] != EMPTY { break; }
                                nx += dx;
                                ny += dy;
                            }
                        }
                    }
                    PieceType::Rook => {
                        for (dx, dy) in &[(1,0),(-1,0),(0,1),(0,-1)] {
                            let mut nx = x as i32 + dx;
                            let mut ny = y as i32 + dy;
                            while is_valid_square(nx, ny) {
                                attacked[sq_index(nx as usize, ny as usize)] = true;
                                if board[sq_index(nx as usize, ny as usize)] != EMPTY { break; }
                                nx += dx;
                                ny += dy;
                            }
                        }
                    }
                    PieceType::Queen => {
                        for (dx, dy) in &[(1,1),(1,-1),(-1,1),(-1,-1),(1,0),(-1,0),(0,1),(0,-1)] {
                            let mut nx = x as i32 + dx;
                            let mut ny = y as i32 + dy;
                            while is_valid_square(nx, ny) {
                                attacked[sq_index(nx as usize, ny as usize)] = true;
                                if board[sq_index(nx as usize, ny as usize)] != EMPTY { break; }
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
                    &self.board, &self.dead_colors, &self.resigned,
                    kx as i32, ky as i32, oc,
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
        let orig_turn = self.turn;
        self.turn = color;
        let has_moves = self.has_any_legal_move();
        let in_check = self.is_king_in_check(color);
        self.turn = orig_turn;
        !has_moves && in_check
    }

    pub fn is_stalemate(&self, color: Color) -> bool {
        if self.dead_colors[color.index()] {
            return false;
        }
        let orig_turn = self.turn;
        self.turn = color;
        let has_moves = self.has_any_legal_move();
        let in_check = self.is_king_in_check(color);
        self.turn = orig_turn;
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
                    if !self.dead_colors[c_idx] && !self.resigned[c_idx]
                        && piece_type(p) != Some(PieceType::King) {
                        return false;
                    }
                }
            }
        }
        true
    }

    pub fn is_game_over(&self) -> bool {
        let active_count = (0..4)
            .filter(|&i| !self.dead_colors[i] && !self.resigned[i])
            .count();
        let alive_count = (0..4).filter(|&i| !self.dead_colors[i]).count();

        if active_count <= 1 && alive_count <= 1 {
            return true;
        }
        if self.halfmove_clock >= 200 {
            return true;
        }
        if *self.hash_counts.get(&self.zobrist_hash).unwrap_or(&0) >= 3 {
            return true;
        }
        if self.is_insufficient_material() {
            return true;
        }
        // Правило 21 очка
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

    // ========================================================
    // Генерация ходов (делегирование в movegen)
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
    // unmake_move для MCTS без клонирования
    // ========================================================
    pub fn make_move_with_undo(&mut self, from: (usize, usize), to: (usize, usize), promotion_piece: Option<PieceType>) -> Option<UndoInfo> {
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

        // Сохраняем состояние для unmake
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
            old_en_passant: self.en_passants[color_idx],
            old_castling_rights: self.castling_rights,
            old_zobrist: self.zobrist_hash,
            old_halfmove: self.halfmove_clock,
            old_king_pos: self.king_positions[color_idx],
            old_scores: self.scores,
            old_dead_colors: self.dead_colors,
            old_turn: self.turn,
        };

        // En passant detection
        let mut ep_captured_idx = None;
        let mut ep_creator_idx = None;
        if p_type == PieceType::Pawn {
            for i in 0..4 {
                if i == color_idx { continue; }
                if let Some(ep) = self.en_passants[i] {
                    if ep.target_sq == to {
                        let creator_idx = ep.creator.index();
                        if self.dead_colors[creator_idx] || self.resigned[creator_idx] {
                            continue;
                        }
                        let v_idx = sq_index(ep.victim_sq.0, ep.victim_sq.1);
                        undo.captured_val = self.board[v_idx];
                        undo.en_passant = true;
                        ep_captured_idx = Some(v_idx);
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

        // Castle detection
        undo.castle = if p_type == PieceType::King {
            let dx = to.0 as i32 - from.0 as i32;
            let dy = to.1 as i32 - from.1 as i32;
            if dx.abs() == 2 || dy.abs() == 2 {
                let ci = color.index();
                let (kx, ky) = (self.castle_king_start[ci].0 as i32, self.castle_king_start[ci].1 as i32);
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
                if dx == step_x * 2 && dy == step_y * 2 { 1 } else { 2 }
            } else {
                0
            }
        } else {
            0
        };

        // Promotion
        undo.promotion = if p_type == PieceType::Pawn
            && Self::is_promotion_square(to.0 as i32, to.1 as i32, color) {
            Some(promotion_piece.unwrap_or(PieceType::Queen))
        } else {
            None
        };

        let final_piece_val = if let Some(prom) = undo.promotion {
            make_piece_val(color, prom, true)
        } else {
            piece_val
        };

        // ---- Zobrist update ----
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

        // ---- Scoring ----
        if undo.captured_val != EMPTY {
            let cap_c = piece_color(undo.captured_val).unwrap().index();
            let cap_p = piece_type(undo.captured_val).unwrap();
            if undo.en_passant {
                let v_idx = undo.ep_captured_idx.unwrap();
                self.zobrist_hash ^= z.pieces[v_idx][undo.captured_val as usize];
            } else {
                self.zobrist_hash ^= z.pieces[to_idx][undo.captured_val as usize];
            }
            if !self.dead_colors[cap_c] && !self.resigned[cap_c]
                && !self.dead_colors[color_idx] && !self.resigned[color_idx] {
                self.scores[color_idx] += cap_p.capture_points(is_promoted(undo.captured_val));
            }
            if cap_p == PieceType::King {
                self.dead_colors[cap_c] = true;
            }
        }

        // ---- Board update ----
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

        // ---- piece_positions update ----
        if let Some(pos) = self.piece_positions[color_idx].iter().position(|&x| x == from_idx) {
            self.piece_positions[color_idx][pos] = to_idx;
        }
        if undo.captured_val != EMPTY {
            let cap_c = piece_color(undo.captured_val).unwrap().index();
            let cap_idx = if undo.en_passant { undo.ep_captured_idx.unwrap() } else { to_idx };
            if let Some(pos) = self.piece_positions[cap_c].iter().position(|&x| x == cap_idx) {
                self.piece_positions[cap_c].swap_remove(pos);
            }
        }

        self.zobrist_hash ^= z.pieces[to_idx][final_piece_val as usize];

        // ---- Castle rook move ----
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
                if let Some(pos) = self.piece_positions[color_idx].iter().position(|&x| x == r_from_idx) {
                    self.piece_positions[color_idx][pos] = r_to_idx;
                }
                self.zobrist_hash ^= z.pieces[r_from_idx][rook_val as usize];
                self.zobrist_hash ^= z.pieces[r_to_idx][rook_val as usize];
            }
        }

        // ---- Castling rights ----
        self.update_castling_rights(color, from, to);

        // ---- En passant creation ----
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

        // ---- Halfmove clock ----
        if p_type == PieceType::Pawn || undo.captured_val != EMPTY {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock += 1;
        }

        // ---- Turn advance ----
        self.turn = self.turn.next();
        let mut safety = 0;
        while self.dead_colors[self.turn.index()] && safety < 5 {
            self.turn = self.turn.next();
            safety += 1;
        }

        // ---- Multi-check bonus ----
        let mut check_count = 0;
        for &c in &Color::ALL {
            if c != color && !self.dead_colors[c.index()] {
                let (kx, ky) = self.king_positions[c.index()];
                if kx != 0 && Self::is_square_attacked(
                    &self.board, &self.dead_colors, &self.resigned,
                    kx as i32, ky as i32, color,
                ) {
                    check_count += 1;
                }
            }
        }
        if check_count >= 2 && !self.dead_colors[color_idx] && !self.resigned[color_idx] {
            let bonus = match piece_type(final_piece_val) {
                Some(PieceType::Queen) => { if check_count == 3 { 5 } else { 1 } }
                _ => { if check_count == 3 { 20 } else { 5 } }
            };
            self.scores[color_idx] += bonus;
        }

        self.zobrist_hash ^= z.turn[self.turn.index()];
        self.hash_history.push(self.zobrist_hash);
        *self.hash_counts.entry(self.zobrist_hash).or_insert(0) += 1;
        Some(undo)
    }

    pub fn unmake_move(&mut self, undo: UndoInfo) {
        let z = get_zobrist();
        
        // Восстанавливаем доску
        self.board[undo.from_idx] = undo.piece_val;
        if undo.promotion.is_some() {
            // Если была промоция, восстанавливаем пешку вместо фигуры
            let color = piece_color(undo.piece_val).unwrap();
            self.board[undo.from_idx] = make_piece_val(color, PieceType::Pawn, false);
        }
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

        // Восстанавливаем позиции королей
        if piece_type(undo.piece_val) == Some(PieceType::King) {
            let color = piece_color(undo.piece_val).unwrap();
            self.king_positions[color.index()] = undo.old_king_pos;
        }

        // Восстанавливаем piece_positions
        let color = piece_color(undo.piece_val).unwrap();
        let color_idx = color.index();
        if let Some(pos) = self.piece_positions[color_idx].iter().position(|&x| x == undo.to_idx) {
            self.piece_positions[color_idx][pos] = undo.from_idx;
        }
        if undo.captured_val != EMPTY {
            let cap_c = piece_color(undo.captured_val).unwrap().index();
            let cap_idx = if undo.en_passant { undo.ep_captured_idx.unwrap() } else { undo.to_idx };
            self.piece_positions[cap_c].push(cap_idx);
        }
        if undo.castle != 0 {
            if let (Some(r_from), Some(_)) = (undo.rook_from_idx, undo.rook_to_idx) {
                if let Some(pos) = self.piece_positions[color_idx].iter().position(|&x| x == r_from) {
                    // already at r_from after board restore
                } else {
                    self.piece_positions[color_idx].push(r_from);
                }
            }
        }

        // Восстанавливаем castling rights
        self.castling_rights = undo.old_castling_rights;
        
        // Восстанавливаем en passant
        self.en_passants[color_idx] = undo.old_en_passant;
        if let Some(ci) = undo.ep_creator_idx {
            // Восстанавливаем en passant создателя
            if let Some(ep) = self.en_passants[ci] {
                // Уже восстановлено через old_en_passant если это тот же цвет
            }
        }

        // Восстанавливаем scores, dead_colors
        self.scores = undo.old_scores;
        self.dead_colors = undo.old_dead_colors;

        // Восстанавливаем halfmove clock
        self.halfmove_clock = undo.old_halfmove;

        // Восстанавливаем turn
        self.turn = undo.old_turn;

        // Восстанавливаем zobrist hash
        self.zobrist_hash = undo.old_zobrist;

        // Удаляем последний hash из истории
        if !self.hash_history.is_empty() {
            let last_hash = self.hash_history.pop().unwrap();
            if let Some(count) = self.hash_counts.get_mut(&last_hash) {
                if *count > 0 {
                    *count -= 1;
                    if *count == 0 {
                        self.hash_counts.remove(&last_hash);
                    }
                }
            }
        }
    }

    // ========================================================
    // Рокировка (вспомогательные)
    // ========================================================
    pub fn get_rook_positions(&self, color: Color, castle: u8) -> ((usize, usize), (usize, usize)) {
        let ci = color.index();
        let (kx, ky) = (self.castle_king_start[ci].0 as i32, self.castle_king_start[ci].1 as i32);
        let (rook1, rook2): ((usize, usize), (usize, usize)) = match color {
            Color::Red => ((4, 1), (11, 1)),
            Color::Blue => ((1, 4), (1, 11)),
            Color::Yellow => ((4, 14), (11, 14)),
            Color::Green => ((14, 4), (14, 11)),
        };
        let d1 = ((rook1.0 as i32 - kx).abs() + (rook1.1 as i32 - ky).abs()) as usize;
        let d2 = ((rook2.0 as i32 - kx).abs() + (rook2.1 as i32 - ky).abs()) as usize;
        let (short_rook, long_rook) = if d1 < d2 { (rook1, rook2) } else { (rook2, rook1) };
        let rook_from = if castle == 1 { short_rook } else { long_rook };
        let step_x = ((rook_from.0 as i32) - kx).signum();
        let step_y = ((rook_from.1 as i32) - ky).signum();
        let rook_to = ((kx + step_x) as usize, (ky + step_y) as usize);
        (rook_from, rook_to)
    }

    // ========================================================
    // make_move_unchecked + make/unmake для MCTS
    // ========================================================
    pub fn make_move_unchecked(
        &mut self,
        from: (usize, usize),
        to: (usize, usize),
        detect_end: bool,
        promotion_piece: Option<PieceType>,
    ) -> bool {
        let from_idx = sq_index(from.0, from.1);
        let to_idx = sq_index(to.0, to.1);
        let piece_val = self.board[from_idx];
        if piece_val == EMPTY {
            return false;
        }
        let color = piece_color(piece_val).unwrap();
        let p_type = piece_type(piece_val).unwrap();
        let color_idx = color.index();
        if self.turn != color {
            return false;
        }

        let mut captured_val = EMPTY;
        let mut en_passant = false;
        let mut ep_captured_idx = None;
        let mut ep_creator_idx = None;

        // En passant detection
        if p_type == PieceType::Pawn {
            for i in 0..4 {
                if i == color_idx { continue; }
                if let Some(ep) = self.en_passants[i] {
                    if ep.target_sq == to {
                        let creator_idx = ep.creator.index();
                        if self.dead_colors[creator_idx] || self.resigned[creator_idx] {
                            continue;
                        }
                        let v_idx = sq_index(ep.victim_sq.0, ep.victim_sq.1);
                        captured_val = self.board[v_idx];
                        en_passant = true;
                        ep_captured_idx = Some(v_idx);
                        ep_creator_idx = Some(i);
                        break;
                    }
                }
            }
        }

        if captured_val == EMPTY {
            captured_val = self.board[to_idx];
        }

        // Castle detection
        let castle = if p_type == PieceType::King {
            let dx = to.0 as i32 - from.0 as i32;
            let dy = to.1 as i32 - from.1 as i32;
            if dx.abs() == 2 || dy.abs() == 2 {
                let ci = color.index();
                let (kx, ky) = (self.castle_king_start[ci].0 as i32, self.castle_king_start[ci].1 as i32);
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
                if dx == step_x * 2 && dy == step_y * 2 { 1 } else { 2 }
            } else {
                0
            }
        } else {
            0
        };

        // Promotion
        let promotion = if p_type == PieceType::Pawn
            && Self::is_promotion_square(to.0 as i32, to.1 as i32, color) {
            Some(promotion_piece.unwrap_or(PieceType::Queen))
        } else {
            None
        };

        let final_piece_val = if let Some(prom) = promotion {
            make_piece_val(color, prom, true)
        } else {
            piece_val
        };

        // ---- Zobrist update ----
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

        // ---- Scoring ----
        if captured_val != EMPTY {
            let cap_c = piece_color(captured_val).unwrap().index();
            let cap_p = piece_type(captured_val).unwrap();
            if en_passant {
                let v_idx = ep_captured_idx.unwrap();
                self.zobrist_hash ^= z.pieces[v_idx][captured_val as usize];
            } else {
                self.zobrist_hash ^= z.pieces[to_idx][captured_val as usize];
            }
            // Очки только за «живые» фигуры (правила FFA)
            if !self.dead_colors[cap_c] && !self.resigned[cap_c]
                && !self.dead_colors[color_idx] && !self.resigned[color_idx] {
                self.scores[color_idx] += cap_p.capture_points(is_promoted(captured_val));
            }
            // Взятие короля = смерть
            if cap_p == PieceType::King {
                self.dead_colors[cap_c] = true;
            }
        }

        // ---- Board update ----
        self.board[from_idx] = EMPTY;
        self.board[to_idx] = final_piece_val;
        if en_passant {
            if let Some(v_idx) = ep_captured_idx {
                self.board[v_idx] = EMPTY;
            }
        }
        if piece_type(final_piece_val) == Some(PieceType::King) {
            self.king_positions[color_idx] = to;
        }

        // ---- piece_positions update ----
        if let Some(pos) = self.piece_positions[color_idx].iter().position(|&x| x == from_idx) {
            self.piece_positions[color_idx][pos] = to_idx;
        }
        if captured_val != EMPTY {
            let cap_c = piece_color(captured_val).unwrap().index();
            let cap_idx = if en_passant { ep_captured_idx.unwrap() } else { to_idx };
            if let Some(pos) = self.piece_positions[cap_c].iter().position(|&x| x == cap_idx) {
                self.piece_positions[cap_c].swap_remove(pos);
            }
        }

        self.zobrist_hash ^= z.pieces[to_idx][final_piece_val as usize];

        // ---- Castle rook move ----
        if castle != 0 {
            let (rook_from, rook_to) = self.get_rook_positions(color, castle);
            let r_from_idx = sq_index(rook_from.0, rook_from.1);
            let r_to_idx = sq_index(rook_to.0, rook_to.1);
            let rook_val = self.board[r_from_idx];
            if rook_val != EMPTY {
                self.board[r_to_idx] = rook_val;
                self.board[r_from_idx] = EMPTY;
                if let Some(pos) = self.piece_positions[color_idx].iter().position(|&x| x == r_from_idx) {
                    self.piece_positions[color_idx][pos] = r_to_idx;
                }
                self.zobrist_hash ^= z.pieces[r_from_idx][rook_val as usize];
                self.zobrist_hash ^= z.pieces[r_to_idx][rook_val as usize];
            }
        }

        // ---- Castling rights ----
        self.update_castling_rights(color, from, to);

        // ---- En passant creation ----
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

        // ---- Halfmove clock ----
        if p_type == PieceType::Pawn || captured_val != EMPTY {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock += 1;
        }

        // ---- Turn advance ----
        self.turn = self.turn.next();
        let mut safety = 0;
        while self.dead_colors[self.turn.index()] && safety < 5 {
            self.turn = self.turn.next();
            safety += 1;
        }

        // ---- Multi-check bonus (правила FFA) ----
        let mut check_count = 0;
        for &c in &Color::ALL {
            if c != color && !self.dead_colors[c.index()] {
                let (kx, ky) = self.king_positions[c.index()];
                if kx != 0 && Self::is_square_attacked(
                    &self.board, &self.dead_colors, &self.resigned,
                    kx as i32, ky as i32, color,
                ) {
                    check_count += 1;
                }
            }
        }
        if check_count >= 2 && !self.dead_colors[color_idx] && !self.resigned[color_idx] {
            let bonus = match piece_type(final_piece_val) {
                Some(PieceType::Queen) => { if check_count == 3 { 5 } else { 1 } }
                _ => { if check_count == 3 { 20 } else { 5 } }
            };
            self.scores[color_idx] += bonus;
        }

        // ---- End detection ----
        if detect_end {
            let next_color = self.turn;
            let next_c_idx = next_color.index();
            if !self.dead_colors[next_c_idx] {
                let has_moves = self.has_any_legal_move();
                if !has_moves {
                    self.dead_colors[next_c_idx] = true;
                    if self.is_king_in_check(next_color) {
                        // Мат: +20 атакующему
                        self.scores[color_idx] += 20;
                    } else {
                        // Пат: +10 каждому оставшемуся (кроме запатованного)
                        for i in 0..4 {
                            if i != next_c_idx && !self.dead_colors[i] && !self.resigned[i] {
                                self.scores[i] += 10;
                            }
                        }
                        // Пат самому себе: +20 (правила FFA)
                        self.scores[next_c_idx] += 20;
                    }
                    self.turn = self.turn.next();
                    let mut safety2 = 0;
                    while self.dead_colors[self.turn.index()] && safety2 < 5 {
                        self.turn = self.turn.next();
                        safety2 += 1;
                    }
                }
            }
        }

        self.zobrist_hash ^= z.turn[self.turn.index()];
        self.hash_history.push(self.zobrist_hash);
        *self.hash_counts.entry(self.zobrist_hash).or_insert(0) += 1;
        true
    }

    pub fn make_move(&mut self, from: (usize, usize), to: (usize, usize), strict: bool) -> bool {
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
        let mv_opt = legal_moves.iter().find(|m| m.from == from && m.to == to).copied();
        if strict && mv_opt.is_none() {
            return false;
        }
        let mv = mv_opt.unwrap_or(Move {
            from, to,
            piece: piece_type(p_val).unwrap(),
            color,
            captured: None, promotion: None, castle: 0,
            en_passant: false, check: false, checkmate: false, skipped: false,
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
        self.make_move_unchecked(from, to, true, promotion);

        let mut recorded = mv;
        recorded.check = self.is_king_in_check(self.turn);
        // Лениво: не вызываем is_checkmate здесь (дорого).
        // Флаг checkmate будет установлен при detect_end в make_move_unchecked.
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
        self.move_history.push(Move {
            from: (0, 0), to: (0, 0),
            piece: PieceType::King, color,
            captured: None, promotion: None, castle: 0,
            en_passant: false, check: false, checkmate: false, skipped: true,
        });
        self.history.push(self.snapshot());
        self.current_move_idx += 1;
    }

    fn update_castling_rights(&mut self, color: Color, from: (usize, usize), to: (usize, usize)) {
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
        let (short_rook, long_rook) = if d1 < d2 { (rook1, rook2) } else { (rook2, rook1) };
        if from == short_rook && self.castling_rights[ci][1] {
            self.zobrist_hash ^= z.castling[ci][1];
            self.castling_rights[ci][1] = false;
        }
        if from == long_rook && self.castling_rights[ci][2] {
            self.zobrist_hash ^= z.castling[ci][2];
            self.castling_rights[ci][2] = false;
        }
        // Проверка захвата ладей соперников
        for &opp in &Color::ALL {
            if opp == color { continue; }
            let oci = opp.index();
            let (or1, or2): ((usize, usize), (usize, usize)) = match opp {
                Color::Red => ((4, 1), (11, 1)),
                Color::Blue => ((1, 4), (1, 11)),
                Color::Yellow => ((4, 14), (11, 14)),
                Color::Green => ((14, 4), (14, 11)),
            };
            let (okx, oky) = (self.castle_king_start[oci].0 as i32, self.castle_king_start[oci].1 as i32);
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

    // ========================================================
    // Dead king walking
    // ========================================================
    pub fn resign_player(&mut self, color: Color) {
        let c_idx = color.index();
        if !self.dead_colors[c_idx] {
            self.resigned[c_idx] = true;
        }
    }

    pub fn play_random_dead_king_move(&mut self) -> bool {
        let c_idx = self.turn.index();
        if self.dead_colors[c_idx] {
            self.pass_turn_and_record_skip(false);
            return true;
        }
        if !self.resigned[c_idx] {
            return false;
        }
        let legal = self.legal_king_moves_for_resigned();
        if legal.is_empty() {
            let in_check = self.is_king_in_check(self.turn);
            self.dead_colors[c_idx] = true;
            if !in_check {
                // Пат: +10 каждому активному
                for i in 0..4 {
                    if i != c_idx && !self.dead_colors[i] && !self.resigned[i] {
                        self.scores[i] += 10;
                    }
                }
            }
            self.pass_turn_and_record_skip(true);
            return true;
        }
        let idx = fast_random(legal.len());
        let mv = legal[idx];
        self.make_move(mv.from, mv.to, true)
    }

    fn legal_king_moves_for_resigned(&mut self) -> Vec<Move> {
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
            let captured_val = self.board[to_idx];
            self.board[from_idx] = EMPTY;
            self.board[to_idx] = piece_val;
            let mut is_attacked = false;
            for &oc in &Color::ALL {
                if oc != color && !self.dead_colors[oc.index()] {
                    if Self::is_square_attacked(
                        &self.board, &self.dead_colors, &self.resigned,
                        mv.to.0 as i32, mv.to.1 as i32, oc,
                    ) {
                        is_attacked = true;
                        break;
                    }
                }
            }
            self.board[from_idx] = piece_val;
            self.board[to_idx] = captured_val;
            if !is_attacked {
                legal.push(mv);
            }
        }
        legal
    }

    fn pass_turn_and_record_skip(&mut self, is_explicit_skip: bool) {
        let skip_color = self.turn;
        let mut mv = Move {
            from: (0, 0), to: (0, 0),
            piece: PieceType::King, color: skip_color,
            captured: None, promotion: None, castle: 0,
            en_passant: false, check: false, checkmate: false, skipped: true,
        };
        if is_explicit_skip {
            mv.castle = 255;
        }
        self.move_history.push(mv);
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
        *self.hash_counts.entry(self.zobrist_hash).or_insert(0) += 1;
        self.history.push(self.snapshot());
        self.current_move_idx += 1;
    }

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
        *self.hash_counts.entry(self.zobrist_hash).or_insert(0) += 1;
    }

    // ========================================================
    // Навигация
    // ========================================================
    pub fn go_to_move(&mut self, move_idx: usize) -> bool {
        if move_idx < self.history.len() {
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
            self.current_move_idx = move_idx;
            true
        } else {
            false
        }
    }

    // ========================================================
    // Final rewards
    // ========================================================
    pub fn final_rewards(&self) -> [f32; 4] {
        let mut scores = self.scores;
        let is_draw = self.halfmove_clock >= 200
            || *self.hash_counts.get(&self.zobrist_hash).unwrap_or(&0) >= 3
            || self.is_insufficient_material();

        // Ничья: +10 всем активным
        if is_draw {
            for i in 0..4 {
                if !self.dead_colors[i] && !self.resigned[i] {
                    scores[i] += 10;
                }
            }
        }

        let active_count = (0..4)
            .filter(|&i| !self.dead_colors[i] && !self.resigned[i])
            .count();

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
                    // Проигрывающий получает +20 (правило 21)
                    if scores[p1] > scores[p2] {
                        scores[p2] += 20;
                    } else {
                        scores[p1] += 20;
                    }
                }
            }
        }

        let mut players = [(0, scores[0]), (1, scores[1]), (2, scores[2]), (3, scores[3])];
        players.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let mut rewards = [0.0f32; 4];
        let rank_rewards = [1.0, 0.33, -0.33, -1.0];
        for (i, &(p_idx, _)) in players.iter().enumerate() {
            rewards[p_idx] = rank_rewards[i];
        }
        rewards
    }

    // ========================================================
    // Индексы ходов для NN
    // ========================================================
    pub fn get_legal_move_indices(moves: &[Move]) -> Vec<usize> {
        moves.iter().map(|m| {
            (m.from.0 - 1) + (m.from.1 - 1) * 14 + (m.to.0 - 1) * 196 + (m.to.1 - 1) * 2744
        }).collect()
    }
}
