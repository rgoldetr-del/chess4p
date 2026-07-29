use crate::board::*;
use crate::state::{GameState, Move};

pub fn new_move(
    from: (usize, usize),
    to: (usize, usize),
    piece: PieceType,
    color: Color,
    captured: Option<u8>,
    promotion: Option<PieceType>,
    castle: u8,
    en_passant: bool,
) -> Move {
    Move {
        from, to, piece, color, captured, promotion, castle, en_passant,
        check: false, checkmate: false, skipped: false,
    }
}

pub fn generate_legal_moves(state: &GameState) -> Vec<Move> {
    let c_idx = state.turn.index();
    if state.dead_colors[c_idx] {
        return Vec::new();
    }
    if state.resigned[c_idx] {
        let mut moves = Vec::new();
        let kp = state.king_positions[c_idx];
        if kp != (0, 0) {
            generate_king_moves(state, kp, state.turn, &mut moves);
        }
        let color = state.turn;
        moves.retain(|mv| {
            let from_idx = sq_index(mv.from.0, mv.from.1);
            let to_idx = sq_index(mv.to.0, mv.to.1);
            let piece_val = state.board[from_idx];
            let _captured_val = state.board[to_idx];
            // Используем временный board для проверки
            let mut temp_board = state.board;
            temp_board[from_idx] = EMPTY;
            temp_board[to_idx] = piece_val;
            let mut is_attacked = false;
            for &oc in &Color::ALL {
                if oc != color && !state.dead_colors[oc.index()] {
                    if GameState::is_square_attacked(
                        &temp_board, &state.dead_colors, &state.resigned,
                        mv.to.0 as i32, mv.to.1 as i32, oc,
                    ) {
                        is_attacked = true;
                        break;
                    }
                }
            }
            !is_attacked
        });
        return moves;
    }

    let pseudo_moves = generate_pseudo_legal_moves(state);
    let color = state.turn;
    let king_pos = state.king_positions[c_idx];
    let mut legal_moves = Vec::new();
    
    for mv in pseudo_moves {
        let from_idx = sq_index(mv.from.0, mv.from.1);
        let to_idx = sq_index(mv.to.0, mv.to.1);
        let piece_val = state.board[from_idx];
        
        // Создаём временную доску для проверки
        let mut temp_board = state.board;
        temp_board[from_idx] = EMPTY;
        let final_piece_val = if let Some(prom) = mv.promotion {
            make_piece_val(color, prom, true)
        } else {
            piece_val
        };
        let _captured_val = temp_board[to_idx];
        temp_board[to_idx] = final_piece_val;

        // Обработка взятия на проходе
        if mv.en_passant {
            for i in 0..4 {
                if let Some(ep) = state.en_passants[i] {
                    if ep.target_sq == mv.to {
                        let v_idx = sq_index(ep.victim_sq.0, ep.victim_sq.1);
                        temp_board[v_idx] = EMPTY;
                        break;
                    }
                }
            }
        }

        // Обработка рокировки
        if mv.castle != 0 {
            let (rook_from, rook_to) = state.get_rook_positions(color, mv.castle);
            let rf_idx = sq_index(rook_from.0, rook_from.1);
            let rt_idx = sq_index(rook_to.0, rook_to.1);
            let rook_val = temp_board[rf_idx];
            temp_board[rf_idx] = EMPTY;
            temp_board[rt_idx] = rook_val;
        }

        let check_king_pos = if piece_type(piece_val) == Some(PieceType::King) {
            mv.to
        } else {
            king_pos
        };

        let mut is_attacked = false;
        for &oc in &Color::ALL {
            if oc != color && !state.dead_colors[oc.index()] {
                if GameState::is_square_attacked(
                    &temp_board, &state.dead_colors, &state.resigned,
                    check_king_pos.0 as i32, check_king_pos.1 as i32, oc,
                ) {
                    is_attacked = true;
                    break;
                }
            }
        }

        if !is_attacked {
            legal_moves.push(mv);
        }
    }
    legal_moves
}

pub fn has_any_legal_move(state: &GameState) -> bool {
    let c_idx = state.turn.index();
    if state.dead_colors[c_idx] {
        return false;
    }
    if state.resigned[c_idx] {
        let kp = state.king_positions[c_idx];
        if kp == (0, 0) { return false; }
        let mut moves = Vec::new();
        generate_king_moves(state, kp, state.turn, &mut moves);
        let color = state.turn;
        for mv in moves {
            let from_idx = sq_index(mv.from.0, mv.from.1);
            let to_idx = sq_index(mv.to.0, mv.to.1);
            let piece_val = state.board[from_idx];
            let _captured_val = state.board[to_idx];
            // Используем временный board для проверки
            let mut temp_board = state.board;
            temp_board[from_idx] = EMPTY;
            temp_board[to_idx] = piece_val;
            let mut is_attacked = false;
            for &oc in &Color::ALL {
                if oc != color && !state.dead_colors[oc.index()] {
                    if GameState::is_square_attacked(
                        &temp_board, &state.dead_colors, &state.resigned,
                        mv.to.0 as i32, mv.to.1 as i32, oc,
                    ) {
                        is_attacked = true;
                        break;
                    }
                }
            }
            
            if !is_attacked { return true; }
        }
        return false;
    }

    let pseudo_moves = generate_pseudo_legal_moves(state);
    let color = state.turn;
    let king_pos = state.king_positions[c_idx];
    for mv in pseudo_moves {
        let from_idx = sq_index(mv.from.0, mv.from.1);
        let to_idx = sq_index(mv.to.0, mv.to.1);
        let piece_val = state.board[from_idx];
        // Создаём временную копию доски
        let mut temp_board = state.board;
        temp_board[from_idx] = EMPTY;
        let final_piece_val = if let Some(prom) = mv.promotion {
            make_piece_val(color, prom, true)
        } else {
            piece_val
        };
        let _captured_val = temp_board[to_idx];
        temp_board[to_idx] = final_piece_val;

        // Обработка взятия на проходе
        if mv.en_passant {
            for i in 0..4 {
                if let Some(ep) = state.en_passants[i] {
                    if ep.target_sq == mv.to {
                        let v_idx = sq_index(ep.victim_sq.0, ep.victim_sq.1);
                        temp_board[v_idx] = EMPTY;
                        break;
                    }
                }
            }
        }
        // Обработка рокировки
        if mv.castle != 0 {
            let (rook_from, rook_to) = state.get_rook_positions(color, mv.castle);
            let rf_idx = sq_index(rook_from.0, rook_from.1);
            let rt_idx = sq_index(rook_to.0, rook_to.1);
            let rook_val = temp_board[rf_idx];
            temp_board[rf_idx] = EMPTY;
            temp_board[rt_idx] = rook_val;
        }

        let check_king_pos = if piece_type(piece_val) == Some(PieceType::King) {
            mv.to
        } else {
            king_pos
        };

        let mut is_attacked = false;
        for &oc in &Color::ALL {
            if oc != color && !state.dead_colors[oc.index()] {
                if GameState::is_square_attacked(
                    &temp_board, &state.dead_colors, &state.resigned,
                    check_king_pos.0 as i32, check_king_pos.1 as i32, oc,
                ) {
                    is_attacked = true;
                    break;
                }
            }
        }



        if !is_attacked { return true; }
    }
    false
}

pub fn generate_pseudo_legal_moves(state: &GameState) -> Vec<Move> {
    let mut moves = Vec::new();
    let color = state.turn;
    let c_idx = color.index();
    for &sq_idx in &state.piece_positions[c_idx] {
        let (x, y) = sq_from_index(sq_idx);
        let p_val = state.board[sq_idx];
        if p_val != EMPTY && piece_color(p_val) == Some(color) {
            if state.resigned[c_idx] && piece_type(p_val) != Some(PieceType::King) {
                continue;
            }
            let from = (x, y);
            match piece_type(p_val).unwrap() {
                PieceType::Pawn => generate_pawn_moves(state, from, color, &mut moves),
                PieceType::Knight => generate_knight_moves(state, from, color, &mut moves),
                PieceType::Bishop => generate_sliding_moves(state, from, color, &[(1,1),(1,-1),(-1,1),(-1,-1)], &mut moves),
                PieceType::Rook => generate_sliding_moves(state, from, color, &[(1,0),(-1,0),(0,1),(0,-1)], &mut moves),
                PieceType::Queen => generate_sliding_moves(state, from, color, &[(1,1),(1,-1),(-1,1),(-1,-1),(1,0),(-1,0),(0,1),(0,-1)], &mut moves),
                PieceType::King => generate_king_moves(state, from, color, &mut moves),
            }
        }
    }
    if !state.resigned[c_idx] {
        generate_castling_moves(state, color, &mut moves);
    }
    moves
}

pub fn generate_pawn_moves(state: &GameState, from: (usize, usize), color: Color, moves: &mut Vec<Move>) {
    let (x, y) = (from.0 as i32, from.1 as i32);
    let (fx, fy) = GameState::forward_vector(color);
    let one_step = (x + fx, y + fy);
    if is_valid_square(one_step.0, one_step.1) {
        if state.board[sq_index(one_step.0 as usize, one_step.1 as usize)] == EMPTY {
            let prom = if GameState::is_promotion_square(one_step.0, one_step.1, color) {
                Some(PieceType::Queen)
            } else {
                None
            };
            moves.push(new_move(from, (one_step.0 as usize, one_step.1 as usize), PieceType::Pawn, color, None, prom, 0, false));
            let is_start = match color {
                Color::Red => y == 2,
                Color::Yellow => y == 13,
                Color::Blue => x == 2,
                Color::Green => x == 13,
            };
            if is_start {
                let two_step = (x + fx * 2, y + fy * 2);
                if is_valid_square(two_step.0, two_step.1)
                    && state.board[sq_index(two_step.0 as usize, two_step.1 as usize)] == EMPTY {
                    moves.push(new_move(from, (two_step.0 as usize, two_step.1 as usize), PieceType::Pawn, color, None, None, 0, false));
                }
            }
        }
    }
    let diags = GameState::pawn_diagonals(color);
    for (dx, dy) in &diags {
        let tx = x + dx;
        let ty = y + dy;
        if is_valid_square(tx, ty) {
            let target_val = state.board[sq_index(tx as usize, ty as usize)];
            if target_val != EMPTY && piece_color(target_val) != Some(color) {
                let prom = if GameState::is_promotion_square(tx, ty, color) {
                    Some(PieceType::Queen)
                } else {
                    None
                };
                moves.push(new_move(from, (tx as usize, ty as usize), PieceType::Pawn, color, Some(target_val), prom, 0, false));
            }
        }
    }
    // En passant
    let color_idx = color.index();
    for i in 0..4 {
        if i == color_idx { continue; }
        if let Some(ep) = state.en_passants[i] {
            let creator_c_idx = ep.creator.index();
            if state.dead_colors[creator_c_idx] || state.resigned[creator_c_idx] { continue; }
            let (epx, epy) = (ep.target_sq.0 as i32, ep.target_sq.1 as i32);
            let is_diag = diags.iter().any(|(dx, dy)| (epx - x == *dx) && (epy - y == *dy));
            if is_diag && state.board[sq_index(epy as usize, epx as usize)] == EMPTY {
                let victim_val = state.board[sq_index(ep.victim_sq.0, ep.victim_sq.1)];
                if victim_val != EMPTY && piece_type(victim_val) == Some(PieceType::Pawn)
                    && piece_color(victim_val) == Some(ep.creator) {
                    moves.push(new_move(from, ep.target_sq, PieceType::Pawn, color, Some(victim_val), None, 0, true));
                }
            }
        }
    }
}

pub fn generate_knight_moves(state: &GameState, from: (usize, usize), color: Color, moves: &mut Vec<Move>) {
    let (x, y) = (from.0 as i32, from.1 as i32);
    for (dx, dy) in &[(1,2),(1,-2),(-1,2),(-1,-2),(2,1),(2,-1),(-2,1),(-2,-1)] {
        let nx = x + dx;
        let ny = y + dy;
        if is_valid_square(nx, ny) {
            let target_val = state.board[sq_index(nx as usize, ny as usize)];
            if target_val == EMPTY {
                moves.push(new_move(from, (nx as usize, ny as usize), PieceType::Knight, color, None, None, 0, false));
            } else if piece_color(target_val) != Some(color) {
                moves.push(new_move(from, (nx as usize, ny as usize), PieceType::Knight, color, Some(target_val), None, 0, false));
            }
        }
    }
}

pub fn generate_king_moves(state: &GameState, from: (usize, usize), color: Color, moves: &mut Vec<Move>) {
    let (x, y) = (from.0 as i32, from.1 as i32);
    for (dx, dy) in &[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)] {
        let nx = x + dx;
        let ny = y + dy;
        if is_valid_square(nx, ny) {
            let target_val = state.board[sq_index(nx as usize, ny as usize)];
            if target_val == EMPTY {
                moves.push(new_move(from, (nx as usize, ny as usize), PieceType::King, color, None, None, 0, false));
            } else if piece_color(target_val) != Some(color) {
                moves.push(new_move(from, (nx as usize, ny as usize), PieceType::King, color, Some(target_val), None, 0, false));
            }
        }
    }
}

fn generate_sliding_moves(state: &GameState, from: (usize, usize), color: Color, directions: &[(i32, i32)], moves: &mut Vec<Move>) {
    let (x, y) = (from.0 as i32, from.1 as i32);
    let p_type = piece_type(state.board[sq_index(x as usize, y as usize)]).unwrap();
    for (dx, dy) in directions {
        let mut nx = x + dx;
        let mut ny = y + dy;
        while is_valid_square(nx, ny) {
            let target_val = state.board[sq_index(nx as usize, ny as usize)];
            if target_val != EMPTY {
                if piece_color(target_val) != Some(color) {
                    moves.push(new_move(from, (nx as usize, ny as usize), p_type, color, Some(target_val), None, 0, false));
                }
                break;
            } else {
                moves.push(new_move(from, (nx as usize, ny as usize), p_type, color, None, None, 0, false));
            }
            nx += dx;
            ny += dy;
        }
    }
}

fn generate_castling_moves(state: &GameState, color: Color, moves: &mut Vec<Move>) {
    let ci = color.index();
    let rights = state.castling_rights[ci];
    if !rights[0] { return; }
    let king_pos = state.king_positions[ci];
    if state.board[sq_index(king_pos.0, king_pos.1)] != make_piece_val(color, PieceType::King, false) {
        return;
    }
    if state.is_king_in_check(color) { return; }

    let (kx, ky) = (state.castle_king_start[ci].0 as i32, state.castle_king_start[ci].1 as i32);
    let (rook1, rook2): ((usize, usize), (usize, usize)) = match color {
        Color::Red => ((4, 1), (11, 1)),
        Color::Blue => ((1, 4), (1, 11)),
        Color::Yellow => ((4, 14), (11, 14)),
        Color::Green => ((14, 4), (14, 11)),
    };
    let d1 = ((rook1.0 as i32 - kx).abs() + (rook1.1 as i32 - ky).abs()) as usize;
    let d2 = ((rook2.0 as i32 - kx).abs() + (rook2.1 as i32 - ky).abs()) as usize;
    let (short_rook, long_rook) = if d1 < d2 { (rook1, rook2) } else { (rook2, rook1) };

    if rights[1] && can_castle(state, color, true, short_rook, (kx, ky)) {
        let step_x = ((short_rook.0 as i32) - kx).signum();
        let step_y = ((short_rook.1 as i32) - ky).signum();
        let king_to = ((kx + step_x * 2) as usize, (ky + step_y * 2) as usize);
        moves.push(new_move(king_pos, king_to, PieceType::King, color, None, None, 1, false));
    }
    if rights[2] && can_castle(state, color, false, long_rook, (kx, ky)) {
        let step_x = ((long_rook.0 as i32) - kx).signum();
        let step_y = ((long_rook.1 as i32) - ky).signum();
        let king_to = ((kx + step_x * 2) as usize, (ky + step_y * 2) as usize);
        moves.push(new_move(king_pos, king_to, PieceType::King, color, None, None, 2, false));
    }
}

fn can_castle(state: &GameState, color: Color, _short: bool, rook_from: (usize, usize), king_start: (i32, i32)) -> bool {
    let (kx, ky) = king_start;
    if state.board[sq_index(rook_from.0, rook_from.1)] != make_piece_val(color, PieceType::Rook, false) {
        return false;
    }
    let (rx, ry) = (rook_from.0 as i32, rook_from.1 as i32);
    let step_x = (rx - kx).signum();
    let step_y = (ry - ky).signum();
    let mut cx = kx + step_x;
    let mut cy = ky + step_y;
    while (cx, cy) != (rx, ry) {
        if state.board[sq_index(cx as usize, cy as usize)] != EMPTY {
            return false;
        }
        cx += step_x;
        cy += step_y;
    }
    let king_path = [(kx, ky), (kx + step_x, ky + step_y), (kx + step_x * 2, ky + step_y * 2)];
    for (px, py) in &king_path {
        for &other in &Color::ALL {
            let oc_idx = other.index();
            if other != color && !state.dead_colors[oc_idx]
                && GameState::is_square_attacked(
                    &state.board, &state.dead_colors, &state.resigned,
                    *px, *py, other,
                ) {
                return false;
            }
        }
    }
    true
}