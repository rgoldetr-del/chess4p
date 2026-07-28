use crate::board::*;
use crate::state::GameState;

pub const TENSOR_PLANES: usize = 73;
pub const TENSOR_SIZE: usize = TENSOR_PLANES * 14 * 14;

/// Заполняет тензор 73×14×14 (flat layout: plane * 196 + y * 14 + x).
pub fn board_to_tensor_slice(state: &GameState, tensor: &mut [f32]) {
    debug_assert!(tensor.len() >= TENSOR_SIZE);
    tensor[..TENSOR_SIZE].fill(0.0);

    // 0-23: pieces, 24-27: promoted, 61-64: pawn distance, 65-68: gray
    for y in 1..=14 {
        for x in 1..=14 {
            let idx = sq_index(x, y);
            let p_val = state.board[idx];
            if p_val == EMPTY { continue; }
            let p_color = piece_color(p_val).unwrap();
            let p_type = piece_type(p_val).unwrap();
            let c_idx = p_color.index();
            let is_dead = state.dead_colors[c_idx];
            let is_resigned = state.resigned[c_idx];

            if is_dead || is_resigned {
                tensor[(65 + c_idx) * 196 + idx] = 1.0;
            }
            if is_promoted(p_val) {
                tensor[(24 + c_idx) * 196 + idx] = 1.0;
            } else {
                let p_idx = match p_type {
                    PieceType::Pawn => 0,
                    PieceType::Knight => 1,
                    PieceType::Bishop => 2,
                    PieceType::Rook => 3,
                    PieceType::Queen => 4,
                    PieceType::King => 5,
                };
                tensor[(c_idx * 6 + p_idx) * 196 + idx] = 1.0;
            }
            if p_type == PieceType::Pawn {
                let dist = match p_color {
                    Color::Red => 8 - y as i32,
                    Color::Yellow => y as i32 - 7,
                    Color::Blue => 8 - x as i32,
                    Color::Green => x as i32 - 7,
                };
                let dist_val = 1.0 / (dist.max(1) as f32 + 1.0);
                tensor[(61 + c_idx) * 196 + idx] = dist_val;
            }
        }
    }

    // 28-31: en passant
    for c in 0..4 {
        if let Some(ep) = state.en_passants[c] {
            let ep_c_idx = ep.creator.index();
            let idx = sq_index(ep.target_sq.0, ep.target_sq.1);
            tensor[(28 + ep_c_idx) * 196 + idx] = 1.0;
        }
    }

    // 32-35: turn
    let turn = state.turn.index();
    for i in 0..196 { tensor[(32 + turn) * 196 + i] = 1.0; }

    // 36-43: castling
    for c in 0..4 {
        if state.castling_rights[c][1] {
            for i in 0..196 { tensor[(36 + c * 2) * 196 + i] = 1.0; }
        }
        if state.castling_rights[c][2] {
            for i in 0..196 { tensor[(37 + c * 2) * 196 + i] = 1.0; }
        }
    }

    // 44-47: dead, 48-51: resigned, 52-55: scores
    for c in 0..4 {
        if state.dead_colors[c] {
            for i in 0..196 { tensor[(44 + c) * 196 + i] = 1.0; }
        }
        if state.resigned[c] {
            for i in 0..196 { tensor[(48 + c) * 196 + i] = 1.0; }
        }
        let score_val = (state.scores[c] as f32 / 40.0).tanh();
        for i in 0..196 { tensor[(52 + c) * 196 + i] = score_val; }
    }

    // 56: halfmove clock
    let halfmove_val = state.halfmove_clock as f32 / 200.0;
    for i in 0..196 { tensor[56 * 196 + i] = halfmove_val; }

    // 57-60: checks
    for c in 0..4 {
        if state.is_king_in_check(Color::ALL[c]) {
            for i in 0..196 { tensor[(57 + c) * 196 + i] = 1.0; }
        }
    }

    // 69-72: hanging pieces of current turn
    let mut enemy_attacks = [[false; 196]; 4];
    for &oc in &Color::ALL {
        if oc != state.turn {
            enemy_attacks[oc.index()] = GameState::get_attacked_squares(
                &state.board, &state.dead_colors, &state.resigned, oc,
            );
        }
    }
    for y in 1..=14 {
        for x in 1..=14 {
            let idx = sq_index(x, y);
            let p_val = state.board[idx];
            if p_val == EMPTY { continue; }
            let p_color = piece_color(p_val).unwrap();
            let p_type = piece_type(p_val).unwrap();
            if p_color != state.turn { continue; }
            let mut is_attacked = false;
            for &oc in &Color::ALL {
                if oc != p_color && enemy_attacks[oc.index()][idx] {
                    is_attacked = true;
                    break;
                }
            }
            if is_attacked {
                let val = p_type.capture_points(is_promoted(p_val)) as f32 / 9.0;
                tensor[(69 + p_color.index()) * 196 + idx] = val;
            }
        }
    }
}