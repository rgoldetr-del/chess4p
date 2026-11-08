use crate::board::*;
use crate::state::GameState;

pub const TENSOR_PLANES: usize = 256;
pub const TENSOR_SIZE: usize = TENSOR_PLANES * 14 * 14;

/// Заполняет тензор 100×14×14.
///
/// Layout:
/// plane * 196 + y * 14 + x
///
/// Первые 73 плоскости совместимы со старой моделью.
/// Новые 27 плоскостей:
/// 73-76  : promoted queen explicit
/// 77-80  : attacked squares by color
/// 81-84  : protected pieces by color
/// 85-88  : hanging pieces for all colors
/// 89-92  : king location by color
/// 93-96  : distance to own king
/// 97     : repetition count
/// 98     : valid square mask
/// 99     : last move signed plane
pub fn board_to_tensor_slice(state: &GameState, tensor: &mut [f32]) {
    debug_assert!(tensor.len() >= TENSOR_SIZE);
    tensor[..TENSOR_SIZE].fill(0.0);

    // ============================================================
    // Кэш атакованных квадратов для всех цветов
    // ============================================================
    let mut all_attacks = [[false; 196]; 4];

    for &c in &Color::ALL {
        all_attacks[c.index()] = GameState::get_attacked_squares(
            &state.board,
            &state.dead_colors,
            &state.resigned,
            c,
        );
    }

    // ============================================================
    // 0-23  : ordinary pieces
    // 24-27 : promoted flag
    // 61-64 : pawn distance
    // 65-68 : gray dead/resigned pieces
    // 69-72 : hanging pieces of current turn
    // 73-76 : promoted queen explicit
    // 81-84 : protected pieces
    // 85-88 : hanging pieces for all colors
    // ============================================================
    for y in 1..=14 {
        for x in 1..=14 {
            let idx = sq_index(x, y);
            let p_val = state.board[idx];

            if p_val == EMPTY {
                continue;
            }

            let p_color = piece_color(p_val).unwrap();
            let p_type = piece_type(p_val).unwrap();
            let c_idx = p_color.index();

            let is_dead = state.dead_colors[c_idx];
            let is_resigned = state.resigned[c_idx];

            // 65-68: gray pieces
            if is_dead || is_resigned {
                tensor[(65 + c_idx) * 196 + idx] = 1.0;
            }

            // 0-23 / 24-27 / 73-76
            if is_promoted(p_val) {
                // Старый совместимый promoted flag
                tensor[(24 + c_idx) * 196 + idx] = 1.0;

                // Новый явный promoted queen plane
                tensor[(73 + c_idx) * 196 + idx] = 1.0;
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

            // 61-64: pawn distance
            // Оставляем совместимо со старой версией:
            // promoted pawn всё ещё может писать сюда.
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

            // Для hanging/protected считаем только живые фигуры.
            // Resigned non-king фигуры фактически серые и не дают очков,
            // но resigned king остаётся живым.
            let alive_for_tensor = !is_dead && (!is_resigned || p_type == PieceType::King);

            if alive_for_tensor {
                let mut attacked_by_enemy = false;

                for &oc in &Color::ALL {
                    let oi = oc.index();

                    if oi == c_idx {
                        continue;
                    }

                    if state.dead_colors[oi] {
                        continue;
                    }

                    if all_attacks[oi][idx] {
                        attacked_by_enemy = true;
                        break;
                    }
                }

                let protected_by_own = all_attacks[c_idx][idx];

                // 81-84: protected pieces by color
                if protected_by_own {
                    tensor[(81 + c_idx) * 196 + idx] = 1.0;
                }

                // 69-72: hanging pieces of current turn
                if p_color == state.turn && attacked_by_enemy {
                    let val = if p_type == PieceType::King {
                        1.0
                    } else {
                        (p_type.capture_points(is_promoted(p_val)) as f32 / 9.0).min(1.0)
                    };

                    tensor[(69 + c_idx) * 196 + idx] = val;
                }

                // 85-88: hanging pieces for all colors
                if attacked_by_enemy && !protected_by_own {
                    let val = if p_type == PieceType::King {
                        1.0
                    } else {
                        (p_type.capture_points(is_promoted(p_val)) as f32 / 9.0).min(1.0)
                    };

                    tensor[(85 + c_idx) * 196 + idx] = val;
                }
            }
        }
    }

    // ============================================================
    // 28-31: en passant
    // ============================================================
    for c in 0..4 {
        if let Some(ep) = state.en_passants[c] {
            let creator_idx = ep.creator.index();

            if !state.dead_colors[creator_idx] && !state.resigned[creator_idx] {
                let idx = sq_index(ep.target_sq.0, ep.target_sq.1);
                tensor[(28 + creator_idx) * 196 + idx] = 1.0;
            }
        }
    }

    // ============================================================
    // 32-35: turn
    // ============================================================
    let turn = state.turn.index();

    for i in 0..196 {
        tensor[(32 + turn) * 196 + i] = 1.0;
    }

    // ============================================================
    // 36-43: castling rights
    // ============================================================
    for c in 0..4 {
        if state.castling_rights[c][1] {
            for i in 0..196 {
                tensor[(36 + c * 2) * 196 + i] = 1.0;
            }
        }

        if state.castling_rights[c][2] {
            for i in 0..196 {
                tensor[(37 + c * 2) * 196 + i] = 1.0;
            }
        }
    }

    // ============================================================
    // 44-47: dead
    // 48-51: resigned
    // 52-55: scores tanh
    // ============================================================
    for c in 0..4 {
        if state.dead_colors[c] {
            for i in 0..196 {
                tensor[(44 + c) * 196 + i] = 1.0;
            }
        }

        if state.resigned[c] {
            for i in 0..196 {
                tensor[(48 + c) * 196 + i] = 1.0;
            }
        }

        let score_val = (state.scores[c] as f32 / 40.0).tanh();

        for i in 0..196 {
            tensor[(52 + c) * 196 + i] = score_val;
        }
    }

    // ============================================================
    // 56: halfmove clock
    // ============================================================
    let halfmove_val = state.halfmove_clock as f32 / 200.0;

    for i in 0..196 {
        tensor[56 * 196 + i] = halfmove_val;
    }

    // ============================================================
    // 57-60: checks
    // Используем кэш атак вместо повторного is_king_in_check.
    // ============================================================
    for c in 0..4 {
        if state.dead_colors[c] {
            continue;
        }

        let (kx, ky) = state.king_positions[c];

        if kx == 0 || ky == 0 {
            continue;
        }

        let king_idx = sq_index(kx, ky);
        let mut in_check = false;

        for &oc in &Color::ALL {
            let oi = oc.index();

            if oi == c {
                continue;
            }

            if state.dead_colors[oi] {
                continue;
            }

            if all_attacks[oi][king_idx] {
                in_check = true;
                break;
            }
        }

        if in_check {
            for i in 0..196 {
                tensor[(57 + c) * 196 + i] = 1.0;
            }
        }
    }

    // ============================================================
    // 77-80: attacked squares by color
    // ============================================================
    for c in 0..4 {
        for idx in 0..196 {
            if all_attacks[c][idx] {
                tensor[(77 + c) * 196 + idx] = 1.0;
            }
        }
    }

    // ============================================================
    // 89-92: king location by color
    // ============================================================
    for c in 0..4 {
        if state.dead_colors[c] {
            continue;
        }

        let (kx, ky) = state.king_positions[c];

        if kx == 0 || ky == 0 {
            continue;
        }

        let idx = sq_index(kx, ky);
        tensor[(89 + c) * 196 + idx] = 1.0;
    }

    // ============================================================
    // 93-96: distance to own king
    // ============================================================
    for c in 0..4 {
        if state.dead_colors[c] {
            continue;
        }

        let (kx, ky) = state.king_positions[c];

        if kx == 0 || ky == 0 {
            continue;
        }

        let kx_i = kx as i32;
        let ky_i = ky as i32;

        for y in 1..=14 {
            for x in 1..=14 {
                if !is_valid_square(x, y) {
                    continue;
                }

                let idx = sq_index(x as usize, y as usize);
                let dist = (x as i32 - kx_i).abs() + (y as i32 - ky_i).abs();
                let val = 1.0 / (dist as f32 + 1.0);

                tensor[(93 + c) * 196 + idx] = val;
            }
        }
    }

    // ============================================================
    // 97: repetition count
    // ============================================================
    let rep_count = *state.hash_counts.get(&state.zobrist_hash).unwrap_or(&0);
    let rep_val = (rep_count as f32 / 3.0).min(1.0);

    for i in 0..196 {
        tensor[97 * 196 + i] = rep_val;
    }

    // ============================================================
    // 98: valid square mask
    // ============================================================
    for y in 1..=14 {
        for x in 1..=14 {
            if is_valid_square(x, y) {
                let idx = sq_index(x as usize, y as usize);
                tensor[98 * 196 + idx] = 1.0;
            }
        }
    }

    // ============================================================
    // 99: last move signed plane
    // from = +1, to = -1
    // ============================================================
    if let Some(mv) = state.last_move {
        if !mv.skipped {
            let from_idx = sq_index(mv.from.0, mv.from.1);
            let to_idx = sq_index(mv.to.0, mv.to.1);

            tensor[99 * 196 + from_idx] = 1.0;
            tensor[99 * 196 + to_idx] = -1.0;
        }
    }
    // ============================================================
    // 100-115: last 16 moves signed
    // ============================================================
    let rm_len = state.recent_moves.len();

    for step in 0..16 {
        if rm_len > step {
            let mv = state.recent_moves[rm_len - 1 - step];

            if !mv.skipped {
                let from_idx = sq_index(mv.from.0, mv.from.1);
                let to_idx = sq_index(mv.to.0, mv.to.1);

                tensor[(100 + step) * 196 + from_idx] = 1.0;
                tensor[(100 + step) * 196 + to_idx] = -1.0;
            }
        }
    }

    // ============================================================
    // 116-119: last move by each color
    // 120-123: last capture by each color
    // 124-127: last checking move by each color
    // ============================================================
    let mut found_move = [false; 4];
    let mut found_capture = [false; 4];
    let mut found_check = [false; 4];

    for &mv in state.recent_moves.iter().rev() {
        if mv.skipped {
            continue;
        }

        let ci = mv.color.index();

        if !found_move[ci] {
            let from_idx = sq_index(mv.from.0, mv.from.1);
            let to_idx = sq_index(mv.to.0, mv.to.1);

            tensor[(116 + ci) * 196 + from_idx] = 1.0;
            tensor[(116 + ci) * 196 + to_idx] = -1.0;

            found_move[ci] = true;
        }

        if mv.captured.is_some() && !found_capture[ci] {
            let from_idx = sq_index(mv.from.0, mv.from.1);
            let to_idx = sq_index(mv.to.0, mv.to.1);

            tensor[(120 + ci) * 196 + from_idx] = 1.0;
            tensor[(120 + ci) * 196 + to_idx] = -1.0;

            found_capture[ci] = true;
        }

        if mv.check && !found_check[ci] {
            let from_idx = sq_index(mv.from.0, mv.from.1);
            let to_idx = sq_index(mv.to.0, mv.to.1);

            tensor[(124 + ci) * 196 + from_idx] = 1.0;
            tensor[(124 + ci) * 196 + to_idx] = -1.0;

            found_check[ci] = true;
        }

        if found_move.iter().all(|&x| x)
            && found_capture.iter().all(|&x| x)
            && found_check.iter().all(|&x| x)
        {
            break;
        }
    }

    // ============================================================
    // 128-155: piece-level tactical features
    // ============================================================
    for y in 1..=14 {
        for x in 1..=14 {
            let idx = sq_index(x, y);
            let p_val = state.board[idx];

            if p_val == EMPTY {
                continue;
            }

            let p_color = piece_color(p_val).unwrap();
            let p_type = piece_type(p_val).unwrap();
            let c_idx = p_color.index();

            let is_dead = state.dead_colors[c_idx];
            let is_resigned = state.resigned[c_idx];

            let alive_for_tensor =
                !is_dead && (!is_resigned || p_type == PieceType::King);

            if !alive_for_tensor {
                continue;
            }

            let next_color = p_color.next();
            let prev_color = match p_color {
                Color::Red => Color::Green,
                Color::Blue => Color::Red,
                Color::Yellow => Color::Blue,
                Color::Green => Color::Yellow,
            };
            let opposite_color = next_color.next();

            let next_idx = next_color.index();
            let prev_idx = prev_color.index();
            let opp_idx = opposite_color.index();

            // 128-131: attacked by next
            if all_attacks[next_idx][idx] {
                tensor[(128 + c_idx) * 196 + idx] = 1.0;
            }

            // 132-135: attacked by previous
            if all_attacks[prev_idx][idx] {
                tensor[(132 + c_idx) * 196 + idx] = 1.0;
            }

            // 136-139: attacked by opposite
            if all_attacks[opp_idx][idx] {
                tensor[(136 + c_idx) * 196 + idx] = 1.0;
            }

            let mut enemy_attackers = 0usize;

            for &oc in &Color::ALL {
                let oi = oc.index();

                if oi == c_idx {
                    continue;
                }

                if state.dead_colors[oi] {
                    continue;
                }

                if all_attacks[oi][idx] {
                    enemy_attackers += 1;
                }
            }

            // 140-143: attacked by any enemy
            if enemy_attackers > 0 {
                tensor[(140 + c_idx) * 196 + idx] = 1.0;
            }

            // 144-147: protected by own color
            if all_attacks[c_idx][idx] {
                tensor[(144 + c_idx) * 196 + idx] = 1.0;
            }

            // 148-151: hanging weighted
            if enemy_attackers > 0 && !all_attacks[c_idx][idx] {
                let val = if p_type == PieceType::King {
                    1.0
                } else {
                    (p_type.capture_points(is_promoted(p_val)) as f32 / 9.0).min(1.0)
                };

                tensor[(148 + c_idx) * 196 + idx] = val;
            }

            // 152-155: attacked by >=2 enemies
            if enemy_attackers >= 2 {
                tensor[(152 + c_idx) * 196 + idx] = 1.0;
            }
        }
    }

    // ============================================================
    // 156-159: enemy pieces attacked by this color
    // ============================================================
    for c in 0..4 {
        if state.dead_colors[c] {
            continue;
        }

        for idx in 0..196 {
            let p_val = state.board[idx];

            if p_val == EMPTY {
                continue;
            }

            let owner = piece_color(p_val).unwrap();
            let owner_idx = owner.index();
            let p_type = piece_type(p_val).unwrap();

            let owner_alive =
                !state.dead_colors[owner_idx]
                && (!state.resigned[owner_idx] || p_type == PieceType::King);

            if owner_idx != c && owner_alive && all_attacks[c][idx] {
                tensor[(156 + c) * 196 + idx] = 1.0;
            }
        }
    }

    // ============================================================
    // 160-167: king safety
    // ============================================================
    for c in 0..4 {
        if state.dead_colors[c] {
            continue;
        }

        let (kx, ky) = state.king_positions[c];

        if kx == 0 || ky == 0 {
            continue;
        }

        for dx in -1..=1 {
            for dy in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }

                let nx = kx as i32 + dx;
                let ny = ky as i32 + dy;

                if !is_valid_square(nx, ny) {
                    continue;
                }

                let nidx = sq_index(nx as usize, ny as usize);

                let mut attacked_by_enemy = false;
                
                for &oc in &Color::ALL {
                    let oi = oc.index();

                    if oi == c {
                        continue;
                    }

                    if state.dead_colors[oi] {
                        continue;
                    }

                    if all_attacks[oi][nidx] {
                        attacked_by_enemy = true;
                        break;
                    }
                }

                // 160-163: attacked squares around king
                if attacked_by_enemy {
                    tensor[(160 + c) * 196 + nidx] = 1.0;
                }

                // 164-167: king escape squares
                if state.board[nidx] == EMPTY && !attacked_by_enemy {
                    tensor[(164 + c) * 196 + nidx] = 1.0;
                }
            }
        }
    }

    // ============================================================
    // 168-175: pawn promotion features
    // ============================================================
    for y in 1..=14 {
        for x in 1..=14 {
            let idx = sq_index(x, y);
            let p_val = state.board[idx];

            if p_val == EMPTY {
                continue;
            }

            if piece_type(p_val) != Some(PieceType::Pawn) {
                continue;
            }

            if is_promoted(p_val) {
                continue;
            }

            let p_color = piece_color(p_val).unwrap();
            let c_idx = p_color.index();

            if state.dead_colors[c_idx] {
                continue;
            }

            if state.resigned[c_idx] {
                continue;
            }

            let dist = match p_color {
                Color::Red => 8 - y as i32,
                Color::Yellow => y as i32 - 7,
                Color::Blue => 8 - x as i32,
                Color::Green => x as i32 - 7,
            };

            let dist = dist.max(1);

            // 168-171: promotion distance
            tensor[(168 + c_idx) * 196 + idx] =
                1.0 / (dist as f32 + 1.0);

            // 172-175: promotion imminent <= 2
            if dist <= 2 {
                tensor[(172 + c_idx) * 196 + idx] = 1.0;
            }
        }
    }

    // ============================================================
    // 176-183: en passant target / victim
    // ============================================================
    for c in 0..4 {
        if let Some(ep) = state.en_passants[c] {
            let creator_idx = ep.creator.index();

            if !state.dead_colors[creator_idx] && !state.resigned[creator_idx] {
                let target_idx = sq_index(ep.target_sq.0, ep.target_sq.1);
                let victim_idx = sq_index(ep.victim_sq.0, ep.victim_sq.1);

                tensor[(176 + creator_idx) * 196 + target_idx] = 1.0;
                tensor[(180 + creator_idx) * 196 + victim_idx] = 1.0;
            }
        }
    }

    // ============================================================
    // 184-191: castling rook squares
    // ============================================================
    for &c in &Color::ALL {
        let ci = c.index();

        if state.dead_colors[ci] || state.resigned[ci] {
            continue;
        }

        if state.castling_rights[ci][1] {
            let (rook_from, _) = state.get_rook_positions(c, 1);
            let idx = sq_index(rook_from.0, rook_from.1);
            tensor[(184 + ci) * 196 + idx] = 1.0;
        }

        if state.castling_rights[ci][2] {
            let (rook_from, _) = state.get_rook_positions(c, 2);
            let idx = sq_index(rook_from.0, rook_from.1);
            tensor[(188 + ci) * 196 + idx] = 1.0;
        }
    }

    // ============================================================
    // 192-211: global color features
    // ============================================================
    let mut material = [0.0f32; 4];

    for idx in 0..196 {
        let p_val = state.board[idx];

        if p_val == EMPTY {
            continue;
        }

        let c = piece_color(p_val).unwrap();
        let p_type = piece_type(p_val).unwrap();
        let ci = c.index();

        let alive = !state.dead_colors[ci]
            && (!state.resigned[ci] || p_type == PieceType::King);

        if alive {
            material[ci] +=
                p_type.capture_points(is_promoted(p_val)) as f32;
        }
    }

    let mut order: Vec<usize> = (0..4).collect();
    order.sort_by(|&a, &b| state.scores[b].cmp(&state.scores[a]));

    let mut ranks = [0.0f32; 4];

    for (place, &ci) in order.iter().enumerate() {
        ranks[ci] = place as f32 / 3.0;
    }

    let leader_score = state.scores[order[0]] as f32;

    for c in 0..4 {
        // 192-195: score rank
        let rank_val = ranks[c];
        for i in 0..196 {
            tensor[(192 + c) * 196 + i] = rank_val;
        }

        // 196-199: material count
        let mat_val = (material[c] / 40.0).tanh();
        for i in 0..196 {
            tensor[(196 + c) * 196 + i] = mat_val;
        }

        // 200-203: score diff vs leader
        let diff_val =
            ((state.scores[c] as f32 - leader_score) / 21.0).tanh();

        for i in 0..196 {
            tensor[(200 + c) * 196 + i] = diff_val;
        }

        // 204-207: active player
        let active =
            !state.dead_colors[c] && !state.resigned[c];

        if active {
            for i in 0..196 {
                tensor[(204 + c) * 196 + i] = 1.0;
            }
        }

        // 208-211: resigned king alive
        let resigned_king_alive =
            state.resigned[c] && !state.dead_colors[c];

        if resigned_king_alive {
            for i in 0..196 {
                tensor[(208 + c) * 196 + i] = 1.0;
            }
        }
    }

    // 212-255 остаются зарезервированными нулями.
}