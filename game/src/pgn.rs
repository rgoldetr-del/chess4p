use crate::board::*;
use crate::state::GameState;

impl GameState {
    pub fn load_pgn(&mut self, pgn: &str) -> Result<(), String> {
        *self = GameState::new_with_variant(0);
        self.load_pgn_internal(pgn, false)
    }

    pub fn load_pgn_internal(&mut self, pgn: &str, lenient: bool) -> Result<(), String> {
        let clean = Self::strip_comments(pgn);
        'outer: for raw_line in clean.lines() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('[') {
                continue;
            }
            for token in line.split_whitespace() {
                let t = token.trim_matches(|c| c == '(' || c == ')');
                if t.is_empty() { continue; }
                if t == "1-0" || t == "0-1" || t == "1/2-1/2" || t == "*" || t == "1/2" || t == "Draw" {
                    break 'outer;
                }
                let chars: Vec<char> = t.chars().collect();
                let mut start = 0;
                while start < chars.len() && (chars[start].is_ascii_digit() || chars[start] == '.') {
                    start += 1;
                }
                let t = &t[start..];
                if t.is_empty() || t == ".." || t == "..." { continue; }

                let mut applied = false;
                for _ in 0..8 {
                    let c_idx = self.turn.index();
                    if self.dead_colors[c_idx] {
                        self.pass_turn_and_record_skip_pub(false);
                        continue;
                    }
                    if t == "Z" || t == "--" || t == "null" {
                        self.pass_turn_and_record_skip_pub(true);
                        applied = true;
                        break;
                    }
                    if t == "R" || t == "T" || t == "Resigns" || t == "Timeout" {
                        self.resigned[c_idx] = true;
                        self.pass_turn_and_record_skip_pub(true);
                        applied = true;
                        break;
                    }
                    if t == "S" || t == "Stalemate" {
                        self.dead_colors[c_idx] = true;
                        self.pass_turn_and_record_skip_pub(true);
                        applied = true;
                        break;
                    }

                    let mut token_str = t.to_string();
                    let mut trailing_action: Option<char> = None;
                    let mut is_mate = false;
                    loop {
                        if let Some(last_char) = token_str.chars().last() {
                            if last_char == '+' || last_char == '!' || last_char == '?' {
                                token_str.pop();
                            } else if last_char == '#' {
                                is_mate = true;
                                token_str.pop();
                            } else if (last_char == 'R' || last_char == 'T' || last_char == 'S')
                                && !token_str.contains('=') {
                                trailing_action = Some(last_char);
                                token_str.pop();
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    }

                    let token_clean = token_str.as_str();
                    let mover = self.turn;
                    let mut move_made = false;

                    if token_clean == "O-O" || token_clean == "0-0"
                        || token_clean == "O-O-O" || token_clean == "0-0-0" {
                        let is_long = token_clean == "O-O-O" || token_clean == "0-0-0";
                        let target_castle = if is_long { 2 } else { 1 };
                        let legal = self.generate_legal_moves();
                        if let Some(&mv) = legal.iter().find(|m| m.castle == target_castle && m.piece == PieceType::King) {
                            move_made = self.make_move(mv.from, mv.to, true);
                        }
                    } else if let Some((_, fx, fy, tx, ty, _prom)) = Self::parse_move_strict(token_clean) {
                        if self.make_move((fx, fy), (tx, ty), true) {
                            move_made = true;
                        } else if lenient {
                            let legal = self.generate_legal_moves();
                            if let Some(&mv) = legal.iter().find(|m| m.to == (tx, ty)) {
                                move_made = self.make_move(mv.from, mv.to, true);
                            }
                        }
                    }

                    if move_made {
                        if let Some(act) = trailing_action {
                            let mover_idx = mover.index();
                            if act == 'R' || act == 'T' {
                                self.resigned[mover_idx] = true;
                            } else if act == 'S' {
                                self.dead_colors[mover_idx] = true;
                            }
                        }
                        if is_mate || trailing_action == Some('S') {
                            let orig_turn = self.turn;
                            for &c in &Color::ALL {
                                if c != mover && !self.dead_colors[c.index()] {
                                    self.turn = c;
                                    if !self.has_any_legal_move() {
                                        self.dead_colors[c.index()] = true;
                                    }
                                }
                            }
                            self.turn = orig_turn;
                        }
                        applied = true;
                        break;
                    } else if lenient {
                        let has = self.has_any_legal_move();
                        if !has {
                            self.dead_colors[c_idx] = true;
                        }
                        self.pass_turn_and_record_skip_pub(false);
                        applied = true;
                        break;
                    } else {
                        return Err(format!("Parse error: '{}' for color {:?}", t, self.turn));
                    }
                }
                if !applied {
                    return Err(format!("Could not apply token '{}'", t));
                }
            }
        }
        self.current_move_idx = self.move_history.len();
        Ok(())
    }

    /// Публичная обёртка для PGN-модуля.
    pub fn pass_turn_and_record_skip_pub(&mut self, is_explicit: bool) {
        // Вызываем приватный метод через повторную реализацию
        let skip_color = self.turn;
        let mut mv = crate::state::Move {
            from: (0, 0), to: (0, 0),
            piece: PieceType::King, color: skip_color,
            captured: None, promotion: None, castle: 0,
            en_passant: false, check: false, checkmate: false, skipped: true,
        };
        if is_explicit { mv.castle = 255; }
        self.move_history.push(mv);
        let z = crate::zobrist::get_zobrist();
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

    fn strip_comments(pgn: &str) -> String {
        let mut s = String::new();
        let mut in_comment = false;
        for c in pgn.chars() {
            if c == '{' { in_comment = true; }
            if !in_comment { s.push(c); }
            if c == '}' { in_comment = false; }
        }
        s
    }

    /// Парсер ходов. Поддерживает ВСЕ типы превращения (=Q/=R/=B/=N)
    /// для совместимости с PGN4 из разных источников.
    pub fn parse_move_strict(
        t: &str,
    ) -> Option<(PieceType, usize, usize, usize, usize, Option<PieceType>)> {
        let mut piece = PieceType::Pawn;
        let mut idx = 0;
        if t.starts_with('K') { piece = PieceType::King; idx = 1; }
        else if t.starts_with('Q') { piece = PieceType::Queen; idx = 1; }
        else if t.starts_with('R') { piece = PieceType::Rook; idx = 1; }
        else if t.starts_with('B') { piece = PieceType::Bishop; idx = 1; }
        else if t.starts_with('N') { piece = PieceType::Knight; idx = 1; }

        let rest = &t[idx..];
        let sep_pos = rest.find(|c| c == '-' || c == 'x')?;
        let from_str = &rest[..sep_pos];
        let to_str_raw = &rest[sep_pos + 1..];

        if from_str.len() < 2 { return None; }
        let fx = (from_str.chars().next()? as u8).to_ascii_lowercase().checked_sub(b'a')? as usize + 1;
        let fy = from_str[1..].parse::<usize>().ok()?;

        let mut to_str = to_str_raw;
        let mut promotion = None;
        if let Some(eq_pos) = to_str_raw.find('=') {
            to_str = &to_str_raw[..eq_pos];
            let prom_char = to_str_raw.chars().nth(eq_pos + 1).unwrap_or('Q');
            // Поддерживаем все типы превращения для совместимости с PGN4
            promotion = match prom_char {
                'Q' | 'D' => Some(PieceType::Queen),
                'R' => Some(PieceType::Rook),
                'B' => Some(PieceType::Bishop),
                'N' => Some(PieceType::Knight),
                _ => Some(PieceType::Queen),
            };
        }

        let to_chars: Vec<char> = to_str.chars().collect();
        let mut to_idx = 0;
        if to_chars.len() > 2 && to_chars[0].is_ascii_uppercase() {
            to_idx = 1;
        }
        if to_chars.len() < to_idx + 2 { return None; }
        let tx = (to_chars[to_idx] as u8).to_ascii_lowercase().checked_sub(b'a')? as usize + 1;
        let ty_str: String = to_chars[to_idx + 1..].iter().collect();
        let ty = ty_str.parse::<usize>().ok()?;

        if fx > 14 || fy > 14 || tx > 14 || ty > 14 { return None; }
        Some((piece, fx, fy, tx, ty, promotion))
    }
}