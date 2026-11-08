// game/src/pgn.rs
use crate::board::*;
use crate::state::GameState;

impl GameState {
    pub fn load_pgn(&mut self, pgn: &str) -> Result<(), String> {
        *self = GameState::new_with_variant(0);
        self.load_pgn_internal(pgn, false)
    }

    pub fn load_pgn_verbose(&mut self, pgn: &str) -> Result<(), String> {
        *self = GameState::new_with_variant(0);
        self.load_pgn_internal(pgn, true)
    }

    pub fn load_pgn_internal(&mut self, pgn: &str, lenient: bool) -> Result<(), String> {
        let clean = Self::strip_comments(pgn);
        let mut skipped_tokens: Vec<String> = Vec::new();

        'outer: for raw_line in clean.lines() {
            let line = raw_line.trim();

            if line.is_empty() || line.starts_with('[') {
                continue;
            }

            for token in line.split_whitespace() {
                let t = token.trim_matches(|c| c == '(' || c == ')');

                if t.is_empty() {
                    continue;
                }

                // Обычные результаты и служебные маркеры.
                if t == "1-0"
                    || t == "0-1"
                    || t == "1/2-1/2"
                    || t == "*"
                    || t == "1/2"
                    || t == "Draw"
                {
                    break 'outer;
                }

                // Результаты chess.com 4PC часто выглядят как:
                // Player1: 55 - Player2: 40 - ...
                if t.contains(':') {
                    break 'outer;
                }

                if t == ".." || t == "..." || t == "." {
                    continue;
                }

                // Удаляем префиксы номеров ходов: 1. 12. 47...
                let chars: Vec<char> = t.chars().collect();
                let mut start = 0;

                while start < chars.len()
                    && (chars[start].is_ascii_digit() || chars[start] == '.')
                {
                    start += 1;
                }

                let t = &t[start..];

                if t.is_empty() || t == ".." || t == "..." {
                    continue;
                }

                let t = t.trim_start_matches('.');

                if t.is_empty() {
                    continue;
                }

                // ============================================================
                // МАРКЕРЫ ОКОНЧАНИЯ / ПРОПУСКА ХОДА (STANDALONE TOKENS)
                // ============================================================
                //
                // R - Resigned (игрок сдался)
                // S - Stalemated / Dead (игрок застейлен или умер)
                // T - Timeout (игрок просрочил время)
                // # - Checkmate (мат)
                // #S - Stalemate + marker
                // Z - Null move / skip
                // -- - Null move
                // null - Null move
                //
                // Эти токены применяются к ТЕКУЩЕМУ цвету (self.turn).

                // Standalone маркеры статусов
                if t == "R" {
                    let c_idx = self.turn.index();
                    self.resigned[c_idx] = true;
                    self.pass_turn_and_record_skip_pub(false);
                    continue;
                }

                if t == "S" || t == "#" || t == "#S" {
                    let c_idx = self.turn.index();
                    self.dead_colors[c_idx] = true;
                    self.pass_turn_and_record_skip_pub(false);
                    continue;
                }

                if t == "T" {
                    // Timeout - считаем как dead
                    let c_idx = self.turn.index();
                    self.dead_colors[c_idx] = true;
                    self.pass_turn_and_record_skip_pub(false);
                    continue;
                }

                // Явные пропуски хода.
                if t == "Z" || t == "--" || t == "null" {
                    self.pass_turn_and_record_skip_pub(true);
                    continue;
                }

                let normalized = Self::normalize_pgn_token(t);

                if normalized.is_empty() {
                    continue;
                }

                // Проверка: не стал ли нормализованный токен маркером статуса?
                // (Например, "S" после strip аннотаций)
                if normalized == "R" {
                    let c_idx = self.turn.index();
                    self.resigned[c_idx] = true;
                    self.pass_turn_and_record_skip_pub(false);
                    continue;
                }

                if normalized == "S" || normalized == "#" || normalized == "#S" {
                    let c_idx = self.turn.index();
                    self.dead_colors[c_idx] = true;
                    self.pass_turn_and_record_skip_pub(false);
                    continue;
                }

                if normalized == "T" {
                    let c_idx = self.turn.index();
                    self.dead_colors[c_idx] = true;
                    self.pass_turn_and_record_skip_pub(false);
                    continue;
                }

                // ============================================================
                // COLOR-SYNC: пробуем все 4 цвета.
                // ============================================================
                let mut applied = false;
                let original_turn = self.turn;

                for _ in 0..4 {
                    let c_idx = self.turn.index();

                    // РОКИРОВКИ
                    if normalized == "O-O"
                        || normalized == "0-0"
                        || normalized == "O-O-O"
                        || normalized == "0-0-0"
                    {
                        if !self.dead_colors[c_idx] && !self.resigned[c_idx] {
                            let legal = self.generate_legal_moves();

                            let castle_type =
                                if normalized == "O-O" || normalized == "0-0" {
                                    1
                                } else {
                                    2
                                };

                            if let Some(&mv) = legal.iter().find(|m| {
                                m.castle == castle_type && m.piece == PieceType::King
                            }) {
                                if self.make_move(mv.from, mv.to, true) {
                                    applied = true;
                                    break;
                                }
                            }

                            if !applied {
                                if let Some(&mv) = legal.iter().find(|m| {
                                    m.piece == PieceType::King
                                        && ((m.from.0 as i32 - m.to.0 as i32).abs() == 2
                                            || (m.from.1 as i32 - m.to.1 as i32).abs() == 2)
                                }) {
                                    if self.make_move(mv.from, mv.to, true) {
                                        applied = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    // ОБЫЧНЫЕ ХОДЫ
                    else if let Some((piece, fx, fy, tx, ty, prom)) =
                        Self::parse_move_strict(&normalized)
                    {
                        if !self.dead_colors[c_idx] && !self.resigned[c_idx] {
                            // Попытка 1: обычный make_move
                            if self.make_move((fx, fy), (tx, ty), true) {
                                applied = true;
                                break;
                            }

                            // Попытка 2-4: lenient mode - ищем в легальных ходах
                            if lenient {
                                let legal = self.generate_legal_moves();

                                // 2. Точное совпадение from/to.
                                if let Some(&mv) = legal
                                    .iter()
                                    .find(|m| m.from == (fx, fy) && m.to == (tx, ty))
                                {
                                    if self.make_move(mv.from, mv.to, true) {
                                        applied = true;
                                        break;
                                    }
                                }

                                // 3. Совпадение фигуры и to.
                                if !applied {
                                    if let Some(&mv) = legal
                                        .iter()
                                        .find(|m| m.to == (tx, ty) && m.piece == piece)
                                    {
                                        if self.make_move(mv.from, mv.to, true) {
                                            applied = true;
                                            break;
                                        }
                                    }
                                }

                                // 4. Совпадение только to.
                                if !applied {
                                    if let Some(&mv) =
                                        legal.iter().find(|m| m.to == (tx, ty))
                                    {
                                        if self.make_move(mv.from, mv.to, true) {
                                            applied = true;
                                            break;
                                        }
                                    }
                                }
                            }
                        }

                        // ============================================================
                        // FALLBACK 1: DEAD KING WALKING
                        // ============================================================
                        // Если цвет уже dead, но PGN продолжает давать ходы короля,
                        // применяем их принудительно.
                        if !applied && self.dead_colors[c_idx] && piece == PieceType::King {
                            let k_pos = self.king_positions[c_idx];

                            if (fx, fy) == k_pos {
                                self.force_apply_dead_king_move(fx, fy, tx, ty);
                                applied = true;
                                break;
                            }
                        }

                        // ============================================================
                        // FALLBACK 2: GENERIC FORCE APPLY
                        // ============================================================
                        // Если ничего не помогло, применяем ход принудительно.
                        // Это ловит:
                        //   - захват короля (xK) = мат или захват dead king
                        //   - ходы в "шах" от мёртвых игроков
                        //   - превращения в сложных позициях
                        //   - ходы, которые make_move отклоняет по другим причинам
                        if !applied {
                            if self.force_apply_move_generic(piece, fx, fy, tx, ty, prom) {
                                applied = true;
                                break;
                            }
                        }
                    }

                    // Сдвигаем цвет без записи пропуска.
                    self.turn = self.turn.next();
                }

                if !applied {
                    self.turn = original_turn;

                    if lenient {
                        skipped_tokens.push(t.to_string());
                        continue;
                    }

                    return Err(format!(
                        "Parse error: '{}' for color {:?}",
                        t,
                        original_turn
                    ));
                }
            }
        }

        self.current_move_idx = self.move_history.len();

        if lenient && !skipped_tokens.is_empty() {
            let preview: Vec<&str> = skipped_tokens
                .iter()
                .take(20)
                .map(|s| s.as_str())
                .collect();

            return Err(format!(
                "WARN skipped {} tokens: {}",
                skipped_tokens.len(),
                preview.join(" ")
            ));
        }

        Ok(())
    }

    /// Нормализация PGN4-токенов chess.com:
    /// - убирает +, #, ?, !, T, S (trailing annotations);
    /// - нормализует 0-0 / o-o;
    /// - upper-case для lowercase piece letters;
    /// - удаляет букву съеденной фигуры после x.
    fn normalize_pgn_token(t: &str) -> String {
        let mut s = t.to_string();

        // Strip trailing annotations: +, #, ?, !, T, S
        // S добавлен для обработки #S и standalone S в конце ходов
        while let Some(c) = s.chars().last() {
            if matches!(c, '+' | '#' | '?' | '!' | 'T' | 'S') {
                s.pop();
            } else {
                break;
            }
        }

        if s.is_empty() {
            return s;
        }

        // Castling normalization.
        if s.eq_ignore_ascii_case("0-0") || s.eq_ignore_ascii_case("o-o") {
            return "O-O".to_string();
        }

        if s.eq_ignore_ascii_case("0-0-0") || s.eq_ignore_ascii_case("o-o-o") {
            return "O-O-O".to_string();
        }

        let chars: Vec<char> = s.chars().collect();
        
        if chars.is_empty() {
            return s;
        }

        let first = chars[0];

        // Uppercase first letter if it's a lowercase piece letter
        // and is followed by a file letter.
        //
        // Examples:
        //   kh11-l10 -> Kh11-l10
        //   qg3-i1   -> Qg3-i1
        //
        // But:
        //   k11-l10 remains as pawn move on file k.
        if matches!(first, 'k' | 'q' | 'r' | 'b' | 'n' | 'p') {
            if chars.len() > 1 {
                let second = chars[1];

                if second.is_ascii_alphabetic()
                    && ('a'..='n').contains(&second.to_ascii_lowercase())
                {
                    let mut new_s = String::with_capacity(s.len());
                    new_s.push(first.to_ascii_uppercase());
                    new_s.push_str(&s[first.len_utf8()..]);
                    s = new_s;
                }
            }
        }

        // Remove captured piece letter after 'x' or 'X'.
        //
        // Examples:
        //   Kh13xBh12 -> Kh13xh12
        //   Kn5xRn4   -> Kn5xn4
        //   Ka10xQb10 -> Ka10xb10
        //   Kf11xNf12 -> Kf11xf12
        if let Some(x_pos) = s.find(|c| c == 'x' || c == 'X') {
            let before_x = &s[..x_pos];
            let after_x = &s[x_pos + 1..];

            if !after_x.is_empty() {
                let next_char = after_x.chars().next().unwrap();

                if matches!(
                    next_char.to_ascii_uppercase(),
                    'K' | 'Q' | 'R' | 'B' | 'N' | 'P' | 'D'
                ) {
                    let rest_after_piece = &after_x[next_char.len_utf8()..];

                    if !rest_after_piece.is_empty() {
                        let file_char = rest_after_piece
                            .chars()
                            .next()
                            .unwrap()
                            .to_ascii_lowercase();

                        if ('a'..='n').contains(&file_char) {
                            s = format!("{}x{}", before_x, rest_after_piece);
                        }
                    }
                }
            }
        }

        s
    }

    fn strip_comments(pgn: &str) -> String {
        let mut s = String::with_capacity(pgn.len());
        let mut in_comment = false;

        let mut chars = pgn.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '{' {
                in_comment = true;
                continue;
            }

            if c == '}' {
                in_comment = false;
                continue;
            }

            if in_comment {
                continue;
            }

            if c == '.' && chars.peek() == Some(&'.') {
                chars.next();
                s.push(' ');
            } else {
                s.push(c);
            }
        }

        s
    }

    /// Гибкий парсер ходов.
    ///
    /// Поддерживает:
    /// - K/Q/R/B/N/P prefixes;
    /// - pawn moves without P: d11xc10;
    /// - captures: d11xc10, Kh13xh12;
    /// - promotions: =Q, =D, =R, =B, =N;
    /// - trailing promotion letter fallback: h1h2Q.
    pub fn parse_move_strict(
        t: &str,
    ) -> Option<(PieceType, usize, usize, usize, usize, Option<PieceType>)> {
        let s = t;

        if s.is_empty() {
            return None;
        }

        let mut piece = PieceType::Pawn;
        let mut idx = 0;

        let first = s.chars().next().unwrap();

        match first {
            'K' => {
                piece = PieceType::King;
                idx = 1;
            }
            'Q' => {
                piece = PieceType::Queen;
                idx = 1;
            }
            'R' => {
                piece = PieceType::Rook;
                idx = 1;
            }
            'B' => {
                piece = PieceType::Bishop;
                idx = 1;
            }
            'N' => {
                piece = PieceType::Knight;
                idx = 1;
            }
            'P' => {
                piece = PieceType::Pawn;
                idx = 1;
            }
            _ => {
                // If it starts with a file letter a-n, treat it as pawn move.
                if first.is_ascii_alphabetic()
                    && ('a'..='n').contains(&first.to_ascii_lowercase())
                {
                    piece = PieceType::Pawn;
                    idx = 0;
                } else {
                    return None;
                }
            }
        }

        let rest = &s[idx..];

        let sep_pos = rest.find(|c| c == '-' || c == 'x')?;
        let from_str = &rest[..sep_pos];

        if from_str.len() < 2 {
            return None;
        }

        let fx_char = from_str.chars().next()?.to_ascii_lowercase();

        if !('a'..='n').contains(&fx_char) {
            return None;
        }

        let fx = fx_char as usize - 'a' as usize + 1;
        let fy = from_str[1..].parse::<usize>().ok()?;

        let mut to_str = &rest[sep_pos + 1..];
        let mut promotion = None;

        // Promotion with '=': h1h2=Q, h1xh2=D.
        if let Some(eq_pos) = to_str.find('=') {
            let prom_part = &to_str[eq_pos + 1..];
            to_str = &to_str[..eq_pos];

            if let Some(pc) = prom_part.chars().next() {
                promotion = match pc.to_ascii_uppercase() {
                    'Q' | 'D' => Some(PieceType::Queen),
                    'R' => Some(PieceType::Rook),
                    'B' => Some(PieceType::Bishop),
                    'N' => Some(PieceType::Knight),
                    _ => Some(PieceType::Queen),
                };
            }
        } else if to_str.len() > 2 {
            // Fallback for trailing promotion letter: h1h2Q.
            let last = to_str.chars().last()?;

            if matches!(last.to_ascii_uppercase(), 'Q' | 'R' | 'B' | 'N' | 'D') {
                promotion = match last.to_ascii_uppercase() {
                    'Q' | 'D' => Some(PieceType::Queen),
                    'R' => Some(PieceType::Rook),
                    'B' => Some(PieceType::Bishop),
                    'N' => Some(PieceType::Knight),
                    _ => Some(PieceType::Queen),
                };

                to_str = &to_str[..to_str.len() - last.len_utf8()];
            }
        }

        // If there is still a captured piece letter before the destination square,
        // skip it.
        if let Some(first_to) = to_str.chars().next() {
            if matches!(
                first_to.to_ascii_uppercase(),
                'K' | 'Q' | 'R' | 'B' | 'N' | 'P' | 'D'
            ) && to_str.len() > 1
            {
                let after = &to_str[first_to.len_utf8()..];

                if after
                    .chars()
                    .next()
                    .map_or(false, |c| c.is_ascii_alphabetic())
                {
                    to_str = after;
                }
            }
        }

        if to_str.len() < 2 {
            return None;
        }

        let tx_char = to_str.chars().next()?.to_ascii_lowercase();

        if !('a'..='n').contains(&tx_char) {
            return None;
        }

        let tx = tx_char as usize - 'a' as usize + 1;
        let ty = to_str[1..].parse::<usize>().ok()?;

        if fx < 1 || fx > 14 || fy < 1 || fy > 14 || tx < 1 || tx > 14 || ty < 1 || ty > 14 {
            return None;
        }

        Some((piece, fx, fy, tx, ty, promotion))
    }

    /// Принудительное применение хода мёртвого короля.
    ///
    /// Это нужно для PGN4 с DeadKingWalking.
    /// Обычный make_move может не разрешать ходы dead-color,
    /// но PGN от chess.com их содержит.
    fn force_apply_dead_king_move(&mut self, fx: usize, fy: usize, tx: usize, ty: usize) {
        let from_idx = (fy - 1) * 14 + (fx - 1);
        let to_idx = (ty - 1) * 14 + (tx - 1);

        let moving_piece = self.board[from_idx];
        let target_piece = self.board[to_idx];

        // Если мёртвый король бьёт фигуру - обрабатываем
        if target_piece != 0 {
            if piece_type(target_piece) == Some(PieceType::King) {
                // Мёртвый король бьёт другого короля
                if let Some(tc) = piece_color(target_piece) {
                    self.dead_colors[tc.index()] = true;
                }
            }
        }

        let captured = if target_piece != 0 {
            Some(target_piece)
        } else {
            None
        };

        self.board[to_idx] = moving_piece;
        self.board[from_idx] = 0;

        let ci = self.turn.index();

        if piece_type(moving_piece) == Some(PieceType::King) {
            self.king_positions[ci] = (tx, ty);
        }

        let mv = crate::state::Move {
            from: (fx, fy),
            to: (tx, ty),
            piece: PieceType::King,
            color: self.turn,
            captured,
            promotion: None,
            castle: 0,
            en_passant: false,
            check: false,
            checkmate: false,
            skipped: false,
        };

        self.move_history.push(mv);
        self.ply_count += 1;
        self.turn = self.turn.next();
    }

    /// Универсальный force-apply: применяет ход без проверки легальности.
    /// Используется как fallback, когда make_move отклоняет ход
    /// (захват короля, ход мёртвого короля, и т.д.).
    fn force_apply_move_generic(
        &mut self,
        piece: PieceType,
        fx: usize,
        fy: usize,
        tx: usize,
        ty: usize,
        promotion: Option<PieceType>,
    ) -> bool {
        let from_idx = (fy - 1) * 14 + (fx - 1);
        let to_idx = (ty - 1) * 14 + (tx - 1);

        // Проверяем границы
        if from_idx >= 196 || to_idx >= 196 {
            return false;
        }

        let moving_piece = self.board[from_idx];

        // На from должна быть фигура текущего хода
        if moving_piece == 0 {
            return false;
        }
        if piece_color(moving_piece) != Some(self.turn) {
            return false;
        }

        let target_piece = self.board[to_idx];

        // Если цель - король, обрабатываем specially
        if target_piece != 0 && piece_type(target_piece) == Some(PieceType::King) {
            if let Some(target_color) = piece_color(target_piece) {
                let tci = target_color.index();

                if self.dead_colors[tci] {
                    // Захват уже мёртвого короля: убираем с доски
                    self.board[to_idx] = 0;
                } else {
                    // Мат живому королю: помечаем как dead
                    self.dead_colors[tci] = true;
                    // В DeadKingWalking король остаётся на доске,
                    // но если его СЪЕЛИ - убираем
                    self.board[to_idx] = 0;
                }
            }
        }

        // Применяем ход
        self.board[to_idx] = moving_piece;
        self.board[from_idx] = 0;

        // Обновляем позицию короля, если двигали короля
        if piece_type(moving_piece) == Some(PieceType::King) {
            self.king_positions[self.turn.index()] = (tx, ty);
        }

        // Обработка превращения
        if let Some(prom) = promotion {
            if piece_type(moving_piece) == Some(PieceType::Pawn) {
                // Заменяем пешку на фигуру превращения
                let color = self.turn;
                let new_piece = make_piece_val(color, prom, true);
                self.board[to_idx] = new_piece;
            }
        }

        let mv = crate::state::Move {
            from: (fx, fy),
            to: (tx, ty),
            piece,
            color: self.turn,
            captured: if target_piece != 0 {
                Some(target_piece)
            } else {
                None
            },
            promotion,
            castle: 0,
            en_passant: false,
            check: false,
            checkmate: false,
            skipped: false,
        };

        self.move_history.push(mv);
        self.ply_count += 1;
        self.turn = self.turn.next();

        true
    }

    pub fn pass_turn_and_record_skip_pub(&mut self, _is_explicit: bool) {
        let skip_color = self.turn;

        let mv = crate::state::Move {
            from: (0, 0),
            to: (0, 0),
            piece: PieceType::King,
            color: skip_color,
            captured: None,
            promotion: None,
            castle: 0,
            en_passant: false,
            check: false,
            checkmate: false,
            skipped: true,
        };

        self.move_history.push(mv);
        self.ply_count += 1;
        self.turn = self.turn.next();
    }
}