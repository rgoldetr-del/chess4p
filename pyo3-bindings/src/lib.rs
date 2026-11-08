use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use numpy::{PyArray3, PyArrayMethods};

use chess4p_game::{
    board_to_tensor_slice,
    Color,
    GameState,
    UndoInfo,
    TENSOR_PLANES,
    TENSOR_SIZE,
};

#[pyclass]
struct PyGameState {
    state: GameState,
    undo_stack: Vec<UndoInfo>,
}

#[pymethods]
impl PyGameState {
    // ========================================================
    // Constructor / clone
    // ========================================================

    #[new]
    fn new() -> Self {
        Self {
            state: GameState::new(),
            undo_stack: Vec::new(),
        }
    }

    fn clone_state(&self) -> Self {
        Self {
            state: self.state.clone_light(),
            undo_stack: Vec::new(),
        }
    }

    // ========================================================
    // Navigation / PGN
    // ========================================================

fn set_board(
    &mut self,
    entries: Vec<(usize, usize, usize, usize, bool)>, // x,y,color,piece,promoted
    turn: usize,
    scores: Vec<i32>,
    dead: Vec<bool>,
    resigned: Vec<bool>,
) -> bool {
    use chess4p_game::*;
    self.undo_stack.clear();
    let mut board = [EMPTY; 196];
    let mut piece_positions: [Vec<usize>; 4] = Default::default();
    let mut king_positions = [(0, 0); 4];
    for (x, y, c, p, prom) in entries {
        if x < 1 || x > 14 || y < 1 || y > 14 || c > 3 || p > 5 { return false; }
        let piece = [PieceType::Pawn, PieceType::Knight, PieceType::Bishop,
                     PieceType::Rook, PieceType::Queen, PieceType::King][p];
        let val = make_piece_val(Color::ALL[c], piece, prom);
        let idx = sq_index(x, y);
        board[idx] = val;
        piece_positions[c].push(idx);
        if piece == PieceType::King { king_positions[c] = (x, y); }
    }
    let st = &mut self.state;
    st.board = board;
    st.piece_positions = piece_positions;
    st.king_positions = king_positions;
    st.turn = Color::ALL[turn.min(3)];
    for i in 0..4 {
        st.scores[i] = *scores.get(i).unwrap_or(&0);
        st.dead_colors[i] = *dead.get(i).unwrap_or(&false);
        st.resigned[i] = *resigned.get(i).unwrap_or(&false);
    }
        st.en_passants = [None; 4];
        st.halfmove_clock = 0;

        // ============================================================
        // Консервативный сброс прав рокировки.
        // Разрешаем рокировку только если король и ладья стоят
        // на стартовых клетках. Иначе сервер отклонит ход,
        // а retry-логика потратит время впустую.
        // ============================================================
        let king_starts: [(usize, usize); 4] = [
            (8, 1), (1, 7), (7, 14), (14, 8),
        ];
        let rook_starts: [[(usize, usize); 2]; 4] = [
            [(4, 1), (11, 1)],    // Red
            [(1, 4), (1, 11)],    // Blue
            [(4, 14), (11, 14)],  // Yellow
            [(14, 4), (14, 11)],  // Green
        ];
        for i in 0..4 {
            let color = Color::ALL[i];
            let (kx, ky) = king_positions[i];
            let king_ok = kx != 0
                && king_positions[i] == king_starts[i]
                && board[sq_index(kx, ky)]
                    == make_piece_val(color, PieceType::King, false);

            st.castling_rights[i][0] = king_ok;
            st.castling_rights[i][1] = king_ok
                && board[sq_index(rook_starts[i][0].0, rook_starts[i][0].1)]
                    == make_piece_val(color, PieceType::Rook, false);
            st.castling_rights[i][2] = king_ok
                && board[sq_index(rook_starts[i][1].0, rook_starts[i][1].1)]
                    == make_piece_val(color, PieceType::Rook, false);
            st.castling_rights[i][3] = false;
        }

        let z = get_zobrist();
        let mut hash = 0u64;
    for i in 0..196 {
        if board[i] != EMPTY { hash ^= z.pieces[i][board[i] as usize]; }
    }
        hash ^= z.turn[st.turn.index()];
        for c in 0..4 {
            for r in 0..4 {
                if st.castling_rights[c][r] {
                    hash ^= z.castling[c][r];
                }
            }
        }
        st.zobrist_hash = hash;
    st.hash_history.clear();
    st.hash_counts.clear();
    st.hash_history.push(hash);
    st.hash_counts.insert(hash, 1);
    true
}

    fn go_to_move(&mut self, move_idx: usize) -> bool {
        self.undo_stack.clear();
        self.state.go_to_move(move_idx)
    }

    fn get_move_at(&self, idx: usize) -> Vec<usize> {
        if idx < self.state.move_history.len() {
            let m = &self.state.move_history[idx];

            if m.skipped {
                return vec![];
            }

            vec![m.from.0, m.from.1, m.to.0, m.to.1]
        } else {
            vec![]
        }
    }

    fn load_pgn(&mut self, pgn: &str) -> bool {
        self.undo_stack.clear();
        self.state.load_pgn(pgn).is_ok()
    }

    fn load_pgn_verbose(&mut self, pgn: &str) -> String {
        self.undo_stack.clear();

        match self.state.load_pgn_verbose(pgn) {
            Ok(()) => String::new(),
            Err(e) => e,
        }
    }

    // ========================================================
    // Legal moves
    // ========================================================

    /// Старый формат:
    /// [fx, fy, tx, ty, fx, fy, tx, ty, ...]
    fn get_legal_moves(&mut self) -> Vec<usize> {
        let moves = self.state.generate_legal_moves();

        let mut result = Vec::with_capacity(moves.len() * 4);

        for m in moves {
            result.push(m.from.0);
            result.push(m.from.1);
            result.push(m.to.0);
            result.push(m.to.1);
        }

        result
    }

    /// Новый формат:
    /// список action indices.
    fn get_legal_move_indices(&mut self) -> Vec<usize> {
        self.state.legal_move_indices()
    }

    // ========================================================
    // Make move API
    // ========================================================

    /// Старый strict API.
    fn make_move(&mut self, fx: usize, fy: usize, tx: usize, ty: usize) -> bool {
        self.undo_stack.clear();
        self.state.make_move((fx, fy), (tx, ty), true)
    }

    /// Старый unchecked API.
    fn make_move_unchecked(
        &mut self,
        fx: usize,
        fy: usize,
        tx: usize,
        ty: usize,
    ) -> bool {
        self.undo_stack.clear();
        self.state
            .make_move_unchecked((fx, fy), (tx, ty), true, None)
    }

    /// Новый быстрый API для self-play:
    /// ход по индексу + detect_end.
    fn make_move_fast_idx(&mut self, idx: usize) -> bool {
        self.undo_stack.clear();
        self.state.make_move_fast_idx(idx)
    }

    // ========================================================
    // MCTS make/unmake stack
    // ========================================================

    /// MCTS push without detect_end.
    /// Кладёт undo в stack.
    fn mcts_push_idx(&mut self, idx: usize) -> bool {
        if let Some(undo) = self.state.mcts_push_idx(idx) {
            self.undo_stack.push(undo);
            true
        } else {
            false
        }
    }

    /// Unmake последнего MCTS-хода.
    fn mcts_pop(&mut self) -> bool {
        if let Some(undo) = self.undo_stack.pop() {
            self.state.unmake_move(undo);
            true
        } else {
            false
        }
    }

    /// Unmake всех MCTS-ходов.
    fn mcts_unmake_all(&mut self) {
        while let Some(undo) = self.undo_stack.pop() {
            self.state.unmake_move(undo);
        }
    }

    /// Опасно: просто очищает stack без unmake.
    /// Использовать только если состояние уже сброшено вручную.
    fn mcts_stack_clear(&mut self) {
        self.undo_stack.clear();
    }

    fn mcts_stack_len(&self) -> usize {
        self.undo_stack.len()
    }

    /// Сделать ход, получить turn, затем откатить.
    /// Полезно для раскрытия детей в MCTS без клонов.
    fn get_turn_after_idx(&mut self, idx: usize) -> Option<usize> {
        let stack_len = self.undo_stack.len();

        if !self.mcts_push_idx(idx) {
            return None;
        }

        let turn = self.state.turn.index();

        while self.undo_stack.len() > stack_len {
            self.mcts_pop();
        }

        Some(turn)
    }

    // ========================================================
    // Dead / resigned API
    // ========================================================

    /// Старый API.
    /// Теперь использует fast dead king logic.
    fn auto_play_dead(&mut self) -> bool {
        self.undo_stack.clear();
        self.state.auto_play_dead_fast()
    }

    /// Новый API.
    fn auto_play_dead_fast(&mut self) -> bool {
        self.undo_stack.clear();
        self.state.auto_play_dead_fast()
    }

    fn resign_player(&mut self, color_idx: usize) -> bool {
        if color_idx >= 4 {
            return false;
        }

        self.undo_stack.clear();
        self.state.resign_player(Color::ALL[color_idx]);

        true
    }

    fn is_turn_dead(&self) -> bool {
        self.state.is_turn_dead()
    }

    fn is_turn_resigned(&self) -> bool {
        self.state.is_turn_resigned()
    }

    fn get_dead(&self) -> Vec<bool> {
        self.state.dead_colors.to_vec()
    }

    fn get_resigned(&self) -> Vec<bool> {
        self.state.resigned.to_vec()
    }

    // ========================================================
    // Game state queries
    // ========================================================

    fn is_game_over(&self) -> bool {
        self.state.is_game_over()
    }

    fn get_turn(&self) -> usize {
        self.state.turn.index()
    }

    fn get_move_count(&self) -> usize {
        self.state.move_history.len()
    }

    fn get_real_move_count(&self) -> usize {
        self.state
            .move_history
            .iter()
            .filter(|m| !m.skipped || m.castle == 255)
            .count()
    }

    fn get_ply_count(&self) -> usize {
        self.state.get_ply_count()
    }

    fn get_halfmove_clock(&self) -> usize {
        self.state.halfmove_clock
    }

    fn get_repetition_count(&self) -> u8 {
        self.state.repetition_count()
    }

    fn get_zobrist_hash(&self) -> u64 {
        self.state.zobrist_hash
    }

    fn get_scores(&self) -> Vec<i32> {
        self.state.scores.to_vec()
    }

    fn get_active_count(&self) -> usize {
        self.state.active_count()
    }

    fn get_alive_count(&self) -> usize {
        self.state.alive_count()
    }

    fn get_piece_count(&self) -> usize {
        self.state.piece_positions.iter().map(|v| v.len()).sum()
    }

    fn get_last_move_idx(&self) -> Option<usize> {
        self.state.last_move_index()
    }

    fn any_active_in_check(&self) -> bool {
        self.state.any_active_in_check()
    }

    fn active_score_spread(&self) -> i32 {
        self.state.active_score_spread()
    }

    /// Training-only adjudication helper.
    fn is_training_adjudication_draw(
        &self,
        min_ply: usize,
        max_score_spread: i32,
    ) -> bool {
        self.state
            .is_training_adjudication_draw(min_ply, max_score_spread)
    }

    // ========================================================
    // Rewards / terminal values
    // ========================================================

    /// Старый совместимый hard reward.
    fn get_final_rewards(&self) -> Vec<f32> {
        self.state.final_rewards().to_vec()
    }

    /// Новый expected reward с усреднением тай-брейков.
    fn get_final_rewards_expected(&self) -> Vec<f32> {
        self.state.final_rewards_expected().to_vec()
    }

    /// Soft placement matrix, flat:
    /// 16 float = 4 players × 4 places.
    fn get_final_placement_soft(&self) -> Vec<f32> {
        self.state.final_placement_soft_flat()
    }

    /// Если у текущего игрока нет legal moves,
    /// возвращает terminal expected rewards.
    /// Иначе None.
    fn terminal_value_if_no_moves(&self) -> Option<Vec<f32>> {
        self.state
            .terminal_value_if_no_moves()
            .map(|v| v.to_vec())
    }

    // ========================================================
    // RNG / determinism
    // ========================================================

    fn set_seed(&mut self, seed: u64) {
        self.state.set_seed(seed);
    }

    fn set_deterministic_dead_king(&mut self, deterministic: bool) {
        self.state.set_deterministic_dead_king(deterministic);
    }

    // ========================================================
    // Tensor
    // ========================================================

    /// Новый tensor API под 100 плоскостей.
    ///
    /// Python side:
    ///   tensor = np.zeros((chess4p_ai.TENSOR_PLANES, 14, 14), dtype=np.float32)
    ///   game.board_to_tensor(tensor)
    fn board_to_tensor<'py>(
        &self,
        _py: Python<'py>,
        tensor: &Bound<'py, PyArray3<f32>>,
    ) -> PyResult<()> {
        let mut arr = unsafe { tensor.as_array_mut() };

        if arr.len() != TENSOR_SIZE {
            return Err(PyValueError::new_err(format!(
                "board_to_tensor expects tensor length {} ({}×14×14), got {}",
                TENSOR_SIZE,
                TENSOR_PLANES,
                arr.len()
            )));
        }

        // Fast path: contiguous slice.
        if let Some(slice) = arr.as_slice_mut() {
            board_to_tensor_slice(&self.state, slice);
            return Ok(());
        }

        // Fallback: non-contiguous array.
        let mut flat = vec![0.0f32; TENSOR_SIZE];
        board_to_tensor_slice(&self.state, &mut flat);

        for plane in 0..TENSOR_PLANES {
            for y in 0..14 {
                for x in 0..14 {
                    arr[[plane, y, x]] = flat[plane * 196 + y * 14 + x];
                }
            }
        }

        Ok(())
    }
}

// ============================================================
// Python module
// ============================================================

#[pymodule]
fn chess4p_ai(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGameState>()?;

    m.add("TENSOR_PLANES", TENSOR_PLANES)?;
    m.add("TENSOR_SIZE", TENSOR_SIZE)?;
    m.add("ACTION_SPACE", 38416usize)?;

    Ok(())
}