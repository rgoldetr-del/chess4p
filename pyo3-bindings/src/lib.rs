use pyo3::prelude::*;
use numpy::{PyArray3, PyArrayMethods};
use ndarray::s;

use chess4p_game::{
    GameState, PieceType, Color, EMPTY,
    piece_color, piece_type, is_promoted,
    board_to_tensor_slice,
};

#[pyclass]
struct PyGameState {
    state: GameState,
}

#[pymethods]
impl PyGameState {
    #[new]
    fn new() -> Self {
        Self {
            state: GameState::new(),
        }
    }

    fn clone_state(&self) -> Self {
        Self {
            state: self.state.clone_light(),
        }
    }

    fn go_to_move(&mut self, move_idx: usize) -> bool {
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
        self.state.load_pgn(pgn).is_ok()
    }

    fn load_pgn_verbose(&mut self, pgn: &str) -> String {
        match self.state.load_pgn(pgn) {
            Ok(()) => String::new(),
            Err(e) => e,
        }
    }

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

    fn get_legal_move_indices(&mut self) -> Vec<usize> {
        let moves = self.state.generate_legal_moves();
        GameState::get_legal_move_indices(&moves)
    }

    fn make_move(&mut self, fx: usize, fy: usize, tx: usize, ty: usize) -> bool {
        self.state.make_move((fx, fy), (tx, ty), true)
    }

    fn make_move_unchecked(&mut self, fx: usize, fy: usize, tx: usize, ty: usize) -> bool {
        self.state.make_move_unchecked((fx, fy), (tx, ty), true, None)
    }

    fn auto_play_dead(&mut self) -> bool {
        self.state.play_random_dead_king_move()
    }

    fn is_game_over(&self) -> bool {
        self.state.is_game_over()
    }

    fn is_turn_dead(&self) -> bool {
        self.state.is_turn_dead()
    }

    fn is_turn_resigned(&self) -> bool {
        let c_idx = GameState::color_index(self.state.turn);
        self.state.resigned[c_idx] && !self.state.dead_colors[c_idx]
    }

    fn get_turn(&self) -> usize {
        GameState::color_index(self.state.turn)
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

    fn board_to_tensor<'py>(&self, _py: Python<'py>, tensor: &Bound<'py, PyArray3<f32>>) {
        let mut arr = unsafe { tensor.as_array_mut() };
        arr.fill(0.0);

        // Используем общий tensor-модуль через flat slice
        let mut flat = vec![0.0f32; 73 * 14 * 14];
        board_to_tensor_slice(&self.state, &mut flat);

        // Копируем в ndarray [73, 14, 14]
        for plane in 0..73 {
            for y in 0..14 {
                for x in 0..14 {
                    arr[[plane, y, x]] = flat[plane * 196 + y * 14 + x];
                }
            }
        }
    }

    fn get_final_rewards(&self) -> Vec<f32> {
        self.state.final_rewards().to_vec()
    }
}

#[pymodule]
fn chess4p_ai(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGameState>()?;
    Ok(())
}