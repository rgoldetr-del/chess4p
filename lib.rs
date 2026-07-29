use godot::prelude::*;
use ort::{
    ep::{DirectML, CUDA},
    session::builder::GraphOptimizationLevel,
    session::Session,
    value::Value,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use chess4p_game::{
    piece_color, piece_type, is_promoted, is_valid_square,
    Board, Color, GameState, Move, PieceType,
    board_to_tensor_slice, TENSOR_PLANES,
};

const CHANNELS: usize = 256;
const RES_BLOCKS: usize = 30;
const TENSOR_SIZE: usize = TENSOR_PLANES * 14 * 14;
const POLICY_SIZE: usize = 38416;
const VALUE_SIZE: usize = 16;

// ============================================================
// MCTS Node
// ============================================================
struct MctsNode {
    move_made: Option<Move>,
    parent: Option<usize>,
    children: Vec<usize>,
    visits: u32,
    total_value: [f32; 4],
    prior: f32,
    is_terminal: bool,
    virtual_loss: f32,
}

impl MctsNode {
    fn new(move_made: Option<Move>, parent: Option<usize>, prior: f32, is_terminal: bool) -> Self {
        Self {
            move_made,
            parent,
            children: Vec::new(),
            visits: 0,
            total_value: [0.0; 4],
            prior,
            is_terminal,
            virtual_loss: 0.0,
        }
    }

    fn q(&self, perspective: Color) -> f32 {
        let idx = GameState::color_index(perspective);
        if self.visits == 0 {
            return -self.virtual_loss;
        }
        (self.total_value[idx] - self.virtual_loss) / self.visits as f32
    }
}

// ============================================================
// ChessAnalyzer
// ============================================================
#[derive(GodotClass)]
#[class(base=Node)]
pub struct ChessAnalyzer {
    game_state: GameState,
    onnx_model: Option<Arc<Mutex<Session>>>,
    tensor_buffer: Vec<f32>,
    bot_thread: Option<thread::JoinHandle<()>>,
    bot_result: Arc<
        Mutex<
            Option<(
                i64,
                i64,
                i64,
                i64,
                String,
                Vec<(usize, usize, usize, usize, String)>,
                f32,
            )>,
        >,
    >,
    bot_cancel_flag: Arc<AtomicBool>,
    analyzer_thread: Option<thread::JoinHandle<()>>,
    analyzer_result: Arc<Mutex<Option<Vec<(f32, i64, i64, i64, i64)>>>>,
    analyzer_cancel_flag: Arc<AtomicBool>,
    bot_intents: HashMap<usize, (String, Vec<(usize, usize, usize, usize, String)>)>,
    transposition_table: HashMap<u64, (u32, [f32; 4], Vec<f32>)>, // hash -> (visits, value, policy)
}

#[godot_api]
impl INode for ChessAnalyzer {
    fn init(_base: Base<Node>) -> Self {
        Self {
            game_state: GameState::new(),
            onnx_model: None,
            tensor_buffer: vec![0.0; TENSOR_SIZE],
            bot_thread: None,
            bot_result: Arc::new(Mutex::new(None)),
            bot_cancel_flag: Arc::new(AtomicBool::new(false)),
            analyzer_thread: None,
            analyzer_result: Arc::new(Mutex::new(None)),
            analyzer_cancel_flag: Arc::new(AtomicBool::new(false)),
            bot_intents: HashMap::new(),
            transposition_table: HashMap::new(),
        }
    }
}

#[godot_api]
impl ChessAnalyzer {
    // ============================================================
    // BOT / MCTS
    // ============================================================
    #[func]
    fn request_neural_move(&mut self, time_ms: i64) -> bool {
        if self.bot_thread.is_some() {
            return false;
        }
        let model = match self.onnx_model.clone() {
            Some(m) => m,
            None => return false,
        };
        if let Ok(mut res) = self.bot_result.lock() {
            *res = None;
        }
        self.bot_cancel_flag.store(false, Ordering::SeqCst);

        let state_clone = self.game_state.clone_light();
        let result_arc = self.bot_result.clone();
        let cancel_arc = self.bot_cancel_flag.clone();
        let time_ms = time_ms.max(0) as u64;

        self.bot_thread = Some(thread::spawn(move || {
            let mut nodes: Vec<MctsNode> = Vec::with_capacity(8192);
            nodes.push(MctsNode::new(None, None, 1.0, state_clone.is_game_over()));

            let batch_size: usize = 16;
            let max_simulations: u32 = if time_ms > 0 { 1_000_000 } else { 1600 };
            let budget: Option<Duration> = if time_ms > 0 {
                Some(Duration::from_millis(time_ms))
            } else {
                None
            };
            let start = Instant::now();
            let mut simulations_done: u32 = 0;

            loop {
                if cancel_arc.load(Ordering::SeqCst) {
                    return;
                }
                if simulations_done >= max_simulations {
                    break;
                }
                if let Some(b) = budget {
                    if simulations_done > 0 && start.elapsed() >= b {
                        break;
                    }
                }

                let mut pending_states: Vec<GameState> = Vec::with_capacity(batch_size);
                let mut pending_indices: Vec<usize> = Vec::with_capacity(batch_size);
                let mut pending_paths: Vec<Vec<usize>> = Vec::with_capacity(batch_size);
                let mut pending_terminal_values: Vec<Option<[f32; 4]>> =
                    Vec::with_capacity(batch_size);

                // 1. Selection
                for _ in 0..batch_size {
                    let mut current_idx = 0;
                    let mut sim_state = state_clone.clone_light();
                    let mut path: Vec<usize> = Vec::new();

                    for _ in 0..512 {
                        if nodes[current_idx].is_terminal || nodes[current_idx].children.is_empty() {
                            break;
                        }
                        let next_idx = Self::select_child(&mut nodes, current_idx, sim_state.turn);
                        path.push(current_idx);
                        current_idx = next_idx;
                        if let Some(mv) = nodes[current_idx].move_made {
                            sim_state.make_move_unchecked(mv.from, mv.to, true, None);
                        }
                    }

                    if sim_state.is_game_over() {
                        pending_terminal_values.push(Some(Self::eval_final_scores(&sim_state)));
                    } else {
                        pending_terminal_values.push(None);
                    }
                    pending_states.push(sim_state);
                    pending_indices.push(current_idx);
                    pending_paths.push(path);
                }

                // 2. Separate terminal / non-terminal
                let mut non_term_states: Vec<GameState> = Vec::new();
                let mut non_term_orig_indices: Vec<usize> = Vec::new();
                for i in 0..batch_size {
                    if pending_terminal_values[i].is_none() {
                        non_term_states.push(pending_states[i].clone_light());
                        non_term_orig_indices.push(i);
                    }
                }

                let mut nn_policies: Vec<Vec<f32>> = Vec::new();
                let mut nn_values: Vec<[f32; 4]> = Vec::new();

                // 3. Batched inference
                if !non_term_states.is_empty() {
                    let mut model_guard = match model.lock() {
                        Ok(g) => g,
                        Err(_) => return,
                    };
                    match run_model_batch(&mut model_guard, &non_term_states, non_term_states.len())
                    {
                        Some((policies, values)) => {
                            nn_policies = policies;
                            nn_values = values;
                        }
                        None => return,
                    }
                }

                // 4. Expansion + backpropagation
                for i in 0..batch_size {
                    let current_idx = pending_indices[i];
                    let value: [f32; 4] = if let Some(val) = pending_terminal_values[i] {
                        val
                    } else {
                        let nt_idx = non_term_orig_indices
                            .iter()
                            .position(|&x| x == i)
                            .unwrap();

                        if nodes[current_idx].children.is_empty() {
                            let mut sim_state_clone = pending_states[i].clone_light();
                            let legal_moves = sim_state_clone.generate_legal_moves();
                            if !legal_moves.is_empty() {
                                let indices = GameState::get_legal_move_indices(&legal_moves);
                                let policy = &nn_policies[nt_idx];
                                let priors = softmax_priors(policy, &indices);
                                for (mv_idx, mv) in legal_moves.iter().enumerate() {
                                    let mut next_state = sim_state_clone.clone_light();
                                    next_state.make_move_unchecked(mv.from, mv.to, true, None);
                                    let child = MctsNode::new(
                                        Some(*mv),
                                        Some(current_idx),
                                        priors[mv_idx],
                                        next_state.is_game_over(),
                                    );
                                    let new_idx = nodes.len();
                                    nodes.push(child);
                                    nodes[current_idx].children.push(new_idx);
                                }
                            }
                        }
                        nn_values[nt_idx]
                    };

                    nodes[current_idx].visits += 1;
                    nodes[current_idx].virtual_loss =
                        (nodes[current_idx].virtual_loss - 1.0).max(0.0);
                    for j in 0..4 {
                        nodes[current_idx].total_value[j] += value[j];
                    }

                    let mut p_idx = current_idx;
                    while let Some(parent) = nodes[p_idx].parent {
                        nodes[parent].visits += 1;
                        nodes[parent].virtual_loss =
                            (nodes[parent].virtual_loss - 1.0).max(0.0);
                        for j in 0..4 {
                            nodes[parent].total_value[j] += value[j];
                        }
                        p_idx = parent;
                    }
                }

                simulations_done += batch_size as u32;

                if nodes[0].children.is_empty() {
                    break;
                }
            }

            if nodes[0].children.is_empty() {
                return;
            }

            // 5. Aggregate visits by move
            let root_children = nodes[0].children.clone();
            let mut visits_by_move: HashMap<(usize, usize, usize, usize), u32> = HashMap::new();
            let mut prior_by_move: HashMap<(usize, usize, usize, usize), f32> = HashMap::new();

            for &child_idx in &root_children {
                if let Some(mv) = nodes[child_idx].move_made {
                    let key = Self::move_key(&mv);
                    *visits_by_move.entry(key).or_insert(0) += nodes[child_idx].visits;
                    let prior_entry = prior_by_move.entry(key).or_insert(0.0);
                    if nodes[child_idx].prior > *prior_entry {
                        *prior_entry = nodes[child_idx].prior;
                    }
                }
            }

            let total_visits: u32 = visits_by_move.values().sum();
            let best_key = if total_visits > 0 {
                match visits_by_move
                    .iter()
                    .max_by_key(|(_, v)| **v)
                    .map(|(k, _)| *k)
                {
                    Some(k) => k,
                    None => return,
                }
            } else {
                match prior_by_move.iter().max_by(|a, b| {
                    a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)
                }) {
                    Some((k, _)) => *k,
                    None => return,
                }
            };

            let mut best_child_idx = root_children[0];
            let mut best_child_visits = 0;
            let mut best_child_prior = -1.0f32;
            for &child_idx in &root_children {
                if let Some(mv) = nodes[child_idx].move_made {
                    if Self::move_key(&mv) == best_key {
                        let vis = nodes[child_idx].visits;
                        let pri = nodes[child_idx].prior;
                        if vis > best_child_visits
                            || (vis == best_child_visits && pri > best_child_prior)
                        {
                            best_child_visits = vis;
                            best_child_prior = pri;
                            best_child_idx = child_idx;
                        }
                    }
                }
            }

            let best_move = match nodes[best_child_idx].move_made {
                Some(mv) => mv,
                None => return,
            };

            // 6. PV line
            let mut pv_moves: Vec<Move> = Vec::new();
            let mut current = best_child_idx;
            for _ in 0..5 {
                if let Some(mv) = nodes[current].move_made {
                    pv_moves.push(mv);
                } else {
                    break;
                }
                if nodes[current].children.is_empty() {
                    break;
                }
                let mut next_best = nodes[current].children[0];
                let mut next_visits = 0;
                let mut next_prior = -1.0f32;
                for &child_idx in &nodes[current].children {
                    let vis = nodes[child_idx].visits;
                    let pri = nodes[child_idx].prior;
                    if vis > next_visits || (vis == next_visits && pri > next_prior) {
                        next_visits = vis;
                        next_prior = pri;
                        next_best = child_idx;
                    }
                }
                if next_visits == 0 && next_prior <= 0.0 {
                    break;
                }
                current = next_best;
            }
            if pv_moves.is_empty() {
                pv_moves.push(best_move);
            }

            let mut intent_text = format!("План бота ({} полуходов): ", pv_moves.len());
            let mut intent_arrows: Vec<(usize, usize, usize, usize, String)> = Vec::new();
            for (i, mv) in pv_moves.iter().enumerate() {
                let arrow_type = if i == 0 {
                    "plan_main".to_string()
                } else {
                    "plan_continuation".to_string()
                };
                intent_arrows.push((mv.from.0, mv.from.1, mv.to.0, mv.to.1, arrow_type));
                intent_text.push_str(&format!(
                    "{}. {} -> {}; ",
                    i + 1,
                    Self::coord_to_str(mv.from.0, mv.from.1),
                    Self::coord_to_str(mv.to.0, mv.to.1)
                ));
            }

            let bot_color = state_clone.turn;
            let root_value = nodes[best_child_idx].q(bot_color);

            if let Ok(mut res) = result_arc.lock() {
                *res = Some((
                    best_move.from.0 as i64,
                    best_move.from.1 as i64,
                    best_move.to.0 as i64,
                    best_move.to.1 as i64,
                    intent_text,
                    intent_arrows,
                    root_value,
                ));
            }
        }));
        true
    }

    fn select_child(nodes: &mut Vec<MctsNode>, parent_idx: usize, perspective: Color) -> usize {
        let c_puct = 1.5;
        let fpu = -0.5; // First Play Urgency для непросмотренных узлов
        let total_parent_visits = (nodes[parent_idx].visits as f32 + 1.0).sqrt();
        let mut best_score = f32::MIN;
        let mut best_child_idx = nodes[parent_idx].children[0];

        for &child_idx in &nodes[parent_idx].children {
            let q = if nodes[child_idx].visits == 0 {
                fpu // FPU для первого посещения
            } else {
                nodes[child_idx].q(perspective)
            };
            let u = c_puct * nodes[child_idx].prior * total_parent_visits
                / (1.0 + nodes[child_idx].visits as f32);
            let score = q + u;
            if score > best_score {
                best_score = score;
                best_child_idx = child_idx;
            }
        }
        nodes[best_child_idx].virtual_loss += 1.0;
        best_child_idx
    }

    fn eval_final_scores(state: &GameState) -> [f32; 4] {
        state.final_rewards()
    }

    fn move_key(m: &Move) -> (usize, usize, usize, usize) {
        (m.from.0, m.from.1, m.to.0, m.to.1)
    }

    fn coord_to_str(x: usize, y: usize) -> String {
        let file = (b'a' + (x as u8) - 1) as char;
        format!("{}{}", file, y)
    }

    #[func]
    fn is_bot_done(&self) -> bool {
        if let Some(thread) = &self.bot_thread {
            return thread.is_finished();
        }
        false
    }

    #[func]
    fn get_bot_result(&mut self) -> Variant {
        if let Some(thread) = self.bot_thread.take() {
            let _ = thread.join();
        }
        if self.bot_cancel_flag.load(Ordering::SeqCst) {
            return Variant::from(Dictionary::<Variant, Variant>::new());
        }
        if let Ok(mut res) = self.bot_result.lock() {
            if let Some((fx, fy, tx, ty, text, arrows_vec, value)) = res.take() {
                let move_idx = self.game_state.current_move_idx;
                self.bot_intents.insert(move_idx, (text, arrows_vec));

                let mut dict = Dictionary::<Variant, Variant>::new();
                dict.set(&Variant::from("fx"), &Variant::from(fx));
                dict.set(&Variant::from("fy"), &Variant::from(fy));
                dict.set(&Variant::from("tx"), &Variant::from(tx));
                dict.set(&Variant::from("ty"), &Variant::from(ty));
                dict.set(&Variant::from("value"), &Variant::from(value));
                return Variant::from(dict);
            }
        }
        Variant::from(Dictionary::<Variant, Variant>::new())
    }

    // ============================================================
    // ONNX model
    // ============================================================
    #[func]
    fn load_neural_model(&mut self, path: String) -> bool {
        let builder = match Session::builder() {
            Ok(b) => b,
            Err(_) => return false,
        };
        let builder = match builder.with_intra_threads(4) {
            Ok(b) => b,
            Err(_) => return false,
        };
        let builder = match builder.with_optimization_level(GraphOptimizationLevel::Level3) {
            Ok(b) => b,
            Err(_) => return false,
        };
        let mut builder = match builder.with_execution_providers([
            DirectML::default().build(),
            CUDA::default().build(),
        ]) {
            Ok(b) => b,
            Err(_) => return false,
        };

        match builder.commit_from_file(&path) {
            Ok(session) => {
                godot_print!("🧠 ONNX модель загружена.");
                self.onnx_model = Some(Arc::new(Mutex::new(session)));
                true
            }
            Err(e) => {
                godot_error!("Не удалось загрузить ONNX: {:?}", e);
                false
            }
        }
    }

    // ============================================================
    // Analysis
    // ============================================================
    #[func]
    fn request_analysis(&mut self, count: i32) -> bool {
        if self.analyzer_thread.is_some() {
            return false;
        }
        let model_arc = match self.onnx_model.clone() {
            Some(m) => m,
            None => return false,
        };
        if let Ok(mut res) = self.analyzer_result.lock() {
            *res = None;
        }
        self.analyzer_cancel_flag.store(false, Ordering::SeqCst);

        let mut state_clone = self.game_state.clone_light();
        let result_arc = self.analyzer_result.clone();
        let cancel_arc = self.analyzer_cancel_flag.clone();
        let count = count.max(0) as usize;

        self.analyzer_thread = Some(thread::spawn(move || {
            let moves = state_clone.generate_legal_moves();
            if moves.is_empty() {
                if let Ok(mut res) = result_arc.lock() {
                    *res = Some(Vec::new());
                }
                return;
            }

            let indices = GameState::get_legal_move_indices(&moves);
            let mut tensor_buffer = vec![0.0; TENSOR_SIZE];
            board_to_tensor_slice(&state_clone, &mut tensor_buffer);

            if cancel_arc.load(Ordering::SeqCst) {
                return;
            }

            let shape = vec![1, TENSOR_PLANES as i64, 14, 14];
            let input_value = match Value::from_array((shape, tensor_buffer)) {
                Ok(v) => v,
                Err(_) => return,
            };
            let mut inputs = HashMap::new();
            inputs.insert("input".to_string(), input_value);

            let mut model = match model_arc.lock() {
                Ok(g) => g,
                Err(_) => return,
            };
            let outputs = match model.run(inputs) {
                Ok(o) => o,
                Err(_) => return,
            };

            if cancel_arc.load(Ordering::SeqCst) {
                return;
            }

            let policy_data = match outputs.get("policy") {
                Some(v) => v,
                None => return,
            };
            let policy_slice = match policy_data.try_extract_tensor::<f32>() {
                Ok(d) => d.1,
                Err(_) => return,
            };

            let scored = scored_legal_moves(&moves, &indices, policy_slice);
            let limit = count.min(scored.len());
            let mut result_vec: Vec<(f32, i64, i64, i64, i64)> = Vec::new();
            for i in 0..limit {
                let (prob, m) = scored[i];
                result_vec.push((
                    prob,
                    m.from.0 as i64,
                    m.from.1 as i64,
                    m.to.0 as i64,
                    m.to.1 as i64,
                ));
            }

            if let Ok(mut res) = result_arc.lock() {
                *res = Some(result_vec);
            }
        }));
        true
    }

    #[func]
    fn is_analysis_done(&self) -> bool {
        if let Some(thread) = &self.analyzer_thread {
            return thread.is_finished();
        }
        false
    }

    #[func]
    fn get_analysis_result(&mut self) -> Variant {
        if let Some(thread) = self.analyzer_thread.take() {
            let _ = thread.join();
        }
        if self.analyzer_cancel_flag.load(Ordering::SeqCst) {
            return Variant::from(Array::<Variant>::new());
        }
        if let Ok(mut res) = self.analyzer_result.lock() {
            if let Some(moves) = res.take() {
                let mut arr = Array::<Variant>::new();
                for (score, fx, fy, tx, ty) in moves {
                    let mut dict = Dictionary::<Variant, Variant>::new();
                    dict.set(&Variant::from("fx"), &Variant::from(fx));
                    dict.set(&Variant::from("fy"), &Variant::from(fy));
                    dict.set(&Variant::from("tx"), &Variant::from(tx));
                    dict.set(&Variant::from("ty"), &Variant::from(ty));
                    dict.set(&Variant::from("score"), &Variant::from(score));
                    arr.push(&Variant::from(dict));
                }
                return Variant::from(arr);
            }
        }
        Variant::from(Array::<Variant>::new())
    }

    // ============================================================
    // Cognitive visualization / AI thoughts
    // ============================================================
    #[func]
    fn get_cognitive_visualization(&mut self) -> Variant {
        let mut dict = Dictionary::<Variant, Variant>::new();
        let state = &self.game_state;
        let my_color = state.turn;

        let mut threats = Array::<Variant>::new();
        let mut targets = Array::<Variant>::new();
        let mut develop = Array::<Variant>::new();

        for y in 1..=14 {
            for x in 1..=14 {
                if !is_valid_square(x as i32, y as i32) {
                    continue;
                }
                let idx = (y - 1) * 14 + (x - 1);
                let p_val = state.board[idx];
                if p_val == 0 {
                    continue;
                }
                let p_color = piece_color(p_val).unwrap();
                let p_type = piece_type(p_val).unwrap();

                if p_color == my_color {
                    if is_piece_hanging(
                        &state.board,
                        &state.dead_colors,
                        &state.resigned,
                        x as i32,
                        y as i32,
                        my_color,
                    ) {
                        let mut c = Dictionary::<Variant, Variant>::new();
                        c.set(&Variant::from("x"), &Variant::from(x as i64));
                        c.set(&Variant::from("y"), &Variant::from(y as i64));
                        threats.push(&Variant::from(c));
                    }
                    let is_start = match my_color {
                        Color::Red => y == 1,
                        Color::Yellow => y == 14,
                        Color::Blue => x == 1,
                        Color::Green => x == 14,
                    };
                    if is_start && (p_type == PieceType::Knight || p_type == PieceType::Bishop) {
                        let mut c = Dictionary::<Variant, Variant>::new();
                        c.set(&Variant::from("x"), &Variant::from(x as i64));
                        c.set(&Variant::from("y"), &Variant::from(y as i64));
                        develop.push(&Variant::from(c));
                    }
                } else {
                    if is_piece_hanging(
                        &state.board,
                        &state.dead_colors,
                        &state.resigned,
                        x as i32,
                        y as i32,
                        p_color,
                    ) {
                        let mut c = Dictionary::<Variant, Variant>::new();
                        c.set(&Variant::from("x"), &Variant::from(x as i64));
                        c.set(&Variant::from("y"), &Variant::from(y as i64));
                        targets.push(&Variant::from(c));
                    }
                }
            }
        }

        dict.set(&Variant::from("threats"), &Variant::from(threats));
        dict.set(&Variant::from("targets"), &Variant::from(targets));
        dict.set(&Variant::from("develop"), &Variant::from(develop));
        Variant::from(dict)
    }

    #[func]
    fn get_ai_thoughts(&mut self) -> Variant {
        let mut dict = Dictionary::<Variant, Variant>::new();
        let current = self.game_state.current_move_idx;

        if current == 0 {
            dict.set(&Variant::from("text"), &Variant::from("Нет хода для анализа."));
            dict.set(&Variant::from("arrows"), &Variant::from(Array::<Variant>::new()));
            return Variant::from(dict);
        }

        let k1 = current.saturating_sub(1);
        let k2 = current.saturating_sub(2);
        let found = if let Some(v) = self.bot_intents.get(&k1) {
            Some(v)
        } else {
            self.bot_intents.get(&k2)
        };

        if let Some((text, arrows_vec)) = found {
            let mut arr = Array::<Variant>::new();
            for (fx, fy, tx, ty, typ) in arrows_vec {
                let mut p = Dictionary::<Variant, Variant>::new();
                p.set(&Variant::from("fx"), &Variant::from(*fx as i64));
                p.set(&Variant::from("fy"), &Variant::from(*fy as i64));
                p.set(&Variant::from("tx"), &Variant::from(*tx as i64));
                p.set(&Variant::from("ty"), &Variant::from(*ty as i64));
                p.set(&Variant::from("type"), &Variant::from(typ.clone()));
                arr.push(&Variant::from(p));
            }
            dict.set(&Variant::from("text"), &Variant::from(text.clone()));
            dict.set(&Variant::from("arrows"), &Variant::from(arr));
        } else {
            dict.set(
                &Variant::from("text"),
                &Variant::from("Ход игрока. Намерение ИИ неизвестно."),
            );
            dict.set(&Variant::from("arrows"), &Variant::from(Array::<Variant>::new()));
        }
        Variant::from(dict)
    }

    // ============================================================
    // Game state API
    // ============================================================
    #[func]
    fn get_current_turn_code(&self) -> i32 {
        match self.game_state.turn {
            Color::Red => 1,
            Color::Blue => 2,
            Color::Yellow => 3,
            Color::Green => 4,
        }
    }

    #[func]
    fn get_dead_colors(&self) -> PackedByteArray {
        let mut arr = PackedByteArray::new();
        arr.resize(4);
        for i in 0..4 {
            arr[i] = if self.game_state.dead_colors[i] { 1 } else { 0 };
        }
        arr
    }

    #[func]
    fn get_resigned_colors(&self) -> PackedByteArray {
        let mut arr = PackedByteArray::new();
        arr.resize(4);
        for i in 0..4 {
            arr[i] = if self.game_state.resigned[i] { 1 } else { 0 };
        }
        arr
    }

    #[func]
    fn get_board_data(&self) -> Variant {
        let mut outer = Array::<Variant>::new();
        for y in 1..=14 {
            let mut inner = Array::<Variant>::new();
            for x in 1..=14 {
                let p_val = self.game_state.board[(y - 1) * 14 + (x - 1)];
                if p_val != 0 {
                    let p_color = piece_color(p_val).unwrap();
                    let p_type = piece_type(p_val).unwrap();
                    let color_code: i64 = match p_color {
                        Color::Red => 1,
                        Color::Blue => 2,
                        Color::Yellow => 3,
                        Color::Green => 4,
                    };
                    let piece_code: i64 = match p_type {
                        PieceType::Pawn => 1,
                        PieceType::Knight => 2,
                        PieceType::Bishop => 3,
                        PieceType::Rook => 4,
                        PieceType::Queen => 5,
                        PieceType::King => 6,
                    };
                    let mut code = color_code * 10 + piece_code;
                    if is_promoted(p_val) {
                        code |= 0x80;
                    }
                    inner.push(&Variant::from(code));
                } else {
                    inner.push(&Variant::from(0));
                }
            }
            outer.push(&Variant::from(inner));
        }
        Variant::from(outer)
    }

    #[func]
    fn get_legal_moves(&mut self) -> Variant {
        let moves = self.game_state.generate_legal_moves();
        let mut arr = Array::<Variant>::new();
        for m in moves {
            let mut dict = Dictionary::<Variant, Variant>::new();
            dict.set(&Variant::from("from_x"), &Variant::from(m.from.0 as i64));
            dict.set(&Variant::from("from_y"), &Variant::from(m.from.1 as i64));
            dict.set(&Variant::from("to_x"), &Variant::from(m.to.0 as i64));
            dict.set(&Variant::from("to_y"), &Variant::from(m.to.1 as i64));
            arr.push(&Variant::from(dict));
        }
        Variant::from(arr)
    }

    #[func]
    fn make_move(&mut self, from_x: i32, from_y: i32, to_x: i32, to_y: i32) -> bool {
        self.game_state.make_move(
            (from_x as usize, from_y as usize),
            (to_x as usize, to_y as usize),
            true,
        )
    }

    #[func]
    fn load_pgn(&mut self, pgn: String) -> bool {
        self.game_state.load_pgn(&pgn).is_ok()
    }

    #[func]
    fn go_to_move(&mut self, move_idx: i32) -> bool {
        self.game_state.go_to_move(move_idx as usize)
    }

    #[func]
    fn is_game_over(&self) -> bool {
        self.game_state.is_game_over()
    }

    #[func]
    fn is_turn_resigned(&self) -> bool {
        let c_idx = GameState::color_index(self.game_state.turn);
        self.game_state.resigned[c_idx] && !self.game_state.dead_colors[c_idx]
    }

    #[func]
    fn resign_player(&mut self, color_code: i32) {
        let c = match color_code {
            1 => Color::Red,
            2 => Color::Blue,
            3 => Color::Yellow,
            4 => Color::Green,
            _ => return,
        };
        self.game_state.resign_player(c);
    }

    #[func]
    fn play_dead_king_move(&mut self) -> bool {
        self.game_state.play_random_dead_king_move()
    }

    #[func]
    fn get_move_list(&self) -> Variant {
        let mut arr = Array::<Variant>::new();
        for m in &self.game_state.move_history {
            let mut dict = Dictionary::<Variant, Variant>::new();
            if m.skipped {
                dict.set(&Variant::from("notation"), &Variant::from(""));
                arr.push(&Variant::from(dict));
                continue;
            }
            dict.set(&Variant::from("from_x"), &Variant::from(m.from.0 as i64));
            dict.set(&Variant::from("from_y"), &Variant::from(m.from.1 as i64));
            dict.set(&Variant::from("to_x"), &Variant::from(m.to.0 as i64));
            dict.set(&Variant::from("to_y"), &Variant::from(m.to.1 as i64));

            let mut notation = if m.castle == 1 {
                "O-O".to_string()
            } else if m.castle == 2 {
                "O-O-O".to_string()
            } else {
                let from_str = Self::coord_to_str(m.from.0, m.from.1);
                let to_str = Self::coord_to_str(m.to.0, m.to.1);
                let dash = if m.captured.is_some() { "x" } else { "-" };
                format!("{}{}{}", from_str, dash, to_str)
            };
            if let Some(prom) = m.promotion {
                let prom_char = match prom {
                    PieceType::Queen => 'Q',
                    PieceType::Rook => 'R',
                    PieceType::Bishop => 'B',
                    PieceType::Knight => 'N',
                    _ => 'Q',
                };
                notation.push_str(&format!("={}", prom_char));
            }
            if m.checkmate {
                notation.push('#');
            } else if m.check {
                notation.push('+');
            }
            dict.set(&Variant::from("notation"), &Variant::from(notation));
            arr.push(&Variant::from(dict));
        }
        Variant::from(arr)
    }

    #[func]
    fn get_last_moves_per_color(&self) -> Variant {
        let mut arr = Array::<Variant>::new();
        let mut found = [false; 4];
        let start_idx = self.game_state.current_move_idx;
        if start_idx == 0 {
            return Variant::from(arr);
        }
        for i in (0..start_idx).rev() {
            let m = &self.game_state.move_history[i];
            if m.skipped {
                continue;
            }
            let c_idx = GameState::color_index(m.color);
            if !found[c_idx] {
                found[c_idx] = true;
                let mut dict = Dictionary::<Variant, Variant>::new();
                dict.set(&Variant::from("from_x"), &Variant::from(m.from.0 as i64));
                dict.set(&Variant::from("from_y"), &Variant::from(m.from.1 as i64));
                dict.set(&Variant::from("to_x"), &Variant::from(m.to.0 as i64));
                dict.set(&Variant::from("to_y"), &Variant::from(m.to.1 as i64));
                dict.set(&Variant::from("color"), &Variant::from(c_idx as i64 + 1));
                arr.push(&Variant::from(dict));
                if found.iter().all(|&x| x) {
                    break;
                }
            }
        }
        Variant::from(arr)
    }

    #[func]
    fn get_scores(&self) -> PackedInt32Array {
        let mut arr = PackedInt32Array::new();
        for &s in &self.game_state.scores {
            arr.push(s);
        }
        arr
    }

    #[func]
    fn get_eval_points(&self) -> PackedFloat32Array {
        let mut arr = PackedFloat32Array::new();
        for &s in &self.game_state.scores {
            arr.push(s as f32);
        }
        arr
    }

    #[func]
    fn cancel_search(&mut self) {
        self.bot_cancel_flag.store(true, Ordering::SeqCst);
        self.analyzer_cancel_flag.store(true, Ordering::SeqCst);
    }
}

// ============================================================
// Free helpers
// ============================================================
fn run_model_batch(
    model: &mut Session,
    states: &[GameState],
    batch_size: usize,
) -> Option<(Vec<Vec<f32>>, Vec<[f32; 4]>)> {
    if batch_size == 0 {
        return Some((Vec::new(), Vec::new()));
    }
    let mut flat_tensor = vec![0.0f32; batch_size * TENSOR_SIZE];
    for i in 0..batch_size {
        let start = i * TENSOR_SIZE;
        let end = start + TENSOR_SIZE;
        board_to_tensor_slice(&states[i], &mut flat_tensor[start..end]);
    }

    let shape = vec![batch_size as i64, TENSOR_PLANES as i64, 14, 14];
    let input_value = Value::from_array((shape, flat_tensor)).ok()?;
    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), input_value);

    let outputs = model.run(inputs).ok()?;

    let policy_data = outputs.get("policy")?.try_extract_tensor::<f32>().ok()?;
    let policy_slice = policy_data.1;
    if policy_slice.len() < batch_size * POLICY_SIZE {
        return None;
    }
    let mut policies = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let start = i * POLICY_SIZE;
        let end = start + POLICY_SIZE;
        policies.push(policy_slice[start..end].to_vec());
    }

    let value_data = outputs.get("value")?.try_extract_tensor::<f32>().ok()?;
    let values = expected_values_from_slice(value_data.1, batch_size)?;

    Some((policies, values))
}

fn expected_values_from_slice(value_slice: &[f32], batch_size: usize) -> Option<Vec<[f32; 4]>> {
    if value_slice.len() < batch_size * VALUE_SIZE {
        return None;
    }
    let placement_centers = [1.0f32, 0.33, -0.33, -1.0];
    let mut values = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let mut expected_vals = [0.0f32; 4];
        for c in 0..4 {
            let off = i * VALUE_SIZE + c * 4;
            let logits = &value_slice[off..off + 4];
            let max_logit = logits.iter().cloned().fold(f32::MIN, f32::max);
            let mut exps = [0.0f32; 4];
            let mut sum_exp = 0.0;
            for j in 0..4 {
                exps[j] = (logits[j] - max_logit).exp();
                sum_exp += exps[j];
            }
            if sum_exp <= 0.0 || !sum_exp.is_finite() {
                expected_vals[c] = 0.0;
            } else {
                let mut expected_val = 0.0;
                for j in 0..4 {
                    expected_val += (exps[j] / sum_exp) * placement_centers[j];
                }
                expected_vals[c] = expected_val;
            }
        }
        values.push(expected_vals);
    }
    Some(values)
}

fn softmax_priors(policy: &[f32], indices: &[usize]) -> Vec<f32> {
    if indices.is_empty() {
        return Vec::new();
    }
    let logits: Vec<f32> = indices
        .iter()
        .map(|&idx| {
            if idx < policy.len() {
                policy[idx]
            } else {
                0.0
            }
        })
        .collect();
    let max_logit = logits.iter().cloned().fold(f32::MIN, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        vec![1.0 / indices.len() as f32; indices.len()]
    } else {
        exps.iter().map(|e| e / sum).collect()
    }
}

fn scored_legal_moves(moves: &[Move], indices: &[usize], policy: &[f32]) -> Vec<(f32, Move)> {
    let priors = softmax_priors(policy, indices);
    let mut scored: Vec<(f32, Move)> = moves
        .iter()
        .enumerate()
        .map(|(i, m)| (priors[i], *m))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored
}

fn is_piece_hanging(
    board: &Board,
    dead_colors: &[bool; 4],
    resigned: &[bool; 4],
    x: i32,
    y: i32,
    color: Color,
) -> bool {
    if !is_valid_square(x, y) {
        return false;
    }
    let is_attacked = [Color::Red, Color::Blue, Color::Yellow, Color::Green]
        .iter()
        .any(|&opp| {
            opp != color
                && !dead_colors[GameState::color_index(opp)]
                && GameState::is_square_attacked(board, dead_colors, resigned, x, y, opp)
        });
    if !is_attacked {
        return false;
    }
    let is_defended = GameState::is_square_attacked(board, dead_colors, resigned, x, y, color);
    !is_defended
}

struct MyExtension;

#[gdextension]
unsafe impl ExtensionLibrary for MyExtension {}