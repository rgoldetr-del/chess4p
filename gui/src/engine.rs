//! engine.rs — нейросетевой движок бота: ONNX-инференс + батчевый MCTS.
//!
//! Портировано из удалённого godot-extension (ChessAnalyzer),
//! адаптировано под нативное окно без зависимости от Godot.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use chess4p_game::{board_to_tensor_slice, GameState, Move, TENSOR_PLANES, TENSOR_SIZE};

use ort::ep::DirectML;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::session::builder::SessionBuilder;

const POLICY_SIZE: usize = 38416; // 196 * 196
const VALUE_SIZE: usize = 16; // 4 * 4
const INF_PLANES: usize = 100; // возможный вариант архитектуры ChessNet100
const INFER_BATCH: usize = 16; // фиксированный batch ONNX-модели (требование DirectML)

// ============================================================
// Публичные типы
// ============================================================

#[derive(Clone, Debug)]
pub struct EngineResult {
    pub best_move: Move,
    pub pv: Vec<Move>,
    pub root_value: f32,
    pub visits: u32,
    pub nodes: usize,
    pub elapsed: Duration,
}

pub struct EngineSearch {
    pub state: GameState,
    pub simulations: u32,
}

/// Загруженный движок: ONNX-сессия + фоновый поток поиска.
pub struct Engine {
    session: Option<Arc<Mutex<Session>>>,
    planes: usize,
    receiver: Option<mpsc::Receiver<EngineResult>>,
    handle: Option<thread::JoinHandle<()>>,
    cancel: Arc<AtomicBool>,
    pub model_name: String,
}

impl Engine {
    pub fn new() -> Self {
        Self {
            session: None,
            planes: TENSOR_PLANES,
            receiver: None,
            handle: None,
            cancel: Arc::new(AtomicBool::new(false)),
            model_name: String::new(),
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.session.is_some()
    }

    /// Загрузка ONNX-модели. Архитектура (100/256 плоскостей) определяется
    /// пробным прогоном на CPU; затем пытаемся включить DirectML, при неудаче
    /// остаёмся на CPU.
    pub fn load(&mut self, path: &str) -> Result<String, String> {
        // DirectML.dll должен быть доступен рядом с exe; и явно указываем,
        // какую onnxruntime.dll грузить (System32 идёт раньше PATH в поиске DLL).
        let ort_dir: Option<std::path::PathBuf> = {
            let mut found = None;
            if let Ok(cwd) = std::env::current_dir() {
                if cwd.join("ort_runtime/onnxruntime.dll").exists() {
                    found = Some(cwd.join("ort_runtime"));
                }
            }
            if found.is_none() {
                if let Ok(exe) = std::env::current_exe() {
                    if let Some(dir) = exe.parent() {
                        if dir.join("ort_runtime/onnxruntime.dll").exists() {
                            found = Some(dir.join("ort_runtime"));
                        }
                    }
                }
            }
            found
        };
        if let Some(ort_dir) = ort_dir {
            std::env::set_var(
                "ORT_DYLIB_PATH",
                ort_dir.join("onnxruntime.dll").to_string_lossy().to_string(),
            );
            // Предзагружаем ПАРУННЫЙ DirectML.dll до onnxruntime.dll —
            // иначе процесс может уже держать старый System32\DirectML.dll,
            // и ORT получит несовместимую версию (E_INVALIDARG).
            // (Точно так же делает python-пакет onnxruntime-directml.)
            let dml = ort_dir.join("DirectML.dll");
            if dml.exists() {
                // SAFETY: загрузка системной библиотеки по явному пути;
                // ошибки загрузки обрабатываются ниже.
                match unsafe { libloading::Library::new(&dml) } {
                    Ok(lib) => std::mem::forget(lib), // держим до конца процесса
                    Err(e) => {
                        return Err(format!("Не удалось предзагрузить {}: {e}", dml.display()))
                    }
                }
            }
            if let Ok(old_path) = std::env::var("PATH") {
                std::env::set_var(
                    "PATH",
                    format!("{};{}", ort_dir.display(), old_path),
                );
            }
        }

        let make_builder = |dml: bool| -> Result<SessionBuilder, String> {
            let b = Session::builder().map_err(|e| format!("builder: {e}"))?;
            let b = b
                .with_intra_threads(4)
                .map_err(|e| format!("threads: {e}"))?
                .with_optimization_level(GraphOptimizationLevel::Level1)
                .map_err(|e| format!("optimization: {e}"))?;
            if dml {
                let b = b
                    .with_execution_providers([DirectML::default().with_device_id(0).build()])
                    .map_err(|e| format!("execution providers: {e}"))?;
                Ok(b)
            } else {
                Ok(b)
            }
        };

        // 1. CPU-сессия: проверяем модель и определяем число плоскостей.
        let mut cpu_session = make_builder(false)?
            .commit_from_file(path)
            .map_err(|e| format!("Не удалось загрузить ONNX: {e}"))?;

        let mut detected: Option<usize> = None;
        for planes in [TENSOR_PLANES, INF_PLANES] {
            if self.probe(&mut cpu_session, planes).is_ok() {
                detected = Some(planes);
                break;
            }
        }
        let planes = detected
            .ok_or("Модель не приняла ни 256, ни 100 плоскостей — несовместимый экспорт")?;

        // 2. Пробуем DirectML; если GPU-сессия не работает — остаёмся на CPU.
        let mut used_dml = false;
        let mut dml_error = String::new();
        match make_builder(true)?.commit_from_file(path) {
            Ok(mut dml_session) => {
                if self.probe(&mut dml_session, planes).is_ok() {
                    self.session = Some(Arc::new(Mutex::new(dml_session)));
                    used_dml = true;
                } else if let Err(e) = self.probe(&mut dml_session, planes) {
                    dml_error = e;
                }
            }
            Err(e) => dml_error = e.to_string(),
        }
        if !used_dml {
            self.session = Some(Arc::new(Mutex::new(cpu_session)));
        }
        self.planes = planes;
        self.model_name = std::path::Path::new(path)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();

        let msg = format!(
            "Модель {} загружена ({} плоскостей, {}{})",
            self.model_name,
            planes,
            if used_dml { "DirectML (GPU)" } else { "CPU" },
            if used_dml {
                String::new()
            } else if dml_error.is_empty() {
                ", DirectML недоступен".to_string()
            } else {
                format!(", DirectML недоступен: {}", dml_error)
            }
        );
        Ok(msg)
    }

    fn probe(&self, session: &mut Session, planes: usize) -> Result<(), String> {
        let data = vec![0.0f32; INFER_BATCH * planes * 196];
        let value = ort::value::Tensor::from_array((
            vec![INFER_BATCH as i64, planes as i64, 14, 14],
            data,
        ))
        .map_err(|e| e.to_string())?;
        let inputs = ort::inputs!["input" => value];
        session.run(inputs).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Запуск поиска в фоне. Возвращает false, если занят или модель не загружена.
    pub fn start_search(&mut self, search: EngineSearch) -> bool {
        if self.session.is_none() || self.handle.as_ref().is_some_and(|h| !h.is_finished()) {
            return false;
        }
        self.cancel.store(false, Ordering::SeqCst);
        let session = self.session.clone().unwrap();
        let planes = self.planes;
        let cancel = self.cancel.clone();
        let (tx, rx) = mpsc::channel();

        self.handle = Some(
            thread::Builder::new()
                .name("mcts-search".into())
                .stack_size(128 * 1024 * 1024)
                .spawn(move || {
                    let result =
                        run_mcts(session, planes, &search.state, search.simulations, &cancel);
                    let _ = tx.send(result);
                })
                .expect("failed to spawn mcts thread"),
        );
        self.receiver = Some(rx);
        true
    }

    /// Забрать результат, если поиск завершён.
    pub fn poll(&mut self) -> Option<EngineResult> {
        let finished = self.handle.as_ref()?.is_finished();
        if !finished {
            return None;
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        let rx = self.receiver.take()?;
        rx.try_recv().ok()
    }

    pub fn cancel(&mut self) {
        self.cancel.store(true, Ordering::SeqCst);
    }
}

// ============================================================
// MCTS
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

    fn q(&self, perspective: usize) -> f32 {
        if self.visits == 0 {
            return -self.virtual_loss;
        }
        (self.total_value[perspective] - self.virtual_loss) / self.visits as f32
    }
}

fn select_child(nodes: &mut [MctsNode], parent_idx: usize, perspective: usize) -> usize {
    let c_puct = 1.5f32;
    let fpu = -0.5f32;
    let total_parent_visits = (nodes[parent_idx].visits as f32 + 1.0).sqrt();
    let mut best_score = f32::MIN;
    let mut best_child_idx = nodes[parent_idx].children[0];

    for &child_idx in &nodes[parent_idx].children {
        let q = if nodes[child_idx].visits == 0 {
            fpu
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

fn run_mcts(
    model: Arc<Mutex<Session>>,
    planes: usize,
    state: &GameState,
    max_simulations: u32,
    cancel: &AtomicBool,
) -> EngineResult {
    let start = Instant::now();
    let state_clone = state.clone_light();
    let root_perspective = state_clone.turn.index();

    let mut nodes: Vec<MctsNode> = Vec::with_capacity(8192);
    nodes.push(MctsNode::new(None, None, 1.0, state_clone.is_game_over()));

    let batch_size: usize = 16;
    let mut simulations_done: u32 = 0;
    let target: u32 = max_simulations.max(batch_size as u32);

    loop {
        if cancel.load(Ordering::SeqCst) {
            break;
        }
        if simulations_done >= target {
            break;
        }

        let mut pending_states: Vec<GameState> = Vec::with_capacity(batch_size);
        let mut pending_indices: Vec<usize> = Vec::with_capacity(batch_size);
        let mut pending_terminal_values: Vec<Option<[f32; 4]>> = Vec::with_capacity(batch_size);

        // 1. Selection
        for _ in 0..batch_size {
            let mut current_idx = 0;
            let mut sim_state = state_clone.clone_light();

            for _ in 0..512 {
                if nodes[current_idx].is_terminal || nodes[current_idx].children.is_empty() {
                    break;
                }
                let next_idx = select_child(&mut nodes, current_idx, sim_state.turn.index());
                current_idx = next_idx;
                if let Some(mv) = nodes[current_idx].move_made {
                    sim_state.make_move_unchecked(mv.from, mv.to, true, None);
                }
            }

            let terminal = if sim_state.is_game_over() {
                Some(sim_state.final_rewards())
            } else {
                None
            };
            pending_terminal_values.push(terminal);
            pending_states.push(sim_state);
            pending_indices.push(current_idx);
        }

        // 2. Отделяем терминальные состояния
        let mut non_term_states: Vec<GameState> = Vec::new();
        let mut non_term_orig_indices: Vec<usize> = Vec::new();
        for i in 0..batch_size {
            if pending_terminal_values[i].is_none() {
                non_term_states.push(pending_states[i].clone_light());
                non_term_orig_indices.push(i);
            }
        }

        // 3. Батчевый инференс
        let mut nn_policies: Vec<Vec<f32>> = Vec::new();
        let mut nn_values: Vec<[f32; 4]> = Vec::new();
        if !non_term_states.is_empty() {
            let mut guard = match model.lock() {
                Ok(g) => g,
                Err(_) => break,
            };
            match run_model_batch(&mut guard, planes, &non_term_states) {
                Some((policies, values)) => {
                    nn_policies = policies;
                    nn_values = values;
                }
                None => break,
            }
        }

        // 4. Expansion + backpropagation
        for i in 0..batch_size {
            let current_idx = pending_indices[i];
            let value: [f32; 4] = if let Some(val) = pending_terminal_values[i] {
                val
            } else {
                match non_term_orig_indices.iter().position(|&x| x == i) {
                    Some(nt_idx) => {
                        if nodes[current_idx].children.is_empty() {
                            let sim_state = &mut pending_states[i];
                            let legal_moves = sim_state.generate_legal_moves();
                            if !legal_moves.is_empty() {
                                let indices = GameState::get_legal_move_indices(&legal_moves);
                                let priors = softmax_priors(&nn_policies[nt_idx], &indices);
                                for (mv_idx, mv) in legal_moves.iter().enumerate() {
                                    let mut next_state = sim_state.clone_light();
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
                    }
                    None => [0.0f32; 4],
                }
            };

            nodes[current_idx].visits += 1;
            nodes[current_idx].virtual_loss = (nodes[current_idx].virtual_loss - 1.0).max(0.0);
            for j in 0..4 {
                nodes[current_idx].total_value[j] += value[j];
            }

            let mut p_idx = current_idx;
            while let Some(parent) = nodes[p_idx].parent {
                nodes[parent].visits += 1;
                nodes[parent].virtual_loss = (nodes[parent].virtual_loss - 1.0).max(0.0);
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

    // 5. Выбор лучшего хода по визитам
    let root_children = nodes[0].children.clone();
    if root_children.is_empty() {
        // Ходов нет — передаём "ничего" через фиктивный результат с нулевыми визитами
        return EngineResult {
            best_move: Move {
                from: (0, 0),
                to: (0, 0),
                piece: chess4p_game::PieceType::King,
                color: state.turn,
                captured: None,
                promotion: None,
                castle: 0,
                en_passant: false,
                check: false,
                checkmate: false,
                skipped: true,
            },
            pv: Vec::new(),
            root_value: 0.0,
            visits: 0,
            nodes: nodes.len(),
            elapsed: start.elapsed(),
        };
    }

    let mut visits_by_move: HashMap<(usize, usize, usize, usize), u32> = HashMap::new();
    for &child_idx in &root_children {
        if let Some(mv) = nodes[child_idx].move_made {
            *visits_by_move
                .entry((mv.from.0, mv.from.1, mv.to.0, mv.to.1))
                .or_insert(0) += nodes[child_idx].visits;
        }
    }

    let best_key = *visits_by_move
        .iter()
        .max_by_key(|(_, v)| **v)
        .map(|(k, _)| k)
        .unwrap();

    let mut best_child_idx = root_children[0];
    let mut best_visits = 0;
    for &child_idx in &root_children {
        if let Some(mv) = nodes[child_idx].move_made {
            if (mv.from.0, mv.from.1, mv.to.0, mv.to.1) == best_key {
                let vis = nodes[child_idx].visits;
                if vis > best_visits {
                    best_visits = vis;
                    best_child_idx = child_idx;
                }
            }
        }
    }

    // 6. PV-линия
    let mut pv: Vec<Move> = Vec::new();
    let mut current = best_child_idx;
    for _ in 0..5 {
        if let Some(mv) = nodes[current].move_made {
            pv.push(mv);
        } else {
            break;
        }
        if nodes[current].children.is_empty() {
            break;
        }
        let mut next_best = nodes[current].children[0];
        let mut next_visits = 0;
        for &child_idx in &nodes[current].children {
            let vis = nodes[child_idx].visits;
            if vis > next_visits {
                next_visits = vis;
                next_best = child_idx;
            }
        }
        if next_visits == 0 {
            break;
        }
        current = next_best;
    }

    EngineResult {
        best_move: nodes[best_child_idx].move_made.unwrap(),
        pv,
        root_value: nodes[best_child_idx].q(root_perspective),
        visits: visits_by_move.values().sum(),
        nodes: nodes.len(),
        elapsed: start.elapsed(),
    }
}

// ============================================================
// Инференс
// ============================================================

fn run_model_batch(
    model: &mut Session,
    planes: usize,
    states: &[GameState],
) -> Option<(Vec<Vec<f32>>, Vec<[f32; 4]>)> {
    if states.is_empty() {
        return Some((Vec::new(), Vec::new()));
    }

    // Модель экспортирована с фиксированным batch (INFER_BATCH): дополняем
    // неполный батч копиями первого состояния (лишние выходы игнорируются).
    let batch = INFER_BATCH;

    // board_to_tensor_slice заполняет буфер 256 плоскостей; модель может
    // принимать подмножество (100), поэтому копируем нужный префикс.
    let mut full = vec![0.0f32; batch * TENSOR_SIZE];
    for i in 0..batch {
        let st = &states[i.min(states.len() - 1)];
        board_to_tensor_slice(st, &mut full[i * TENSOR_SIZE..(i + 1) * TENSOR_SIZE]);
    }
    let used = planes * 196;
    let mut flat = vec![0.0f32; batch * used];
    for i in 0..batch {
        flat[i * used..(i + 1) * used]
            .copy_from_slice(&full[i * TENSOR_SIZE..i * TENSOR_SIZE + used]);
    }

    let value = ort::value::Tensor::from_array((
        vec![batch as i64, planes as i64, 14, 14],
        flat,
    ))
    .ok()?;
    let inputs = ort::inputs!["input" => value];

    let outputs = model.run(inputs).ok()?;
    let policy_out = outputs["policy"].try_extract_tensor::<f32>().ok()?;
    let policy_slice: Vec<f32> = policy_out.1.to_vec();
    if policy_slice.len() < batch * POLICY_SIZE {
        return None;
    }

    let value_out = outputs["value"].try_extract_tensor::<f32>().ok()?;
    let value_slice: Vec<f32> = value_out.1.to_vec();

    let mut policies = Vec::with_capacity(batch);
    for i in 0..batch {
        policies.push(policy_slice[i * POLICY_SIZE..(i + 1) * POLICY_SIZE].to_vec());
    }
    let values = expected_values_from_slice(&value_slice, batch)?;

    Some((policies, values))
}

fn expected_values_from_slice(value_slice: &[f32], batch: usize) -> Option<Vec<[f32; 4]>> {
    if value_slice.len() < batch * VALUE_SIZE {
        return None;
    }
    let placement_centers = [1.0f32, 0.33, -0.33, -1.0];
    let mut values = Vec::with_capacity(batch);

    for i in 0..batch {
        let mut expected_vals = [0.0f32; 4];
        for c in 0..4 {
            let off = i * VALUE_SIZE + c * 4;
            let logits = &value_slice[off..off + 4];
            let max_logit = logits.iter().cloned().fold(f32::MIN, f32::max);
            let mut exps = [0.0f32; 4];
            let mut sum_exp = 0.0f32;
            for j in 0..4 {
                exps[j] = (logits[j] - max_logit).exp();
                sum_exp += exps[j];
            }
            if sum_exp <= 0.0 || !sum_exp.is_finite() {
                expected_vals[c] = 0.0;
            } else {
                let mut expected_val = 0.0f32;
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
        .map(|&idx| if idx < policy.len() { policy[idx] } else { 0.0 })
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
