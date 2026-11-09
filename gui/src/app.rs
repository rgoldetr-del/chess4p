//! app.rs — главное окно Chess4P: игра с ботами, история, скрипты, настройки.

use std::path::PathBuf;
use std::sync::Arc;

use egui::{Color32, ScrollArea, RichText};

use chess4p_game::{Color, GameState, Move};

use crate::board::{self, BoardResponse, PLAYER_COLORS, PLAYER_NAMES};
use crate::engine::{Engine, EngineSearch};
use crate::scripts::ScriptRunner;
use crate::sound::{SoundEvent, SoundPlayer};

// ============================================================
// Настройки (персистятся в gui_settings.json)
// ============================================================

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct Settings {
    model_path: String,
    python_path: String,
    bot_dir: String,
    #[serde(default)]
    board_zoom: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            python_path: "python".to_string(),
            bot_dir: std::env::current_dir()
                .map(|d| d.join("bot").to_string_lossy().to_string())
                .unwrap_or_else(|_| "bot".to_string()),
            board_zoom: 1.0,
        }
    }
}

fn settings_path() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("gui_settings.json")))
        .unwrap_or_else(|| PathBuf::from("gui_settings.json"))
}

fn load_settings() -> Settings {
    std::fs::read_to_string(settings_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

// ============================================================
// Места игроков
// ============================================================

#[derive(Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
enum SeatKind {
    Human,
    Bot,
}

#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
struct Seat {
    kind: SeatKind,
    simulations: u32,
}

impl Default for Seat {
    fn default() -> Self {
        Seat { kind: SeatKind::Human, simulations: 400 }
    }
}

// ============================================================
// Приложение
// ============================================================

pub struct Chess4pApp {
    state: GameState,
    seats: [Seat; 4],
    engine: Engine,
    /// Для какого цвета идёт поиск
    pending_for: Option<usize>,
    selected: Option<(usize, usize)>,
    legal_cache: Option<Vec<Move>>,
    last_result: Option<crate::engine::EngineResult>,
    status: String,
    error: Option<String>,
    tab: Tab,
    review_idx: Option<usize>,
    settings: Settings,
    runner: ScriptRunner,
    font_ok: bool,
    piece_textures: Option<board::PieceTextures>,
    sounds: SoundPlayer,
    board_zoom: f32,
    game_over_played: bool,
    marks: board::BoardMarks,
    bot_mode: String,
    bot_sims: u32,
    calib_cache: Option<u64>,
    calib_status: String,
}

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Game,
    Scripts,
    Settings,
}

impl Chess4pApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Подключаем шрифт с шахматными символами (Segoe UI Symbol есть на любой Windows)
        let font_ok = install_piece_font(cc);

        let mut settings = load_settings();
        // Поиск папки bot/: заданная в настройках -> рядом с exe -> рядом с exe на уровень выше -> C:\chess4p\bot
        {
            let exe_dir = std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.to_path_buf()));
            let mut candidates: Vec<std::path::PathBuf> = Vec::new();
            if !settings.bot_dir.is_empty() {
                candidates.push(std::path::PathBuf::from(&settings.bot_dir));
            }
            if let Ok(cwd) = std::env::current_dir() {
                candidates.push(cwd.join("bot"));
            }
            if let Some(d) = &exe_dir {
                candidates.push(d.join("bot"));
                if let Some(parent) = d.parent() {
                    candidates.push(parent.join("bot"));
                }
            }
            candidates.push(std::path::PathBuf::from(r"C:\chess4p\bot"));

            if let Some(found) = candidates
                .iter()
                .find(|p| p.join("config.py").exists() || p.join("calibration.py").exists())
            {
                settings.bot_dir = found.to_string_lossy().to_string();
            } else if settings.bot_dir.is_empty() {
                if let Ok(cwd) = std::env::current_dir() {
                    settings.bot_dir = cwd.join("bot").to_string_lossy().to_string();
                }
            }
        }

        // Автопоиск model.onnx рядом
        if settings.model_path.is_empty() {
            for cand in ["model.onnx", "bot/model.onnx", "../model.onnx"] {
                if std::path::Path::new(cand).exists() {
                    settings.model_path = cand.to_string();
                    break;
                }
            }
        }
        app_log(&format!("model_path = {:?}", settings.model_path));

        let board_zoom = { let z = settings.board_zoom; if z > 0.0 { z } else { 1.0 } };

        // Спрайты фигур: assets/pieces рядом с exe или в текущей папке
        let piece_textures = {
            let mut dir = None;
            if std::path::Path::new("assets/pieces").exists() {
                dir = Some(std::path::PathBuf::from("assets/pieces"));
            } else if let Ok(exe) = std::env::current_exe() {
                if let Some(d) = exe.parent() {
                    let p = d.join("assets/pieces");
                    if p.exists() {
                        dir = Some(p);
                    }
                }
            }
            dir.and_then(|d| board::PieceTextures::load(&cc.egui_ctx, &d))
        };

        let mut app = Self {
            state: GameState::new(),
            seats: Default::default(),
            engine: Engine::new(),
            pending_for: None,
            selected: None,
            legal_cache: None,
            last_result: None,
            status: "Готово".to_string(),
            error: None,
            tab: Tab::Game,
            review_idx: None,
            settings,
            runner: ScriptRunner::default(),
            font_ok,
            piece_textures,
            sounds: SoundPlayer::new(&{
                // assets/sounds рядом с exe или в текущей папке
                let mut dir = std::path::PathBuf::from("assets/sounds");
                if !dir.exists() {
                    if let Ok(exe) = std::env::current_exe() {
                        let d = exe.parent().unwrap().join("assets/sounds");
                        if d.exists() {
                            dir = d;
                        }
                    }
                }
                dir
            }),
            board_zoom,
            game_over_played: false,
            marks: board::BoardMarks::default(),
            bot_mode: "ever".to_string(),
            bot_sims: 400,
            calib_cache: None,
            calib_status: String::new(),
        };

        // Диагностический режим: CHESS4P_GUI_AUTOBOT=1 — все 4 места боты
        if std::env::var("CHESS4P_GUI_AUTOBOT").as_deref() == Ok("1") {
            for s in &mut app.seats {
                s.kind = SeatKind::Bot;
                s.simulations = 32;
            }
            app.log_line("режим AUTOBOT включён (32 симуляции)");
        }

        if !app.settings.model_path.is_empty() {
            app_log("загрузка модели...");
            match app.engine.load(&app.settings.model_path.clone()) {
                Ok(msg) => {
                    app.status = msg.clone();
                    app_log(&msg);
                }
                Err(e) => {
                    app.error = Some(e.clone());
                    app_log(&format!("ОШИБКА ЗАГРУЗКИ: {e}"));
                }
            }
        } else {
            app.status = "Модель не загружена — укажите model.onnx во вкладке «Настройки»".into();
        }

        app
    }

    fn save_settings(&self) {
        if let Ok(json) = serde_json::to_string_pretty(&self.settings) {
            let _ = std::fs::write(settings_path(), json);
        }
    }

    /// Диагностический лог (включается env CHESS4P_GUI_LOG=1)
    fn log_line(&self, line: &str) {
        if std::env::var("CHESS4P_GUI_LOG").as_deref() != Ok("1") {
            return;
        }
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("gui_debug.log")
        {
            let _ = writeln!(f, "{line}");
        }
    }

    // ========================================================
    // Игровая логика
    // ========================================================

    fn is_reviewing(&self) -> bool {
        self.review_idx.is_some()
    }

    fn current_legal_moves(&mut self) -> &[Move] {
        if self.legal_cache.is_none() {
            self.legal_cache = Some(self.state.generate_legal_moves());
        }
        self.legal_cache.as_deref().unwrap()
    }

    fn invalidate(&mut self) {
        self.legal_cache = None;
        self.selected = None;
    }

    /// Обработка автоходов: мёртвые/сдавшиеся пропускаются автоматически.
    fn auto_dead(&mut self) {
        let c = self.state.turn.index();
        if self.state.dead_colors[c] || self.state.resigned[c] {
            if self.state.auto_play_dead_fast() {
                self.invalidate();
            }
        }
    }

    /// Автопилот ботов. Возвращает true, если ход применён.
    fn bot_autopilot(&mut self) {
        if self.is_reviewing() || self.state.is_game_over() {
            return;
        }
        self.auto_dead();
        if self.state.is_game_over() {
            return;
        }

        let turn = self.state.turn.index();
        if self.seats[turn].kind != SeatKind::Bot {
            return;
        }

        if !self.engine.is_loaded() {
            return;
        }

        if self.pending_for.is_none() {
            let sims = self.seats[turn].simulations;
            let state_copy = self.state.clone_light();
            if self.engine.start_search(EngineSearch { state: state_copy, simulations: sims }) {
                self.pending_for = Some(turn);
                self.status = format!("🤖 {} думает ({} симуляций)...", PLAYER_NAMES[turn], sims);
            }
        } else if self.pending_for == Some(turn) {
            if let Some(result) = self.engine.poll() {
                self.pending_for = None;
                self.last_result = Some(result.clone());
                let mv = result.best_move;
                if mv.skipped {
                    self.status = "🤖 у бота нет ходов".into();
                    self.log_line("бот: нет ходов (skip)");
                } else {
                    // make_move(false): без строгой проверки, но С записью истории
                    let ok = self.state.make_move(mv.from, mv.to, false);
                    if ok {
                        self.invalidate();
                        self.status = format!(
                            "🤖 {}: {} → {} | visits {}, eval {:.2}, {:.2?}",
                            PLAYER_NAMES[turn],
                            coord_str(mv.from),
                            coord_str(mv.to),
                            result.visits,
                            result.root_value,
                            result.elapsed,
                        );
                        self.log_line(&format!(
                            "{} | ходов в истории: {}",
                            self.status,
                            self.state.move_history.len()
                        ));
                        self.sounds.play(sound_for_move(&mv));
                        self.on_move_finished();
                    }
                }
            }
        }
    }

    fn human_click(&mut self, sq: (usize, usize)) {
        if self.is_reviewing() || self.state.is_game_over() {
            return;
        }
        self.auto_dead();
        let turn = self.state.turn.index();
        if self.seats[turn].kind != SeatKind::Human {
            return;
        }

        let (x, y) = sq;
        let idx = chess4p_game::sq_index(x, y);

        if let Some(sel) = self.selected {
            // Попытка хода
            let mv_opt = {
                let legal = self.current_legal_moves();
                legal
                    .iter()
                    .find(|m| m.from == sel && m.to == sq)
                    .cloned()
            };
            if let Some(mv) = mv_opt {
                let ok = self.state.make_move(sel, sq, true);
                if ok {
                    self.invalidate();
                    self.status = format!("Ход {}: {} → {}", PLAYER_NAMES[turn], coord_str(sel), coord_str(sq));
                    self.sounds.play(sound_for_move(&mv));
                    self.on_move_finished();
                    return;
                } else {
                    self.sounds.play(SoundEvent::Illegal);
                }
            }
            // Перевыбор своей фигуры или сброс
            let val = self.state.board[idx];
            if val != chess4p_game::EMPTY
                && chess4p_game::piece_color(val) == Some(self.state.turn)
            {
                self.selected = Some(sq);
            } else {
                self.selected = None;
            }
            self.legal_cache = None;
        } else {
            let val = self.state.board[idx];
            if val != chess4p_game::EMPTY && chess4p_game::piece_color(val) == Some(self.state.turn) {
                self.selected = Some(sq);
                self.legal_cache = None;
            }
        }
    }

    /// После любого применённого хода: разовый звук окончания партии.
    fn on_move_finished(&mut self) {
        if self.state.is_game_over() && !self.game_over_played {
            self.game_over_played = true;
            self.sounds.play(SoundEvent::GameEnd);
        }
    }

    /// Статус калибровки (перечитывается при изменении файла)
    fn refresh_calibration_status(&mut self) {
        let path = std::path::Path::new(&self.settings.bot_dir).join("calibration_data.json");
        let len = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        if self.calib_cache != Some(len) {
            self.calib_cache = Some(len);
            self.calib_status = if len == 0 {
                "❌ калибровка не выполнена".to_string()
            } else {
                match std::fs::read_to_string(&path)
                    .ok()
                    .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
                {
                    Some(v) => {
                        let cell = v.get("cell_size").and_then(|x| x.as_u64()).unwrap_or(0);
                        let w = v.get("board_width").and_then(|x| x.as_u64()).unwrap_or(0);
                        format!("✅ калибровка: клетка {cell}px, доска {w}×{w}px")
                    }
                    None => "⚠ файл калибровки повреждён".to_string(),
                }
            };
        }
    }

    fn delete_calibration(&mut self) {
        let dir = std::path::Path::new(&self.settings.bot_dir);
        for f in ["calibration_data.json", "calibration_screenshot.png", "board_crop.png"] {
            let _ = std::fs::remove_file(dir.join(f));
        }
        self.calib_cache = None;
        self.calib_status = "❌ калибровка не выполнена".to_string();
        self.status = "Калибровка удалена".into();
    }

    fn new_game(&mut self) {
        self.engine.cancel();
        self.state = GameState::new();
        self.invalidate();
        self.review_idx = None;
        self.pending_for = None;
        self.last_result = None;
        self.game_over_played = false;
        self.marks.clear();
        self.status = "Новая партия".into();
    }

    fn load_pgn(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("PGN", &["pgn", "pgn4"])
            .add_filter("Все файлы", &["*"])
            .pick_file()
        {
            match std::fs::read_to_string(&path) {
                Ok(text) => {
                    self.engine.cancel();
                    let mut st = GameState::new();
                    match st.load_pgn_verbose(&text) {
                        Ok(()) => {
                            self.state = st;
                            self.invalidate();
                            self.review_idx = None;
                            self.pending_for = None;
                            self.game_over_played = false;
                            self.status = format!("Загружено ходов: {}", self.state.move_history.len());
                        }
                        Err(e) => self.error = Some(format!("Ошибка PGN: {e}")),
                    }
                }
                Err(e) => self.error = Some(format!("Не удалось прочитать файл: {e}")),
            }
        }
    }

    fn save_game_json(&mut self) {
        #[derive(serde::Serialize)]
        struct SM {
            from: [usize; 2],
            to: [usize; 2],
            skipped: bool,
        }
        #[derive(serde::Serialize)]
        struct SG {
            version: u32,
            moves: Vec<SM>,
        }
        let moves = self
            .state
            .move_history
            .iter()
            .map(|m| SM {
                from: [m.from.0, m.from.1],
                to: [m.to.0, m.to.1],
                skipped: m.skipped,
            })
            .collect();
        let sg = SG { version: 1, moves };
        if let Some(path) = rfd::FileDialog::new()
            .set_file_name("game.json")
            .add_filter("JSON", &["json"])
            .save_file()
        {
            match serde_json::to_string_pretty(&sg)
                .map_err(|e| e.to_string())
                .and_then(|j| std::fs::write(&path, j).map_err(|e| e.to_string()))
            {
                Ok(()) => self.status = format!("Сохранено: {}", path.display()),
                Err(e) => self.error = Some(format!("Ошибка сохранения: {e}")),
            }
        }
    }

    fn load_game_json(&mut self) {
        #[derive(serde::Deserialize)]
        struct SM {
            from: [usize; 2],
            to: [usize; 2],
            #[serde(default)]
            skipped: bool,
        }
        #[derive(serde::Deserialize)]
        struct SG {
            moves: Vec<SM>,
        }
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("JSON", &["json"])
            .pick_file()
        {
            match std::fs::read_to_string(&path) {
                Ok(text) => match serde_json::from_str::<SG>(&text) {
                    Ok(sg) => {
                        self.engine.cancel();
                        let mut st = GameState::new();
                        let mut failed = 0;
                        for m in &sg.moves {
                            let ok = if m.skipped {
                                st.pass_turn_and_record_skip_pub(false);
                                true
                            } else {
                                st.make_move_unchecked(
                                    (m.from[0], m.from[1]),
                                    (m.to[0], m.to[1]),
                                    true,
                                    None,
                                )
                            };
                            if !ok {
                                failed += 1;
                            }
                        }
                        self.state = st;
                        self.invalidate();
                        self.review_idx = None;
                        self.pending_for = None;
                        self.game_over_played = false;
                        self.status = if failed == 0 {
                            format!("Загружено ходов: {}", sg.moves.len())
                        } else {
                            format!("Загружено ходов: {}, пропущено: {}", sg.moves.len(), failed)
                        };
                    }
                    Err(e) => self.error = Some(format!("Ошибка JSON: {e}")),
                },
                Err(e) => self.error = Some(format!("Не удалось прочитать файл: {e}")),
            }
        }
    }
}

fn sound_for_move(m: &Move) -> SoundEvent {
    if m.castle > 0 {
        SoundEvent::Castle
    } else if m.promotion.is_some() {
        SoundEvent::Promote
    } else if m.captured.is_some() {
        SoundEvent::Capture
    } else if m.check {
        SoundEvent::Check
    } else {
        SoundEvent::Move
    }
}

fn coord_str((x, y): (usize, usize)) -> String {
    format!("{}{}", (b'a' + (x as u8 - 1)) as char, y)
}

fn app_log(line: &str) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("gui_debug.log")
    {
        let _ = writeln!(f, "{line}");
    }
}

fn install_piece_font(cc: &eframe::CreationContext<'_>) -> bool {
    let candidates = [
        r"C:\Windows\Fonts\seguisym.ttf",
        r"C:\Windows\Fonts\seguiemj.ttf",
    ];
    for path in candidates {
        if let Ok(bytes) = std::fs::read(path) {
            let mut fonts = egui::FontDefinitions::default();
            fonts.font_data.insert(
                "chess_symbols".to_owned(),
                Arc::new(egui::FontData::from_owned(bytes)),
            );
            for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
                fonts
                    .families
                    .entry(family)
                    .or_default()
                    .push("chess_symbols".to_owned());
            }
            cc.egui_ctx.set_fonts(fonts);
            return true;
        }
    }
    false
}

// ============================================================
// eframe::App
// ============================================================

impl eframe::App for Chess4pApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Навигация по истории стрелками клавиатуры (только на вкладке «Игра»)
        let (left, right) = ctx.input(|i| (
            i.key_pressed(egui::Key::ArrowLeft),
            i.key_pressed(egui::Key::ArrowRight),
        ));
        if self.tab == Tab::Game && (left || right) {
            let n = self.state.move_history.len();
            let cur = self.review_idx.unwrap_or(n);
            if left && cur > 0 {
                self.review_idx = Some(cur - 1);
                self.invalidate();
            } else if right {
                let next = cur + 1;
                self.review_idx = if next >= n { None } else { Some(next) };
                self.invalidate();
            }
        }

        // Автопилот ботов
        self.bot_autopilot();

        // Переапи завершившихся скриптов
        for (title, code) in self.runner.reap() {
            match code {
                Some(0) => self.status = format!("✅ {title} — завершён"),
                Some(c) => self.status = format!("⚠ {title} — код выхода {c}"),
                None => self.status = format!("⚠ {title} — аварийно завершён"),
            }
        }

        // Сообщение об ошибке сверху
        if self.error.is_some() {
            let err = self.error.clone().unwrap();
            egui::Window::new("⚠ Ошибка")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.label(err.clone());
                    ui.add_space(6.0);
                    if ui.button("Закрыть").clicked() {
                        self.error = None;
                    }
                });
        }

        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("♞ Chess4P");
                ui.separator();
                ui.selectable_value(&mut self.tab, Tab::Game, "🎮 Игра");
                ui.selectable_value(&mut self.tab, Tab::Scripts, "📜 Скрипты");
                ui.selectable_value(&mut self.tab, Tab::Settings, "⚙ Настройки");
            });
        });

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(&self.status).weak());
            });
        });

        match self.tab {
            Tab::Game => self.show_game(ctx),
            Tab::Scripts => self.show_scripts(ctx),
            Tab::Settings => self.show_settings(ctx),
        }

        // Перерисовка пока бот думает или скрипт пишет лог
        if self.pending_for.is_some() || !self.runner.running.is_empty() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.save_settings();
    }
}

impl Chess4pApp {
    // ========================================================
    // Вкладка «Игра»
    // ========================================================

    fn show_game(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("seats_panel")
            .min_width(330.0)
            .show(ctx, |ui| {
                ScrollArea::vertical().show(ui, |ui| {
                    self.seats_panel(ui);
                    ui.add_space(10.0);
                    self.history_panel(ui);
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            // Статус игры сверху
            let over = self.state.is_game_over();
            if over {
                let mut order: Vec<(usize, i32)> = self
                    .state
                    .scores
                    .iter()
                    .copied()
                    .enumerate()
                    .collect();
                order.sort_by_key(|(_, s)| -*s);
                let line: String = order
                    .iter()
                    .map(|(i, s)| format!("{}: {s}", PLAYER_NAMES[*i]))
                    .collect::<Vec<_>>()
                    .join("  |  ");
                ui.label(RichText::new("🏁 Игра окончена").heading());
                ui.label(RichText::new(line).strong());
                ui.add_space(6.0);
            }

            // Отображаемое состояние (live или review)
            let (display, review_mode) = if let Some(idx) = self.review_idx {
                // ВАЖНО: полный клон — clone_light() не копирует history,
                // и go_to_move() на нём всегда проваливался.
                let mut st = self.state.clone();
                st.go_to_move(idx);
                (st, true)
            } else {
                (self.state.clone_light(), false)
            };

            let last_move = display.last_move.map(|m| (m.from, m.to));

            // Подсветка легальных целей
            let mut targets: Vec<(usize, usize)> = Vec::new();
            if !review_mode {
                if let Some(sel) = self.selected {
                    let legal = self.current_legal_moves();
                    for m in legal.iter() {
                        if m.from == sel {
                            targets.push(m.to);
                        }
                    }
                }
            }

            let cell = (board::DEFAULT_CELL * self.board_zoom)
                .clamp(board::MIN_CELL, board::MAX_CELL);
            let void_color = ui.style().visuals.panel_fill;

            // Зум: кнопки + колесо мыши над доской
            ui.horizontal(|ui| {
                ui.label("Доска:");
                if ui.small_button("➖").clicked() {
                    self.board_zoom = ((self.board_zoom - 0.1).max(0.5) * 10.0).round() / 10.0;
                    self.save_settings();
                }
                if ui.small_button("➕").clicked() {
                    self.board_zoom = ((self.board_zoom + 0.1).min(2.0) * 10.0).round() / 10.0;
                    self.save_settings();
                }
                if ui.small_button("стандарт").clicked() {
                    self.board_zoom = 1.0;
                    self.save_settings();
                }
                ui.label(format!("{:.0}%", self.board_zoom * 100.0));
            });

            let resp: BoardResponse = board::draw_board(
                ui,
                cell,
                void_color,
                &display,
                self.selected,
                &targets,
                last_move,
                self.piece_textures.as_ref(),
                self.font_ok,
                &mut self.marks,
            );

            // Колесо мыши над доской тоже меняет зум
            if resp.hovered {
                let scroll = ui.input(|i| i.raw_scroll_delta.y);
                if scroll.abs() > 0.5 {
                    let factor = 1.0 - scroll * 0.0007;
                    let old = self.board_zoom;
                    self.board_zoom = (self.board_zoom * factor).clamp(0.5, 2.0);
                    if (self.board_zoom - old).abs() > 1e-4 {
                        self.save_settings();
                    }
                }
            }

            if let Some(sq) = resp.clicked {
                if review_mode {
                    self.review_idx = None; // клик по доске возвращает в игру
                    self.invalidate();
                } else {
                    self.human_click(sq);
                }
            }

            ui.add_space(4.0);
            if review_mode {
                ui.label(RichText::new("👀 Режим просмотра — клик по доске или кнопка вернёт к текущей позиции").weak().small());
            }
        });
    }

    fn seats_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Игроки");
        ui.add_space(4.0);

        let turn_idx = self.state.turn.index();
        let model_loaded = self.engine.is_loaded();

        for i in 0..4usize {
            let _color = Color::ALL[i];
            let is_turn = turn_idx == i && !self.state.is_game_over();
            let dead = self.state.dead_colors[i];
            let resigned = self.state.resigned[i];

            ui.horizontal(|ui| {
                let (rect, _) = ui.allocate_exact_size(egui::vec2(16.0, 16.0), egui::Sense::hover());
                ui.painter().rect_filled(rect, 4.0, PLAYER_COLORS[i]);
                let label = format!(
                    "{} {}",
                    PLAYER_NAMES[i],
                    if dead { "☠" } else if resigned { "🏳" } else { "" }
                );
                if is_turn {
                    ui.label(RichText::new(label).strong());
                } else {
                    ui.label(label);
                }
            });

            ui.horizontal(|ui| {
                let mut kind = self.seats[i].kind;
                let mut changed = false;
                if ui
                    .selectable_label(kind == SeatKind::Human, "👤 Человек")
                    .clicked()
                {
                    kind = SeatKind::Human;
                    changed = true;
                }
                if ui
                    .add_enabled(
                        model_loaded,
                        egui::Button::new(if kind == SeatKind::Bot { "🤖 Бот ✓" } else { "🤖 Бот" }),
                    )
                    .clicked()
                {
                    kind = SeatKind::Bot;
                    changed = true;
                }
                if changed {
                    self.seats[i].kind = kind;
                    self.pending_for = None;
                }
            });

            if self.seats[i].kind == SeatKind::Bot {
                ui.horizontal(|ui| {
                    ui.label("Симуляций:");
                    let mut sims = self.seats[i].simulations;
                    let r = ui.add(
                        egui::DragValue::new(&mut sims)
                            .speed(10)
                            .range(16..=100_000),
                    );
                    if r.changed() {
                        self.seats[i].simulations = sims;
                    }
                    for (label, v) in [("128", 128u32), ("400", 400), ("1600", 1600)] {
                        if ui.small_button(label).clicked() {
                            self.seats[i].simulations = v;
                        }
                    }
                });
            }

            ui.add_space(4.0);
        }

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(4.0);

        // Счёт
        ui.heading("Счёт");
        for i in 0..4usize {
            ui.label(format!("{}: {} очк.", PLAYER_NAMES[i], self.state.scores[i]));
        }

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(4.0);

        // Кнопки управления
        ui.horizontal(|ui| {
            if ui.button("🆕 Новая партия").clicked() {
                self.new_game();
            }
            if ui.button("📂 PGN").clicked() {
                self.load_pgn();
            }
        });
        ui.horizontal(|ui| {
            if ui.button("💾 JSON").clicked() {
                self.save_game_json();
            }
            if ui.button("📥 JSON").clicked() {
                self.load_game_json();
            }
        });

        // Сдаться (для человек-места)
        let turn_idx = self.state.turn.index();
        if !self.state.is_game_over()
            && self.seats[turn_idx].kind == SeatKind::Human
            && !self.state.resigned[turn_idx]
        {
            ui.horizontal(|ui| {
                if ui.button("🏳 Сдаться (текущий игрок)").clicked() {
                    self.state.resign_player(Color::ALL[turn_idx]);
                    self.invalidate();
                }
                // Пропуск хода, если легальных ходов нет
                let no_moves = self.state.generate_legal_moves().is_empty();
                if no_moves && ui.button("⏭ Пропуск хода").clicked() {
                    self.state.pass_turn_and_record_skip_pub(false);
                    self.invalidate();
                }
            });
        }

        // Прогресс поиска
        if self.pending_for.is_some() {
            ui.add_space(4.0);
            ui.label("⏳ Поиск хода...");
        }
        if let Some(res) = &self.last_result {
            ui.add_space(4.0);
            ui.collapsing("Последний ход бота", |ui| {
                ui.label(format!("Оценка: {:.3}", res.root_value));
                ui.label(format!("Визиты: {}, узлов: {}", res.visits, res.nodes));
                ui.label(format!("Время: {:.2?}", res.elapsed));
                if !res.pv.is_empty() {
                    let pv: Vec<String> = res
                        .pv
                        .iter()
                        .map(|m| format!("{}→{}", coord_str(m.from), coord_str(m.to)))
                        .collect();
                    ui.label(format!("PV: {}", pv.join("  ")));
                }
            });
        }
    }

    fn history_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("История ходов");
        let n = self.state.move_history.len();
        ui.label(format!("Ходов: {n}"));

        ui.horizontal(|ui| {
            let n = self.state.move_history.len();
            let live = !self.is_reviewing();
            let cur = self.review_idx.unwrap_or(n);

            // ⏮ в начало (позиция 0 = стартовая расстановка)
            if ui.add_enabled(cur > 0, egui::Button::new("⏮")).clicked() {
                self.review_idx = Some(0);
                self.invalidate();
            }
            // ◀ на ход назад
            if ui.add_enabled(cur > 0, egui::Button::new("◀")).clicked() {
                self.review_idx = Some(cur - 1);
                self.invalidate();
            }
            // ▶ на ход вперёд
            if ui.add_enabled(live || cur < n, egui::Button::new("▶")).clicked() {
                let next = cur + 1;
                if next >= n {
                    self.review_idx = None; // вернулись к текущей позиции
                } else {
                    self.review_idx = Some(next);
                }
                self.invalidate();
            }
            // ⏭ к текущей позиции
            if ui.add_enabled(live || cur < n, egui::Button::new("⏭")).clicked() {
                self.review_idx = None;
                self.invalidate();
            }
            if live {
                ui.label(RichText::new("(текущая)").weak().small());
            } else {
                ui.label(RichText::new(format!("{} / {}", cur, n)).weak().small());
            }
        });

        ScrollArea::vertical()
            .id_salt("history_scroll")
            .max_height(240.0)
            .stick_to_bottom(true)
            .show(ui, |ui| {
                let mut goto: Option<usize> = None;
                for i in 0..n {
                    let m = &self.state.move_history[i];
                    let text = if m.skipped {
                        format!("{}. пропуск ({})", i + 1, PLAYER_NAMES[m.color.index()])
                    } else {
                        format!(
                            "{}. {} {} → {}",
                            i + 1,
                            PLAYER_NAMES[m.color.index()],
                            coord_str(m.from),
                            coord_str(m.to)
                        )
                    };
                    let viewing = self.review_idx == Some(i + 1);
                    if ui
                        .selectable_label(viewing, text)
                        .clicked()
                    {
                        goto = Some(i + 1);
                    }
                }
                if let Some(idx) = goto {
                    self.review_idx = Some(idx);
                    self.invalidate();
                }
            });
    }

    // ========================================================
    // Вкладка «Скрипты»
    // ========================================================

    fn show_scripts(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Скрипты bot/");
            ui.label(
                RichText::new(format!(
                    "Python: {}   |   Папка: {}",
                    self.settings.python_path, self.settings.bot_dir
                ))
                .weak()
                .small(),
            );
            ui.add_space(6.0);

            ScrollArea::vertical().show(ui, |ui| {
                // ================================================
                // Секция «Экранный бот»
                // ================================================
                self.refresh_calibration_status();
                let busy_cal = self.runner.running.iter().any(|r| r.title == "Калибровка");
                let busy_check = self.runner.running.iter().any(|r| r.title == "Проверка калибровки");
                let busy_bot = self.runner.running.iter().any(|r| r.title == "Игра ботом");
                let calib_exists = self.calib_cache.unwrap_or(0) > 0;
                let calib_status = self.calib_status.clone();
                let kill_bot = self
                    .runner
                    .running
                    .iter()
                    .find(|r| r.title == "Игра ботом")
                    .map(|r| r.id);

                ui.group(|ui| {
                    ui.heading("🎮 Экранный бот (chess.com)");

                    // --- Калибровка ---
                    ui.label("Калибровка доски:");
                    ui.label(
                        RichText::new(calib_status.clone()).color(if calib_exists {
                            Color32::from_rgb(80, 200, 90)
                        } else {
                            Color32::from_rgb(230, 120, 60)
                        }),
                    );
                    ui.label(
                        RichText::new(format!("Папка бота: {}", self.settings.bot_dir)).weak().small(),
                    );
                    ui.horizontal(|ui| {
                        if ui.small_button("📁 Указать папку bot...").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .set_title("Выберите папку bot (где config.py)")
                                .pick_folder()
                            {
                                self.settings.bot_dir = path.to_string_lossy().to_string();
                                self.calib_cache = None; // перечитать статус
                                self.save_settings();
                            }
                        }
                    });
                    ui.horizontal(|ui| {
                        let python = self.settings.python_path.clone();
                        let bot_dir = self.settings.bot_dir.clone();
                        let mut fail = |msg: String| self.error = Some(msg);

                        let cal_label = if calib_exists {
                            "✏️ Перекалибровать"
                        } else {
                            "📐 Калибровать"
                        };
                        if ui
                            .add_enabled(!busy_cal, egui::Button::new(cal_label))
                            .clicked()
                        {
                            let def = crate::scripts::ScriptDef {
                                path: "calibration.py",
                                title: "Калибровка",
                                desc: "",
                            };
                            self.runner.spawn(&def, &python, &bot_dir, &[], true, &mut fail);
                            self.status = "Калибровка: следуйте инструкциям в открывшемся окне (2 клика по углам доски, WASD/Q/E подгонка, Enter — сохранить)".into();
                        }
                        if ui
                            .add_enabled(!busy_check, egui::Button::new("🔍 Проверить"))
                            .clicked()
                        {
                            let def = crate::scripts::ScriptDef {
                                path: "tools/check_calibration.py",
                                title: "Проверка калибровки",
                                desc: "",
                            };
                            self.runner.spawn(&def, &python, &bot_dir, &[], true, &mut fail);
                        }
                        if ui
                            .add_enabled(calib_exists && !busy_cal, egui::Button::new("🗑 Удалить"))
                            .clicked()
                        {
                            self.delete_calibration();
                        }
                    });
                    ui.label(
                        RichText::new(
                            "Выделите доску двумя кликами (верхний-левый и нижний-правый углы),                              подгоните сетку (WASD/Q/E), Enter — сохранить.",
                        )
                        .weak()
                        .small(),
                    );

                    ui.separator();

                    // --- Игра ботом ---
                    ui.label("Игра ботом на chess.com:");
                    ui.horizontal(|ui| {
                        ui.label("Цвет:");
                        egui::ComboBox::from_id_salt("bot_mode")
                            .selected_text(match self.bot_mode.as_str() {
                                "red" => "красный",
                                "blue" => "синий",
                                "yellow" => "жёлтый",
                                "green" => "зелёный",
                                _ => "все цвета",
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut self.bot_mode, "ever".to_string(), "все цвета");
                                ui.selectable_value(&mut self.bot_mode, "red".to_string(), "красный");
                                ui.selectable_value(&mut self.bot_mode, "blue".to_string(), "синий");
                                ui.selectable_value(&mut self.bot_mode, "yellow".to_string(), "жёлтый");
                                ui.selectable_value(&mut self.bot_mode, "green".to_string(), "зелёный");
                            });

                        ui.label("Симуляций:");
                        ui.add(
                            egui::DragValue::new(&mut self.bot_sims)
                                .speed(10)
                                .range(16..=100_000),
                        );
                        for (label, v) in [("128", 128u32), ("400", 400), ("1600", 1600)] {
                            if ui.small_button(label).clicked() {
                                self.bot_sims = v;
                            }
                        }
                    });
                    ui.horizontal(|ui| {
                        if busy_bot {
                            ui.label(RichText::new("🤖 Бот играет (лог в консоли ниже)...").strong());
                            if let Some(id) = kill_bot {
                                if ui.button("⏹ Остановить").clicked() {
                                    self.runner.kill(id);
                                }
                            }
                        } else if ui.button("▶ Играть ботом").clicked() {
                            let python = self.settings.python_path.clone();
                            let bot_dir = self.settings.bot_dir.clone();
                            let mut fail = |msg: String| self.error = Some(msg);
                            let def = crate::scripts::ScriptDef {
                                path: "main.py",
                                title: "Игра ботом",
                                desc: "",
                            };
                            let args = vec![
                                "--mode".to_string(),
                                self.bot_mode.clone(),
                                "--sims".to_string(),
                                self.bot_sims.to_string(),
                            ];
                            self.runner.spawn(&def, &python, &bot_dir, &args, false, &mut fail);
                            self.status = "🤖 Бот запущен: сам определит свой цвет и будет ждать его таймер".into();
                        }
                    });
                    ui.label(
                        RichText::new(
                            "Бот определит свой цвет, будет ждать свой таймер и делать ходы сам.                              Лог — в консоли ниже.",
                        )
                        .weak()
                        .small(),
                    );
                });

                ui.add_space(8.0);

                if !self.runner.running.is_empty() {
                    ui.heading("Запущенные процессы");
                    let mut kill: Option<u64> = None;
                    for rs in &self.runner.running {
                        ui.horizontal(|ui| {
                            ui.label(format!("▶ {} (pid {:?})", rs.title, rs.child.lock().map(|c| c.id()).unwrap_or(0)));
                            if ui.small_button("⛔ Стоп").clicked() {
                                kill = Some(rs.id);
                            }
                        });
                    }
                    if let Some(id) = kill {
                        self.runner.kill(id);
                    }
                }

                ui.add_space(8.0);
                ui.heading("Вывод");
                ui.checkbox(&mut self.runner.auto_scroll, "Автопрокрутка");
                egui::ScrollArea::vertical()
                    .id_salt("script_log")
                    .auto_shrink([false, false])
                    .stick_to_bottom(self.runner.auto_scroll)
                    .show(ui, |ui| {
                        let log = self.runner.log.lock().map(|l| l.clone()).unwrap_or_default();
                        for (_, line) in log {
                            ui.monospace(line);
                        }
                    });
            });
        });
    }

    // ========================================================
    // Вкладка «Настройки»
    // ========================================================

    fn show_settings(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Настройки");
            ui.add_space(6.0);

            ui.group(|ui| {
                ui.label(RichText::new("Модель (ONNX)").strong());
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    if ui.button("Выбрать model.onnx...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("ONNX", &["onnx"])
                            .pick_file()
                        {
                            self.settings.model_path = path.to_string_lossy().to_string();
                            match self.engine.load(&self.settings.model_path.clone()) {
                                Ok(msg) => self.status = msg,
                                Err(e) => self.error = Some(e),
                            }
                            self.save_settings();
                        }
                    }
                    if self.engine.is_loaded() {
                        ui.label(RichText::new(format!("✅ {}", self.engine.model_name)).color(Color32::from_rgb(80, 200, 90)));
                    } else {
                        ui.label(RichText::new("❌ не загружена").weak());
                    }
                });
                ui.label(
                    RichText::new(&self.settings.model_path)
                        .weak()
                        .small(),
                );
                ui.add_space(4.0);
                ui.label(
                    RichText::new(
                        "Нет model.onnx? Экспортируйте:  python scripts/export_onnx.py path/to/model.pt",
                    )
                    .weak()
                    .small(),
                );
            });

            ui.add_space(8.0);
            ui.group(|ui| {
                ui.label(RichText::new("Скрипты").strong());
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label("Python:");
                    if ui.button("Выбрать python.exe...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Python", &["exe"])
                            .pick_file()
                        {
                            self.settings.python_path = path.to_string_lossy().to_string();
                            self.save_settings();
                        }
                    }
                });
                ui.text_edit_singleline(&mut self.settings.python_path);
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label("Папка bot/:");
                    if ui.button("Выбрать...").clicked() {
                        if let Some(path) = rfd::FileDialog::new().pick_folder() {
                            self.settings.bot_dir = path.to_string_lossy().to_string();
                            self.save_settings();
                        }
                    }
                });
                ui.text_edit_singleline(&mut self.settings.bot_dir);
                if ui.button("Сохранить").clicked() {
                    self.save_settings();
                    self.status = "Настройки сохранены".into();
                }
            });

            ui.add_space(8.0);
            ui.label(
                RichText::new(format!(
                    "Шрифт шахматных символов: {}",
                    if self.font_ok { "✅ Segoe UI Symbol" } else { "❌ не найден — фигуры буквами" }
                ))
                .weak(),
            );
        });
    }
}
