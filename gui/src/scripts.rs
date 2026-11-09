//! scripts.rs — лаунчер скриптов из папки bot/ с выводом в консоль GUI.

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy)]
pub struct ScriptDef {
    pub path: &'static str,   // относительный путь от bot/
    pub title: &'static str,  // название для кнопки
    pub desc: &'static str,   // что делает
}

pub const SCRIPTS: &[ScriptDef] = &[
    ScriptDef { path: "create_templates.py", title: "Создать шаблоны", desc: "Снятие шаблонов фигур с доски" },
    ScriptDef { path: "create_promoted_gray_templates.py", title: "Шаблоны серых", desc: "Шаблоны превращённых/серых фигур" },
    ScriptDef { path: "clean_gray_templates.py", title: "Очистка серых шаблонов", desc: "Удаление пустых/мусорных шаблонов" },
    ScriptDef { path: "tools/check_environment.py", title: "Проверка окружения", desc: "Диагностика зависимостей и модели" },
    ScriptDef { path: "tools/benchmark.py", title: "Бенчмарк бота", desc: "Скорость распознавания/MCTS" },
    ScriptDef { path: "download_pgn4_auto.py", title: "Скачать PGN4 (auto)", desc: "Скачивание партий chess.com" },
    ScriptDef { path: "download_all_pgn4_v2.py", title: "Скачать PGN4 (все)", desc: "Массовое скачивание архива партий" },
    ScriptDef { path: "parse_ffa_dataset.py", title: "Парсинг FFA", desc: "Разбор скачанных PGN в датасет" },
    ScriptDef { path: "filter_clean_ffa.py", title: "Фильтр чистых FFA", desc: "Отбор чистых FFA-партий для обучения" },
    ScriptDef { path: "validate_and_filter_pgn4.py", title: "Валидация PGN4", desc: "Проверка и фильтрация PGN4 файлов" },
    ScriptDef { path: "build.py", title: "Сборка chess4p_ai", desc: "cargo build + maturin develop" },
    ScriptDef { path: "build_soup.py", title: "Soup-модель", desc: "Обучение soup из чекпойнтов" },
];

pub struct RunningScript {
    pub id: u64,
    pub title: String,
    pub child: Arc<Mutex<Child>>,
    pub generation: u64,
}

#[derive(Default)]
pub struct ScriptRunner {
    pub running: Vec<RunningScript>,
    pub next_id: u64,
    /// Общий журнал: (generation, строка)
    pub log: Arc<Mutex<Vec<(u64, String)>>>,
    pub generation: u64,
    pub auto_scroll: bool,
}

impl ScriptRunner {
    pub fn spawn(
        &mut self,
        script: &ScriptDef,
        python: &str,
        bot_dir: &str,
        args: &[String],
        new_console: bool,
        on_fail: &mut dyn FnMut(String),
    ) {
        self.generation += 1;
        let gen = self.generation;
        let title = script.title.to_string();

        let mut cmd = Command::new(python);
        cmd.arg(script.path)
            .args(args)
            .current_dir(bot_dir);

        // Интерактивным скриптам (input(), OpenCV-окна) нужна своя консоль:
        // вывод идёт в неё, а не в лог GUI.
        if new_console {
            cmd.creation_flags(0x00000010); // CREATE_NEW_CONSOLE
        } else {
            cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
        }

        let child = cmd.spawn();

        let child = match child {
            Ok(c) => c,
            Err(e) => {
                let mut msg = format!("❌ Не удалось запустить {}: {e}", script.path);
                if e.raw_os_error() == Some(267) {
                    msg.push_str(
                        " — папка бота не найдена. Укажите её кнопкой «📁 Указать папку bot»",
                    );
                }
                on_fail(msg);
                return;
            }
        };
        let child = Arc::new(Mutex::new(child));
        let shared_log = self.log.clone();

        // Читаем stdout и stderr в фоновых потоках (только если пайп есть)
        if !new_console {
        {
            let child_ref = child.clone();
            let shared = shared_log.clone();
            thread::spawn(move || {
                // Ждём появления stdout
                let stdout = loop {
                    if let Ok(mut c) = child_ref.lock() {
                        if let Some(out) = c.stdout.take() {
                            break out;
                        }
                    }
                    std::thread::sleep(std::time::Duration::from_millis(20));
                };
                for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                    if let Ok(mut sl) = shared.lock() {
                        sl.push((gen, line));
                        if sl.len() > 5000 {
                            sl.drain(0..1000);
                        }
                    }
                }
            });
        }
        {
            let child_ref = child.clone();
            let shared = shared_log.clone();
            thread::spawn(move || {
                let stderr = loop {
                    if let Ok(mut c) = child_ref.lock() {
                        if let Some(err) = c.stderr.take() {
                            break err;
                        }
                    }
                    std::thread::sleep(std::time::Duration::from_millis(20));
                };
                for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                    if let Ok(mut sl) = shared.lock() {
                        sl.push((gen, line));
                    }
                }
            });
        }
        } // if !new_console

        self.next_id += 1;
        self.running.push(RunningScript {
            id: self.next_id,
            title,
            child,
            generation: gen,
        });
    }

    /// Удалить завершившиеся процессы из списка (возвращает их коды выхода).
    pub fn reap(&mut self) -> Vec<(String, Option<i32>)> {
        let mut finished = Vec::new();
        self.running.retain_mut(|rs| {
            let status = match rs.child.try_lock() {
                Ok(mut c) => c.try_wait().ok().flatten(),
                Err(_) => None, // ещё занят reader-потоком — считаем живым
            };
            match status {
                Some(code) => {
                    finished.push((rs.title.clone(), code.code()));
                    false
                }
                None => true,
            }
        });
        finished
    }

    pub fn kill(&mut self, id: u64) {
        if let Some(pos) = self.running.iter().position(|r| r.id == id) {
            let rs = self.running.remove(pos);
            if let Ok(mut c) = rs.child.lock() {
                // taskkill /T убивает и дочерние процессы (python-скрипты часто плодят дети)
                let _ = Command::new("taskkill")
                    .args(["/PID", &c.id().to_string(), "/T", "/F"])
                    .creation_flags(0x08000000) // CREATE_NO_WINDOW
                    .status();
                let _ = c.kill();
            }
            if let Ok(mut l) = self.log.lock() {
                l.push((rs.generation, format!("⛔ [{}] остановлен пользователем", rs.title)));
            }
        }
    }
}

use std::os::windows::process::CommandExt;
use std::thread;
