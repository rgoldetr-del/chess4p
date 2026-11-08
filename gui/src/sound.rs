//! sound.rs — воспроизведение звуков с «антидребезгом».
//!
//! Звуки играются в отдельном потоке. Когда ходы идут быстро (например,
//! играют 4 бота), каждая категория звука имеет минимальный интервал —
//! слишком частые повторы просто пропускаются, чтобы не было каши.

use std::collections::HashMap;
use std::io::Cursor;
use std::path::Path;
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SoundEvent {
    Move,     // обычный ход
    Check,    // ход с шахом
    Capture,  // взятие
    Castle,   // рокировка
    Promote,  // превращение
    Illegal,  // запрещённый ход
    GameEnd,  // партия окончена
}

impl SoundEvent {
    fn file(self) -> &'static str {
        match self {
            SoundEvent::Move => "move-opponent.wav",
            SoundEvent::Check => "move-check.wav",
            SoundEvent::Capture => "capture.wav",
            SoundEvent::Castle => "castle.wav",
            SoundEvent::Promote => "promote.wav",
            SoundEvent::Illegal => "illegal.wav",
            SoundEvent::GameEnd => "game-end.wav",
        }
    }

    fn cooldown(self) -> Duration {
        match self {
            SoundEvent::Move => Duration::from_millis(260),
            SoundEvent::Check => Duration::from_millis(350),
            SoundEvent::Capture => Duration::from_millis(350),
            SoundEvent::Castle => Duration::from_millis(350),
            SoundEvent::Promote => Duration::from_millis(350),
            SoundEvent::Illegal => Duration::from_millis(300),
            SoundEvent::GameEnd => Duration::from_millis(2000),
        }
    }
}

pub struct SoundPlayer {
    tx: Option<Sender<SoundEvent>>,
}

impl SoundPlayer {
    /// Загружает звуки из каталога и запускает поток воспроизведения.
    /// Если аудио недоступно или каталог пуст — играет молча.
    pub fn new(dir: &Path) -> Self {
        // Читаем файлы заранее (wav 30-70 КБ каждый)
        let mut files: HashMap<SoundEvent, Vec<u8>> = HashMap::new();
        for ev in [
            SoundEvent::Move,
            SoundEvent::Check,
            SoundEvent::Capture,
            SoundEvent::Castle,
            SoundEvent::Promote,
            SoundEvent::Illegal,
            SoundEvent::GameEnd,
        ] {
            let path = dir.join(ev.file());
            if let Ok(bytes) = std::fs::read(&path) {
                files.insert(ev, bytes);
            }
        }
        if files.is_empty() {
            return Self { tx: None };
        }

        let (tx, rx) = mpsc::channel::<SoundEvent>();
        let ok = std::thread::Builder::new()
            .name("sound-player".into())
            .spawn(move || sound_thread(rx, files))
            .is_ok();
        if ok {
            Self { tx: Some(tx) }
        } else {
            Self { tx: None }
        }
    }

    /// Неблокирующий запрос на звук; слишком частые повторы отсеются в потоке.
    pub fn play(&self, ev: SoundEvent) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(ev);
        }
    }
}

fn sound_thread(rx: Receiver<SoundEvent>, files: HashMap<SoundEvent, Vec<u8>>) {
    let (_stream, handle) = match rodio::OutputStream::try_default() {
        Ok(s) => s,
        Err(_) => return,
    };
    let sink = match rodio::Sink::try_new(&handle) {
        Ok(s) => s,
        Err(_) => return,
    };
    sink.set_volume(0.6);

    let mut last: HashMap<SoundEvent, Instant> = HashMap::new();

    while let Ok(ev) = rx.recv() {
        let Some(bytes) = files.get(&ev) else { continue };

        // Антидребезг: слишком частые звуки одной категории пропускаем
        let now = Instant::now();
        if let Some(t) = last.get(&ev) {
            if now.duration_since(*t) < ev.cooldown() {
                continue;
            }
        }
        last.insert(ev, now);

        if sink.len() > 3 {
            // Очередь переполнена (быстрая игра) — самые старые звуки не важны
            sink.clear();
        }

        let cursor = Cursor::new(bytes.clone());
        if let Ok(source) = rodio::Decoder::new(cursor) {
            sink.append(source);
        }
    }
}
