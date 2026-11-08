//! Chess4P — нативный GUI (egui): игра с ботами 4-игрочных шахмат
//! + лаунчер скриптов из bot/.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod board;
mod engine;
mod scripts;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Chess4P")
            .with_inner_size([1360.0, 900.0])
            .with_min_inner_size([1100.0, 700.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Chess4P",
        options,
        Box::new(|cc| Ok(Box::new(app::Chess4pApp::new(cc)))),
    )
}
