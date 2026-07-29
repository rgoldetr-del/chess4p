# justfile — автоматизация сборки
set shell := ["cmd", "/C"]

godot_project := "C:\\chess4p\\godot-project"
godot_bin := godot_project + "\\bin"

# Только Godot extension
godot:
    cargo build -p chess-analyzer --release
    copy /Y target\release\chess_analyzer.dll {{godot_bin}}\libchess_analyzer.dll
    @echo ✅ Godot DLL обновлена

# Собрать всё и скопировать DLL
build:
    cargo build --release
    copy /Y "target\release\chess_analyzer.dll" "{{godot_bin}}\libchess_analyzer.dll"
    @echo ✅ DLL скопирована в {{godot_bin}}

# Только PyO3
pyo3:
    cargo build -p chess4p-ai --release
    @echo ✅ PyO3 собран.

# Экспорт ONNX
onnx:
    cd training && python model.py export
    copy /Y "training\model.onnx" "{{godot_project}}\model.onnx"
    @echo ✅ ONNX экспортирован и скопирован

# Полный цикл
all: build onnx
    @echo 🎉 Всё готово. Открывайте Godot.

# Запустить Godot
run:
    "{{godot_bin}}\..\..\Godot.exe" --path "{{godot_project}}"

# Очистка
clean:
    cargo clean
    del /Q "{{godot_bin}}\*.dll" 2>nul