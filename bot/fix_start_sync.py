import re

# ============================================================
# 1. state_sync.py: диагностика + восстановление серых королей
# ============================================================
p = "state_sync.py"
s = open(p, encoding="utf-8").read()

# Найти функцию try_set_board_resync и добавить восстановление серых королей
old_block = '''def try_set_board_resync(game, board, tracker, dead, resigned, turn=None):
    from coordinates import screen_to_game
    from config import COLOR_ORDER

    entries = []
    kings_found = [False, False, False, False]

    for x, y, (color, piece, prom, is_dead) in iter_pieces(board):
        if is_dead: continue'''

new_block = '''def try_set_board_resync(game, board, tracker, dead, resigned, turn=None):
    from coordinates import screen_to_game
    from config import COLOR_ORDER

    # 🚀 НОВОЕ: Попытка восстановить серых королей и пометить их цвета как dead
    for x, y, (color, piece, prom, is_dead) in iter_pieces(board):
        if is_dead and piece == 5:  # серый король
            ci = tracker.nearest_color(x, y, max_dist=8)
            if ci is not None and ci >= 0 and not dead[ci]:
                print(f"  [sync] ⚠ Обнаружен серый король на ({x},{y}), помечаю {COLOR_ORDER[ci]} как dead")
                dead[ci] = True

    entries = []
    kings_found = [False, False, False, False]

    for x, y, (color, piece, prom, is_dead) in iter_pieces(board):
        if is_dead: continue'''

if old_block in s and "# 🚀 НОВОЕ" not in s:
    s = s.replace(old_block, new_block, 1)
    print("✅ state_sync.py: gray king recovery added")
else:
    print("⚠ state_sync.py: patch already applied or block not found")

# Добавить диагностику перед "ОТКЛОНЕН"
old_diag = '''            print(f"  [sync] set_board ОТКЛОНЕН: Король {COLOR_ORDER[i]} не найден и нет в трекере.")
            return False'''

new_diag = '''            print(f"  [sync] set_board ОТКЛОНЕН: Король {COLOR_ORDER[i]} не найден и нет в трекере.")
            print(f"  [sync] kings_found: {kings_found}")
            print(f"  [sync] dead: {[bool(d) for d in dead]}, resigned: {[bool(r) for r in resigned]}")
            print(f"  [sync] tracker.last: {tracker.last}")
            # Показать все найденные короли
            for yy in range(1, 15):
                for xx in range(1, 15):
                    cell = board[yy][xx]
                    if cell and cell[1] == 5:  # king
                        print(f"  [sync] Found king at ({xx},{yy}): color={cell[0]}, dead={cell[3]}")
            return False'''

if old_diag in s and "kings_found: {kings_found}" not in s:
    s = s.replace(old_diag, new_diag, 1)
    print("✅ state_sync.py: diagnostic logging added")
else:
    print("⚠ state_sync.py: diagnostic already present")

open(p, "w", encoding="utf-8").write(s)

# ============================================================
# 2. main.py: остановка после провала set_board
# ============================================================
p = "main.py"
s = open(p, encoding="utf-8").read()

# Найти initial_resync и добавить sys.exit после провала
old_resync = '''            print("❌ set_board не удался. Бот запущен в середине чужой партии без истории PGN.")
    print("✅ Стартовая позиция (или set_board не нужен).")
    return last_board'''

new_resync = '''            print("❌ set_board не удался. Бот запущен в середине чужой партии без истории PGN.")
            print("🛑 Остановка бота. Пожалуйста, запустите бота в начале новой партии.")
            import sys
            sys.exit(1)
    print("✅ Стартовая позиция (или set_board не нужен).")
    return last_board'''

if old_resync in s and "sys.exit(1)" not in s:
    s = s.replace(old_resync, new_resync, 1)
    print("✅ main.py: stop after set_board failure added")
else:
    print("⚠ main.py: exit already present")

# Добавить проверку синхронизации перед ходом бота
old_move = '''    if color_name in bot_colors:
        human_think_delay()
        best = engine.get_best_move(
            game,
            simulations=simulations,
            max_time=max_time,
        )'''

new_move = '''    if color_name in bot_colors:
        # 🛡 ПРОВЕРКА СИНХРОНИЗАЦИИ
        engine_turn = game.get_turn()
        if engine_turn != turn:
            print(f"  ⚠ Десинк: движок считает ход {COLOR_ORDER[engine_turn]}, но таймер показывает {color_name}")
            print("  🛑 Остановка бота. Состояние движка рассинхронизировано.")
            break
        
        human_think_delay()
        best = engine.get_best_move(
            game,
            simulations=simulations,
            max_time=max_time,
        )'''

if old_move in s and "Десинк: движок считает ход" not in s:
    s = s.replace(old_move, new_move, 1)
    print("✅ main.py: sync check before bot move added")
else:
    print("⚠ main.py: sync check already present")

open(p, "w", encoding="utf-8").write(s)

print("\n" + "="*60)
print("✅ ALL FIXES APPLIED")
print("="*60)
print("\nЧто изменилось:")
print("1. state_sync.py: бот пытается восстановить серых королей")
print("2. state_sync.py: расширенная диагностика при провале set_board")
print("3. main.py: бот останавливается после провала set_board")
print("4. main.py: проверка синхронизации перед каждым ходом бота")
print("\nТеперь запусти: python main.py --mode red")