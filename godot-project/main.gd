extends Node2D

const CELL_SIZE = 64
const BOARD_SIZE = 14
const OFFSET = Vector2(30, 30)
const DRAG_THRESHOLD = 5.0
const DEAD_KING_DELAY = 0.45
const PIECE_SCALE_FACTOR = 1.15

# ==========================================================
# Архитектура модели: 30 блоков по 256 каналов (синхронизировано с model.py и lib.rs)
# ==========================================================
const CHANNELS = 256
const RES_BLOCKS = 30
const NUM_PLANES = 73

# Пути к звукам
const SOUNDS = {
	"capture": "res://assets/sounds/capture.wav",
	"castle": "res://assets/sounds/castle.wav",
	"click": "res://assets/sounds/click.wav",
	"game_end": "res://assets/sounds/game-end.wav",
	"illegal": "res://assets/sounds/illegal.wav",
	"move_check": "res://assets/sounds/move-check.wav",
	"move_opponent": "res://assets/sounds/move-opponent.wav",
	"premove": "res://assets/sounds/premove.wav",
	"promote": "res://assets/sounds/promote.wav"
}

# Пути к фигурам (цвет-фигура)
const PIECE_PATHS = {
	"red-pawn": "res://assets/pieces/red-pawn.png",
	"red-rook": "res://assets/pieces/red-rook.png",
	"red-knight": "res://assets/pieces/red-knight.png",
	"red-bishop": "res://assets/pieces/red-bishop.png",
	"red-queen": "res://assets/pieces/red-queen.png",
	"red-king": "res://assets/pieces/red-king.png",
	"blue-pawn": "res://assets/pieces/blue-pawn.png",
	"blue-rook": "res://assets/pieces/blue-rook.png",
	"blue-knight": "res://assets/pieces/blue-knight.png",
	"blue-bishop": "res://assets/pieces/blue-bishop.png",
	"blue-queen": "res://assets/pieces/blue-queen.png",
	"blue-king": "res://assets/pieces/blue-king.png",
	"yellow-pawn": "res://assets/pieces/yellow-pawn.png",
	"yellow-rook": "res://assets/pieces/yellow-rook.png",
	"yellow-knight": "res://assets/pieces/yellow-knight.png",
	"yellow-bishop": "res://assets/pieces/yellow-bishop.png",
	"yellow-queen": "res://assets/pieces/yellow-queen.png",
	"yellow-king": "res://assets/pieces/yellow-king.png",
	"green-pawn": "res://assets/pieces/green-pawn.png",
	"green-rook": "res://assets/pieces/green-rook.png",
	"green-knight": "res://assets/pieces/green-knight.png",
	"green-bishop": "res://assets/pieces/green-bishop.png",
	"green-queen": "res://assets/pieces/green-queen.png",
	"green-king": "res://assets/pieces/green-king.png"
}

# Настройки сложности бота (количество симуляций)
const BOT_DIFFICULTIES = [100, 400, 1600]
const BOT_DIFFICULTY_NAMES = ["Легко (100)", "Средне (400)", "Сложно (1600)"]

var piece_textures = {}
var sound_players = {}

# ==========================================================
# Узел дерева ходов
# ==========================================================
class MoveNode:
	var move: Dictionary = {}
	var notation: String = ""
	var parent: Variant = null
	var children: Array = []

# ==========================================================
# OverlayLayer
# ==========================================================
class OverlayLayer extends Node2D:
	var items: Array = []
	var temp_arrow: Dictionary = {}
	var needs_redraw: bool = false

	func clear() -> void:
		items.clear()
		temp_arrow = {}
		mark_redraw()

	func mark_redraw() -> void:
		if not needs_redraw:
			needs_redraw = true
			queue_redraw()

	func add_arrow(from: Vector2, to: Vector2, col: Color, width: float = 8.0, outline: bool = true) -> void:
		items.append({"t": "arrow", "from": from, "to": to, "col": col, "w": width, "outline": outline})
		mark_redraw()

	func add_circle(pos: Vector2, radius: float, col: Color, width: float = 4.0, filled: bool = false) -> void:
		items.append({"t": "circle", "pos": pos, "r": radius, "col": col, "w": width, "filled": filled})
		mark_redraw()

	func add_rect(pos: Vector2, size: Vector2, col: Color) -> void:
		items.append({"t": "rect", "pos": pos, "size": size, "col": col})
		mark_redraw()

	func add_text(pos: Vector2, text: String, col: Color, size: int = 14) -> void:
		items.append({"t": "text", "pos": pos, "text": text, "col": col, "size": size})
		mark_redraw()

	func _draw() -> void:
		needs_redraw = false
		for it in items:
			match it["t"]:
				"arrow":
					_draw_arrow(it["from"], it["to"], it["col"], it["w"], it["outline"])
				"circle":
					if it["filled"]:
						draw_circle(it["pos"], it["r"], it["col"])
					else:
						draw_arc(it["pos"], it["r"], 0.0, TAU, 48, it["col"], it["w"], true)
				"rect":
					draw_rect(Rect2(it["pos"], it["size"]), it["col"])
				"text":
					_draw_text(it["pos"], it["text"], it["col"], it["size"])
		if not temp_arrow.is_empty():
			_draw_arrow(
				temp_arrow["from"], temp_arrow["to"],
				temp_arrow.get("col", Color(1, 0.15, 0.15, 0.5)),
				8.0, false
			)

	func _draw_text(pos: Vector2, text: String, col: Color, size: int) -> void:
		var font := ThemeDB.fallback_font
		for off in [Vector2(-1, 0), Vector2(1, 0), Vector2(0, -1), Vector2(0, 1)]:
			draw_string(font, pos + off, text, HORIZONTAL_ALIGNMENT_LEFT, -1, size, Color(0, 0, 0, 0.75))
		draw_string(font, pos, text, HORIZONTAL_ALIGNMENT_LEFT, -1, size, col)

	func _draw_arrow(from: Vector2, to: Vector2, col: Color, w: float, outline: bool) -> void:
		if from.distance_to(to) < 4.0:
			return
		var dir := (to - from).normalized()
		var perp := Vector2(-dir.y, dir.x)
		var head_len := clampf(w * 2.6, 14.0, 26.0)
		var line_end := to - dir * head_len * 0.55
		draw_line(from, line_end, col, w, true)
		var p1 := to
		var p2 := to - dir * head_len + perp * w * 1.15
		var p3 := to - dir * head_len - perp * w * 1.15
		draw_polygon(PackedVector2Array([p1, p2, p3]), [col])

# ==========================================================
# Переменные сцены
# ==========================================================
var analyzer: ChessAnalyzer
var selected_cell: Vector2i = -Vector2i.ONE
var board_data: Array = []
var board_rotation: int = 0
var bots_enabled = [false, false, false, false]
var bot_task_active = false
var analyzer_enabled = false
var dead_king_timer = 0.0
var active_bot_color = 0
var low_eval_counts = [0, 0, 0, 0]
var auto_resign_check: CheckBox

# Режим Teams (2v2)
var teams_mode: bool = false
var team_red_yellow: bool = true  # true = red+yellow vs blue+green, false = red+blue vs yellow+green

# Индикатор "бот думает"
var bot_thinking_label: Label
var bot_progress_bar: ProgressBar
var bot_difficulty_index: int = 1  # 0=100, 1=400, 2=1600

# Таблица результатов
var results_dialog: AcceptDialog
var results_table: VBoxContainer

@onready var board_layer = $Board
@onready var pieces_layer = $Pieces
@onready var highlights_root = $Highlights

var last_move_layer: OverlayLayer
var player_arrows_layer: OverlayLayer
var analysis_arrows_layer: OverlayLayer
var cognitive_layer: OverlayLayer
var highlights_layer: OverlayLayer
var premove_layer: OverlayLayer
var player_arrow_cells: Array = []
var premoves = {}
var view_stack: Array = []

var root_node: MoveNode = null
var current_node: MoveNode = null
var pgn_text_edit: TextEdit
var load_button: Button
var prev_button: Button
var next_button: Button
var move_label: Label
var move_list_container: VBoxContainer
var rotate_button: Button
var analyze_button: Button
var score_labels = []
var eval_bar_segments = []
var bot_time_spinbox: SpinBox
var analyzer_lines_spinbox: SpinBox
var show_threats_check: CheckBox
var show_targets_check: CheckBox
var show_develop_check: CheckBox
var show_plans_check: CheckBox
var piece_sprites = []
var drag_sprite: Sprite2D
var analysis_task_active = false
var right_dragging = false
var right_drag_start_pos = Vector2.ZERO
var arrow_start_cell = Vector2i.ZERO
var ai_thoughts_label: Label
var dragging = false
var dragged_cell = Vector2i.ZERO
var drag_start_pos = Vector2.ZERO

# Кэш клеток доски (создаются один раз)
var board_cells_created = false

# ==========================================================
# _ready
# ==========================================================
func _ready():
	analyzer = ChessAnalyzer.new()
	add_child(analyzer)

	var global_path = ProjectSettings.globalize_path("res://model.onnx")
	if analyzer.load_neural_model(global_path):
		print("Нейросеть загружена в Godot!")
	else:
		printerr("Не удалось загрузить нейросеть.")

	last_move_layer = OverlayLayer.new()
	add_child(last_move_layer)
	player_arrows_layer = OverlayLayer.new()
	add_child(player_arrows_layer)
	analysis_arrows_layer = OverlayLayer.new()
	add_child(analysis_arrows_layer)
	cognitive_layer = OverlayLayer.new()
	add_child(cognitive_layer)
	premove_layer = OverlayLayer.new()
	add_child(premove_layer)
	highlights_layer = OverlayLayer.new()
	highlights_root.add_child(highlights_layer)

	drag_sprite = Sprite2D.new()
	drag_sprite.centered = true
	drag_sprite.z_index = 100
	drag_sprite.visible = false
	add_child(drag_sprite)

	_create_ui()
	_create_corner_buttons()

	for i in range(128):
		var sprite = Sprite2D.new()
		sprite.centered = true
		sprite.visible = false
		pieces_layer.add_child(sprite)
		piece_sprites.append(sprite)

	root_node = MoveNode.new()
	current_node = root_node

	draw_board()
	update_pieces(true)
	update_move_label()
	_update_move_list()
	_update_scores()

# ==========================================================
# UI
# ==========================================================
func _create_ui():
	var board_width = BOARD_SIZE * CELL_SIZE
	var panel_width = board_width / 2

	var scroll_container = ScrollContainer.new()
	scroll_container.size = Vector2(panel_width, board_width)
	scroll_container.position = Vector2(OFFSET.x + board_width + 20, OFFSET.y)
	add_child(scroll_container)

	var vbox = VBoxContainer.new()
	vbox.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	vbox.size_flags_vertical = Control.SIZE_EXPAND_FILL
	vbox.custom_minimum_size = Vector2(panel_width - 20, 0)
	scroll_container.add_child(vbox)

	var pgn_label = Label.new()
	pgn_label.text = "PGN4:"
	vbox.add_child(pgn_label)

	pgn_text_edit = TextEdit.new()
	pgn_text_edit.custom_minimum_size = Vector2(panel_width - 40, 200)
	vbox.add_child(pgn_text_edit)

	load_button = Button.new()
	load_button.text = "Загрузить PGN"
	load_button.pressed.connect(_on_load_pressed)
	vbox.add_child(load_button)

	rotate_button = Button.new()
	rotate_button.text = "Повернуть доску"
	rotate_button.pressed.connect(_on_rotate_pressed)
	vbox.add_child(rotate_button)

	var copy_pgn_button = Button.new()
	copy_pgn_button.text = "Копировать PGN"
	copy_pgn_button.pressed.connect(_on_copy_pgn_pressed)
	vbox.add_child(copy_pgn_button)

	# Eval bar
	var eval_label = Label.new()
	eval_label.text = "Оценка позиции:"
	vbox.add_child(eval_label)

	var eval_hbox = HBoxContainer.new()
	eval_hbox.custom_minimum_size = Vector2(panel_width - 40, 30)
	for i in range(4):
		var rect = ColorRect.new()
		rect.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		rect.color = get_color_by_code(i + 1)
		eval_hbox.add_child(rect)
		eval_bar_segments.append(rect)
	vbox.add_child(eval_hbox)

	var eval_nums_hbox = HBoxContainer.new()
	eval_nums_hbox.custom_minimum_size = Vector2(panel_width - 40, 20)
	for i in range(4):
		var lbl = Label.new()
		lbl.text = "1.0"
		lbl.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
		lbl.modulate = get_color_by_code(i + 1)
		lbl.visible = false
		eval_nums_hbox.add_child(lbl)
		eval_bar_segments.append(lbl)
	vbox.add_child(eval_nums_hbox)

	# Bot settings
	var bot_settings_hbox = HBoxContainer.new()
	var bot_time_label = Label.new()
	bot_time_label.text = "Время бота (мс, 0=без лимита):"
	bot_settings_hbox.add_child(bot_time_label)
	bot_time_spinbox = SpinBox.new()
	bot_time_spinbox.min_value = 0
	bot_time_spinbox.max_value = 60000
	bot_time_spinbox.value = 1500
	bot_time_spinbox.step = 100
	bot_settings_hbox.add_child(bot_time_spinbox)
	vbox.add_child(bot_settings_hbox)

	auto_resign_check = CheckBox.new()
	auto_resign_check.text = "Авто-сдача ботов (3 хода с оценкой < -0.95)"
	auto_resign_check.button_pressed = true
	vbox.add_child(auto_resign_check)

	# Сложность бота
	var bot_diff_hbox = HBoxContainer.new()
	var bot_diff_label = Label.new()
	bot_diff_label.text = "Сложность бота:"
	bot_diff_hbox.add_child(bot_diff_label)
	
	var bot_diff_option = OptionButton.new()
	for i in range(BOT_DIFFICULTIES.size()):
		bot_diff_option.add_item(BOT_DIFFICULTY_NAMES[i], i)
	bot_diff_option.selected = bot_difficulty_index
	bot_diff_option.item_selected.connect(_on_bot_difficulty_selected)
	bot_diff_hbox.add_child(bot_diff_option)
	vbox.add_child(bot_diff_hbox)
	
	# Режим Teams (2v2)
	var teams_check = CheckBox.new()
	teams_check.text = "Режим команд (2v2)"
	teams_check.toggled.connect(_on_teams_mode_toggled)
	vbox.add_child(teams_check)
	
	# Кнопка перезапуска
	var restart_button = Button.new()
	restart_button.text = "Перезапустить партию"
	restart_button.pressed.connect(_on_restart_pressed)
	vbox.add_child(restart_button)

	# Analyzer settings
	var analyzer_settings_hbox = HBoxContainer.new()
	var lines_label = Label.new()
	lines_label.text = "Линий анализа (1-3):"
	analyzer_settings_hbox.add_child(lines_label)
	analyzer_lines_spinbox = SpinBox.new()
	analyzer_lines_spinbox.min_value = 1
	analyzer_lines_spinbox.max_value = 3
	analyzer_lines_spinbox.value = 3
	analyzer_settings_hbox.add_child(analyzer_lines_spinbox)
	vbox.add_child(analyzer_settings_hbox)

	# Cognitive visualization
	var cog_label = Label.new()
	cog_label.text = "Визуализация ИИ:"
	vbox.add_child(cog_label)

	show_threats_check = CheckBox.new()
	show_threats_check.text = "Угрозы (Красный)"
	show_threats_check.toggled.connect(_on_cognitive_toggled)
	vbox.add_child(show_threats_check)

	show_targets_check = CheckBox.new()
	show_targets_check.text = "Цели (Синий)"
	show_targets_check.toggled.connect(_on_cognitive_toggled)
	vbox.add_child(show_targets_check)

	show_develop_check = CheckBox.new()
	show_develop_check.text = "Развитие (Зеленый)"
	show_develop_check.toggled.connect(_on_cognitive_toggled)
	vbox.add_child(show_develop_check)

	show_plans_check = CheckBox.new()
	show_plans_check.text = "Планы атак (Стрелки)"
	show_plans_check.toggled.connect(_on_cognitive_toggled)
	vbox.add_child(show_plans_check)

	# AI thoughts
	var thoughts_label_title = Label.new()
	thoughts_label_title.text = "Мысли ИИ (Почему этот ход?):"
	thoughts_label_title.add_theme_color_override("font_color", Color.BLACK)
	vbox.add_child(thoughts_label_title)

	ai_thoughts_label = Label.new()
	ai_thoughts_label.text = "Включите ботов или сделайте ход."
	ai_thoughts_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	ai_thoughts_label.custom_minimum_size = Vector2(panel_width - 40, 100)
	ai_thoughts_label.add_theme_color_override("font_color", Color(0.1, 0.1, 0.1))
	ai_thoughts_label.add_theme_color_override("font_outline_color", Color.WHITE)
	ai_thoughts_label.add_theme_constant_override("outline_size", 2)
	vbox.add_child(ai_thoughts_label)

	analyze_button = Button.new()
	analyze_button.text = "Включить анализатор"
	analyze_button.toggle_mode = true
	analyze_button.toggled.connect(_on_analyze_toggled)
	vbox.add_child(analyze_button)

	# Navigation
	var hbox_nav = HBoxContainer.new()
	prev_button = Button.new()
	prev_button.text = "←"
	prev_button.pressed.connect(_on_prev_pressed)
	hbox_nav.add_child(prev_button)

	move_label = Label.new()
	move_label.text = "Ход: 0"
	move_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	move_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	hbox_nav.add_child(move_label)

	next_button = Button.new()
	next_button.text = "→"
	next_button.pressed.connect(_on_next_pressed)
	hbox_nav.add_child(next_button)
	vbox.add_child(hbox_nav)

	var moves_label = Label.new()
	moves_label.text = "Ходы:"
	vbox.add_child(moves_label)

	move_list_container = VBoxContainer.new()
	vbox.add_child(move_list_container)


func _create_corner_buttons():
	var corners = [Vector2i(2, 2), Vector2i(2, 13), Vector2i(13, 13), Vector2i(13, 2)]
	var names = ["Red Bot", "Blue Bot", "Yellow Bot", "Green Bot"]
	var colors = [Color.RED, Color.BLUE, Color.YELLOW, Color.GREEN]
	for i in range(4):
		var btn = Button.new()
		btn.name = "BotButton_%d" % i
		btn.text = names[i]
		btn.size = Vector2(120, 64)
		var pos = cell_to_pixel(corners[i].x, corners[i].y)
		btn.position = pos + Vector2(CELL_SIZE / 2 - 60, CELL_SIZE / 2 - 32)
		btn.modulate = colors[i]
		btn.add_theme_color_override("font_color", Color.BLACK)
		btn.toggle_mode = true
		btn.toggled.connect(func(pressed): _on_bot_toggled(i, pressed))
		add_child(btn)

		var score_lbl = Label.new()
		score_lbl.name = "ScoreLabel_%d" % i
		score_lbl.text = "0"
		score_lbl.add_theme_font_size_override("font_size", 24)
		score_lbl.add_theme_color_override("font_color", Color.BLACK)
		score_lbl.add_theme_color_override("font_outline_color", Color.WHITE)
		score_lbl.add_theme_constant_override("outline_size", 4)
		score_lbl.position = pos + Vector2(CELL_SIZE / 2 - 15, CELL_SIZE / 2 + 32)
		add_child(score_lbl)
		score_labels.append(score_lbl)


func _on_bot_toggled(bot_idx: int, pressed: bool):
	bots_enabled[bot_idx] = pressed
	if not pressed:
		bot_task_active = false
		analyzer.cancel_search()


func _disable_bot(bot_idx: int):
	"""Отключает бота и снимает кнопку."""
	if bot_idx >= 0 and bot_idx < 4:
		bots_enabled[bot_idx] = false
		var btn = get_node_or_null("BotButton_%d" % bot_idx)
		if btn:
			btn.set_pressed_no_signal(false)


func _on_bot_difficulty_selected(index: int):
	bot_difficulty_index = index


func _on_teams_mode_toggled(pressed: bool):
	teams_mode = pressed


func _on_restart_pressed():
	# Перезапуск партии
	analyzer.go_to_move(0)
	root_node = MoveNode.new()
	current_node = root_node
	view_stack.clear()
	premoves.clear()
	player_arrow_cells.clear()
	low_eval_counts = [0, 0, 0, 0]
	dead_king_timer = 0.0
	
	# Сброс состояния доски
	analyzer.reset_board()
	
	# Обновление UI
	update_pieces(true)
	update_move_label()
	_update_move_list()
	_update_scores()
	last_move_layer.clear()
	player_arrows_layer.clear()
	analysis_arrows_layer.clear()
	cognitive_layer.clear()
	premove_layer.clear()
	
	play_click_sound()

# ==========================================================
# Звук
# ==========================================================
var audio_pool: Array = []
const MAX_AUDIO_PLAYERS = 16

func _play_sound(sound_name: String, blocking: bool = false) -> void:
	if not SOUNDS.has(sound_name):
		push_warning("Звук '%s' не найден" % sound_name)
		return
	
	var path = SOUNDS[sound_name]
	if not ResourceLoader.exists(path):
		push_warning("Файл звука '%s' не существует" % path)
		return
	
	# Очищаем пул от завершённых плееров
	for i in range(audio_pool.size() - 1, -1, -1):
		if not is_instance_valid(audio_pool[i]) or not audio_pool[i].playing:
			audio_pool.remove_at(i)
	
	# Используем пул готовых плееров или создаём новый
	var player: AudioStreamPlayer = null
	if audio_pool.size() < MAX_AUDIO_PLAYERS:
		player = AudioStreamPlayer.new()
		player.bus = "Master"
		add_child(player)
		audio_pool.append(player)
	else:
		# Берём самый старый неиграющий плеер
		for p in audio_pool:
			if is_instance_valid(p) and not p.playing:
				player = p
				break
		if not player:
			# Все играют - пропускаем звук (лучше чем обрезать)
			return
	
	player.stream = load(path)
	player.volume_db = 0.0
	player.play()
	if blocking:
		await player.finished


func play_move_sound():
	_play_sound("move_opponent")


func play_capture_sound():
	_play_sound("capture")


func play_castle_sound():
	_play_sound("castle")


func play_click_sound():
	_play_sound("click")


func play_game_end_sound():
	_play_sound("game_end")


func play_illegal_sound():
	_play_sound("illegal")


func play_check_sound():
	_play_sound("move_check")


func play_premove_sound():
	_play_sound("premove")


func play_promote_sound():
	_play_sound("promote")

# ==========================================================
# Дерево ходов и навигация
# ==========================================================
func get_path_to_root(target_node: MoveNode) -> Array:
	var path = []
	var n = target_node
	while n != root_node and n != null:
		path.push_front(n)
		n = n.parent
	return path


func replay_to_current():
	var path = get_path_to_root(current_node)
	analyzer.go_to_move(0)
	for node in path:
		analyzer.make_move(node.move["fx"], node.move["fy"], node.move["tx"], node.move["ty"])
	update_pieces(true)
	_update_scores()
	if analyzer_enabled:
		_update_analysis_arrow()


func _get_last_move_notation() -> String:
	"""Ищет последний ход с координатами (пропускает skipped)."""
	var moves = analyzer.get_move_list()
	for i in range(moves.size() - 1, -1, -1):
		var m = moves[i]
		if m.has("from_x"):
			return m.get("notation", "?")
	return "?"


func add_move_to_tree(move_dict: Dictionary):
	var existing_child = null
	for child in current_node.children:
		if (child.move["fx"] == move_dict["fx"] and child.move["fy"] == move_dict["fy"]
				and child.move["tx"] == move_dict["tx"] and child.move["ty"] == move_dict["ty"]):
			existing_child = child
			break
	if existing_child:
		current_node = existing_child
	else:
		var new_node = MoveNode.new()
		new_node.move = move_dict
		new_node.parent = current_node
		new_node.notation = _get_last_move_notation()
		current_node.children.append(new_node)
		current_node = new_node


func _on_move_clicked(node: MoveNode):
	current_node = node
	replay_to_current()
	_update_move_list()
	update_move_label()
	play_move_sound()


func _on_branch_clicked(node: MoveNode):
	view_stack.push_back(node)
	_update_move_list()


func _on_prev_pressed():
	if current_node != root_node:
		current_node = current_node.parent
		replay_to_current()
		_update_move_list()
		update_move_label()
		play_move_sound()


func _on_next_pressed():
	if current_node.children.size() > 0:
		current_node = current_node.children[0]
		replay_to_current()
		_update_move_list()
		update_move_label()
		play_move_sound()


func update_move_label():
	move_label.text = "Ход: %d" % get_path_to_root(current_node).size()


func get_absolute_index(node: MoveNode) -> int:
	var count = 0
	var n = node
	while n != root_node and n != null:
		count += 1
		n = n.parent
	return count


func get_move_color_by_idx(idx: int) -> Color:
	match idx % 4:
		0: return Color.RED
		1: return Color.BLUE
		2: return Color(1.0, 0.84, 0.0)
		3: return Color.GREEN
	return Color.BLACK


func _update_move_list():
	for child in move_list_container.get_children():
		child.queue_free()

	var current_view_node = root_node.children[0] if root_node.children.size() > 0 else null
	if view_stack.size() > 0:
		current_view_node = view_stack.back()

	if current_view_node == null:
		var empty_label = Label.new()
		empty_label.text = "Нет ходов"
		move_list_container.add_child(empty_label)
		return

	if view_stack.size() > 0:
		var back_btn = Button.new()
		var parent_node = current_view_node.parent
		back_btn.text = "← Назад к ходу: %s" % parent_node.notation
		back_btn.add_theme_font_size_override("font_size", 14)
		back_btn.add_theme_color_override("font_color", Color(0.1, 0.5, 0.8))
		back_btn.pressed.connect(func():
			view_stack.pop_back()
			_update_move_list()
		)
		move_list_container.add_child(back_btn)

	_render_line(current_view_node)


func _render_line(start_node: MoveNode):
	var hbox = HBoxContainer.new()
	move_list_container.add_child(hbox)
	var n = start_node
	var start_abs_idx = get_absolute_index(n)

	while n != null:
		var abs_idx = get_absolute_index(n)
		if abs_idx != start_abs_idx and abs_idx % 4 == 1:
			hbox = HBoxContainer.new()
			move_list_container.add_child(hbox)

		if abs_idx % 4 == 1:
			var num_label = Label.new()
			num_label.text = str((abs_idx - 1) / 4 + 1) + "."
			num_label.add_theme_font_size_override("font_size", 14)
			num_label.add_theme_color_override("font_color", Color.BLACK)
			hbox.add_child(num_label)

		var btn = Button.new()
		btn.text = n.notation
		btn.flat = true
		btn.add_theme_font_size_override("font_size", 14)
		btn.add_theme_color_override("font_color", get_move_color_by_idx(abs_idx - 1))
		if n == current_node:
			btn.modulate = Color(1, 0.2, 0.2, 1)
		btn.pressed.connect(_on_move_clicked.bind(n))
		hbox.add_child(btn)

		if n.children.size() > 1:
			for j in range(1, n.children.size()):
				var var_node = n.children[j]
				var branch_btn = Button.new()
				branch_btn.text = "[Ветка %d]" % j
				branch_btn.flat = true
				branch_btn.modulate = Color(0.8, 0.6, 0.1)
				branch_btn.add_theme_font_size_override("font_size", 14)
				branch_btn.pressed.connect(_on_branch_clicked.bind(var_node))
				hbox.add_child(branch_btn)

		n = n.children[0] if n.children.size() > 0 else null


func _on_copy_pgn_pressed():
	var pgn = ""
	var n = root_node.children[0] if root_node.children.size() > 0 else null
	var round_num = 1
	var i = 0
	while n != null:
		if i % 4 == 0:
			pgn += str(round_num) + ". "
		pgn += n.notation + " "
		if i % 4 == 3:
			pgn += "\n"
			round_num += 1
		n = n.children[0] if n.children.size() > 0 else null
		i += 1
	DisplayServer.clipboard_set(pgn)
	print("PGN скопирован в буфер обмена!")

# ==========================================================
# Главный цикл
# ==========================================================
func _process(delta):
	# 1. Бот думает в фоне
	if bot_task_active:
		ai_thoughts_label.text = "🤖 Бот думает... (пожалуйста, подождите)"
		if analyzer.is_bot_done():
			bot_task_active = false
			var result = analyzer.get_bot_result()
			if result.has("fx"):
				var fx = int(result["fx"])
				var fy = int(result["fy"])
				var tx = int(result["tx"])
				var ty = int(result["ty"])
				var bot_value = float(result.get("value", 0.0))
				if analyzer.make_move(fx, fy, tx, ty):
					add_move_to_tree({"fx": fx, "fy": fy, "tx": tx, "ty": ty})
					_refresh_after_move()
					play_move_sound()
					_check_auto_resign(bot_value)
				else:
					printerr("Бот попытался сделать нелегальный ход.")
					_disable_bot(active_bot_color - 1)
			else:
				printerr("Бот отменён или нейросеть не вернула ход.")
				_disable_bot(active_bot_color - 1)
		return

	# 2. Анализатор (асинхронный) — ИСПРАВЛЕНО: результат через is_analysis_done
	if analysis_task_active:
		if analyzer.is_analysis_done():
			analysis_task_active = false
			var result = analyzer.get_analysis_result()
			_draw_analysis_arrows(result)
		return

	# 3. Dead king walking
	if analyzer.is_turn_resigned() and not analyzer.is_game_over():
		dead_king_timer += delta
		if dead_king_timer >= DEAD_KING_DELAY:
			dead_king_timer = 0.0
			if analyzer.play_dead_king_move():
				var moves = analyzer.get_move_list()
				if moves.size() > 0:
					var lm = moves[moves.size() - 1]
					if lm.has("from_x"):
						add_move_to_tree({
							"fx": int(lm["from_x"]),
							"fy": int(lm["from_y"]),
							"tx": int(lm["to_x"]),
							"ty": int(lm["to_y"]),
						})
				_refresh_after_move()
				play_move_sound()
			else:
				_refresh_after_move()
		return
	else:
		dead_king_timer = 0.0

	# Восстанавливаем текст
	if not bot_task_active and ai_thoughts_label.text.begins_with("🤖"):
		ai_thoughts_label.text = "Включите ботов или сделайте ход."

	var turn_code = analyzer.get_current_turn_code()

	# Авто-выполнение премува
	if (turn_code >= 1 and turn_code <= 4
			and not bots_enabled[turn_code - 1]
			and not analyzer.is_game_over()
			and not bot_task_active
			and not analysis_task_active):
		if premoves.has(turn_code):
			var pm = premoves[turn_code]
			var legal = analyzer.get_legal_moves()
			var is_valid = false
			for m in legal:
				if (m.from_x == pm.from.x and m.from_y == pm.from.y
						and m.to_x == pm.to.x and m.to_y == pm.to.y):
					is_valid = true
					break
			if is_valid:
				if analyzer.make_move(pm.from.x, pm.from.y, pm.to.x, pm.to.y):
					add_move_to_tree({"fx": pm.from.x, "fy": pm.from.y, "tx": pm.to.x, "ty": pm.to.y})
					_refresh_after_move()
					play_move_sound()
				else:
					print("Premove invalid, canceled.")
			premoves.erase(turn_code)
			_draw_premoves()

	# Запуск бота
	if (turn_code >= 1 and turn_code <= 4
			and bots_enabled[turn_code - 1]
			and not analyzer.is_game_over()):
		active_bot_color = turn_code
		bot_task_active = true
		var time_ms = int(bot_time_spinbox.value)
		analyzer.request_neural_move(time_ms)


func _check_auto_resign(bot_value: float):
	if not auto_resign_check.button_pressed:
		return
	if active_bot_color < 1 or active_bot_color > 4:
		return
	var idx = active_bot_color - 1
	if bot_value < -0.95:
		low_eval_counts[idx] += 1
	else:
		low_eval_counts[idx] = 0
	if low_eval_counts[idx] >= 3:
		analyzer.resign_player(active_bot_color)
		low_eval_counts[idx] = 0
		var names = ["Red", "Blue", "Yellow", "Green"]
		print("🏳️ Бот %s сдался (оценка %.2f)" % [names[idx], bot_value])


func _refresh_after_move(instant: bool = false):
	# Определяем тип хода для звука — только один звук за ход
	var last_moves = analyzer.get_last_moves_per_color()
	for lm in last_moves:
		if lm.has("is_capture") and lm["is_capture"]:
			play_capture_sound()
		elif lm.has("is_castle") and lm["is_castle"]:
			play_castle_sound()
		elif lm.has("is_promotion") and lm["is_promotion"]:
			play_promote_sound()
		elif lm.has("is_check") and lm["is_check"]:
			play_check_sound()
		else:
			play_move_sound()
		break  # Только первый ход (чтобы не было двойного срабатывания)
	
	update_pieces(instant)
	_update_move_list()
	update_move_label()
	_update_scores()
	if analyzer_enabled:
		_update_analysis_arrow()

# ==========================================================
# Анализатор
# ==========================================================
func _on_analyze_toggled(pressed: bool):
	analyzer_enabled = pressed
	if pressed:
		analyze_button.text = "Выключить анализатор"
		_update_analysis_arrow()
	else:
		analyze_button.text = "Включить анализатор"
		analyzer.cancel_search()
		analysis_arrows_layer.clear()


func _update_analysis_arrow():
	"""ИСПРАВЛЕНО: запускает асинхронный анализ, результат придёт в _process()."""
	analysis_arrows_layer.clear()
	if bot_task_active or analysis_task_active:
		return
	var lines = int(analyzer_lines_spinbox.value)
	analysis_task_active = true
	analyzer.request_analysis(lines)
	# НЕ вызываем get_analysis_result() здесь!
	# Результат обрабатывается в _process() через is_analysis_done().


func _draw_analysis_arrows(result) -> void:
	analysis_arrows_layer.clear()
	if result == null:
		return
	var colors = [
		Color(0.1, 0.9, 0.2, 0.8),
		Color(1.0, 0.85, 0.1, 0.75),
		Color(1.0, 0.55, 0.1, 0.7),
	]
	for i in range(result.size()):
		var res_move = result[i]
		var from = Vector2i(int(res_move["fx"]), int(res_move["fy"]))
		var to = Vector2i(int(res_move["tx"]), int(res_move["ty"]))
		var col = colors[i % colors.size()]
		analysis_arrows_layer.add_arrow(cell_center(from), cell_center(to), col, 9.0, false)
		analysis_arrows_layer.add_text(
			cell_center(to) + Vector2(10, -14),
			"%.1f%%" % (float(res_move["score"]) * 100.0),
			col, 13
		)


func _update_scores():
	var scores = analyzer.get_scores()
	for i in range(4):
		score_labels[i].text = str(scores[i])

	var evals = analyzer.get_eval_points()
	var total = 0.0
	for i in range(4):
		total += abs(evals[i])
	if total == 0:
		for i in range(4):
			eval_bar_segments[i].custom_minimum_size.x = 10
	else:
		for i in range(4):
			eval_bar_segments[i].custom_minimum_size.x = abs(float(evals[i])) / total * 200
			eval_bar_segments[i + 4].text = "%.2f" % evals[i]
			eval_bar_segments[i + 4].visible = analyzer_enabled

# ==========================================================
# Ввод
# ==========================================================
func _unhandled_input(event):
	if event is InputEventKey and event.pressed:
		if event.keycode == KEY_LEFT:
			_on_prev_pressed()
		elif event.keycode == KEY_RIGHT:
			_on_next_pressed()


func _input(event):
	# Правая кнопка — стрелки
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_RIGHT:
		if event.pressed:
			var cell = pixel_to_cell(get_global_mouse_position())
			if cell != -Vector2i.ONE:
				right_dragging = true
				right_drag_start_pos = get_global_mouse_position()
				arrow_start_cell = cell
				player_arrows_layer.temp_arrow = {
					"from": cell_center(cell),
					"to": get_global_mouse_position(),
					"col": Color(1, 0.15, 0.15, 0.5),
				}
				player_arrows_layer.mark_redraw()
		else:
			if right_dragging:
				right_dragging = false
				var end_pos = get_global_mouse_position()
				player_arrows_layer.temp_arrow = {}
				if right_drag_start_pos.distance_to(end_pos) >= 10.0:
					var cell = pixel_to_cell(end_pos)
					if cell != -Vector2i.ONE and cell != arrow_start_cell:
						player_arrow_cells.append([arrow_start_cell, cell])
						rebuild_player_arrows()
				else:
					player_arrows_layer.mark_redraw()

	if event is InputEventMouseMotion and right_dragging:
		player_arrows_layer.temp_arrow["to"] = get_global_mouse_position()
		player_arrows_layer.mark_redraw()

	# Drag & drop фигур - мгновенное появление при начале движения
	if event is InputEventMouseMotion and dragging and not drag_sprite.visible:
		if event.position.distance_to(drag_start_pos) > DRAG_THRESHOLD:
			var orig_sprite = get_sprite_at(dragged_cell)
			if orig_sprite:
				orig_sprite.modulate.a = 0.5
			drag_sprite.visible = true
			drag_sprite.position = get_global_mouse_position()

	# Обновление позиции перетаскиваемой фигуры без задержек
	if event is InputEventMouseMotion and drag_sprite.visible:
		drag_sprite.position = get_global_mouse_position()
		drag_sprite.get_viewport().set_input_as_handled()

	# Левая кнопка
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT:
		if event.pressed:
			if bot_task_active:
				return
			if not player_arrow_cells.is_empty():
				player_arrow_cells.clear()
				player_arrows_layer.clear()

			var cell = pixel_to_cell(get_global_mouse_position())
			if cell == -Vector2i.ONE:
				selected_cell = -Vector2i.ONE
				clear_highlights()
				return

			var code: int = board_data[cell.y - 1][cell.x - 1]

			if selected_cell != -Vector2i.ONE:
				if attempt_move(selected_cell, cell, false):
					return

			if code != 0:
				var piece_color_code = (code & 0x7F) / 10
				if premoves.has(piece_color_code) and premoves[piece_color_code]["from"] == cell:
					premoves.erase(piece_color_code)
					_draw_premoves()
					selected_cell = -Vector2i.ONE
					clear_highlights()
					return

				if selected_cell == cell:
				# Повторный клик по той же фигуре - сбрасываем выделение
					selected_cell = -Vector2i.ONE
					clear_highlights()
					update_pieces(true)
					return

				# Выделяем новую фигуру
				selected_cell = cell
				if piece_color_code == analyzer.get_current_turn_code():
					show_possible_moves(cell)

				dragging = true
				dragged_cell = cell
				drag_start_pos = event.position
				var tex = get_piece_texture(code)
				if tex:
					drag_sprite.texture = tex
					var tex_size = tex.get_size()
					if tex_size.x > 0:
						var sc = (CELL_SIZE * 0.9 * PIECE_SCALE_FACTOR) / tex_size.x
						drag_sprite.scale = Vector2(sc, sc)
			else:
				selected_cell = -Vector2i.ONE
				clear_highlights()
			update_pieces(true)

		elif not event.pressed:
			if dragging:
				dragging = false
				if drag_sprite.visible:
					drag_sprite.visible = false
					var target_cell = pixel_to_cell(get_global_mouse_position())
					if target_cell != dragged_cell and target_cell != -Vector2i.ONE:
						if not attempt_move(dragged_cell, target_cell, true):
							update_pieces(true)
					else:
						update_pieces(true)
                                # Не сбрасываем selected_cell и highlights здесь - это происходит при следующем нажатии


func is_shape_valid(code: int, from_cell: Vector2i, to_cell: Vector2i) -> bool:
	var dx = abs(to_cell.x - from_cell.x)
	var dy = abs(to_cell.y - from_cell.y)
	var raw_code = code & 0x7F
	var pt = raw_code % 10
	var cc = raw_code / 10
	if dx == 0 and dy == 0:
		return false
	match pt:
		2: return (dx == 1 and dy == 2) or (dx == 2 and dy == 1)
		3: return dx == dy
		4: return dx == 0 or dy == 0
		5: return dx == dy or dx == 0 or dy == 0
		6: return dx <= 1 and dy <= 1
		1:
			var fx = from_cell.x; var fy = from_cell.y
			var tx = to_cell.x; var ty = to_cell.y
			var is_start = false
			match cc:
				1:
					is_start = (fy == 2)
					return ((tx == fx and ty == fy + 1)
						or (is_start and tx == fx and ty == fy + 2)
						or (abs(tx - fx) == 1 and ty == fy + 1))
				2:
					is_start = (fx == 2)
					return ((ty == fy and tx == fx + 1)
						or (is_start and ty == fy and tx == fx + 2)
						or (abs(ty - fy) == 1 and tx == fx + 1))
				3:
					is_start = (fy == 13)
					return ((tx == fx and ty == fy - 1)
						or (is_start and tx == fx and ty == fy - 2)
						or (abs(tx - fx) == 1 and ty == fy - 1))
				4:
					is_start = (fx == 13)
					return ((ty == fy and tx == fx - 1)
						or (is_start and ty == fy and tx == fx - 2)
						or (abs(ty - fy) == 1 and tx == fx - 1))
	return false


func attempt_move(from_cell: Vector2i, to_cell: Vector2i, instant: bool = false) -> bool:
	if from_cell == -Vector2i.ONE or to_cell == -Vector2i.ONE:
		return false
	var code = board_data[from_cell.y - 1][from_cell.x - 1]
	if code == 0:
		return false
	var piece_color_code = (code & 0x7F) / 10
	var current_turn = analyzer.get_current_turn_code()

	if piece_color_code == current_turn:
		var success = analyzer.make_move(from_cell.x, from_cell.y, to_cell.x, to_cell.y)
		if success:
			add_move_to_tree({"fx": from_cell.x, "fy": from_cell.y, "tx": to_cell.x, "ty": to_cell.y})
			clear_highlights()
			selected_cell = -Vector2i.ONE
			_refresh_after_move(instant)
			play_move_sound()
			return true
		else:
			selected_cell = -Vector2i.ONE
			clear_highlights()
			return false
	else:
		# Premove
		if not is_shape_valid(code, from_cell, to_cell):
			return false
		premoves[piece_color_code] = {"from": from_cell, "to": to_cell}
		selected_cell = -Vector2i.ONE
		clear_highlights()
		_draw_premoves()
		update_pieces(true)
		return true


func _draw_premoves():
	premove_layer.clear()
	for color_key in premoves.keys():
		var pm = premoves[color_key]
		var col = get_color_by_code(color_key)
		col.a = 0.6
		premove_layer.add_arrow(cell_center(pm.from), cell_center(pm.to), col, 6.0, false)
		premove_layer.add_circle(cell_center(pm.to), CELL_SIZE * 0.2, col, 4.0)


func get_sprite_at(cell: Vector2i) -> Sprite2D:
	var target_pos = cell_center(cell)
	for sprite in piece_sprites:
		if sprite.visible and sprite.position.distance_to(target_pos) < 2.0:
			return sprite
	return null


func rebuild_player_arrows() -> void:
	player_arrows_layer.clear()
	for pair in player_arrow_cells:
		player_arrows_layer.add_arrow(
			cell_center(pair[0]), cell_center(pair[1]),
			Color(1, 0.15, 0.15, 0.5), 8.0, false
		)

# ==========================================================
# Доска и координаты
# ==========================================================
func _on_rotate_pressed():
	board_rotation = (board_rotation + 90) % 360
	# Пересоздаём доску только при повороте
	for child in board_layer.get_children():
		child.queue_free()
	board_cells_created = false
	draw_board()
	update_pieces(true)
	rebuild_player_arrows()
	_draw_premoves()
	clear_highlights()
	selected_cell = -Vector2i.ONE
	if analyzer_enabled:
		_update_analysis_arrow()


func rotate_coords(x: int, y: int, angle: int) -> Vector2i:
	match ((angle % 360) + 360) % 360:
		90:  return Vector2i(15 - y, x)
		180: return Vector2i(15 - x, 15 - y)
		270: return Vector2i(y, 15 - x)
		_:   return Vector2i(x, y)


func cell_to_pixel(x: int, y: int) -> Vector2:
	var rotated = rotate_coords(x, y, board_rotation)
	return Vector2((rotated.x - 1) * CELL_SIZE, (14 - rotated.y) * CELL_SIZE) + OFFSET


func cell_center(cell: Vector2i) -> Vector2:
	return cell_to_pixel(cell.x, cell.y) + Vector2(CELL_SIZE / 2.0, CELL_SIZE / 2.0)


func pixel_to_cell(global_pos: Vector2) -> Vector2i:
	var local = global_pos - OFFSET
	var rx = int(local.x / CELL_SIZE) + 1
	var ry = 14 - int(local.y / CELL_SIZE)
	var rotated = rotate_coords(rx, ry, -board_rotation)
	if is_board_cell(rotated.x, rotated.y):
		return rotated
	return -Vector2i.ONE


func draw_board():
	"""Создаёт клетки доски. Вызывается только в _ready() и при повороте."""
	if board_cells_created:
		return
	for y in range(1, BOARD_SIZE + 1):
		for x in range(1, BOARD_SIZE + 1):
			if not is_board_cell(x, y):
				continue
			var rect = ColorRect.new()
			rect.size = Vector2(CELL_SIZE, CELL_SIZE)
			rect.position = cell_to_pixel(x, y)
			rect.color = Color(0.9, 0.9, 0.9) if (x + y) % 2 == 0 else Color(0.7, 0.7, 0.7)
			board_layer.add_child(rect)
	board_cells_created = true


func is_board_cell(x: int, y: int) -> bool:
	if x < 1 or x > 14 or y < 1 or y > 14:
		return false
	return not ((x <= 3 or x >= 12) and (y <= 3 or y >= 12))

# ==========================================================
# Фигуры
# ==========================================================
func get_piece_texture(code: int) -> Texture2D:
	var promoted = false
	if code & 0x80 != 0:
		promoted = true
	code = code & 0x7F
	var color_code = code / 10
	var piece_code = code % 10

	var color_name = ""
	match color_code:
		1: color_name = "red"
		2: color_name = "blue"
		3: color_name = "yellow"
		4: color_name = "green"

	var piece_name = ""
	match piece_code:
		1: piece_name = "pawn"
		2: piece_name = "knight"
		3: piece_name = "bishop"
		4: piece_name = "rook"
		5: piece_name = "queen"
		6: piece_name = "king"

	var key = "%s-%s" % [color_name, piece_name]
	if not PIECE_PATHS.has(key):
		printerr("Путь к фигуре не найден: ", key)
		return null
	
	var path = PIECE_PATHS[key]
	
	# Для превращённой пешки используем текстуру ферзя
	if promoted and piece_code == 1:
		key = "%s-queen" % color_name
		path = PIECE_PATHS[key]

	if not piece_textures.has(path):
		if ResourceLoader.exists(path):
			piece_textures[path] = load(path)
		else:
			printerr("Текстура не найдена: ", path)
			return null
	return piece_textures[path]


func update_pieces(instant: bool = false):
	"""ИСПРАВЛЕНО: НЕ пересоздаёт доску каждый кадр."""
	last_move_layer.clear()
	board_data = analyzer.get_board_data()

	# Подсветка последних ходов
	var last_moves = analyzer.get_last_moves_per_color()
	for i in range(last_moves.size()):
		var lm = last_moves[i]
		var col_code = lm.get("color", 0)
		if col_code != 0:
			var col = get_color_by_code(col_code)
			col.a = 0.3
			var from = Vector2i(lm.get("from_x"), lm.get("from_y"))
			var to = Vector2i(lm.get("to_x"), lm.get("to_y"))
			last_move_layer.add_rect(cell_to_pixel(from.x, from.y), Vector2(CELL_SIZE, CELL_SIZE), col)
			last_move_layer.add_rect(cell_to_pixel(to.x, to.y), Vector2(CELL_SIZE, CELL_SIZE), col)

	# Запоминаем старые позиции
	var old_pos = {}
	var old_code = {}
	for i in range(piece_sprites.size()):
		var sp = piece_sprites[i]
		if sp.visible:
			old_pos[i] = sp.get_meta("pos", Vector2i.ZERO)
			old_code[i] = sp.get_meta("code", 0)
		sp.visible = false
		sp.z_index = 0
		sp.modulate.a = 1.0

	var used_sprites = []
	var dead_colors = analyzer.get_dead_colors()

	# Первый проход: переиспользуем спрайты на тех же позициях
	for y in range(1, BOARD_SIZE + 1):
		for x in range(1, BOARD_SIZE + 1):
			if not is_board_cell(x, y):
				continue
			var code: int = board_data[y - 1][x - 1]
			if code != 0:
				var target_pos = Vector2i(x, y)
				for idx in old_pos.keys():
					if not idx in used_sprites and old_pos[idx] == target_pos:
						if old_code[idx] == code or (old_code[idx] % 10 == 1 and code % 10 == 5):
							used_sprites.append(idx)
							var sp = piece_sprites[idx]
							var tex = get_piece_texture(code)
							if tex:
								sp.texture = tex
								var tex_size = tex.get_size()
								if tex_size.x > 0:
									var sc = (CELL_SIZE * 0.9 * PIECE_SCALE_FACTOR) / tex_size.x
									sp.scale = Vector2(sc, sc)
							sp.position = cell_center(target_pos)
							sp.set_meta("pos", target_pos)
							sp.set_meta("code", code)
							_apply_dead_modulate(sp, code, dead_colors)
							sp.visible = true
							break

	# Второй проход: анимируем перемещения
	for y in range(1, BOARD_SIZE + 1):
		for x in range(1, BOARD_SIZE + 1):
			if not is_board_cell(x, y):
				continue
			var code: int = board_data[y - 1][x - 1]
			if code != 0:
				var target_pos = Vector2i(x, y)
				var already_assigned = false
				for idx in used_sprites:
					if piece_sprites[idx].get_meta("pos") == target_pos:
						already_assigned = true
						break
				if already_assigned:
					continue

				var best_sprite_idx = -1
				var best_dist = 9999.0
				for idx in old_pos.keys():
					if not idx in used_sprites and (old_code[idx] == code or (old_code[idx] % 10 == 1 and code % 10 == 5)):
						var dist = old_pos[idx].distance_to(target_pos)
						if dist < best_dist:
							best_dist = dist
							best_sprite_idx = idx

				if best_sprite_idx == -1:
					for i in range(piece_sprites.size()):
						if not i in used_sprites:
							best_sprite_idx = i
							break

				if best_sprite_idx != -1:
					used_sprites.append(best_sprite_idx)
					var sp = piece_sprites[best_sprite_idx]
					var tex = get_piece_texture(code)
					if tex:
						sp.texture = tex
						var tex_size = tex.get_size()
						if tex_size.x > 0:
							var sc = (CELL_SIZE * 0.9 * PIECE_SCALE_FACTOR) / tex_size.x
							sp.scale = Vector2(sc, sc)

					if old_pos.has(best_sprite_idx) and not instant:
						var dist = old_pos[best_sprite_idx].distance_to(target_pos)
						if dist > 1.0:
							var duration = clamp(dist / 1000.0, 0.08, 0.25)
							sp.z_index = 10
							var tween = create_tween()
							tween.tween_property(sp, "position", cell_center(target_pos), duration) \
								.set_trans(Tween.TRANS_SINE).set_ease(Tween.EASE_IN_OUT)
							tween.tween_property(sp, "z_index", 0, 0.0)
						else:
							sp.position = cell_center(target_pos)
					else:
						sp.position = cell_center(target_pos)

					sp.set_meta("pos", target_pos)
					sp.set_meta("code", code)
					_apply_dead_modulate(sp, code, dead_colors)
					sp.visible = true

	_update_cognitive_visualization()


func _apply_dead_modulate(sp: Sprite2D, code: int, dead_colors: PackedByteArray):
	"""Серые фигуры для мёртвых цветов."""
	var raw_code = code & 0x7F
	var p_color = raw_code / 10
	if p_color >= 1 and p_color <= 4 and dead_colors[p_color - 1] == 1:
		sp.modulate = Color(0.4, 0.4, 0.4, 1.0)
	else:
		sp.modulate = Color(1, 1, 1, 1.0)


func get_color_by_code(c: int) -> Color:
	match c:
		1: return Color.RED
		2: return Color.BLUE
		3: return Color.YELLOW
		4: return Color.GREEN
		0: return Color(0.5, 0.5, 0.5)
		_: return Color.WHITE


func clear_highlights():
	highlights_layer.clear()


func show_possible_moves(from_cell: Vector2i):
	clear_highlights()
	if from_cell == -Vector2i.ONE:
		return
	# Не сбрасываем выделение, если показываем ходы для той же клетки
	if from_cell == selected_cell:
		var moves = analyzer.get_legal_moves()
		for m in moves:
			if m.from_x == from_cell.x and m.from_y == from_cell.y:
				highlights_layer.add_circle(
					cell_center(Vector2i(m.to_x, m.to_y)),
					CELL_SIZE * 0.15,
					Color(0.2, 0.6, 1.0, 0.8),
					4.0, true
				)

# ==========================================================
# Когнитивная визуализация
# ==========================================================
func _on_cognitive_toggled(_pressed: bool):
	_update_cognitive_visualization()


func _update_cognitive_visualization():
	cognitive_layer.clear()
	if not (show_threats_check.button_pressed or show_targets_check.button_pressed
			or show_develop_check.button_pressed or show_plans_check.button_pressed):
		return

	var data = analyzer.get_cognitive_visualization()
	var thoughts = analyzer.get_ai_thoughts()
	ai_thoughts_label.text = thoughts["text"]

	var radius = CELL_SIZE * 0.45

	if show_threats_check.button_pressed:
		for t in data["threats"]:
			cognitive_layer.add_circle(
				cell_center(Vector2i(int(t["x"]), int(t["y"]))),
				radius, Color(1, 0.15, 0.15, 0.9), 4.0
			)

	if show_targets_check.button_pressed:
		for t in data["targets"]:
			cognitive_layer.add_circle(
				cell_center(Vector2i(int(t["x"]), int(t["y"]))),
				radius, Color(0.2, 0.45, 1.0, 0.9), 4.0
			)

	if show_develop_check.button_pressed:
		for t in data["develop"]:
			cognitive_layer.add_circle(
				cell_center(Vector2i(int(t["x"]), int(t["y"]))),
				radius, Color(0.2, 0.85, 0.3, 0.9), 4.0
			)

	if show_plans_check.button_pressed:
		var arrows = thoughts.get("arrows", [])
		var plan_colors = [
			Color(1, 1, 1, 0.9),
			Color(0.65, 0.65, 0.65, 0.75),
			Color(0.25, 0.85, 0.3, 0.85),
		]
		for i in range(arrows.size()):
			var r = arrows[i]
			var from = Vector2i(int(r["fx"]), int(r["fy"]))
			var to = Vector2i(int(r["tx"]), int(r["ty"]))
			var r_type = r.get("type", "plan_main")
			var col = Color(0.1, 0.1, 0.1, 0.75)
			if r_type == "plan_main":
				col = plan_colors[0]
			elif r_type == "plan_continuation":
				col = plan_colors[i % plan_colors.size()]
			cognitive_layer.add_arrow(cell_center(from), cell_center(to), col, 7.0, false)
			cognitive_layer.add_text(cell_center(from) + Vector2(-6, -16), str(i + 1), col, 18)

# ==========================================================
# PGN
# ==========================================================
func _on_load_pressed():
	if analyzer.load_pgn(pgn_text_edit.text):
		root_node = MoveNode.new()
		current_node = root_node
		var moves = analyzer.get_move_list()
		var parent = root_node
		for m in moves:
			if m.get("from_x", 0) == 0:
				continue
			var node = MoveNode.new()
			node.move = {
				"fx": int(m["from_x"]),
				"fy": int(m["from_y"]),
				"tx": int(m["to_x"]),
				"ty": int(m["to_y"]),
			}
			node.notation = m.get("notation", "?")
			node.parent = parent
			parent.children.append(node)
			parent = node
		current_node = parent
		replay_to_current()
		_update_move_list()
		update_move_label()
		if analyzer_enabled:
			_update_analysis_arrow()
	else:
		printerr("Ошибка загрузки PGN!")
