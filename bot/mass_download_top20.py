"""
mass_download_top20.py — массовое скачивание FFA партий топ-20 игроков.

Автоматически:
  - Скачивает FFA партии каждого игрока
  - Создаёт папку "партии FFA/{ник}/"
  - Поддерживает возобновление (resume) при перезапуске
  - Пропускает уже скачанных игроков

Запуск:
    python mass_download_top20.py
    python mass_download_top20.py --relogin
    python mass_download_top20.py --skip-existing   # пропускать уже скачанных
    python mass_download_top20.py --limit 5          # только первые 5 игроков
"""

import asyncio
import json
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

try:
    import requests
except ImportError:
    print("pip install requests")
    sys.exit(1)

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("pip install playwright && playwright install chromium")
    sys.exit(1)


# ============================================================
# СПИСОК ТОП-20 ИГРОКОВ (из в.txt)
# ============================================================
TOP_20_PLAYERS = [
    "Prathu_19",        # 3380
    "HSCCCB",           # 3367
    "rojitto",          # 3346
    "Clock_bht",        # 3151
    "shadow1729",       # 3123
    "Puzzles",          # 3109
    "Gudadev",          # 3080
    "DNSPhantom",       # 3036
    "hdnight",          # 3035
    "neoserbian",       # 3027
    "Articpolarbear",   # 3024
    "PlayingFor2nd",    # 3011
    "MillzGambit",      # 2984
    "Slayer950",        # 2982
    "ItsRatingHen",     # 2963
    "Mnbcion",          # 2960
    "CowDownUnder",     # 2945
    "Myway98",          # 2930
    "Pioleen",          # 2928
]


# ============================================================
# Конфигурация
# ============================================================
COOKIES_FILE = Path("chess_com_cookies.json")
BASE_DIR = Path("партии FFA")
PROGRESS_FILE = Path("mass_download_progress.json")

# ============================================================
# Обновлённые константы
# ============================================================
DELAY_BETWEEN_GAMES = 0.7
DELAY_BETWEEN_PLAYERS = 5.0
ARCHIVE_WAIT_TIMEOUT = 600       # Увеличен до 5 минут (было 90 сек)
STABLE_TIMEOUT = 12              # Секунд без новых партий перед "Show more"
MAX_SHOW_MORE_CLICKS = 200       # Максимум кликов "Show more" (защита от бесконечного цикла)
MAX_TOTAL_GAMES = 0              # Максимум партий на одного игрока (0 = без лимита)

# Селекторы кнопки "Show more" / "Load more"
SHOW_MORE_SELECTORS = [
    'button:has-text("Show more")',
    'button:has-text("Show More")',
    'button:has-text("Load more")',
    'button:has-text("Load More")',
    'button:has-text("See more")',
    'button:has-text("See More")',
    'text="Show more"',
    'text="Load more"',
    'a:has-text("Show more")',
    'div:has-text("Show more")',
    '[data-test-id="show-more"]',
    '[data-test-id="load-more"]',
]

# ============================================================
# JS для перехвата WebSocket
# ============================================================
JS_INJECT_WS = """
window.__intercepted_ws = window.__intercepted_ws || [];
const _OrigWS = window.WebSocket;
window.WebSocket = function(url, protocols) {
    const ws = protocols ? new _OrigWS(url, protocols) : new _OrigWS(url);
    if (url.includes('variants')) {
        window.__intercepted_ws.push(ws);
    }
    return ws;
};
window.WebSocket.prototype = _OrigWS.prototype;
window.WebSocket.CONNECTING = _OrigWS.CONNECTING;
window.WebSocket.OPEN = _OrigWS.OPEN;
window.WebSocket.CLOSING = _OrigWS.CLOSING;
window.WebSocket.CLOSED = _OrigWS.CLOSED;
"""


# ============================================================
# Получение Player ID
# ============================================================
def get_player_id(username: str) -> int | None:
    url = f"https://api.chess.com/pub/player/{username.lower()}"
    try:
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        })
        if resp.status_code == 200:
            return resp.json().get("player_id")
        return None
    except Exception:
        return None


# ============================================================
# Класс перехвата WS
# ============================================================
class WSCapture:
    def __init__(self):
        self.all_games = {}
        self.pgn_games = {}
        self.ws_connections = []

    def reset(self):
        """Сброс состояния для нового игрока."""
        self.all_games = {}
        self.pgn_games = {}
        self.ws_connections = []

    def on_websocket(self, ws):
        if "variants" not in ws.url:
            return
        self.ws_connections.append(ws)

        def on_frame(data):
            try:
                if isinstance(data, bytes):
                    data = data.decode('utf-8')
                msg = json.loads(data)
                mutation = msg.get("mutation", "")

                if mutation in ("set_games_search_result", "add_games_search_result"):
                    data_obj = msg.get("data", {})
                    if isinstance(data_obj, str):
                        try: data_obj = json.loads(data_obj)
                        except: return

                    games_list = []
                    if isinstance(data_obj, dict):
                        games_list = data_obj.get("games", [])
                    elif isinstance(data_obj, list):
                        games_list = data_obj

                    new = 0
                    for g in games_list:
                        if isinstance(g, dict) and "gameNr" in g:
                            nr = g["gameNr"]
                            if nr not in self.all_games:
                                self.all_games[nr] = g
                                new += 1

                    if new > 0:
                        ffa_count = sum(1 for g in self.all_games.values() if self._is_ffa(g))
                        print(f"  📋 [{mutation}] +{new} партий "
                              f"(всего: {len(self.all_games)}, FFA: {ffa_count})")

                elif mutation == "archive_game":
                    data_obj = msg.get("data", {})
                    if isinstance(data_obj, str):
                        try: data_obj = json.loads(data_obj)
                        except: return

                    if isinstance(data_obj, dict):
                        game_nr = data_obj.get("gameNr")
                        pgn4 = data_obj.get("pgn4") or data_obj.get("pgn")

                        if game_nr and pgn4 and len(pgn4) > 50:
                            if '[Variant "FFA"]' in pgn4 or '[Variant "4PC FFA"]' in pgn4:
                                self.pgn_games[game_nr] = pgn4
            except Exception:
                pass

        ws.on("framereceived", on_frame)

    @staticmethod
    def _is_ffa(game_meta: dict) -> bool:
        q = game_meta.get("q", {})
        if isinstance(q, dict):
            if q.get("type") == "ffa":
                return True
        return False

    def get_ffa_game_nrs(self) -> list:
        return [nr for nr, meta in self.all_games.items() if self._is_ffa(meta)]


# ============================================================
# Прогресс
# ============================================================
def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"completed": {}, "failed": []}


def save_progress(progress: dict):
    PROGRESS_FILE.write_text(
        json.dumps(progress, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


# ============================================================
# Селекторы для кликов
# ============================================================
ARCHIVE_SELECTORS = [
    'text="Archive"', 'a:has-text("Archive")',
    'button:has-text("Archive")', 'div[role="tab"]:has-text("Archive")',
]

INPUT_SELECTORS = [
    'input[placeholder*="player" i]',
    'input[placeholder*="search" i]',
    'input[placeholder*="username" i]',
    'input[type="text"]', 'input[type="search"]',
]

ALL_TIME_SELECTORS = [
    'text="All time"', 'text="All Time"',
    'button:has-text("All time")', 'div:has-text("All time")',
]

FOUR_PLAYER_SELECTORS = [
    'text="4 Player"', 'text="4-Player"', 'text="4PC"',
    'button:has-text("4 Player")', 'div:has-text("4 Player")',
]

FFA_SELECTORS = [
    'text="FFA"', 'text="Free for All"',
    'button:has-text("FFA")', 'div:has-text("FFA")',
]


# ============================================================
# Вспомогательные функции
# ============================================================
async def try_click(page, selectors, description, timeout=4000) -> bool:
    for sel in selectors:
        try:
            elem = await page.wait_for_selector(sel, timeout=timeout)
            if elem:
                await elem.click()
                print(f"  ✅ {description}")
                return True
        except Exception:
            continue
    return False


async def try_type(page, selectors, text, timeout=4000) -> bool:
    for sel in selectors:
        try:
            elem = await page.wait_for_selector(sel, timeout=timeout)
            if elem:
                await elem.click()
                await asyncio.sleep(0.3)
                await elem.fill(text)
                print(f"  ✅ Ввёл '{text}'")
                return True
        except Exception:
            continue
    return False


# ============================================================
# Скачивание одного игрока
# ============================================================
async def download_player(
    page, capture: WSCapture, username: str, player_id: int, player_dir: Path
) -> dict:
    """Скачивает все FFA партии одного игрока. Возвращает статистику."""
    stats = {
        "username": username,
        "player_id": player_id,
        "total_in_archive": 0,
        "ffa_found": 0,
        "pgn_downloaded": 0,
        "pgn_saved": 0,
        "success": False,
    }

    capture.reset()

    # 1. Переход на 4PC
    try:
        await page.goto(
            "https://www.chess.com/variants/4-player-chess",
            wait_until="domcontentloaded",
            timeout=30000
        )
    except Exception as e:
        print(f"  ⚠ Навигация: {e}")

    await asyncio.sleep(3)

    if "/login" in page.url.lower():
        print("  ❌ Сессия истекла!")
        return stats

    # 2. Открытие архива
    await try_click(page, ARCHIVE_SELECTORS, "Открытие Archive")
    await asyncio.sleep(3)

    # 3. Ввод ника
    typed = await try_type(page, INPUT_SELECTORS, username)
    if typed:
        await asyncio.sleep(1.5)
        try:
            elem = await page.wait_for_selector(f'text="{username}"', timeout=3000)
            if elem:
                await elem.click()
                print(f"  ✅ Выбран вариант автокомплита")
                await asyncio.sleep(1)
        except Exception:
            await page.keyboard.press("Enter")
            await asyncio.sleep(1)

    # 4. Установка фильтров
    await try_click(page, ALL_TIME_SELECTORS, "Фильтр: All time")
    await asyncio.sleep(1)
    await try_click(page, FOUR_PLAYER_SELECTORS, "Фильтр: 4 Player")
    await asyncio.sleep(1)
    await try_click(page, FFA_SELECTORS, "Фильтр: FFA")
    await asyncio.sleep(2)

    # 5. Ожидание списка партий С АВТОМАТИЧЕСКОЙ ПОДГРУЗКОЙ
    print(f"  ⏳ Ожидание списка партий (с автоподгрузкой)...")
    start_time = time.time()
    last_count = 0
    stable_secs = 0
    show_more_clicks = 0

    while time.time() - start_time < ARCHIVE_WAIT_TIMEOUT:
        await asyncio.sleep(1)
        cur = len(capture.all_games)

        if cur > last_count:
            last_count = cur
            stable_secs = 0
        else:
            stable_secs += 1

        # Проверка лимита
        if MAX_TOTAL_GAMES > 0 and cur >= MAX_TOTAL_GAMES:
            print(f"  ✅ Достигнут лимит {MAX_TOTAL_GAMES} партий")
            break

        # Список стабилизировался — пробуем подгрузить ещё
        if stable_secs >= STABLE_TIMEOUT and cur > 0:
            # Проверяем, есть ли кнопка "Show more"
            clicked_more = False

            for sel in SHOW_MORE_SELECTORS:
                try:
                    btn = await page.wait_for_selector(sel, timeout=2000)
                    if btn:
                        await btn.scroll_into_view_if_needed()
                        await asyncio.sleep(0.5)
                        await btn.click()
                        show_more_clicks += 1
                        clicked_more = True
                        print(f"  📥 Клик 'Show more' #{show_more_clicks} "
                              f"(сейчас {cur} партий)")
                        stable_secs = 0  # Сбрасываем таймер
                        await asyncio.sleep(2)  # Ждём новые данные
                        break
                except Exception:
                    continue

            if not clicked_more:
                # Пробуем прокрутку вниз (может триггернуть автоподгрузку)
                try:
                    await page.evaluate("window.scrollBy(0, 3000)")
                    await asyncio.sleep(1)
                except Exception:
                    pass

                # Если прокрутка не дала новых партий — пробуем ещё раз
                # отправить search-archive через WS
                if show_more_clicks == 0 and stable_secs >= STABLE_TIMEOUT + 5:
                    # Отправляем повторный запрос через WS
                    query_id = int(time.time() * 1000)
                    js_resend = f"""
                    (() => {{
                        for (let ws of window.__intercepted_ws) {{
                            if (ws.readyState === 1) {{
                                ws.send(JSON.stringify({{
                                    action: "search-archive",
                                    data: {{
                                        days: "0-9999",
                                        playerId: {player_id},
                                        gameType: "",
                                        ratingType: "4player",
                                        queryId: {query_id}
                                    }},
                                    windowId: "{player_id}_{query_id}"
                                }}));
                                return true;
                            }}
                        }}
                        return false;
                    }})();
                    """
                    try:
                        await page.evaluate(js_resend)
                        print(f"  🔄 Повторный search-archive отправлен")
                        stable_secs = 0
                        await asyncio.sleep(3)
                    except Exception:
                        pass

            # Защита от бесконечного цикла
            if show_more_clicks >= MAX_SHOW_MORE_CLICKS:
                print(f"  ⚠ Достигнут лимит кликов 'Show more'")
                break

            # Если после Show more + прокрутки + повторного запроса
            # всё ещё нет новых партий в течение длительного времени — выходим
            if stable_secs >= STABLE_TIMEOUT + 20:
                print(f"  ✅ Больше партий нет. Итого: {cur}")
                break

    stats["total_in_archive"] = len(capture.all_games)
    print(f"  📊 Всего партий в архиве: {len(capture.all_games)}")

    if len(capture.all_games) == 0:
        print(f"  ❌ Список партий пуст")
        return stats

    # 6. Фильтрация FFA
    ffa_nrs = capture.get_ffa_game_nrs()
    stats["ffa_found"] = len(ffa_nrs)
    print(f"  ♟️ FFA партий: {len(ffa_nrs)}")

    if len(ffa_nrs) == 0:
        return stats

    # 7. Скачивание PGN4
    query_id = int(time.time() * 1000)

    for i, game_nr in enumerate(ffa_nrs, 1):
        if game_nr in capture.pgn_games:
            continue

        js_send = f"""
        (() => {{
            for (let ws of window.__intercepted_ws) {{
                if (ws.readyState === 1) {{
                    ws.send(JSON.stringify({{
                        action: "game",
                        data: {{ gameNr: {game_nr}, action: "join" }},
                        windowId: "{player_id}_{query_id}"
                    }}));
                    return true;
                }}
            }}
            return false;
        }})();
        """
        try:
            await page.evaluate(js_send)
        except Exception:
            pass

        if i % 10 == 0:
            print(f"    📥 [{i}/{len(ffa_nrs)}] (получено PGN4: {len(capture.pgn_games)})")

        await asyncio.sleep(DELAY_BETWEEN_GAMES)

    stats["pgn_downloaded"] = len(capture.pgn_games)
    print(f"  ⏳ Ожидание последних ответов (10 сек)...")
    await asyncio.sleep(10)

    # 8. Сохранение файлов
    saved = 0
    for game_nr, pgn4 in capture.pgn_games.items():
        if '[Variant "FFA"]' not in pgn4:
            continue
        fname = player_dir / f"game_{game_nr}.pgn4"
        fname.write_text(pgn4, encoding="utf-8")
        saved += 1

    stats["pgn_saved"] = saved
    stats["success"] = saved > 0
    print(f"  ✅ Сохранено {saved} FFA партий")

    return stats


# ============================================================
# Основной скрипт
# ============================================================
async def main():
    parser = argparse.ArgumentParser(description="Массовое скачивание топ-20 FFA")
    parser.add_argument("--relogin", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Пропускать игроков с уже скачанными партиями")
    parser.add_argument("--limit", type=int, default=0,
                        help="Скачать только первых N игроков")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Начать с N-го игрока (для возобновления)")
    args = parser.parse_args()

    print("=" * 70)
    print("🏆 МАССОВОЕ СКАЧИВАНИЕ ТОП-20 ИГРОКОВ FFA")
    print("=" * 70)

    # Загружаем прогресс
    progress = load_progress()
    completed = progress.get("completed", {})

    # Фильтруем список игроков
    players_to_download = []
    for i, username in enumerate(TOP_20_PLAYERS):
        if args.limit > 0 and len(players_to_download) >= args.limit:
            break
        if i < args.start_from:
            continue
        if args.skip_existing and username in completed:
            print(f"  ⏭️ Пропуск {username} (уже скачан)")
            continue
        players_to_download.append(username)

    print(f"\n📋 Планируется скачать: {len(players_to_download)} игроков")
    print(f"📂 Папка: {BASE_DIR.absolute()}")

    if len(players_to_download) == 0:
        print("✅ Все игроки уже скачаны!")
        return

    capture = WSCapture()

    async with async_playwright() as p:
        need_login = args.relogin or not COOKIES_FILE.exists()

        if need_login:
            browser = await p.chromium.launch(headless=False)
        else:
            browser = await p.chromium.launch(
                headless=args.headless,
                channel="msedge",   # использовать Edge вместо Chromium
            )

        if not need_login and COOKIES_FILE.exists():
            print(f"\n📂 Загружаю cookies из {COOKIES_FILE}...")
            context = await browser.new_context(
                storage_state=str(COOKIES_FILE),
                viewport={"width": 1920, "height": 1080},
                locale="en-US",
            )
        else:
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                locale="en-US",
            )

        await context.add_init_script(JS_INJECT_WS)
        page = await context.new_page()
        page.on("websocket", capture.on_websocket)

        # Логин
        if need_login:
            print("\n" + "=" * 70)
            print("ШАГ 0: Логин (вручную, один раз)")
            print("=" * 70)
            await page.goto("https://www.chess.com/login", wait_until="domcontentloaded")
            print("👉 Залогиньтесь в браузере...")

            for _ in range(300):
                if "/login" not in page.url.lower():
                    print(f"✅ Логин успешен!")
                    break
                await asyncio.sleep(1)
            else:
                print("❌ Таймаут логина")
                await browser.close()
                return

            await context.storage_state(path=str(COOKIES_FILE))
            print(f"💾 Cookies сохранены в {COOKIES_FILE}")

        # Основной цикл по игрокам
        total_stats = []
        start_all = time.time()

        for player_idx, username in enumerate(players_to_download, 1):
            print(f"\n{'=' * 70}")
            print(f"🎯 [{player_idx}/{len(players_to_download)}] {username}")
            print(f"{'=' * 70}")

            # Получаем Player ID
            player_id = get_player_id(username)
            if player_id is None:
                print(f"  ❌ Игрок не найден через API, пропускаем")
                progress["failed"].append(username)
                save_progress(progress)
                continue

            print(f"  🔍 Player ID: {player_id}")

            # Создаём папку игрока
            player_dir = BASE_DIR / username
            player_dir.mkdir(parents=True, exist_ok=True)

            # Проверяем, не скачан ли уже
            existing_files = list(player_dir.glob("*.pgn4"))
            if existing_files and args.skip_existing:
                print(f"  ⏭️ Уже скачано {len(existing_files)} партий, пропускаем")
                progress["completed"][username] = {
                    "count": len(existing_files),
                    "timestamp": datetime.now().isoformat(),
                }
                save_progress(progress)
                continue

            # Скачиваем
            player_start = time.time()
            stats = await download_player(page, capture, username, player_id, player_dir)
            player_time = time.time() - player_start

            total_stats.append(stats)

            # Сохраняем прогресс
            if stats["success"]:
                progress["completed"][username] = {
                    "player_id": player_id,
                    "count": stats["pgn_saved"],
                    "ffa_found": stats["ffa_found"],
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                progress["failed"].append(username)
            save_progress(progress)

            print(f"  ⏱️ Время: {player_time:.1f}с")

            # Задержка между игроками
            if player_idx < len(players_to_download):
                print(f"  ⏳ Пауза {DELAY_BETWEEN_PLAYERS}с перед следующим игроком...")
                await asyncio.sleep(DELAY_BETWEEN_PLAYERS)

        total_time = time.time() - start_all
        await browser.close()

    # ============================================================
    # Итоговая статистика
    # ============================================================
    print("\n" + "=" * 70)
    print("🏁 ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 70)
    print(f"⏱️ Общее время: {total_time / 60:.1f} мин")
    print(f"📊 Игроков обработано: {len(total_stats)}")

    total_saved = 0
    total_ffa = 0
    success_count = 0

    print(f"\n{'Игрок':<20} {'FFA':<6} {'PGN':<6} {'Сохранено':<10}")
    print("-" * 50)
    for s in total_stats:
        print(f"{s['username']:<20} {s['ffa_found']:<6} "
              f"{s['pgn_downloaded']:<6} {s['pgn_saved']:<10}")
        total_saved += s["pgn_saved"]
        total_ffa += s["ffa_found"]
        if s["success"]:
            success_count += 1

    print("-" * 50)
    print(f"{'ИТОГО':<20} {total_ffa:<6} {total_saved:<6} {total_saved:<10}")
    print(f"\n✅ Успешно скачано игроков: {success_count}/{len(total_stats)}")
    print(f"💾 Всего FFA партий сохранено: {total_saved}")
    print(f"📂 Папка: {BASE_DIR.absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())