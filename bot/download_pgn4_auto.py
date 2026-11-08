"""
download_pgn4_auto.py — автоматическое скачивание FFA партий с chess.com.

Автоматически:
  1. Запрашивает ник игрока в консоли
  2. Вводит ник в поле поиска на странице архива
  3. Устанавливает фильтры: All time + 4 Player + FFA
  4. Скачивает только FFA партии

Первый запуск: нужно залогиниться вручную (один раз).
Последующие запуски: полностью автоматические через сохранённые cookies.

Запуск:
    python download_pgn4_auto.py
    python download_pgn4_auto.py --relogin
    python download_pgn4_auto.py --headless
"""

import asyncio
import json
import sys
import time
import argparse
from pathlib import Path

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
# Конфигурация
# ============================================================
COOKIES_FILE = Path("chess_com_cookies.json")
BASE_DIR = Path("партии FFA")

DELAY_BETWEEN_GAMES = 0.7
ARCHIVE_WAIT_TIMEOUT = 90

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
# Получение Player ID по нику
# ============================================================
def get_player_id(username: str) -> int | None:
    url = f"https://api.chess.com/pub/player/{username.lower()}"
    try:
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        })
        if resp.status_code == 200:
            return resp.json().get("player_id")
        print(f"❌ Игрок '{username}' не найден (HTTP {resp.status_code})")
        return None
    except Exception as e:
        print(f"❌ Ошибка получения Player ID: {e}")
        return None


# ============================================================
# Класс перехвата WS
# ============================================================
class WSCapture:
    def __init__(self):
        self.all_games = {}
        self.pgn_games = {}
        self.ws_connections = []
        self._last_archive_time = 0
        self.username_confirmed = False

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

                # Подтверждение выбора игрока из автокомплита
                if mutation == "set_username_choices":
                    data_obj = msg.get("data", {})
                    if isinstance(data_obj, dict):
                        players = data_obj.get("players", [])
                        if players:
                            self.username_confirmed = True

                # Список партий из архива
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
                        self._last_archive_time = time.time()
                        ffa_count = sum(1 for g in self.all_games.values() if self._is_ffa(g))
                        print(f"  📋 [{mutation}] +{new} партий "
                              f"(всего: {len(self.all_games)}, FFA: {ffa_count})")

                # PGN4 конкретной партии
                elif mutation == "archive_game":
                    data_obj = msg.get("data", {})
                    if isinstance(data_obj, str):
                        try: data_obj = json.loads(data_obj)
                        except: return

                    if isinstance(data_obj, dict):
                        game_nr = data_obj.get("gameNr")
                        pgn4 = data_obj.get("pgn4") or data_obj.get("pgn")

                        if game_nr and pgn4 and len(pgn4) > 50:
                            q = data_obj.get("q", {})
                            if isinstance(q, dict) and q.get("type") == "ffa":
                                self.pgn_games[game_nr] = pgn4
                            elif '[Variant "FFA"]' in pgn4:
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
        if game_meta.get("gameType") == "singles":
            rv = game_meta.get("ruleVariants", {})
            if isinstance(rv, dict):
                if "dkw" in rv or "promoteTo" in rv:
                    return True
        return False

    def get_ffa_game_nrs(self) -> list:
        return [nr for nr, meta in self.all_games.items() if self._is_ffa(meta)]


# ============================================================
# Вспомогательные функции
# ============================================================
async def click_filter(page, selectors: list, description: str, timeout=5000) -> bool:
    """Пробует кликнуть на элемент по списку селекторов."""
    for sel in selectors:
        try:
            elem = await page.wait_for_selector(sel, timeout=timeout)
            if elem:
                await elem.click()
                print(f"  ✅ {description} ({sel})")
                return True
        except Exception:
            continue
    print(f"  ⚠ {description} — не найден")
    return False


async def type_in_field(page, selectors: list, text: str, description: str, timeout=5000) -> bool:
    """Вводит текст в поле по списку селекторов."""
    for sel in selectors:
        try:
            elem = await page.wait_for_selector(sel, timeout=timeout)
            if elem:
                await elem.click()
                await asyncio.sleep(0.3)
                await elem.fill(text)
                print(f"  ✅ {description}: ввёл '{text}' ({sel})")
                return True
        except Exception:
            continue
    print(f"  ⚠ {description} — поле не найдено")
    return False


# ============================================================
# Основной скрипт
# ============================================================
async def main():
    parser = argparse.ArgumentParser(description="Скачивание FFA партий")
    parser.add_argument("--relogin", action="store_true")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("📥 СКАЧИВАНИЕ FFA ПАРТИЙ С CHESS.COM")
    print("=" * 70)
    username = input("👤 Введите ник игрока: ").strip()
    if not username:
        print("❌ Ник не введён.")
        return

    print(f"\n🔍 Получаю Player ID для '{username}'...")
    player_id = get_player_id(username)
    if player_id is None:
        return
    print(f"✅ Player ID: {player_id}")

    player_dir = BASE_DIR / username
    player_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 Папка: {player_dir.absolute()}")

    capture = WSCapture()

    async with async_playwright() as p:
        need_login = args.relogin or not COOKIES_FILE.exists()

        if need_login:
            browser = await p.chromium.launch(headless=False)
        else:
            browser = await p.chromium.launch(headless=args.headless)

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

        # ============================================================
        # ШАГ 1: Логин
        # ============================================================
        if need_login:
            print("\n" + "=" * 70)
            print("ШАГ 1: Логин (вручную, один раз)")
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
        else:
            print("\n✅ Используем сохранённые cookies")

        # ============================================================
        # ШАГ 2: Переход на страницу 4PC
        # ============================================================
        print("\n" + "=" * 70)
        print("ШАГ 2: Переход на 4-Player Chess")
        print("=" * 70)

        try:
            await page.goto(
                "https://www.chess.com/variants/4-player-chess",
                wait_until="domcontentloaded",
                timeout=30000
            )
        except Exception as e:
            print(f"⚠ Навигация: {e}")

        await asyncio.sleep(3)

        if "/login" in page.url.lower():
            print("❌ Cookies истекли. Запустите с --relogin")
            await browser.close()
            return

        print(f"✅ На странице: {page.url}")

        # ============================================================
        # ШАГ 3: Открытие архива
        # ============================================================
        print("\n" + "=" * 70)
        print("ШАГ 3: Открытие архива")
        print("=" * 70)

        archive_selectors = [
            'text="Archive"',
            'a:has-text("Archive")',
            'button:has-text("Archive")',
            '[data-test-id="archive"]',
            'nav >> text="Archive"',
            'div[role="tab"]:has-text("Archive")',
        ]

        clicked = await click_filter(page, archive_selectors, "Клик на Archive")

        if not clicked:
            print("  Пробую прямой URL...")
            try:
                await page.goto(
                    "https://www.chess.com/variants/4-player-chess/archive",
                    wait_until="domcontentloaded",
                    timeout=15000
                )
                print("  ✅ Перешёл на /archive")
            except Exception:
                pass

        await asyncio.sleep(3)

        # ============================================================
        # ШАГ 4: Ввод ника в поле поиска
        # ============================================================
        print("\n" + "=" * 70)
        print(f"ШАГ 4: Ввод ника '{username}' в поле поиска")
        print("=" * 70)

        # Ищем поле ввода для поиска игрока
        input_selectors = [
            'input[placeholder*="player"]',
            'input[placeholder*="Player"]',
            'input[placeholder*="search"]',
            'input[placeholder*="Search"]',
            'input[placeholder*="username"]',
            'input[placeholder*="Username"]',
            'input[placeholder*="name"]',
            'input[type="text"]',
            'input[type="search"]',
            '[data-test-id="player-search"] input',
            '[data-test-id="search-input"] input',
        ]

        typed = await type_in_field(page, input_selectors, username, "Ввод ника")

        if typed:
            # Ждём автокомплит
            await asyncio.sleep(1.5)

            # Пробуем кликнуть на первый вариант автокомплита
            autocomplete_selectors = [
                f'text="{username}"',
                f'div:has-text("{username}")',
                f'li:has-text("{username}")',
                f'span:has-text("{username}")',
                '[data-test-id="username-choice"]',
                '.username-choice',
            ]

            # Сначала пробуем точное совпадение
            for sel in autocomplete_selectors[:2]:
                try:
                    elem = await page.wait_for_selector(sel, timeout=3000)
                    if elem:
                        await elem.click()
                        print(f"  ✅ Кликнул на вариант автокомплита: {username}")
                        await asyncio.sleep(1)
                        break
                except Exception:
                    continue
            else:
                # Если не нашли вариант, пробуем просто нажать Enter
                print("  ⚠ Вариант автокомплита не найден, пробую Enter...")
                try:
                    await page.keyboard.press("Enter")
                    await asyncio.sleep(1)
                except Exception:
                    pass
        else:
            print("  ⚠ Поле ввода не найдено, пропускаю ввод ника")

        await asyncio.sleep(1)

        # ============================================================
        # ШАГ 5: Установка фильтров
        # ============================================================
        print("\n" + "=" * 70)
        print("ШАГ 5: Установка фильтров")
        print("=" * 70)

        # 5.1. Фильтр "All time" (за всё время)
        print("\n  [5.1] Фильтр времени: All time")
        all_time_selectors = [
            'text="All time"',
            'text="All Time"',
            'text="За всё время"',
            'button:has-text("All time")',
            'div:has-text("All time")',
            '[data-test-id="time-all"]',
        ]
        await click_filter(page, all_time_selectors, "All time")
        await asyncio.sleep(1)

        # 5.2. Фильтр "4 Player"
        print("\n  [5.2] Фильтр варианта: 4 Player")
        four_player_selectors = [
            'text="4 Player"',
            'text="4-Player"',
            'text="4PC"',
            'button:has-text("4 Player")',
            'div:has-text("4 Player")',
            '[data-test-id="variant-4pc"]',
        ]
        await click_filter(page, four_player_selectors, "4 Player")
        await asyncio.sleep(1)

        # 5.3. Фильтр Game mode = FFA
        print("\n  [5.3] Фильтр Game mode: FFA")
        ffa_selectors = [
            'text="FFA"',
            'text="Free for All"',
            'text="Free-for-All"',
            'button:has-text("FFA")',
            'div:has-text("FFA")',
            '[data-test-id="gamemode-ffa"]',
        ]
        await click_filter(page, ffa_selectors, "FFA")
        await asyncio.sleep(1)

        # 5.4. Пробуем применить фильтры
        print("\n  [5.4] Применение фильтров")
        apply_selectors = [
            'button:has-text("Apply")',
            'button:has-text("Search")',
            'button:has-text("Filter")',
            'text="Apply"',
            'text="Search"',
        ]
        await click_filter(page, apply_selectors, "Apply/Search", timeout=3000)
        await asyncio.sleep(2)

        # ============================================================
        # ШАГ 6: Ожидание списка партий
        # ============================================================
        print("\n" + "=" * 70)
        print("ШАГ 6: Ожидание списка партий из архива")
        print("=" * 70)

        start_time = time.time()
        last_count = 0
        stable_secs = 0

        while time.time() - start_time < ARCHIVE_WAIT_TIMEOUT:
            await asyncio.sleep(1)
            cur = len(capture.all_games)

            if cur > last_count:
                last_count = cur
                stable_secs = 0
            else:
                stable_secs += 1

            if stable_secs >= 7 and cur > 0:
                print(f"✅ Список стабилизировался: {cur} партий")
                break

            # Прокручиваем вниз для загрузки дополнительных партий
            if stable_secs == 4 and cur > 0:
                try:
                    await page.evaluate("window.scrollBy(0, 2000)")
                    await asyncio.sleep(1)
                except Exception:
                    pass

        if len(capture.all_games) == 0:
            print("❌ Список партий не получен.")
            print("   Попробуйте без --headless и вручную проверить фильтры.")
            await browser.close()
            return

        # ============================================================
        # ШАГ 7: Фильтрация FFA
        # ============================================================
        print("\n" + "=" * 70)
        print("ШАГ 7: Фильтрация FFA партий")
        print("=" * 70)

        ffa_nrs = capture.get_ffa_game_nrs()
        total = len(capture.all_games)
        print(f"📋 Всего партий в архиве: {total}")
        print(f"♟️ Из них FFA: {len(ffa_nrs)}")
        print(f"❌ Не-FFA (пропущено): {total - len(ffa_nrs)}")

        if len(ffa_nrs) == 0:
            print("❌ FFA партий не найдено.")
            await browser.close()
            return

        # ============================================================
        # ШАГ 8: Скачивание PGN4
        # ============================================================
        print("\n" + "=" * 70)
        print(f"ШАГ 8: Скачивание PGN4 для {len(ffa_nrs)} FFA партий")
        print("=" * 70)

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
                print(f"  📥 [{i}/{len(ffa_nrs)}] "
                      f"(получено PGN4: {len(capture.pgn_games)})")

            await asyncio.sleep(DELAY_BETWEEN_GAMES)

        print(f"\n⏳ Ожидание последних ответов (15 сек)...")
        await asyncio.sleep(15)

        # ============================================================
        # ШАГ 9: Сохранение файлов
        # ============================================================
        print("\n" + "=" * 70)
        print("ШАГ 9: Сохранение PGN4 файлов")
        print("=" * 70)

        saved = 0
        for game_nr, pgn4 in capture.pgn_games.items():
            if '[Variant "FFA"]' not in pgn4 and '[Variant "4PC FFA"]' not in pgn4:
                continue

            fname = player_dir / f"game_{game_nr}.pgn4"
            fname.write_text(pgn4, encoding="utf-8")
            saved += 1

        print(f"✅ Сохранено {saved} FFA партий")

        # ============================================================
        # Итоги
        # ============================================================
        print("\n" + "=" * 70)
        print("📊 ИТОГИ")
        print("=" * 70)
        print(f"👤 Игрок: {username} (ID: {player_id})")
        print(f"📋 Всего партий в архиве: {len(capture.all_games)}")
        print(f"♟️ FFA партий найдено: {len(ffa_nrs)}")
        print(f"💾 PGN4 сохранено: {saved}")
        print(f"📂 Папка: {player_dir.absolute()}")
        print("=" * 70)

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())