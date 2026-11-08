"""
download_all_pgn4_v2.py — улучшенная версия с полной диагностикой.
Сохраняет ВСЕ полученные мутации для анализа.
"""

import asyncio
import json
import time
from pathlib import Path
from playwright.async_api import async_playwright

OUTPUT_DIR = Path("pgn4_downloads")
OUTPUT_DIR.mkdir(exist_ok=True)

PLAYER_ID = 371668759
USERNAME = "good123S"

DELAY_BETWEEN_GAMES = 1.5

JS_INJECT_WS = """
window.__intercepted_ws = [];
const OriginalWebSocket = window.WebSocket;

window.WebSocket = function(url, protocols) {
    const ws = protocols ? new OriginalWebSocket(url, protocols) : new OriginalWebSocket(url);
    if (url.includes('variants.gcp-prod.chess.com') || url.includes('pubsub')) {
        window.__intercepted_ws.push(ws);
    }
    return ws;
};
window.WebSocket.prototype = OriginalWebSocket.prototype;
window.WebSocket.CONNECTING = OriginalWebSocket.CONNECTING;
window.WebSocket.OPEN = OriginalWebSocket.OPEN;
window.WebSocket.CLOSING = OriginalWebSocket.CLOSING;
window.WebSocket.CLOSED = OriginalWebSocket.CLOSED;
"""

async def async_input(prompt=""):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, prompt)

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={"width": 1920, "height": 1080}, locale="en-US")
        await context.add_init_script(JS_INJECT_WS)
        page = await context.new_page()
        
        # Хранилища данных
        all_mutations = []  # ВСЕ полученные мутации для анализа
        archive_game_nrs = []
        pgn_games = {}
        game_metadata = {}
        
        def on_websocket(ws):
            if "variants" in ws.url:
                print(f"🔌 [WS] Подключен: {ws.url}")
                
                def on_frame_received(data):
                    try:
                        if isinstance(data, bytes): data = data.decode('utf-8')
                        msg = json.loads(data)
                        mut = msg.get("mutation", "")
                        
                        # Сохраняем ВСЕ мутации для анализа
                        if mut:
                            all_mutations.append({
                                "mutation": mut,
                                "timestamp": time.time(),
                                "data_preview": str(msg.get("data", ""))[:200]
                            })
                            print(f"  📨 Мутация: {mut}")
                        
                        # Ищем список игр в ЛЮБОЙ мутации
                        data_obj = msg.get("data", {})
                        if isinstance(data_obj, str):
                            try:
                                data_obj = json.loads(data_obj)
                            except Exception:
                                return
                        
                        # Проверяем различные структуры
                        games_list = None
                        if isinstance(data_obj, dict):
                            # Пробуем разные ключи
                            for key in ["games", "archive", "results", "items", "list"]:
                                if key in data_obj and isinstance(data_obj[key], list):
                                    games_list = data_obj[key]
                                    print(f"     ✓ Найден список в ключе '{key}': {len(games_list)} элементов")
                                    break
                        elif isinstance(data_obj, list):
                            games_list = data_obj
                            print(f"     ✓ Data является списком: {len(games_list)} элементов")
                        
                        if games_list:
                            new_games = 0
                            for g in games_list:
                                if isinstance(g, dict) and "gameNr" in g:
                                    nr = g["gameNr"]
                                    if nr not in game_metadata:
                                        game_metadata[nr] = g
                                        archive_game_nrs.append(nr)
                                        new_games += 1
                            
                            if new_games > 0:
                                print(f"  📋 +{new_games} новых игр (всего: {len(archive_game_nrs)})")
                        
                        # Перехватываем PGN4
                        if mut == "archive_game":
                            if not isinstance(data_obj, dict):
                                return
                            
                            game_nr = data_obj.get("gameNr")
                            pgn4 = data_obj.get("pgn4") or data_obj.get("pgn")
                            
                            if game_nr and pgn4 and len(pgn4) > 50:
                                pgn_games[game_nr] = pgn4
                                fname = OUTPUT_DIR / f"game_{game_nr}.pgn4"
                                with open(fname, "w", encoding="utf-8") as f:
                                    f.write(pgn4)
                                print(f"  🎯 PGN4 игры {game_nr} сохранён ({len(pgn4)} символов)")
                            elif game_nr:
                                print(f"  ⚠ Игра {game_nr}: PGN4 отсутствует")
                                
                    except Exception as e:
                        print(f"  ⚠ Ошибка парсинга: {e}")

                ws.on("framereceived", on_frame_received)
                
        page.on("websocket", on_websocket)
        
        # ========== ШАГ 1: Авторизация ==========
        print("="*60)
        print("ШАГ 1: Авторизация")
        print("="*60)
        await page.goto("https://www.chess.com/login", wait_until="domcontentloaded")
        await async_input("👉 Залогиньтесь в браузере и нажмите Enter здесь...")
        
        for _ in range(180):
            if "/login" not in page.url.lower():
                print(f"✅ Логин успешен! URL: {page.url}")
                break
            await asyncio.sleep(1)
        else:
            print("❌ Таймаут логина.")
            await browser.close()
            return
        
        # ========== ШАГ 2: Переход на 4PC ==========
        print("\n" + "="*60)
        print("ШАГ 2: Загрузка архива")
        print("="*60)
        
        try:
            await page.goto("https://www.chess.com/variants/4-player-chess", 
                           wait_until="domcontentloaded", timeout=30000)
        except Exception as e:
            print(f"⚠ Навигация: {e}")
            
        await asyncio.sleep(3)
        if "/login" in page.url.lower():
            print("❌ Редирект на логин.")
            await browser.close()
            return
        
        # ========== ШАГ 3: Запрос архива ==========
        print("🚀 Отправка запроса search-archive...")
        query_id = int(time.time() * 1000)
        
        js_send_archive = f"""
        (() => {{
            for (let ws of window.__intercepted_ws) {{
                if (ws.readyState === 1) {{
                    ws.send(JSON.stringify({{
                        action: "search-archive",
                        data: {{
                            days: "0-9999",
                            playerId: {PLAYER_ID},
                            gameType: "",
                            ratingType: "4player",
                            queryId: {query_id}
                        }},
                        windowId: "{PLAYER_ID}_{query_id}"
                    }}));
                    return true;
                }}
            }}
            return false;
        }})();
        """
        
        sent = await page.evaluate(js_send_archive)
        if not sent:
            print("❌ WS не готов.")
            await async_input("Откройте Archive -> 4 Player вручную и нажмите Enter...")
        
        # Увеличенное ожидание
        print("⏳ Ожидание ответа (30 сек)...")
        for i in range(30):
            await asyncio.sleep(1)
            if len(archive_game_nrs) > 0:
                print(f"  [{i+1}s] Получено игр: {len(archive_game_nrs)}")
        
        # ========== ДИАГНОСТИКА ==========
        print("\n" + "="*60)
        print("ДИАГНОСТИКА")
        print("="*60)
        print(f"📨 Всего получено мутаций: {len(all_mutations)}")
        print("\nСписок всех мутаций:")
        for i, m in enumerate(all_mutations):
            print(f"  {i+1}. {m['mutation']}: {m['data_preview'][:100]}")
        
        # Сохраняем все мутации для анализа
        with open(OUTPUT_DIR / "all_mutations_debug.json", "w", encoding="utf-8") as f:
            json.dump(all_mutations, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Все мутации сохранены в: {OUTPUT_DIR}/all_mutations_debug.json")
        
        if not archive_game_nrs:
            print("\n❌ Список игр не получен.")
            print("   Проверьте файл all_mutations_debug.json")
            print("   Возможно, мутация со списком называется иначе.")
            
            # Альтернатива: ручной режим
            print("\n🔄 Переключение в ручной режим...")
            print("👉 Откройте вкладку Archive -> 4 Player в браузере")
            print("👉 Кликните на 5-10 партий для загрузки PGN4")
            await async_input("Нажмите Enter после кликов...")
            
            print(f"\n⏳ Ожидание PGN4 (20 сек)...")
            await asyncio.sleep(20)
        
        # ========== Загрузка PGN4 ==========
        if archive_game_nrs:
            print(f"\n📊 В архиве найдено {len(archive_game_nrs)} партий!")
            print(f"🚀 Загрузка PGN4 для всех партий...")
            
            total = len(archive_game_nrs)
            for i, game_nr in enumerate(archive_game_nrs, 1):
                if game_nr in pgn_games:
                    continue
                    
                js_load_game = f"""
                (() => {{
                    for (let ws of window.__intercepted_ws) {{
                        if (ws.readyState === 1) {{
                            ws.send(JSON.stringify({{
                                action: "game",
                                data: {{ gameNr: {game_nr}, action: "join" }},
                                windowId: "{PLAYER_ID}_{query_id}"
                            }}));
                            return true;
                        }}
                    }}
                    return false;
                }})();
                """
                await page.evaluate(js_load_game)
                
                if i % 10 == 0:
                    print(f"  📥 [{i}/{total}] Игра {game_nr} (PGN4: {len(pgn_games)})")
                
                await asyncio.sleep(DELAY_BETWEEN_GAMES)
            
            print(f"\n⏳ Ожидание последних ответов (30 сек)...")
            await asyncio.sleep(30)
        
        # ========== Итоги ==========
        print("\n" + "="*60)
        print("✅ ГОТОВО!")
        print("="*60)
        print(f"📋 Всего партий в архиве: {len(archive_game_nrs)}")
        print(f"♟️ Загружено PGN4 файлов: {len(pgn_games)}")
        print(f"📂 Папка: {OUTPUT_DIR.absolute()}")
        print("="*60)
        
        # Сохраняем статистику
        stats = {
            "total_games": len(archive_game_nrs),
            "downloaded_pgn": len(pgn_games),
            "missing_pgn": len(archive_game_nrs) - len(pgn_games),
            "all_mutations_count": len(all_mutations),
        }
        with open(OUTPUT_DIR / "download_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        
        if len(pgn_games) > 0:
            with open(OUTPUT_DIR / "games_metadata.json", "w", encoding="utf-8") as f:
                json.dump(game_metadata, f, indent=2, ensure_ascii=False, default=str)
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())