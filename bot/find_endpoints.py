"""
find_endpoints.py — автоматический перехват всех запросов chess.com.
Сохраняет всё в endpoints_log.jsonl для анализа.
"""
import asyncio
import json
from datetime import datetime
from playwright.async_api import async_playwright

LOG_FILE = "endpoints_log.jsonl"

async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        def on_request(request):
            url = request.url
            # Фильтруем статику, чтобы не тонуть в мусоре
            skip = [".png", ".jpg", ".css", ".woff", ".svg", 
                    "google", "analytics", "ads", "facebook"]
            if any(s in url.lower() for s in skip):
                return
            
            entry = {
                "type": "REQUEST",
                "method": request.method,
                "url": url,
                "time": datetime.now().isoformat(),
            }
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
            # Печатаем подозрительные (с game/4p/pgn/live)
            keywords = ["game", "4p", "pgn", "live", "multiplayer", "four"]
            if any(k in url.lower() for k in keywords):
                print(f"🔥 {request.method} {url}")

        async def on_response(response):
            url = response.url
            keywords = ["game", "4p", "pgn", "live", "multiplayer", "four"]
            if not any(k in url.lower() for k in keywords):
                return
            try:
                body = await response.text()
                entry = {
                    "type": "RESPONSE",
                    "url": url,
                    "status": response.status,
                    "body_preview": body[:500],
                    "time": datetime.now().isoformat(),
                }
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                
                # Если нашли PGN — кричим
                if "1." in body and ("[" in body or "Event" in body):
                    print(f"🎯 PGN FOUND: {url}")
            except Exception:
                pass

        page.on("request", on_request)
        page.on("response", on_response)

        print("=" * 60)
        print("Откроется браузер. Ваша задача:")
        print("1. Войдите в аккаунт chess.com")
        print("2. Перейдите в 4-Player Chess")
        print("3. Откройте любую сыгранную партию (архив)")
        print("4. Полистайте архив, пооткрывайте игры")
        print("5. Вернитесь сюда и нажмите Ctrl+C")
        print("=" * 60)

        await page.goto("https://www.chess.com/4-player-chess")
        
        # Держим браузер открытым
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        
        await browser.close()
        print(f"\n✅ Лог сохранён: {LOG_FILE}")

asyncio.run(main())