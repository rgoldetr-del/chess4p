"""
analyze_ws.py — анализ перехваченных WebSocket сообщений chess.com 4PC.
Ищет сообщения с данными игры (ходы, позиции, PGN).
"""

import json
import sys
from pathlib import Path
from collections import Counter

WS_FILE = Path("api_discovery/websocket_messages.json")
OUTPUT_DIR = Path("api_discovery")


def analyze_messages():
    if not WS_FILE.exists():
        print(f"❌ Файл не найден: {WS_FILE}")
        return
    
    with open(WS_FILE, "r", encoding="utf-8") as f:
        messages = json.load(f)
    
    print(f"📨 Всего сообщений: {len(messages)}")
    
    # Статистика по направлениям
    directions = Counter(m["direction"] for m in messages)
    print(f"   received: {directions['received']}, sent: {directions['sent']}")
    
    # Статистика по URL
    urls = Counter()
    for m in messages:
        # URL может быть в metadata
        urls[m.get("url", "unknown")] += 1
    
    print(f"\n🔌 WebSocket URLs:")
    for url, count in urls.most_common():
        print(f"   {url}: {count} сообщений")
    
    # Ищем интересные сообщения
    print(f"\n🔍 Поиск сообщений с данными игры...")
    
    keywords = ["move", "game", "pgn", "position", "fen", "board", 
                "turn", "player", "capture", "check", "mate", "castle",
                "four", "4p", "variant", "red", "blue", "green", "yellow"]
    
    interesting = []
    binary_count = 0
    json_count = 0
    text_count = 0
    
    for i, msg in enumerate(messages):
        data = msg.get("data", "")
        data_str = str(data)
        
        # Подсчет типов
        if isinstance(data, bytes):
            binary_count += 1
        elif isinstance(data, str):
            if data.startswith("{") or data.startswith("["):
                json_count += 1
            else:
                text_count += 1
        
        # Поиск по ключевым словам
        data_lower = data_str.lower()
        matched_keywords = [kw for kw in keywords if kw in data_lower]
        
        if matched_keywords:
            interesting.append({
                "index": i,
                "direction": msg["direction"],
                "keywords": matched_keywords,
                "preview": data_str[:300],
                "full": data_str,
                "time": msg.get("time", ""),
            })
    
    print(f"\n📊 Типы сообщений:")
    print(f"   JSON: {json_count}")
    print(f"   Text: {text_count}")
    print(f"   Binary: {binary_count}")
    
    print(f"\n🎯 Интересных сообщений: {len(interesting)}")
    
    if interesting:
        # Сохраняем интересные сообщения
        with open(OUTPUT_DIR / "ws_interesting.json", "w", encoding="utf-8") as f:
            json.dump(interesting, f, indent=2, ensure_ascii=False)
        
        print(f"   Сохранено в: {OUTPUT_DIR / 'ws_interesting.json'}")
        
        # Выводим первые 10
        print(f"\n{'='*70}")
        print("ТОП-10 интересных сообщений:")
        print('='*70)
        for item in interesting[:10]:
            print(f"\n[{item['index']}] {item['direction']} | keywords: {item['keywords']}")
            print(f"  Time: {item['time']}")
            print(f"  Preview: {item['preview'][:200]}")
            print("-" * 70)
    
    # Анализируем все JSON сообщения
    print(f"\n📦 Анализ JSON сообщений...")
    json_messages = []
    for i, msg in enumerate(messages):
        data = msg.get("data", "")
        if isinstance(data, str) and (data.startswith("{") or data.startswith("[")):
            try:
                parsed = json.loads(data)
                json_messages.append({
                    "index": i,
                    "direction": msg["direction"],
                    "parsed": parsed,
                    "time": msg.get("time", ""),
                })
            except json.JSONDecodeError:
                pass
    
    print(f"   Успешно распарсено: {len(json_messages)}")
    
    if json_messages:
        with open(OUTPUT_DIR / "ws_json_parsed.json", "w", encoding="utf-8") as f:
            json.dump(json_messages, f, indent=2, ensure_ascii=False, default=str)
        
        # Анализируем структуру JSON
        print(f"\n🔎 Анализ структуры JSON...")
        all_keys = Counter()
        all_types = Counter()
        
        def walk_json(obj, prefix=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    full_key = f"{prefix}.{k}" if prefix else k
                    all_keys[full_key] += 1
                    all_types[type(v).__name__] += 1
                    walk_json(v, full_key)
            elif isinstance(obj, list):
                for item in obj:
                    walk_json(item, prefix + "[]")
        
        for jm in json_messages:
            walk_json(jm["parsed"])
        
        print(f"\n   Все ключи JSON:")
        for key, count in all_keys.most_common(30):
            print(f"      {key}: {count}")
    
    # Анализируем бинарные сообщения (могут быть Protobuf)
    if binary_count > 0:
        print(f"\n🔢 Бинарные сообщения: {binary_count}")
        print("   Возможно Protobuf или кастомный бинарный протокол")
        print("   Для анализа нужен protobuf decoder")
        
        # Сохраняем первые несколько бинарных
        binary_samples = []
        for msg in messages:
            data = msg.get("data", "")
            if isinstance(data, bytes):
                binary_samples.append({
                    "direction": msg["direction"],
                    "length": len(data),
                    "hex_preview": data[:64].hex(),
                    "time": msg.get("time", ""),
                })
                if len(binary_samples) >= 5:
                    break
        
        with open(OUTPUT_DIR / "ws_binary_samples.json", "w") as f:
            json.dump(binary_samples, f, indent=2)


if __name__ == "__main__":
    analyze_messages()