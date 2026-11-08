import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import chess4p_ai
from engine_wrapper import EngineWrapper
from selfplay_utils import mcts_search

def get_vram_mb():
    if not torch.cuda.is_available():
        return 0, 0
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1024**2
    max_alloc = torch.cuda.max_memory_allocated() / 1024**2
    return alloc, max_alloc

def main():
    print("=== DIAG 03: MCTS & Model Benchmark (GTX 1060 3GB) ===")
    print("Загружаю модель и грею GPU...")
    engine = EngineWrapper()
    
    game = chess4p_ai.PyGameState()
    
    # Делаем пару случайных ходов, чтобы получить позицию мидгейма (там больше легальных ходов)
    for _ in range(6):
        legal = game.get_legal_move_indices()
        if legal:
            game.make_move_fast_idx(legal[0])
            
    print("\nТестируем разные MCTS_BATCH_SIZE...")
    print(f"{'Batch':<6} | {'Sims':<6} | {'Time (s)':<9} | {'NPS':<8} | {'VRAM Max (MB)':<14} | {'Status'}")
    print("-" * 75)
    
    for batch_size in [4, 8, 16, 32]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        cfg = {
            "min_sims": 100,
            "max_sims": 2000,
            "batch_size": batch_size,
            "max_time": 5.0, # Тестируем 5 секунд
            "max_depth": 256,
            "forced_depth": 32,
            "c_init": 1.25,
            "c_base": 19652.0,
            "fpu_reduction": 0.3,
            "virtual_loss": 1.0,
            "noise_eps": 0.0,
            "early_stop_top_share": 0.99,
        }
        
        start_time = time.perf_counter()
        try:
            res = mcts_search(game, engine.model, engine.device, cfg=cfg)
            elapsed = time.perf_counter() - start_time
            
            _, max_vram = get_vram_mb()
            
            if res:
                _, _, _, _, _, dbg = res
                sims = dbg.get("simulations", 0)
                nps = int(sims / elapsed) if elapsed > 0 else 0
                status = "OK"
                # У GTX 1060 3GB всего ~3000MB. Windows забирает ~300-500MB. 
                # Безопасный лимит для PyTorch ~ 2400-2500 MB.
                if max_vram > 2500: 
                    status = "⚠ OOM RISK"
            else:
                sims = 0
                nps = 0
                status = "FAIL"
                
            print(f"{batch_size:<6} | {sims:<6} | {elapsed:<9.2f} | {nps:<8} | {max_vram:<14.1f} | {status}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{batch_size:<6} | {'CRASH':<6} | {'OOM':<9} | {'---':<8} | {'---':<14} | ❌ CUDA OOM")
                torch.cuda.empty_cache()
            else:
                raise e

    print("\n✅ Бенчмарк завершен.")
    print("Жду результат, чтобы прописать идеальные параметры в config.py!")

if __name__ == "__main__":
    main()