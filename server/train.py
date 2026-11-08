import os

# Важно выставить до импорта torch / numpy.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    sys.path.insert(0, "/kaggle/working")

sys.path.insert(0, "/kaggle/working")

import glob
import math
import time
import shutil
import random
import queue as pyqueue

import numpy as np
import torch
import torch.optim as optim
import torch.multiprocessing as tmp
from torch.optim.lr_scheduler import LambdaLR

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from config import (
    WORKDIR,
    LOGDIR,
    LATEST_MODEL_PATH,
    BEST_MODEL_PATH,
    CHECKPOINT_PATH,
    INPUT_MODEL_PATTERNS,
    MODEL_CFG,
    WORKER_CFG,
    TRAIN_CFG,
)

from model_utils import create_model, load_model_weights

from selfplay_utils import (
    ReplayBufferSparse,
    augment_sparse_batch,
    sparse_policy_loss,
    soft_value_loss,
    policy_metrics,
    self_play_worker,
)


# ============================================================
# Utils
# ============================================================

def set_global_seeds(seed: int = 12345):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_save(obj, path: str):
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def find_input_model():
    for pattern in INPUT_MODEL_PATTERNS:
        files = glob.glob(pattern, recursive=True)

        if files:
            files = sorted(files)
            return files[-1]

    return None


def get_hf_token(cfg: dict):
    # Сначала пробуем Kaggle Secrets.
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret(cfg["hf_secret_name"])
        if token:
            return token
    except Exception:
        pass

    # Потом переменные окружения.
    return os.getenv("HF_TOKEN", None)


def download_best_model_from_hf(cfg: dict):
    """
    Скачивает best_model.pt из HF в /kaggle/working/input_model.pt.
    """
    if not cfg.get("download_best_from_hf", False):
        return None

    token = get_hf_token(cfg)

    if not token:
        print("[HF] no token found, download skipped", flush=True)
        return None

    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=cfg["hf_repo_id"],
            filename=cfg["hf_model_filename"],
            repo_type="dataset",
            token=token,
        )

        dst = os.path.join(WORKDIR, "input_model.pt")
        shutil.copy(path, dst)

        print(f"[HF] downloaded {cfg['hf_repo_id']}/{cfg['hf_model_filename']} -> {dst}", flush=True)

        return dst

    except Exception as e:
        print(f"[HF] download failed: {e}", flush=True)
        return None


def upload_to_hf(cfg: dict, local_path: str, repo_path: str):
    if not cfg.get("upload_hf", False):
        return

    if not os.path.exists(local_path):
        return

    token = get_hf_token(cfg)

    if not token:
        print("[HF] no token found, upload skipped", flush=True)
        return

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)

        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=cfg["hf_repo_id"],
            repo_type="dataset",
        )

        print(f"[HF] uploaded {local_path} -> {repo_path}", flush=True)

    except Exception as e:
        print(f"[HF] upload failed: {e}", flush=True)


# ============================================================
# Model / checkpoint initialization
# ============================================================

def initialize_model(device: torch.device, cfg: dict):
    model = create_model(
        planes=MODEL_CFG["planes"],
        blocks=MODEL_CFG["blocks"],
        channels=MODEL_CFG["channels"],
    )

    model.to(device)

    ckpt = None

    if cfg.get("resume", True) and os.path.exists(CHECKPOINT_PATH):
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
            print("[init] checkpoint found", flush=True)
        except Exception as e:
            print(f"[init] failed to load checkpoint: {e}", flush=True)
            ckpt = None

    if ckpt is not None and "model" in ckpt:
        load_model_weights(model, ckpt["model"], verbose=True)
        print("[init] loaded model weights from checkpoint", flush=True)

    elif os.path.exists(LATEST_MODEL_PATH):
        sd = torch.load(LATEST_MODEL_PATH, map_location="cpu", weights_only=True)
        load_model_weights(model, sd, verbose=True)
        print("[init] loaded latest_model.pt", flush=True)

    else:
        input_model = find_input_model()

        if input_model is None:
            input_model = download_best_model_from_hf(cfg)

        if input_model is not None:
            try:
                sd = torch.load(input_model, map_location="cpu", weights_only=True)
            except Exception:
                sd = torch.load(input_model, map_location="cpu", weights_only=False)

            if isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]

            load_model_weights(model, sd, verbose=True)
            print(f"[init] loaded input model: {input_model}", flush=True)

        else:
            print("[init] no model found, starting from random weights", flush=True)

    # Сохраняем latest, чтобы воркеры могли его подхватить.
    atomic_save(model.state_dict(), LATEST_MODEL_PATH)

    if not os.path.exists(BEST_MODEL_PATH):
        shutil.copy(LATEST_MODEL_PATH, BEST_MODEL_PATH)

    return model, ckpt


def save_checkpoint(model, optimizer, scheduler, scaler, step: int):
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "model_cfg": MODEL_CFG,
    }

    atomic_save(ckpt, CHECKPOINT_PATH)


# ============================================================
# LR schedule
# ============================================================

def make_scheduler(optimizer, cfg: dict):
    warmup = int(cfg["warmup_steps"])
    total = int(cfg["total_steps"])
    base_lr = float(cfg["lr"])
    eta_min = 1e-5
    ratio = eta_min / base_lr

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return max(1e-6 / base_lr, float(step + 1) / float(max(1, warmup)))

        if step >= total:
            return ratio

        t = step - warmup
        T = max(1, total - warmup)

        return ratio + 0.5 * (1.0 - ratio) * (1.0 + math.cos(math.pi * t / T))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ============================================================
# Workers
# ============================================================

def create_workers(cfg: dict, data_queue):
    workers = []

    num_workers = int(cfg["num_workers"])
    selfplay_gpus = cfg["selfplay_gpus"]

    for i in range(num_workers):
        gpu_id = selfplay_gpus[i % len(selfplay_gpus)]

        p = tmp.Process(
            target=self_play_worker,
            args=(
                i,
                gpu_id,
                LATEST_MODEL_PATH,
                data_queue,
                10_000_000,
                WORKER_CFG,
            ),
        )

        p.daemon = True
        p.start()

        workers.append(p)

    return workers


def restart_workers(workers, cfg: dict, data_queue):
    selfplay_gpus = cfg["selfplay_gpus"]

    for i, p in enumerate(workers):
        if not p.is_alive():
            print(f"[workers] worker {i} died, restarting", flush=True)

            try:
                p.terminate()
                p.join(timeout=5)
            except Exception:
                pass

            gpu_id = selfplay_gpus[i % len(selfplay_gpus)]

            new_p = tmp.Process(
                target=self_play_worker,
                args=(
                    i,
                    gpu_id,
                    LATEST_MODEL_PATH,
                    data_queue,
                    10_000_000,
                    WORKER_CFG,
                ),
            )

            new_p.daemon = True
            new_p.start()

            workers[i] = new_p


# ============================================================
# Data queue / buffer
# ============================================================

def collect_queue(data_queue, buffer: ReplayBufferSparse, max_items: int) -> int:
    collected = 0

    while max_items is None or collected < max_items:
        try:
            item = data_queue.get_nowait()
        except pyqueue.Empty:
            break

        buffer.push(*item)
        collected += 1

    return collected


def warmup_buffer(
    buffer: ReplayBufferSparse,
    data_queue,
    workers,
    cfg: dict,
    target: int = 5000,
    timeout: float = 3600.0,
):
    print(f"[warmup] waiting for {target} positions...", flush=True)

    start = time.time()
    last_print = 0

    while len(buffer) < target:
        restart_workers(workers, cfg, data_queue)

        collected = collect_queue(data_queue, buffer, 512)

        if len(buffer) - last_print >= 500:
            print(f"[warmup] {len(buffer)} / {target}", flush=True)
            last_print = len(buffer)

        if collected == 0:
            if time.time() - start > timeout:
                raise RuntimeError("Warmup timeout: no data from workers")

            time.sleep(0.5)

    print("[warmup] buffer ready", flush=True)


# ============================================================
# Training step
# ============================================================

def train_step(
    model,
    optimizer,
    scheduler,
    scaler,
    buffer: ReplayBufferSparse,
    cfg: dict,
    device: torch.device,
):
    model.train()

    accum = int(cfg["accum_steps"])
    batch_size = int(cfg["batch_size"])

    metrics = {
        "loss": 0.0,
        "loss_p": 0.0,
        "loss_v": 0.0,
        "loss_v_ce": 0.0,
        "loss_v_mse": 0.0,
        "top1": 0.0,
        "top3": 0.0,
        "entropy": 0.0,
    }

    optimizer.zero_grad(set_to_none=True)

    for micro in range(accum):
        states, policy_idx, policy_prob, masks, values_soft = buffer.sample(batch_size)

        s, idx, prob, m, v = augment_sparse_batch(
            states,
            policy_idx,
            policy_prob,
            masks,
            values_soft,
        )

        s = s.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        prob = prob.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)
        v = v.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            pred_p, pred_v = model(s)

            loss_p = sparse_policy_loss(pred_p, idx, prob, m)

            loss_v, loss_v_ce, loss_v_mse = soft_value_loss(
                pred_v,
                v,
                mse_weight=float(cfg["value_mse_weight"]),
            )

            loss = (loss_p + loss_v) / accum

        scaler.scale(loss).backward()

        with torch.no_grad():
            top1, top3, entropy = policy_metrics(pred_p, idx, prob, m)

        metrics["loss"] += (loss_p.item() + loss_v.item()) / accum
        metrics["loss_p"] += loss_p.item() / accum
        metrics["loss_v"] += loss_v.item() / accum
        metrics["loss_v_ce"] += loss_v_ce.item() / accum
        metrics["loss_v_mse"] += loss_v_mse.item() / accum
        metrics["top1"] += top1 / accum
        metrics["top3"] += top3 / accum
        metrics["entropy"] += entropy / accum

    scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        float(cfg["grad_clip"]),
    )

    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    model.eval()

    metrics["grad_norm"] = float(grad_norm.item())
    metrics["grad_scale"] = float(scaler.get_scale())
    metrics["lr"] = float(optimizer.param_groups[0]["lr"])

    return metrics


# ============================================================
# Main training loop
# ============================================================

def run_training():
    os.makedirs(LOGDIR, exist_ok=True)

    cfg = TRAIN_CFG

    set_global_seeds(12345)

    device = torch.device(f"cuda:{cfg['trainer_gpu']}")
    torch.cuda.set_device(device)

    if cfg.get("trainer_memory_fraction") is not None:
        try:
            torch.cuda.set_per_process_memory_fraction(
                float(cfg["trainer_memory_fraction"]),
                device,
            )
        except Exception as e:
            print(f"[init] failed to set trainer memory fraction: {e}", flush=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("[init] initializing model...", flush=True)

    model, ckpt = initialize_model(device, cfg)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    scheduler = make_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda")

    start_step = 1

    if ckpt is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"])
            start_step = int(ckpt.get("step", 0)) + 1

            print(f"[init] resumed from checkpoint step {start_step}", flush=True)

        except Exception as e:
            print(f"[init] failed to load optimizer/scheduler/scaler: {e}", flush=True)
            print("[init] continuing with fresh optimizer state", flush=True)

            optimizer = optim.AdamW(
                model.parameters(),
                lr=float(cfg["lr"]),
                weight_decay=float(cfg["weight_decay"]),
            )

            scheduler = make_scheduler(optimizer, cfg)
            scaler = torch.amp.GradScaler("cuda")

            start_step = 1

    # После загрузки checkpoint снова сохраняем latest для workers.
    atomic_save(model.state_dict(), LATEST_MODEL_PATH)

    buffer = ReplayBufferSparse(
        capacity=int(cfg["buffer_capacity"]),
        policy_k=int(WORKER_CFG.get("policy_k", 192)),
        planes=int(MODEL_CFG["planes"]),
    )

    tmp.set_start_method("spawn", force=True)

    data_queue = tmp.Queue(maxsize=int(cfg["queue_maxsize"]))

    print("[init] starting workers...", flush=True)

    workers = create_workers(cfg, data_queue)

    warmup_buffer(
        buffer=buffer,
        data_queue=data_queue,
        workers=workers,
        cfg=cfg,
        target=max(int(cfg["batch_size"]) * 4, 2048),
        timeout=7200.0,
    )

    writer = None

    if cfg.get("tensorboard", True) and SummaryWriter is not None:
        writer = SummaryWriter(LOGDIR)

    print("[train] starting training loop", flush=True)

    step = 0

    try:
        for step in range(start_step, int(cfg["total_steps"]) + 1):
            restart_workers(workers, cfg, data_queue)

            collected = collect_queue(
                data_queue,
                buffer,
                max_items=int(cfg["batch_size"]) * 2,
            )

            if len(buffer) < int(cfg["batch_size"]):
                time.sleep(0.5)
                continue

            metrics = train_step(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                buffer=buffer,
                cfg=cfg,
                device=device,
            )

            if step % int(cfg["log_every"]) == 0:
                qsize = -1

                try:
                    qsize = data_queue.qsize()
                except Exception:
                    pass

                alive = sum(1 for p in workers if p.is_alive())

                msg = (
                    f"step {step} | "
                    f"loss {metrics['loss']:.3f} | "
                    f"p {metrics['loss_p']:.3f} | "
                    f"v {metrics['loss_v']:.3f} | "
                    f"top1 {metrics['top1']:.3f} | "
                    f"top3 {metrics['top3']:.3f} | "
                    f"ent {metrics['entropy']:.3f} | "
                    f"lr {metrics['lr']:.6f} | "
                    f"buf {len(buffer)} | "
                    f"q {qsize} | "
                    f"col {collected} | "
                    f"w {alive}/{len(workers)} | "
                    f"scale {metrics['grad_scale']:.0f}"
                )

                print(msg, flush=True)

                if writer is not None:
                    writer.add_scalar("Loss/total", metrics["loss"], step)
                    writer.add_scalar("Loss/policy", metrics["loss_p"], step)
                    writer.add_scalar("Loss/value", metrics["loss_v"], step)
                    writer.add_scalar("Loss/value_ce", metrics["loss_v_ce"], step)
                    writer.add_scalar("Loss/value_mse", metrics["loss_v_mse"], step)
                    writer.add_scalar("Policy/top1", metrics["top1"], step)
                    writer.add_scalar("Policy/top3", metrics["top3"], step)
                    writer.add_scalar("Policy/entropy", metrics["entropy"], step)
                    writer.add_scalar("Optim/lr", metrics["lr"], step)
                    writer.add_scalar("Optim/grad_norm", metrics["grad_norm"], step)
                    writer.add_scalar("Optim/grad_scale", metrics["grad_scale"], step)
                    writer.add_scalar("Data/buffer_size", len(buffer), step)
                    writer.add_scalar("Data/queue_size", qsize, step)
                    writer.add_scalar("Data/collected", collected, step)
                    writer.add_scalar("Workers/alive", alive, step)

            if step % int(cfg["save_every"]) == 0:
                print(f"[save] step {step}", flush=True)

                save_checkpoint(model, optimizer, scheduler, scaler, step)
                atomic_save(model.state_dict(), LATEST_MODEL_PATH)

                if cfg.get("copy_latest_to_best_each_save", True):
                    shutil.copy(LATEST_MODEL_PATH, BEST_MODEL_PATH)

                upload_to_hf(cfg, LATEST_MODEL_PATH, "latest_model.pt")
                upload_to_hf(cfg, BEST_MODEL_PATH, "best_model.pt")

                if cfg.get("upload_checkpoint", False):
                    upload_to_hf(cfg, CHECKPOINT_PATH, "checkpoint.pt")

        print("[train] finished total_steps", flush=True)

    except KeyboardInterrupt:
        print("[train] KeyboardInterrupt, saving checkpoint...", flush=True)
        save_checkpoint(model, optimizer, scheduler, scaler, step)
        atomic_save(model.state_dict(), LATEST_MODEL_PATH)

    except Exception as e:
        print(f"[train] exception: {e}", flush=True)

        try:
            save_checkpoint(model, optimizer, scheduler, scaler, step)
            atomic_save(model.state_dict(), LATEST_MODEL_PATH)
        except Exception:
            pass

        raise

    finally:
        print("[train] terminating workers", flush=True)

        for p in workers:
            try:
                p.terminate()
                p.join(timeout=5)
            except Exception:
                pass

        if writer is not None:
            writer.close()


if __name__ == "__main__":
    run_training()