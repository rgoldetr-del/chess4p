"""
export_onnx.py — конвертация model.pt -> model.onnx для нативного GUI (Rust + ort).

Автоматически определяет архитектуру по форме весов:
  - ChessNet256 (30x256, 256 плоскостей, full-rank policy) — bot/config.py
  - ChessNet100 (19x192, 100 плоскостей, low-rank policy) — server/config.py

Экспортирует:
  input  : [batch, planes, 14, 14]
  policy : [batch, 38416]
  value  : [batch, 4, 4]

Использование:
    python scripts/export_onnx.py path/to/model.pt
    python scripts/export_onnx.py path/to/model.pt -o model.onnx
"""

import argparse
import os
import re
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "bot"))
sys.path.insert(0, os.path.join(REPO_ROOT, "server"))

from bot.model_utils import ChessNet256, load_state_dict_from_path  # noqa: E402
from server.model_utils import ChessNet100  # noqa: E402


class ExportNet256(torch.nn.Module):
    """Обёртка над ChessNet256 для DML-совместимого ONNX-экспорта.

    Заменяет F.adaptive_avg_pool2d(v, 4) (не экспортируется legacy-экспортёром
    при 14->4 и рождает Reshape-узлы, которые ломает DirectML) на точный
    эквивалент — два matmul с фиксированной матрицей окна усреднения.
    """

    def __init__(self, m: ChessNet256):
        super().__init__()
        self.m = m
        # Окна adaptive_avg_pool2d(14 -> 4): start = floor(i*14/4), end = ceil((i+1)*14/4)
        w = torch.zeros(4, 14)
        for i in range(4):
            s = (i * 14) // 4
            e = -(-((i + 1) * 14) // 4)  # ceil
            w[i, s:e] = 1.0 / (e - s)
        self.register_buffer("pool_w", w)

    def forward(self, x):
        m = self.m
        x = m.conv_init(x)
        x = m.res_blocks(x)

        p = m.policy_head(x)
        p = m.policy_out(p)
        policy_logits = p.flatten(1)

        v = m.value_head(x)                              # [B, C, 14, 14]
        v = torch.matmul(self.pool_w, v)                 # усреднение строк -> [B, C, 4, 14]
        v = torch.matmul(v, self.pool_w.transpose(0, 1)) # усреднение столбцов -> [B, C, 4, 4]
        v = v.flatten(1)
        v = m.value_fc(v)
        v = v.view(-1, 4, 4)
        return policy_logits, v


def detect_architecture(sd: dict) -> dict:
    """Определяет planes/blocks/channels и вариант policy head по state_dict."""
    conv_w = sd.get("conv_init.0.weight")
    if conv_w is None:
        raise RuntimeError("В checkpoint нет conv_init.0.weight — неизвестная архитектура")

    channels, planes = int(conv_w.shape[0]), int(conv_w.shape[1])

    block_ids = [
        int(m.group(1))
        for k in sd
        for m in [re.match(r"res_blocks\.(\d+)\.conv1\.weight", k)]
        if m
    ]
    blocks = (max(block_ids) + 1) if block_ids else 0

    low_rank_policy = "policy_from.weight" in sd and "policy_to.weight" in sd

    policy_channels = None
    pw = sd.get("policy_head.0.weight")
    if pw is not None:
        policy_channels = int(pw.shape[0])

    value_channels = None
    vw = sd.get("value_head.0.weight")
    if vw is not None:
        value_channels = int(vw.shape[0])

    return {
        "planes": planes,
        "blocks": blocks,
        "channels": channels,
        "low_rank_policy": low_rank_policy,
        "policy_channels": policy_channels,
        "value_channels": value_channels,
    }


def build_model(sd: dict):
    arch = detect_architecture(sd)

    if arch["low_rank_policy"]:
        model = ChessNet100(
            planes=arch["planes"],
            blocks=arch["blocks"],
            channels=arch["channels"],
            policy_channels=arch["policy_channels"] or 64,
            value_channels=arch["value_channels"] or 32,
        )
        missing, unexpected = model.load_state_dict(sd, strict=False)
    else:
        inner = ChessNet256(
            planes=arch["planes"],
            blocks=arch["blocks"],
            channels=arch["channels"],
            policy_channels=arch["policy_channels"] or 256,
            value_channels=arch["value_channels"] or 64,
        )
        missing, unexpected = inner.load_state_dict(sd, strict=False)
        model = ExportNet256(inner)

    # strict=False допускает отсутствие только НЕобязательных частей;
    # критичные слои проверяем вручную.
    real_missing = [
        k for k in missing
        if not k.startswith(("policy_from", "policy_to", "policy_out"))
        and "num_batches_tracked" not in k
    ]
    if real_missing:
        print(f"⚠ Отсутствуют слои: {real_missing[:8]}{'...' if len(real_missing) > 8 else ''}")

    model.eval()

    # Проверяем, что обёртка даёт побитово те же выходы, что исходная модель
    if not arch["low_rank_policy"]:
        with torch.no_grad():
            xin = torch.randn(2, arch["planes"], 14, 14)
            po, vo = model.m(xin)
            pw, vw = model(xin)
            if torch.allclose(po, pw, atol=1e-4) and torch.allclose(vo, vw, atol=1e-4):
                print("✅ Обёртка эквивалентна исходной модели")
            else:
                raise RuntimeError("Обёртка экспорта отличается от исходной модели!")

    return model, arch


def verify_onnx(path: str, planes: int, batch: int = 16):
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("(onnxruntime не установлен — проверка пропущена)")
        return

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = np.random.randn(batch, planes, 14, 14).astype(np.float32)
    policy, value = sess.run(None, {"input": x})
    print(f"✅ Проверка ONNX: policy {policy.shape}, value {value.shape}")


def main():
    parser = argparse.ArgumentParser(description="Export chess4p model.pt to ONNX")
    parser.add_argument("model", nargs="?", default="model.pt", help="путь к model.pt")
    parser.add_argument("-o", "--out", default=None, help="путь к выходному .onnx")
    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        # типичные места
        for cand in [
            os.path.join(REPO_ROOT, "model.pt"),
            os.path.join(REPO_ROOT, "bot", "model.pt"),
            os.path.join(REPO_ROOT, "scripts", "model.pt"),
        ]:
            if os.path.exists(cand):
                model_path = cand
                break
        else:
            print(f"❌ Файл модели не найден: {args.model}")
            sys.exit(1)

    out_path = args.out or os.path.splitext(model_path)[0] + ".onnx"

    print(f"📂 Загрузка: {model_path}")
    sd = load_state_dict_from_path(model_path)
    model, arch = build_model(sd)
    print(f"🏗 Архитектура: planes={arch['planes']}, blocks={arch['blocks']}, "
          f"channels={arch['channels']}, "
          f"policy={'low-rank (ChessNet100)' if arch['low_rank_policy'] else 'full-rank (ChessNet256)'}")

    dummy = torch.randn(16, arch["planes"], 14, 14)

    print(f"💾 Экспорт в {out_path} (статический batch=16, для DirectML) ...")
    # ВАЖНО:
    # 1) без dynamic_axes — DML не поддерживает Reshape с символьными размерностями;
    # 2) dynamo=False (legacy TorchScript-экспортёр) — даёт простой граф из
    #    стандартных операций (Conv/Relu/MatMul/Gemm), который принимает DirectML;
    #    новый dynamo-экспортёр генерирует view/reshape-узлы, на которых DML падает.
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["policy", "value"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print("✅ Экспорт завершён")

    verify_onnx(out_path, arch["planes"])


if __name__ == "__main__":
    main()