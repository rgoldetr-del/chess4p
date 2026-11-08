import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        mid = max(1, channels // reduction)

        self.fc1 = nn.Conv2d(channels, mid, kernel_size=1)
        self.fc2 = nn.Conv2d(mid, channels, kernel_size=1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)

        out += residual
        return F.relu(out)


class ChessNet100(nn.Module):
    """
    обратно совместимая модель:
    - вход теперь 100 плоскостей;
    - архитектура пока 19x192, чтобы можно было подхватить старые веса;
    - policy head остаётся совместимым со старым action space 38416;
    - value head остаётся [B, 4, 4].
    """

    def __init__(
        self,
        planes: int = 100,
        blocks: int = 19,
        channels: int = 192,
        policy_channels: int = 64,
        value_channels: int = 32,
    ):
        super().__init__()

        self.planes = planes
        self.blocks = blocks
        self.channels = channels

        self.conv_init = nn.Sequential(
            nn.Conv2d(planes, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        self.res_blocks = nn.Sequential(
            *[ResBlock(channels) for _ in range(blocks)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, policy_channels, kernel_size=1),
            nn.BatchNorm2d(policy_channels),
            nn.ReLU(),
        )

        self.policy_from = nn.Conv2d(policy_channels, 196, kernel_size=1)
        self.policy_to = nn.Conv2d(policy_channels, 196, kernel_size=1)

        self.value_head = nn.Sequential(
            nn.Conv2d(channels, value_channels, kernel_size=1),
            nn.BatchNorm2d(value_channels),
            nn.ReLU(),
        )

        self.value_fc = nn.Sequential(
            nn.Linear(value_channels * 14 * 14, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 4 * 4),
        )

    def forward(self, x):
        x = self.conv_init(x)
        x = self.res_blocks(x)

        # Policy
        p = self.policy_head(x)

        p_from = self.policy_from(p).flatten(2)  # [B, 196, 196]
        p_to = self.policy_to(p).flatten(2)      # [B, 196, 196]

        # [B, to, from] -> flatten => from + to*196
        policy_logits = torch.bmm(p_to.transpose(1, 2), p_from).flatten(1)
        policy_logits = policy_logits / 14.0

        # Value
        v = self.value_head(x)
        v = v.flatten(1)
        v = self.value_fc(v)
        v = v.view(-1, 4, 4)

        return policy_logits, v


def clean_state_dict(state_dict):
    """
    Убирает module. от DataParallel и мусорные префиксы.
    """
    out = {}

    for k, v in state_dict.items():
        nk = k.replace("module.", "")
        out[nk] = v

    return out


def upgrade_conv_input_weight(
    old_weight: torch.Tensor,
    new_planes: int,
    init_scale: float = 0.01,
) -> torch.Tensor:
    """
    Старый conv_init.weight: [C, 73, 3, 3]
    Новый conv_init.weight: [C, 100, 3, 3]

    Первые 73 канала копируем, новые 27 инициализируем маленькими весами.
    """
    out_channels, old_planes, kh, kw = old_weight.shape

    new_weight = torch.zeros(
        (out_channels, new_planes, kh, kw),
        dtype=old_weight.dtype,
    )

    copy_planes = min(old_planes, new_planes)
    new_weight[:, :copy_planes, :, :] = old_weight[:, :copy_planes, :, :]

    if new_planes > old_planes:
        new_part = new_weight[:, old_planes:, :, :]
        nn.init.kaiming_normal_(new_part, nonlinearity="relu")
        new_part.mul_(init_scale)

    return new_weight


def load_model_weights(
    model: nn.Module,
    state_dict: dict,
    verbose: bool = True,
) -> dict:
    """
    Умная загрузка весов:
    - чистит module.;
    - апгрейдит conv_init 73 -> 100;
    - пропускает несовместимые слои;
    - полезно для будущего перехода на 30x256.
    """
    sd = clean_state_dict(state_dict)
    model_sd = model.state_dict()

    loaded = []
    skipped = []
    upgraded = []

    # Апгрейд входного conv, если нужно.
    conv_key = "conv_init.0.weight"

    if conv_key in sd and conv_key in model_sd:
        old_w = sd[conv_key]
        new_w_expected = model_sd[conv_key]

        if old_w.shape != new_w_expected.shape:
            if (
                len(old_w.shape) == 4
                and len(new_w_expected.shape) == 4
                and old_w.shape[0] == new_w_expected.shape[0]
                and old_w.shape[2:] == new_w_expected.shape[2:]
            ):
                sd[conv_key] = upgrade_conv_input_weight(
                    old_w,
                    new_planes=new_w_expected.shape[1],
                )
                upgraded.append(conv_key)

    for k, v in sd.items():
        if k not in model_sd:
            skipped.append((k, "missing_in_model"))
            continue

        if model_sd[k].shape != v.shape:
            skipped.append((k, f"shape {tuple(v.shape)} != {tuple(model_sd[k].shape)}"))
            continue

        model_sd[k] = v
        loaded.append(k)

    model.load_state_dict(model_sd, strict=False)

    if verbose:
        print(f"[model_utils] loaded tensors: {len(loaded)}")
        print(f"[model_utils] upgraded tensors: {len(upgraded)}")
        print(f"[model_utils] skipped tensors: {len(skipped)}")

        if upgraded:
            print("[model_utils] upgraded:")
            for k in upgraded[:10]:
                print("   ", k)

        if skipped:
            print("[model_utils] skipped:")
            for k, reason in skipped[:20]:
                print(f"    {k}: {reason}")

    return {
        "loaded": loaded,
        "upgraded": upgraded,
        "skipped": skipped,
    }


def create_model(
    planes: int = 100,
    blocks: int = 19,
    channels: int = 192,
) -> ChessNet100:
    return ChessNet100(
        planes=planes,
        blocks=blocks,
        channels=channels,
    )