"""
Chess4P Neural Network Architecture.
30 residual blocks with 256 channels, SE-attention, and auxiliary heads.
Supports Transformer policy head (experimental).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    
    def __init__(self, channels: int, se_channels: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, se_channels)
        self.fc2 = nn.Linear(se_channels, channels * 2)  # Scale + Bias
        self.channels = channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        
        # Global average pooling
        z = x.mean(dim=(2, 3))  # [B, C]
        
        # FC layers
        z = F.relu(self.fc1(z))
        z = self.fc2(z)  # [B, 2C]
        
        # Split into scale and bias
        scale, bias = z[:, :c], z[:, c:]
        scale = torch.sigmoid(scale).view(b, c, 1, 1)
        bias = bias.view(b, c, 1, 1)
        
        return x * scale + bias


class ResBlock(nn.Module):
    """Residual block with SE-attention."""
    
    def __init__(self, channels: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, se_channels=channels // 16)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        
        out = F.relu(out + identity)
        return out


class PolicyHead(nn.Module):
    """Standard factorized policy head."""
    
    def __init__(self, channels: int = 256, policy_channels: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(channels, policy_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(policy_channels)
        
        # From squares (196) and To squares (196)
        self.policy_from = nn.Linear(policy_channels * 196, 196)
        self.policy_to = nn.Linear(policy_channels * 196, 196)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        
        x = F.relu(self.bn(self.conv(x)))  # [B, PC, 14, 14]
        x_flat = x.flatten(1)  # [B, PC*196]
        
        p_from = self.policy_from(x_flat)  # [B, 196]
        p_to = self.policy_to(x_flat)  # [B, 196]
        
        # Outer product for all from->to combinations
        policy = torch.bmm(p_to.unsqueeze(2), p_from.unsqueeze(1))  # [B, 196, 196]
        return policy.flatten(1)  # [B, 38416]


class TransformerPolicyHead(nn.Module):
    """Experimental transformer-based policy head."""
    
    def __init__(self, channels: int = 256, heads: int = 8, layers: int = 2):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, 196, channels))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=heads,
            dim_feedforward=channels * 2,
            batch_first=True,
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.to_logits = nn.Linear(channels, 196)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        
        # [B, C, 14, 14] -> [B, 196, C]
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed.expand(b, -1, -1)
        
        encoded = self.encoder(x)  # [B, 196, C]
        logits = self.to_logits(encoded)  # [B, 196, 196]
        
        return logits.flatten(1)  # [B, 38416]


class ValueHead(nn.Module):
    """Value head returning per-player values."""
    
    def __init__(self, channels: int = 256, value_channels: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(channels, value_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(value_channels)
        
        # Flatten: value_channels * 14 * 14 = 6272
        self.fc1 = nn.Linear(value_channels * 14 * 14, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 16)  # 4 players * 4 outcome classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        
        x = F.relu(self.bn(self.conv(x)))  # [B, VC, 14, 14]
        x = x.flatten(1)  # [B, VC*196]
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # [B, 16]
        
        return x.view(b, 4, 4)  # [B, 4, 4]


class ThreatHead(nn.Module):
    """Auxiliary head for threat level prediction per opponent."""
    
    def __init__(self, channels: int = 256):
        super().__init__()
        self.conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 14 * 14, 64)
        self.fc2 = nn.Linear(64, 4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        
        x = F.relu(self.bn(self.conv(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # [B, 4] in range [0, 1]
        
        return x


class MaterialHead(nn.Module):
    """Auxiliary head for material count prediction."""
    
    def __init__(self, channels: int = 256):
        super().__init__()
        self.conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 14 * 14, 64)
        self.fc2 = nn.Linear(64, 4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        
        x = F.relu(self.bn(self.conv(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Normalized material [0, 1]
        
        return x


class ChessNet(nn.Module):
    """
    Main chess neural network.
    Architecture: 30 residual blocks, 256 channels.
    """
    
    def __init__(
        self,
        num_blocks: int = 30,
        channels: int = 256,
        policy_channels: int = 64,
        value_channels: int = 32,
        use_transformer: bool = False
    ):
        super().__init__()
        
        self.num_blocks = num_blocks
        self.channels = channels
        
        # Input convolution
        self.conv_init = nn.Conv2d(73, channels, 3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(channels)
        
        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResBlock(channels) for _ in range(num_blocks)
        ])
        
        # Policy head
        if use_transformer:
            self.policy_head = TransformerPolicyHead(channels)
        else:
            self.policy_head = PolicyHead(channels, policy_channels)
        
        # Value head
        self.value_head = ValueHead(channels, value_channels)
        
        # Auxiliary heads (training only)
        self.threat_head = ThreatHead(channels)
        self.material_head = MaterialHead(channels)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for inference (policy, value only)."""
        x = F.relu(self.bn_init(self.conv_init(x)))
        
        for block in self.res_blocks:
            x = block(x)
        
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value
    
    def forward_training(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for training (includes auxiliary heads)."""
        x = F.relu(self.bn_init(self.conv_init(x)))
        
        for block in self.res_blocks:
            x = block(x)
        
        policy = self.policy_head(x)
        value = self.value_head(x)
        threat = self.threat_head(x)
        material = self.material_head(x)
        
        return policy, value, threat, material


class AveragedModel(nn.Module):
    """
    Stochastic Weight Averaging wrapper.
    Used for SWA during training.
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.module = model
        self.n_averaged = 0
        self.averaged_state = None
    
    def update_parameters(self, model: nn.Module):
        """Update averaged parameters with new model state."""
        if self.averaged_state is None:
            self.averaged_state = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
            self.n_averaged = 1
        else:
            self.n_averaged += 1
            alpha = 1.0 / self.n_averaged
            
            for k, v in model.state_dict().items():
                self.averaged_state[k].mul_(1 - alpha).add_(v.detach(), alpha=alpha)
    
    def load_averaged_weights(self):
        """Load averaged weights into the wrapped model."""
        if self.averaged_state is not None:
            self.module.load_state_dict(self.averaged_state)
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
