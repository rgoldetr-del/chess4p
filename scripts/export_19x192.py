"""
Export ChessNet model (19 blocks, 192 channels) to ONNX format.
This script exports only the inference heads (policy, value).
Auxiliary heads (threat, material) are NOT included in the ONNX export.
"""

import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.model import ChessNet


def export_model_19x192(
    checkpoint_path: str,
    output_path: str = "chessnet_19x192.onnx",
    use_transformer: bool = False
):
    """
    Export a ChessNet model with 19 residual blocks and 192 channels to ONNX.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        output_path: Path for the output .onnx file
        use_transformer: Whether to use transformer policy head
    """
    
    # Configuration for 19 blocks, 192 channels
    num_blocks = 19
    channels = 192
    policy_channels = 48  # Scaled proportionally from 64
    value_channels = 24   # Scaled proportionally from 32
    
    print(f"Loading model configuration: {num_blocks} blocks, {channels} channels")
    print(f"Policy channels: {policy_channels}, Value channels: {value_channels}")
    
    # Create model instance
    model = ChessNet(
        num_blocks=num_blocks,
        channels=channels,
        policy_channels=policy_channels,
        value_channels=value_channels,
        use_transformer=use_transformer
    )
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                # Try to load directly if it's just state dict
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Filter state dict to match current model (in case of mismatched keys)
        model_state = model.state_dict()
        filtered_state = {}
        for k, v in state_dict.items():
            if k in model_state and model_state[k].shape == v.shape:
                filtered_state[k] = v
            elif k in model_state:
                print(f"Skipping mismatched key: {k} (expected {model_state[k].shape}, got {v.shape})")
            else:
                print(f"Skipping unknown key: {k}")
        
        missing_keys = set(model_state.keys()) - set(filtered_state.keys())
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys}")
        
        model.load_state_dict(filtered_state, strict=False)
        print(f"Loaded checkpoint from: {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}, using random weights")
    
    model.eval()
    
    # Create dummy input (batch_size=1, 73 planes, 14x14 board)
    dummy_input = torch.randn(1, 73, 14, 14)
    
    # Verify forward pass works
    with torch.no_grad():
        policy_out, value_out = model(dummy_input)
        print(f"Model output shapes - Policy: {policy_out.shape}, Value: {value_out.shape}")
    
    # Export to ONNX
    print(f"Exporting to ONNX: {output_path}")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['policy', 'value'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'policy': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print(f"Successfully exported model to {output_path}")
    
    # Verify the exported model
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed!")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export ChessNet 19x192 to ONNX")
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to the .pt checkpoint file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="chessnet_19x192.onnx",
        help="Output ONNX file path (default: chessnet_19x192.onnx)"
    )
    parser.add_argument(
        "--transformer",
        action="store_true",
        help="Use transformer policy head instead of standard"
    )
    
    args = parser.parse_args()
    
    export_model_19x192(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        use_transformer=args.transformer
    )
