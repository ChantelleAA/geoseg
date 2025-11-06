#!/usr/bin/env python3
"""
Export FTUNetFormer segmenter to TorchScript for fast filtering in GAN generation.

Usage:
  python export_segmenter_ts.py \
    --ckpt /home/chantelle/Desktop/UCD/ai_sandbox/geoseg/model_weights/biodiversity/ftunetformer-kd-512-crop-ms-e45/ftunetformer-kd-512-crop-ms-e45-v4.ckpt \
    --out segmenter_ts.pt \
    --img_size 512 \
    --device cpu
"""

import argparse, torch, re
import torch.nn as nn

# --- import your model constructor ---
from geoseg.models.FTUNetFormer import ft_unetformer

def strip_prefix(state_dict, prefixes=("net.", "model.", "module.")):
    """Remove common Lightning/DataParallel prefixes from state_dict keys."""
    out = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out

class NormalizeWrapper(nn.Module):
    """
    Wrap the core model so the exported TorchScript model accepts 0..1 RGB
    and applies mean/std normalization internally.
    """
    def __init__(self, core, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
        super().__init__()
        self.core = core
        self.register_buffer("mean", torch.tensor(mean).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor(std).view(1,3,1,1))

    def forward(self, x):
        # x: [B,3,H,W] in 0..1
        x = (x - self.mean) / self.std
        logits = self.core(x)
        # Ensure we return [B,C,H,W] logits (float32)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        return logits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to Lightning .ckpt or state_dict .pth")
    ap.add_argument("--out",  default="segmenter_ts.pt", help="TorchScript output file")
    ap.add_argument("--num_classes", type=int, default=6)
    ap.add_argument("--decoder_channels", type=int, default=256)
    ap.add_argument("--img_size", type=int, default=512, help="dummy trace size (use 512 to match segmenter input)")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], 
                    help="Device to export on (use 'cpu' for CPU-only inference)")
    args = ap.parse_args()

    # CRITICAL: Export on the device you'll use for inference
    # Since your GAN gen script loads with map_location="cpu", export on CPU
    device = args.device
    print(f"[INFO] Exporting TorchScript model on device: {device}")

    # 1) Build the exact model you trained
    core = ft_unetformer(num_classes=args.num_classes, decoder_channels=args.decoder_channels)
    
    # 2) Load checkpoint FIRST on CPU, then move to target device
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt  # assume plain state dict

    state = strip_prefix(state, prefixes=("net.", "model.", "module."))
    
    # Load state dict while model is still on CPU
    missing, unexpected = core.load_state_dict(state, strict=False)
    print(f"[load_state_dict] missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:   print("  missing:", missing[:10], "..." if len(missing)>10 else "")
    if unexpected:print("  unexpected:", unexpected[:10], "..." if len(unexpected)>10 else "")

    # NOW move to target device and set to eval
    core = core.to(device).eval()

    # 3) Wrap with normalization so TorchScript takes 0..1 tensors
    model = NormalizeWrapper(core).to(device).eval()

    # 4) Trace & save TorchScript
    dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    print(f"[INFO] Tracing with dummy input shape: {dummy.shape}")
    
    with torch.inference_mode():
        # Test forward pass before tracing
        try:
            output = model(dummy)
            print(f"[INFO] Test forward pass successful, output shape: {output.shape}")
        except Exception as e:
            print(f"[ERROR] Forward pass failed: {e}")
            return
        
        # Trace the model
        ts = torch.jit.trace(model, dummy, check_trace=False)
    
    ts.save(args.out)
    print(f"[OK] Saved TorchScript to: {args.out}")
    print(f"[INFO] Model exported on '{device}' - load with map_location='{device}' for inference")

if __name__ == "__main__":
    main()