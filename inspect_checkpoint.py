"""Inspect the structure of the pretrained checkpoint."""
import torch
from collections import OrderedDict

checkpoint_path = 'pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth'

print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Check if it's a state_dict or full checkpoint
if isinstance(checkpoint, dict):
    if 'state_dict' in checkpoint:
        print("\nCheckpoint contains 'state_dict' key")
        state_dict = checkpoint['state_dict']
        print(f"Other keys in checkpoint: {list(checkpoint.keys())}")
    else:
        print("\nCheckpoint is a state_dict")
        state_dict = checkpoint
else:
    print("\nUnexpected checkpoint format")
    exit(1)

# Analyze the architecture
print(f"\nTotal parameters: {len(state_dict)}")

# Group by module
encoder_keys = [k for k in state_dict.keys() if k.startswith('encoder')]
decoder_keys = [k for k in state_dict.keys() if k.startswith('decoder')]
other_keys = [k for k in state_dict.keys() if not k.startswith('encoder') and not k.startswith('decoder')]

print(f"\nEncoder parameters: {len(encoder_keys)}")
print(f"Decoder parameters: {len(decoder_keys)}")
print(f"Other parameters: {len(other_keys)}")

print("\n=== First 20 Encoder Keys ===")
for key in encoder_keys[:20]:
    print(f"  {key}: {state_dict[key].shape}")

print("\n=== First 20 Decoder Keys ===")
for key in decoder_keys[:20]:
    print(f"  {key}: {state_dict[key].shape}")

print("\n=== Other Keys ===")
for key in other_keys:
    print(f"  {key}: {state_dict[key].shape}")

# Check the number of classes
if 'segmentation_head.0.weight' in state_dict:
    num_classes = state_dict['segmentation_head.0.weight'].shape[0]
    print(f"\n=== Number of output classes: {num_classes} ===")
elif 'final_conv.weight' in state_dict:
    num_classes = state_dict['final_conv.weight'].shape[0]
    print(f"\n=== Number of output classes: {num_classes} ===")

# Find unique module prefixes
all_modules = set()
for key in state_dict.keys():
    parts = key.split('.')
    if len(parts) >= 2:
        all_modules.add('.'.join(parts[:2]))

print(f"\n=== Unique module prefixes (first 2 levels) ===")
for module in sorted(all_modules):
    count = sum(1 for k in state_dict.keys() if k.startswith(module))
    print(f"  {module}: {count} params")
