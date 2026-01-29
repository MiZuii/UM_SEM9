#!/usr/bin/env python3
"""
Convert DINO checkpoint to mmpretrain-compatible format.
DINO saves: ckpt['teacher']['module.backbone.xxx'] 
We need: ckpt['state_dict']['backbone.xxx']
"""
import argparse
import torch


def convert_dino_checkpoint(input_path: str, output_path: str):
    """Convert DINO checkpoint to mmpretrain format."""
    print(f"Loading checkpoint from {input_path}")
    ckpt = torch.load(input_path, map_location='cpu')
    
    # DINO stores weights under 'teacher' or 'student' key
    if 'teacher' in ckpt:
        teacher_state = ckpt['teacher']
    elif 'student' in ckpt:
        teacher_state = ckpt['student']
    else:
        raise ValueError(f"Cannot find 'teacher' or 'student' in checkpoint. Keys: {list(ckpt.keys())}")
    
    # Convert keys from 'module.backbone.xxx' to 'backbone.xxx'
    new_state_dict = {}
    for key, value in teacher_state.items():
        if key.startswith('module.backbone.'):
            # Remove 'module.' prefix
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = value
            print(f"  {key} -> {new_key}")
        elif key.startswith('module.head.'):
            # Skip DINO head weights (projection head)
            print(f"  Skipping: {key}")
        else:
            print(f"  Skipping unknown key: {key}")
    
    # Create mmpretrain-compatible checkpoint
    converted_ckpt = {
        'state_dict': new_state_dict,
        'meta': {
            'epoch': ckpt.get('epoch', 0),
            'converted_from': 'dino',
        }
    }
    
    print(f"\nSaving converted checkpoint to {output_path}")
    print(f"Total backbone parameters: {len(new_state_dict)}")
    torch.save(converted_ckpt, output_path)
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert DINO checkpoint to mmpretrain format')
    parser.add_argument('input', help='Input DINO checkpoint path')
    parser.add_argument('output', help='Output mmpretrain checkpoint path')
    args = parser.parse_args()
    
    convert_dino_checkpoint(args.input, args.output)
