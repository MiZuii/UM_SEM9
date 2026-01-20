# MoCo v2 on CIFAR10 with ResNet50_CIFAR backbone
# 2 GPUs x 128 batch size, 50 epochs

_base_ = [
    '../_base_/datasets/cifar10_bs128_mocov2.py',
    '../_base_/schedules/cifar10_sgd_coslr_50e.py',
    '../_base_/default_runtime.py',
]

# Model settings - using ResNet_CIFAR for 32x32 images
model = dict(
    type='MoCo',
    queue_len=4096,  # Smaller queue for CIFAR10 (less data)
    feat_dim=128,
    momentum=0.001,
    backbone=dict(
        type='ResNet_CIFAR',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(
        type='MoCoV2Neck',
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(
        type='ContrastiveHead',
        loss=dict(type='CrossEntropyLoss'),
        temperature=0.2))

# Optimizer - scaled for CIFAR10
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.03, weight_decay=1e-4, momentum=0.9))

# Only keeps the latest 3 checkpoints
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)
