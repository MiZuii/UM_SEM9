# BYOL on CIFAR10 with ResNet50_CIFAR backbone
# 2 GPUs x 128 batch size, 50 epochs

_base_ = [
    '../_base_/datasets/cifar10_bs128_byol.py',
    '../_base_/schedules/cifar10_sgd_coslr_50e.py',
    '../_base_/default_runtime.py',
]

# Model settings - using ResNet_CIFAR for 32x32 images
model = dict(
    type='BYOL',
    base_momentum=0.01,
    backbone=dict(
        type='ResNet_CIFAR',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(
        type='NonLinearNeck',
        in_channels=2048,
        hid_channels=4096,
        out_channels=256,
        num_layers=2,
        with_bias=True,
        with_last_bn=False,
        with_avg_pool=True),
    head=dict(
        type='LatentPredictHead',
        predictor=dict(
            type='NonLinearNeck',
            in_channels=256,
            hid_channels=4096,
            out_channels=256,
            num_layers=2,
            with_bias=True,
            with_last_bn=False,
            with_avg_pool=False),
        loss=dict(type='CosineSimilarityLoss')),
)

# Optimizer - LARS scaled for CIFAR10
optimizer = dict(type='LARS', lr=1.0, momentum=0.9, weight_decay=1e-6)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'bn': dict(decay_mult=0, lars_exclude=True),
            'bias': dict(decay_mult=0, lars_exclude=True),
            'downsample.1': dict(decay_mult=0, lars_exclude=True),
        }),
)

# Runtime settings
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)
