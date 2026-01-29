# Linear probing for MoCo v2 pretrained ResNet50 on CIFAR100
# Uses frozen backbone + single linear classifier head

_base_ = [
    '../../pretraining/mmpretrain/configs/_base_/default_runtime.py',
]

# Dataset settings for CIFAR100
dataset_type = 'CIFAR100'
data_root = 'data/cifar100'

data_preprocessor = dict(
    num_classes=100,
    mean=[129.304, 124.070, 112.434],  # CIFAR100 mean
    std=[68.170, 65.392, 70.418],       # CIFAR100 std
    to_rgb=False,
)

train_pipeline = [
    # No LoadImageFromFile - CIFAR100 provides images directly
    dict(type='RandomCrop', crop_size=32, padding=4),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    # No LoadImageFromFile - CIFAR100 provides images directly
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=128,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=128,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split='test',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

test_dataloader = val_dataloader
test_evaluator = val_evaluator

# Model: Frozen MoCo v2 backbone + Linear classifier head
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=4,  # Freeze all stages
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='../../outputs/mocov2_cifar100/epoch_50.pth'
        )
    ),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=2048,
        with_avg_pool=True,
        loss=dict(type='CrossEntropyLoss'),
        topk=(1, 5),
    )
)

# Optimizer - only train the linear head
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0))

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR', T_max=95, by_epoch=True, begin=5, end=100)
]

# Training settings
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# Checkpointing
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3))

auto_scale_lr = dict(base_batch_size=256)
