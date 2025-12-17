_base_ = [
    'configs/_base_/schedules/cifar10_bs128.py', # Inherit default scheduler
    'configs/_base_/default_runtime.py',
]

# ==========================================================
# 1. PATHS & CHECKPOINT
# ==========================================================
data_root = '/media/hiro/T7/UM_datasets/data/cifar-10-batches-py'
# PATH TO YOUR SAVED EPOCH 2 CHECKPOINT
checkpoint_path = 'work_dirs/mocov2_bcosresnet50_4xb64-coslr-200e_in1k/epoch_2.pth'

# ==========================================================
# 2. MODEL SETTINGS (Linear Probe)
# ==========================================================
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',     # Must match the architecture you pretrained
        depth=50,
        num_stages=4,
        out_indices=(3, ), # Output features from the last layer
        style='pytorch',
        # THIS IS THE KEY FOR PROBING:
        frozen_stages=4,   # Freeze everything! 4 = all stages frozen.
        init_cfg=dict(
            type='Pretrained', 
            checkpoint=checkpoint_path, 
            prefix='backbone.' # MoCo saves weights with 'backbone.' prefix
        )
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,    # CIFAR-10 has 10 classes
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, ),
    )
)

# ==========================================================
# 3. DATASET & PIPELINE
# ==========================================================
dataset_type = 'CIFAR10'

# Normalize using CIFAR stats (same as your pretraining)
data_preprocessor = dict(
    num_classes=10,
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False
)

train_pipeline = [
    dict(type='RandomResizedCrop', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='Resize', scale=256, backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=128, # Frozen backbone uses very little memory, so 128 might fit!
    num_workers=4,
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
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, ))
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# ==========================================================
# 4. OPTIMIZER & SCHEDULE
# ==========================================================
# Linear probing typically uses a slightly higher LR and no weight decay for the head
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0)
)

# Train for 100 epochs (Linear probing converges fast)
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# Simple scheduler
param_scheduler = [
    dict(type='MultiStepLR', by_epoch=True, milestones=[60, 80], gamma=0.1)
]