auto_scale_lr = dict(base_batch_size=128)
checkpoint_path = 'work_dirs/mocov2_bcosresnet50_4xb64-coslr-200e_in1k/epoch_2.pth'
data_preprocessor = dict(
    mean=[
        125.307,
        122.961,
        113.8575,
    ],
    num_classes=10,
    std=[
        51.5865,
        50.847,
        51.255,
    ],
    to_rgb=False)
data_root = '/media/hiro/T7/UM_datasets/data/cifar-10-batches-py'
dataset_type = 'CIFAR10'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=4,
        init_cfg=dict(
            checkpoint=
            'work_dirs/mocov2_bcosresnet50_4xb64-coslr-200e_in1k/epoch_2.pth',
            prefix='backbone.',
            type='Pretrained'),
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        type='ResNet'),
    head=dict(
        in_channels=2048,
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss'),
        num_classes=10,
        topk=(1, ),
        type='LinearClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    type='ImageClassifier')
optim_wrapper = dict(
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0))
param_scheduler = [
    dict(by_epoch=True, gamma=0.1, milestones=[
        60,
        80,
    ], type='MultiStepLR'),
]
randomness = dict(deterministic=False, seed=None)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=128,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='/media/hiro/T7/UM_datasets/data/cifar-10-batches-py',
        pipeline=[
            dict(backend='pillow', scale=256, type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        test_mode=True,
        type='CIFAR10'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(topk=(1, ), type='Accuracy')
test_pipeline = [
    dict(backend='pillow', scale=256, type='Resize'),
    dict(crop_size=224, type='CenterCrop'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
train_dataloader = dict(
    batch_size=128,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='/media/hiro/T7/UM_datasets/data/cifar-10-batches-py',
        pipeline=[
            dict(backend='pillow', scale=224, type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        split='train',
        type='CIFAR10'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend='pillow', scale=224, type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=128,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='/media/hiro/T7/UM_datasets/data/cifar-10-batches-py',
        pipeline=[
            dict(backend='pillow', scale=256, type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        test_mode=True,
        type='CIFAR10'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(topk=(1, ), type='Accuracy')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/linear_probe_cifar10'
