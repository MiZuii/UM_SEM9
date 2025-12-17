auto_scale_lr = dict(base_batch_size=256)
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True,
    type='SelfSupDataPreprocessor')
data_root = '/media/hiro/T7/UM_datasets/data/cifar-10-batches-py'
dataset_type = 'CIFAR10'
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=3, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
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
        norm_cfg=dict(type='BN'),
        type='ResNet',
        zero_init_residual=False),
    feat_dim=128,
    head=dict(
        loss=dict(type='CrossEntropyLoss'),
        temperature=0.2,
        type='ContrastiveHead'),
    momentum=0.001,
    neck=dict(
        hid_channels=2048,
        in_channels=2048,
        out_channels=128,
        type='MoCoV2Neck',
        with_avg_pool=True),
    queue_len=65536,
    type='MoCo')
optim_wrapper = dict(
    loss_scale='dynamic',
    optimizer=dict(lr=0.03, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(T_max=2, begin=0, by_epoch=True, end=2, type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, seed=None)
resume = False
test_pipeline = [
    dict(backend='pillow', scale=256, type='Resize'),
    dict(crop_size=224, type='CenterCrop'),
    dict(type='PackInputs'),
]
train_cfg = dict(max_epochs=2, type='EpochBasedTrainLoop')
train_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='/media/hiro/T7/UM_datasets/data/cifar-10-batches-py',
        pipeline=[
            dict(
                num_views=[
                    2,
                ],
                transforms=[
                    [
                        dict(
                            backend='pillow',
                            scale=224,
                            type='RandomResizedCrop'),
                        dict(
                            prob=0.8,
                            transforms=[
                                dict(
                                    brightness=0.4,
                                    contrast=0.4,
                                    hue=0.1,
                                    saturation=0.4,
                                    type='ColorJitter'),
                            ],
                            type='RandomApply'),
                        dict(
                            keep_channels=True,
                            prob=0.2,
                            type='RandomGrayscale'),
                        dict(
                            magnitude_range=(
                                0.1,
                                2.0,
                            ),
                            magnitude_std='inf',
                            prob=0.5,
                            type='GaussianBlur'),
                        dict(prob=0.5, type='RandomFlip'),
                    ],
                ],
                type='MultiView'),
            dict(type='PackInputs'),
        ],
        split='train',
        type='CIFAR10'),
    drop_last=True,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        num_views=[
            2,
        ],
        transforms=[
            [
                dict(backend='pillow', scale=224, type='RandomResizedCrop'),
                dict(
                    prob=0.8,
                    transforms=[
                        dict(
                            brightness=0.4,
                            contrast=0.4,
                            hue=0.1,
                            saturation=0.4,
                            type='ColorJitter'),
                    ],
                    type='RandomApply'),
                dict(keep_channels=True, prob=0.2, type='RandomGrayscale'),
                dict(
                    magnitude_range=(
                        0.1,
                        2.0,
                    ),
                    magnitude_std='inf',
                    prob=0.5,
                    type='GaussianBlur'),
                dict(prob=0.5, type='RandomFlip'),
            ],
        ],
        type='MultiView'),
    dict(type='PackInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=32,
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
    pin_memory=True)
val_evaluator = dict(topk=(1, ), type='Accuracy')
view_pipeline = [
    dict(backend='pillow', scale=224, type='RandomResizedCrop'),
    dict(
        prob=0.8,
        transforms=[
            dict(
                brightness=0.4,
                contrast=0.4,
                hue=0.1,
                saturation=0.4,
                type='ColorJitter'),
        ],
        type='RandomApply'),
    dict(keep_channels=True, prob=0.2, type='RandomGrayscale'),
    dict(
        magnitude_range=(
            0.1,
            2.0,
        ),
        magnitude_std='inf',
        prob=0.5,
        type='GaussianBlur'),
    dict(prob=0.5, type='RandomFlip'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/mocov2_bcosresnet50_4xb64-coslr-200e_in1k'
