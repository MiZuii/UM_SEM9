# CIFAR100 dataset config for MoCo v2 SSL training
# Adapted for 32x32 images with SSL augmentations

dataset_type = 'CIFAR100'
data_root = 'data/cifar100'

data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[129.304, 124.070, 112.434],  # CIFAR100 mean
    std=[68.170, 65.392, 70.418],       # CIFAR100 std
    to_rgb=False)

# MoCo v2 augmentation pipeline adapted for CIFAR100 (32x32 images)
view_pipeline = [
    dict(
        type='RandomResizedCrop',
        scale=32,
        crop_ratio_range=(0.2, 1.0),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1)
        ],
        prob=0.8),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(
        type='GaussianBlur',
        magnitude_range=(0.1, 2.0),
        magnitude_std='inf',
        prob=0.5),
]

train_pipeline = [
    # Note: No LoadImageFromFile needed - CIFAR100 provides images directly in memory
    dict(
        type='MultiView',
        num_views=[1, 1],
        transforms=[view_pipeline, view_pipeline]),
    dict(type='PackInputs')
]

train_dataloader = dict(
    batch_size=128,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=True,  # Required for MoCo: queue_len must be divisible by batch_size
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split='train',
        pipeline=train_pipeline))
