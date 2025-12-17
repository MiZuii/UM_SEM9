_base_ = './mocov2_resnet50_8xb32-coslr-200e_in1k.py'

# ==========================================================
# 1. DATA PATH SETTINGS
# ==========================================================
data_root = '/media/hiro/T7/UM_datasets/data/cifar-10-batches-py'
dataset_type = 'CIFAR10'

# ==========================================================
# 2. PIPELINE SETTINGS
# ==========================================================
# The "View" pipeline is for generating the 2 crops for MoCo
view_pipeline = [
    dict(type='RandomResizedCrop', scale=224, backend='pillow'),
    dict(type='RandomApply', transforms=[
        dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    ], prob=0.8),
    dict(type='RandomGrayscale', prob=0.2, keep_channels=True),
    dict(type='GaussianBlur', magnitude_range=(0.1, 2.0), magnitude_std='inf', prob=0.5),
    dict(type='RandomFlip', prob=0.5),
]

# TRAIN PIPELINE: Uses MultiView (Output: [Img1, Img2])
train_pipeline = [
    dict(type='MultiView', num_views=[2], transforms=[view_pipeline]),
    dict(type='PackInputs')
]

# TEST PIPELINE: Standard classification transform (Output: Img)
test_pipeline = [
    dict(type='Resize', scale=256, backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs')
]

# ==========================================================
# 3. DATALOADER SETTINGS
# ==========================================================
train_dataloader = dict(
    batch_size=32,
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    drop_last=True,
    collate_fn=dict(type='default_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CIFAR10',
        data_root=data_root,
        split='train',
        pipeline=train_pipeline 
    )
)

val_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        pipeline=test_pipeline, # Uses single image pipeline
    )
)

# This was the missing line causing the crash!
val_cfg = dict(type='ValLoop')
val_evaluator = dict(type='Accuracy', topk=(1, ))

# ==========================================================
# 4. RUNTIME SETTINGS
# ==========================================================
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=3)

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=2,       
        by_epoch=True,
        begin=0,
        end=2          
    )
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1),
    logger=dict(type='LoggerHook', interval=50)
)