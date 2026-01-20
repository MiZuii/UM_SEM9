# CIFAR10 schedule: 50 epochs with SGD and cosine LR
# Downsized from 200 epochs for faster training

# optimizer wrapper
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.03, weight_decay=1e-4, momentum=0.9))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR', T_max=45, by_epoch=True, begin=5, end=50)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50)
