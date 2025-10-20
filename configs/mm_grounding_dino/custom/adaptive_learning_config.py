_base_ = '../grounding_dino_swin-t_pretrain_obj365.py'

# Import custom classes
import sys
sys.path.append('/root/mmdetection')
import adaptive_learning

# Data settings
data_root = 'data/insulator-defect-detection/'
train_ann_file = 'fsod_coco_idx0/train_30shot_seed0.json'
val_ann_file_50 = 'fsod_coco_idx0/val_50.json'
val_ann_file_full = 'fsod_coco_idx0/val.json'
train_dir = 'project/train/img'
val_dir = 'project/val/img'

classes = ('broken', 'insulator', 'pollution-flashover')
metainfo = dict(classes=classes, palette=[(220, 20, 60), (0, 0, 142), (119, 11, 32)])

# Override model for our classes
model = dict(bbox_head=dict(num_classes=3))

# Override train dataloader to use adaptive dataset
train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='AdaptiveCocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_dir),
        return_classes=True,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        max_samples=30,
        samples_per_stage=1
    ))

# Override val/test dataloaders
val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file=val_ann_file_50,
        data_prefix=dict(img=val_dir)))

test_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file=val_ann_file_full,
        data_prefix=dict(img=val_dir)))

# Override evaluators
val_evaluator = dict(ann_file=data_root + val_ann_file_50)
test_evaluator = dict(ann_file=data_root + val_ann_file_full)

# Training settings
max_epochs = 300
train_cfg = dict(max_epochs=max_epochs, val_interval=10)

# Add adaptive learning hook
custom_hooks = [
    dict(
        type='AdaptiveLearningHook',
        epochs_per_stage=10,
        save_checkpoint=True
    )
]

# Override work dir
work_dir = 'work_dirs/adaptive_learning_30shot'