max_epochs=300

model_type = "n"

model = dict(type='TrainYOLO11Seg',
    backbone_config=dict(
        model_type = "n",
        num_classes = 80,
        freeze_patterns = 'none'),
    loss_config=dict(
        strides = [8, 16, 32],
        num_classes = 80,
        dfl_bins = 16,
        loss_cls_weight = 0.5,
        loss_iou_weight = 7.5,
        loss_dfl_weight = 1.5,
        loss_seg_weight = 12,
        cls_loss_type = "bce",
        label_smoothing = 0.0,
        use_focal = True,
        focal_alpha = 0.25,
        focal_gamma = 2.0,
        debug = False,
        o2m_weight = 0.8,
        pos_cls_weight = 1.0,
        neg_cls_weight = 0.1,
        seg_debug = False))

work_dir = (
    f"/lpai/volumes/vc-profile-bd-ga/others/wubo/Projects/Code/hauaworkspace/yolo11seg{model_type}")

train_dataloader = dict(
    dataset = dict(type = 'YOLOCOCOSeg',
        root = (
            "/lpai/volumes/vc-profile-bd-ga/others/wubo/Datasets/OpenDataLab___COCO_2017/raw/Images"
            "/train2017"),
        ann_file = (
            "/lpai/volumes/vc-profile-bd-ga/others/wubo/Datasets/OpenDataLab___COCO_2017/raw"
            "/Annotations/instances_train2017.json")),
    sampler = dict(type='DefaultSampler', shuffle=True),
    collate_fn = dict(type='coco_seg_collate'),
    batch_size = 32,
    drop_last = True,
    pin_memory = False,
    persistent_workers = True,
    num_workers = 4)

train_cfg = dict(
    by_epoch = True,
    max_epochs = max_epochs,
    val_begin = 1,
    val_interval = 1)

optim_wrapper = dict(
    optimizer = dict(
        type = 'AdamW',
        lr = 1e-3,
        weight_decay = 1e-2))

param_scheduler = [
    dict(  # 预热
        type = 'LinearLR',
        start_factor = .001,
        by_epoch = True,
        begin = 0,
        end = 6,
        verbose = True),
    dict(  # 阶梯下降
        type = 'MultiStepLR',
        by_epoch = True,
        milestones = [60, 120, 180],
        gamma = 0.1)]


# val_dataloader = dict(
#     dataset=dict(type='YOLOCOCOSeg',
#         root = (
        #     "/lpai/volumes/vc-profile-bd-ga/others/wubo/Datasets/OpenDataLab___COCO_2017/raw/Images"
        #     "/val2017"),
        # ann_file = (
        #     "/lpai/volumes/vc-profile-bd-ga/others/wubo/Datasets/OpenDataLab___COCO_2017/raw"
        #     "/Annotations/instances_val2017.json")),
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     collate_fn=dict(type='default_collate'),
#     batch_size=128,
#     drop_last=False,
#     pin_memory=False,
#     persistent_workers=False,
#     num_workers=8)
# val_evaluator = dict(type='GenderAgeMetric')

# val_cfg = dict(type='ValLoop')

default_hooks = dict(
    logger = dict(type = 'LoggerHook',
        interval = 10,
        log_metric_by_epoch = True),
    checkpoint = dict(type='CheckpointHook', interval=1))

launcher = 'pytorch'

env_cfg = dict(
    cudnn_benchmark = False,
    backend = 'nccl',
    mp_cfg = dict(mp_start_method='spawn'),
    dist_cfg = dict(backend='nccl'))

log_level = 'INFO'

load_from = "/mnt/volumes/vc-profile-bd-ga/others/wubo/Projects/Code/hauaworkspace/yolo11segn/epoch_53.pth"

resume = True