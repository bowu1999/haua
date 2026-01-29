max_epochs = 300


model = dict(type = 'TrainRTDETRv2InSeg',
    model_config = dict(
        num_classes = 80,
        feat_strides = [8, 16, 32],
        hidden_dim = 256,
        num_encoder_layers = 1,
        num_queries = 300,
        num_denoising = 100,
        prototype_dim = 32),
    loss_config = dict(
        weight_dict = dict(
            loss_vfl = 1, 
            loss_bbox = 5, 
            loss_giou = 2,
            loss_mask = 1,
            loss_dice = 1),
        losses = ['vfl', 'boxes', 'masks'],
        alpha = 0.75,
        gamma = 2.0,
        boxes_weight_format = 'giou',
        share_matched_indices = True))

work_dir = (
    "/lpai/volumes/vc-profile-bd-ga/others/wubo/Projects/Code/hauaworkspace"
    "/rtdetrv2inseg_b_darnet")

train_dataloader = dict(
    dataset = dict(type = 'RTDETRCOCO',
        root = (
            "/lpai/volumes/vc-profile-bd-ga/others/wubo/Datasets/OpenDataLab___COCO_2017/raw/Images"
            "/train2017"),
        ann_file = (
            "/lpai/volumes/vc-profile-bd-ga/others/wubo/Datasets/OpenDataLab___COCO_2017/raw"
            "/Annotations/instances_train2017.json"),
        return_masks = True),
    sampler = dict(type = 'DefaultSampler',
        shuffle = True),
    collate_fn = dict(type = 'rtdetr_collate_fn'),
    batch_size = 16,
    drop_last = True,
    pin_memory = True,
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
#     dataset=dict(type='YOLOCOCO',
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
    checkpoint = dict(type = 'CheckpointHook',
        interval = 1))

launcher = 'pytorch'

env_cfg = dict(
    cudnn_benchmark = False,
    backend = 'nccl',
    mp_cfg = dict(mp_start_method='spawn'),
    dist_cfg = dict(backend='nccl'))

log_level = 'INFO'

load_from = None

resume = True