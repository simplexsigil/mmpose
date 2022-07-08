_base_ = ['../../../../_base_/datasets/coco_wholebody.py']
log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=10)
evaluation = dict(interval=10, metric='mAP', save_best='AP')

optimizer = dict(
    type='AdamW',
    lr=5e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
)

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])
total_epochs = 210
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    num_output_channels=133,
    dataset_joints=133,
    dataset_channel=[
        list(range(133)),
    ],
    inference_channel=list(range(133)))

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='TopDown',
    pretrained='https://download.openmmlab.com/mmpose/'
    'pretrain_models/tcformer-4e1adbf1_20220421.pth',
    backbone=dict(
        type='TCFormer',
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        num_layers=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        drop_path_rate=0.1),
    neck=dict(
        type='MTA',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        start_level=0,
        num_heads=[4, 4, 4, 4],
        mlp_ratios=[4, 4, 4, 4],
        num_outs=4,
        use_sr_conv=False,
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=256,
        out_channels=channel_cfg['num_output_channels'],
        num_deconv_layers=0,
        extra=dict(final_conv_kernel=1, ),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))

data_cfg = dict(
    image_size=[288, 384],
    heatmap_size=[72, 96],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    use_nms=True,
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=False,
    det_bbox_thr=0.2,
    bbox_file='data/posetrack18/posetrack18_precomputed_boxes/'
    'val_boxes.json',
    # frame_indices_train=[-1, 0],
    frame_index_rand=True,
    frame_index_range=[-2, 2],
    num_adj_frames=1,
    frame_indices_test=[-2, -1, 0, 1, 2],
    # the first weight is the current frame,
    # then on ascending order of frame indices
    frame_weight_train=(0.0, 1.0),
    frame_weight_test=(0.3, 0.1, 0.25, 0.25, 0.1),
)

# take care of orders of the transforms
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownRandomShiftBboxCenter', shift_factor=0.16, prob=0.3),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=45,
        scale_factor=0.35),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=3),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs', 'frame_weight'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=[
            'img',
        ],
        meta_keys=[
            'image_file',
            'center',
            'scale',
            'rotation',
            'bbox_score',
            'flip_pairs',
            'frame_weight',
        ]),
]

test_pipeline = val_pipeline

data_root = 'data/posetrack18'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=4),
    test_dataloader=dict(samples_per_gpu=4),
    train=dict(
        type='TopDownPoseTrack18VideoDataset',
        ann_file=f'{data_root}/annotations/posetrack18_train.json',
        img_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='TopDownPoseTrack18VideoDataset',
        ann_file=f'{data_root}/annotations/posetrack18_val.json',
        img_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='TopDownPoseTrack18VideoDataset',
        ann_file=f'{data_root}/annotations/posetrack18_val.json',
        img_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
