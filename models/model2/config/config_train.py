# Train Config
config = {
    "dtype": "fp32",  # todo: does not have any impact for now
    "device": "gpu",
    "device_id": 0,  # todo: does not have any impact for now

    # learning policy
    # tf.keras.experimental.LinearCosineDecay
    "init_lr": 0.004,
    "decay_steps": 9000,
    "num_periods": 3,
    "alpha": 0.05,
    "beta": 0.01,
    "weight_decay": 0.00001,
    "max_step": 9000,

    # logging
    "logging_step": 1,
    "saving_step": 400,
    "val_step": 100,
    "summary_step": 100,

    # input
    "train_tfrecord_dir": '/content/kidney_datasets/x16_tile_trainval1/train',  # tfrecord_folder
    "train_batch_size": 32,
    "val_tfrecord_dir": '/content/kidney_datasets/x16_tile_trainval1/val',  # tfrecord_folder
    "val_batch_size": 32,

    # input - augmentation
    "random_scale_range": [1.0, 1.2],  # scale before cropping. None for skipping
    "crop_size": [257, 257],
    "flip_probability": 0.5,
    "rotate_probability": 0.5,
    "rotate_angle_by90": True,
    "rotate_angle_range": None,  # works only if "rotate_angle_by90: False"
    "random_quality_prob": 0.0,  # in tf-v2 it does not work. Do not use it
    "random_quality": [80, 100],
    "rgb_permutation_prob": 0.5,
    "brightness_prob": 0.5,
    "brightness_constant": 0.1,
    "contrast_prob": 0.5,
    "contrast_constant": [1.0, 2.0],
    "hue_prob": 0.5,
    "hue_constant": [-0.3, 0.3],
    "saturation_prob": 0.5,
    "saturation_constant": [0.4, 3.0],
    "gaussian_noise_prob": 0.5,
    "gaussian_noise_std": [0.03, 0.1],
    "shred_prob": 0.0,
    "shred_piece_range": None,  # [max, min] number of shredded pieces
    "shred_shift_ratio": None,
    "shade_prob": 0.0,
    "shade_file": None,
    "warp_prob": 0.0,
    "warp_ratio": 0.0,
    "warp_crop_prob": 0.0,
    "elastic_distortion_prob": 0.0,
}
