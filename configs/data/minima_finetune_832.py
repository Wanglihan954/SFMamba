from configs.data.base import cfg


TRAIN_BASE_PATH = "data/megadepth/index"
cfg.DATASET.TRAINVAL_DATA_SOURCE = "MegaDepth"

# MiNIMA pseudo-infrared training images + MegaDepth indices/depth
cfg.DATASET.TRAIN_DATA_ROOT = "data/megadepth/train"
cfg.DATASET.TRAIN_IMAGE_ROOT = "data/MINIMA/train/Infrared"
cfg.DATASET.TRAIN_DEPTH_ROOT = "data/megadepth/train"
cfg.DATASET.TRAIN_IMAGE_PATH_PREFIX_TO_STRIP = "phoenix/S6/zl548/"
cfg.DATASET.TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7_no_sfm"
cfg.DATASET.TRAIN_LIST_PATH = "assets/minima_train_scenes.txt"
cfg.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.0

# Validation/testing: use MiNIMA pseudo-infrared images with MegaDepth depth/index
TEST_BASE_PATH = "data/megadepth/index"
cfg.DATASET.TEST_DATA_SOURCE = "MegaDepth"
cfg.DATASET.VAL_DATA_ROOT = cfg.DATASET.TEST_DATA_ROOT = "data/MINIMA/test/Megadepth-1500-syn/Infrared"
cfg.DATASET.VAL_IMAGE_ROOT = cfg.DATASET.TEST_IMAGE_ROOT = "data/MINIMA/test/Megadepth-1500-syn/Infrared"
cfg.DATASET.VAL_DEPTH_ROOT = cfg.DATASET.TEST_DEPTH_ROOT = "data/megadepth/test"
cfg.DATASET.VAL_NPZ_ROOT = cfg.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}/scene_info_val_1500"
cfg.DATASET.VAL_LIST_PATH = cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/trainvaltest_list/val_list.txt"
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0

cfg.DATASET.MGDPT_IMG_PAD = True
cfg.DATASET.MGDPT_DEPTH_PAD = True
cfg.TRAINER.N_SAMPLES_PER_SUBSET = 100

cfg.DATASET.MGDPT_IMG_RESIZE = 832
cfg.DATASET.CORR_TH = 5
cfg.TRAINER.EPI_ERR_THR = 1e-4
