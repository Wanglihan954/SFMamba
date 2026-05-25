import math
import argparse
import pprint
from distutils.util import strtobool
from pathlib import Path
from loguru import logger as loguru_logger

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

from src.config.default import get_cfg_defaults
from src.utils.misc import get_rank_zero_only_logger, setup_gpus
from src.utils.profiler import build_profiler
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_jamma import PL_JamMa
loguru_logger = get_rank_zero_only_logger(loguru_logger)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_cfg_path', type=str, help='data config path')
    parser.add_argument('main_cfg_path', type=str, help='main config path')
    parser.add_argument('--exp_name', type=str, default='default_exp_name')
    parser.add_argument('--dump_dir', type=str, default=None, help="if set, the matching results will be dump to dump_dir")
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=lambda x: bool(strtobool(x)), nargs='?', default=True, help='whether loading data to pinned memory or not')
    parser.add_argument('--ckpt_path', type=str, default=None, help='pretrained checkpoint path, helpful for using a pre-trained coarse-only LoFTR')
    parser.add_argument('--disable_ckpt', action='store_true', help='disable checkpoint saving (useful for debugging).')
    parser.add_argument('--profiler_name', type=str, default=None, help='options: [inference, pytorch], or leave it unset')
    parser.add_argument('--parallel_load_data', action='store_true', help='load datasets in with multiple processes.')
    # 添加 Trainer 相关参数
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--num_nodes', type=int, default=1, help='number of nodes to use')
    parser.add_argument('--accelerator', type=str, default='ddp', help='training accelerator')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='validation frequency (epochs)')
    parser.add_argument('--log_every_n_steps', type=int, default=1000, help='logging frequency (steps)')
    parser.add_argument('--flush_logs_every_n_steps', type=int, default=1000, help='flush logs frequency (steps)')
    parser.add_argument('--limit_val_batches', type=float, default=1.0, help='limit validation batches')
    parser.add_argument('--num_sanity_val_steps', type=int, default=10, help='number of sanity validation steps')
    parser.add_argument('--benchmark', type=bool, default=True, help='cudnn benchmark')
    parser.add_argument('--max_epochs', type=int, default=30, help='max number of epochs')
    return parser.parse_args()


def main():
    # parse arguments
    args = parse_args()
    rank_zero_only(pprint.pprint)(vars(args))

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility
    # scale lr and warmup-step automatically
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.SCALING = _scaling
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    config.TRAINER.WARMUP_STEP = math.floor(config.TRAINER.WARMUP_STEP / _scaling)
    
    # lightning module
    profiler = build_profiler(args.profiler_name)
    model = PL_JamMa(config, pretrained_ckpt=args.ckpt_path, profiler=profiler, dump_dir=args.dump_dir)
    loguru_logger.info(f"LoFTR LightningModule initialized!")
    
    # lightning data
    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info(f"LoFTR DataModule initialized!")
    
    # TensorBoard Logger
    logger = TensorBoardLogger(save_dir='jamma_log/', name=args.exp_name, default_hp_metric=False)
    ckpt_dir = Path(logger.log_dir) / 'checkpoints'

    # Callbacks
    # TODO: update ModelCheckpoint to monitor multiple metrics
    ckpt_callback = ModelCheckpoint(monitor='auc@10', verbose=True, save_top_k=3, mode='max',
                                    save_last=True,
                                    dirpath=str(ckpt_dir),
                                    filename='{epoch}-{auc@5:.3f}-{auc@10:.3f}-{auc@20:.3f}')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor]
    if not args.disable_ckpt:
        callbacks.append(ckpt_callback)
    
    # Lightning Trainer
    # 设置 cudnn benchmark
    try:
        import torch as _torch
        _torch.backends.cudnn.benchmark = bool(args.benchmark)
    except Exception:
        pass

    # map legacy accelerator names to valid ones for PL 2.x
    accel = args.accelerator
    _valid_accels = ('auto', 'tpu', 'mps', 'cuda', 'cpu')
    if accel not in _valid_accels:
        accel = 'cuda' if getattr(args, 'gpus', 0) and args.gpus > 0 else 'cpu'

    strategy = DDPStrategy(find_unused_parameters=False)

    trainer = pl.Trainer(
        strategy=strategy,
        devices=args.gpus,
        num_nodes=args.num_nodes,
        accelerator=accel,
        max_epochs=args.max_epochs,
        gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
        callbacks=callbacks,
        logger=logger,
        profiler=profiler,
    )

    loguru_logger.info(f"Trainer initialized!")
    loguru_logger.info(f"Start training!")
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    main()
