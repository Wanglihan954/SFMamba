import argparse
import pprint

import pytorch_lightning as pl
from loguru import logger as loguru_logger

from src.config.default import get_cfg_defaults
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_jamma import PL_JamMa
from src.utils.profiler import build_profiler


def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'data_cfg_path', type=str, help='data config path')
    parser.add_argument(
        'main_cfg_path', type=str, help='main config path')
    parser.add_argument(
        '--ckpt_path', type=str, default="weights/indoor_ds.ckpt", help='path to the checkpoint')
    parser.add_argument(
        '--dump_dir', type=str, default=None, help="if set, the matching results will be dump to dump_dir")
    parser.add_argument(
        '--profiler_name', type=str, default=None, help='options: [inference, pytorch], or leave it unset')
    parser.add_argument(
        '--batch_size', type=int, default=1, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=2)
    parser.add_argument(
        '--thr', type=float, default=None, help='modify the coarse-level matching threshold.')

    # Older PyTorch Lightning exposes CLI helpers on Trainer.
    # Newer Lightning versions have removed these; fall back to manually
    # defining the most relevant Trainer arguments.
    if hasattr(pl.Trainer, 'add_argparse_args'):
        parser = pl.Trainer.add_argparse_args(parser)
    else:
        parser.add_argument('--gpus', type=int, default=-1, help='number of gpus, -1 for all available')
        parser.add_argument('--num_nodes', type=int, default=1, help='number of nodes for distributed run')
        parser.add_argument('--accelerator', type=str, default='ddp', help='training strategy (e.g. ddp)')
        parser.add_argument('--benchmark', action='store_true', help='enable cudnn benchmark')

    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    pprint.pprint(vars(args))

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility

    # tune when testing
    if args.thr is not None:
        config.LOFTR.MATCH_COARSE.THR = args.thr

    loguru_logger.info(f"Args and config initialized!")

    # lightning module
    profiler = build_profiler(args.profiler_name)
    model = PL_JamMa(config, pretrained_ckpt=args.ckpt_path, profiler=profiler, dump_dir=args.dump_dir)
    loguru_logger.info(f"JamMa-lightning initialized!")

    # lightning data
    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info(f"DataModule initialized!")

    # lightning trainer
    if hasattr(pl.Trainer, 'from_argparse_args'):
        trainer = pl.Trainer.from_argparse_args(args, replace_sampler_ddp=False, logger=False)
    else:
        # Fallback for newer Lightning where CLI helpers are removed
        gpus = getattr(args, 'gpus', -1)
        num_nodes = getattr(args, 'num_nodes', 1)
        strategy = getattr(args, 'accelerator', 'ddp')
        benchmark = getattr(args, 'benchmark', False)

        # Map legacy gpus flag to new devices argument
        devices = 'auto' if gpus is None or gpus == -1 else gpus

        trainer = pl.Trainer(
            accelerator='gpu',
            devices=devices,
            num_nodes=num_nodes,
            strategy=strategy,
            benchmark=benchmark,
            logger=False,
        )

    loguru_logger.info(f"Start testing!")
    trainer.test(model, datamodule=data_module, verbose=False)
