from collections import defaultdict
import pprint
from loguru import logger
from pathlib import Path

import torch
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from src.jamma.jamma import JamMa
from src.jamma.backbone import CovNextV2_nano
from src.jamma.utils.supervision import compute_supervision_fine, compute_supervision_coarse
from src.losses.loss import Loss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.metrics import (
    compute_f1,
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    aggregate_metrics_train_val, aggregate_metrics_test
)
from src.utils.comm import gather, all_gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler
from thop import profile
from src.utils.plotting import make_matching_figures


class PL_JamMa(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        super().__init__()
        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)
        self.JAMMA_cfg = lower_config(_config['jamma'])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)
        self.viz_path = Path('visualization')
        self.viz_path.mkdir(parents=True, exist_ok=True)
        # Matcher: JamMa
        self.backbone = CovNextV2_nano()
        self.matcher = JamMa(config=_config['jamma'], profiler=profiler)
        self.loss = Loss(_config)

        if pretrained_ckpt == 'official':
            state_dict = torch.hub.load_state_dict_from_url(
                'https://github.com/leoluxxx/JamMa/releases/download/v0.1/jamma.ckpt',
                file_name='jamma.ckpt')['state_dict']
            self.load_state_dict(state_dict, strict=True)
            logger.info(f"Load Official JamMa Weight")
        elif pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
            self.load_state_dict(state_dict, strict=True)
            logger.info(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")

        # Testing
        self.dump_dir = dump_dir
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.total_ms = 0
        self._test_outputs = []
        self._train_epoch_outputs = []
        self._val_epoch_outputs = []
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('number of params:', n_parameters / 1e6)

    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]

    def optimizer_step(self, *args, **kwargs):
        """Flexible optimizer_step to support different Lightning versions.

        This function tries to locate the optimizer and the optimizer_closure
        from `args` or `kwargs`, applies lr warmup, executes the closure
        (if present), calls `optimizer.step()` with fallbacks, and zeros grads.
        """
        # locate optimizer and closure
        optimizer = kwargs.get('optimizer', None)
        optimizer_closure = kwargs.get('optimizer_closure', None) or kwargs.get('closure', None)

        for a in args:
            if optimizer is None and hasattr(a, 'param_groups'):
                optimizer = a
            # pick the first callable that's not a module/batch/tensor as closure
            if optimizer_closure is None and callable(a):
                optimizer_closure = a

        # fallback: if optimizer still None, try conventional position
        if optimizer is None and len(args) >= 3 and hasattr(args[2], 'param_groups'):
            optimizer = args[2]

        # learning rate warm up
        try:
            warmup_step = self.config.TRAINER.WARMUP_STEP
        except Exception:
            warmup_step = 0

        if optimizer is not None and getattr(self.trainer, 'global_step', 0) < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + (self.trainer.global_step / warmup_step) * abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif self.config.TRAINER.WARMUP_TYPE == 'constant':
                pass
            else:
                raise ValueError(f'Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}')

        # execute closure if provided (Lightning v2 expects closure to be executed)
        result = None
        if callable(optimizer_closure):
            result = optimizer_closure()

        # perform optimizer step with fallbacks
        if optimizer is not None:
            try:
                optimizer.step()
            except TypeError:
                try:
                    optimizer.step(closure=optimizer_closure)
                except TypeError:
                    try:
                        optimizer.step(optimizer_closure)
                    except Exception:
                        optimizer.step()

            try:
                optimizer.zero_grad()
            except Exception:
                pass

        return result

    def _train_inference(self, batch):

        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config)

        with self.profiler.profile("Backbone"):
            with torch.autocast(enabled=self.config.JAMMA.MP, device_type='cuda'):
                self.backbone(batch)

        with self.profiler.profile("Matcher"):
            self.matcher(batch, mode='train')

        with self.profiler.profile("Compute fine supervision"):
            with torch.autocast(enabled=self.config.JAMMA.MP, device_type='cuda'):
                compute_supervision_fine(batch, self.config)

        with self.profiler.profile("Compute losses"):
            with torch.autocast(enabled=self.config.JAMMA.MP, device_type='cuda'):
                self.loss(batch)

    def _val_inference(self, batch):

        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config)

        with self.profiler.profile("Backbone"):
            with torch.autocast(enabled=self.config.JAMMA.MP, device_type='cuda'):
                self.backbone(batch)

        with self.profiler.profile("Matcher"):
            self.matcher(batch, mode='val')

        with self.profiler.profile("Compute fine supervision"):
            with torch.autocast(enabled=self.config.JAMMA.MP, device_type='cuda'):
                compute_supervision_fine(batch, self.config)

        with self.profiler.profile("Compute losses"):
            with torch.autocast(enabled=self.config.JAMMA.MP, device_type='cuda'):
                self.loss(batch)

    def _compute_metrics_val(self, batch):
        with self.profiler.profile("Copmute metrics"):
            compute_f1(batch)  # compute fi-score
            compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
            compute_pose_errors(batch, self.config)  # compute R_errs, t_errs, pose_errs for each pair

            rel_pair_names = list(zip(*batch['pair_names']))
            bs = batch['imagec_0'].size(0)
            metrics = {
                # to filter duplicate pairs caused by DistributedSampler
                'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
                'epi_errs': [batch['epi_errs'][batch['m_bids'] == b].cpu().numpy() for b in range(bs)],
                'precision': batch['precision'],
                'recall': batch['recall'],
                'f1_score': batch['f1_score'],
                'R_errs': batch['R_errs'],
                't_errs': batch['t_errs'],
                'inliers': batch['inliers']}
            ret_dict = {'metrics': metrics}
        return ret_dict, rel_pair_names

    def _compute_metrics(self, batch):
        with self.profiler.profile("Copmute metrics"):
            compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
            compute_pose_errors(batch, self.config)  # compute R_errs, t_errs, pose_errs for each pair

            rel_pair_names = list(zip(*batch['pair_names']))
            bs = batch['imagec_0'].size(0)
            metrics = {
                # to filter duplicate pairs caused by DistributedSampler
                'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
                'epi_errs': [batch['epi_errs'][batch['m_bids'] == b].cpu().numpy() for b in range(bs)],
                'R_errs': batch['R_errs'],
                't_errs': batch['t_errs'],
                'inliers': batch['inliers']}
            ret_dict = {'metrics': metrics}
        return ret_dict, rel_pair_names

    def training_step(self, batch, batch_idx):

        self._train_inference(batch)
        # logging
        if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            # scalars
            self.logger.experiment.add_scalar(f'train_loss', batch['loss'], self.global_step)

        # store outputs for on_train_epoch_end (Lightning v2+)
        try:
            self._train_epoch_outputs.append({'loss': batch['loss'].detach()})
        except Exception:
            # fallback if detach fails
            self._train_epoch_outputs.append({'loss': batch['loss']})

        return {'loss': batch['loss']}
    def on_train_epoch_end(self):
        outputs = self._train_epoch_outputs
        if len(outputs) == 0:
            return
        # compute average loss
        try:
            avg_loss = torch.stack([x['loss'].cpu() for x in outputs]).mean()
        except Exception:
            avg_loss = torch.tensor(0.0)

        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar('train/avg_loss_on_epoch', avg_loss, global_step=self.current_epoch)

        # clear stored outputs
        self._train_epoch_outputs = []

    def validation_step(self, batch, batch_idx):
        self._val_inference(batch)
        ret_dict, _ = self._compute_metrics_val(batch)
        ret_dict['metrics'] = {**ret_dict['metrics'], 'max_matches': [batch['num_candidates_max']]}
        val_plot_interval = max(self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        figures = {self.config.TRAINER.PLOT_MODE: []}

        out = {
            **ret_dict,
            'loss_scalars': batch['loss_scalars'],
            'figures': figures,
        }
        # store for on_validation_epoch_end (Lightning v2+)
        try:
            self._val_epoch_outputs.append(out)
        except Exception:
            pass
        return out

    def on_validation_epoch_end(self):
        outputs = self._val_epoch_outputs
        if len(outputs) == 0:
            return
        # handle multiple validation sets
        multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        multi_val_metrics = defaultdict(list)

        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training
            cur_epoch = self.trainer.current_epoch
            # Lightning v2 removed `resume_from_checkpoint`; use `running_sanity_check` alone
            if getattr(self.trainer, 'running_sanity_check', False):
                cur_epoch = -1

            # 1. loss_scalars: dict of list, on cpu
            _loss_scalars = [o['loss_scalars'] for o in outputs]
            loss_scalars = {k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars])) for k in _loss_scalars[0]}

            # 2. val metrics: dict of list, numpy
            _metrics = [o['metrics'] for o in outputs]
            metrics = {k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
            # NOTE: all ranks need to `aggregate_merics`, but only log at rank-0
            val_metrics_4tb = aggregate_metrics_train_val(metrics, self.config.TRAINER.EPI_ERR_THR, config=self.config)
            for thr in [5, 10, 20]:
                multi_val_metrics[f'auc@{thr}'].append(val_metrics_4tb[f'auc@{thr}'])

            # 3. figures
            _figures = [o['figures'] for o in outputs]
            figures = {k: flattenList(gather(flattenList([_me[k] for _me in _figures]))) for k in _figures[0]}

            # tensorboard records only on rank 0
            if self.trainer.global_rank == 0:
                for k, v in loss_scalars.items():
                    mean_v = torch.stack(v).mean()
                    self.logger.experiment.add_scalar(f'val_{valset_idx}/avg_{k}', mean_v, global_step=cur_epoch)

                for k, v in val_metrics_4tb.items():
                    self.logger.experiment.add_scalar(f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch)

                for k, v in figures.items():
                    if self.trainer.global_rank == 0:
                        for plot_idx, fig in enumerate(v):
                            self.logger.experiment.add_figure(
                                f'val_match_{valset_idx}/{k}/pair-{plot_idx}', fig, cur_epoch, close=True)
            plt.close('all')

        for thr in [5, 10, 20]:
            # log on all ranks for ModelCheckpoint callback to work properly
            self.log(f'auc@{thr}', torch.tensor(np.mean(multi_val_metrics[f'auc@{thr}'])))  # ckpt monitors on this
        print(val_metrics_4tb)

        # clear stored outputs
        self._val_epoch_outputs = []

    def test_step(self, batch, batch_idx):
        with torch.autocast(enabled=self.config.JAMMA.MP, device_type='cuda'):
            self.start_event.record()
            with self.profiler.profile("Backbone"):
                self.backbone(batch)

            with self.profiler.profile("Matcher"):
                self.matcher(batch, mode='test')

            self.end_event.record()
            torch.cuda.synchronize()
            self.total_ms += self.start_event.elapsed_time(self.end_event)
            batch['runtime'] = self.start_event.elapsed_time(self.end_event)

        ret_dict, rel_pair_names = self._compute_metrics(batch)

        # Visualization #
        # path = str(self.viz_path) + '/' + str(batch_idx)
        # make_matching_figures(batch, 'confidence', path=path+'_confidence.png')
        # make_matching_figures(batch, 'evaluation', path=path+'_evaluation.png')
        # make_matching_figures(batch, 'wheel', path=path+'_wheel.png')

        with self.profiler.profile("dump_results"):
            if self.dump_dir is not None:
                # dump results for further analysis
                bs = batch['imagec_0'].shape[0]
                dumps = []
                for b_id in range(bs):
                    item = {}
                    mask = batch['m_bids'] == b_id
                    epi_errs = batch['epi_errs'][mask].cpu().numpy()
                    correct_mask = epi_errs < 1e-4
                    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
                    n_correct = np.sum(correct_mask)
                    item['precision'] = precision
                    item['n_correct'] = n_correct
                    item['runtime'] = batch['runtime']
                    for key in ['R_errs', 't_errs']:
                        item[key] = batch[key][b_id][0]
                    dumps.append(item)
                ret_dict['dumps'] = dumps
        # store outputs for use in on_test_epoch_end (Lightning v2+)
        self._test_outputs.append(ret_dict)
        return ret_dict

    def on_test_epoch_end(self):
        outputs = self._test_outputs
        # metrics: dict of list, numpy
        _metrics = [o['metrics'] for o in outputs]
        metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}

        # [{key: [{...}, *#bs]}, *#batch]
        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList([o['dumps'] for o in outputs])  # [{...}, #bs*#batch]
            dumps = flattenList(gather(_dumps))  # [{...}, #proc*#bs*#batch]
            logger.info(f'Prediction and evaluation results will be saved to: {self.dump_dir}')

        if self.trainer.global_rank == 0:
            print(self.profiler.summary())
            val_metrics_4tb = aggregate_metrics_test(metrics, self.config.TRAINER.EPI_ERR_THR, config=self.config)
            logger.info('\n' + pprint.pformat(val_metrics_4tb))
            print('Averaged Matching time over 1500 pairs: {:.2f} ms'.format(self.total_ms / 1500))
            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / 'JAMMA_pred_eval', dumps)

        # clear stored outputs
        self._test_outputs = []

    # def test_step(self, batch, batch_idx):
    #     flops1, params1 = profile(self.backbone, inputs=[batch])
    #     flops2, params2 = profile(self.matcher, inputs=[batch])
    #     return flops1+flops2
    #
    # def test_epoch_end(self, outputs):
    #     flops_mean = sum(outputs)/len(outputs) / 1e9
    #     print("mean flops: {}G".format(flops_mean))
