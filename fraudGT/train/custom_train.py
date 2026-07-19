import copy
import json
import logging
import os
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData, Data
from torch_geometric.utils import mask_to_index, index_to_mask
from fraudGT.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.loader import create_loader, get_loader
from fraudGT.graphgym.loss import compute_loss
from fraudGT.graphgym.register import register_train
from fraudGT.graphgym.model_builder import create_model
from fraudGT.graphgym.optimizer import create_optimizer, create_scheduler
from fraudGT.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from fraudGT.graphgym.utils.comp_budget import params_count

from fraudGT.utils import cfg_to_dict, flatten_dict, make_wandb_name
from fraudGT.utils import (new_optimizer_config, new_scheduler_config)
from fraudGT.timer import runtime_stats_cuda, is_performance_stats_enabled, enable_runtime_stats, disable_runtime_stats

def check_grad(model):
    for name, param in model.named_parameters():
        if 'attn_bias' in name:
            print(param.grad)
        # if param.requires_grad:
        #     if param.grad is None:
        #         print(f'{name} has no gradient')
        #     elif torch.isnan(param.grad).any():
        #         print(f'{name} has NaN gradients')
        #     elif torch.isinf(param.grad).any():
        #         print(f'{name} has Inf gradients')

def get_best_epoch(val_perf):
    best_epoch = int(np.array([vp['loss'] for vp in val_perf]).argmin())
    if cfg.metric_best == 'auto':
        return best_epoch

    metric = cfg.metric_best
    primary_values = np.array([vp[metric] for vp in val_perf])
    best_epoch = int(getattr(primary_values, cfg.metric_agg)())

    topk = int(getattr(cfg.train, 'selection_topk_by_metric', 1))
    tiebreak_metric = getattr(cfg.train, 'selection_tiebreak_metric', "")
    tiebreak_agg = getattr(cfg.train, 'selection_tiebreak_agg', 'argmax')
    if topk <= 1 or not tiebreak_metric:
        return best_epoch

    ranked = np.argsort(primary_values)
    if cfg.metric_agg == 'argmax':
        candidate_idx = ranked[-topk:]
    elif cfg.metric_agg == 'argmin':
        candidate_idx = ranked[:topk]
    else:
        raise ValueError(f'Unsupported metric aggregation: {cfg.metric_agg}')

    secondary_values = np.array([val_perf[idx][tiebreak_metric]
                                 for idx in candidate_idx])
    if tiebreak_agg == 'argmax':
        return int(candidate_idx[int(secondary_values.argmax())])
    if tiebreak_agg == 'argmin':
        return int(candidate_idx[int(secondary_values.argmin())])
    raise ValueError(f'Unsupported tiebreak aggregation: {tiebreak_agg}')


def get_early_stop_metric():
    metric = getattr(cfg.train, 'early_stop_metric', "")
    if metric:
        return metric
    if cfg.metric_best != 'auto':
        return cfg.metric_best
    return 'loss'


def is_early_stop_better(value, best_value, delta):
    if best_value is None:
        return True
    if get_early_stop_metric() == 'loss':
        return value < best_value - delta
    if cfg.metric_agg == 'argmin':
        return value < best_value - delta
    return value > best_value + delta


def update_early_stop_state(val_perf, cur_epoch, state):
    if not getattr(cfg.train, 'early_stop', False):
        return False
    metric = get_early_stop_metric()
    if not val_perf or metric not in val_perf[-1]:
        return False

    value = float(val_perf[-1][metric])
    delta = float(getattr(cfg.train, 'early_stop_delta', 0.0))
    if is_early_stop_better(value, state['best_value'], delta):
        state['best_value'] = value
        state['best_epoch'] = cur_epoch

    min_epoch = int(getattr(cfg.train, 'early_stop_min_epoch', 0))
    patience = int(getattr(cfg.train, 'early_stop_patience', 0))
    if patience <= 0 or (cur_epoch + 1) < min_epoch:
        return False

    stale_epochs = cur_epoch - state['best_epoch']
    if stale_epochs < patience:
        return False

    payload = {
        'stopped_epoch': cur_epoch,
        'best_epoch': state['best_epoch'],
        'metric': metric,
        'best_value': state['best_value'],
        'patience': patience,
        'min_epoch': min_epoch,
        'max_epoch': int(cfg.optim.max_epoch),
    }
    path = os.path.join(cfg.run_dir, 'early_stop.json')
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    logging.info(
        'Early stop at epoch %s: best %s=%.5f at epoch %s; '
        'patience=%s, min_epoch=%s, max_epoch=%s',
        cur_epoch, metric, state['best_value'], state['best_epoch'],
        patience, min_epoch, cfg.optim.max_epoch)
    return True


# def train_epoch(logger, loader, model, optimizer, scheduler, batch_accumulation):
#     model.train()
#     optimizer.zero_grad()
#     time_start = time.time()
#     for iter, batch in enumerate(tqdm(loader, disable=not cfg.train.tqdm)):
#         batch.split = 'train'
#         batch.to(torch.device(cfg.device))
#         pred, true = model(batch)

#         loss, pred_score = compute_loss(pred, true)
#         _true = true.detach().to('cpu', non_blocking=True)
#         _pred = pred_score.detach().to('cpu', non_blocking=True)

#         loss.backward()
#         # check_grad(model)
#         # Parameters update after accumulating gradients for given num. batches.
#         if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
#             if cfg.optim.clip_grad_norm:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(),
#                                                cfg.optim.clip_grad_norm_value)
#             optimizer.step()
#             optimizer.zero_grad()
#         cfg.params = params_count(model)
#         logger.update_stats(true=_true,
#                             pred=_pred,
#                             loss=loss.detach().cpu().item(),
#                             lr=scheduler.get_last_lr()[0],
#                             time_used=time.time() - time_start,
#                             params=cfg.params,
#                             dataset_name=cfg.dataset.name)
#         time_start = time.time()

def _set_evidence_gate_epoch(model, cur_epoch):
    '''Tell evidence-gate heads the current epoch (for warm-up schedules).'''
    for module in model.modules():
        if hasattr(module, '_eg_cur_epoch'):
            module._eg_cur_epoch = cur_epoch


def _evidence_gate_penalty(model):
    '''Sum any stashed evidence-gate routing penalties (0 if none).'''
    penalty = 0.0
    for module in model.modules():
        pen = getattr(module, '_eg_gate_penalty', None)
        if pen is not None:
            penalty = penalty + pen
    return penalty


def _evidence_gate_aux_loss(model):
    '''Train a v4 structural candidate before the learned router trusts it.'''
    aux_total = 0.0
    for module in model.modules():
        logits = getattr(module, '_eg_struct_aux_logits', None)
        labels = getattr(module, '_eg_struct_aux_labels', None)
        scale = float(getattr(module, '_eg_struct_aux_scale', 0.0))
        if logits is not None and labels is not None and scale > 0.0:
            aux_loss, _ = compute_loss(logits, labels)
            aux_total = aux_total + scale * aux_loss
    return aux_total


def _uprc_training_anchor(model):
    anchors = [
        anchor for module in model.modules()
        for anchor in [getattr(module, '_uprc_anchor_logits', None)]
        if anchor is not None
    ]
    if len(anchors) > 1:
        raise RuntimeError("multiple UPRC heads are not supported")
    return anchors[0] if anchors else None


def _uprc_margin(logits):
    if logits.ndim == 1:
        return logits
    if logits.size(-1) == 1:
        return logits.view(-1)
    return logits[:, 1] - logits[:, 0]


def _uprc_hard_pair_rank_loss(anchor, candidate, labels, pair_limit,
                              temperature):
    """Pair the lowest-scored positives with highest-scored negatives."""
    labels = labels.view(-1).long()
    positive = (labels == 1).nonzero(as_tuple=False).view(-1)
    negative = (labels == 0).nonzero(as_tuple=False).view(-1)
    if positive.numel() == 0 or negative.numel() == 0:
        return candidate.sum() * 0.0
    anchor_score = _uprc_margin(anchor).detach()
    positive = positive[
        anchor_score[positive].argsort()[:pair_limit]]
    negative = negative[
        anchor_score[negative].argsort(descending=True)[:pair_limit]]
    score = _uprc_margin(candidate)
    difference = score[positive].unsqueeze(1) - score[negative].unsqueeze(0)
    return F.softplus(-difference / temperature).mean()


def _uprc_balanced_point_loss(candidate, labels):
    labels = labels.view(-1).long()
    score = _uprc_margin(candidate)
    losses = []
    positive = labels == 1
    negative = labels == 0
    if positive.any():
        losses.append(F.softplus(-score[positive]).mean())
    if negative.any():
        losses.append(F.softplus(score[negative]).mean())
    return torch.stack(losses).mean() if losses else score.sum() * 0.0


def _uprc_aux_loss(model):
    """Train only UPRC with balanced point and hard-pair ranking losses."""
    total = 0.0
    for module in model.modules():
        anchor = getattr(module, '_uprc_anchor_logits', None)
        candidate = getattr(module, '_uprc_candidate_logits', None)
        labels = getattr(module, '_uprc_labels', None)
        correction = getattr(module, '_uprc_correction', None)
        if anchor is None or candidate is None or labels is None:
            continue
        rank_loss = _uprc_hard_pair_rank_loss(
            anchor, candidate, labels, int(module.uprc_pair_limit),
            float(module.uprc_rank_temperature))
        balanced_loss = _uprc_balanced_point_loss(candidate, labels)
        center_loss = candidate.new_zeros(())
        norm_loss = candidate.new_zeros(())
        if correction is not None:
            signed = _uprc_margin(correction)
            center_loss = signed.mean().pow(2)
            norm_loss = correction.pow(2).sum(dim=-1).mean()
        total = total + (
            float(module.uprc_rank_loss_weight) * rank_loss +
            float(module.uprc_balanced_loss_weight) * balanced_loss +
            float(module.uprc_center_loss_weight) * center_loss +
            float(module.uprc_norm_loss_weight) * norm_loss)
    return total


def _uprc_parameters(model):
    parameters = []
    for module in model.modules():
        editor = getattr(module, 'uprc_editor', None)
        if editor is not None:
            parameters.extend(
                parameter for parameter in editor.parameters()
                if parameter.requires_grad)
    return parameters


def _clip_gradients(model, max_norm, isolate_uprc=False):
    if not isolate_uprc:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        return
    branch_parameters = _uprc_parameters(model)
    branch_ids = {id(parameter) for parameter in branch_parameters}
    anchor_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in branch_ids]
    if anchor_parameters:
        torch.nn.utils.clip_grad_norm_(anchor_parameters, max_norm)
    if branch_parameters:
        torch.nn.utils.clip_grad_norm_(branch_parameters, max_norm)


def train_epoch(cur_epoch, logger, loader, model, optimizer, scheduler, batch_accumulation):
    pbar = tqdm(total=len(loader), disable=not cfg.train.tqdm)
    pbar.set_description(f'Train epoch')
    _set_evidence_gate_epoch(model, cur_epoch)

    model.train()

    runtime_stats_cuda.start_epoch()

    runtime_stats_cuda.start_region("total")
    runtime_stats_cuda.start_region(
        "sampling", runtime_stats_cuda.get_last_event())
    iterator = iter(loader)
    runtime_stats_cuda.end_region("sampling")
    runtime_stats_cuda.end_region("total", runtime_stats_cuda.get_last_event())

    if cfg.model.type == 'LPModel': # Handle label propagation specially
        # We don't need to train label propagation
        time_start = time.time()
        batch = next(iterator, None)
        batch.split = 'train'
        batch.to(torch.device(cfg.device))
        
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)
        _true = true.detach().to('cpu', non_blocking=True)
        _pred = pred_score.detach().to('cpu', non_blocking=True)
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=0,
                            dataset_name=cfg.dataset.name)
        pbar.update(1)
        return

    optimizer.zero_grad()
    it = 0
    time_start = time.time()
    # with torch.autograd.set_detect_anomaly(True):
    while True:
        try:
            torch.cuda.empty_cache() 
            runtime_stats_cuda.start_region(
                "total", runtime_stats_cuda.get_last_event())
            runtime_stats_cuda.start_region(
                "sampling", runtime_stats_cuda.get_last_event())
            # print('Crashed?')
            batch = next(iterator, None)
            # print('Crashed?')
            it += 1
            if batch is None:
                runtime_stats_cuda.end_region("sampling")
                runtime_stats_cuda.end_region(
                    "total", runtime_stats_cuda.get_last_event())
                break
            runtime_stats_cuda.end_region("sampling")

            runtime_stats_cuda.start_region("data_transfer", runtime_stats_cuda.get_last_event())
            if isinstance(batch, Data) or isinstance(batch, HeteroData):
                batch.split = 'train'
                batch.to(torch.device(cfg.device))
            else: # NAGphormer, HINo
                batch = [x.to(torch.device(cfg.device)) for x in batch]
            runtime_stats_cuda.end_region("data_transfer")

            runtime_stats_cuda.start_region("train", runtime_stats_cuda.get_last_event())
            runtime_stats_cuda.start_region("forward", runtime_stats_cuda.get_last_event())
            pred, true = model(batch)
            runtime_stats_cuda.end_region("forward")
            runtime_stats_cuda.start_region("loss", runtime_stats_cuda.get_last_event())
            if cfg.model.loss_fun == 'curriculum_learning_loss':
                reported_loss, pred_score = compute_loss(pred, true, cur_epoch)
            else:
                reported_loss, pred_score = compute_loss(pred, true)
            uprc_anchor = _uprc_training_anchor(model)
            if uprc_anchor is None:
                loss = reported_loss
            elif cfg.model.loss_fun == 'curriculum_learning_loss':
                loss, _ = compute_loss(uprc_anchor, true, cur_epoch)
            else:
                loss, _ = compute_loss(uprc_anchor, true)
            loss = (
                loss +
                _evidence_gate_penalty(model) +
                _evidence_gate_aux_loss(model) +
                _uprc_aux_loss(model)
            )
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
            runtime_stats_cuda.end_region("loss")

            runtime_stats_cuda.start_region("backward", runtime_stats_cuda.get_last_event())
            loss.backward()
            runtime_stats_cuda.end_region("backward")
            # print(loss.detach().cpu().item())
            # check_grad(model)
            # Parameters update after accumulating gradients for given num. batches.
            if ((it + 1) % batch_accumulation == 0) or (it + 1 == len(loader)):
                if cfg.optim.clip_grad_norm:
                    _clip_gradients(
                        model, cfg.optim.clip_grad_norm_value,
                        isolate_uprc=(uprc_anchor is not None))
                optimizer.step()
                optimizer.zero_grad()
            runtime_stats_cuda.end_region("train")
            runtime_stats_cuda.end_region("total", runtime_stats_cuda.get_last_event())
            cfg.params = params_count(model)
            logger.update_stats(true=_true,
                                pred=_pred,
                                loss=loss.detach().cpu().item(),
                                lr=scheduler.get_last_lr()[0],
                                time_used=time.time() - time_start,
                                params=cfg.params,
                                dataset_name=cfg.dataset.name)
            pbar.update(1)
            time_start = time.time()
        except RuntimeError as e:
            if "cannot sample n_sample <= 0 samples" in str(e):
                print(f"Skipping batch due to error: {e}")
                continue
            else:
                # If it's a different error, re-raise it
                raise
    
    runtime_stats_cuda.end_epoch()
    runtime_stats_cuda.report_stats(
        {'total': 'Total', 'data_transfer': 'Data Transfer', 'sampling': 'Sampling + Slicing', 'train': 'Train', \
         'attention': 'Attention', 'gt-layer': 'GT-Layer', 'forward': 'Forward', 'loss': 'Loss', 'backward': 'Backward'})


@torch.no_grad()
def eval_epoch(logger, loader, model, split='val'):
    model.eval()
    pbar = tqdm(total=len(loader), disable=not cfg.val.tqdm)
    iterator = iter(loader)

    time_start = time.time()
    it = 0
    while True:
        try:
            batch = next(iterator, None)
            it += 1
            if batch is None:
                break
            if isinstance(batch, Data) or isinstance(batch, HeteroData):
                batch.split = split
                batch.to(torch.device(cfg.device))
            else: # NAGphormer
                batch = [x.to(torch.device(cfg.device)) for x in batch]
            if cfg.gnn.head == 'inductive_edge':
                pred, true, extra_stats = model(batch)
            else:
                pred, true = model(batch)
                extra_stats = {}

            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
            
            logger.update_stats(true=_true,
                                pred=_pred,
                                loss=loss.detach().cpu().item(),
                                lr=0, time_used=time.time() - time_start,
                                params=cfg.params,
                                dataset_name=cfg.dataset.name,
                                **extra_stats)
            pbar.update(1)
            time_start = time.time()
        except RuntimeError as e:
            if "cannot sample n_sample <= 0 samples" in str(e):
                print(f"Skipping batch due to error: {e}")
                continue
            else:
                # If it's a different error, re-raise it
                raise


@register_train('custom')
def custom_train(loggers, loaders, model, optimizer, scheduler):
    """
    Customized training pipeline.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler,
                                cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch %s', start_epoch)

    if cfg.wandb.use:
        try:
            import wandb
        except:
            raise ImportError('WandB is not installed.')
        if cfg.wandb.name == '':
            wandb_name = make_wandb_name(cfg)
        else:
            wandb_name = cfg.wandb.name
        run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
                         name=wandb_name, dir='/nobackup/users/junhong/Logs/')
        run.config.update(cfg_to_dict(cfg))

    num_splits = len(loggers)
    split_names = ['val', 'test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]
    early_stop_state = {'best_value': None, 'best_epoch': start_epoch}
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        start_time = time.perf_counter()
        # enable_runtime_stats()
        train_epoch(cur_epoch, loggers[0], loaders[0], model, optimizer, scheduler,
                    cfg.optim.batch_accumulation)
        # disable_runtime_stats()
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        if is_eval_epoch(cur_epoch, start_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model,
                           split=split_names[i - 1])
                if i == 1 and cfg.model.auto_tune_thresh and \
                        loggers[i].task_type == 'classification_binary':
                    cfg.model.thresh = loggers[i].find_best_threshold()
                perf[i].append(loggers[i].write_epoch(cur_epoch))
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        val_perf = perf[1]
        if cfg.optim.scheduler == 'reduce_on_plateau':
            scheduler.step(val_perf[-1]['loss'])
        else:
            scheduler.step()
        full_epoch_times.append(time.perf_counter() - start_time)
        # Checkpoint with regular frequency (if enabled).
        if cfg.train.enable_ckpt and not cfg.train.ckpt_best \
                and is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

        if cfg.wandb.use:
            run.log(flatten_dict(perf), step=cur_epoch)

        # Log current best stats on eval epoch.
        if is_eval_epoch(cur_epoch, start_epoch):
            best_epoch = get_best_epoch(val_perf)
            best_train = best_val = best_test = ""
            if cfg.metric_best != 'auto':
                m = cfg.metric_best
                if m in perf[0][best_epoch]:
                    best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
                else:
                    # Note: For some datasets it is too expensive to compute
                    # the main metric on the training set.
                    best_train = f"train_{m}: {0:.4f}"
                best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
                best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

                if cfg.wandb.use:
                    bstats = {"best/epoch": best_epoch}
                    for i, s in enumerate(['train', 'val', 'test']):
                        bstats[f"best/{s}_loss"] = perf[i][best_epoch]['loss']
                        if m in perf[i][best_epoch]:
                            bstats[f"best/{s}_{m}"] = perf[i][best_epoch][m]
                            run.summary[f"best_{s}_perf"] = \
                                perf[i][best_epoch][m]
                        for x in ['hits@1', 'hits@3', 'hits@10', 'mrr']:
                            if x in perf[i][best_epoch]:
                                bstats[f"best/{s}_{x}"] = perf[i][best_epoch][x]
                    run.log(bstats, step=cur_epoch)
                    run.summary["full_epoch_time_avg"] = np.mean(full_epoch_times)
                    run.summary["full_epoch_time_sum"] = np.sum(full_epoch_times)
            # Checkpoint the best epoch params (if enabled).
            if cfg.train.enable_ckpt and cfg.train.ckpt_best and \
                    best_epoch == cur_epoch:
                save_ckpt(model, optimizer, scheduler, cur_epoch)
                if cfg.train.ckpt_clean:  # Delete old ckpt each time.
                    clean_ckpt()
            logging.info(
                f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
                f"(avg {np.mean(full_epoch_times):.1f}s) | "
                f"Best so far: epoch {best_epoch}\t"
                f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
                f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
                f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
            )
            if hasattr(model, 'trf_layers'):
                # Log SAN's gamma parameter values if they are trainable.
                for li, gtl in enumerate(model.trf_layers):
                    if torch.is_tensor(gtl.attention.gamma) and \
                            gtl.attention.gamma.requires_grad:
                        logging.info(f"    {gtl.__class__.__name__} {li}: "
                                     f"gamma={gtl.attention.gamma.item()}")
            if update_early_stop_state(val_perf, cur_epoch, early_stop_state):
                break
    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    # close wandb
    if cfg.wandb.use:
        run.finish()
        run = None

    logging.info('Task done, results saved in %s', cfg.run_dir)


@register_train('multi-stage')
def multi_stage_train(loggers, loaders, model, optimizer, scheduler):
    """
    Multi-stage training pipeline.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    del loaders
    # We don't want the val/test set be shuffled
    loaders, dataset = create_loader(shuffle=False, returnDataset=True)
    
    start_epoch = 0
    # if cfg.train.auto_resume:
    #     start_epoch = load_ckpt(model, optimizer, scheduler,
    #                             cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch %s', start_epoch)

    for stage in range(3): # cfg.stages
        if stage >= 0:
            # Load from last checkpoint and perform inference on whole dataset
            epoch = load_ckpt(model, None, None, -1)
            print(f'Loaded checkpoint at {epoch}')
            model.eval()
            time_start = time.time()
            new_dataset = copy.deepcopy(dataset)
            new_data = new_dataset.data
            num_splits = len(loggers)
            split_names = ['val', 'test']
            for split in range(1, num_splits):
                logger, loader = loggers[split], loaders[split]
                step = loader.step
                loader.set_step(-1)
                all_preds = []
                all_inds = []
                for batch in tqdm(loader, disable=not cfg.val.tqdm):
                    all_inds.append(batch[cfg.dataset.task_entity].input_id)
                    if isinstance(batch, Data) or isinstance(batch, HeteroData):
                        batch.split = split_names[split - 1]
                        batch.to(torch.device(cfg.device))
                    else: # NAGphormer
                        batch = batch = [x.to(torch.device(cfg.device)) for x in batch]
                    if cfg.gnn.head == 'inductive_edge':
                        pred, true, extra_stats = model(batch)
                    else:
                        pred, true = model(batch)
                        extra_stats = {}

                    loss, pred_score = compute_loss(pred, true)
                    _true = true.detach().to('cpu', non_blocking=True)
                    _pred = pred_score.detach().to('cpu', non_blocking=True)

                    all_preds.append(_pred)
                    
                    logger.update_stats(true=_true,
                                        pred=_pred,
                                        loss=loss.detach().cpu().item(),
                                        lr=0, time_used=time.time() - time_start,
                                        params=cfg.params,
                                        dataset_name=cfg.dataset.name,
                                        **extra_stats)
                    time_start = time.time()
                logger.write_epoch(0)
                loader.set_step(step)
                all_inds, permu = torch.sort(torch.cat(all_inds))
                all_preds = torch.cat(all_preds)[permu]
                preds = all_preds.argmax(dim=-1)
                predict_prob = all_preds.softmax(dim=-1)

                task = cfg.dataset.task_entity

                mask_name = f'{split_names[split - 1]}_mask'
                confident_mask = predict_prob.max(dim=1)[0] > 0.95 #args.threshold
                confident_inds = all_inds[confident_mask]
                # new_data[task].y[confident_inds] = preds[confident_mask].unsqueeze(-1)
                # new_data[task].y[confident_inds] = preds[confident_mask]
                # new_data[task].train_mask |= index_to_mask(confident_inds, new_data[task].train_mask.shape[0])
                # new_data[task].train_mask |= new_data[task].val_mask
                # new_data[task].train_mask |= new_data[task].test_mask

                confident_level = (new_data[task].y[confident_inds] == 
                                   dataset[0][task].y[confident_inds]).sum() / confident_inds.numel()
                print(f"{split_names[split - 1]} confident nodes: {confident_inds.numel()} / {dataset[0][task][mask_name].sum()}, {split_names[split - 1]} confident_level: {confident_level}")

            # Update the training loader
            new_loaders = create_loader(dataset=new_dataset)
            loaders[0] = new_loaders[0]

            # Reinitialize the model
            model = create_model(dataset=new_dataset)
            optimizer = create_optimizer(model.named_parameters(), #model.named_parameters(),
                                     new_optimizer_config(cfg))
            scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
            start_epoch = 0

        if cfg.wandb.use:
            try:
                import wandb
            except:
                raise ImportError('WandB is not installed.')
            if cfg.wandb.name == '':
                wandb_name = make_wandb_name(cfg)
            else:
                wandb_name = cfg.wandb.name
            run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
                            name=f'{wandb_name}_MS{stage}', dir='/nobackup/users/junhong/Logs/')
            run.config.update(cfg_to_dict(cfg))

        num_splits = len(loggers)
        split_names = ['val', 'test']
        full_epoch_times = []
        perf = [[] for _ in range(num_splits)]
        for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
            start_time = time.perf_counter()
            # enable_runtime_stats()
            train_epoch(cur_epoch, loggers[0], loaders[0], model, optimizer, scheduler,
                        cfg.optim.batch_accumulation)
            # disable_runtime_stats()
            perf[0].append(loggers[0].write_epoch(cur_epoch))

            if is_eval_epoch(cur_epoch, start_epoch):
                for i in range(1, num_splits):
                    eval_epoch(loggers[i], loaders[i], model,
                            split=split_names[i - 1])
                    perf[i].append(loggers[i].write_epoch(cur_epoch))
            else:
                for i in range(1, num_splits):
                    perf[i].append(perf[i][-1])

            val_perf = perf[1]
            if cfg.optim.scheduler == 'reduce_on_plateau':
                scheduler.step(val_perf[-1]['loss'])
            else:
                scheduler.step()
            full_epoch_times.append(time.perf_counter() - start_time)
            # Checkpoint with regular frequency (if enabled).
            if cfg.train.enable_ckpt and not cfg.train.ckpt_best \
                    and is_ckpt_epoch(cur_epoch):
                save_ckpt(model, optimizer, scheduler, cur_epoch)

            if cfg.wandb.use:
                run.log(flatten_dict(perf), step=cur_epoch)

            # Log current best stats on eval epoch.
            if is_eval_epoch(cur_epoch, start_epoch):
                best_epoch = get_best_epoch(val_perf)
                best_train = best_val = best_test = ""
                if cfg.metric_best != 'auto':
                    m = cfg.metric_best
                    if m in perf[0][best_epoch]:
                        best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
                    else:
                        # Note: For some datasets it is too expensive to compute
                        # the main metric on the training set.
                        best_train = f"train_{m}: {0:.4f}"
                    best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
                    best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

                    if cfg.wandb.use:
                        bstats = {"best/epoch": best_epoch}
                        for i, s in enumerate(['train', 'val', 'test']):
                            bstats[f"best/{s}_loss"] = perf[i][best_epoch]['loss']
                            if m in perf[i][best_epoch]:
                                bstats[f"best/{s}_{m}"] = perf[i][best_epoch][m]
                                run.summary[f"best_{s}_perf"] = \
                                    perf[i][best_epoch][m]
                            for x in ['hits@1', 'hits@3', 'hits@10', 'mrr']:
                                if x in perf[i][best_epoch]:
                                    bstats[f"best/{s}_{x}"] = perf[i][best_epoch][x]
                        run.log(bstats, step=cur_epoch)
                        run.summary["full_epoch_time_avg"] = np.mean(full_epoch_times)
                        run.summary["full_epoch_time_sum"] = np.sum(full_epoch_times)
                # Checkpoint the best epoch params (if enabled).
                if cfg.train.enable_ckpt and cfg.train.ckpt_best and \
                        best_epoch == cur_epoch:
                    save_ckpt(model, optimizer, scheduler, cur_epoch)
                    if cfg.train.ckpt_clean:  # Delete old ckpt each time.
                        clean_ckpt()
                logging.info(
                    f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
                    f"(avg {np.mean(full_epoch_times):.1f}s) | "
                    f"Best so far: epoch {best_epoch}\t"
                    f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
                    f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
                    f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
                )
                if hasattr(model, 'trf_layers'):
                    # Log SAN's gamma parameter values if they are trainable.
                    for li, gtl in enumerate(model.trf_layers):
                        if torch.is_tensor(gtl.attention.gamma) and \
                                gtl.attention.gamma.requires_grad:
                            logging.info(f"    {gtl.__class__.__name__} {li}: "
                                        f"gamma={gtl.attention.gamma.item()}")
        logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
        logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")
        # close wandb
        if cfg.wandb.use:
            run.finish()
            run = None
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in %s', cfg.run_dir)


@register_train('inference-only')
def inference_only(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to run inference only.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    num_splits = len(loggers)
    split_names = ['train', 'val', 'test']
    perf = [[] for _ in range(num_splits)]
    cur_epoch = 0
    start_time = time.perf_counter()

    for i in range(0, num_splits):
        eval_epoch(loggers[i], loaders[i], model,
                   split=split_names[i])
        perf[i].append(loggers[i].write_epoch(cur_epoch))

    best_epoch = 0
    best_train = best_val = best_test = ""
    if cfg.metric_best != 'auto':
        # Select again based on val perf of `cfg.metric_best`.
        m = cfg.metric_best
        if m in perf[0][best_epoch]:
            best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
        else:
            # Note: For some datasets it is too expensive to compute
            # the main metric on the training set.
            best_train = f"train_{m}: {0:.4f}"
        best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
        best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

    logging.info(
        f"> Inference | "
        f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
        f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
        f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
    )
    logging.info(f'Done! took: {time.perf_counter() - start_time:.2f}s')
    for logger in loggers:
        logger.close()


@ register_train('log-attn-weights')
def log_attn_weights(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to inference on the test set and log the attention
    weights in Transformer modules.

    Args:
        loggers: Unused, exists just for API compatibility
        loaders: List of loaders
        model (torch.nn.Module): GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    import os.path as osp
    from torch_geometric.loader.dataloader import DataLoader
    from fraudGT.utils import unbatch, unbatch_edge_index

    start_time = time.perf_counter()

    # The last loader is a test set.
    l = loaders[-1]
    # To get a random sample, create a new loader that shuffles the test set.
    loader = DataLoader(l.dataset, batch_size=l.batch_size,
                        shuffle=True, num_workers=0)

    output = []
    # batch = next(iter(loader))  # Run one random batch.
    for b_index, batch in enumerate(loader):
        bsize = batch.batch.max().item() + 1  # Batch size.
        if len(output) >= 128:
            break
        print(f">> Batch {b_index}:")

        X_orig = unbatch(batch.x.cpu(), batch.batch.cpu())
        batch.to(torch.device(cfg.device))
        model.eval()
        model(batch)

        # Unbatch to individual graphs.
        X = unbatch(batch.x.cpu(), batch.batch.cpu())
        edge_indices = unbatch_edge_index(batch.edge_index.cpu(),
                                          batch.batch.cpu())
        graphs = []
        for i in range(bsize):
            graphs.append({'num_nodes': len(X[i]),
                           'x_orig': X_orig[i],
                           'x_final': X[i],
                           'edge_index': edge_indices[i],
                           'attn_weights': []  # List with attn weights in layers from 0 to L-1.
                           })

        # Iterate through GPS layers and pull out stored attn weights.
        for l_i, (name, module) in enumerate(model.model.layers.named_children()):
            if hasattr(module, 'attn_weights'):
                print(l_i, name, module.attn_weights.shape)
                for g_i in range(bsize):
                    # Clip to the number of nodes in this graph.
                    # num_nodes = graphs[g_i]['num_nodes']
                    # aw = module.attn_weights[g_i, :num_nodes, :num_nodes]
                    aw = module.attn_weights[g_i]
                    graphs[g_i]['attn_weights'].append(aw.cpu())
        output += graphs

    logging.info(
        f"[*] Collected a total of {len(output)} graphs and their "
        f"attention weights for {len(output[0]['attn_weights'])} layers.")

    # Save the graphs and their attention stats.
    save_file = osp.join(cfg.run_dir, 'graph_attn_stats.pt')
    logging.info(f"Saving to file: {save_file}")
    torch.save(output, save_file)

    logging.info(f'Done! took: {time.perf_counter() - start_time:.2f}s')
