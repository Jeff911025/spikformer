#!/usr/bin/env python3
"""
Spikformer 完整訓練流程：訓練 -> Post-training Pruning -> Finetune

這個腳本實現了完整的模型壓縮流程：
1. 完整訓練 300 epochs
2. 基於 spike map 的 post-training pruning
3. 對 pruned 模型進行 finetune
"""

import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from spikingjelly.clock_driven import functional

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.data import create_dataset, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from loader import create_loader
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, \
    convert_splitbn_model, model_parameters
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler
import model
from pruning_utils import SpikeMapPruner

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

# 參數解析
parser = argparse.ArgumentParser(description='Spikformer: Train -> Prune -> Finetune')

# Model parameters
parser.add_argument('--model', default='spikformer', type=str, metavar='MODEL',
                    help='Name of model to train (default: "spikformer"')
parser.add_argument('-T', '--time-step', type=int, default=4, metavar='time',
                    help='simulation time step of spiking neuron (default: 4)')
parser.add_argument('-L', '--layer', type=int, default=4, metavar='layer',
                    help='model layer (default: 4)')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--dim', type=int, default=None, metavar='N',
                    help='embedding dimsension of feature')
parser.add_argument('--num_heads', type=int, default=None, metavar='N',
                    help='attention head number')
parser.add_argument('--patch-size', type=int, default=None, metavar='N',
                    help='Image patch size')
parser.add_argument('--mlp-ratio', type=int, default=None, metavar='N',
                    help='expand ration of embedding dimension in MLP block')

# Dataset parameters
parser.add_argument('-data-dir', metavar='DIR', default="/root/data/nas07/PersonalData/Jeff0102030433/CIFAR10/",
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='torch/cifar10',
                    help='dataset type (default: torch/cifar10)')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')

# Training parameters
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb', '--val-batch-size', type=int, default=16, metavar='N',
                    help='input val batch size for training (default: 16)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--finetune-epochs', type=int, default=50, metavar='N',
                    help='number of epochs to finetune (default: 50)')

# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--finetune-lr', type=float, default=0.001, metavar='LR',
                    help='finetune learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')

# Pruning parameters
parser.add_argument('--pruning-ratio', type=float, default=0.3, metavar='RATIO',
                    help='Pruning ratio (default: 0.3)')
parser.add_argument('--pruning-samples', type=int, default=100, metavar='N',
                    help='Number of samples to use for pruning evaluation (default: 100)')
parser.add_argument('--prune-type', type=str, default='masking', choices=['masking', 'structural'],
                    help='剪枝方式：masking（unstructured baseline）或 structural（論文主結果）')

# Output parameters
parser.add_argument('--output-dir', default='./output/train_prune_finetune', type=str,
                    help='Output directory for results')
parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='Experiment name (default: auto-generated)')

# Workflow control
parser.add_argument('--do-train', action='store_true', default=True, help='執行訓練')
parser.add_argument('--do-prune', action='store_true', default=True, help='執行 pruning')
parser.add_argument('--do-finetune', action='store_true', default=True, help='執行 finetune')
parser.add_argument('--eval-unpruned', action='store_true', help='只評估訓練好（未 prune）模型')
parser.add_argument('--eval-pruned', action='store_true', help='只評估 pruning 後模型')
parser.add_argument('--eval-finetuned', action='store_true', help='只評估 finetuned 模型')
parser.add_argument('--load-unpruned', type=str, default='', help='直接載入訓練好模型（跳過訓練）')
parser.add_argument('--load-pruned', type=str, default='', help='直接載入 pruning 後模型（跳過訓練和 pruning）')

# Other parameters
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--workers', type=int, default=4, metavar='N',
                    help='dataloader workers (default: 4)')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient GPU transfer')


def train_epoch(epoch, model, loader, optimizer, loss_fn, args, device):
    """訓練一個 epoch"""
    model.train()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    end = time.time()
    last_idx = len(loader) - 1
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        
        input, target = input.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        # 重置神經元狀態
        functional.reset_net(model)

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        torch.cuda.synchronize()
        batch_time_m.update(time.time() - end)
        end = time.time()

        if last_batch or batch_idx % 100 == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            _logger.info(
                'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                'LR: {lr:.3e}  '
                'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                    epoch,
                    batch_idx, len(loader),
                    100. * batch_idx / last_idx,
                    loss=losses_m,
                    batch_time=batch_time_m,
                    rate=input.size(0) / batch_time_m.val,
                    lr=lr,
                    data_time=data_time_m))

    return OrderedDict([('loss', losses_m.avg)])


def validate(model, loader, loss_fn, args, device):
    """驗證模型"""
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            input = input.to(device)
            target = target.to(device)

            # 重置神經元狀態
            functional.reset_net(model)

            output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()

            if last_batch or batch_idx % 100 == 0:
                _logger.info(
                    'Test: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
    return metrics


def post_training_pruning(model, train_loader, args, device):
    """Post-training pruning"""
    _logger.info("Starting post-training pruning...")
    
    # 創建 pruner
    pruner = SpikeMapPruner(model, pruning_ratio=args.pruning_ratio)
    
    # 獲取樣本數據用於 spike map 計算
    sample_inputs = []
    sample_count = 0
    
    model.eval()
    with torch.no_grad():
        for inputs, _ in train_loader:
            if sample_count >= args.pruning_samples:
                break
            inputs = inputs.to(device)
            sample_inputs.append(inputs)
            sample_count += inputs.size(0)
    
    # 合併樣本輸入
    sample_input = torch.cat(sample_inputs, dim=0)
    
    # 計算 channel 分數
    _logger.info("Computing channel scores...")
    scores = pruner.compute_channel_scores(sample_input)
    _logger.info(f"Computed scores for {len(scores)} modules")
    
    # 選擇要 pruning 的 channels
    channels_to_prune = pruner.select_channels_to_prune(scores)
    
    # 應用 pruning
    _logger.info(f"Applying {args.prune_type} pruning...")
    if args.prune_type == 'masking':
        pruned_model = pruner.apply_pruning(channels_to_prune)
    else:
        keep_indices_dict = {}
        for k, v in channels_to_prune.items():
            total = len(v) + len([i for i in range(10000) if i not in v])  # 粗略估計
            keep_indices = [i for i in range(total) if i not in v]
            keep_indices_dict[k] = keep_indices
        pruned_model = pruner.structural_prune_and_rebuild(keep_indices_dict)
    pruned_model = pruned_model.to(device)
    
    # 計算參數減少
    original_params = sum(p.numel() for p in model.parameters())
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    reduction_ratio = (original_params - pruned_params) / original_params
    
    _logger.info(f"Pruning completed!")
    _logger.info(f"Original parameters: {original_params:,}")
    _logger.info(f"Pruned parameters: {pruned_params:,}")
    _logger.info(f"Reduction ratio: {reduction_ratio:.2%}")
    
    return pruned_model, reduction_ratio


def main():
    args = parser.parse_args()
    
    # 設置日誌
    setup_default_logging()
    
    # 設置設備
    args.device = 'cuda'
    args.distributed = False
    args.world_size = 1
    args.rank = 0
    args.local_rank = 0
    
    # 設置隨機種子
    random_seed(args.seed, args.rank)
    
    # 創建模型
    model = create_model(
        'spikformer',
        pretrained=False,
        drop_rate=0.,
        drop_path_rate=0.,
        drop_block_rate=None,
        img_size_h=args.img_size, img_size_w=args.img_size,
        patch_size=args.patch_size, embed_dims=args.dim, num_heads=args.num_heads, mlp_ratios=args.mlp_ratio,
        in_channels=3, num_classes=args.num_classes, qkv_bias=False,
        depths=args.layer, sr_ratios=1,
        T=args.time_step
    )
    
    model = model.cuda()
    
    # 創建數據加載器
    data_config = resolve_data_config(vars(args), model=model, verbose=True)
    
    dataset_train = create_dataset(
        args.dataset, root=args.data_dir, split=args.train_split, is_training=True,
        batch_size=args.batch_size)
    dataset_eval = create_dataset(
        args.dataset, root=args.data_dir, split=args.val_split, is_training=False, batch_size=args.batch_size)
    
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=True,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        pin_memory=args.pin_mem,
    )
    
    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.val_batch_size,
        is_training=False,
        use_prefetcher=True,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )
    
    # 設置損失函數
    train_loss_fn = nn.CrossEntropyLoss().cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()
    
    # 創建輸出目錄
    if args.experiment:
        exp_name = args.experiment
    else:
        exp_name = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            safe_model_name(args.model),
            str(data_config['input_size'][-1])
        ])
    output_dir = get_outdir(args.output_dir, exp_name)
    
    _logger.info(f'Output directory: {output_dir}')
    
    # 載入模型（如指定）
    if args.load_unpruned:
        _logger.info(f'Loading unpruned model from {args.load_unpruned}')
        checkpoint = torch.load(args.load_unpruned, map_location=args.device)
        model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
        best_metric = checkpoint.get('best_metric', None)
        best_epoch = checkpoint.get('epoch', None)
        _logger.info('Unpruned model loaded!')
        do_train = False
    else:
        do_train = args.do_train
    
    # 訓練階段
    if do_train:
        _logger.info("=" * 50)
        _logger.info("STAGE 1: Full Training")
        _logger.info("=" * 50)
        
        optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
        lr_scheduler, _ = create_scheduler(args, optimizer)
        
        best_metric = None
        best_epoch = None
        
        for epoch in range(args.epochs):
            # 訓練
            train_metrics = train_epoch(epoch, model, loader_train, optimizer, train_loss_fn, args, args.device)
            
            # 驗證
            eval_metrics = validate(model, loader_eval, validate_loss_fn, args, args.device)
            
            # 更新學習率
            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1, eval_metrics['top1'])
            
            # 保存最佳模型
            if best_metric is None or eval_metrics['top1'] > best_metric:
                best_metric = eval_metrics['top1']
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_metric': best_metric,
                    'args': args,
                }, os.path.join(output_dir, 'model_best.pth'))
            
            _logger.info(f'Epoch {epoch+1}/{args.epochs}: Train Loss: {train_metrics["loss"]:.4f}, '
                        f'Val Acc@1: {eval_metrics["top1"]:.4f}, Best: {best_metric:.4f}')
        
        _logger.info(f"Full training completed! Best accuracy: {best_metric:.4f} at epoch {best_epoch+1}")
    
    # 只評估 unpruned
    if args.eval_unpruned:
        _logger.info('Evaluating unpruned model...')
        eval_metrics = validate(model, loader_eval, validate_loss_fn, args, args.device)
        _logger.info(f'Unpruned model accuracy: {eval_metrics["top1"]:.4f}')
        return
    
    # pruning 階段
    if args.load_pruned:
        _logger.info(f'Loading pruned model from {args.load_pruned}')
        pruned_model = create_model(
            'spikformer',
            pretrained=False,
            drop_rate=0.,
            drop_path_rate=0.,
            drop_block_rate=None,
            img_size_h=args.img_size, img_size_w=args.img_size,
            patch_size=args.patch_size, embed_dims=args.dim, num_heads=args.num_heads, mlp_ratios=args.mlp_ratio,
            in_channels=3, num_classes=args.num_classes, qkv_bias=False,
            depths=args.layer, sr_ratios=1,
            T=args.time_step
        ).cuda()
        checkpoint = torch.load(args.load_pruned, map_location=args.device)
        pruned_model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
        reduction_ratio = checkpoint.get('reduction_ratio', None)
        _logger.info('Pruned model loaded!')
        do_prune = False
    else:
        do_prune = args.do_prune
        pruned_model = None
    
    if do_prune:
        pruned_model, reduction_ratio = post_training_pruning(model, loader_train, args, args.device)
        torch.save({
            'model': pruned_model.state_dict(),
            'reduction_ratio': reduction_ratio,
            'pruning_ratio': args.pruning_ratio,
            'args': args,
        }, os.path.join(output_dir, 'model_pruned.pth'))
    
    # 只評估 pruned
    if args.eval_pruned:
        _logger.info('Evaluating pruned model...')
        eval_metrics = validate(pruned_model, loader_eval, validate_loss_fn, args, args.device)
        _logger.info(f'Pruned model accuracy: {eval_metrics["top1"]:.4f}')
        return
    
    # finetune 階段
    do_finetune = args.do_finetune
    if do_finetune:
        _logger.info("=" * 50)
        _logger.info("STAGE 3: Finetune")
        _logger.info("=" * 50)
        
        # 設置 finetune 參數
        args.lr = args.finetune_lr
        args.epochs = args.finetune_epochs
        
        optimizer = create_optimizer_v2(pruned_model, **optimizer_kwargs(cfg=args))
        lr_scheduler, _ = create_scheduler(args, optimizer)
        
        best_finetune_metric = None
        best_finetune_epoch = None
        
        for epoch in range(args.finetune_epochs):
            # 訓練
            train_metrics = train_epoch(epoch, pruned_model, loader_train, optimizer, train_loss_fn, args, args.device)
            
            # 驗證
            eval_metrics = validate(pruned_model, loader_eval, validate_loss_fn, args, args.device)
            
            # 更新學習率
            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1, eval_metrics['top1'])
            
            # 保存最佳模型
            if best_finetune_metric is None or eval_metrics['top1'] > best_finetune_metric:
                best_finetune_metric = eval_metrics['top1']
                best_finetune_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model': pruned_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_metric': best_finetune_metric,
                    'reduction_ratio': reduction_ratio,
                    'args': args,
                }, os.path.join(output_dir, 'model_finetuned.pth'))
            
            _logger.info(f'Finetune Epoch {epoch+1}/{args.finetune_epochs}: Train Loss: {train_metrics["loss"]:.4f}, '
                        f'Val Acc@1: {eval_metrics["top1"]:.4f}, Best: {best_finetune_metric:.4f}')
        
        # 最終結果總結
        _logger.info("=" * 50)
        _logger.info("FINAL RESULTS")
        _logger.info("=" * 50)
        _logger.info(f"Original model best accuracy: {best_metric:.4f}")
        _logger.info(f"Pruned model best accuracy: {best_finetune_metric:.4f}")
        _logger.info(f"Parameter reduction: {reduction_ratio:.2%}")
        _logger.info(f"Accuracy change: {best_finetune_metric - best_metric:+.4f}")
        _logger.info(f"All models saved in: {output_dir}")
    
    # 只評估 finetuned
    if args.eval_finetuned:
        _logger.info('Evaluating finetuned model...')
        finetuned_model = create_model(
            'spikformer',
            pretrained=False,
            drop_rate=0.,
            drop_path_rate=0.,
            drop_block_rate=None,
            img_size_h=args.img_size, img_size_w=args.img_size,
            patch_size=args.patch_size, embed_dims=args.dim, num_heads=args.num_heads, mlp_ratios=args.mlp_ratio,
            in_channels=3, num_classes=args.num_classes, qkv_bias=False,
            depths=args.layer, sr_ratios=1,
            T=args.time_step
        ).cuda()
        checkpoint = torch.load(os.path.join(output_dir, 'model_finetuned.pth'), map_location=args.device)
        finetuned_model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
        eval_metrics = validate(finetuned_model, loader_eval, validate_loss_fn, args, args.device)
        _logger.info(f'Finetuned model accuracy: {eval_metrics["top1"]:.4f}')
        return


if __name__ == '__main__':
    main() 