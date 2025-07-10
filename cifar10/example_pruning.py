#!/usr/bin/env python3
"""
Spikformer Pruning 示例腳本

這個腳本展示了如何在 Spikformer 訓練中使用基於 spike map 的 pruning 機制。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from pruning_utils import SpikeMapPruner, create_pruning_scheduler
import model
import argparse


def create_cifar10_dataloader(batch_size=32, num_workers=4):
    """創建 CIFAR-10 數據加載器"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='/root/data/nas07/PersonalData/Jeff0102030433/CIFAR10/', train=True, download=False, transform=transform_train)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    
    testset = torchvision.datasets.CIFAR10(
        root='/root/data/nas07/PersonalData/Jeff0102030433/CIFAR10/', train=False, download=False, transform=transform_test)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    
    return trainloader, testloader


def evaluate_model(model, testloader, device):
    """評估模型準確率"""
    model.eval()
    correct = 0
    total = 0
    
    # 在評估開始前重置神經元狀態
    from spikingjelly.clock_driven import functional
    functional.reset_net(model)
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 每個 batch 前也重置神經元狀態
            functional.reset_net(model)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    return correct / total


def main():
    parser = argparse.ArgumentParser(description='Spikformer Pruning Example')
    parser.add_argument('--pruning-ratio', type=float, default=0.3, 
                       help='Pruning ratio (default: 0.3)')
    parser.add_argument('--epochs', type=int, default=10, 
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.01, 
                       help='Learning rate (default: 0.01)')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 創建模型
    model_instance = model.spikformer(
        img_size_h=32, img_size_w=32, patch_size=4, in_channels=3, 
        num_classes=10, embed_dims=256, num_heads=4, mlp_ratios=4, 
        depths=4, sr_ratios=1, T=4
    )
    model_instance = model_instance.to(device)
    
    # 創建數據加載器
    trainloader, testloader = create_cifar10_dataloader(args.batch_size)
    
    # 創建 pruning 工具
    pruner = SpikeMapPruner(model_instance, pruning_ratio=args.pruning_ratio)
    pruning_scheduler = create_pruning_scheduler(
        initial_ratio=0.1, 
        final_ratio=args.pruning_ratio, 
        epochs=args.epochs
    )
    
    # 創建優化器和損失函數
    optimizer = torch.optim.SGD(model_instance.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    print(f'Initial model parameters: {sum(p.numel() for p in model_instance.parameters())}')
    
    # 訓練循環
    for epoch in range(args.epochs):
        model_instance.train()
        running_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model_instance(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # 重置神經元狀態
            from spikingjelly.clock_driven import functional
            functional.reset_net(model_instance)
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # 評估當前模型（只在特定 epoch 進行評估以節省時間）
        if epoch % 2 == 0 or epoch == 0:  # 每2個epoch評估一次，或第一個epoch
            accuracy = evaluate_model(model_instance, testloader, device)
            print(f'Epoch {epoch+1}, Accuracy: {accuracy:.4f}, Loss: {running_loss/len(trainloader):.4f}')
        else:
            print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}')
        
        # 應用 pruning（在特定 epoch）
        if epoch >= 3 and epoch % 2 == 0:  # 從第3個epoch開始，每2個epoch進行一次pruning
            print(f'Applying pruning at epoch {epoch+1}')
            
            # 獲取當前 pruning 比例
            current_pruning_ratio = pruning_scheduler(epoch)
            pruner.pruning_ratio = current_pruning_ratio
            
            # 獲取樣本輸入並重置神經元狀態
            sample_batch = next(iter(trainloader))
            sample_input = sample_batch[0].to(device)
            
            # 重置神經元狀態
            from spikingjelly.clock_driven import functional
            functional.reset_net(model_instance)
            
            # 計算 channel 分數
            scores = pruner.compute_channel_scores(sample_input)
            print(f'Computed scores for {len(scores)} modules')
            
            # 選擇要 pruning 的 channels
            channels_to_prune = pruner.select_channels_to_prune(scores)
            
            # 應用 pruning
            pruned_model = pruner.apply_pruning(channels_to_prune)
            pruned_model = pruned_model.to(device)
            
            # 替換模型
            model_instance = pruned_model
            
            # 重新創建優化器
            optimizer = torch.optim.SGD(model_instance.parameters(), lr=args.lr, momentum=0.9)
            
            print(f'Pruning applied with ratio {current_pruning_ratio:.3f}')
            print(f'Pruned model parameters: {sum(p.numel() for p in model_instance.parameters())}')
            
            # 評估 pruning 後的模型
            pruned_accuracy = evaluate_model(model_instance, testloader, device)
            print(f'Pruned model accuracy: {pruned_accuracy:.4f}')
    
    print('Training completed!')
    
    # 最終評估
    final_accuracy = evaluate_model(model_instance, testloader, device)
    print(f'Final model accuracy: {final_accuracy:.4f}')
    print(f'Final model parameters: {sum(p.numel() for p in model_instance.parameters())}')


if __name__ == '__main__':
    main() 