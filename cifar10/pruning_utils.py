import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import copy
from model import SSA, MLP


class SpikeMapPruner:
    """
    基於 spike map 分數的 pruning 工具類
    """
    
    def __init__(self, model: nn.Module, pruning_ratio: float = 0.3):
        """
        Args:
            model: 要進行 pruning 的模型
            pruning_ratio: pruning 比例 (0-1)
        """
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.original_state = None
        self.spike_maps = {}
        self.channel_scores = {}
        
    def register_hooks(self):
        """註冊 hooks 來捕獲 spike maps"""
        self.spike_maps.clear()
        
        def get_spike_hook(name):
            def hook(module, input, output):
                # 捕獲 spike map (T, B, N, C)
                if isinstance(output, torch.Tensor) and len(output.shape) == 4:
                    # 壓縮 T 維度，計算每個 channel 的活躍度
                    spike_map = output.detach()
                    # 計算每個 channel 的平均 spike 頻率
                    channel_activity = torch.mean(spike_map, dim=(0, 1, 2))  # (C,)
                    self.spike_maps[name] = channel_activity
            return hook
        
        # 為 SSA 模組註冊 hooks
        for name, module in self.model.named_modules():
            if isinstance(module, SSA):
                module.register_forward_hook(get_spike_hook(f"ssa_{name}"))
            elif isinstance(module, MLP):
                module.register_forward_hook(get_spike_hook(f"mlp_{name}"))
    
    def compute_channel_scores(self, sample_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        計算每個 channel 的分數
        
        Args:
            sample_input: 樣本輸入 (B, C, H, W)
            
        Returns:
            Dict[str, torch.Tensor]: 每個模組的 channel 分數
        """
        self.register_hooks()
        
        # 重置神經元狀態
        from spikingjelly.clock_driven import functional
        functional.reset_net(self.model)
        
        # 前向傳播以捕獲 spike maps
        with torch.no_grad():
            self.model(sample_input)
        
        # 計算分數
        scores = {}
        for name, spike_map in self.spike_maps.items():
            # 使用 spike 頻率作為分數
            # 也可以使用其他指標，如變異數、熵等
            scores[name] = spike_map
            
        self.channel_scores = scores
        return scores
    
    def select_channels_to_prune(self, scores: Dict[str, torch.Tensor]) -> Dict[str, List[int]]:
        """
        根據分數選擇要 pruning 的 channels
        
        Args:
            scores: 每個模組的 channel 分數
            
        Returns:
            Dict[str, List[int]]: 每個模組要 pruning 的 channel 索引
        """
        channels_to_prune = {}
        
        for name, score in scores.items():
            num_channels = len(score)
            num_to_prune = int(num_channels * self.pruning_ratio)
            
            # 選擇分數最低的 channels 進行 pruning
            _, indices = torch.sort(score)
            prune_indices = indices[:num_to_prune].tolist()
            channels_to_prune[name] = prune_indices
            
        return channels_to_prune
    
    def apply_pruning(self, channels_to_prune: Dict[str, List[int]]) -> nn.Module:
        """
        應用 pruning 到模型
        
        Args:
            channels_to_prune: 要 pruning 的 channel 索引
            
        Returns:
            nn.Module: pruning 後的模型
        """
        pruned_model = copy.deepcopy(self.model)
        
        for name, module in pruned_model.named_modules():
            if isinstance(module, SSA):
                # Prune SSA 模組
                self._prune_ssa_module(module, channels_to_prune, name)
            elif isinstance(module, MLP):
                # Prune MLP 模組
                self._prune_mlp_module(module, channels_to_prune, name)
        
        return pruned_model
    
    def _prune_ssa_module(self, module: SSA, channels_to_prune: Dict[str, List[int]], module_name: str):
        """Prune SSA 模組的特定 channels (masking baseline)"""
        prune_key = f"ssa_{module_name}"
        if prune_key not in channels_to_prune:
            return
        prune_indices = channels_to_prune[prune_key]
        if not prune_indices:
            return
        dim = module.dim
        keep_indices = [i for i in range(dim) if i not in prune_indices]
        if hasattr(module, 'q_linear'):
            module.q_linear = self._prune_linear_layer(module.q_linear, keep_indices, dim)
        if hasattr(module, 'k_linear'):
            module.k_linear = self._prune_linear_layer(module.k_linear, keep_indices, dim)
        if hasattr(module, 'v_linear'):
            module.v_linear = self._prune_linear_layer(module.v_linear, keep_indices, dim)
        if hasattr(module, 'proj_linear'):
            module.proj_linear = self._prune_linear_layer(module.proj_linear, keep_indices, dim)
        # masking baseline 不要動結構參數
        # module.dim = len(keep_indices)
        # module.num_heads = min(module.num_heads, module.dim // (module.dim // module.num_heads))

    def _prune_mlp_module(self, module: MLP, channels_to_prune: Dict[str, List[int]], module_name: str):
        """Prune MLP 模組的特定 channels (masking baseline)"""
        prune_key = f"mlp_{module_name}"
        if prune_key not in channels_to_prune:
            return
        prune_indices = channels_to_prune[prune_key]
        if not prune_indices:
            return
        in_features = module.c_output
        keep_indices = [i for i in range(in_features) if i not in prune_indices]
        if hasattr(module, 'fc1_linear'):
            module.fc1_linear = self._prune_linear_layer(module.fc1_linear, keep_indices, in_features)
        if hasattr(module, 'fc2_linear'):
            module.fc2_linear = self._prune_linear_layer(module.fc2_linear, keep_indices, in_features)
        # masking baseline 不要動結構參數
        # module.c_hidden = len(keep_indices)
        # module.c_output = len(keep_indices)
    
    def _prune_linear_layer(self, linear_layer: nn.Linear, keep_indices: List[int], original_dim: int) -> nn.Linear:
        # Masking: 將要砍掉的 channel 權重設為 0
        device = linear_layer.weight.device
        mask = torch.ones(original_dim, dtype=torch.bool, device=device)
        mask[keep_indices] = False  # mask=True 代表要砍掉
        with torch.no_grad():
            # 如果是 input features pruning
            if linear_layer.in_features == original_dim:
                linear_layer.weight.data[:, mask] = 0
            # 如果是 output features pruning
            elif linear_layer.out_features == original_dim:
                linear_layer.weight.data[mask, :] = 0
                if linear_layer.bias is not None:
                    linear_layer.bias.data[mask] = 0
        return linear_layer
    
    def evaluate_pruning_impact(self, original_model: nn.Module, pruned_model: nn.Module, 
                               test_loader, device: torch.device) -> Dict[str, float]:
        """
        評估 pruning 對模型性能的影響
        
        Args:
            original_model: 原始模型
            pruned_model: pruning 後的模型
            test_loader: 測試數據加載器
            device: 設備
            
        Returns:
            Dict[str, float]: 性能指標
        """
        original_model.eval()
        pruned_model.eval()
        
        original_acc = self._evaluate_model(original_model, test_loader, device)
        pruned_acc = self._evaluate_model(pruned_model, test_loader, device)
        
        # 計算參數減少比例
        original_params = sum(p.numel() for p in original_model.parameters())
        pruned_params = sum(p.numel() for p in pruned_model.parameters())
        param_reduction = (original_params - pruned_params) / original_params
        
        return {
            'original_accuracy': original_acc,
            'pruned_accuracy': pruned_acc,
            'accuracy_drop': original_acc - pruned_acc,
            'parameter_reduction': param_reduction
        }
    
    def _evaluate_model(self, model: nn.Module, test_loader, device: torch.device) -> float:
        """評估單個模型的準確率"""
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        return correct / total

    def structural_prune_and_rebuild(self, keep_indices_dict: Dict[str, List[int]]) -> nn.Module:
        """
        根據 keep_indices_dict（每個 module 要保留的 channel 索引）重建一個新的模型
        只支援 Linear/MLP block 的結構化 pruning demo
        """
        # 深拷貝模型，避免污染原始模型
        new_model = copy.deepcopy(self.model)
        # 遞迴重建所有 MLP/SSA block
        for name, module in new_model.named_modules():
            # MLP block
            if isinstance(module, MLP):
                prune_key = f"mlp_{name}"
                if prune_key in keep_indices_dict:
                    keep_indices = keep_indices_dict[prune_key]
                    # 重建 fc1_linear
                    old_fc1 = module.fc1_linear
                    new_fc1 = nn.Linear(old_fc1.in_features, len(keep_indices), bias=old_fc1.bias is not None).to(old_fc1.weight.device)
                    with torch.no_grad():
                        new_fc1.weight.copy_(old_fc1.weight[keep_indices, :])
                        if old_fc1.bias is not None:
                            new_fc1.bias.copy_(old_fc1.bias[keep_indices])
                    module.fc1_linear = new_fc1
                    # 重建 fc1_bn（如果有）
                    if hasattr(module, 'fc1_bn') and module.fc1_bn is not None:
                        old_bn = module.fc1_bn
                        new_bn = nn.BatchNorm1d(len(keep_indices)).to(old_bn.weight.device)
                        with torch.no_grad():
                            new_bn.weight.copy_(old_bn.weight[keep_indices])
                            new_bn.bias.copy_(old_bn.bias[keep_indices])
                            new_bn.running_mean.copy_(old_bn.running_mean[keep_indices])
                            new_bn.running_var.copy_(old_bn.running_var[keep_indices])
                        module.fc1_bn = new_bn
                    # 重建 fc2_linear
                    old_fc2 = module.fc2_linear
                    new_fc2 = nn.Linear(len(keep_indices), old_fc2.out_features, bias=old_fc2.bias is not None).to(old_fc2.weight.device)
                    with torch.no_grad():
                        new_fc2.weight.copy_(old_fc2.weight[:, keep_indices])
                        if old_fc2.bias is not None:
                            new_fc2.bias.copy_(old_fc2.bias)
                    module.fc2_linear = new_fc2
                    # 重建 fc2_bn（如果有）
                    if hasattr(module, 'fc2_bn') and module.fc2_bn is not None:
                        old_bn2 = module.fc2_bn
                        new_bn2 = nn.BatchNorm1d(old_fc2.out_features).to(old_bn2.weight.device)
                        with torch.no_grad():
                            new_bn2.weight.copy_(old_bn2.weight)
                            new_bn2.bias.copy_(old_bn2.bias)
                            new_bn2.running_mean.copy_(old_bn2.running_mean)
                            new_bn2.running_var.copy_(old_bn2.running_var)
                        module.fc2_bn = new_bn2
                    # 更新 c_hidden
                    module.c_hidden = len(keep_indices)
            # SSA block（可依需求擴充）
            # ... 你可以仿照 MLP 實作 SSA ...
        return new_model


def create_pruning_scheduler(initial_ratio: float = 0.1, final_ratio: float = 0.5, 
                           epochs: int = 200) -> callable:
    """
    創建 pruning 比例調度器
    
    Args:
        initial_ratio: 初始 pruning 比例
        final_ratio: 最終 pruning 比例
        epochs: 總訓練輪數
        
    Returns:
        callable: 返回當前 epoch 的 pruning 比例
    """
    def get_pruning_ratio(epoch: int) -> float:
        if epoch < epochs * 0.3:  # 前 30% 的 epochs 不進行 pruning
            return 0.0
        elif epoch < epochs * 0.7:  # 30%-70% 的 epochs 逐漸增加 pruning
            progress = (epoch - epochs * 0.3) / (epochs * 0.4)
            return initial_ratio + (final_ratio - initial_ratio) * progress
        else:  # 70% 之後保持最終比例
            return final_ratio
    
    return get_pruning_ratio 