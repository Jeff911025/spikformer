# prune_heads_spike.py
import torch
from collections import defaultdict
from timm.models.vision_transformer import Attention  # 或者替换成你的 SSA 类

def compute_pruned_heads_by_spike(model, dataloader, keep_ratio=0.5, device='cuda'):
    """
    对 model 中每个 Attention 模块（多头注意力），
    用一个 batch 的输入跑一次 forward，hook 下 attn 输出的 spike map，
    按时间轴（T）求和后对每个 head 做 SVD，取奇异值之和作为该 head 的分数，
    然后根据 keep_ratio 保留得分最高的若干 head，返回 pruned_heads_map.

    Args:
        model:            spikformer 实例
        dataloader:       DataLoader，只会从里头拿一个 batch
        keep_ratio:       保留 head 的比例，例如 0.5
        device:           cuda 设备

    Returns:
        pruned_heads_map: dict, key=block index，value=set(要保留的 head idx)
    """
    # 存储每个 Attention 模块的每个 head 分数
    scores = defaultdict(lambda: defaultdict(float))
    handles = []

    def hook_attn(spike_out, module, _input, _output):
        # spike_out：MultiStepLIFNode 的输出，[T, B, N, C]
        spike = _output.detach().cpu()  # shape (T, B, N, C)
        T, B, N, C = spike.shape
        H = module.num_heads
        C_h = C // H

        # 按时间求和，shape -> (B, N, C)
        m = spike.sum(dim=0)  # ([B,N,C])

        # 对每个 head 做 SVD 评分
        for h in range(H):
            # 取出这个 head 对应的通道
            block = m[..., h*C_h:(h+1)*C_h]  # ([B,N,C_h])
            mat = block.reshape(B*N, C_h)   # ([B*N, C_h])
            # SVD
            try:
                U, S, Vt = torch.linalg.svd(mat, full_matrices=False)
                scores[module][h] += S.sum().item()
            except Exception:
                # 如果 mat 全零或奇异，就退而求其次用 L1
                scores[module][h] += mat.abs().sum().item()

    # 给所有 Attention（或 SSA）模块的 LIF 输出打 hook
    for module in model.modules():
        if isinstance(module, Attention) or module.__class__.__name__ == 'SSA':
            # hook 在 attn_lif 上，output 是 spike map
            handles.append(
                module.attn_lif.register_forward_hook(
                    lambda m,i,o, mod=module: hook_attn(o, mod, i, o)
                )
            )

    # 只跑一个 batch 来收集 spike data
    model.eval()
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            model(x)
            break

    # 卸载 hook
    for h in handles:
        h.remove()

    # 根据每个模块的 head 分数选 top-k
    pruned_heads_map = {}
    for module, head_scores in scores.items():
        H = module.num_heads
        k = max(1, int(H * keep_ratio))
        # 按分数排序，保留前 k
        best = sorted(head_scores.items(), key=lambda kv: -kv[1])[:k]
        keep = {h for h, _ in best}
        # 下面根据 module 在 model 中的位置给它一个索引，你可以改成对应的 block id
        layer_idx = None
        for name, mod in model.named_modules():
            if mod is module:
                # 假设你的 block 名叫 block.3.attn
                layer_idx = int(name.split('.')[1])
                break
        pruned_heads_map[layer_idx] = keep

    return pruned_heads_map
