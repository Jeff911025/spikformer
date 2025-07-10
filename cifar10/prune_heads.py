import torch


def prune_spikformer_heads(model, ratio=0.5):
    """
    Prune attention heads in a Spikformer model by using weight-norm as an importance proxy.

    Args:
        model: the Spikformer instance (with attribute `block`, a list of Blocks containing an `attn` SSA module)
        ratio: fraction of heads to KEEP in each attention layer (0 < ratio <= 1)

    Returns:
        keep_map: dict mapping block index -> list of kept head indices
    """
    keep_map = {}
    # Iterate through each transformer block
    for blk_idx, blk in enumerate(model.block):
        attn = blk.attn
        num_heads = attn.num_heads
        dim = attn.dim
        head_dim = dim // num_heads

        # Compute importance score for each head via L2-norm of its output projection weights
        W = attn.proj_linear.weight.data  # shape: (dim, dim)
        scores = torch.empty(num_heads, device=W.device)
        for h in range(num_heads):
            start = h * head_dim
            end = (h + 1) * head_dim
            w_h = W[start:end, :]  # rows corresponding to head h
            scores[h] = w_h.norm()

        # Select top-k heads to KEEP
        k = max(1, int(num_heads * ratio))
        topk = torch.topk(scores, k)
        keep = topk.indices.cpu().tolist()
        keep_map[blk_idx] = keep

        # Zero out pruned heads in Q, K, V and projection
        mask = torch.zeros(num_heads, dtype=torch.bool, device=W.device)
        mask[keep] = True

        # Zero Q/K/V weight slices for pruned heads
        for lin in (attn.q_linear, attn.k_linear, attn.v_linear):
            w = lin.weight.data
            for h in range(num_heads):
                if not mask[h]:
                    w[:, h*head_dim:(h+1)*head_dim] = 0

        # Zero output projection slices for pruned heads
        for h in range(num_heads):
            if not mask[h]:
                attn.proj_linear.weight.data[h*head_dim:(h+1)*head_dim, :] = 0

        print(f"  ↓ Block {blk_idx}: kept heads {keep}")

    return keep_map


def _prune_attention_module(attn, keep_heads):
    """
    把给定 attention 模块里不在 keep_heads 中的 head 对应的 Q/K/V 以及输出 projection 全部置 0
    """
    num_heads = attn.num_heads
    dim = attn.dim
    head_dim = dim // num_heads
    device = attn.proj_linear.weight.device

    mask = torch.zeros(num_heads, dtype=torch.bool, device=device)
    mask[keep_heads] = True

    # Q, K, V linear: weight shape [dim_out, dim_in] = [dim, dim]
    for lin in (attn.q_linear, attn.k_linear, attn.v_linear):
        w = lin.weight.data
        # Q/K/V 按 head 拆分在 feature 维上
        for h in range(num_heads):
            if not mask[h]:
                # 把该 head 对应的所有列清零
                w[:, h * head_dim: (h + 1) * head_dim].zero_()

    # 输出投影：weight shape [dim, dim]
    w_proj = attn.proj_linear.weight.data
    # 输出头是把每个 head 的输出按行堆起来
    for h in range(num_heads):
        if not mask[h]:
            w_proj[h * head_dim: (h + 1) * head_dim, :].zero_()


def apply_keep_map_to_spikformer(model, keep_map):
    """
    在 eval/load 完 checkpoint 之后调用，把 state_dict 中的 keep_map 重新作用到模型上：
      - 遍历每个 block idx
      - 把不在 keep_map[blk_idx] 中的 head 全部 zero_out
    """
    for blk_idx, keep in keep_map.items():
        attn = model.block[blk_idx].attn
        _prune_attention_module(attn, keep)
