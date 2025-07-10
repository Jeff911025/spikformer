# model_svd.py
import torch
import torch.nn as nn

class SVDLinear(nn.Module):
    def __init__(self, orig_linear: nn.Linear, rank_ratio: float):
        super().__init__()
        W = orig_linear.weight.data
        m, n = W.shape
        k = min(m, n)                       # SVD 的最大可用秩
        r = max(1, int(k * rank_ratio))    # 保留 k * ratio 条

        # 完整 SVD
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        U_r  = U[:, :r]            # (m, r)
        S_r  = torch.diag(S[:r])   # (r, r)
        Vt_r = Vt[:r, :]           # (r, n)

        # 线性拆成两层： first Vt_r，再 U_r @ S_r
        # 第一层：n -> r
        self.linear1 = nn.Linear(n, r, bias=False)
        self.linear1.weight.data.copy_(Vt_r)

        # 第二层：r -> m
        self.linear2 = nn.Linear(r, m, bias=(orig_linear.bias is not None))
        self.linear2.weight.data.copy_(U_r @ S_r)
        if orig_linear.bias is not None:
            self.linear2.bias.data.copy_(orig_linear.bias.data)

    def forward(self, x):
        return self.linear2(self.linear1(x))


# model_svd.py (继续)
def apply_svd_to_model(model: nn.Module, rank_ratio: float, verbose: bool = True):
    for name, module in list(model.named_children()):
        # 如果是 Linear，替换
        if isinstance(module, nn.Linear):
            if verbose:
                print(f"  ↓ Replace {name}: {module.in_features}->{module.out_features}")
            svd_layer = SVDLinear(module, rank_ratio)
            setattr(model, name, svd_layer)
        else:
            # 递归子模块
            apply_svd_to_model(module, rank_ratio, verbose)
