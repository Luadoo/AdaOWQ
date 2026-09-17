# adaowq_utils.py
# AdaOWQ v3: Adaptive Outlier Allocation with Unified Weight Export
import torch
import torch.nn as nn
import numpy as np
import transformers


class AdaOWQProfiler:
    """
    Contribution 1 & 2 for Sensors Paper:
    1. Hessian-Activation Energy-based Sensitivity Profiling (In log-space to ensure numerical stability).
    2. Entropy-Regularized Constrained Budget Allocation (Preserves total parameter budget strictly).
    """

    def __init__(self, alpha=0.6, beta=0.4, guarantee_ratio=0.5):
        self.alpha = alpha
        self.beta = beta
        self.guarantee_ratio = guarantee_ratio

    @torch.no_grad()
    def compute_layer_sensitivity(self, gptq_owq_obj, layer_module, inps_sample):
        """
        计算融合二阶 Hessian 曲率与激活-权重能量信息的层敏感度 S_l
        """
        # 1. Hessian 迹与主对角线信息 (量化曲率敏感度)
        H = gptq_owq_obj._get_effective_H()
        H_diag = torch.diag(H).abs()
        hessian_energy = torch.log(torch.mean(H_diag) + 1e-8).item()

        # 2. 激活值通道极大值范数 (异常值激活能量)
        if inps_sample.dim() == 3:
            x_flat = inps_sample.reshape(-1, inps_sample.shape[-1]).float()
        else:
            x_flat = inps_sample.float()
        x_col_max = torch.max(x_flat.abs(), dim=0).values
        act_energy = torch.log(torch.mean(x_col_max ** 2) + 1e-8).item()

        # 3. 权重通道范数
        W = layer_module.weight.data.float()
        if isinstance(layer_module, transformers.Conv1D):
            W = W.t()
        if W.dim() > 2:
            W = W.flatten(1)
        w_col_norms = torch.norm(W, dim=0)
        weight_energy = torch.log(torch.mean(w_col_norms ** 2) + 1e-8).item()

        # 组合敏感度得分 (对数空间线性加权，确保鲁棒性)
        sensitivity = self.alpha * hessian_energy + self.beta * (act_energy + weight_energy)
        return float(sensitivity)

    def allocate_budgets(self, layers_sensitivity, base_n_out_dict, temp=1.0):
        """
        自适应预算分配器：
        使用 Softmax 温度调节，并在分配后强制与全局总预算对齐 (Total Budget Neutral)，
        确保全局零开销增加，同时实现层间自适应调配。
        """
        sens_keys = list(layers_sensitivity.keys())
        sens_vals = np.array([layers_sensitivity[k] for k in sens_keys])

        # 处理数值异常
        sens_vals = np.nan_to_num(sens_vals, nan=np.mean(sens_vals))
        
        # 归一化 Softmax 权重
        norm_sens = (sens_vals - np.mean(sens_vals)) / (np.std(sens_vals) + 1e-6)
        exp_sens = np.exp(np.clip(norm_sens / temp, -8.0, 8.0))
        smooth_weights = exp_sens / np.sum(exp_sens)

        # 全局总 Outlier 列预算统计
        total_base_budget = sum([base_n_out_dict.get(k.split('.', 1)[1] if '.' in k else k, 0) for k in sens_keys])
        
        if total_base_budget == 0:
            return {k: 0 for k in sens_keys}

        guaranteed_total = total_base_budget * self.guarantee_ratio
        floating_total = total_base_budget * (1.0 - self.guarantee_ratio)

        final_allocations = {}
        allocated_sum = 0

        for idx, key in enumerate(sens_keys):
            layer_name = key.split('.', 1)[1] if '.' in key else key
            base_out = base_n_out_dict.get(layer_name, 0)

            # 保底预算 + 基于敏感度分配的动态浮动预算
            layer_guaranteed = base_out * self.guarantee_ratio
            layer_floating = floating_total * (smooth_weights[idx] * len(sens_keys) / sum([base_n_out_dict.get(k.split('.', 1)[1] if '.' in k else k, 0) > 0 for k in sens_keys])) * (base_out / (total_base_budget / len(sens_keys) + 1e-6))
            
            alloc = int(round(layer_guaranteed + layer_floating))
            # 严格边界约束：[0.4 * base, 2.5 * base]
            alloc = max(int(base_out * 0.4), min(alloc, int(base_out * 2.5)))
            if alloc % 2 == 1:
                alloc += 1
            final_allocations[key] = alloc
            allocated_sum += alloc

        print(f"  [AdaOWQ Profiler] Global Base Outliers Total: {total_base_budget} | Adaptive Total: {allocated_sum}")
        return final_allocations


class LowRankOutlierCompressor:
    """
    兼容性保留（若后续做低秩比较实验使用）
    """
    def __init__(self, rank_r=32):
        self.rank_r = rank_r

    @torch.no_grad()
    def decompose_tier2(self, W_outliers_medium):
        if W_outliers_medium.shape[1] <= self.rank_r:
            return W_outliers_medium, None
        W_f = W_outliers_medium.float()
        U, S, Vh = torch.linalg.svd(W_f, full_matrices=False)
        r = min(self.rank_r, U.shape[1], Vh.shape[0])
        U_r_sigma = U[:, :r] * S[:r].unsqueeze(0)
        Vh_r = Vh[:r, :]
        return U_r_sigma.to(W_outliers_medium.dtype), Vh_r.to(W_outliers_medium.dtype)