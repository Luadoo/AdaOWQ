# recon_soark_ena.py
# AdaOWQ v3 Engine: In-place Fake Quantization for OWQ Compatibility
import math
import time
import torch
import torch.nn as nn
import transformers

from .quant import *

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ_OWQ:
    def __init__(self, layer, n_out, expansion_factor=1.5, tier1_ratio=0.3,
                 tier2_bits=8, sym=False):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()

        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()

        self.rows = W.shape[0]
        self.columns = W.shape[1]

        # === Hessian 矩阵累加 ===
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

        # === OWQ 自适应保护列 ===
        # 保留自适应 allocate 后的 n_out，完全兼容 OWQ 原生维度逻辑
        self.n_out = n_out
        self.n_out_total = min(int(n_out * expansion_factor), self.columns - 1)
        self.n_nonout = self.columns - self.n_out_total
        self.owq = self.n_out_total > 0

        self.tier1_ratio = tier1_ratio
        self.tier2_bits = tier2_bits
        self.sym = sym
        self.tier1_ids = torch.tensor([], dtype=torch.long, device=self.dev)
        self.tier2_ids = torch.tensor([], dtype=torch.long, device=self.dev)

        self.ids = None
        self.quantizer = None

    def _prep_inp(self, inp):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        return inp

    def add_batch(self, inp, out):
        inp = self._prep_inp(inp)
        tmp = inp.shape[1]
        self.nsamples += tmp

        scale = math.sqrt(2 / self.nsamples)
        X = (scale * inp.float())
        H_new = X.matmul(X.t())

        self.H *= (self.nsamples - tmp) / self.nsamples
        self.H += H_new

    def _get_effective_H(self):
        if self.H is None:
            raise RuntimeError("Hessian is None")
        return 0.5 * (self.H + self.H.t())

    def hessian_sorting(self, actorder=False, frob_norm=None, tier1_ratio=None):
        H = self._get_effective_H()

        if self.n_out_total == 0 or not self.owq:
            self.ids = torch.arange(self.columns, device=self.dev)
            self.tier1_ids = torch.tensor([], dtype=torch.long, device=self.dev)
            self.tier2_ids = torch.tensor([], dtype=torch.long, device=self.dev)
            return torch.tensor([], dtype=torch.int32, device=self.dev)

        if tier1_ratio is not None:
            self.tier1_ratio = tier1_ratio

        w = frob_norm if frob_norm is not None else 1.0
        scores = torch.diag(H) * w
        out_ids = torch.topk(scores, self.n_out_total).indices

        temp_mask = torch.full([self.columns], True, device=self.dev)
        temp_mask[out_ids] = False
        if actorder:
            descending_ids = torch.argsort(scores, descending=True)
            self.ids = torch.cat([descending_ids[self.n_out_total:], descending_ids[:self.n_out_total]])
        else:
            self.ids = torch.cat([torch.arange(self.columns, device=self.dev)[temp_mask], out_ids])

        # Tier1 与 Tier2 的划定
        n_tier1 = max(1, round(self.n_out_total * self.tier1_ratio))
        n_tier1 = min(n_tier1, len(out_ids))
        self.tier1_ids = out_ids[:n_tier1]
        self.tier2_ids = out_ids[n_tier1:]

        # 返回与 OWQ 完全一致的 out_ids，保证维度匹配！
        return torch.sort(out_ids)[0].to(torch.int32)

    def fasterquant(self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if actorder or self.owq:
            W = W[:, self.ids]
            self.H = self.H[self.ids][:, self.ids]

        self.quantizer.find_params(W[:, :self.n_nonout], weight=True)

        H = self._get_effective_H()
        self.H = None

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp

        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.n_nonout, blocksize):
            i2 = min(i1 + blocksize, self.n_nonout)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(
                            W[:, (i1 + i):min((i1 + i + groupsize), (self.columns - self.n_out_total))],
                            weight=True, num=40
                        )

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        # 保护列先全部保留 FP16 原始精度
        if actorder or self.owq:
            Q[:, self.n_nonout:] = W[:, self.n_nonout:]
            invids = torch.argsort(self.ids)
            Q = Q[:, invids]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()

        # 写回全量 FP16 格式权重的矩阵，保持维度一致
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        # ==================== Tier 2: 原位伪量化 (In-place Fake Quantization) ====================
        # 核心改进：不改变 tensor shape，只对 Tier2 列应用 INT8 伪量化限制，无缝兼容后续 eval/save !
        if hasattr(self, 'tier2_ids') and len(self.tier2_ids) > 0 and self.tier2_bits < 16:
            W_tier2 = self.layer.weight.data[:, self.tier2_ids].clone().float()
            tier2_quant = Quantizer(
                self.tier2_bits, perchannel=True, sym=self.sym, mse=True
            )
            tier2_quant.find_params(W_tier2, weight=True, num=40)
            W_tier2_q = tier2_quant.quantize(W_tier2)

            # 写回伪量化后的数值
            self.layer.weight.data[:, self.tier2_ids] = W_tier2_q.to(self.layer.weight.data.dtype)

            tier2_rel_err = torch.norm(W_tier2 - W_tier2_q).item() / (torch.norm(W_tier2).item() + 1e-8)
            print(f"    [Tier2 In-place Quant] {len(self.tier2_ids)} cols -> {self.tier2_bits}-bit Fake Quant, rel_err={tier2_rel_err:.6f}")
        
        print(f"    [Summary] Outlier total: {self.n_out_total} (Tier1 FP16: {len(self.tier1_ids)}, Tier2 INT8: {len(self.tier2_ids)}) | Base INT3: {self.n_nonout}")

    def free(self):
        self.H = None
        self.ids = None
        torch.cuda.empty_cache()