import math
from functools import partial
from typing import Tuple, List

import torch
from torch import nn
import torch.nn.functional as F

from src.jamma.utils.utils import GLU_3
from mamba_ssm import Mamba

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm, LayerNorm
except ImportError:
    RMSNorm, LayerNorm = None, None

from src.utils.profiler import PassThroughProfiler


class FrequencyEncoding2D(nn.Module):
    """
    Full-channel frequency enhancement before grouped Mamba.

    Input:
        x: [B, C, H, W]

    Steps:
        1) FFT
        2) low/high split
        3) inverse FFT
        4) refine high-frequency with lightweight conv
        5) fuse spatial + low + refined high
    """
    def __init__(self, dim, reduction=4, low_freq_ratio=0.25):
        super().__init__()
        self.dim = dim
        self.low_freq_ratio = low_freq_ratio

        hidden = max(dim // reduction, 8)

        self.low_freq_weight = nn.Parameter(torch.tensor(1.0))
        self.high_freq_weight = nn.Parameter(torch.tensor(1.0))

        self.low_proj = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=True),
        )

        self.high_proj = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=True),
        )

        # 高频卷积 refinement：轻量、局部、残差式
        self.high_refine = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 3, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=True),
        )

        self.output_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def _build_frequency_masks(self, x_freq: torch.Tensor):
        _, _, fh, fw = x_freq.shape
        low_h = max(1, int(fh * self.low_freq_ratio))
        low_w = max(1, int(fw * self.low_freq_ratio))

        low_mask = torch.zeros(
            (1, 1, fh, fw), device=x_freq.device, dtype=x_freq.real.dtype
        )
        low_mask[:, :, :low_h, :low_w] = 1.0
        high_mask = 1.0 - low_mask
        return low_mask, high_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape

        x_freq = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        low_mask, high_mask = self._build_frequency_masks(x_freq)

        low_freq = x_freq * low_mask
        high_freq = x_freq * high_mask

        low_spatial = torch.fft.irfft2(low_freq, s=(H, W), dim=(-2, -1), norm="ortho")
        high_spatial = torch.fft.irfft2(high_freq, s=(H, W), dim=(-2, -1), norm="ortho")

        low_spatial = self.low_proj(low_spatial)
        high_spatial = self.high_proj(high_spatial)

        # 高频再做卷积 refinement
        high_spatial = high_spatial + self.high_refine(high_spatial)

        alpha = torch.sigmoid(self.low_freq_weight)
        beta = torch.sigmoid(self.high_freq_weight)

        low_spatial = alpha * low_spatial
        high_spatial = beta * high_spatial

        fused = self.fusion(torch.cat([x, low_spatial, high_spatial], dim=1))
        gate = self.output_gate(x + low_spatial + high_spatial)

        out = x + gate * fused
        return out


class GroupedMambaLayer(nn.Module):
    """
    Grouped Mamba:
        Input  [B, L, C]
        Split along channel -> G groups
        Each group passes its own Mamba
        Concat back to [B, L, C]
    """
    def __init__(
        self,
        dim,
        mixer_cls,
        num_groups=4,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
    ):
        super().__init__()
        self.dim = dim
        self.num_groups = num_groups
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        self.norm = norm_cls(dim)

        self.group_dim = math.ceil(dim / num_groups)
        padded_dim = self.group_dim * num_groups
        self.padded_dim = padded_dim
        self.pad_dim = padded_dim - dim

        self.mixers = nn.ModuleList([mixer_cls(self.group_dim) for _ in range(num_groups)])

        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), \
                "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def _pad_channels(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        if self.pad_dim == 0:
            return x
        pad = x.new_zeros(x.shape[0], x.shape[1], self.pad_dim)
        return torch.cat([x, pad], dim=-1)

    def forward(self, desc, inference_params=None):
        """
        desc: [B, L, C]
        """
        hidden_states = self.norm(desc.to(dtype=self.norm.weight.dtype))
        residual = desc.to(torch.float32) if self.residual_in_fp32 else desc

        hidden_states = self._pad_channels(hidden_states)
        groups = torch.chunk(hidden_states, self.num_groups, dim=-1)

        out_groups = []
        for i, g in enumerate(groups):
            out_groups.append(self.mixers[i](g, inference_params=inference_params))

        out = torch.cat(out_groups, dim=-1)
        out = out[..., :self.dim]
        return residual + out

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return [
            mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for mixer in self.mixers
        ]


def create_grouped_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    num_groups=4,
):
    if ssm_cfg is None:
        ssm_cfg = {}

    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm,
        eps=norm_epsilon,
        **factory_kwargs
    )

    block = GroupedMambaLayer(
        d_model,
        mixer_cls=mixer_cls,
        num_groups=num_groups,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class JointMamba(nn.Module):
    """
    Final version:
        1) full-channel frequency enhancement
        2) split channels into groups
        3) grouped directional scan + grouped Mamba
        4) merge
        5) channel attention + aggregator
        6) residual add
    """
    def __init__(
        self,
        feature_dim: int,
        depth,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        reduction: int = 16,
        step_size: int = 2,
        num_groups: int = 4,
        freq_reduction: int = 4,
        low_freq_ratio: float = 0.25,
        profiler=None,
    ):
        super().__init__()
        self.profiler = profiler or PassThroughProfiler()
        self.feature_dim = feature_dim
        self.step_size = step_size
        self.num_groups = num_groups

        self.freq_encoder = FrequencyEncoding2D(
            dim=feature_dim,
            reduction=freq_reduction,
            low_freq_ratio=low_freq_ratio,
        )

        self.group_dim = math.ceil(feature_dim / num_groups)
        self.padded_dim = self.group_dim * num_groups
        self.pad_dim = self.padded_dim - feature_dim

        # 每个方向一个 grouped Mamba block；depth>4 时循环复用
        self.layers = nn.ModuleList()
        for i in range(max(depth, 4)):
            self.layers.append(
                create_grouped_block(
                    self.group_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    num_groups=1,  # 外层已经分组了，这里每个分组内部直接单组 Mamba
                )
            )

        reduced_dim = max(16, max(1, feature_dim // max(1, reduction)))
        self.ca = nn.Sequential(
            nn.Linear(feature_dim, reduced_dim, bias=True),
            nn.ReLU(),
            nn.Linear(reduced_dim, feature_dim, bias=True),
            nn.Sigmoid()
        )

        self.aggregator = GLU_3(feature_dim, feature_dim)

        self.apply(
            partial(
                _init_weights,
                n_layer=max(depth, 4),
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def _pad_channels_2d(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        if self.pad_dim == 0:
            return x
        pad = x.new_zeros(x.shape[0], self.pad_dim, x.shape[2], x.shape[3])
        return torch.cat([x, pad], dim=1)

    def _safe_chunk_2d(self, x: torch.Tensor, chunks: int, dim: int = 1) -> List[torch.Tensor]:
        x = self._pad_channels_2d(x)
        return list(torch.chunk(x, chunks, dim=dim))

    def forward(self, data):
        desc0, desc1 = data['feat_8_0'], data['feat_8_1']
        B_size = data['bs']
        H, W = data['h_8'], data['w_8']

        desc0 = desc0.view(B_size, -1, H, W)
        desc1 = desc1.view(B_size, -1, H, W)

        # 保留原始特征作 residual
        residual0 = desc0
        residual1 = desc1

        # 1) 先做全通道频域增强
        desc0 = self.freq_encoder(desc0)
        desc1 = self.freq_encoder(desc1)

        # 2) 再按通道分组
        g_descs0 = self._safe_chunk_2d(desc0, chunks=self.num_groups, dim=1)
        g_descs1 = self._safe_chunk_2d(desc1, chunks=self.num_groups, dim=1)

        processed_groups_0 = []
        processed_groups_1 = []

        # 3) 每个组各自做扫描 + Mamba
        for g in range(self.num_groups):
            d0 = g_descs0[g]
            d1 = g_descs1[g]

            # 4 个方向
            x0, h0, w0 = self._grouped_scan(d0, d1, direction=0)
            x0 = self.layers[0](x0)
            g00, g01 = self._grouped_merge(x0, h0, w0, direction=0)

            x1, h1, w1 = self._grouped_scan(d0, d1, direction=1)
            x1 = self.layers[1](x1)
            g10, g11 = self._grouped_merge(x1, h1, w1, direction=1)

            x2, h2, w2 = self._grouped_scan(d0, d1, direction=2)
            x2 = self.layers[2](x2)
            g20, g21 = self._grouped_merge(x2, h2, w2, direction=2)

            x3, h3, w3 = self._grouped_scan(d0, d1, direction=3)
            x3 = self.layers[3](x3)
            g30, g31 = self._grouped_merge(x3, h3, w3, direction=3)

            # 组内四方向融合：这里直接求和，比拼接更稳
            out_g0 = g00 + g10 + g20 + g30
            out_g1 = g01 + g11 + g21 + g31

            processed_groups_0.append(out_g0)
            processed_groups_1.append(out_g1)

        # 4) 合并所有组
        out0 = torch.cat(processed_groups_0, dim=1)[..., :H, :W]
        out1 = torch.cat(processed_groups_1, dim=1)[..., :H, :W]

        # 去掉 pad 的通道
        out0 = out0[:, :self.feature_dim]
        out1 = out1[:, :self.feature_dim]

        # 5) Channel Attention
        combined = torch.cat([out0, out1], dim=0)  # [2B, C, H, W]
        weights = self.ca(combined.mean(dim=(2, 3))).unsqueeze(-1).unsqueeze(-1)
        combined = combined * weights

        # 6) Aggregator
        desc = self.aggregator(combined)
        out0, out1 = torch.chunk(desc, 2, dim=0)

        # 7) residual add 回原始输入
        d0 = residual0 + out0
        d1 = residual1 + out1

        data.update({
            'feat_8_0': d0.flatten(2, 3),
            'feat_8_1': d1.flatten(2, 3)
        })
        return data

    def _grouped_scan(self, d0, d1, direction):
        """
        d0, d1: [B, Cg, H, W]
        returns:
            x: [B, L, Cg]
        """
        d_2w = torch.cat([d0, d1], dim=3)   # [B, Cg, H, 2W]
        d_2h = torch.cat([d0, d1], dim=2)   # [B, Cg, 2H, W]

        if direction == 0:
            x = d_2w[:, :, ::self.step_size, ::self.step_size]
        elif direction == 1:
            x = d_2h.transpose(2, 3)[:, :, ::self.step_size, ::self.step_size]
        elif direction == 2:
            x = d_2w[:, :, ::self.step_size, ::self.step_size].flip([-1])
        elif direction == 3:
            x = d_2h.transpose(2, 3)[:, :, ::self.step_size, ::self.step_size].flip([-1])
        else:
            raise ValueError(f"Invalid direction: {direction}")

        return x.flatten(2).transpose(1, 2).contiguous(), d0.shape[2], d0.shape[3]

    def _grouped_merge(self, x, h, w, direction):
        """
        x: [B, L, Cg]
        return:
            tuple of two tensors:
            [B, Cg, h, w], [B, Cg, h, w]
        """
        B, L, C = x.shape
        H_s = math.ceil(h / self.step_size)
        W_s = math.ceil((2 * w) / self.step_size)

        feat = x.transpose(1, 2).contiguous().view(B, C, H_s, W_s)

        if direction == 0 or direction == 2:
            if direction == 2:
                feat = feat.flip([-1])
            res = F.interpolate(feat, size=(h, 2 * w), mode='bilinear', align_corners=False)
            a, b = torch.chunk(res, 2, dim=3)
            return a.contiguous(), b.contiguous()
        else:
            if direction == 3:
                feat = feat.flip([-1])
            res = F.interpolate(
                feat.transpose(2, 3),
                size=(2 * h, w),
                mode='bilinear',
                align_corners=False
            )
            a, b = torch.chunk(res, 2, dim=2)
            return a.contiguous(), b.contiguous()