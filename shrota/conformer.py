"""Cache-Aware Streaming FastConformer with 8x Causal Depthwise Subsampling.

Designed for 100% ONNX Runtime Mobile / XNNPACK / NNAPI / WASM compatibility
(Conv, MatMul, LayerNormalization, SiLU, Sigmoid, Mul, Concat, Slice) and exact
equivalence between chunked streaming (160 ms / 320 ms) and full-sequence execution.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSubsampling8x(nn.Module):
    """3-stage 2x strided causal depthwise-separable Conv1D stem (10 ms -> 80 ms frames).

    Each stage has kernel_size=3, stride=2 and takes a 1-frame left cache:
      input (B, C, T) + cache (B, C, 1) -> (B, C, T+1) -> conv(k=3, s=2) -> (B, C_out, T//2)
      next_cache = input[:, :, -1:]
    For any T divisible by 8 (e.g. 16 frames = 160 ms -> 2 output frames), chunked
    execution is 100% mathematically identical to full-sequence execution.
    """

    def __init__(self, n_mels: int = 80, d_model: int = 256):
        super().__init__()
        self.n_mels = n_mels
        self.d_model = d_model

        self.conv0 = nn.Conv1d(n_mels, d_model, kernel_size=3, stride=2, padding=0)
        self.act0 = nn.SiLU()

        self.dw1 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=0, groups=d_model)
        self.pw1 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.act1 = nn.SiLU()

        self.dw2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=0, groups=d_model)
        self.pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.act2 = nn.SiLU()

        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        mel: torch.Tensor,
        cache_s0: torch.Tensor,
        cache_s1: torch.Tensor,
        cache_s2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            mel: (B, T, 80) where T is a multiple of 8
            cache_s0: (B, 80, 1)
            cache_s1: (B, d_model, 1)
            cache_s2: (B, d_model, 1)
        Returns:
            x: (B, T // 8, d_model)
            next_s0, next_s1, next_s2
        """
        x0 = mel.transpose(1, 2)  # (B, 80, T)
        next_s0 = x0[:, :, -1:]
        x0_pad = torch.cat([cache_s0, x0], dim=2)
        x1 = self.act0(self.conv0(x0_pad))  # (B, d_model, T//2)

        next_s1 = x1[:, :, -1:]
        x1_pad = torch.cat([cache_s1, x1], dim=2)
        x2 = self.act1(self.pw1(self.dw1(x1_pad)))  # (B, d_model, T//4)

        next_s2 = x2[:, :, -1:]
        x2_pad = torch.cat([cache_s2, x2], dim=2)
        x3 = self.act2(self.pw2(self.dw2(x2_pad)))  # (B, d_model, T//8)

        out = self.out_norm(x3.transpose(1, 2))  # (B, T//8, d_model)
        return out, next_s0, next_s1, next_s2


class FeedForwardModule(nn.Module):
    """Macaron-style half-step Feed-Forward module."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.w1 = nn.Linear(d_model, d_ff)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.drop(self.act(self.w1(self.norm(x)))))


class CacheAwareCausalAttention(nn.Module):
    """Streaming Multi-Head Self-Attention with explicit K/V left-context cache."""

    def __init__(self, d_model: int, n_heads: int = 4, left_ctx: int = 64, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.left_ctx = left_ctx
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D)
            cache_k: (B, H, L_ctx, Dh)
            cache_v: (B, H, L_ctx, Dh)
        Returns:
            out: (B, T, D)
            next_k: (B, H, L_ctx, Dh)
            next_v: (B, H, L_ctx, Dh)
        """
        B, T, D = x.shape
        L_ctx = cache_k.size(2)
        h = self.norm(x)

        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k_new = self.k_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v_new = self.v_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        k_all = torch.cat([cache_k, k_new], dim=2)  # (B, H, L_ctx + T, Dh)
        v_all = torch.cat([cache_v, v_new], dim=2)

        next_k = k_all[:, :, -L_ctx:, :]
        next_v = v_all[:, :, -L_ctx:, :]

        scores = torch.matmul(q, k_all.transpose(-2, -1)) * self.scale  # (B, H, T, L_ctx + T)

        # Strictly causal + bounded left_ctx window mask so every query i in [0..T-1]
        # attends to exactly [i .. i + L_ctx] in k_all (which is L_ctx past tokens + self)!
        # This guarantees 100% equivalence between chunked streaming (even T=1 or T=2)
        # and full-sequence execution!
        if T > 1:
            q_idx = torch.arange(T, device=x.device).unsqueeze(1)  # (T, 1)
            k_idx = torch.arange(L_ctx + T, device=x.device).unsqueeze(0)  # (1, L_ctx + T)
            # Valid keys for query q_idx are in [q_idx, q_idx + L_ctx]
            invalid = (k_idx > (q_idx + L_ctx)) | (k_idx < q_idx)
            scores = scores.masked_fill(invalid.unsqueeze(0).unsqueeze(0), -1e4)

        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        ctx = torch.matmul(attn, v_all)  # (B, H, T, Dh)
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(ctx), next_k, next_v


class CacheAwareConvModule(nn.Module):
    """Streaming Conformer Convolution Module with GLU and causal depthwise Conv1d."""

    def __init__(self, d_model: int, kernel_size: int = 9, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.cache_len = kernel_size - 1

        self.norm = nn.LayerNorm(d_model)
        self.pw1 = nn.Linear(d_model, 2 * d_model)
        self.dw = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            groups=d_model,
        )
        self.conv_norm = nn.LayerNorm(d_model)
        self.act = nn.SiLU()
        self.pw2 = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cache_conv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D)
            cache_conv: (B, D, kernel_size - 1)
        Returns:
            out: (B, T, D)
            next_cache_conv: (B, D, kernel_size - 1)
        """
        h = self.norm(x)
        u, v = self.pw1(h).chunk(2, dim=-1)
        gated = u * torch.sigmoid(v)  # GLU: (B, T, D)

        g_t = gated.transpose(1, 2)  # (B, D, T)
        padded = torch.cat([cache_conv, g_t], dim=2)  # (B, D, K - 1 + T)
        next_cache = padded[:, :, -self.cache_len :]

        conv_out = self.dw(padded).transpose(1, 2)  # (B, T, D)
        conv_out = self.act(self.conv_norm(conv_out))
        out = self.drop(self.pw2(conv_out))
        return out, next_cache


class FastConformerBlock(nn.Module):
    """Single Cache-Aware FastConformer Block (FFN1 -> MHSA -> Conv -> FFN2 -> LayerNorm)."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 1024,
        conv_kernel: int = 9,
        left_ctx: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, d_ff, dropout)
        self.attn = CacheAwareCausalAttention(d_model, n_heads, left_ctx, dropout)
        self.conv = CacheAwareConvModule(d_model, conv_kernel, dropout)
        self.ff2 = FeedForwardModule(d_model, d_ff, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        cache_conv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x + 0.5 * self.ff1(x)
        attn_out, next_k, next_v = self.attn(x, cache_k, cache_v)
        x = x + attn_out
        conv_out, next_conv = self.conv(x, cache_conv)
        x = x + conv_out
        x = x + 0.5 * self.ff2(x)
        x = self.final_norm(x)
        return x, next_k, next_v, next_conv


class CacheAwareFastConformerEncoder(nn.Module):
    """Full Cache-Aware Streaming FastConformer Encoder."""

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 256,
        n_layers: int = 12,
        n_heads: int = 4,
        d_ff: int = 1024,
        conv_kernel: int = 9,
        left_ctx: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_ff = d_ff
        self.conv_kernel = conv_kernel
        self.left_ctx = left_ctx

        self.subsampling = CausalSubsampling8x(n_mels, d_model)
        self.blocks = nn.ModuleList(
            [
                FastConformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    conv_kernel=conv_kernel,
                    left_ctx=left_ctx,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def init_caches(
        self, batch_size: int = 1, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_s0 = torch.zeros(batch_size, self.n_mels, 1, device=device, dtype=dtype)
        cache_s1 = torch.zeros(batch_size, self.d_model, 1, device=device, dtype=dtype)
        cache_s2 = torch.zeros(batch_size, self.d_model, 1, device=device, dtype=dtype)
        cache_k = torch.zeros(
            self.n_layers, batch_size, self.n_heads, self.left_ctx, self.head_dim, device=device, dtype=dtype
        )
        cache_v = torch.zeros(
            self.n_layers, batch_size, self.n_heads, self.left_ctx, self.head_dim, device=device, dtype=dtype
        )
        cache_conv = torch.zeros(
            self.n_layers, batch_size, self.d_model, self.conv_kernel - 1, device=device, dtype=dtype
        )
        return cache_s0, cache_s1, cache_s2, cache_k, cache_v, cache_conv

    def forward_streaming(
        self,
        mel: torch.Tensor,
        cache_s0: torch.Tensor,
        cache_s1: torch.Tensor,
        cache_s2: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        cache_conv: torch.Tensor,
        return_layer_states: bool = False,
    ):
        """Run one streaming chunk (or full sequence) with explicit cache tensors."""
        x, next_s0, next_s1, next_s2 = self.subsampling(mel, cache_s0, cache_s1, cache_s2)

        next_k_list = []
        next_v_list = []
        next_conv_list = []
        layer_states = []

        for i, blk in enumerate(self.blocks):
            x, nk, nv, nc = blk(x, cache_k[i], cache_v[i], cache_conv[i])
            next_k_list.append(nk)
            next_v_list.append(nv)
            next_conv_list.append(nc)
            if return_layer_states:
                layer_states.append(x)

        next_k = torch.stack(next_k_list, dim=0)
        next_v = torch.stack(next_v_list, dim=0)
        next_conv = torch.stack(next_conv_list, dim=0)

        if return_layer_states:
            return x, next_s0, next_s1, next_s2, next_k, next_v, next_conv, layer_states
        return x, next_s0, next_s1, next_s2, next_k, next_v, next_conv

    def forward(self, mel: torch.Tensor, return_layer_states: bool = False):
        """Convenience full-sequence forward starting from zero caches."""
        B = mel.size(0)
        T = mel.size(1)
        rem = T % 8
        if rem != 0:
            mel = F.pad(mel, (0, 0, 0, 8 - rem))
        caches = self.init_caches(B, device=mel.device, dtype=mel.dtype)
        out = self.forward_streaming(mel, *caches, return_layer_states=return_layer_states)
        if return_layer_states:
            return out[0], out[-1]
        return out[0]
