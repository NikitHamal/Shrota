"""Cache-Aware Streaming FastConformer v2 with 4x Causal Subsampling & Self-Conditioned InterCTC.

Designed for 100% ONNX Runtime Mobile / XNNPACK / NNAPI / WASM compatibility
(Conv, MatMul, LayerNormalization, SiLU, Sigmoid, Softmax, Mul, Concat, Slice) and exact
equivalence between chunked streaming (160 ms = 16 mel frames -> 4 encoder frames) and
full-sequence offline execution.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSubsampling4x(nn.Module):
    """2-strided + 1-refinement causal depthwise-separable Conv1D stem (10 ms -> 40 ms frames).

    Stage 0: k=3, stride=2 with 1-frame left cache_s0 (B, 80, 1) -> (B, d_model, T//2) [20 ms]
    Stage 1: k=3, stride=2 depthwise-separable with 1-frame left cache_s1 (B, d_model, 1) -> (B, d_model, T//4) [40 ms]
    Stage 2: k=2, stride=1 causal depthwise-separable refinement with 1-frame left cache_s2 (B, d_model, 1) -> (B, T//4, d_model) [40 ms]
    Plus a direct 4-frame (40 ms) spectrogram patch linear residual connection.

    For any T divisible by 4 (e.g. 16 frames = 160 ms -> 4 output frames), chunked
    streaming execution is 100% mathematically identical to full-sequence execution.
    """

    def __init__(self, n_mels: int = 80, d_model: int = 256):
        super().__init__()
        self.n_mels = n_mels
        self.d_model = d_model
        self.subsampling_factor = 4

        # Direct causal 4-frame (40ms) spectrogram patch residual projection
        self.frame_proj = nn.Linear(n_mels * 4, d_model)

        self.conv0 = nn.Conv1d(n_mels, d_model, kernel_size=3, stride=2, padding=0)
        self.norm0 = nn.LayerNorm(d_model)
        self.act0 = nn.SiLU()

        self.dw1 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=0, groups=d_model)
        self.pw1 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.act1 = nn.SiLU()

        # Causal stride-1 refinement stage (preserves 40 ms frame rate & cache_s2 signature)
        self.dw2 = nn.Conv1d(d_model, d_model, kernel_size=2, stride=1, padding=0, groups=d_model)
        self.pw2 = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
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
            mel: (B, T, 80) where T is a multiple of 4
            cache_s0: (B, 80, 1)
            cache_s1: (B, d_model, 1)
            cache_s2: (B, d_model, 1)
        Returns:
            x: (B, T // 4, d_model)
            next_s0, next_s1, next_s2
        """
        B, T, _ = mel.shape
        direct = self.frame_proj(mel.reshape(B, T // 4, self.n_mels * 4))

        x0 = mel.transpose(1, 2)  # (B, 80, T)
        next_s0 = x0[:, :, -1:]
        x0_pad = torch.cat([cache_s0, x0], dim=2)
        x1 = self.act0(self.norm0(self.conv0(x0_pad).transpose(1, 2))).transpose(1, 2)  # (B, d_model, T//2)

        next_s1 = x1[:, :, -1:]
        x1_pad = torch.cat([cache_s1, x1], dim=2)
        dw1_out = self.dw1(x1_pad).transpose(1, 2)
        pw1_out = self.pw1(dw1_out)
        x2_t = self.act1(self.norm1(pw1_out))  # (B, T//4, d_model)
        x2 = x2_t.transpose(1, 2)  # (B, d_model, T//4)

        next_s2 = x2[:, :, -1:]
        x2_pad = torch.cat([cache_s2, x2], dim=2)
        dw2_out = self.dw2(x2_pad).transpose(1, 2)
        pw2_out = self.pw2(dw2_out)
        x3 = self.act2(self.norm2(pw2_out))  # (B, T//4, d_model)

        out = self.out_norm(x3 + x2_t + direct)  # (B, T//4, d_model)
        return out, next_s0, next_s1, next_s2


# Alias for backward compatibility
CausalSubsampling8x = CausalSubsampling4x


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
    """Streaming Multi-Head Self-Attention with explicit K/V left-context cache and learnable relative slopes."""

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
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        # Gentle learnable relative distance slopes per head (allows full 2.56s context horizon)
        init_slopes = [0.004, 0.012, 0.035, 0.090][:n_heads]
        self.head_slopes = nn.Parameter(torch.tensor(init_slopes, dtype=torch.float32).view(1, n_heads, 1, 1))

    def forward(
        self,
        x: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        q_idx = torch.arange(T, device=x.device).unsqueeze(1)  # (T, 1)
        k_idx = torch.arange(L_ctx + T, device=x.device).unsqueeze(0)  # (1, L_ctx + T)
        rel_pos = k_idx - (q_idx + L_ctx)  # (T, L_ctx + T), 0 at self, negative for past
        invalid = (rel_pos > 0) | (rel_pos < -L_ctx)

        slopes = torch.abs(self.head_slopes).to(scores.dtype)
        rel_clamped = torch.clamp(rel_pos, min=-L_ctx, max=0).to(scores.dtype).unsqueeze(0).unsqueeze(0)
        scores = scores + slopes * rel_clamped
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
    """Single Cache-Aware Pre-LN FastConformer Block (FFN1 -> MHSA -> Conv -> FFN2)."""

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
        return x, next_k, next_v, next_conv


class CacheAwareFastConformerEncoder(nn.Module):
    """Full Cache-Aware Streaming FastConformer Encoder v2 with 4x Subsampling & SC-InterCTC."""

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 256,
        n_layers: int = 12,
        n_heads: int = 4,
        d_ff: int = 1024,
        conv_kernel: int = 9,
        left_ctx: int = 64,
        vocab_size: int = 1024,
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
        self.vocab_size = vocab_size
        self.subsampling_factor = 4

        # Self-Conditioned Intermediate CTC at layers 4 & 8 (for 12L Base) or layers 3 & 6 (for 8L Tiny)
        if n_layers >= 12:
            self.inter_layers = (3, 7)
        elif n_layers >= 8:
            self.inter_layers = (2, 5)
        else:
            self.inter_layers = (max(0, n_layers // 2 - 1),)

        self.subsampling = CausalSubsampling4x(n_mels, d_model)
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
        self.final_norm = nn.LayerNorm(d_model)

        # Shared CTC projection head + SC-InterCTC norms & conditioning projections
        self.ctc_head = nn.Linear(d_model, vocab_size)
        with torch.no_grad():
            self.ctc_head.bias.zero_()
        self.inter_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in self.inter_layers])
        self.inter_cond_proj = nn.Linear(vocab_size, d_model, bias=False)
        nn.init.normal_(self.inter_cond_proj.weight, mean=0.0, std=0.02)

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
        """Run one streaming chunk (or full sequence) with explicit cache tensors & SC-InterCTC."""
        x, next_s0, next_s1, next_s2 = self.subsampling(mel, cache_s0, cache_s1, cache_s2)

        next_k_list = []
        next_v_list = []
        next_conv_list = []
        inter_logits_list = []

        inter_idx = 0
        for i, blk in enumerate(self.blocks):
            x, nk, nv, nc = blk(x, cache_k[i], cache_v[i], cache_conv[i])
            next_k_list.append(nk)
            next_v_list.append(nv)
            next_conv_list.append(nc)
            if i in self.inter_layers:
                h_inter = self.inter_norms[inter_idx](x)
                inter_idx += 1
                logits_inter = self.ctc_head(h_inter)
                probs_inter = torch.softmax(logits_inter, dim=-1)
                x = x + self.inter_cond_proj(probs_inter)
                if return_layer_states:
                    inter_logits_list.append(logits_inter)

        x = self.final_norm(x)
        next_k = torch.stack(next_k_list, dim=0)
        next_v = torch.stack(next_v_list, dim=0)
        next_conv = torch.stack(next_conv_list, dim=0)

        if return_layer_states:
            return x, next_s0, next_s1, next_s2, next_k, next_v, next_conv, inter_logits_list
        return x, next_s0, next_s1, next_s2, next_k, next_v, next_conv

    def forward(self, mel: torch.Tensor, return_layer_states: bool = False):
        """Convenience full-sequence forward starting from zero caches."""
        B = mel.size(0)
        T = mel.size(1)
        rem = T % 4
        if rem != 0:
            mel = F.pad(mel, (0, 0, 0, 4 - rem))
        caches = self.init_caches(B, device=mel.device, dtype=mel.dtype)
        out = self.forward_streaming(mel, *caches, return_layer_states=return_layer_states)
        if return_layer_states:
            return out[0], out[-1]
        return out[0]
