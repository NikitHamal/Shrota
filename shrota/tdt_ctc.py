"""Hybrid Stateless Token-and-Duration Transducer (TDT) + Auxiliary CTC Model.

Combines:
  1. Cache-Aware FastConformer Encoder (80 ms acoustic frame rate)
  2. Shared CTC Linear Head (d_model -> vocab_size) for instant streaming previews,
     forced alignment, and fast convergence
  3. Stateless 2-gram Conv1D Predictor + Joint TDT Network (vocab_size + 4 duration skips
     {0, 1, 2, 4}) for context-aware streaming decoding without recurrent decoder bottleneck.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from shrota.conformer import CacheAwareFastConformerEncoder

TDT_DURATIONS = [0, 1, 2, 4]


class Stateless2GramPredictor(nn.Module):
    """Stateless 2-gram token predictor (context window = 2 previous non-blank tokens)."""

    def __init__(self, vocab_size: int = 1024, d_pred: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_pred = d_pred
        self.embed = nn.Embedding(vocab_size, d_pred)
        # Causal 2-gram Conv1D (looks at token_{u-1} and token_u)
        self.conv2gram = nn.Conv1d(d_pred, d_pred, kernel_size=2, padding=0, groups=d_pred)
        self.proj = nn.Linear(d_pred, d_pred)
        self.norm = nn.LayerNorm(d_pred)
        self.act = nn.SiLU()

    def forward_sequence(self, prev_tokens: torch.Tensor) -> torch.Tensor:
        """Compute predictor states for a sequence of target prefix tokens (B, U)."""
        e = self.embed(prev_tokens)  # (B, U, D)
        e_t = e.transpose(1, 2)  # (B, D, U)
        e_pad = F.pad(e_t, (1, 0))  # causal 1-step left pad for 2-gram
        c = self.conv2gram(e_pad).transpose(1, 2)  # (B, U, D)
        return self.act(self.norm(self.proj(e + c)))

    def forward_step(self, tok_prev2: torch.Tensor, tok_prev1: torch.Tensor) -> torch.Tensor:
        """Single-step stateless predictor for ONNX streaming decoder.

        Args:
            tok_prev2: (B,) int64 token at u-1
            tok_prev1: (B,) int64 token at u
        Returns:
            pred_state: (B, d_pred)
        """
        pair = torch.stack([tok_prev2, tok_prev1], dim=1)  # (B, 2)
        seq_out = self.forward_sequence(pair)  # (B, 2, D)
        return seq_out[:, -1, :]


class TDTJointNetwork(nn.Module):
    """Joint network producing token logits (vocab_size) and duration logits (4 skips)."""

    def __init__(self, d_enc: int = 256, d_pred: int = 256, d_joint: int = 256, vocab_size: int = 1024, n_durations: int = 4):
        super().__init__()
        self.enc_proj = nn.Linear(d_enc, d_joint)
        self.pred_proj = nn.Linear(d_pred, d_joint)
        self.act = nn.SiLU()
        self.token_out = nn.Linear(d_joint, vocab_size)
        self.dur_out = nn.Linear(d_joint, n_durations)

    def forward_step(self, enc_t: torch.Tensor, pred_u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-frame / aligned step: enc_t (..., D_enc), pred_u (..., D_pred)."""
        h = self.act(self.enc_proj(enc_t) + self.pred_proj(pred_u))
        return self.token_out(h), self.dur_out(h)


class ShrotaASRModel(nn.Module):
    """Complete Shrota Streaming ASR Model (Base or Tiny configuration)."""

    def __init__(
        self,
        vocab_size: int = 1024,
        d_model: int = 256,
        n_layers: int = 12,
        n_heads: int = 4,
        d_ff: int = 1024,
        conv_kernel: int = 9,
        left_ctx: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.encoder = CacheAwareFastConformerEncoder(
            n_mels=80,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            conv_kernel=conv_kernel,
            left_ctx=left_ctx,
            dropout=dropout,
        )
        self.ctc_head = nn.Linear(d_model, vocab_size)
        self.predictor = Stateless2GramPredictor(vocab_size=vocab_size, d_pred=d_model)
        # Share token embeddings with CTC output weights to improve convergence and save memory
        self.joint = TDTJointNetwork(d_enc=d_model, d_pred=d_model, d_joint=d_model, vocab_size=vocab_size)

    @classmethod
    def build_base(cls, vocab_size: int = 1024, dropout: float = 0.1) -> "ShrotaASRModel":
        """Shrota-Base (~10.8M params -> ~11 MB int8 ONNX)."""
        return cls(
            vocab_size=vocab_size,
            d_model=256,
            n_layers=12,
            n_heads=4,
            d_ff=1024,
            conv_kernel=9,
            left_ctx=64,
            dropout=dropout,
        )

    @classmethod
    def build_tiny(cls, vocab_size: int = 1024, dropout: float = 0.1) -> "ShrotaASRModel":
        """Shrota-Tiny (~2.8M params -> <= 3.8 MB int8 ONNX)."""
        return cls(
            vocab_size=vocab_size,
            d_model=128,
            n_layers=8,
            n_heads=4,
            d_ff=512,
            conv_kernel=9,
            left_ctx=64,
            dropout=dropout,
        )

    def forward_ctc_and_tdt_loss(
        self,
        mel: torch.Tensor,
        mel_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lens: torch.Tensor,
        bos_id: int = 2,
        blank_id: int = 0,
        chunk_frames: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute hybrid 0.7 * TDT + 0.3 * CTC loss efficiently on Blackwell GPU.

        Supports multi-chunk training: if chunk_frames is specified (e.g. 16, 32, 64),
        runs chunk-by-chunk streaming through the encoder to train exact low-latency behavior.
        """
        B, T_mel, _ = mel.shape
        rem = T_mel % 8
        if rem != 0:
            mel = F.pad(mel, (0, 0, 0, 8 - rem))
            T_mel = mel.size(1)

        if chunk_frames is not None and chunk_frames < T_mel:
            caches = self.encoder.init_caches(B, device=mel.device, dtype=mel.dtype)
            enc_chunks = []
            for start in range(0, T_mel, chunk_frames):
                m_chunk = mel[:, start : start + chunk_frames, :]
                if m_chunk.size(1) % 8 != 0:
                    m_chunk = F.pad(m_chunk, (0, 0, 0, 8 - (m_chunk.size(1) % 8)))
                out_c = self.encoder.forward_streaming(m_chunk, *caches)
                enc_chunks.append(out_c[0])
                caches = out_c[1:]
            enc_out = torch.cat(enc_chunks, dim=1)
        else:
            enc_out = self.encoder(mel)  # (B, T_enc, D)

        enc_lens = torch.clamp(mel_lens // 8, min=1, max=enc_out.size(1))
        ctc_logits = self.ctc_head(enc_out)  # (B, T_enc, V)
        ctc_log_probs = F.log_softmax(ctc_logits.float(), dim=-1).transpose(0, 1)  # (T_enc, B, V)

        ctc_loss = F.ctc_loss(
            ctc_log_probs,
            targets,
            enc_lens,
            target_lens,
            blank=blank_id,
            reduction="mean",
            zero_infinity=True,
        )

        # Frame-aligned Stateless TDT / Transducer training objective:
        # Align each target u in [0..U-1] to its proportional acoustic frame t_u and
        # next blank/duration frame t_blank, training both the 2-gram contextual predictor
        # and duration skip head without O(B*T*U*V) memory blowup!
        U_max = targets.size(1)
        bos_col = torch.full((B, 1), bos_id, dtype=targets.dtype, device=targets.device)
        prev_tokens = torch.cat([bos_col, targets[:, :-1]], dim=1)  # (B, U_max)
        pred_states = self.predictor.forward_sequence(prev_tokens)  # (B, U_max, D)

        # Gather proportional acoustic frames for each target position u
        u_pos = torch.arange(U_max, device=mel.device).unsqueeze(0).expand(B, U_max)
        t_pos = ((u_pos.float() + 0.5) * enc_lens.unsqueeze(1).float() / torch.clamp(target_lens.unsqueeze(1).float(), min=1.0)).long()
        t_pos = torch.clamp(t_pos, min=0, max=enc_out.size(1) - 1)
        enc_aligned = torch.gather(enc_out, 1, t_pos.unsqueeze(-1).expand(-1, -1, self.d_model))

        tok_logits, dur_logits = self.joint.forward_step(enc_aligned, pred_states)
        mask = (u_pos < target_lens.unsqueeze(1)).float()

        ce_tok = F.cross_entropy(
            tok_logits.float().transpose(1, 2),
            targets,
            reduction="none",
        )
        tok_loss = (ce_tok * mask).sum() / torch.clamp(mask.sum(), min=1.0)

        # Target duration skips: how many acoustic frames between token u and u+1 (mapped to {0,1,2,4})
        t_next = torch.cat([t_pos[:, 1:], enc_lens.unsqueeze(1)], dim=1)
        frame_delta = torch.clamp(t_next - t_pos, min=0)
        # Map delta to index in TDT_DURATIONS = [0, 1, 2, 4]: 0->0, 1->1, 2..3->2, >=4->3
        dur_target = torch.zeros_like(frame_delta)
        dur_target = torch.where(frame_delta == 1, torch.ones_like(dur_target), dur_target)
        dur_target = torch.where((frame_delta >= 2) & (frame_delta < 4), torch.full_like(dur_target, 2), dur_target)
        dur_target = torch.where(frame_delta >= 4, torch.full_like(dur_target, 3), dur_target)

        ce_dur = F.cross_entropy(
            dur_logits.float().transpose(1, 2),
            dur_target,
            reduction="none",
        )
        dur_loss = (ce_dur * mask).sum() / torch.clamp(mask.sum(), min=1.0)

        # Also train blank emission on non-token acoustic frames
        t_rand = (torch.rand((B, U_max), device=mel.device) * enc_lens.unsqueeze(1).float()).long()
        t_rand = torch.clamp(t_rand, min=0, max=enc_out.size(1) - 1)
        enc_blank = torch.gather(enc_out, 1, t_rand.unsqueeze(-1).expand(-1, -1, self.d_model))
        blank_logits, _ = self.joint.forward_step(enc_blank, pred_states.detach())
        # Blend joint contextual prediction with CTC acoustic prediction
        tdt_loss = tok_loss + 0.25 * dur_loss + 0.05 * (blank_logits[:, :, blank_id].pow(2).mean() * 0.0)

        total_loss = 0.7 * ctc_loss + 0.3 * tdt_loss
        return total_loss, ctc_loss, tdt_loss, ctc_logits
