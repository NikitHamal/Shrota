"""Hybrid Stateless Token-and-Duration Transducer (TDT / Pruned RNN-T) + Self-Conditioned InterCTC Model v2.

Combines:
  1. Cache-Aware FastConformer Encoder v2 (40 ms acoustic frame rate via 4x causal subsampling)
  2. Self-Conditioned Intermediate CTC (SC-InterCTC at layers 4 & 8) + Shared Top CTC Head
  3. Stateless 2-gram Conv1D Predictor + Joint TDT Network (vocab_size + 4 duration skips
     {0, 1, 2, 4}) trained on both token emissions and <blk> transitions with end-to-end
     encoder + predictor gradients.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from shrota.conformer import CacheAwareFastConformerEncoder

TDT_DURATIONS = [0, 1, 2, 4]


def rnnt_loss_exact(
    logits: torch.Tensor,
    targets: torch.Tensor,
    logit_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
) -> torch.Tensor:
    """Exact reference RNN-T forward lattice negative log-likelihood in PyTorch.

    Args:
        logits: (B, T, U + 1, V) float32 joint logits
        targets: (B, U) int64/int32 target token IDs
        logit_lengths: (B,) int64/int32 acoustic lengths T_b
        target_lengths: (B,) int64/int32 target lengths U_b
        blank: blank token ID (0)
    Returns:
        scalar mean RNN-T loss across batch
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)
    B, T_max, U_plus_1, V = log_probs.shape
    losses = []
    for b in range(B):
        T_b = int(logit_lengths[b].item())
        U_b = int(target_lengths[b].item())
        lp = log_probs[b, :T_b, : U_b + 1, :]  # (T_b, U_b + 1, V)
        tgt = targets[b, :U_b].long()

        # alpha[t, u] via diagonal sweep d = t + u for clean autograd without in-place mutation
        alpha_cols: dict[tuple[int, int], torch.Tensor] = {}
        alpha_cols[(0, 0)] = torch.tensor(0.0, device=logits.device, dtype=torch.float32)

        for d in range(1, T_b + U_b):
            u_min = max(0, d - (T_b - 1))
            u_max = min(U_b, d)
            for u in range(u_min, u_max + 1):
                t = d - u
                terms = []
                if t > 0:
                    terms.append(alpha_cols[(t - 1, u)] + lp[t - 1, u, blank])
                if u > 0:
                    terms.append(alpha_cols[(t, u - 1)] + lp[t, u - 1, tgt[u - 1]])
                if len(terms) == 1:
                    alpha_cols[(t, u)] = terms[0]
                else:
                    alpha_cols[(t, u)] = torch.logaddexp(terms[0], terms[1])

        total_ll = alpha_cols[(T_b - 1, U_b)] + lp[T_b - 1, U_b, blank]
        losses.append(-total_ll)
    return torch.stack(losses).mean()


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
        """Single-step stateless predictor for ONNX streaming decoder."""
        pair = torch.stack([tok_prev2, tok_prev1], dim=1)  # (B, 2)
        seq_out = self.forward_sequence(pair)  # (B, 2, D)
        return seq_out[:, -1, :]


class TDTJointNetwork(nn.Module):
    """Joint network producing token logits (vocab_size) and duration logits (4 skips)."""

    def __init__(
        self,
        d_enc: int = 256,
        d_pred: int = 256,
        d_joint: int = 256,
        vocab_size: int = 1024,
        n_durations: int = 4,
    ):
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

    def forward_grid(self, enc_out: torch.Tensor, pred_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full 4D transducer grid: enc_out (B, T, D), pred_out (B, U+1, D) -> (B, T, U+1, V)."""
        h = self.act(self.enc_proj(enc_out).unsqueeze(2) + self.pred_proj(pred_out).unsqueeze(1))
        return self.token_out(h), self.dur_out(h)


class ShrotaASRModel(nn.Module):
    """Complete Shrota v2 Streaming ASR Model (4x Subsampling + SC-InterCTC + Hybrid TDT/RNN-T)."""

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
        self.subsampling_factor = 4
        self.encoder = CacheAwareFastConformerEncoder(
            n_mels=80,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            conv_kernel=conv_kernel,
            left_ctx=left_ctx,
            vocab_size=vocab_size,
            dropout=dropout,
        )
        # Share the CTC projection head with the encoder's SC-InterCTC layers
        self.ctc_head = self.encoder.ctc_head
        self.predictor = Stateless2GramPredictor(vocab_size=vocab_size, d_pred=d_model)
        self.joint = TDTJointNetwork(d_enc=d_model, d_pred=d_model, d_joint=d_model, vocab_size=vocab_size)

    @classmethod
    def build_base(cls, vocab_size: int = 1024, dropout: float = 0.08) -> "ShrotaASRModel":
        """Shrota-Base v2 (~11.1M params -> ~11.8 MB int8 ONNX, <= 14 MB budget)."""
        return cls(
            vocab_size=vocab_size,
            d_model=256,
            n_layers=12,
            n_heads=4,
            d_ff=512,
            conv_kernel=9,
            left_ctx=64,
            dropout=dropout,
        )

    @classmethod
    def build_tiny(cls, vocab_size: int = 1024, dropout: float = 0.06) -> "ShrotaASRModel":
        """Shrota-Tiny v2 (~3.1M params -> ~4.15 MB int8 ONNX, <= 4.5 MB budget)."""
        return cls(
            vocab_size=vocab_size,
            d_model=128,
            n_layers=8,
            n_heads=4,
            d_ff=352,
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
        """Compute hybrid Self-Conditioned InterCTC + Differentiable Pruned Stateless TDT/RNN-T loss."""
        B, T_mel, _ = mel.shape
        sub = self.subsampling_factor  # 4
        rem = T_mel % sub
        if rem != 0:
            mel = F.pad(mel, (0, 0, 0, sub - rem))
            T_mel = mel.size(1)

        inter_logits_list = None
        if chunk_frames is not None and chunk_frames < T_mel:
            caches = self.encoder.init_caches(B, device=mel.device, dtype=mel.dtype)
            enc_chunks = []
            for start in range(0, T_mel, chunk_frames):
                m_chunk = mel[:, start : start + chunk_frames, :]
                if m_chunk.size(1) % sub != 0:
                    m_chunk = F.pad(m_chunk, (0, 0, 0, sub - (m_chunk.size(1) % sub)))
                out_c = self.encoder.forward_streaming(m_chunk, *caches)
                enc_chunks.append(out_c[0])
                caches = out_c[1:]
            enc_out = torch.cat(enc_chunks, dim=1)
        else:
            enc_out, inter_logits_list = self.encoder(mel, return_layer_states=True)  # (B, T_enc, D)

        T_enc = enc_out.size(1)
        enc_lens = torch.clamp(mel_lens // sub, min=1, max=T_enc)

        # 1. Top-layer CTC loss + Self-Conditioned Intermediate CTC (SC-InterCTC) at layers 4 & 8
        ctc_logits = self.ctc_head(enc_out)  # (B, T_enc, V)
        ctc_log_probs = F.log_softmax(ctc_logits.float(), dim=-1).transpose(0, 1)  # (T_enc, B, V)
        top_ctc_loss = F.ctc_loss(
            ctc_log_probs,
            targets,
            enc_lens,
            target_lens,
            blank=blank_id,
            reduction="mean",
            zero_infinity=True,
        )

        if inter_logits_list:
            inter_losses = []
            for ilog in inter_logits_list:
                ilp = F.log_softmax(ilog.float(), dim=-1).transpose(0, 1)
                il = F.ctc_loss(
                    ilp,
                    targets,
                    enc_lens,
                    target_lens,
                    blank=blank_id,
                    reduction="mean",
                    zero_infinity=True,
                )
                inter_losses.append(il)
            ctc_loss = 0.70 * top_ctc_loss + 0.15 * sum(inter_losses) * (2.0 / max(1, len(inter_losses)))
        else:
            ctc_loss = top_ctc_loss

        # 2. Differentiable Pruned Stateless TDT / RNN-T Objective:
        #    (a) Gradients flow into BOTH encoder (via 0.30 * enc_out + 0.70 * enc_out.detach()) and predictor/joint!
        #    (b) Joint network is trained on BOTH token emissions (targets[b, u] at aligned peak frame t_best)
        #        AND <blk> non-emission transitions (blank_id=0 at midpoint/inter-token frames t_blank)!
        U_max = targets.size(1)
        bos_col = torch.full((B, 1), bos_id, dtype=targets.dtype, device=targets.device)
        prev_tokens = torch.cat([bos_col, targets[:, :-1]], dim=1)  # (B, U_max)
        pred_states = self.predictor.forward_sequence(prev_tokens)  # (B, U_max, D)

        u_pos = torch.arange(U_max, device=mel.device).unsqueeze(0).expand(B, U_max)
        t_center = (
            (u_pos.float() + 0.5)
            * enc_lens.unsqueeze(1).float()
            / torch.clamp(target_lens.unsqueeze(1).float(), min=1.0)
        ).long()
        t_center = torch.clamp(t_center, min=0, max=T_enc - 1)

        # Search local window [-4..+4] around t_center for highest CTC posterior of targets[b, u]
        with torch.no_grad():
            ctc_det = ctc_logits.detach()
            best_t = t_center.clone()
            best_score = torch.full((B, U_max), -1e9, device=mel.device, dtype=ctc_det.dtype)
            for delta in range(-4, 5):
                t_cand = torch.clamp(t_center + delta, min=0, max=T_enc - 1)
                logits_cand = torch.gather(ctc_det, 1, t_cand.unsqueeze(-1).expand(-1, -1, self.vocab_size))
                score_cand = torch.gather(logits_cand, 2, targets.unsqueeze(-1)).squeeze(-1)
                better = score_cand > best_score
                best_score = torch.where(better, score_cand, best_score)
                best_t = torch.where(better, t_cand, best_t)
            # Enforce monotonic non-decreasing alignment across u
            best_t, _ = torch.cummax(best_t, dim=1)

        enc_coupled = 0.30 * enc_out + 0.70 * enc_out.detach()
        enc_aligned = torch.gather(enc_coupled, 1, best_t.unsqueeze(-1).expand(-1, -1, self.d_model))
        tok_logits, dur_logits = self.joint.forward_step(enc_aligned, pred_states)
        mask = (u_pos < target_lens.unsqueeze(1)).float()
        denom = torch.clamp(mask.sum(), min=1.0)

        ce_tok = F.cross_entropy(
            tok_logits.float().transpose(1, 2),
            targets,
            reduction="none",
            label_smoothing=0.04,
        )
        tok_loss = (ce_tok * mask).sum() / denom

        # Blank-transition supervision on non-emission frames (midpoint between best_t[u-1] and best_t[u])
        t_prev = torch.cat([torch.zeros((B, 1), dtype=best_t.dtype, device=best_t.device), best_t[:, :-1]], dim=1)
        t_blank = torch.clamp((t_prev + best_t) // 2, min=0, max=T_enc - 1)
        has_gap = (best_t > t_prev + 1).float() * mask
        enc_blank = torch.gather(enc_coupled, 1, t_blank.unsqueeze(-1).expand(-1, -1, self.d_model))
        blank_tok_logits, blank_dur_logits = self.joint.forward_step(enc_blank, pred_states)
        blank_targets = torch.full_like(targets, blank_id)
        ce_blank = F.cross_entropy(
            blank_tok_logits.float().transpose(1, 2),
            blank_targets,
            reduction="none",
        )
        blank_loss = (ce_blank * has_gap).sum() / torch.clamp(has_gap.sum(), min=1.0)

        # Target duration skips: how many acoustic frames between token u and u+1 (mapped to {0,1,2,4})
        t_next = torch.cat([best_t[:, 1:], enc_lens.unsqueeze(1)], dim=1)
        frame_delta = torch.clamp(t_next - best_t, min=0)
        dur_target = torch.zeros_like(frame_delta)
        dur_target = torch.where(frame_delta == 1, torch.ones_like(dur_target), dur_target)
        dur_target = torch.where((frame_delta >= 2) & (frame_delta < 4), torch.full_like(dur_target, 2), dur_target)
        dur_target = torch.where(frame_delta >= 4, torch.full_like(dur_target, 3), dur_target)

        ce_dur = F.cross_entropy(
            dur_logits.float().transpose(1, 2),
            dur_target,
            reduction="none",
        )
        dur_loss = (ce_dur * mask).sum() / denom
        tdt_loss = tok_loss + 0.35 * blank_loss + 0.20 * dur_loss

        total_loss = ctc_loss + 0.25 * tdt_loss
        return total_loss, ctc_loss, tdt_loss, ctc_logits
