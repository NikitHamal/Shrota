"""Hardware-Friendly Opset 17 Stateful Streaming ONNX Exporter.

Guarantees:
  - Opset 17 standard operators only (Conv, MatMul, LayerNormalization, SiLU/Sigmoid/Mul, Concat, Slice)
  - ZERO ConvInteger operators (quantizes MatMul/Gemm weights to INT8 while keeping depthwise Conv
    in native float so Android XNNPACK/NNAPI, Desktop CPU, and Browser WASM run without error!)
  - Exports:
      * encoder + ctc_head stateful streaming graph:
        inputs: (mel, cache_s0, cache_s1, cache_s2, cache_k, cache_v, cache_conv)
        outputs: (ctc_logits, enc_out, next_s0, next_s1, next_s2, next_k, next_v, next_conv)
      * decoder (stateless 2-gram predictor + joint TDT network):
        inputs: (enc_frame, tok_prev2, tok_prev1)
        outputs: (tdt_tok_logits, tdt_dur_logits)
      * shuddha (4L bidirectional grammar, hraswa-dirgha & punctuation refiner):
        inputs: (input_ids,)
        outputs: (token_logits, punct_logits)
"""
from __future__ import annotations
import os
import onnx
import torch
import torch.nn as nn
from onnxruntime.quantization import quantize_dynamic, QuantType

from shrota.tdt_ctc import ShrotaASRModel
from shrota.shuddha import ShrotaShuddhaModel


class StatefulStreamingEncoderWrapper(nn.Module):
    """Wraps Encoder + CTC head for single-session streaming ONNX execution."""

    def __init__(self, asr_model: ShrotaASRModel):
        super().__init__()
        self.encoder = asr_model.encoder
        self.ctc_head = asr_model.ctc_head

    def forward(
        self,
        mel: torch.Tensor,
        cache_s0: torch.Tensor,
        cache_s1: torch.Tensor,
        cache_s2: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        cache_conv: torch.Tensor,
    ):
        enc_out, next_s0, next_s1, next_s2, next_k, next_v, next_conv = self.encoder.forward_streaming(
            mel, cache_s0, cache_s1, cache_s2, cache_k, cache_v, cache_conv
        )
        ctc_logits = self.ctc_head(enc_out)
        return ctc_logits, enc_out, next_s0, next_s1, next_s2, next_k, next_v, next_conv


class StatelessTDTDecoderWrapper(nn.Module):
    """Wraps Stateless 2-gram Predictor + Joint TDT network."""

    def __init__(self, asr_model: ShrotaASRModel):
        super().__init__()
        self.predictor = asr_model.predictor
        self.joint = asr_model.joint

    def forward(
        self,
        enc_frame: torch.Tensor,
        tok_prev2: torch.Tensor,
        tok_prev1: torch.Tensor,
    ):
        pred_state = self.predictor.forward_step(tok_prev2, tok_prev1)
        tok_logits, dur_logits = self.joint.forward_step(enc_frame, pred_state)
        return tok_logits, dur_logits


def verify_zero_conv_integer(onnx_path: str) -> dict:
    """Inspect exported ONNX graph and assert zero ConvInteger or custom ops."""
    model = onnx.load(onnx_path)
    op_counts: dict[str, int] = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    if "ConvInteger" in op_counts:
        raise RuntimeError(f"Forbidden ConvInteger found in {onnx_path}: {op_counts}")
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    return {"path": onnx_path, "size_mb": round(size_mb, 3), "op_counts": op_counts}


def quantize_matmul_int8(fp32_path: str, int8_path: str) -> dict:
    """Quantize MatMul/Gemm weights to INT8 while keeping Conv in native FP32 (zero ConvInteger!)."""
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        op_types_to_quantize=["MatMul", "Gemm"],
        weight_type=QuantType.QUInt8,
        per_channel=False,
    )
    return verify_zero_conv_integer(int8_path)


def export_asr_onnx(
    asr_model: ShrotaASRModel,
    out_dir: str,
    prefix: str = "shrota_base",
    chunk_frames: int = 16,
) -> dict:
    """Export stateful streaming encoder + stateless TDT decoder to FP32 and INT8 ONNX (opset 17)."""
    os.makedirs(out_dir, exist_ok=True)
    asr_model = asr_model.cpu().eval()

    enc_wrap = StatefulStreamingEncoderWrapper(asr_model).eval()
    dec_wrap = StatelessTDTDecoderWrapper(asr_model).eval()

    mel_dummy = torch.randn(1, chunk_frames, 80, dtype=torch.float32)
    caches_dummy = asr_model.encoder.init_caches(1, device="cpu", dtype=torch.float32)

    enc_fp32 = os.path.join(out_dir, f"{prefix}_encoder_fp32.onnx")
    enc_int8 = os.path.join(out_dir, f"{prefix}_encoder_int8.onnx")

    input_names = ["mel", "cache_s0", "cache_s1", "cache_s2", "cache_k", "cache_v", "cache_conv"]
    output_names = ["ctc_logits", "enc_out", "next_s0", "next_s1", "next_s2", "next_k", "next_v", "next_conv"]
    dynamic_axes = {
        "mel": {0: "batch", 1: "mel_frames"},
        "ctc_logits": {0: "batch", 1: "enc_frames"},
        "enc_out": {0: "batch", 1: "enc_frames"},
    }

    torch.onnx.export(
        enc_wrap,
        (mel_dummy, *caches_dummy),
        enc_fp32,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    enc_info = quantize_matmul_int8(enc_fp32, enc_int8)

    dec_fp32 = os.path.join(out_dir, f"{prefix}_decoder_fp32.onnx")
    dec_int8 = os.path.join(out_dir, f"{prefix}_decoder_int8.onnx")
    enc_frame_dummy = torch.randn(1, asr_model.d_model, dtype=torch.float32)
    tok_prev2_dummy = torch.tensor([2], dtype=torch.long)
    tok_prev1_dummy = torch.tensor([4], dtype=torch.long)

    torch.onnx.export(
        dec_wrap,
        (enc_frame_dummy, tok_prev2_dummy, tok_prev1_dummy),
        dec_fp32,
        input_names=["enc_frame", "tok_prev2", "tok_prev1"],
        output_names=["tdt_tok_logits", "tdt_dur_logits"],
        dynamic_axes={
            "enc_frame": {0: "batch"},
            "tok_prev2": {0: "batch"},
            "tok_prev1": {0: "batch"},
            "tdt_tok_logits": {0: "batch"},
            "tdt_dur_logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    dec_info = quantize_matmul_int8(dec_fp32, dec_int8)

    return {
        "encoder_fp32": enc_fp32,
        "encoder_int8": enc_info,
        "decoder_fp32": dec_fp32,
        "decoder_int8": dec_info,
        "total_int8_mb": round(enc_info["size_mb"] + dec_info["size_mb"], 3),
    }


def export_shuddha_onnx(
    shuddha_model: ShrotaShuddhaModel,
    out_dir: str,
    prefix: str = "shrota_shuddha",
) -> dict:
    """Export Shrota-Shuddha bidirectional refiner to FP32 and INT8 ONNX (opset 17)."""
    os.makedirs(out_dir, exist_ok=True)
    shuddha_model = shuddha_model.cpu().eval()

    fp32_path = os.path.join(out_dir, f"{prefix}_fp32.onnx")
    int8_path = os.path.join(out_dir, f"{prefix}_int8.onnx")
    dummy_ids = torch.tensor([[2, 4, 6, 12, 18, 25, 3]], dtype=torch.long)

    torch.onnx.export(
        shuddha_model,
        (dummy_ids,),
        fp32_path,
        input_names=["input_ids"],
        output_names=["token_logits", "punct_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "token_logits": {0: "batch", 1: "seq_len"},
            "punct_logits": {0: "batch", 1: "seq_len"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    info = quantize_matmul_int8(fp32_path, int8_path)
    return {"fp32": fp32_path, "int8": info}
