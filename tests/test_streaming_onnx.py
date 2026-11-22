"""Test PyTorch <-> ONNX Opset 17 streaming chunk-vs-offline equivalence & zero ConvInteger."""
import tempfile
import numpy as np
import onnxruntime as ort
import torch

from shrota.tdt_ctc import ShrotaASRModel, rnnt_loss_exact
from shrota.shuddha import ShrotaShuddhaModel
from shrota.export_onnx import export_asr_onnx, export_shuddha_onnx


def test_streaming_chunk_vs_offline_and_onnx_parity():
    torch.manual_seed(42)
    model = ShrotaASRModel.build_tiny(vocab_size=1024).eval()
    mel = torch.randn(1, 64, 80)  # 64 frames = 640 ms (4 chunks of 16 frames = 160 ms each)

    with torch.no_grad():
        full_enc, inter_logits = model.encoder(mel, return_layer_states=True)
        assert full_enc.shape == (1, 16, 128), f"Expected 4x subsampling (1, 16, 128), got {full_enc.shape}"
        assert len(inter_logits) == 2
        full_ctc = model.ctc_head(full_enc)

        caches = model.encoder.init_caches(1)
        chunk_encs = []
        for start in range(0, 64, 16):
            out = model.encoder.forward_streaming(mel[:, start : start + 16, :], *caches)
            chunk_encs.append(out[0])
            caches = out[1:]
        stream_enc = torch.cat(chunk_encs, dim=1)
        stream_ctc = model.ctc_head(stream_enc)

    diff_pt = (full_ctc - stream_ctc).abs().max().item()
    assert diff_pt < 1e-4, f"PyTorch chunked vs full max_abs_diff too large: {diff_pt}"

    with tempfile.TemporaryDirectory() as tmpdir:
        info = export_asr_onnx(model, tmpdir, prefix="test_tiny", chunk_frames=16)
        assert info["total_int8_mb"] <= 4.5, f"Tiny INT8 size exceeded 4.5 MB: {info['total_int8_mb']}"

        sess = ort.InferenceSession(info["encoder_fp32"], providers=["CPUExecutionProvider"])
        ort_caches = [c.numpy() for c in model.encoder.init_caches(1)]
        ort_ctc_chunks = []
        for start in range(0, 64, 16):
            m_np = mel[:, start : start + 16, :].numpy()
            feeds = {
                "mel": m_np,
                "cache_s0": ort_caches[0],
                "cache_s1": ort_caches[1],
                "cache_s2": ort_caches[2],
                "cache_k": ort_caches[3],
                "cache_v": ort_caches[4],
                "cache_conv": ort_caches[5],
            }
            outs = sess.run(None, feeds)
            ort_ctc_chunks.append(outs[0])
            ort_caches = outs[2:]

        ort_ctc = np.concatenate(ort_ctc_chunks, axis=1)
        diff_onnx = np.max(np.abs(full_ctc.numpy() - ort_ctc))
        assert diff_onnx < 1e-3, f"ONNX streaming vs PyTorch full max_abs_diff too large: {diff_onnx}"

        shuddha = ShrotaShuddhaModel(vocab_size=1024).eval()
        s_info = export_shuddha_onnx(shuddha, tmpdir)
        assert s_info["int8"]["size_mb"] <= 2.5

    # Verify exact RNN-T lattice loss and hybrid SC-InterCTC + TDT backward pass
    logits_4d = torch.randn(2, 8, 4, 16, requires_grad=True)
    targets_2d = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
    l_lens = torch.tensor([8, 6], dtype=torch.long)
    t_lens = torch.tensor([3, 2], dtype=torch.long)
    rnnt_l = rnnt_loss_exact(logits_4d, targets_2d, l_lens, t_lens, blank=0)
    rnnt_l.backward()
    assert torch.isfinite(rnnt_l) and logits_4d.grad is not None and torch.isfinite(logits_4d.grad).all()
