import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import os
    import re
    import subprocess
    import time
    import urllib.request
    import marimo as mo
    return json, mo, os, re, subprocess, time, urllib


@app.cell
def _(mo):
    tick = mo.ui.refresh(options=["3s"], default_interval="3s", label="tick")
    heartbeat = mo.ui.refresh(options=["60s"], default_interval="60s", label="keep-alive")
    mo.hstack([tick, heartbeat], justify="start", gap=1)
    return heartbeat, tick


@app.cell
def _(heartbeat, json, mo, os, re, subprocess, tick, time, urllib):
    _ = (tick.value, heartbeat.value)
    EVAL, LOGS = "/work/shrota/eval", "/work/shrota/logs"

    def read_json(p):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def tail(paths, n=4):
        if isinstance(paths, str):
            paths = [paths]
        for p in paths:
            try:
                if os.path.exists(p) and os.path.getsize(p) > 0:
                    with open(p, encoding="utf-8", errors="replace") as f:
                        return [ln.rstrip()[:150] for ln in f.readlines()[-n:]]
            except Exception:
                pass
        return []

    def sh(cmd, t=3):
        try:
            return subprocess.run(cmd, shell=True, text=True, capture_output=True, timeout=t).stdout.strip()
        except Exception:
            return ""

    def esc(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def progress(lines):
        for ln in reversed(lines):
            m = re.search(r"[sS]tep[= ]+(\d+)/(\d+)", str(ln))
            if m:
                return int(m.group(1)), int(m.group(2))
        return None

    def state_of(text):
        t = (text or "").strip()
        if not t or t.lower().startswith("pending"):
            return "Pending", "#ECE4DD", "#51443D"
        if t.lower().startswith("done"):
            return "Done", "#D7F5D9", "#07290B"
        if "fail" in t.lower() or "error" in t.lower():
            return "Failed", "#FFD9E1", "#3A0014"
        return "Running", "#FFE9B3", "#3B2A00"

    def card(title, status, rows, log):
        label, bg, fg = state_of(status)
        pr = progress((log or []) + ([status] if status else []))
        bar = ""
        if pr and label == "Running":
            pct = 100.0 * pr[0] / max(pr[1], 1)
            bar = (f"<div style='height:6px;background:#ECE4DD;border-radius:3px;margin:10px 0'>"
                   f"<div style='height:6px;width:{pct:.1f}%;background:#8C1D40;border-radius:3px'></div></div>"
                   f"<div style='font-size:12px;color:#51443D'>{pr[0]} / {pr[1]} steps ({pct:.0f}%)</div>")
        kv = "".join(
            f"<div style='display:flex;justify-content:space-between;gap:12px;padding:4px 0;border-bottom:1px solid #F4EFEA'>"
            f"<span style='color:#51443D'>{esc(k)}</span><span style='font-weight:500;text-align:right'>{esc(v)}</span></div>"
            for k, v in rows)
        logs = ("<pre style='margin:10px 0 0;padding:10px;background:#F4EFEA;border-radius:12px;font-size:11px;"
                "line-height:1.5;white-space:pre-wrap;overflow:hidden'>" + esc("\n".join(log)) + "</pre>") if log else ""
        return (f"<div style='background:#fff;border:1px solid #E0D3CA;border-radius:24px;padding:20px;min-width:0'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;gap:8px'>"
                f"<div style='font-weight:600;font-size:16px'>{esc(title)}</div>"
                f"<span style='background:{bg};color:{fg};padding:3px 12px;border-radius:999px;font-size:12px;font-weight:500'>{label}</span></div>"
                f"<div style='font-size:13px;color:#51443D;margin:8px 0'>{esc(status or 'Waiting')}</div>{bar}"
                f"<div style='font-size:13px;margin-top:8px'>{kv}</div>{logs}</div>")

    status = read_json(f"{EVAL}/status.json")
    base, tiny = read_json(f"{EVAL}/shrota_base.json"), read_json(f"{EVAL}/shrota_tiny.json")
    shud, bl = read_json(f"{EVAL}/shrota_shuddha.json"), read_json(f"{EVAL}/baselines.json")
    final = read_json(f"{EVAL}/metrics.json")

    gpu = sh("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw "
             "--format=csv,noheader,nounits").split(",")
    gpu = [g.strip() for g in gpu] if len(gpu) == 5 else ["-"] * 5
    jobs = [ln.split(None, 1)[1][:90] for ln in sh("ps -eo pid,args | grep -E 'scripts/(train|distill|eval|build|bench|gpu_experiments|complete_phase1)' | grep -v grep").splitlines() if ln.strip()]
    jobs = [j for j in jobs if not j.startswith(("bash -c", "/bin/sh"))]

    def pages():
        if not hasattr(pages, "c") or time.time() - pages.c[0] > 60:
            try:
                with urllib.request.urlopen("https://nikithamal.github.io/Shrota/metrics.json", timeout=3) as r:
                    pages.c = (time.time(), f"HTTP {r.status}")
            except Exception as e:
                pages.c = (time.time(), f"Unreachable ({getattr(e, 'code', 'net')})")
        return pages.c[1]

    def dom(m, d):
        v = (m.get("domains") or {}).get(d)
        return f"{v['wer']}% / {v['cer']}%" if v else "-"

    onnx = sh("ls -l /work/shrota/checkpoints/onnx 2>/dev/null | awk 'NF>8 {printf \"%s %.2f MB\\n\", $9, $5/1048576}'").splitlines()
    lm_mb = f"{os.path.getsize('/work/shrota/checkpoints/nepali_4gram_lm.json')/1048576:.2f} MB" if os.path.exists("/work/shrota/checkpoints/nepali_4gram_lm.json") else "pending"

    c1 = card("1  Data & Autopsy (v2)", status.get("phase1"), [
        ("Autopsy report", "eval/v1_autopsy.md" if os.path.exists("/work/shrota/eval/v1_autopsy.md") else "pending"),
        ("Gold suites", "FLEURS, SLR54, Conv, Vedic"),
        ("Global Fixed CMVN", "baked in ONNX" if os.path.exists("/work/shrota/checkpoints/mel_cmvn.json") else "pending"),
        ("4-gram Nepali LM", lm_mb),
        ("Whisper-small FLEURS WER", dom(bl.get("whisper_small", {}), "eval_fleurs_ne_np").split(" /")[0]),
        ("MMS-1b FLEURS WER", dom(bl.get("mms_1b_npi", {}), "eval_fleurs_ne_np").split(" /")[0])],
        tail([f"{LOGS}/phase1_v2b.log", f"{LOGS}/phase1_v2.log", f"{LOGS}/phase1_data.log"], 3))

    c2 = card("2  Shrota-Base v2 (4x + SC-InterCTC)", status.get("phase2"), [
        ("Params / INT8", f"{base.get('params_m', '-')} M / {base.get('size_int8_mb', '-')} MB"),
        ("1-thread RTF / Lat", f"{base.get('cpu_1thread_rtf', '-')} ({base.get('streaming_latency_ms', '-')} ms)"),
        ("SLR54 Gold WER / CER", dom(base, "eval_slr54_gold")),
        ("FLEURS WER / CER", dom(base, "eval_fleurs_ne_np")),
        ("Conv WER / CER", dom(base, "eval_conv_story")),
        ("Vedic WER / CER", dom(base, "eval_vedic_gold"))],
        tail([f"{LOGS}/train_base_v2_clean.log", f"{LOGS}/eval_base_v2.log", f"{LOGS}/train_base_v2.log", f"{LOGS}/train_base.log"], 3))

    b, a = shud.get("before_shuddha", {}), shud.get("after_shuddha", {})
    c3 = card("3  Shrota-Shuddha v2", status.get("phase3"), [
        ("INT8 size / Lat", f"{shud.get('size_int8_mb', '-')} MB / {shud.get('clause_latency_ms', '-')} ms"),
        ("Real Error WER", f"{b.get('wer', '-')}% -> {a.get('wer', '-')}%"),
        ("Punct F1", f"{b.get('punct_f1', '-')}% -> {a.get('punct_f1', '-')}%"),
        ("Hraswa-dirgha Acc", f"{b.get('hraswa_dirgha_acc', '-')}% -> {a.get('hraswa_dirgha_acc', '-')}%")],
        tail([f"{LOGS}/train_shuddha_v2.log", f"{LOGS}/train_shuddha.log"], 3))

    c4 = card("4  Shrota-Tiny v2 (KD)", status.get("phase4"), [
        ("Params / INT8", f"{tiny.get('params_m', '-')} M / {tiny.get('size_int8_mb', 'pending')} MB" if tiny else "pending"),
        ("1-thread RTF", f"{tiny.get('cpu_1thread_rtf', '-')}" if tiny else "-"),
        ("SLR54 Gold WER / CER", dom(tiny, "eval_slr54_gold") if tiny else "-"),
        ("FLEURS WER / CER", dom(tiny, "eval_fleurs_ne_np") if tiny else "-"),
        ("Vedic WER / CER", dom(tiny, "eval_vedic_gold") if tiny else "-")],
        tail([f"{LOGS}/distill_tiny_v2.log", f"{LOGS}/distill_tiny.log"], 4))

    c5 = card("5  Studio & Export (v2)", status.get("phase5", "Done" if final else "Running"), [
        ("Final metrics", "written" if final else "pending"), ("Studio page", pages()),
        ("ONNX bundles", str(len(onnx)))] + [(n.split()[0], " ".join(n.split()[1:])) for n in onnx[:6]], [])

    hdr = (f"<div style='display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:16px'>"
           f"<span style='font-size:22px;font-weight:600;margin-right:12px'>Shrota v2</span>"
           + "".join(f"<span style='background:#F4EFEA;padding:4px 12px;border-radius:999px;font-size:12px'>{esc(t)}</span>" for t in [
               f"GPU {gpu[0]}%", f"VRAM {gpu[1]} / {gpu[2]} MiB", f"{gpu[3]} C", f"{gpu[4]} W",
               time.strftime("%H:%M:%S UTC", time.gmtime())]) + "</div>")
    jobs_html = ("<div style='margin-top:16px;font-size:12px;color:#51443D'>Running: " + esc(" | ".join(jobs)) + "</div>") if jobs else ""

    grid = "display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px;font-family:Poppins,system-ui,sans-serif;color:#1D1B1A"
    mo.Html(f"<div style='font-family:Poppins,system-ui,sans-serif;color:#1D1B1A'>{hdr}<div style='{grid}'>{c1}{c2}{c3}{c4}{c5}</div>{jobs_html}</div>")
    return


if __name__ == "__main__":
    app.run()
