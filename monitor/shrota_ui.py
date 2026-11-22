import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import os
    import subprocess
    import time
    import marimo as mo
    return json, mo, os, subprocess, time


@app.cell
def _(mo):
    tick = mo.ui.refresh(options=["3s", "10s"], default_interval="3s")
    heartbeat = mo.ui.refresh(options=["60s"], default_interval="60s")
    mo.hstack([tick, heartbeat])
    return heartbeat, tick


@app.cell
def _(heartbeat, json, mo, os, subprocess, tick, time):
    _ = (tick.value, heartbeat.value)
    now_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    def read_json(p):
        try:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def tail_log(p, n=12):
        try:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                    return "".join(lines[-n:]).strip()
        except Exception:
            pass
        return "Waiting for job log..."

    gpu_info = "GPU Idle"
    try:
        p = subprocess.run(
            "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader",
            shell=True,
            text=True,
            capture_output=True,
            timeout=3,
        )
        if p.stdout.strip():
            gpu_info = p.stdout.strip()
    except Exception:
        pass

    status = read_json("/work/shrota/eval/status.json")
    baselines = read_json("/work/shrota/eval/baselines.json")
    metrics = read_json("/work/shrota/eval/metrics.json")

    c1 = mo.md(
        f"""### Card 1 — Phase 1: Data Engine & Gold Eval
- **Updated:** `{now_str}`
- **GPU:** `{gpu_info}`
- **Data Status:** `{status.get('phase1', 'Running / Ready')}`
- **Log Tail:**
```text
{tail_log('/work/shrota/logs/phase1_data.log', 6)}
```"""
    )

    c2 = mo.md(
        f"""### Card 2 — Phase 2: Shrota-Base (FastConformer TDT+CTC)
- **Stage:** `{status.get('phase2', 'Pending')}`
- **Log Tail:**
```text
{tail_log('/work/shrota/logs/train_base.log', 6)}
```"""
    )

    c3 = mo.md(
        f"""### Card 3 — Phase 3: Shrota-Shuddha (<2.5 MB Grammar & Hraswa-Dirgha)
- **Stage:** `{status.get('phase3', 'Pending')}`
- **Log Tail:**
```text
{tail_log('/work/shrota/logs/train_shuddha.log', 6)}
```"""
    )

    c4 = mo.md(
        f"""### Card 4 — Phase 4: Shrota-Tiny (<=4.5 MB Distilled Mobile/Browser ASR)
- **Stage:** `{status.get('phase4', 'Pending')}`
- **Log Tail:**
```text
{tail_log('/work/shrota/logs/distill_tiny.log', 6)}
```"""
    )

    c5 = mo.md(
        f"""### Card 5 — Phase 5: Benchmarks, ONNX Parity & Android/Browser Studio
- **Baselines Recorded:** `{bool(baselines)}`
- **Final Metrics Recorded:** `{bool(metrics)}`
- **Studio / Export Status:** `{status.get('phase5', 'In Progress')}`"""
    )

    mo.vstack([
        mo.md("# Shrota (श्रोता) v1 — Live Blackwell Training & Verification Monitor (BAD 0)"),
        mo.hstack([c1, c2], widths="equal"),
        mo.hstack([c3, c4], widths="equal"),
        c5,
    ])
    return


if __name__ == "__main__":
    app.run()
