#!/usr/bin/env python3
"""Minimal MoLab remote execution & file transfer helper."""
import argparse
import base64
import http.cookiejar
import json
import os
import re
import sys
import urllib.request
import urllib.error

MOLAB_URL = os.environ.get("MOLAB_URL", "https://sb-909826ab46bc5ba6.sb.molab.run").rstrip("/")
TOKEN_PATH = "/tmp/molab_token"


def get_token() -> str:
    if os.environ.get("MOLAB_TOKEN"):
        return os.environ["MOLAB_TOKEN"].strip()
    with open(TOKEN_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()


class MolabClient:
    def __init__(self):
        self.tok = get_token()
        self.cj = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.cj))
        self.srv_tok = None
        self.session_id = None
        self._init_session()

    def _init_session(self):
        req = urllib.request.Request(
            f"{MOLAB_URL}/?access_token={self.tok}",
            headers={"Authorization": f"Bearer {self.tok}", "User-Agent": "Mozilla/5.0"},
        )
        html = self.opener.open(req, timeout=30).read().decode("utf-8", errors="replace")
        m = re.search(r'<marimo-server-token data-token="([^"]+)"', html)
        self.srv_tok = m.group(1) if m else ""

        req_nb = urllib.request.Request(
            f"{MOLAB_URL}/api/home/running_notebooks",
            data=b"{}",
            headers={
                "Authorization": f"Bearer {self.tok}",
                "Marimo-Server-Token": self.srv_tok,
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0",
            },
            method="POST",
        )
        nb_data = json.loads(self.opener.open(req_nb, timeout=30).read().decode("utf-8"))
        files = nb_data.get("files", [])
        if files and files[0].get("sessionId"):
            self.session_id = files[0]["sessionId"]
        else:
            self.session_id = "s_default"

    def api_post(self, endpoint: str, payload: dict, timeout: int = 120) -> dict:
        url = f"{MOLAB_URL}{endpoint}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.tok}",
                "Marimo-Server-Token": self.srv_tok,
                "Marimo-Session-Id": self.session_id,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0",
            },
            method="POST",
        )
        with self.opener.open(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return json.loads(raw)
            except Exception:
                return {"raw": raw}


_CLIENT = None


def get_client() -> MolabClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = MolabClient()
    return _CLIENT


def run_code(code: str, timeout: int = 120) -> dict:
    return get_client().api_post("/api/kernel/execute", {"code": code}, timeout=timeout)


def parse_sse(raw: str) -> str:
    out = []
    for line in raw.splitlines():
        if line.startswith("data: "):
            try:
                obj = json.loads(line[6:])
                if isinstance(obj, dict):
                    if isinstance(obj.get("data"), str) and obj.get("data"):
                        out.append(obj["data"])
                    elif isinstance(obj.get("output"), dict) and obj["output"].get("data"):
                        out.append(str(obj["output"]["data"]))
                    elif obj.get("error"):
                        out.append(f"[ERROR] {obj['error']}")
            except Exception:
                pass
    return "".join(out) if out else raw


def run_bash(cmd: str, timeout: int = 120) -> str:
    py_code = (
        "import subprocess, sys\n"
        f"p = subprocess.run({cmd!r}, shell=True, text=True, capture_output=True)\n"
        "if p.stdout: sys.stdout.write(p.stdout)\n"
        "if p.stderr: sys.stderr.write(p.stderr)\n"
        "if p.returncode != 0: print(f'[EXIT {p.returncode}]')\n"
    )
    res = run_code(py_code, timeout=timeout)
    if isinstance(res, dict) and "raw" in res:
        return parse_sse(res["raw"]).rstrip()
    return json.dumps(res)


def upload_file(local_path: str, remote_path: str, mode: int = 0o644) -> str:
    with open(local_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    py_code = (
        "import base64, os\n"
        f"rp = {remote_path!r}\n"
        "os.makedirs(os.path.dirname(rp) or '/', exist_ok=True)\n"
        f"with open(rp, 'wb') as f: f.write(base64.b64decode({b64!r}))\n"
        f"os.chmod(rp, {mode})\n"
        "print(f'Uploaded {rp} ({os.path.getsize(rp)} bytes)')\n"
    )
    res = run_code(py_code, timeout=180)
    return json.dumps(res)


def download_file(remote_path: str, local_path: str) -> str:
    py_code = (
        "import base64\n"
        f"with open({remote_path!r}, 'rb') as f:\n"
        "    print('B64_START:' + base64.b64encode(f.read()).decode('ascii') + ':B64_END')\n"
    )
    res = run_code(py_code, timeout=300)
    raw = json.dumps(res)
    if "B64_START:" not in raw:
        raise RuntimeError(f"Failed to download {remote_path}: {raw[:400]}")
    b64 = raw.split("B64_START:", 1)[1].split(":B64_END", 1)[0]
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    data = base64.b64decode(b64)
    with open(local_path, "wb") as f:
        f.write(data)
    return f"Downloaded {local_path} ({len(data)} bytes)"


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="sub")
    p_sh = sub.add_parser("sh")
    p_sh.add_argument("cmd")
    p_sh.add_argument("--timeout", type=int, default=120)
    p_py = sub.add_parser("py")
    p_py.add_argument("code")
    p_py.add_argument("--timeout", type=int, default=120)
    p_up = sub.add_parser("up")
    p_up.add_argument("local")
    p_up.add_argument("remote")
    p_up.add_argument("--mode", type=lambda x: int(x, 8), default=0o644)
    p_dn = sub.add_parser("down")
    p_dn.add_argument("remote")
    p_dn.add_argument("local")
    args = parser.parse_args()
    if args.sub == "sh":
        print(run_bash(args.cmd, timeout=args.timeout))
    elif args.sub == "py":
        print(json.dumps(run_code(args.code, timeout=args.timeout), indent=2))
    elif args.sub == "up":
        print(upload_file(args.local, args.remote, mode=args.mode))
    elif args.sub == "down":
        print(download_file(args.remote, args.local))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
