#!/usr/bin/env python3
"""Get Oodle Texture's encoded BCn blocks out of a stock Unreal Engine install, using UE only as a user would.

Pipeline (no engine source, no direct DLL calls):
  1. Create a minimal project (OodleBench.uproject + Config/DefaultEngine.ini): Oodle Texture enabled for the
     editor, Final encode speed with RDO for both cooks and the editor, DDC kept inside the project
     (UE-LocalDataCachePath env var -> <project>/DerivedDataCache).
  2. UnrealEditor-Cmd -run=pythonscript ue_import.py: import every PNG once per (format, lambda) as
     /Game/Bench/<fmt>/<image>_<hash>_l<lambda> with no mips, sRGB off, never stream, no VT, the format's
     compression setting and LossyCompressionAmount for the lambda.
  3. UnrealEditor-Cmd -run=cook -targetplatform=Windows with -OodleDebugDumpFilter=*Bench*. The cook builds the
     textures through TextureFormatOodle, whose debug dump writes, per encoded surface,
     <project>/Saved/OodleDebugImages/<asset path>_<input fmt>_<W>x<H>_S0_IN.dds   (what Oodle was given)
     <project>/Saved/OodleDebugImages/<asset path>_RDO<l>_<BCn>_UNORM_<W>x<H>_S0_OUT.dds (what Oodle produced)
     (no "RDO<l>_" for lambda 0). Verbose LogTexture logs each build's DDC key, whose Oodle part
     (e.g. Oodle_CPF7_L30_E20_V2916) shows the lambda and SDK version actually used.
  4. Copy each OUT payload to <out>/<fmt>/<image>_l<lambda>.dds (DX10 header, mip 0) after checking the IN
     image against the PNG and that the cooked package (loose files, Saved/Cooked/Windows/.../*.uexp) contains
     the payload byte for byte, and merge rows into <out>/manifest.csv. When the input Oodle got is not the PNG
     (BC5: renormalized normal XY as R16G16), the IN dump is kept in <out>/<fmt>/inputs/.

The DDC caches builds: a texture whose cache key was already built is fetched, not re-encoded, and no dump is
written. ue_import.py gives every imported texture a fresh CompressionCacheId (a documented Texture property that
only changes the cache key), so every import is really encoded by the following cook. Old dumps of the assets
are deleted before the cook, and the manifest's "fresh" column confirms each dump was written by this cook.

Usage:
  python scripts/ue_oodle/run_ue_oodle.py --images test-images/oodle/kodim18_original.png \
      --formats bc1 bc7 bc5 bc4 --lambdas 0 1 10 20 30 40 --project <ue project dir> --out <dir>
"""
import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dds import level_bytes, write_dds  # noqa: E402

HERE = Path(__file__).resolve().parent
UE_ROOT = Path(r"C:\Program Files\Epic Games\UE_5.8")
UE_CMD = UE_ROOT / "Engine" / "Binaries" / "Win64" / "UnrealEditor-Cmd.exe"
CONTENT = "/Game/Bench"
DXGI_OUT = {"bc1": "BC1_UNORM", "bc4": "BC4_UNORM", "bc5": "BC5_UNORM", "bc7": "BC7_UNORM"}

UPROJECT = {
    "FileVersion": 3,
    "EngineAssociation": "5.8",
    "Category": "",
    "Description": "Oodle Texture extraction harness",
    "Plugins": [
        {"Name": "PythonScriptPlugin", "Enabled": True},
        {"Name": "EditorScriptingUtilities", "Enabled": True},
        {"Name": "TextureFormatOodle", "Enabled": True},
    ],
}

DEFAULT_ENGINE_INI = """\
[AlternateTextureCompression]
TextureCompressionFormat="TextureFormatOodle"
TextureFormatPrefix="OODLE_"
bEnableInEditor=True

[TextureFormatOodle]
DefaultRDOLambda=30
GlobalLambdaMultiplier=1.0

; Project Settings > Engine > Texture Encoding. Stock defaults have RDO off for both speeds, so the
; per-texture LossyCompressionAmount would be ignored; only RDO is switched on, effort/tiling stay default.
[/Script/Engine.TextureEncodingProjectSettings]
bFinalUsesRDO=True
CookUsesSpeed=Final
EditorUsesSpeed=Final
"""

# Loose cooked files instead of the Zen store, so the shipped bytes can be checked against the dump.
DEFAULT_GAME_INI = """[/Script/UnrealEd.ProjectPackagingSettings]
bUseZenStore=False
"""


def log(*a):
    print("[ue_oodle]", *a, flush=True)


def make_project(proj: Path):
    (proj / "Config").mkdir(parents=True, exist_ok=True)
    (proj / "OodleBench.uproject").write_text(json.dumps(UPROJECT, indent=1))
    (proj / "Config" / "DefaultEngine.ini").write_text(DEFAULT_ENGINE_INI)
    (proj / "Config" / "DefaultGame.ini").write_text(DEFAULT_GAME_INI)


def run_ue(proj: Path, args, logfile: Path):
    env = dict(os.environ)
    env["UE-LocalDataCachePath"] = str(proj / "DerivedDataCache")
    cmd = [str(UE_CMD), str(proj / "OodleBench.uproject"), *args,
           "-unattended", "-nosplash", "-nullrhi", "-nop4", "-stdout", "-FullStdOutLogOutput", "-ddc=NoShared"]
    log("running:", " ".join(cmd[1:]))
    t0 = time.time()
    child_cpu = {}
    with open(logfile, "w", encoding="utf-8", errors="replace") as f:
        r = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        # Child processes' CPU time (shader workers and the like), sampled until each exits.
        try:
            import psutil
            me = psutil.Process(r.pid)
            while r.poll() is None:
                try:  # UE itself may exit between the poll and the listing
                    kids = me.children(recursive=True)
                except psutil.Error:
                    break
                for c in kids:
                    try:
                        t = c.cpu_times()
                        child_cpu[c.pid] = t.user + t.system
                    except psutil.Error:
                        pass
                time.sleep(0.2)
        except ImportError:
            pass
        r.wait()
    dt = time.time() - t0
    cpu = process_cpu_s(r) + sum(child_cpu.values())
    log(f"  exit {r.returncode} after {dt:.0f}s ({cpu:.1f} CPU s), log {logfile}")
    if r.returncode != 0:
        tail = logfile.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        sys.exit("UE failed:\n" + "\n".join(tail))
    return dt, cpu


def process_cpu_s(p: subprocess.Popen) -> float:
    """User + kernel CPU seconds of an exited Popen process (Windows: GetProcessTimes on its handle)."""
    if os.name != "nt":
        return 0.0
    import ctypes
    from ctypes import wintypes
    ft = [wintypes.FILETIME() for _ in range(4)]
    if not ctypes.windll.kernel32.GetProcessTimes(wintypes.HANDLE(int(p._handle)), *map(ctypes.byref, ft)):
        return 0.0
    return sum((t.dwHighDateTime << 32 | t.dwLowDateTime) / 1e7 for t in ft[2:])


def parse_ts(ts: str):
    """UE log timestamp 2026.11.09-19.22.53:912 -> seconds."""
    return time.mktime(time.strptime(ts[:19], "%Y.%m.%d-%H.%M.%S")) + int(ts[20:]) / 1000.0


def short_hash(path: Path):
    return hashlib.sha1(path.read_bytes()).hexdigest()[:8]


def read_dds_payload(path: Path):
    b = path.read_bytes()
    assert b[:4] == b"DDS "
    h, w = int.from_bytes(b[12:16], "little"), int.from_bytes(b[16:20], "little")
    off = 128 + (20 if b[84:88] == b"DX10" else 0)
    dxgi = int.from_bytes(b[128:132], "little") if b[84:88] == b"DX10" else None
    return w, h, dxgi, b[off:]


def check_input(in_dds: Path, png: Path, fmt: str):
    """Compare the image Oodle was given against the PNG; returns a short note."""
    w, h, dxgi, data = read_dds_payload(in_dds)
    src = np.asarray(Image.open(png).convert("RGBA"))
    if (src.shape[1], src.shape[0]) != (w, h):
        return f"SIZE {w}x{h} != source {src.shape[1]}x{src.shape[0]}"
    if dxgi in (87, 88):  # B8G8R8A8 / B8G8R8X8
        img = np.frombuffer(data, np.uint8)[: w * h * 4].reshape(h, w, 4)[:, :, [2, 1, 0, 3]]
        chans = {"bc4": [0], "bc5": [0, 1]}.get(fmt, [0, 1, 2])
        d = np.abs(img[:, :, chans].astype(int) - src[:, :, chans].astype(int)).max()
        return "identical" if d == 0 else f"differs (max {d})"
    if dxgi == 28:  # R8G8B8A8
        img = np.frombuffer(data, np.uint8)[: w * h * 4].reshape(h, w, 4)
        chans = {"bc4": [0], "bc5": [0, 1]}.get(fmt, [0, 1, 2])
        d = np.abs(img[:, :, chans].astype(int) - src[:, :, chans].astype(int)).max()
        return "identical" if d == 0 else f"differs (max {d})"
    if dxgi == 61:  # R8
        img = np.frombuffer(data, np.uint8)[: w * h].reshape(h, w)
        d = np.abs(img.astype(int) - src[:, :, 0].astype(int)).max()
        return "identical(R)" if d == 0 else f"differs R (max {d})"
    if dxgi == 35:  # R16G16 (BC4 from TC_Alpha, BC5 from TC_Normalmap)
        img = np.frombuffer(data, np.uint16)[: w * h * 2].reshape(h, w, 2).astype(float)
        if fmt == "bc4":
            d = np.abs(img[:, :, 0] - src[:, :, 0] * 257.0).max()
            return "identical(R as 16-bit R*257)" if d == 0 else f"differs R*257 (max {d:.0f})"
        # normal map: XY of normalize(rgb/255*2-1), remapped to 0..65535
        n = src[:, :, :3] / 255.0 * 2 - 1
        u = n / np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-9)
        d = np.abs((u[:, :, :2] + 1) / 2 * 65535 - img).max()
        return f"renormalized normal XY 16-bit (max dev {d:.2f}/65535)"
    return f"input dxgi {dxgi} not compared"


def save_input(in_dds: Path, dst_dir: Path, stem: str):
    """Keep what Oodle was given: the raw IN dump, plus an 8-bit PNG view when the input is not 8-bit RGB."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{stem}_IN.dds"
    if not dst.exists():
        dst.write_bytes(in_dds.read_bytes())
    w, h, dxgi, data = read_dds_payload(in_dds)
    if dxgi == 35:
        img = np.frombuffer(data, np.uint16)[: w * h * 2].reshape(h, w, 2).astype(float)
        rgb = np.zeros((h, w, 3), np.uint8)
        rgb[:, :, :2] = np.round(img / 257.0).astype(np.uint8)
        Image.fromarray(rgb).save(dst_dir / f"{stem}_IN_8bit_view.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--formats", nargs="+", default=["bc1", "bc7"], choices=["bc1", "bc4", "bc5", "bc7"])
    ap.add_argument("--lambdas", nargs="+", type=int, default=[0, 1, 10, 20, 30, 40])
    ap.add_argument("--project", type=Path, required=True, help="UE project dir (created if missing)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-import", action="store_true")
    ap.add_argument("--skip-cook", action="store_true")
    ap.add_argument("--cook-args", nargs="*", default=[], help='extra cook switches, e.g. -LogCmds="LogTextureFormatOodle Verbose"')
    a = ap.parse_args()

    proj = a.project.resolve()
    make_project(proj)
    logs = proj / "Saved" / "HarnessLogs"
    logs.mkdir(parents=True, exist_ok=True)

    images = [Path(p).resolve() for p in a.images]
    names = {p: f"{p.stem}_{short_hash(p)}" for p in images}
    job = {"images": [str(p) for p in images], "names": {str(p): n for p, n in names.items()},
           "formats": a.formats, "lambdas": a.lambdas, "content_path": CONTENT}
    job_path = logs / "job.json"
    job_path.write_text(json.dumps(job, indent=1))

    timing = {}
    if not a.skip_import:
        os.environ["UE_OODLE_JOB"] = str(job_path)
        timing["import_s"], timing["import_cpu_s"] = run_ue(proj, ["-run=pythonscript", f"-script={HERE / 'ue_import.py'}"],
                                    logs / "import.log")
    dump = proj / "Saved" / "OodleDebugImages"
    cook_start = time.time()
    if not a.skip_cook:
        for fmt in a.formats:  # stale dumps of these assets would otherwise mask a missing encode
            for p in images:
                for lam in a.lambdas:
                    for f in dump.glob(f"{CONTENT}/{fmt}/{names[p]}_l{lam}".replace("/", "_") + ".*"):
                        f.unlink()
        timing["cook_s"], timing["cook_cpu_s"] = run_ue(proj, ["-run=cook", "-targetplatform=Windows",
                                         f"-cookdir={proj / 'Content' / 'Bench'}",
                                         "-OodleDebugDumpFilter=*Bench*",
                                         # Verbose LogTexture prints "Storing texture in DDC" with the cache key
                                         # (format, lambda, effort, SDK version) after each build.
                                         "-LogCmds=LogTexture Verbose, LogTextureFormatOodle Verbose",
                                         *a.cook_args], logs / "cook.log")
    # with --skip-cook, the previous cook's log is parsed
    cook_log = (logs / "cook.log").read_text(encoding="utf-8", errors="replace") if (logs / "cook.log").exists() else ""

    # "Building texture TwoD: /Game/Bench/bc7/x_l30.x_l30 (TFO_BC7, 512x512 ...)" lines, with timestamps
    built = {}
    for m in re.finditer(r"\[(\d{4}\.\d\d\.\d\d-\d\d\.\d\d\.\d\d:\d{3})\]\[\s*\d+\]LogTexture: Display: Building "
                         r"texture TwoD: (/Game/Bench/\S+?)\.\S+ \((\S+), (\d+)x(\d+)", cook_log):
        built[m.group(2)] = {"ts": m.group(1), "ue_format": m.group(3), "built_size": f"{m.group(4)}x{m.group(5)}"}
    # "LogTexture: Verbose: Storing texture in DDC:" / "  Name: /Game/..." / "  Key: TEXTURE_..._Oodle_CPF7_L30_E20_V2916..."
    for m in re.finditer(r"\[(\d{4}\.\d\d\.\d\d-\d\d\.\d\d\.\d\d:\d{3})\]\[\s*\d+\]LogTexture: Verbose: Storing "
                         r"texture in DDC:\s*\n\s*Name: (/Game/Bench/\S+?)\.\S+\s*\n\s*Key: (\S+)", cook_log):
        b = built.setdefault(m.group(2), {})
        b["stored_ts"] = m.group(1)
        if b.get("ts"):
            b["build_to_store_s"] = f"{(parse_ts(m.group(1)) - parse_ts(b['ts'])):.3f}"
        k = re.search(r"Oodle_CPF\d+_L\d+_E\d+_V\d+?(?=0000803F)", m.group(3)) or re.search(r"Oodle_\w{0,40}", m.group(3))
        b["ddc_key_oodle"] = k.group(0) if k else ""
    versions = sorted(set(re.findall(r"Oodle Texture loading DLL: (\S+)", cook_log)))
    speed = re.findall(r"Oodle Texture Encode Speed settings: (.*)", cook_log)

    rows = []
    for fmt in a.formats:
        (a.out / fmt).mkdir(parents=True, exist_ok=True)
        for p in images:
            for lam in a.lambdas:
                asset = f"{CONTENT}/{fmt}/{names[p]}_l{lam}"
                prefix = asset.replace("/", "_") + "." + f"{names[p]}_l{lam}" + "_"
                outs = sorted(dump.glob(prefix + f"*{DXGI_OUT[fmt]}_*_S0_OUT.dds"))
                ins = sorted(dump.glob(prefix + "*_S0_IN.dds"))
                row = {"fmt": fmt, "image": p.stem, "lambda": lam, "asset": asset}
                row.update(built.get(asset, {"ts": "", "ue_format": "(not built: DDC hit?)", "built_size": ""}))
                if not outs:
                    row["status"] = "NO DUMP"
                    rows.append(row)
                    log("missing dump for", asset)
                    continue
                out = outs[-1]
                rdo = re.search(r"_RDO(\d+)_", out.name[len(prefix) - 1:])
                row["dump_rdo"] = rdo.group(1) if rdo else ""
                row["fresh"] = out.stat().st_mtime >= cook_start - 1 if not a.skip_cook else ""
                w, h, dxgi, payload = read_dds_payload(out)
                n = level_bytes(fmt, w, h)
                dst = a.out / fmt / f"{p.stem}_l{lam}.dds"
                write_dds(dst, fmt, w, h, payload[:n])
                # the cooked package (what the game ships) must contain exactly these bytes
                cooked = proj / "Saved" / "Cooked" / "Windows" / "OodleBench" / "Content" / "Bench" / fmt
                row["in_cooked"] = "no cooked file"
                for ext in (".uexp", ".ubulk"):
                    f = cooked / f"{names[p]}_l{lam}{ext}"
                    if f.exists():
                        pos = f.read_bytes().find(payload[:n])
                        row["in_cooked"] = f"yes {ext}@{pos}" if pos >= 0 else "NOT FOUND"
                        if pos >= 0:
                            break
                note = check_input(ins[0], p, fmt) if ins else "no IN dump"
                if ins and not note.startswith("identical"):
                    save_input(ins[0], a.out / fmt / "inputs", p.stem)
                row.update({"w": w, "h": h, "bytes": n, "extra_bytes": len(payload) - n,
                            "input_check": note, "dds": str(dst), "status": "ok"})
                rows.append(row)
    # ts = "Building texture" log time, stored_ts = "Storing texture in DDC" log time. build_to_store_s is the
    # wall time between them: UE overhead included, and the cook builds several textures concurrently.
    keys = ["fmt", "image", "lambda", "status", "w", "h", "bytes", "extra_bytes", "dump_rdo", "ue_format",
            "built_size", "ddc_key_oodle", "input_check", "in_cooked", "fresh", "ts", "stored_ts",
            "build_to_store_s", "asset", "dds"]
    a.out.mkdir(parents=True, exist_ok=True)
    # merge with an existing manifest (rows of this run replace same fmt/image/lambda)
    man = a.out / "manifest.csv"
    merged = {}
    if man.exists():
        for r in csv.DictReader(open(man, newline="")):
            merged[(r["fmt"], r["image"], str(r["lambda"]))] = r
    for r in rows:
        merged[(r["fmt"], r["image"], str(r["lambda"]))] = r
    order = {"bc1": 0, "bc7": 1, "bc5": 2, "bc4": 3}
    with open(man, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(sorted(merged.values(), key=lambda r: (order[r["fmt"]], r["image"], int(r["lambda"]))))
    meta = {"args": sys.argv[1:], "oodle_dlls_loaded_during_cook": versions, "encode_speed_settings": speed,
            "timing": timing}
    info = a.out / "run_info.json"
    runs = json.loads(info.read_text()) if info.exists() else []
    (info).write_text(json.dumps((runs if isinstance(runs, list) else [runs]) + [meta], indent=1))
    ok = sum(r["status"] == "ok" for r in rows)
    log(f"{ok}/{len(rows)} written to {a.out}; {meta}")


if __name__ == "__main__":
    main()
