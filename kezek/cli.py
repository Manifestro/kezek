"""kezek-separate input.wav [more.wav ...] --out DIR: writes <name>_A.wav, <name>_B.wav and <name>.json."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import soundfile as sf


def read16k(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != 16_000:
        import torch
        import torchaudio.functional as AF

        x = AF.resample(torch.from_numpy(x), sr, 16_000).numpy()
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=None, help="Hugging Face model repository (default: manifestro/kezek-v1)")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    a = ap.parse_args()
    import torch

    from kezek import Kezek

    a.device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    kz = Kezek.from_pretrained(**({"repo": a.repo} if a.repo else {}), dev=a.device)
    os.makedirs(a.out, exist_ok=True)
    for path in a.inputs:
        x = read16k(path)
        peak = max(float(np.abs(x).max()), 1e-4)
        y, act, info = kz((x * 0.9 / peak).astype(np.float32))
        name = os.path.splitext(os.path.basename(path))[0]
        for k, t in zip("AB", y, strict=True):
            sf.write(os.path.join(a.out, f"{name}_{k}.wav"), t * peak / 0.9, 16_000)
        json.dump(info, open(os.path.join(a.out, f"{name}.json"), "w"), indent=1)
        print(f"{path}: {info['seconds']:.1f} s, A {info['speech_s']['A']:.1f} s, B {info['speech_s']['B']:.1f} s, "
              f"overlap {info['overlap_s']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
