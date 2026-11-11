"""Kezek v1: one mono recording of two people -> one track per person, time-aligned with the input.

    stage 1    PairNet: raw audio -> who speaks when (20 ms frames) for the two voices of the recording
    profiles   the voice fingerprint (SpeakerNet) of up to 30 s of each person's solo speech
    separator  KezekTSE-L: mixture + stage 1 activity + the two profiles -> two tracks summing to the input

Usage:

    from kezek import Kezek
    kz = Kezek.from_pretrained()               # weights from the Hugging Face model repository
    tracks, activity, info = kz(audio_16k)     # audio: float32 [T] at 16 kHz, mono

This is the research pipeline of Kezek v1 (ke-zek KZK-43 + KZK-33 + KZK-35) with its default settings:
threshold 0.3, voices found by the fingerprint, one refinement pass, the separator over the whole
recording in 8 s windows with 6 s hop.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .pairnet import PairNet
from .rawenc import SpeakerNet
from .separator import KezekTSEL

SR, F20 = 16_000, 320
REPO = "manifestro/kezek-v1"
FILES = {"stage1": "stage1_pairnet.pt", "fingerprint": "fingerprint.pt", "separator": "separator.pt"}


def _spans(mask, min_len):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True)
            if b - a >= min_len]


def profiles(x, act, spk, dev):
    """Fingerprint of up to 30 s of each person's solo speech spread over the recording."""
    prof = []
    for s in (0, 1):
        fr = np.flatnonzero(act[s] & ~act[1 - s])
        if len(fr) < 50:
            fr = np.flatnonzero(act[s])
        if len(fr) == 0:
            prof.append(None)
            continue
        fr = fr[np.linspace(0, len(fr) - 1, min(len(fr), 30 * 50)).astype(int)]
        idx = (fr[:, None] * F20 + np.arange(F20)[None]).ravel()
        idx = idx[idx < len(x)]
        prof.append(spk(torch.from_numpy(np.ascontiguousarray(x[idx]))[None].to(dev).float()).float()[0])
    if prof[0] is None or prof[1] is None:  # one voice only: the other profile is a far copy
        k = 0 if prof[0] is not None else 1
        prof[1 - k] = -prof[k]
    return prof


@torch.no_grad()
def separate(x, act, prof, sep, dev, win_s=8.0, hop_s=6.0, bs=32):
    """The separator over overlapping windows with linear cross-fades -> [2, n]."""
    n = len(x)
    win, hop = int(win_s * SR), int(hop_s * SR)
    starts = list(range(0, max(1, n - win + hop), hop))
    hint_full = act.astype(np.float32)
    y = np.zeros((2, n), np.float32)
    wsum = np.zeros(n, np.float32)
    fade = win - hop
    ramp = np.ones(win, np.float32)
    ramp[:fade] = np.linspace(0, 1, fade)
    ramp[-fade:] = np.linspace(1, 0, fade)
    pa, pb = prof[0][None].to(dev), prof[1][None].to(dev)
    for i in range(0, len(starts), bs):
        chunk = starts[i: i + bs]
        segs, hints = [], []
        for s in chunk:
            seg = np.zeros(win, np.float32)
            seg[: min(win, n - s)] = x[s: s + win]
            segs.append(seg)
            fr = np.minimum((s + np.arange(win // 128 + 1) * 128) // F20, hint_full.shape[1] - 1)
            hints.append(hint_full[:, fr])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev != "cpu"):
            out = sep(torch.from_numpy(np.stack(segs)).to(dev), torch.from_numpy(np.stack(hints)).to(dev),
                      pa.expand(len(chunk), -1), pb.expand(len(chunk), -1)).float().cpu().numpy()
        for s, o in zip(chunk, out, strict=True):
            L = min(win, n - s)
            r = np.maximum(ramp[:L], 1e-3)
            y[:, s: s + L] += o[:, :L] * r
            wsum[s: s + L] += r
    ok = wsum > 1e-6
    y[:, ok] /= wsum[ok]
    return y


class Kezek:
    def __init__(self, stage1, fingerprint, separator, dev="cuda", threshold=0.3, refine=1):
        d = torch.load(stage1, map_location="cpu", weights_only=False)
        self.pn = PairNet(d.get("config", "S"), front=d.get("front", "conv"), fp=d.get("fp", False),
                          mel=d.get("mel", False), ssl=d.get("ssl", False)).to(dev).eval()
        self.pn.load_state_dict(d["model"])
        d = torch.load(fingerprint, map_location="cpu", weights_only=False)
        self.spk = SpeakerNet(d["config"], d.get("front", "conv")).to(dev).eval()
        self.spk.load_state_dict(d["model"])
        sd = torch.load(separator, map_location="cpu", weights_only=False)
        sd = sd["model"] if "model" in sd else sd
        self.sep = KezekTSEL("L").to(dev).eval()
        self.sep.load_state_dict(sd)
        self.dev, self.thr, self.refine = dev, threshold, refine

    @classmethod
    def from_pretrained(cls, repo=REPO, revision=None, dev=None, **kw):
        from huggingface_hub import hf_hub_download

        dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
        paths = {k: hf_hub_download(repo, f, revision=revision) for k, f in FILES.items()}
        return cls(paths["stage1"], paths["fingerprint"], paths["separator"], dev=dev, **kw)

    @torch.no_grad()
    def _spk(self, wav):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev != "cpu"):
            return self.spk(wav.float()).float()

    @torch.no_grad()
    def __call__(self, x):
        """x: float32 [T], 16 kHz mono -> (tracks [2, T], activity [2, frames of 20 ms] bool, info)."""
        x = np.ascontiguousarray(np.asarray(x, np.float32))
        p, _ = self.pn.infer(torch.from_numpy(x).to(self.dev), refine=self.refine, thr=self.thr, spk=self.spk)
        p = p.cpu().numpy()
        act = p > self.thr
        prof = profiles(x, act, self._spk, self.dev)
        y = separate(x, act, prof, self.sep, self.dev)
        both = act[0] & act[1]
        info = {
            "seconds": len(x) / SR,
            "speech_s": {"A": float(act[0].sum() * 0.02), "B": float(act[1].sum() * 0.02)},
            "overlap_s": float(both.sum() * 0.02),
            "profile_cosine": float(F.cosine_similarity(prof[0][None], prof[1][None]).item()),
            "uncertain_spans": [[round(s0 * 0.02, 2), round(e0 * 0.02, 2)]
                                for s0, e0 in _spans((np.abs(p - 0.5) < 0.2).any(0), 25)],
        }
        return y, act, info
