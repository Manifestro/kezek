"""Version 1 stage 1, "who speaks when" for exactly two people: PairNet.

    waveform ─► RawEncoder (from the KZK-25 fingerprint, chunks of 10 s with 2 s context)
             ─► frames h [F, D] ─┬─► voice z_t (unit vector)        ─┐
                                 ├─► speech / overlap per frame      ─┼─► two voices of the
                                 │                                    │   recording: S_A, S_B
                                 └─────────────► decoder per voice ◄──┘   (spectral split + soft
                                                 (shared weights)          k-means over the WHOLE
                                                      │                    recording)
                                                      ▼
                                          p_A(t), p_B(t) on the 20 ms grid

What is not standard here:

* No windows to stitch. The two voices are found once for the whole recording: the frames
  where one person speaks are split in two along the first principal direction of their voice
  vectors (two people -> two clusters -> the largest spread lies between them), then refined by a
  few soft k-means steps. Everything after that is local and refers to the same two voices, so
  A cannot turn into B half way through: the prototype's swaps came from linking 16 s windows.
* The comparison is inside the recording, not against the world: the voices are the two
  centres of this recording's own frames, so the decoder only has to tell these two apart
  (a relative, per-recording contrast), not place them among thousands of people.
* One decoder, shared by both voices and run once per voice with (own voice, other voice,
  cosine of every frame to each): swapping the voices swaps the outputs exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rawenc import HOP, RawEncoder

SR = 16_000


def _windows(n, chunk, ctx):
    """Chunks of ``chunk`` frames, each inside a window of chunk + 2 * ctx frames that lies
    wholly within [0, n) (moved inwards at the edges), so every window has the same length and
    a batch of them needs no padding: [(window start, chunk start, chunk length)]."""
    W = chunk + 2 * ctx
    return [(min(max(0, s - ctx), n - W), s, min(chunk, n - s)) for s in range(0, n, chunk)], W


def chunked(fn, x, chunk, ctx, bs=64):
    """fn over frames [B, F, C] in windows of chunk + 2 * ctx frames (batched); keeps the chunks."""
    B, Fn = x.shape[:2]
    wins, W = _windows(Fn, chunk, ctx)
    if Fn <= W:
        return fn(x)
    xs = torch.stack([x[:, a : a + W] for a, _, _ in wins], 1).flatten(0, 1)  # [B * K, W, C]
    ys = torch.cat([fn(xs[i : i + bs]) for i in range(0, len(xs), bs)]).view(B, len(wins), W, -1)
    return torch.cat([ys[:, k, s - a : s - a + L] for k, (a, s, L) in enumerate(wins)], 1)


def encode_wave(enc, wav, chunk_s=10.0, ctx_s=2.0, bs=64):
    """RawEncoder over a long waveform in windows of chunk_s + 2 ctx_s (batched) -> [B, F, D]."""
    n = wav.shape[-1] // HOP
    c, k = int(chunk_s * SR / HOP), int(ctx_s * SR / HOP)
    wins, W = _windows(n, c, k)
    if n <= W:
        return enc(wav[:, : n * HOP])
    B = wav.shape[0]
    xs = torch.stack([wav[:, a * HOP : (a + W) * HOP] for a, _, _ in wins], 1).flatten(0, 1)
    hs = torch.cat([enc(xs[i : i + bs]) for i in range(0, len(xs), bs)])
    hs = hs.view(B, len(wins), hs.shape[1], -1)
    return torch.cat([hs[:, j, s - a : s - a + L] for j, (a, s, L) in enumerate(wins)], 1)


def _ssl_feats(enc, wav, k=4):
    """Instance-normalised mean of the last k Conformer layers (the pretraining targets' form)."""
    rms = wav.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-4)
    x = enc.front(wav / rms * 0.1).transpose(0, 1)
    outs = []
    for lay in enc.conf.conformer_layers:
        x = lay(x, None)
        outs.append(x.transpose(0, 1))
    t = 0
    for o in outs[-k:]:
        o = o.float()
        t = t + (o - o.mean(1, keepdim=True)) / (o.std(1, keepdim=True) + 1e-5)
    return F.layer_norm(t / k, t.shape[-1:])


def two_voices(z, w, iters=6, tau=10.0):
    """z [B, F, Z] unit voice vectors, w [B, F] weights (frames where one person speaks) ->
    slots [B, 2, Z]. Spectral split along the first principal direction, then soft k-means."""
    B, Fn, Zd = z.shape
    wn = w / w.sum(1, keepdim=True).clamp_min(1e-6)
    mu = (wn.unsqueeze(-1) * z).sum(1, keepdim=True)
    zc = z - mu
    C = torch.einsum("bf,bfi,bfj->bij", wn, zc, zc)
    u = torch.randn(B, Zd, 1, device=z.device, generator=torch.Generator(device=z.device).manual_seed(0))
    for _ in range(20):
        u = F.normalize(C @ u, dim=1)
    proj = (zc @ u).squeeze(-1)  # [B, F]
    r = torch.sigmoid(proj / (proj.abs().mean(1, keepdim=True) + 1e-6) * 3)
    resp = torch.stack([r, 1 - r], -1)  # [B, F, 2]
    for _ in range(iters):
        s = F.normalize(torch.einsum("bf,bfk,bfz->bkz", w, resp, z), dim=-1)
        resp = torch.softmax(tau * torch.einsum("bfz,bkz->bfk", z, s), -1)
    return s


class Decoder(nn.Module):
    def __init__(self, dim, zdim, layers=2, heads=4, extra=0):
        super().__init__()
        self.extra = extra
        self.inp = nn.Linear(dim + 2 * zdim + 4 + extra, dim)
        lay = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.1, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(lay, layers)
        self.out = nn.Linear(dim, 1)

    def forward(self, h, z, s_own, s_oth, sp, ov, f_own=None, f_oth=None):
        B, Fn, _ = h.shape
        c_own = (z * s_own[:, None]).sum(-1, keepdim=True)
        c_oth = (z * s_oth[:, None]).sum(-1, keepdim=True)
        parts = [h, s_own[:, None].expand(B, Fn, -1), s_oth[:, None].expand(B, Fn, -1),
                 c_own, c_oth, sp.unsqueeze(-1), ov.unsqueeze(-1)]
        if self.extra:  # fingerprint: cosine of the frame's fingerprint to each voice of the recording
            zero = torch.zeros_like(sp)
            parts += [(f_own if f_own is not None else zero).unsqueeze(-1), (f_oth if f_oth is not None else zero).unsqueeze(-1)]
        x = torch.cat(parts, -1)
        x = self.inp(x.to(h.dtype))
        x = chunked(self.tr, x, 500, 100)  # 10 s chunks, 2 s context
        return self.out(x).squeeze(-1)


class PairNet(nn.Module):
    def __init__(self, enc_config="S", zdim=128, dec_layers=2, front="conv", fp=False, mel=False, ssl=False):
        super().__init__()
        self.enc = RawEncoder(enc_config, front=front)
        D = self.enc.dim
        self.use_ssl = ssl
        if ssl:  # frozen encoder from our own self-supervised pretraining (ssl_pre.py), added to the frames
            self.ssl = RawEncoder(enc_config, front=front)
            for p in self.ssl.parameters():
                p.requires_grad_(False)
            self.ssl_proj = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
            nn.init.zeros_(self.ssl_proj[2].weight)
            nn.init.zeros_(self.ssl_proj[2].bias)
        self.mel = mel
        if mel:  # fine acoustic detail beside the voice-trained encoder: log-mel 80 per 20 ms frame
            import torchaudio

            self._melspec = torchaudio.transforms.MelSpectrogram(SR, n_fft=640, hop_length=HOP, n_mels=80)
            self.mel_proj = nn.Sequential(nn.Conv1d(80, D, 5, padding=2), nn.GELU(), nn.Conv1d(D, D, 1))
            nn.init.zeros_(self.mel_proj[2].weight)
            nn.init.zeros_(self.mel_proj[2].bias)
        self.zproj = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, zdim))
        self.frame = nn.Linear(D, 2)  # speech, overlap
        self.fp = fp
        self.dec = Decoder(D, zdim, dec_layers, extra=2 if fp else 0)

    def load_from(self, sd):
        """Weights of a PairNet without fingerprint inputs: the new input columns start at zero."""
        own = self.state_dict()
        sd = {**{k: v for k, v in own.items() if k.startswith(("mel_proj.", "_melspec.", "ssl.", "ssl_proj."))}, **sd}
        k = "dec.inp.weight"
        if k in sd and sd[k].shape != own[k].shape:
            w = torch.zeros_like(own[k])
            w[:, : sd[k].shape[1]] = sd[k]
            sd = {**sd, k: w}
        self.load_state_dict(sd)

    def frames(self, wav):
        """wav [B, T] -> h [B, F, D], z [B, F, Z] (unit), speech and overlap logits [B, F]."""
        rms = wav.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-4)
        h = encode_wave(self.enc, wav / rms * 0.1)
        z = F.normalize(self.zproj(h).float(), dim=-1)
        if self.use_ssl:
            with torch.no_grad():
                self.ssl.eval()
                g = encode_wave(lambda w: _ssl_feats(self.ssl, w), wav / rms * 0.1)
            h = h + self.ssl_proj(g.to(h.dtype))
        if self.mel:  # added after the voice vectors: they stay those of the encoder
            m = torch.log(self._melspec((wav / rms * 0.1).float()) + 1e-5)[..., : h.shape[1]]
            m = (m - m.mean(-1, keepdim=True)) / (m.std(-1, keepdim=True) + 1e-3)
            h = h + self.mel_proj(m.to(h.dtype)).transpose(1, 2)
        fl = self.frame(h).float()
        return h, z, fl[..., 0], fl[..., 1]

    def decode(self, h, z, s, sp_logit, ov_logit, fpc=None):
        """fpc [B, 2, F]: cosine of each frame's fingerprint to voice A and voice B (or None)."""
        sp, ov = torch.sigmoid(sp_logit), torch.sigmoid(ov_logit)
        fa, fb = (fpc[:, 0], fpc[:, 1]) if fpc is not None else (None, None)
        la = self.dec(h, z, s[:, 0], s[:, 1], sp, ov, fa, fb)
        lb = self.dec(h, z, s[:, 1], s[:, 0], sp, ov, fb, fa)
        return torch.stack([la, lb], 1).float()  # [B, 2, F]

    def forward(self, wav, slots=None):
        h, z, sp, ov = self.frames(wav)
        if slots is None:
            w = (torch.sigmoid(sp) * (1 - torch.sigmoid(ov))).detach()
            slots = two_voices(z, w)
        return self.decode(h, z, slots, sp, ov), z, slots, sp, ov

    @torch.no_grad()
    def infer(self, wav, refine=1, thr=0.5, spk=None):
        """Whole recording [T] -> probabilities [2, F]. ``refine``: passes in which the two voices are
        taken again from the frames that the previous pass gave to one person alone. ``spk``: a
        speaker fingerprint (SpeakerNet); the two voices of the recording are then found by it
        (fingerprint_voices) instead of by PairNet's own frame vectors."""
        x = wav[None]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            h, z, sp, ov = self.frames(x)
        w = torch.sigmoid(sp) * (1 - torch.sigmoid(ov))
        fpc = None
        if spk is not None:
            fv = fingerprint_voices(spk, wav, w[0])
            if fv is not None:
                assign, fpc = fv
                s = F.normalize(torch.einsum("kf,fz->kz", assign * w[0][None], z[0]), dim=-1)[None]
                fpc = fpc[None] if self.fp else None
                refine = 0
            else:
                s = two_voices(z, w)
        else:
            s = two_voices(z, w)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            p = torch.sigmoid(self.decode(h, z, s, sp, ov, fpc))
        for _ in range(refine):
            solo = torch.stack([(p[:, 0] > thr) & (p[:, 1] < 1 - thr), (p[:, 1] > thr) & (p[:, 0] < 1 - thr)], 1).float()
            if (solo.sum(-1) < 25).any():
                break
            s = F.normalize(torch.einsum("bkf,bfz->bkz", solo * w[:, None], z), dim=-1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                p = torch.sigmoid(self.decode(h, z, s, sp, ov, None))
        return p[0].float(), s[0]


@torch.no_grad()
def fingerprint_track(spk, wav, win_s=1.5, hop_s=0.25, bs=512):
    """Fingerprints of 1.5 s windows every 0.25 s over a waveform [T] -> (unit vectors [J, 192],
    window start frames [J], window length in frames)."""
    W, H = int(win_s * SR), int(hop_s * SR)
    if len(wav) < W:
        wav = F.pad(wav, (0, W - len(wav)))
    st = torch.arange(0, len(wav) - W + 1, H, device=wav.device)
    segs = wav.unfold(0, W, H)[: len(st)]
    e = []
    for i in range(0, len(segs), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            e.append(F.normalize(spk(segs[i : i + bs].float()).float(), dim=-1))
    return torch.cat(e), st // HOP, W // HOP


def frame_cos(e, st, fw, cents, Fn):
    """Cosine of every frame's fingerprint (the window centred nearest to it) to each centroid
    [K, 192] -> [K, F]."""
    centres = st + fw // 2
    f = torch.arange(Fn, device=e.device)
    j = torch.clamp(torch.searchsorted(centres.contiguous(), f), 0, len(centres) - 1)
    jm = torch.clamp(j - 1, 0)
    j = torch.where((centres[jm] - f).abs() < (centres[j] - f).abs(), jm, j)
    return (e[j] @ cents.T).T


def window_share(m, st, fw):
    """Mean of a frame mask [K, F] over every window -> [J, K]."""
    c = torch.nn.functional.pad(torch.cumsum(m.float(), -1), (1, 0))
    a, b = st.clamp(max=m.shape[-1]), (st + fw).clamp(max=m.shape[-1])
    return ((c[:, b] - c[:, a]) / fw).T


@torch.no_grad()
def fingerprint_voices(spk, wav, w):
    """The two voices of a recording by a speaker fingerprint: fingerprints of windows where one
    person mostly speaks (mean weight w >= 0.5), split in two along their first principal
    direction and refined by k-means -> (frame assignment [2, F], cosine of every frame's
    fingerprint to the two voices [2, F]); None when too little speech."""
    Fn = len(w)
    e, st, fw = fingerprint_track(spk, wav)
    keep = window_share(w[None], st, fw)[:, 0] >= 0.5
    if keep.sum() < 4:
        return None
    ek = e[keep]
    c = ek - ek.mean(0, keepdim=True)
    u = torch.linalg.svd(c, full_matrices=False).Vh[0]
    lab = (c @ u > 0).long()
    for _ in range(10):
        if lab.min() == lab.max():
            return None
        cen = F.normalize(torch.stack([ek[lab == k].mean(0) for k in (0, 1)]), dim=-1)
        lab = (ek @ cen.T).argmax(1)
    cover = torch.zeros(2, Fn, device=wav.device)
    for s0, k in zip(st[keep].tolist(), lab.tolist()):
        cover[k, s0 : s0 + fw] += 1
    return cover / cover.sum(0, keepdim=True).clamp_min(1), frame_cos(e, st, fw, cen, Fn)


def pit_bce(logits, act, mask=None, weight=None):
    """logits, act [B, 2, F]; the A/B order that fits each example best. ``weight`` [B, 2, F]:
    per-frame weight that goes with the labels (it is flipped with them)."""
    if mask is None:
        mask = torch.ones_like(act[:, 0])
    if weight is None:
        weight = torch.ones_like(act)
    l0 = ((F.binary_cross_entropy_with_logits(logits, act, reduction="none") * weight).sum(1) * mask).sum(-1)
    l1 = ((F.binary_cross_entropy_with_logits(logits, act.flip(1), reduction="none") * weight.flip(1)).sum(1) * mask).sum(-1)
    swap = l1 < l0
    n = mask.sum(-1).clamp_min(1) * 2
    return (torch.where(swap, l1, l0) / n).mean(), swap
