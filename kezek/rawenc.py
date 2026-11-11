"""Version 1 encoder on raw audio: one network under both the speaker fingerprint and (next)
"who speaks when".

    waveform 16 kHz ─► RMS normalisation ─► conv front end (stride 320: one frame per 20 ms, the
    stage 1 grid) with learned log compression ─► Conformer (local context) ─► frames [B, F, D]
                                                                         └─► SpeakerHead ─► 192-d

* The front end replaces WavLM: strided 1-d convolutions on the waveform (strides 5·4·4·4), the
  first layer followed by a per-channel learned compression log(1 + e^a · relu(x)), so the
  network sees level differences on a log scale (recordings differ by tens of dB).
* The Conformer is torchaudio's; frames keep only local context (up to a few seconds), so the
  whole recording can be encoded in chunks at any length.
* SpeakerHead: attentive statistics pooling (the frames that carry the voice weigh more) ->
  192-d fingerprint, compared by cosine.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

HOP = 320  # samples per frame (20 ms)

CONFIGS = {
    # dim, layers, heads, ffn, conv kernel
    "S": (256, 6, 4, 1024, 15),
    "M": (320, 8, 4, 1280, 15),
}


class LogCompress(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.a = nn.Parameter(torch.full((1, ch, 1), 2.0))

    def forward(self, x):
        return torch.log1p(torch.exp(self.a) * F.relu(x))


class FrontEnd(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c1 = nn.Conv1d(1, 128, 10, stride=5, bias=False)
        self.comp = LogCompress(128)
        self.n1 = nn.GroupNorm(128, 128)
        self.rest = nn.Sequential(
            nn.Conv1d(128, 192, 8, stride=4), nn.GELU(),
            nn.Conv1d(192, 256, 4, stride=4), nn.GELU(),
            nn.Conv1d(256, dim, 4, stride=4), nn.GELU(),
        )
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, wav):  # [B, T] -> [B, F, D], F = T // 320
        B, T = wav.shape
        n = T // HOP
        x = F.pad(wav, (5, HOP * n + 5 + 345 - 10 - T)).unsqueeze(1)  # centre: frame k covers ~[320k-5, 320k+340)
        h = self.n1(self.comp(self.c1(x)))
        h = self.rest(h).transpose(1, 2)[:, :n]
        if h.shape[1] < n:
            h = F.pad(h, (0, 0, 0, n - h.shape[1]))
        return self.proj(h)


class GaborFront(nn.Module):
    """Phase-blind front end: a learnable complex (analytic) filterbank on the waveform, 128
    filters of 25 ms, hop 10 ms, initialised as Gabor filters on a mel-like scale; only the
    modulus is kept, so a change of phase inside a band (allpass, IIR filters of a recording
    chain) does not reach the network. Then learned log compression, and a strided convolution
    to the 20 ms grid."""

    def __init__(self, dim, n=128, win=401, hop=160):
        super().__init__()
        import math

        self.hop, self.win = hop, win
        mel = torch.linspace(2595 * math.log10(1 + 60 / 700), 2595 * math.log10(1 + 7800 / 700), n)
        fc = 700 * (10 ** (mel / 2595) - 1)  # centre frequencies, Hz
        bw = torch.clamp(fc * 0.25, min=60.0)  # bandwidths
        t = (torch.arange(win) - win // 2).float() / 16000
        env = torch.exp(-0.5 * (t[None] * bw[:, None] * 2 * math.pi / 2.355) ** 2)
        env = env / env.sum(-1, keepdim=True)
        self.re = nn.Conv1d(1, n, win, stride=hop, bias=False)
        self.im = nn.Conv1d(1, n, win, stride=hop, bias=False)
        with torch.no_grad():
            self.re.weight.copy_((env * torch.cos(2 * math.pi * fc[:, None] * t[None]))[:, None])
            self.im.weight.copy_((env * torch.sin(2 * math.pi * fc[:, None] * t[None]))[:, None])
        self.comp = LogCompress(n)
        self.norm = nn.GroupNorm(n, n)
        self.down = nn.Sequential(nn.Conv1d(n, dim, 3, stride=2, padding=1), nn.GELU(), nn.Conv1d(dim, dim, 3, padding=1), nn.GELU())
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, wav):
        B, T = wav.shape
        n = T // HOP
        x = F.pad(wav, (self.win // 2, self.win // 2 + 2 * self.hop)).unsqueeze(1)
        m = torch.sqrt(self.re(x).float() ** 2 + self.im(x).float() ** 2 + 1e-8).to(wav.dtype)
        h = self.norm(self.comp(m))
        h = self.down(h).transpose(1, 2)[:, :n]
        if h.shape[1] < n:
            h = F.pad(h, (0, 0, 0, n - h.shape[1]))
        return self.proj(h)


class RawEncoder(nn.Module):
    def __init__(self, config="S", dropout=0.1, front="conv"):
        super().__init__()
        from torchaudio.models import Conformer

        dim, layers, heads, ffn, k = CONFIGS[config]
        self.config, self.dim = config, dim
        self.front = GaborFront(dim) if front == "gabor" else FrontEnd(dim)
        self.conf = Conformer(input_dim=dim, num_heads=heads, ffn_dim=ffn, num_layers=layers,
                              depthwise_conv_kernel_size=k, dropout=dropout)

    def forward(self, wav, lengths=None):
        """wav [B, T] float -> frames [B, F, D]; ``lengths`` [B] in frames (padding at the end)."""
        rms = wav.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-4)
        h = self.front(wav / rms * 0.1)
        B, Fn, _ = h.shape
        if lengths is None:
            lengths = torch.full((B,), Fn, device=wav.device, dtype=torch.long)
        h, _ = self.conf(h, lengths)
        return h


class SpeakerHead(nn.Module):
    """Attentive statistics pooling over frames -> 192-d."""

    def __init__(self, dim, out=192, att=128):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(3 * dim, att), nn.Tanh(), nn.Linear(att, dim))
        self.out = nn.Sequential(nn.Linear(2 * dim, out), nn.BatchNorm1d(out))

    def forward(self, h, mask=None):
        """h [B, F, D], mask [B, F] (True = use) -> [B, 192]."""
        if mask is None:
            mask = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        m = mask.unsqueeze(-1).float()
        n = m.sum(1).clamp_min(1)
        mu = (h * m).sum(1) / n
        sd = (((h - mu[:, None]) ** 2 * m).sum(1) / n).clamp_min(1e-6).sqrt()
        ctx = torch.cat([h, mu[:, None].expand_as(h), sd[:, None].expand_as(h)], -1)
        a = self.att(ctx).float().masked_fill(~mask.unsqueeze(-1), -1e4).softmax(1).to(h.dtype)
        mu2 = (a * h).sum(1)
        sd2 = ((a * (h - mu2[:, None]) ** 2).sum(1)).clamp_min(1e-6).sqrt()
        return self.out(torch.cat([mu2, sd2], -1).float())


class SpeakerNet(nn.Module):
    def __init__(self, config="S", front="conv"):
        super().__init__()
        self.enc = RawEncoder(config, front=front)
        self.head = SpeakerHead(self.enc.dim)

    def forward(self, wav, mask=None):
        return self.head(self.enc(wav), mask)


class AAM(nn.Module):
    """Additive angular margin softmax over classes."""

    def __init__(self, n_classes, dim=192, scale=30.0):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_classes, dim) * 0.01)
        self.scale = scale

    def forward(self, emb, y, margin):
        cos = F.linear(F.normalize(emb.float(), dim=-1), F.normalize(self.w.float(), dim=-1))
        th = torch.acos(cos.gather(1, y[:, None]).clamp(-1 + 1e-6, 1 - 1e-6))
        tgt = torch.cos(th + margin)
        logits = cos.scatter(1, y[:, None], tgt) * self.scale
        return F.cross_entropy(logits, y), (cos.argmax(1) == y).float().mean()
