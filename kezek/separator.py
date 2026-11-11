"""Kezek v3 stage 2: KezekTSE-L, the K-T0c / K-R1 Conv-TasNet separator made larger and given the
stage 1 decision as its input.

    mixture ─► Conv encoder ─► gLN ─► bottleneck ─(+ hint a_A, a_B)─► [FiLM TCN block × X·R] ─► two
    softmax masks ─► shared decoder ─► y_A, y_B ─► projection y_i += (x − y_A − y_B) / 2

* the activity hint (2 channels, per STFT hop of 128 samples) is resampled to the encoder frames
  and added to the bottleneck through a 1×1 convolution;
* the profiles A and B (ECAPA, 192-d each) are normalised, concatenated and modulate every block
  with FiLM, as the speaker embedding did in KezekTSE;
* the two masks are a softmax over the sources (every encoder bin goes to A or B), and the outputs
  are projected so that y_A + y_B = x exactly: solo passes through, nothing is lost.

Call: ``model(mix [B, T], hint [B, 2, F], pa [B, 192], pb [B, 192]) -> [B, 2, T]`` (the same as
the TF-GridNet KezekSep, so the stage 2 training and evaluation code takes either).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalLN(nn.Module):
    """Global layer norm over (channels, time)."""

    def __init__(self, ch: int):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, ch, 1))
        self.b = nn.Parameter(torch.zeros(1, ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = x.mean(dim=(1, 2), keepdim=True)
        v = ((x - m) ** 2).mean(dim=(1, 2), keepdim=True)
        return self.g * (x - m) / torch.sqrt(v + 1e-8) + self.b


class FiLMBlock(nn.Module):
    """TCN block with FiLM conditioning and a dilated depthwise convolution."""

    def __init__(self, bn: int, hid: int, kernel: int, dilation: int, cond: int):
        super().__init__()
        self.inp = nn.Conv1d(bn, hid, 1)
        self.act1, self.norm1 = nn.PReLU(), GlobalLN(hid)
        self.film = nn.Linear(cond, 2 * hid)
        pad = (kernel - 1) * dilation // 2
        self.dw = nn.Conv1d(hid, hid, kernel, padding=pad, dilation=dilation, groups=hid)
        self.act2, self.norm2 = nn.PReLU(), GlobalLN(hid)
        self.out = nn.Conv1d(hid, bn, 1)
        nn.init.zeros_(self.film.weight)  # start as identity: gamma=1, beta=0
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        h = self.norm1(self.act1(self.inp(x)))
        gamma, beta = self.film(d).unsqueeze(-1).chunk(2, dim=1)
        h = (1 + gamma) * h + beta
        h = self.norm2(self.act2(self.dw(h)))
        return x + self.out(h)


HINT_HOP = 128

CONFIGS = {
    # n_filters, kernel, bottleneck, hidden, blocks, repeats, cond
    "M": (512, 32, 256, 512, 8, 3, 256),
    "L": (512, 32, 256, 768, 8, 3, 256),
    "XL": (512, 32, 256, 1024, 8, 3, 256),
    "S": (256, 32, 128, 512, 8, 3, 192),
}


class KezekTSEL(nn.Module):
    def __init__(self, config: str = "M", tcn_kernel: int = 3):
        super().__init__()
        n_filters, kernel, bottleneck, hidden, blocks, repeats, cond = CONFIGS[config]
        self.config = config
        self.kernel, self.stride = kernel, kernel // 2
        self.enc = nn.Conv1d(1, n_filters, kernel, stride=self.stride, bias=False)
        self.norm = GlobalLN(n_filters)
        self.bottleneck = nn.Conv1d(n_filters, bottleneck, 1)
        self.hint = nn.Conv1d(2, bottleneck, 1)
        self.d_proj = nn.Sequential(nn.Linear(2 * 192, cond), nn.PReLU())
        self.blocks = nn.ModuleList(
            FiLMBlock(bottleneck, hidden, tcn_kernel, 2**i, cond) for _ in range(repeats) for i in range(blocks)
        )
        self.mask = nn.Sequential(nn.PReLU(), nn.Conv1d(bottleneck, 2 * n_filters, 1))
        self.dec = nn.ConvTranspose1d(n_filters, 1, kernel, stride=self.stride, bias=False)

    def forward(self, mix: torch.Tensor, hint: torch.Tensor, pa: torch.Tensor, pb: torch.Tensor) -> torch.Tensor:
        B, T = mix.shape
        pad = (-(T - self.kernel) % self.stride) if T >= self.kernel else self.kernel - T
        x = F.pad(mix, (0, pad)).unsqueeze(1)
        w = F.relu(self.enc(x))  # [B, N, L]
        L = w.shape[-1]
        idx = torch.clamp(torch.arange(L, device=mix.device) * self.stride // HINT_HOP, max=hint.shape[-1] - 1)
        h = self.bottleneck(self.norm(w)) + self.hint(hint.to(w.dtype)[..., idx])
        d = torch.cat([F.normalize(pa.float(), dim=-1), F.normalize(pb.float(), dim=-1)], -1) * 192**0.5
        d = self.d_proj(d.to(h.dtype))
        for blk in self.blocks:
            h = blk(h, d)
        m = self.mask(h).view(B, 2, w.shape[1], L).float().softmax(dim=1).to(w.dtype)
        y = self.dec((w.unsqueeze(1) * m).flatten(0, 1)).view(B, 2, -1)[..., :T].float()
        x0 = mix.float()
        return y + ((x0 - y.sum(1)) / 2).unsqueeze(1)
