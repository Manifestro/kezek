---
license: apache-2.0
language:
- en
library_name: kezek
pipeline_tag: audio-to-audio
tags:
- speech-separation
- speaker-separation
- conversation
- full-duplex
- dialogue
datasets:
- TheAgenticDataCompany/open-yap-1k
- facebook/voxpopuli
---

# Kezek v1

**A mono recording of exactly two people → one track per person, with the original length and timing.**

Overlaps and short replies ("yeah", "mm-hm") stay on the track of the person who said them. The two tracks always sum exactly to the input.

Kezek is built for preparing full-duplex dialogue data from conversational audio that exists only as a mono mix.

[Code (MIT)](https://github.com/Manifestro/kezek) · [Demo](https://manifestro.io/models/kezek) · Paper (arXiv, link at release)

## Usage

```bash
pip install git+https://github.com/Manifestro/kezek
kezek-separate conversation.wav --out separated/
```

```python
from kezek import Kezek
kz = Kezek.from_pretrained()            # this repository
tracks, activity, info = kz(audio_16k)  # float32 mono at 16 kHz -> tracks [2, T]
```

A CUDA GPU with bf16 is required.

## Model

Kezek decides *who speaks when* and *who is who* once for the whole recording (stage 1). A conditioned separator then carries out that decision (stage 2).

| File | Component | Parameters |
|---|---|---|
| `fingerprint.pt` | speaker fingerprint: raw-audio speaker embedding, trained to ignore phase | 10.1 M |
| `stage1_pairnet.pt` | PairNet: activity of the two voices over the whole recording (20 ms), guided by the fingerprint | 11.8 M |
| `separator.pt` | KezekTSE-L: Conv-TasNet-style separator conditioned on the activity and both voice profiles; outputs sum to the input | 19.6 M |

Total: 41 M parameters, all trained from our own runs, with no frozen third-party model.

Inference defaults: activity threshold 0.3; the separator runs over the whole recording in 8 s windows with a 6 s hop.

## Evaluation

The comparison is against pyannote → DialogueSidon, as published in the DuplexChat code, run without changes on the same recordings. Whole recordings are scored; what a system does not return counts as lost.

| Set | System | Short replies kept ↑ | Leak (dB) ↓ | No-swap ↑ | Coverage ↑ | h / GPU-h ↑ |
|---|---|---|---|---|---|---|
| Open Yap test, as recorded (121 conv.) | pyannote → DialogueSidon | 0.687 | −10.1 | 0.793 | 0.735 | 51 |
| | **Kezek v1** | **0.830** | **−13.9** | **0.934** | **1.000** | **231** |
| CallHome English, unseen telephone (128 calls) | pyannote → DialogueSidon | 0.711 / 0.798 | −9.7 | 0.797 | 0.816 | 46 |
| | **Kezek v1** | **0.796 / 0.907** | **−11.8** | **0.859** | **1.000** | **232** |

How each column is scored:
- **Open Yap:** scored against the reference channels.
- **CallHome:** scored by the utterance annotation. Replies are given over the other's speech / in pauses.
- **Leak:** energy on a track while only the other person speaks.
- **No-swap:** share of recordings without swapped channels.
- **Speed:** one RTX PRO 6000 GPU.

On Open Yap test, clean audio:
- overlap SI-SDRi +10.61 dB;
- short replies 0.878;
- no-swap 0.967.

## Limitations

- The remaining gap is quiet replies under a louder voice. On Open Yap test, as recorded, 0.83 of short replies are kept.
- Exactly two speakers, offline; no enrollment; no speech restoration. Kezek splits the audio and does not clean it.
- Training and quantitative evaluation are in English.
- On telephone speech with similar voices, channels can swap for part of a call: 14% of CallHome calls have at least one swapped window.

## Training data

| Data | Use | Terms |
|---|---|---|
| Open Yap 1K, build `open-yap-1k_2026-11-04_wav_0e7bcf1e_0d3abb77` | conversations: activity, separation | [Open Yap 1K Data Use Agreement v1](https://theagenticdatacompany.com/open-yap-1k/license). Commercial use and open weights allowed. No Open Yap audio or derived tracks are distributed here. |
| Libri-Light (LibriVox) | synthetic overlaps, fingerprint | public domain audio |
| VoxPopuli | fingerprint | CC0. We acknowledge the European Parliament for creating and sharing these materials. |
| MUSAN | augmentation (noise, music) | CC BY 4.0 |
| RIRS_NOISES | augmentation (room impulse responses) | Apache-2.0 |

CallHome and DuplexChat audio were not used for training.

## Responsible use

Kezek only reassigns audio that is already in a recording; it does not synthesise speech. Users must have the right to process their recordings. Do not use Kezek to identify speakers or to clone a person's voice.

## Licence

Weights: Apache-2.0. Code: MIT ([github.com/Manifestro/kezek](https://github.com/Manifestro/kezek)).

## Citation

```bibtex
@misc{karl2026kezek,
  title  = {Kezek: Splitting Mono Two-Person Conversations into Per-Speaker Tracks for Full-Duplex Dialogue Data},
  author = {Karl, Bagzhan and Karl, Alizhan},
  year   = {2026},
  note   = {arXiv identifier to be added at release}
}
```

Please also cite Open Yap 1K (The Agentic Data Company), build `open-yap-1k_2026-11-04_wav_0e7bcf1e_0d3abb77`.
