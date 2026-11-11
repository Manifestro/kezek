---
license: apache-2.0
language:
- en
- kk
- ru
library_name: kezek
pipeline_tag: audio-to-audio
tags:
- speech-separation
- speaker-separation
- conversation
- diarization
---

# Kezek v1

**Two people, one mono recording → one time-aligned track per person.**

Kezek splits a conversation between two people, recorded on one channel, into two tracks: everything person A says on one track, everything person B says on the other. Both tracks are aligned sample by sample with the input.

Short replies ("yeah", "mm-hm") said while the other person talks stay on the right track. Nothing is cut out: the two tracks sum exactly to the input.

> DRAFT — the report text and the final wording come from the release report. The numbers below are final: they come from the evaluations behind it.

## Usage

```bash
pip install git+https://github.com/Manifestro/kezek
kezek-separate conversation.wav --out separated/
```

```python
from kezek import Kezek
kz = Kezek.from_pretrained()            # this repository
tracks, activity, info = kz(audio_16k)  # float32 mono at 16 kHz -> [2, T]
```

A CUDA GPU is required (bf16).

## What is inside

The model has three parts. All of them were trained by us, and none is a third-party pretrained model.

| Part | File | Parameters | What it does |
|---|---|---|---|
| Stage 1, PairNet | `stage1_pairnet.pt` | see `params.json` | raw audio → who speaks when (20 ms) for the two voices of the recording |
| Voice fingerprint | `fingerprint.pt` | see `params.json` | a speaker embedding on raw audio; finds the two voices and builds a profile of each from up to 30 s of their solo speech |
| Separator, KezekTSE-L | `separator.pt` | 19.6 M | mixture + stage 1 activity + the two profiles → two tracks that sum exactly to the input |

## Results

### Against the open pipeline

The comparison is against pyannote `speaker-diarization-community-1` → DialogueSidon, as published in the DuplexChat repository. Both systems ran on the same recordings and were scored by the same rules, fixed before the run.

**Open Yap 1K test.** 121 real two-person conversations, up to 300 s each. The condition is "as recorded": the conversation rendered as one room recording with reverberation, noise, EQ, compression and codec.

| | Kezek v1 | open pipeline |
|---|---|---|
| short replies kept | **0.830** | 0.687 |
| recordings without swapped tracks | **0.934** | 0.793 |
| leak of the other voice | **−13.9 dB** | −10.1 dB |
| audio returned | **100%** | 73.5% |

**CallHome English.** 128 private telephone calls, first 300 s each, 10.7 h. This is telephone audio, unseen in training by both systems. It is scored by the human speaker segments, because the dataset has no reference tracks.

| | Kezek v1 | open pipeline |
|---|---|---|
| replies over the other's speech kept | **0.796** | 0.711 |
| replies in a pause kept | **0.907** | 0.798 |
| leak | **−11.8 dB** | −9.7 dB |
| calls without swapped tracks | **0.859** | 0.797 |
| audio returned | **100%** | 81.6% |

**Speed** on one RTX PRO 6000 (Blackwell):

| | Kezek v1 | open pipeline |
|---|---|---|
| hours of audio per GPU-hour | **≈ 230** | ≈ 50 |

### Limitations

- Very quiet replies, more than 10 dB below the other person, are often lost.
- On telephone audio with two similar voices, the tracks can swap for part of a call. In CallHome this happened on a large part of the call in 7 of 128 calls.
- Kezek does not denoise. Background noise is shared between the tracks.
- Exactly two people. Recordings with three or more speakers are out of scope.

## Training data

| Data | Licence / terms |
|---|---|
| Open Yap 1K (The Agentic Data Company) | under its data use agreement, which allows commercial use and sharing of trained weights |
| Libri-Light (LibriVox audio, public domain) | — |
| VoxPopuli | CC0; contains European Parliament recordings — source: European Parliament |
| Augmentation: RIRS_NOISES | Apache-2.0 |
| Augmentation: MUSAN | per OpenSLR 17 |

CallHome and the DuplexChat audio were not used for training.

## Licence

Weights: Apache-2.0. Code (github.com/Manifestro/kezek): MIT.
