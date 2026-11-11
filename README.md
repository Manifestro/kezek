<p align="center"><b>Kezek</b><br>Splitting mono two-person conversations into per-speaker tracks</p>

<p align="center">
  <a href="https://huggingface.co/manifestro/kezek-v1">Weights</a> ·
  <a href="https://manifestro.io/models/kezek">Demo</a> ·
  Paper (arXiv, link at release)
</p>

Kezek turns a mono recording of exactly two people into **one track per person**. Each track keeps the original length and timing.

- Overlaps and short replies ("yeah", "mm-hm") stay on the track of the person who said them.
- The two tracks always sum exactly to the input.
- It is built for preparing full-duplex dialogue data from conversational audio that exists only as a mono mix: podcasts, interviews, calls.

## How it works

Kezek decides *who speaks when* and *who is who* once for the whole recording. A separator then carries out that decision.

| Stage | Model | Parameters |
|---|---|---|
| Speaker fingerprint | raw-audio speaker embedding, trained to ignore phase | 10.1 M |
| Stage 1 | **PairNet**: activity of the two voices over the whole recording, guided by the fingerprint | 11.8 M |
| Stage 2 | **KezekTSE-L**: Conv-TasNet-style separator conditioned on the activity and the two voice profiles; outputs sum to the input | 19.6 M |

Total: 41 M parameters, all trained by us. No frozen third-party model is inside.

## Results

The comparison is against the open pipeline pyannote → DialogueSidon (as published in the DuplexChat code), on the same recordings, over whole recordings.

| Set | System | Short replies kept ↑ | Leak (dB) ↓ | No-swap ↑ | Coverage ↑ | h audio / GPU-h ↑ |
|---|---|---|---|---|---|---|
| Open Yap test (as recorded) | pyannote → DialogueSidon | 0.687 | −10.1 | 0.793 | 0.735 | 51 |
| | **Kezek** | **0.830** | **−13.9** | **0.934** | **1.000** | **231** |
| CallHome English (unseen telephone speech) | pyannote → DialogueSidon | 0.711 / 0.798 | −9.7 | 0.797 | 0.816 | 46 |
| | **Kezek** | **0.796 / 0.907** | **−11.8** | **0.859** | **1.000** | **232** |

- **Replies kept.** The share of short replies (< 1 s) kept on the right track. On CallHome: over the other's speech / in pauses.
- **Leak.** Energy on a track while only the other person speaks.
- **No-swap.** The share of recordings in which the two people's tracks are never swapped.
- **Coverage.** The share of the recording returned.
- **Speed.** One RTX PRO 6000 GPU.

The metrics, the ablations and an ideal-mask ceiling are in the paper.

## Install

```bash
pip install kezek
```

You need Python ≥ 3.10 and PyTorch ≥ 2.4. A CUDA GPU with bf16 is recommended; on CPU it runs in float32 and is much slower.

## Use

From the command line:

```bash
kezek-separate interview.wav podcast.mp3 --out separated/
# separated/interview_A.wav, separated/interview_B.wav, separated/interview.json
```

From Python:

```python
import soundfile as sf
from kezek import Kezek

kz = Kezek.from_pretrained()                    # weights from huggingface.co/manifestro/kezek-v1
x, sr = sf.read("interview_16k.wav", dtype="float32")   # mono, 16 kHz
tracks, activity, info = kz(x)                  # tracks [2, T]; activity [2, frames of 20 ms]
```

The CLI resamples any input to 16 kHz and mixes it down to mono. The JSON next to each pair of tracks gives each person's speech time, the overlap time and the stretches where stage 1 was unsure.

## Limitations

- Quiet replies said under a louder voice are the main remaining loss.
- Exactly two speakers, offline. No enrollment and no speech restoration: Kezek splits the audio, it does not clean it.
- Trained and evaluated quantitatively in English.
- On telephone speech with similar voices, the tracks can swap for part of a call.

## Responsible use

Kezek only reassigns audio that is already in a recording; it does not synthesise speech. You must have the right to process the recordings you give it. Do not use it to identify speakers or to clone a person's voice.

## Licence

- Code: MIT (this repository).
- Weights: Apache-2.0 ([Hugging Face](https://huggingface.co/manifestro/kezek-v1)).

The model was trained on Open Yap 1K under its [Data Use Agreement](https://theagenticdatacompany.com/open-yap-1k/license), on Libri-Light and VoxPopuli, with MUSAN and RIRS noises for augmentation. The training data is not distributed here. See the model card.

## Citation

```bibtex
@misc{karl2026kezek,
  title  = {Kezek: Splitting Mono Two-Person Conversations into Per-Speaker Tracks for Full-Duplex Dialogue Data},
  author = {Karl, Bagzhan and Karl, Alizhan},
  year   = {2026},
  note   = {arXiv identifier to be added at release}
}
```

## Acknowledgements

We thank The Agentic Data Company for Open Yap 1K. VoxPopuli: we acknowledge the European Parliament for creating and sharing these materials.
