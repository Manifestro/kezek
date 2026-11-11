# Kezek

**Two people, one mono recording → one time-aligned track per person.**

> DRAFT — the intro and the results section follow the release report. The code and the usage below are final.

Kezek splits a two-person conversation recorded on one channel into two tracks, aligned sample by sample with the input. Short replies said while the other person talks stay on the speaker's own track. The two tracks always sum exactly to the input.

## Install

```bash
pip install git+https://github.com/Manifestro/kezek
```

Requirements:
- Python ≥ 3.10;
- PyTorch ≥ 2.4 with CUDA;
- a GPU with bf16 support.

## Use

From the command line:

```bash
kezek-separate interview.wav podcast.mp3 --out separated/
# separated/interview_A.wav, separated/interview_B.wav, separated/interview.json (speech and overlap seconds)
```

From Python:

```python
import soundfile as sf
from kezek import Kezek

kz = Kezek.from_pretrained()                    # weights: huggingface.co/manifestro/kezek-v1
x, sr = sf.read("interview_16k.wav", dtype="float32")
tracks, activity, info = kz(x)                  # tracks [2, T]; activity [2, frames of 20 ms]
```

Input is mono, 16 kHz. The CLI resamples other rates and mixes multi-channel audio down to mono.

## How it works

1. **Stage 1 (PairNet)** marks who speaks when, every 20 ms, for the two voices of the recording.
2. **The voice fingerprint** builds a profile of each person from their solo speech.
3. **The separator (KezekTSE-L, 19.6 M parameters)** takes the recording, the activity and both profiles, and splits the recording into two tracks. It works in 8 s windows and joins them with cross-fades.

## Results

See the [model card](https://huggingface.co/manifestro/kezek-v1) and the release report.

## Licence

- Code: MIT (this repository).
- Weights: Apache-2.0 (the Hugging Face repository).
