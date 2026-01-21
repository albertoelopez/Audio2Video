# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

HeartLib is a Python library providing open-source music foundation models. It includes:
- **HeartMuLa**: Music language model (3B/7B parameters) generating music from lyrics + style tags
- **HeartCodec**: 12.5Hz music codec for high-fidelity audio encoding/decoding
- **HeartTranscriptor**: Whisper-based model fine-tuned for lyrics transcription

## Common Commands

```bash
# Install locally (editable)
pip install -e .

# Generate music
python ./examples/run_music_generation.py \
  --model_path=./ckpt \
  --lyrics=./assets/lyrics.txt \
  --tags=./assets/tags.txt \
  --save_path=./output.mp3 \
  --version="3B"

# Transcribe lyrics
python ./examples/run_lyrics_transcription.py \
  --model_path=./ckpt \
  --music_path=./assets/output.mp3

# Download checkpoints (HuggingFace)
huggingface-cli download HeartMuLa/HeartMuLaGen --local-dir ./ckpt
huggingface-cli download HeartMuLa/HeartMuLa-oss-3B --local-dir ./ckpt/HeartMuLa-oss-3B
huggingface-cli download HeartMuLa/HeartCodec-oss --local-dir ./ckpt/HeartCodec-oss
huggingface-cli download HeartMuLa/HeartTranscriptor-oss --local-dir ./ckpt/HeartTranscriptor-oss
```

## Architecture

### Pipeline Layer (`src/heartlib/pipelines/`)
- `HeartMuLaGenPipeline`: Main entry point for music generation - handles text tokenization, prompt encoding, autoregressive generation, and audio decoding
- `HeartTranscriptorPipeline`: Wraps Whisper for lyrics transcription with chunked processing

### Model Layer (`src/heartlib/`)
- `heartmula/modeling_heartmula.py`: LLaMA 3.2-based transformer with dual backbone/decoder architecture for parallel codebook prediction
- `heartcodec/modeling_heartcodec.py`: Flow-matching based neural codec combining `FlowMatching` (latent generation) and `ScalarModel` (waveform synthesis)

### Generation Flow
1. Text (lyrics + tags) → tokenized with special `<tag></tag>` tokens
2. Backbone transformer generates first codebook token via classifier-free guidance
3. Decoder transformer generates remaining codebook tokens (31 total) per frame
4. HeartCodec detokenizes audio codes → latent space → waveform at 48kHz

### Key Parameters
- `max_audio_length_ms`: Maximum generation length (default 240000ms = 4min)
- `cfg_scale`: Classifier-free guidance strength (default 1.5)
- `topk`, `temperature`: Sampling controls (defaults: 50, 1.0)
- `version`: Model variant ("3B" or "7B")

## Checkpoint Structure

```
./ckpt/
├── HeartCodec-oss/          # Audio codec weights
├── HeartMuLa-oss-3B/        # Main model weights
├── HeartTranscriptor-oss/   # Transcription model (optional)
├── gen_config.json          # Generation config (token IDs)
└── tokenizer.json           # Text tokenizer (required)
```

## API Usage

```python
from heartlib import HeartMuLaGenPipeline
import torch

pipe = HeartMuLaGenPipeline.from_pretrained(
    "./ckpt",
    device=torch.device("cuda"),
    dtype=torch.bfloat16,
    version="3B"
)

with torch.no_grad():
    pipe(
        {"lyrics": "path/to/lyrics.txt", "tags": "path/to/tags.txt"},
        save_path="output.mp3",
        max_audio_length_ms=240000
    )
```

## Input Formats

**Lyrics** - Use section markers:
```
[Verse]
First line of verse
Second line of verse

[Chorus]
Chorus lyrics here
```

**Tags** - Comma-separated, no spaces:
```
piano,happy,romantic,synthesizer
```
