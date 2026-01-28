# Qwen3-TTS One-Command Fine-Tuning

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> End-to-end automation for fine-tuning Qwen3-TTS with your own voice samples.

An automated pipeline that handles everything from environment setup to model training. Simply provide a directory of WAV files and a reference audio—this tool automatically sets up the environment, transcribes your audio, prepares the dataset, and fine-tunes the Qwen3-TTS model to clone your voice.

## Features

- **True One-Command** – Single command handles setup, transcription, and training
- **Automatic Setup** – Environment and dependencies installed automatically on first run
- **Automatic Transcription** – Uses WhisperX to transcribe your audio files in 99+ languages
- **Complete Pipeline** – Handles transcription, JSONL creation, data preparation, and training
- **Multi-Language Support** – Works with any language supported by Whisper
- **Flash Attention Fallback** – Automatically uses flash_attention_2 if available, falls back to eager attention
- **Isolated Environment** – HuggingFace models cached in `venv/hf_cache/` for portability
- **GPU Optimized** – CUDA support for faster training and inference

## Architecture

![Architecture Diagram](./docs/diagrams/architecture.svg)

The system consists of three main phases:

1. **Auto-Setup Phase**: Automatically detects if environment is ready, creates virtual environment, installs PyTorch (with CUDA detection), downloads all dependencies
2. **Training Pipeline**: Validates audio, transcribes with WhisperX, creates JSONL files, extracts audio codes, fine-tunes the model
3. **External Models**: Integrates with WhisperX for transcription and HuggingFace for Qwen3-TTS models

## Data Flow

![Data Flow Diagram](./docs/diagrams/data_flow.svg)

The pipeline processes your audio through 6 steps:
1. **Audio Validation** – Verifies all WAV files are loadable
2. **WhisperX Transcription** – Converts speech to text
3. **JSONL Creation** – Creates `train_raw.jsonl` with audio paths and transcripts
4. **Audio Encoding** – Extracts `audio_codes` using Qwen3 Tokenizer (16-layer codec)
5. **JSONL with Codes** – Creates `train_with_codes.jsonl` for training
6. **Model Fine-Tuning** – Trains Qwen3-TTS on your voice

## Quick Start

### Prerequisites

- Python 3.12
- CUDA 12.x (for GPU support, ~16GB VRAM recommended)
- SoX audio library (required by qwen-tts)
- ~10GB disk space for models

**Install SoX:**
```bash
# Ubuntu/Debian
sudo apt install sox libsox-fmt-all

# RHEL/CentOS
sudo yum install sox

# macOS
brew install sox
```

### Pinned Versions

The setup uses specific tested versions:
- **PyTorch:** 2.8.0 (CUDA 12.8)
- **flash-attn:** 2.8.1 (pre-built wheel for Python 3.12 + CUDA 12 + PyTorch 2.8)

### One-Command Training

```bash
# Navigate to the project directory
cd /path/to/Qwen3-TTS-finetune

# Run training - setup happens automatically if needed!
./train.sh \
    --audio_dir ./my_audio_files \
    --ref_audio ./reference.wav \
    --speaker_name my_voice
```

That's it! The script automatically:
1. Detects if environment is ready
2. Runs setup if needed (non-interactive)
3. Configures HuggingFace cache
4. Runs the complete training pipeline

### Manual Setup (Optional)

If you prefer to set up the environment separately:

```bash
# Interactive setup (prompts for model pre-download)
./setup.sh

# Non-interactive setup
./setup.sh --auto
```

## What Gets Installed Automatically

The setup handles all dependencies:

| Category | Packages |
|----------|----------|
| **Core ML** | torch 2.8.0, torchaudio 2.8.0, transformers, accelerate |
| **TTS** | qwen-tts (Qwen3-TTS models) |
| **Audio** | librosa, soundfile |
| **Transcription** | whisperx |
| **Acceleration** | flash-attn 2.8.1 (pre-built wheel, with fallback) |
| **Utilities** | tqdm, safetensors, datasets, huggingface-hub |

Models are automatically downloaded from HuggingFace on first use:
- `Qwen/Qwen3-TTS-Tokenizer-12Hz` (~2GB)
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (~3.5GB)
- WhisperX model (~3GB for large-v3)

## Usage

### Directory Structure

Your audio files should be organized as follows:

```
my_project/
├── my_audio_files/          # Directory containing training WAV files
│   ├── sample1.wav
│   ├── sample2.wav
│   └── ...
├── reference.wav            # Reference audio for speaker embedding
├── train_from_audio.py      # Main automation script
├── train.sh                 # Shell wrapper (entry point)
└── setup.sh                 # Setup script (called automatically)
```

### Basic Usage

```bash
# One command does everything!
./train.sh \
    --audio_dir ./my_audio_files \
    --ref_audio ./reference.wav \
    --speaker_name my_voice
```

### Advanced Options

```bash
./train.sh \
    --audio_dir ./my_audio_files \
    --ref_audio ./reference.wav \
    --speaker_name my_voice \
    --output_dir ./my_output \
    --batch_size 4 \
    --lr 1e-5 \
    --epochs 5 \
    --whisper_model large-v3 \
    --language en
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `--audio_dir` | Directory containing WAV files | *required* |
| `--ref_audio` | Path to reference audio file | *required* |
| `--speaker_name` | Name for the speaker | `my_speaker` |
| `--output_dir` | Output directory | `./output` |
| `--device` | Device to use | `cuda:0` |
| `--batch_size` | Training batch size | `2` |
| `--lr` | Learning rate | `2e-5` |
| `--epochs` | Number of training epochs | `3` |
| `--whisper_model` | Whisper model size | `large-v3` |
| `--language` | Language code or `auto` | `auto` |

## Output

After completion, you'll find:

```
output/
├── checkpoint-epoch-0/     # Checkpoint after epoch 0
├── checkpoint-epoch-1/     # Checkpoint after epoch 1
├── checkpoint-epoch-2/     # Checkpoint after epoch 2 (use this)
├── train_raw.jsonl         # Raw training data
└── train_with_codes.jsonl  # Data with audio codes
```

## Inference

After training, use your fine-tuned model:

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda:0"
tts = Qwen3TTSModel.from_pretrained(
    "output/checkpoint-epoch-2",
    device_map=device,
    dtype=torch.bfloat16,
    # Uses flash_attention_2 if available, falls back to eager
)

wavs, sr = tts.generate_custom_voice(
    text="Hello, this is a test.",
    speaker="my_voice",
)
sf.write("output.wav", wavs[0], sr)
```

## Tips for Best Results

1. **Audio Quality**: Use clean, high-quality recordings (16kHz or higher)
2. **Reference Audio**: Choose a clear, representative sample as your reference
3. **Dataset Size**: 10-100 samples work well for single-speaker fine-tuning
4. **Audio Length**: 5-30 second clips are optimal
5. **Consistency**: All audio should be from the same speaker in similar conditions

## Supported Languages

Whisper supports 99 languages. Common codes:

| Code | Language |
|------|----------|
| `en` | English |
| `zh` | Chinese |
| `es` | Spanish |
| `fr` | French |
| `de` | German |
| `ja` | Japanese |
| `ko` | Korean |

Use `auto` for automatic language detection.

## Troubleshooting

### Out of Memory

Reduce batch size:

```bash
./train.sh --audio_dir ./audio --ref_audio ./ref.wav --batch_size 1
```

### Slow Transcription

Use a smaller Whisper model:

```bash
./train.sh --audio_dir ./audio --ref_audio ./ref.wav --whisper_model base
```

### Poor Transcription Quality

Specify the language explicitly:

```bash
./train.sh --audio_dir ./audio --ref_audio ./ref.wav --language en
```

### Flash Attention Not Available

The system automatically falls back to eager attention if flash_attn cannot be installed. You'll see this message during setup:

```
flash-attn installation failed - will use eager attention (slower but compatible)
```

This is normal on some systems. Training will still work, just slightly slower.

### Setup Issues

If you want to manually run setup with verbose output:

```bash
bash -x setup.sh --auto
```

## Complete One-Command Example

From a fresh clone to a trained model:

```bash
# Clone and enter the project
git clone <repo-url>
cd Qwen3-TTS-finetune

# Prepare your audio files in ./my_audio_files/
# Place reference.wav in the project root

# One command does everything!
./train.sh \
    --audio_dir ./my_audio_files \
    --ref_audio ./reference.wav \
    --speaker_name alice \
    --epochs 3

# Your trained model will be in output/checkpoint-epoch-2/
```

## Virtual Environment

The setup creates a virtual environment in `venv/` with an isolated HuggingFace cache:

```
venv/
├── bin/
│   └── activate          # Contains HF_HOME exports
├── lib/
├── hf_cache/             # HuggingFace models stored here
│   ├── transformers/
│   └── datasets/
└── ...
```

To manage it manually:

```bash
# Activate (only needed if running Python scripts directly)
source activate.sh
# or
source venv/bin/activate

# Deactivate
deactivate

# Delete and recreate (train.sh will auto-setup again)
rm -rf venv/
./train.sh --audio_dir ./audio --ref_audio ./ref.wav
```

## Project Structure

```
Qwen3-TTS-finetune/
├── README.md                   # This file
├── CLAUDE.md                   # Claude Code instructions
├── setup.sh                    # Auto-setup script (--auto for non-interactive)
├── train.sh                    # Entry point - handles everything
├── activate.sh                 # Venv activation helper (created by setup)
├── train_from_audio.py         # Main end-to-end automation
├── dataset.py                  # Dataset class for training
├── prepare_data.py             # Data preparation script
├── sft_12hz.py                 # Training script
├── venv/                       # Virtual environment (created automatically)
│   └── hf_cache/               # HuggingFace cache (isolated)
├── docs/
│   └── diagrams/
│       ├── architecture.drawio # System architecture diagram (source)
│       ├── architecture.svg    # System architecture diagram (rendered)
│       ├── data_flow.drawio    # Data flow diagram (source)
│       └── data_flow.svg       # Data flow diagram (rendered)
└── output/                     # Training output (created during training)
```

## License

This project includes code from the [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) project, which is licensed under the Apache License 2.0.

## References

- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- [WhisperX GitHub](https://github.com/m-bain/whisperX)
- [Qwen3-TTS Finetuning Guide](https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning)
