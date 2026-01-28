#!/usr/bin/env bash
# Qwen3-TTS Complete Setup Script
# This script sets up everything needed for the fine-tuning pipeline
#
# Requirements:
#   - Python 3.12
#   - CUDA 12.x (for GPU support)
#
# Usage:
#   ./setup.sh          # Interactive mode (prompts for model download)
#   ./setup.sh --auto   # Non-interactive mode (skips model download prompt)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
VENV_DIR="venv"
PYTHON_VERSION="python3.12"

# Specific versions
PYTORCH_VERSION="2.8.0"
TORCHVISION_VERSION="0.23.0"
TORCHAUDIO_VERSION="2.8.0"
FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

# Parse arguments
AUTO_MODE=false
if [[ "$1" == "--auto" ]]; then
    AUTO_MODE=true
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Qwen3-TTS Setup Script${NC}"
if [ "$AUTO_MODE" = true ]; then
    echo -e "${YELLOW}(Running in auto mode)${NC}"
fi
echo -e "${GREEN}========================================${NC}"
echo ""

# Check for Python 3.12
echo -e "${GREEN}Checking Python version...${NC}"
if ! command -v $PYTHON_VERSION &> /dev/null; then
    echo -e "${RED}Error: $PYTHON_VERSION not found${NC}"
    echo -e "${YELLOW}Please install Python 3.12:${NC}"
    echo -e "  Ubuntu/Debian: sudo apt install python3.12 python3.12-venv"
    echo -e "  Or use pyenv: pyenv install 3.12"
    exit 1
fi

PYTHON_FULL_VERSION=$($PYTHON_VERSION --version)
echo -e "${GREEN}Found: $PYTHON_FULL_VERSION${NC}"
echo ""

# Check for sox (required by qwen-tts)
echo -e "${GREEN}Checking for SoX...${NC}"
if ! command -v sox &> /dev/null; then
    echo -e "${RED}Error: SoX not found${NC}"
    echo -e "${YELLOW}SoX is required by qwen-tts for audio processing.${NC}"
    echo -e "${YELLOW}Please install SoX:${NC}"
    echo -e "  Ubuntu/Debian: sudo apt install sox libsox-fmt-all"
    echo -e "  RHEL/CentOS:   sudo yum install sox"
    echo -e "  macOS:         brew install sox"
    exit 1
fi
echo -e "${GREEN}Found: $(sox --version | head -1)${NC}"
echo ""

# Detect CUDA
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d':' -f2 | tr -d ' ')
    echo -e "${GREEN}Detected CUDA Version: $CUDA_VERSION${NC}"
    HAS_CUDA=true
else
    echo -e "${YELLOW}No CUDA detected. Will install CPU-only PyTorch.${NC}"
    HAS_CUDA=false
fi
echo ""

# Create virtual environment
echo -e "${GREEN}Step 1: Creating virtual environment${NC}"
if [ ! -d "$VENV_DIR" ]; then
    $PYTHON_VERSION -m venv "$VENV_DIR"
    echo -e "${GREEN}Virtual environment created at $VENV_DIR${NC}"
else
    echo -e "${YELLOW}Virtual environment already exists at $VENV_DIR${NC}"
fi
echo ""

# Activate virtual environment
echo -e "${GREEN}Step 2: Activating virtual environment${NC}"
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}Virtual environment activated${NC}"
echo ""

# Upgrade pip
echo -e "${GREEN}Step 3: Upgrading pip${NC}"
pip install --upgrade pip setuptools wheel -q
echo -e "${GREEN}pip upgraded${NC}"
echo ""

# Install PyTorch
echo -e "${GREEN}Step 4: Installing PyTorch ${PYTORCH_VERSION}${NC}"
if [ "$HAS_CUDA" = true ]; then
    echo -e "${YELLOW}Installing PyTorch with CUDA 12.8 support...${NC}"
    pip install torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url https://download.pytorch.org/whl/cu128
else
    echo -e "${YELLOW}Installing CPU-only PyTorch...${NC}"
    pip install torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url https://download.pytorch.org/whl/cpu
fi
echo -e "${GREEN}PyTorch ${PYTORCH_VERSION} installed${NC}"
echo ""

# Install core dependencies
echo -e "${GREEN}Step 5: Installing core dependencies${NC}"
pip install -q \
    numpy \
    librosa \
    soundfile \
    tqdm \
    transformers \
    accelerate \
    safetensors \
    datasets \
    evaluate \
    huggingface-hub \
    hf_transfer

echo -e "${GREEN}Core dependencies installed${NC}"
echo ""

# Install qwen-tts
echo -e "${GREEN}Step 6: Installing qwen-tts${NC}"
pip install qwen-tts -q
echo -e "${GREEN}qwen-tts installed${NC}"
echo ""

# Install WhisperX
echo -e "${GREEN}Step 7: Installing WhisperX${NC}"
pip install -q whisperx
echo -e "${GREEN}WhisperX installed${NC}"
echo ""

# Install additional dependencies
echo -e "${GREEN}Step 8: Installing additional dependencies${NC}"
pip install -q \
    ffmpeg-python \
    gruut \
    cn2an \
    pypinyin \
    jieba \
    nemo_text_processing

echo -e "${GREEN}Additional dependencies installed${NC}"
echo ""

# Install flash-attn (optional, for faster training)
echo -e "${GREEN}Step 9: Installing flash-attn (optional)${NC}"
if [ "$HAS_CUDA" = true ]; then
    echo -e "${YELLOW}Installing flash-attn from pre-built wheel...${NC}"
    echo -e "${YELLOW}URL: ${FLASH_ATTN_WHEEL}${NC}"
    pip install "${FLASH_ATTN_WHEEL}" 2>/dev/null && \
        echo -e "${GREEN}flash-attn installed successfully${NC}" || \
        echo -e "${YELLOW}flash-attn installation failed - will use eager attention (slower but compatible)${NC}"
else
    echo -e "${YELLOW}Skipping flash-attn (CPU-only mode)${NC}"
fi
echo ""

# Pre-download models (optional - saves time during training)
echo -e "${YELLOW}Step 10: Pre-downloading models (this may take a while)...${NC}"

if [ "$AUTO_MODE" = true ]; then
    # Auto mode: skip model pre-download (will download on first run)
    echo -e "${YELLOW}Auto mode: Skipping model pre-download. Models will be downloaded during first run.${NC}"
else
    # Interactive mode: ask user
    read -p "Do you want to pre-download models now? This saves time during training. (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then

        echo -e "${GREEN}Downloading Qwen3-TTS Tokenizer...${NC}"
        python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Qwen/Qwen3-TTS-Tokenizer-12Hz', local_dir='./models/Qwen3-TTS-Tokenizer-12Hz')
" 2>/dev/null || echo "Tokenizer download will happen during first run"

        echo -e "${GREEN}Downloading Qwen3-TTS Base Model...${NC}"
        python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Qwen/Qwen3-TTS-12Hz-1.7B-Base', local_dir='./models/Qwen3-TTS-12Hz-1.7B-Base')
" 2>/dev/null || echo "Base model download will happen during first run"

        echo -e "${GREEN}Downloading WhisperX model...${NC}"
        python -c "
import whisperx
whisperx.load_model('large-v3', device='cpu')
" 2>/dev/null || echo "WhisperX model download will happen during first run"

        echo -e "${GREEN}Models downloaded to ./models/${NC}"
    else
        echo -e "${YELLOW}Skipping model pre-download. Models will be downloaded during first run.${NC}"
    fi
fi
echo ""

# Configure HuggingFace cache in venv
echo -e "${GREEN}Step 11: Configuring HuggingFace cache${NC}"
mkdir -p "$VENV_DIR/hf_cache"

# Add HF environment variables to activate script
if ! grep -q "HF_HOME" "$VENV_DIR/bin/activate" 2>/dev/null; then
    cat >> "$VENV_DIR/bin/activate" << 'HFEOF'

# HuggingFace cache configuration (added by setup.sh)
export HF_HOME="${VIRTUAL_ENV}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
HFEOF
    echo -e "${GREEN}HuggingFace cache configured at $VENV_DIR/hf_cache${NC}"
else
    echo -e "${YELLOW}HuggingFace cache already configured${NC}"
fi
echo ""

# Create activation script for convenience
echo -e "${GREEN}Step 12: Creating activation helper${NC}"
cat > activate.sh << 'EOF'
#!/usr/bin/env bash
source venv/bin/activate
echo "Virtual environment activated."
echo "Python: $(python --version)"
echo "HF_HOME: $HF_HOME"
echo "Use 'deactivate' to exit."
EOF
chmod +x activate.sh
echo -e "${GREEN}Created 'activate.sh' for easy venv activation${NC}"
echo ""

# Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Configuration:"
echo -e "  Python:    $PYTHON_FULL_VERSION"
echo -e "  PyTorch:   ${PYTORCH_VERSION} (CUDA 12.8)"
echo -e "  venv:      $VENV_DIR/"
echo -e "  HF cache:  $VENV_DIR/hf_cache/"
echo ""
echo -e "To activate the virtual environment, run:"
echo -e "  ${YELLOW}source activate.sh${NC}"
echo -e "  or:"
echo -e "  ${YELLOW}source venv/bin/activate${NC}"
echo ""
echo -e "To run fine-tuning:"
echo -e "  ${YELLOW}./train.sh --audio_dir ./audio_files --ref_audio ./reference.wav --speaker_name my_voice${NC}"
echo ""
echo -e "To deactivate:"
echo -e "  ${YELLOW}deactivate${NC}"
echo ""
