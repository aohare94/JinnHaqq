# Windows Setup Guide - Quranic AI Alignment Project v2.0

**Optimized for Windows 11 + RTX 4070 Super (12GB VRAM)**

## 🚀 Quick Start

### Prerequisites
1. **Windows 11** (recommended)
2. **Python 3.9, 3.10, or 3.11** (avoid 3.12+ for compatibility)
3. **NVIDIA GPU drivers** (version 525.85+ for CUDA 12.1)
4. **CUDA Toolkit 12.1+** from [NVIDIA website](https://developer.nvidia.com/cuda-downloads)

### Installation

1. **Run the improved setup script:**
   ```cmd
   scripts\setup_windows_improved.bat
   ```

2. **Verify installation:**
   ```cmd
   quranic_alignment_env\Scripts\activate.bat
   python test_windows_setup.py
   ```

## 🔧 What the New Setup Does

### Intelligent PyTorch Installation
- **CUDA 12.1 first** (primary target for RTX 4070 Super)
- **CUDA 11.8 fallback** (if 12.1 fails)
- **CPU-only fallback** (if all CUDA versions fail)

### Windows-Compatible Packages
- **xFormers instead of FlashAttention** (better Windows support)
- **Removes problematic packages** (graph-tool, deepspeed, etc.)
- **Essential packages only** for stable installation

### Smart Error Handling
- **Version compatibility checks** (Python 3.9-3.11 recommended)
- **GPU detection and fallbacks**
- **Package installation with retries**
- **Comprehensive testing at the end**

## 📋 Key Differences from Original Setup

| Issue | Original | Improved |
|-------|----------|----------|
| PyTorch Index | Single CUDA 12.1 only | Multiple fallbacks |
| Flash Attention | Attempted (fails on Windows) | xFormers alternative |
| Requirements | Full list (many fail) | Windows-specific subset |
| Error Handling | Minimal | Comprehensive with fallbacks |
| Testing | Basic | Full system verification |

## 🛠️ Troubleshooting

### PyTorch Installation Fails
```cmd
# Manual PyTorch installation options:

# Option 1: CUDA 12.1 (recommended for RTX 4070 Super)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Option 2: CUDA 11.8 (if 12.1 fails)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Option 3: CPU only (fallback)
pip install torch torchvision torchaudio
```

### xFormers Installation Fails
```cmd
# Try precompiled wheels
pip install xformers --no-deps

# If that fails, use einops for basic tensor operations
pip install einops
```

### CUDA Not Detected
1. **Check NVIDIA drivers:**
   ```cmd
   nvidia-smi
   ```

2. **Verify CUDA installation:**
   ```cmd
   nvcc --version
   ```

3. **Download latest drivers:**
   - Visit [NVIDIA Driver Downloads](https://www.nvidia.com/drivers/)
   - Select RTX 4070 Super / Windows 11

### Package Installation Issues
If individual packages fail:
```cmd
# Install essential packages manually
pip install transformers accelerate peft
pip install sentence-transformers faiss-cpu
pip install arabic-reshaper python-bidi pyarabic
```

## 🧪 Testing Your Setup

### Quick Test
```cmd
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Comprehensive Test
```cmd
python test_windows_setup.py
```

The test script checks:
- ✅ Python version compatibility
- ✅ PyTorch and CUDA functionality
- ✅ Transformers library
- ✅ Attention optimization (xFormers)
- ✅ Arabic text processing
- ✅ Core ML packages
- ✅ GPU memory (should show ~12GB for RTX 4070 Super)
- ✅ Project structure

## 🎯 Expected Results

### Successful Setup Should Show:
```
CUDA Available: True
CUDA Device: NVIDIA GeForce RTX 4070 SUPER
GPU Memory: ~12GB
xFormers: Available (optimized attention enabled)
```

### Performance Expectations:
- **Model Loading:** 3-10 seconds for BERT-base models
- **Inference Speed:** 50-200 tokens/second (depends on model size)
- **Memory Usage:** Efficiently uses 12GB VRAM for large models

## 🚦 What's Different for RTX 4070 Super

### Optimizations:
- **CUDA 12.1 support** for latest optimizations
- **12GB VRAM utilization** for larger models
- **xFormers attention** for memory efficiency
- **Mixed precision training** enabled by default

### Limitations on Windows:
- **No FlashAttention-2** (Linux/WSL only)
- **No DeepSpeed** (limited Windows support)
- **Some quantization libraries** may not work

## 📚 Next Steps

### 1. Download Arabic Models
```cmd
python -c "from transformers import AutoModel; AutoModel.from_pretrained('aubmindlab/bert-base-arabertv2')"
```

### 2. Run the Application
```cmd
python -m src.core.engine
```

### 3. Optimize for Your Use Case
- Adjust batch sizes for your 12GB VRAM
- Enable mixed precision for faster training
- Use xFormers for memory-efficient attention

## 🔍 Still Having Issues?

### Common Solutions:
1. **Restart your terminal** after CUDA installation
2. **Update Windows** to latest version
3. **Use Windows Subsystem for Linux (WSL2)** for better compatibility
4. **Check antivirus software** - some block CUDA operations

### WSL2 Alternative:
If Windows continues to have issues:
```cmd
# Install WSL2 with Ubuntu
wsl --install -d Ubuntu

# Then follow Linux setup instructions
```

### Getting Help:
- Check `test_windows_setup.py` output for specific failing components
- Verify GPU drivers with Device Manager
- Ensure Python virtual environment is activated
- Check Windows Event Viewer for system-level errors

---

**Note:** This setup is specifically optimized for Windows 11 + RTX 4070 Super. For other configurations, you may need to adjust CUDA versions or memory settings.