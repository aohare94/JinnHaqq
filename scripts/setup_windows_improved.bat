@echo off
echo ============================================================================
echo           QURANIC AI ALIGNMENT PROJECT v2.0 - WINDOWS SETUP (IMPROVED)
echo ============================================================================
echo Setting up environment for Windows 11 + RTX 4070 Super
echo.

REM Check if Python is installed and get version
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.9-3.11 from python.org
    echo Python 3.12+ may have compatibility issues with some packages
    pause
    exit /b 1
)

REM Display Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python %PYTHON_VERSION%

REM Check Python version compatibility
python -c "import sys; exit(0 if (3,9) <= sys.version_info < (3,12) else 1)" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Python %PYTHON_VERSION% may have compatibility issues.
    echo Recommended: Python 3.9, 3.10, or 3.11
    set /p continue="Continue anyway? (y/n): "
    if /i not "%continue%"=="y" (
        echo Setup cancelled.
        pause
        exit /b 1
    )
)

REM Check if NVIDIA GPU is available
echo Checking NVIDIA GPU...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo WARNING: nvidia-smi not found. Please ensure:
    echo 1. NVIDIA GPU drivers are installed (version 525.85+ for CUDA 12.1)
    echo 2. CUDA Toolkit 12.1+ is installed from NVIDIA website
    echo.
    set /p continue="Continue with CPU-only setup? (y/n): "
    if /i not "%continue%"=="y" (
        echo Setup cancelled.
        pause
        exit /b 1
    )
    set USE_CUDA=false
) else (
    set USE_CUDA=true
    echo NVIDIA GPU detected:
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
)

REM Create virtual environment
echo.
echo Creating Python virtual environment...
if exist "quranic_alignment_env" (
    echo Virtual environment already exists. Removing old one...
    rmdir /s /q quranic_alignment_env
)

python -m venv quranic_alignment_env
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo Activating virtual environment...
call quranic_alignment_env\Scripts\activate.bat

REM Upgrade pip and essential tools
echo.
echo Upgrading pip and essential tools...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo WARNING: Failed to upgrade some tools, continuing...
)

REM Install PyTorch with multiple fallback options
echo.
echo Installing PyTorch...
if "%USE_CUDA%"=="true" (
    echo Attempting CUDA 12.1 installation...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 (
        echo CUDA 12.1 failed, trying CUDA 11.8...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        if errorlevel 1 (
            echo CUDA installations failed, falling back to CPU version...
            pip install torch torchvision torchaudio
            if errorlevel 1 (
                echo ERROR: All PyTorch installation attempts failed
                echo Please check your internet connection and try again
                pause
                exit /b 1
            )
        )
    )
) else (
    echo Installing CPU-only version...
    pip install torch torchvision torchaudio
    if errorlevel 1 (
        echo ERROR: PyTorch CPU installation failed
        pause
        exit /b 1
    )
)

REM Test PyTorch installation
echo.
echo Testing PyTorch installation...
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"
if errorlevel 1 (
    echo ERROR: PyTorch installation verification failed
    pause
    exit /b 1
)

REM Install Windows-compatible requirements
echo.
echo Installing Windows-compatible requirements...
if exist "requirements_windows.txt" (
    echo Using Windows-specific requirements file...
    pip install -r requirements_windows.txt
    if errorlevel 1 (
        echo Some packages failed to install, continuing with essential packages...
        pip install transformers>=4.37.0 accelerate>=0.25.0 peft>=0.7.0
        pip install sentence-transformers faiss-cpu scikit-learn scipy
        pip install datasets tokenizers arabic-reshaper python-bidi pyarabic
        pip install numpy pandas matplotlib seaborn plotly rich typer tqdm loguru
        pip install fastapi uvicorn gradio psutil python-dotenv pyyaml
        pip install huggingface-hub gitpython orjson
    )
) else (
    echo Installing essential packages manually...
    pip install transformers>=4.37.0 accelerate>=0.25.0 peft>=0.7.0
    pip install sentence-transformers faiss-cpu scikit-learn scipy
    pip install datasets tokenizers arabic-reshaper python-bidi pyarabic
    pip install numpy pandas matplotlib seaborn plotly rich typer tqdm loguru
    pip install fastapi uvicorn gradio psutil python-dotenv pyyaml
    pip install huggingface-hub gitpython orjson
)

REM Try to install attention optimization (with fallbacks)
echo.
echo Installing attention optimization packages...
echo Attempting xFormers installation (Windows-compatible alternative to FlashAttention)...
pip install xformers
if errorlevel 1 (
    echo xFormers installation failed, trying alternative packages...
    pip install einops
    echo Will use standard attention mechanisms
)

REM Install optional GPU acceleration (if CUDA available)
if "%USE_CUDA%"=="true" (
    echo.
    echo Installing optional GPU acceleration packages...
    pip install bitsandbytes
    if errorlevel 1 (
        echo bitsandbytes installation failed (this is common on Windows)
        echo Model quantization will use alternative methods
    )
)

REM Test final setup
echo.
echo ============================================================================
echo                           TESTING INSTALLATION
echo ============================================================================

echo Testing PyTorch CUDA availability...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU Only\"}')"

echo Testing xFormers...
python -c "try: import xformers; print('xFormers: Available (optimized attention enabled)'); except ImportError: print('xFormers: Not available - using standard attention')"

echo Testing transformers...
python -c "import transformers; print(f'Transformers {transformers.__version__} ready')"

echo Testing other key packages...
python -c "import sentence_transformers, faiss, numpy, pandas; print('Core packages ready')"

echo.
echo ============================================================================
echo                              SETUP COMPLETE!
echo ============================================================================
echo.
echo Environment: quranic_alignment_env
echo PyTorch: CUDA %USE_CUDA%
echo.
echo To activate the environment in the future:
echo     quranic_alignment_env\Scripts\activate.bat
echo.
echo To run the project:
echo     python -m src.core.engine
echo.
echo NEXT STEPS:
echo 1. Test with: python test_setup.py
echo 2. Download models: python -c "from transformers import AutoModel; AutoModel.from_pretrained('aubmindlab/bert-base-arabertv2')"
echo.
echo ============================================================================
pause