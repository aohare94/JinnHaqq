@echo off
echo ============================================================================
echo           QURANIC AI ALIGNMENT PROJECT v2.0 - WINDOWS SETUP
echo ============================================================================
echo Setting up environment for Windows 11 + RTX 4070 Super
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.11+ from python.org or Microsoft Store
    echo Then run this script again.
    pause
    exit /b 1
)

REM Check if NVIDIA GPU is available
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo WARNING: nvidia-smi not found. Please ensure:
    echo 1. NVIDIA GPU drivers are installed
    echo 2. CUDA Toolkit 12.1+ is installed from NVIDIA website
    echo.
    set /p continue="Continue anyway? (y/n): "
    if /i not "%continue%"=="y" (
        echo Setup cancelled.
        pause
        exit /b 1
    )
)

REM Create virtual environment
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

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch with CUDA support
echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch with CUDA
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

REM Install xFormers for optimized attention
echo Installing xFormers for optimized attention...
pip install xformers
if errorlevel 1 (
    echo WARNING: Failed to install xFormers
    echo Continuing with standard attention mechanisms...
)

REM Install other requirements
echo Installing project requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements
    echo Please check requirements.txt and try again
    pause
    exit /b 1
)

REM Test CUDA availability
echo Testing CUDA availability...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

REM Test xFormers
echo Testing xFormers...
python -c "try: import xformers; print('xFormers: Available'); except ImportError: print('xFormers: Not available - using standard attention')"

echo.
echo ============================================================================
echo                              SETUP COMPLETE!
echo ============================================================================
echo.
echo To activate the environment in the future, run:
echo     quranic_alignment_env\Scripts\activate.bat
echo.
echo To start the interactive terminal:
echo     python -m quranic_alignment.terminal
echo.
echo To run the setup script for Qur'an data:
echo     python scripts\setup_quran_data.py
echo.
echo ============================================================================
pause