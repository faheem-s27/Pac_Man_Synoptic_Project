@echo off
title Pac-Man AI Project - Setup
echo.
echo  ==========================================
echo       Pac-Man AI Project - First Setup
echo  ==========================================
echo.

:: Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found.
    echo  Install Python 3.11 from https://www.python.org/downloads/
    echo  Make sure to tick "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo  Python found:
python --version
echo.

:: Install dependencies
echo  Installing required packages...
echo  (This may take a few minutes on first run)
echo.
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo  ERROR: Package installation failed.
    echo  Try running this file as Administrator, or check your internet connection.
    pause
    exit /b 1
)

echo.
echo  ==========================================
echo   All packages installed successfully.
echo.
echo   NOTE: PyTorch installs CPU-only by default.
echo   For GPU-accelerated training (recommended),
echo   run this command manually after setup:
echo.
echo     pip install torch --index-url https://download.pytorch.org/whl/cu121
echo.
echo   (Requires an NVIDIA GPU with CUDA 12.1+)
echo  ==========================================
echo.
echo  Setup complete. Run run.bat to start the launcher.
echo.
pause
