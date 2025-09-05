@echo off
title SMOTE Image Synthesis Application
echo.
echo ================================================================
echo                SMOTE IMAGE SYNTHESIS APPLICATION
echo ================================================================
echo.
echo This application demonstrates synthetic image generation using 
echo SMOTE (Synthetic Minority Over-sampling Technique) for images.
echo.
echo Starting the application...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

REM Install requirements if needed
echo Checking requirements...
pip install -r requirements.txt >nul 2>&1

REM Run the application
python run_app.py

echo.
echo Application finished.
pause