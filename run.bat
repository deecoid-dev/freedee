@echo off
cd /d "%~dp0"
python -c "import pygame, OpenGL" 2>nul
if errorlevel 1 (
    echo Installing requirements...
    python -m pip install -r requirements.txt
)
python main.py
if errorlevel 1 pause
