@echo off
setlocal EnableDelayedExpansion
title MosaicPy Demo

:: Move to the folder containing this script
cd /d "%~dp0"

:: ── Locate or install uv ──────────────────────────────────────────────────────
where uv >nul 2>&1
if %errorlevel% equ 0 goto :launch

:: Not on PATH — check the default Windows install location
if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "PATH=%USERPROFILE%\.local\bin;!PATH!"
    goto :launch
)

echo uv package manager not found. Installing now (one-time, requires internet)...
echo.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Could not install uv automatically.
    echo Please install it manually: https://docs.astral.sh/uv/getting-started/installation/
    pause
    exit /b 1
)
set "PATH=%USERPROFILE%\.local\bin;!PATH!"

where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ERROR: uv was installed but is not reachable in this session.
    echo Close this window, then double-click run_mosaic.bat again.
    pause
    exit /b 1
)

:launch
echo Starting MosaicPy Demo...
echo (First launch downloads and installs dependencies -- this takes 2-3 minutes.)
echo Subsequent launches are instant.
echo.

uv run --link-mode copy python -m mosaic.gui.app

if %errorlevel% neq 0 (
    echo.
    echo MosaicPy exited with an error. See the messages above for details.
    pause
)
