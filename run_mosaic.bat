@echo off
setlocal EnableDelayedExpansion
title Mosaic

:: Move to the folder containing this script
cd /d "%~dp0"

:: ── Check Microsoft VC++ Runtime (required by Dear PyGui) ─────────────────────
:: On fresh Windows installs this is often missing and produces a cryptic
:: "DLL load failed while importing _dearpygui" trace. Catch it here instead.
if not exist "%SystemRoot%\System32\vcruntime140.dll" goto :need_vcrt
if not exist "%SystemRoot%\System32\msvcp140.dll" goto :need_vcrt
goto :find_uv

:need_vcrt
echo.
echo Mosaic needs a small Microsoft system component (the VC++ Runtime)
echo that is not installed on this PC. We'll open the download page now.
echo.
echo Steps:
echo   1) Save and run VC_redist.x64.exe from the page that opens
echo   2) Click Yes if Windows asks for permission
echo   3) Close this window, then double-click run_mosaic.bat again
echo.
start "" "https://aka.ms/vs/17/release/vc_redist.x64.exe"
pause
exit /b 1

:find_uv
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
echo Starting Mosaic...
if not exist ".venv" (
    echo First launch: downloading and installing dependencies. This takes
    echo 2-3 minutes. Subsequent launches are instant.
)
echo.

uv run python -m mosaic

if %errorlevel% neq 0 (
    echo.
    echo Mosaic exited with an error. See the messages above for details.
    pause
)
