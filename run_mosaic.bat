@echo off
setlocal
title Mosaic

:: No EnableDelayedExpansion: it strips "!" from %~dp0.
:: %~dp0 also stays out of ( ) blocks -- cmd expands at parse time, so a ")"
:: in the path closes the block early. Branch with labels instead.

:: Move to the folder containing this script
cd /d "%~dp0"
if not errorlevel 1 goto :in_folder
echo.
echo ERROR: Could not open the Mosaic folder:
echo   %~dp0
echo.
pause
exit /b 1

:in_folder

:: Explorer's zip preview copies only this file to a temp folder when opened
:: from inside the archive, leaving nothing else to run.
if not exist "pyproject.toml" (
    echo.
    echo ERROR: Mosaic's files are missing from this folder.
    echo.
    echo This usually means run_mosaic.bat was opened directly from inside
    echo the .zip. Extract the whole zip to a real folder first, then run
    echo run_mosaic.bat from the extracted copy.
    echo.
    pause
    exit /b 1
)

:: ---- Keep the environment off OneDrive ---------------------------------------
:: OneDrive holds handles on freshly written .pyd/.dll files, and hardlinks into
:: a reparse point fail with "Access is denied". LOCALAPPDATA is never synced.
if not defined UV_PROJECT_ENVIRONMENT set "UV_PROJECT_ENVIRONMENT=%LOCALAPPDATA%\Mosaic\venv"
if not defined UV_CACHE_DIR set "UV_CACHE_DIR=%LOCALAPPDATA%\Mosaic\uv-cache"
if not defined UV_LINK_MODE set "UV_LINK_MODE=copy"

:: Substring tests rather than "find": Git/MSYS on PATH shadows Windows find.exe
:: with the Unix one. Substitution is case-insensitive, so no /i is needed.
set "MOSAIC_DIR=%~dp0"
set "MOSAIC_SYNC="
if not "%MOSAIC_DIR%"=="%MOSAIC_DIR:OneDrive=%" set "MOSAIC_SYNC=OneDrive"
if not "%MOSAIC_DIR%"=="%MOSAIC_DIR:Dropbox=%" set "MOSAIC_SYNC=Dropbox"
if not "%MOSAIC_DIR%"=="%MOSAIC_DIR:Google Drive=%" set "MOSAIC_SYNC=Google Drive"
if not "%MOSAIC_DIR%"=="%MOSAIC_DIR:GoogleDrive=%" set "MOSAIC_SYNC=Google Drive"
:: Anchored: a bare "Box" would also match "Dropbox".
if not "%MOSAIC_DIR%"=="%MOSAIC_DIR:\Box\=%" set "MOSAIC_SYNC=Box"
if not "%MOSAIC_DIR%"=="%MOSAIC_DIR:iCloudDrive=%" set "MOSAIC_SYNC=iCloud Drive"

if defined MOSAIC_SYNC (
    echo.
    echo Note: this folder is inside %MOSAIC_SYNC%. Mosaic keeps its Python
    echo environment outside it, so the install itself is fine, but saved maps
    echo and exports here will sync. A plain folder such as C:\Mosaic avoids
    echo that.
    echo.
)

:: ---- Check Microsoft VC++ Runtime (required by Dear PyGui) -------------------
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
:: ---- Locate or install uv ----------------------------------------------------
where uv >nul 2>&1
if %errorlevel% equ 0 goto :check_uv_version

:: Not on PATH -- check the default Windows install location
if not exist "%USERPROFILE%\.local\bin\uv.exe" goto :install_uv
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
goto :launch

:install_uv
echo uv package manager not found. Installing now (one-time, requires internet)...
echo.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

:: Test for the file, not the exit code: powershell -c returns 0 even when
:: Group Policy or AMSI blocks the download.
if not exist "%USERPROFILE%\.local\bin\uv.exe" (
    echo.
    echo ERROR: uv could not be downloaded.
    echo.
    echo This is usually a corporate firewall, a TLS-inspecting proxy, or a
    echo PowerShell execution policy set by your IT department -- retrying
    echo will not help.
    echo.
    echo Workaround: download uv manually from
    echo   https://github.com/astral-sh/uv/releases
    echo and place uv.exe in:
    echo   %USERPROFILE%\.local\bin\
    echo then double-click run_mosaic.bat again.
    echo.
    pause
    exit /b 1
)
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

:check_uv_version
:: uv.lock is lockfile revision 3; uv older than 0.5 rejects it with a message
:: no user can act on. Prefix tests only, so an unknown format falls through.
for /f "tokens=2" %%v in ('uv --version 2^>nul') do set "UV_VER=%%v"
if not defined UV_VER goto :launch
set "UV_P=%UV_VER:~0,4%"
set "UV_OLD="
if "%UV_P%"=="0.0." set "UV_OLD=1"
if "%UV_P%"=="0.1." set "UV_OLD=1"
if "%UV_P%"=="0.2." set "UV_OLD=1"
if "%UV_P%"=="0.3." set "UV_OLD=1"
if "%UV_P%"=="0.4." set "UV_OLD=1"
if not defined UV_OLD goto :launch
echo.
echo Your installed uv (%UV_VER%) is too old to read Mosaic's lock file.
echo.
echo Update it by running this command, then try again:
echo   uv self update
echo.
pause
exit /b 1

:launch
echo Starting Mosaic...
if not exist "%UV_PROJECT_ENVIRONMENT%" (
    echo First launch: downloading and installing dependencies. This takes
    echo 2-3 minutes and uses about 500 MB of disk. Later launches are instant.
)
echo.

uv run python -m mosaic

if %errorlevel% neq 0 (
    echo.
    echo Mosaic exited with an error. The details are in the messages above.
    pause
)
