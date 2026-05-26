@echo off
REM ===========================================================================
REM PEMFC Simulator - one-click launcher for the Streamlit GUI
REM ===========================================================================
REM Behaviour:
REM   * If a PEMFC GUI server is already running on 8501..8510 (e.g. you closed
REM     the browser tab but the server console is still open), just open a new
REM     browser tab pointing at it and exit. No duplicate server is spawned.
REM   * Otherwise, find the first free port in 8501..8510 and start a fresh
REM     server there. Streamlit opens the browser automatically.
REM
REM End-user UX: double-click this file any time you want the GUI on screen.
REM ===========================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

REM --- Locate streamlit.exe ------------------------------------------------
set "ST="
if exist "C:\ProgramData\anaconda3\Scripts\streamlit.exe" set "ST=C:\ProgramData\anaconda3\Scripts\streamlit.exe"
if not defined ST if exist "%~dp0.venv\Scripts\streamlit.exe" set "ST=%~dp0.venv\Scripts\streamlit.exe"
if not defined ST if exist "%USERPROFILE%\anaconda3\Scripts\streamlit.exe" set "ST=%USERPROFILE%\anaconda3\Scripts\streamlit.exe"
if not defined ST if exist "%USERPROFILE%\miniconda3\Scripts\streamlit.exe" set "ST=%USERPROFILE%\miniconda3\Scripts\streamlit.exe"
if not defined ST (
    echo Could not locate streamlit.exe. Install it once with:
    echo    pip install streamlit
    echo or with Anaconda:
    echo    conda install -c conda-forge streamlit
    pause
    exit /b 1
)

REM --- Step 1: probe 8501..8510 for a running Streamlit health endpoint ----
set "REUSE="
for %%P in (8501 8502 8503 8504 8505 8506 8507 8508 8509 8510) do (
    if not defined REUSE call :probe %%P
)
if defined REUSE goto :reuse

REM --- Step 2: no existing server -- pick the first free port --------------
set "PORT="
for %%P in (8501 8502 8503 8504 8505 8506 8507 8508 8509 8510) do (
    if not defined PORT call :checkfree %%P
)
if not defined PORT set "PORT=8501"

echo ============================================================
echo  Starting PEMFC Simulator GUI ...
echo    Streamlit : %ST%
echo    App       : %CD%\gui\app.py
echo    Port      : %PORT%
echo ============================================================
echo.
echo (Your browser will open automatically. Close this window to stop.)
echo.
"%ST%" run gui\app.py --server.port %PORT%
exit /b %errorlevel%


REM ===========================================================================
REM Subroutines
REM ===========================================================================

:probe
REM Issue a 1-second HTTP GET against /_stcore/health. Streamlit returns "ok"
REM with status 200 if a server is alive. Any error -> port is dormant.
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:%1/_stcore/health' -UseBasicParsing -TimeoutSec 1 -ErrorAction Stop; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 set "REUSE=%1"
goto :eof

:checkfree
REM We only care about LISTENING sockets (a server bound to the port).
REM Filter netstat to LISTENING rows first, then look for ":PORT " (literal
REM port-then-space) to avoid 8501 matching 85011, and to skip SYN_SENT /
REM ESTABLISHED rows that have ":8501" only as the remote endpoint.
netstat -ano | findstr "LISTENING" | findstr /C:":%1 " >nul
if errorlevel 1 set "PORT=%1"
goto :eof

:reuse
echo ============================================================
echo  Existing PEMFC GUI detected on port !REUSE!.
echo  Opening a new browser tab at http://localhost:!REUSE!/
echo ============================================================
echo.
echo  (The server keeps running in its original console window.
echo   Close that window if you want to stop the simulator.)
echo.
start "" "http://localhost:!REUSE!/"
timeout /t 3 /nobreak >nul
exit /b 0
