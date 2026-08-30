@echo off
setlocal
chcp 65001 >nul 2>&1

set "PROJECT_DIR=%~dp0"
set "PYTHON="
set "CHECK_ONLY=0"

if /I "%~1"=="--check" set "CHECK_ONLY=1"

for %%V in (.venv-project .venv venv) do (
    if not defined PYTHON if exist "%PROJECT_DIR%%%V\Scripts\python.exe" (
        set "PYTHON=%PROJECT_DIR%%%V\Scripts\python.exe"
    )
)

if not defined PYTHON (
    for /f "delims=" %%P in ('where python 2^>nul') do (
        if not defined PYTHON set "PYTHON=%%P"
    )
)

if not defined PYTHON (
    echo LamellarSAXS2D could not find Python.
    echo 未找到 Python 解释器。
    echo.
    echo Create a supported project environment from this directory:
    echo   py -3.13 -m venv ".venv-project"
    echo   ".venv-project\Scripts\python.exe" -m pip install -c constraints\validation-py311-313.txt -e ".[all]"
    if "%CHECK_ONLY%"=="0" pause
    exit /b 1
)

"%PYTHON%" -m butterfly_saxs.doctor --require-ui --quiet
if errorlevel 1 (
    echo LamellarSAXS2D environment check failed.
    echo 环境检查未通过：
    echo.
    "%PYTHON%" -m butterfly_saxs.doctor --require-ui
    echo.
    echo Run the repair command above from:
    echo   "%PROJECT_DIR%"
    if "%CHECK_ONLY%"=="0" pause
    exit /b 1
)

if "%CHECK_ONLY%"=="1" (
    echo LamellarSAXS2D environment check passed.
    exit /b 0
)

for %%P in ("%PYTHON%") do set "PYTHONW=%%~dpPpythonw.exe"
if not exist "%PYTHONW%" set "PYTHONW=%PYTHON%"

start "" /D "%PROJECT_DIR%" "%PYTHONW%" -m butterfly_saxs.ui.launcher %*
if errorlevel 1 (
    echo Failed to start LamellarSAXS2D.
    echo GUI startup diagnostics are written to the per-user launcher log.
    if "%CHECK_ONLY%"=="0" pause
    exit /b 1
)

exit /b 0
