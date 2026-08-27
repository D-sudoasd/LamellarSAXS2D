@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "PYTHONW=%PROJECT_DIR%.venv-project\Scripts\pythonw.exe"

if not exist "%PYTHONW%" (
    echo Verified project environment was not found.
    echo Expected file: "%PYTHONW%"
    echo Repair command:
    echo   python -m venv "%PROJECT_DIR%.venv-project"
    echo   cd /d "%PROJECT_DIR%"
    echo   .venv-project\Scripts\python.exe -m pip install -c constraints\validation-py311-313.txt -e ".[all]"
    pause
    exit /b 1
)

start "" /D "%PROJECT_DIR%" "%PYTHONW%" -m butterfly_saxs gui
if errorlevel 1 (
    echo Failed to start LamellarSAXS2D.
    pause
    exit /b 1
)

exit /b 0
