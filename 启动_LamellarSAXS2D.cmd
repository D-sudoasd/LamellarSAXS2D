@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul 2>&1

set "PROJECT_DIR=%~dp0"
set "PYTHON="
set "PYTHON_ARGS="
set "CHECK_ONLY=0"
if /I "%~1"=="--check" set "CHECK_ONLY=1"
set "PYTHONPATH=%PROJECT_DIR%src;%PYTHONPATH%"

rem Prefer project environments when available.
for %%V in (.venv-project .venv venv) do call :try_project_python "%%V"
if defined PYTHON goto :python_ready

rem Then ask the Python launcher for a supported interpreter. This avoids
rem the Microsoft Store python.exe alias returned by where python.
for %%V in (3.13 3.12 3.11) do call :try_launcher "%%V"
if defined PYTHON_ARGS goto :python_ready

echo LamellarSAXS2D could not find Python.
echo 未找到 Python 解释器。
echo.
echo Create a supported project environment from this directory:
echo   py -3.13 -m venv ".venv-project"
echo   ".venv-project\Scripts\python.exe" -m pip install -c constraints\validation-py311-313.txt -e ".[all]"
if "%CHECK_ONLY%"=="0" pause
exit /b 1

:python_ready
set "PYTHONPATH=%PROJECT_DIR%src;%PYTHONPATH%"
if defined PYTHON goto :doctor_project
%PYTHON_ARGS% -m butterfly_saxs.doctor --require-ui --quiet
set "DOCTOR_STATUS=%ERRORLEVEL%"
goto :doctor_status

:doctor_project
"%PYTHON%" -m butterfly_saxs.doctor --require-ui --quiet
set "DOCTOR_STATUS=%ERRORLEVEL%"

:doctor_status
if "%DOCTOR_STATUS%"=="0" goto :doctor_ok
echo LamellarSAXS2D environment check failed.
echo 环境检查未通过：
echo.
if defined PYTHON goto :doctor_project_verbose
%PYTHON_ARGS% -m butterfly_saxs.doctor --require-ui
goto :doctor_verbose_done

:doctor_project_verbose
"%PYTHON%" -m butterfly_saxs.doctor --require-ui

:doctor_verbose_done
echo.
echo Run the repair command above from:
echo   "%PROJECT_DIR%"
if "%CHECK_ONLY%"=="0" pause
exit /b 1

:doctor_ok
if "%CHECK_ONLY%"=="1" (
    echo LamellarSAXS2D environment check passed.
    exit /b 0
)
if defined PYTHON goto :start_project_gui
start "" /D "%PROJECT_DIR%" %PYTHON_ARGS% -m butterfly_saxs.ui.launcher %*
goto :start_status

:start_project_gui
for %%P in ("%PYTHON%") do set "PYTHONW=%%~dpPpythonw.exe"
if exist "%PYTHONW%" goto :start_project_gui_ready
set "PYTHONW=%PYTHON%"
:start_project_gui_ready
start "" /D "%PROJECT_DIR%" "%PYTHONW%" -m butterfly_saxs.ui.launcher %*

:start_status
if not errorlevel 1 exit /b 0
echo Failed to start LamellarSAXS2D.
echo GUI startup diagnostics are written to the per-user launcher log.
if "%CHECK_ONLY%"=="0" pause
exit /b 1

:try_project_python
if defined PYTHON exit /b 0
if not exist "%PROJECT_DIR%%~1\Scripts\python.exe" exit /b 0
set "PYTHON=%PROJECT_DIR%%~1\Scripts\python.exe"
exit /b 0

:try_launcher
if defined PYTHON_ARGS exit /b 0
py -%~1 --version >nul 2>&1
if errorlevel 1 exit /b 0
py -%~1 -c "import butterfly_saxs; import PySide6, pyqtgraph, fabio, pyFAI" >nul 2>&1
if errorlevel 1 exit /b 0
set "PYTHON_ARGS=py -%~1"
exit /b 0
