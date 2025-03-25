@echo off
setlocal enabledelayedexpansion
set VENV_DIR=venv

:: Create virtual environment if not exists
if not exist %VENV_DIR% (
    echo Creating virtual environment...
    python -m venv %VENV_DIR%
)

:: Activate virtual environment
call %VENV_DIR%\Scripts\activate

:: Install dependencies
echo Installing Python dependencies...
pip install --upgrade pip
pip install -r requirements.txt

:: Check for Jupyter mode
set JUPYTER_MODE=false
if "%1"=="--jupyter" (
    set JUPYTER_MODE=true
    shift
)

if "%JUPYTER_MODE%"=="true" (
    echo Starting Jupyter Notebook for run_demo.ipynb...
    jupyter notebook run_demo.ipynb
    goto end
)

:: Parse arguments
set PROCESS=all
set TARGET=
set WITH_RAINFALL=
set CONFIG_FILE=

:parse_args
if "%1"=="" goto run_main
if "%1"=="--process" (set PROCESS=%2 & shift & shift & goto parse_args)
if "%1"=="--target" (set TARGET=--target %2 & shift & shift & goto parse_args)
if "%1"=="--with_rainfall" (set WITH_RAINFALL=--with_rainfall & shift & goto parse_args)
if "%1"=="--config" (set CONFIG_FILE=--config %2 & shift & shift & goto parse_args)
echo Unknown parameter passed: %1
exit /b 1

:run_main
:: main.pyが正しい場所にあるか確認
set MAIN_SCRIPT=src\main.py
if not exist %MAIN_SCRIPT% (
    set MAIN_SCRIPT=main.py
)

echo Running: python %MAIN_SCRIPT% --process %PROCESS% %TARGET% %WITH_RAINFALL% %CONFIG_FILE%
python %MAIN_SCRIPT% --process %PROCESS% %TARGET% %WITH_RAINFALL% %CONFIG_FILE%

:end
echo Done.