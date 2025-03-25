@echo off
setlocal enabledelayedexpansion
set VENV_DIR=venv

:: -----------------------------
:: オプション初期化
:: -----------------------------
set DOCKER_MODE=false
set JUPYTER_MODE=false
set PROCESS=all
set TARGET=
set WITH_RAINFALL=
set CONFIG_FILE=

:: -----------------------------
:: ヘルプ表示
:: -----------------------------
if "%1"=="--help" (
    echo.
    echo [Water Level Forecast CLI ヘルプ]
    echo ---------------------------------
    echo --process [step]       実行ステップを指定（all, train, predictなど）
    echo --target [name]        対象の頭首工（例：Headworks_A）
    echo --with_rainfall        降雨データを使う場合に指定
    echo --config [path]        特徴量設定ファイルのパス
    echo --docker               Dockerで実行（Python環境不要）
    echo --jupyter              Jupyter Notebook を起動
    echo --help                 本ヘルプを表示
    echo.
    exit /b 0
)

:: -----------------------------
:: 特殊モードの検出
:: -----------------------------
if "%1"=="--docker" (
    set DOCKER_MODE=true
    shift
)
if "%1"=="--jupyter" (
    set JUPYTER_MODE=true
    shift
)

:: -----------------------------
:: 引数解析
:: -----------------------------
:parse_args
if "%1"=="" goto run_main
if "%1"=="--process" (set PROCESS=%2 & shift & shift & goto parse_args)
if "%1"=="--target" (set TARGET=--target %2 & shift & shift & goto parse_args)
if "%1"=="--with_rainfall" (set WITH_RAINFALL=--with_rainfall & shift & goto parse_args)
if "%1"=="--config" (set CONFIG_FILE=--config %2 & shift & shift & goto parse_args)
echo ❌ Unknown parameter passed: %1
exit /b 1

:: -----------------------------
:: メイン処理
:: -----------------------------
:run_main

:: Jupyter モード
if "%JUPYTER_MODE%"=="true" (
    where jupyter >nul 2>&1
    if errorlevel 1 (
        echo ❌ Jupyter がインストールされていません。`pip install notebook` を実行してください。
        exit /b 1
    )
    echo 🚀 Starting Jupyter Notebook...
    jupyter notebook run_demo.ipynb
    goto end
)

:: Docker モード
if "%DOCKER_MODE%"=="true" (
    where docker >nul 2>&1
    if errorlevel 1 (
        echo ❌ Docker がインストールされていません。https://www.docker.com/ からインストールしてください。
        exit /b 1
    )
    echo 🐳 Executing via Docker...
    docker run --rm ^
        -v "G:/マイドライブ/農林水産省/データファイル:/app/G/マイドライブ/農林水産省/データファイル" ^
        water-level-forecast ^
        python main.py --process %PROCESS% %TARGET% %WITH_RAINFALL% %CONFIG_FILE%
    goto end
)

:: Python 環境チェック
where python >nul 2>&1
if errorlevel 1 (
    echo ❌ Python が見つかりません。PATHを確認するか、Pythonをインストールしてください。
    exit /b 1
)

:: 仮想環境作成（初回のみ）
if not exist %VENV_DIR% (
    echo 🔧 Creating virtual environment...
    python -m venv %VENV_DIR%
)

:: 仮想環境を有効化
call %VENV_DIR%\Scripts\activate

:: 依存パッケージのインストール
echo 📦 Installing Python dependencies...
pip install --upgrade pip >nul
pip install -r requirements.txt >nul

:: main.pyの位置を確認
set MAIN_SCRIPT=src\main.py
if not exist %MAIN_SCRIPT% (
    set MAIN_SCRIPT=main.py
)

:: 実行
echo 🚀 Running: python %MAIN_SCRIPT% --process %PROCESS% %TARGET% %WITH_RAINFALL% %CONFIG_FILE%
python %MAIN_SCRIPT% --process %PROCESS% %TARGET% %WITH_RAINFALL% %CONFIG_FILE%

:end
echo ✅ Done.
