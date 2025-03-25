#  水位予測システム

本プロジェクトは、機械学習を用いた水位予測システムです。データの前処理からモデルの学習・予測まで一連の処理を自動化できます。

---

## 📂 プロジェクト構成
<pre>
WATER_LEVEL_FORECAST/
│── data/                           # データフォルダ
│   ├── outputs/                     # 予測結果の出力フォルダ
│   ├── test/                        # テストデータセット
│   └── データファイル/               # 入力データファイル
│── models/                          # 学習済みモデルを保存するフォルダ
│── src/                             # メインソースコード
│   └── water_level_forecast/        # 水位予測モジュール
│       ├── anomaly_handler.py       # 異常値の検出と補正
│       ├── data_cleaning.py         # 生データのクリーニング（欠損値処理、ノイズ除去など）
│       ├── data_filter.py           # ルールに基づいたデータのフィルタリング
│       ├── feature_library.py       # 特徴量の作成
│       ├── japanese_data_processor.py # 水位データファイルの処理
│       ├── lightgbm_predictor.py    # 学習済みLightGBMモデルによる予測
│       ├── ml_data_preparer.py      # 機械学習用の入力データ準備
│       └── train_lightgbm.py        # LightGBMモデルの学習
│── demo.py                          # 動作確認用のデモスクリプト
│── run_demo.ipynb                   # Jupyter Notebookによる予測結果確認用デモ
│── requirements.txt                 # Pythonの依存パッケージリスト
│── README.md                        # プロジェクトの説明ドキュメント
└── run.bat                          # Windows環境でパイプラインを実行するバッチスクリプト

</pre>

---

## 🛠 環境構築とインストール手順

### **1️⃣ Python環境の準備**
本プロジェクトは **Python 3.11.9** で動作します。以下のコマンドでバージョンを確認してください：
```bash
python --version
```

### **2️⃣ 仮想環境 (venv) の作成**
プロジェクトフォルダのルートで以下を実行してください：
```bash
python -m venv venv
```

仮想環境を有効化：
```bash
venv\Scripts\activate
```

### **3️⃣ 依存パッケージ (requirements.txt) のインストール**
仮想環境を有効化した状態で、以下のコマンドを実行してください：
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 コードの実行方法

### **コマンドラインでの実行**
Windows環境では、`run.bat` を使用して簡単に処理を実行できます。

```bat
run.bat --process japanese	          # 水位データファイル処理
run.bat --process filter	          # データフィルタリング
run.bat --process clean	              # データクリーニング
run.bat --process anomaly	          # 異常値処理
run.bat --process ml	              # 機械学習データの準備
run.bat --process dynamic_features	  # 特徴量の計算
run.bat --process train --target "Headworks_A" --with_rainfall	   # モデル学習
run.bat --process predict --target "Headworks_A" --with_rainfall   # モデル予測
run.bat --process all --target "Headworks_A" --with_rainfall	   # すべてのプロセスを実行
```

### **Jupyter Notebookでの実行**
予測結果を対話的に確認したい場合は、Jupyter Notebookを使用できます：

```bash
# 仮想環境を有効化した状態で実行
jupyter notebook run_demo.ipynb
```

または、JupyterLabを使用する場合：

```bash
jupyter lab
```
そして、`run_demo.ipynb`ファイルを開いて実行してください。

### **💡 事前準備済みモデルを使って予測するだけの場合**

すでにモデルとテストデータが提供されている場合、以下のコマンドで予測のみ実行できます：

```bat
run.bat --process predict --target "Headworks_A" --with_rainfall
```


## 📊 出力ファイルについて

予測結果は以下の場所に保存されます：
- コマンドライン実行時: `data/outputs/{target}_predictions.csv`
- Jupyter Notebook実行時: ノートブック内で直接確認可能

---

## 🚀 その他
- データは `data/データファイル` フォルダに配置してください。
- 学習済みモデルは `models/` フォルダに保存されます。
- テストデータは `data/test` フォルダに保存されます。
- デモプログラム(demo.pyまたはrun_demo.ipynb)を実行する場合、予測結果は data/outputs フォルダに出力されます。

---

## 🔍 よくある質問

**Q1: 依存関係のインストールでエラーが発生する場合？**  
Pythonのバージョンを 3.11以上 にしてください。また、以下を試してください：

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Q2: 仮想環境 (venv) を有効化できない？**  
Windowsの場合、以下を試してください：

```powershell
Set-ExecutionPolicy Unrestricted -Scope Process
venv\Scripts\activate
```

**Q3: Jupyter Notebookが起動しない場合は？**  
以下のコマンドを実行してJupyterがインストールされていることを確認してください：

```bash
pip install notebook jupyterlab
jupyter --version
```
