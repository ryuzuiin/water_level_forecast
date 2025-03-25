# 軽量な Python イメージを使用
FROM python:3.11-slim

# 作業ディレクトリを設定（コンテナ内）
WORKDIR /app

# プロジェクトファイルをすべてコピー（requirements.txt や src など含む）
COPY . /app

# パッケージをインストール（pip をアップグレードしてから requirements を読み込む）
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# デフォルトの実行コマンドを設定（main.py を使って処理を一括実行）
CMD ["python", "main.py", "--process", "all", "--target", "Headworks_A", "--with_rainfall"]
