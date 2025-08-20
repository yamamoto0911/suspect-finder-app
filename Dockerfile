FROM python:3.10-slim

# システムの依存関係をインストール
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /app

# Pythonの依存関係をコピーしてインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションのソースコードをコピー
COPY . .

# アップロード用ディレクトリを作成
RUN mkdir -p uploads results static

# ポート5000を公開
EXPOSE 5000

# Flaskアプリケーションを起動
CMD ["python", "web_app.py"]