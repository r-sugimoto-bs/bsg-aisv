# lixil-aisv-backend

本リポジトリは、LIXIL AISV（AI Support Virtual）システムのバックエンド実装です。
FastAPI・LangChain・Google Cloud・Gemini などの最新技術を活用し、AIチャットやデータ連携を実現します。

## 主な構成

- **FastAPI**: APIサーバー
- **LangChain / LangGraph**: LLMワークフロー・チャットフロー
- **Google Cloud**: BigQuery, Storage, Vertex AI, Discovery Engine など
- **Gemini**: Googleの生成AI
- **cloud_run/app/**: アプリケーション本体
- **cloud_run/requirements.txt**: 依存パッケージ管理
- **setup.py**: 自作モジュールインストール用

## セットアップ手順

1. 仮想環境の作成・有効化（プロジェクトルートで）
    ```powershell
    python -m venv venv
    venv\Scripts\activate
    ```

2. 依存パッケージのインストール
    ```powershell
    pip install -r cloud_run/requirements.txt
    pip install -e .
    ```

3. 環境変数の設定  
   `.env` ファイルを作成し、必要なGoogle CloudやAPIキー等を記載してください
   設定する項目は下記の通りです
   ```
   BEARER_TOKEN
   GOOGLE_CLOUD_PROJECT
   DATASTORE
   ```

4. サーバー起動例
    ```powershell
    uvicorn app.main:app --reload
    ```

## ディレクトリ構成例

```
lixil-aisv-backend/
├─ cloud_run/
│  ├─ app/
│  │  ├─ services/
│  │  ├─ schemas/
│  │  └─ ...
│  ├─ requirements.txt
├─ setup.py
├─ README.md
└─ ...
```