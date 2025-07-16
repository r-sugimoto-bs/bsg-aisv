from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.api import api_router
from app.core.middleware import AuthMiddleware
from app.core.load_env import load_environment
import os

# 環境変数読み込み
load_environment()

app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)

# CORS origins を取得・整形
raw = os.getenv("CORS_ORIGINS", "")
origins = [o.strip() for o in raw.split(",") if o.strip()]
print("CORS orig:", origins)

# ─────────────────────────────────────
# ① 認証ミドルウェアを「内側」に登録
app.add_middleware(AuthMiddleware)

# ② CORSミドルウェアを「外側」に登録 ←必ず後！
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ─────────────────────────────────────

# ルーター登録
app.include_router(api_router)
