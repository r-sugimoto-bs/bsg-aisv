from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.api import api_router
from app.core.middleware import AuthMiddleware
from app.core.load_env import load_environment
import os

# 環境変数の読み込み
load_environment()

app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)

# CORS ミドルウェアの設定
origins = os.getenv("CORS_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# その他のミドルウェア
app.add_middleware(AuthMiddleware)

app.include_router(api_router)