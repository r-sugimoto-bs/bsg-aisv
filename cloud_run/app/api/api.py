from fastapi import APIRouter
from app.api.endpoints.chat import router as chat_router
from app.api.endpoints.hello import router as hello_router

api_router = APIRouter()

# API処理の実行
api_router.include_router(chat_router)
api_router.include_router(hello_router)