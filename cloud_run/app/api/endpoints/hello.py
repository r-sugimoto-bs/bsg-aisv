from fastapi import APIRouter
from app.schemas.chat_schema import ChatRequest

router = APIRouter()

@router.post("/api/v1/hello")
async def hello_world(chat_request: ChatRequest):
    return {"message": "Hello, World!"}