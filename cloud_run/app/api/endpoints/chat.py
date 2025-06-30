from fastapi import APIRouter
from app.schemas.chat_schema import ChatRequest, ChatResponse, State
from app.services import chat_langgraph as lg

router = APIRouter()

@router.post("/api/v1/chat")
async def chatbot(req: ChatRequest):
    config = {"configurable": {"thread_id": "example-1"}}
    user_query = State(query=req.message, chat_id=req.session_id)
    try:

        first_response = lg.langgraph().invoke({"query": req.message, "chat_id": req.session_id}, config, debug=True)
        return ChatResponse(
        message=first_response.get("message")[-1]
        )
    except Exception as e:
        print("langgraph:", e)
        raise

