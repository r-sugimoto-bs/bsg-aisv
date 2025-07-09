from fastapi import APIRouter
from app.schemas.chat_schema import ChatRequest, ChatResponse, State
from app.services.chat_langgraph import LangGraph
from app.services.chat_gemini import Geminijob

router = APIRouter()

@router.post("/api/v1/chat_lg")
async def chat_bot_lg(req: ChatRequest):
    config = {"configurable": {"thread_id": "example-1"}}
    user_query = State(query=req.message, chat_id=req.session_id)
    try:

        first_response = LangGraph().langgraph().invoke({"query": req.message, "chat_id": req.session_id}, config, debug=True)
        print()
        return ChatResponse(
        message=first_response.get("messages")[-1],
        grounding_data=first_response.get("grounding_data")
        )
    except Exception as e:
        print("langgraph:", e)
        raise


@router.post("/api/v1/chat_gm")
async def chat_bot_gm(req: ChatRequest):
    user_query = State(query=req.message, chat_id=req.session_id)
    try:
        response = Geminijob(user_query).flow()
        #print(response)
        return ChatResponse(
        message=response.messages[-1],
        grounding_data=response.grounding_data if response.grounding_data else []
        )
    except Exception as e:
        print("gemini:", e)
        raise