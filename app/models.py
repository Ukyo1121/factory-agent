# app/models.py
from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str          # 用户的问题
    thread_id: str      # 用于 LangGraph 记忆的会话 ID