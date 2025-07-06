from pydantic import BaseModel
from typing import List, Optional

class CreateKbRequest(BaseModel):
    name: Optional[str] = None
    embedding_model: Optional[str] = "openai"

class CreateKbResponse(BaseModel):
    id: str

class IngestRequest(BaseModel):
    texts: List[str]

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str 