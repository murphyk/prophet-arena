import os
import time
from typing import List, Optional

import anthropic
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

class ChatMessage(BaseModel):
    model_config = {"extra": "ignore"}
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model_config = {"extra": "ignore"}
    model: str = "claude-sonnet-4-6"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False

app = FastAPI(title="Claude-powered OpenAI-compatible API for ProphetArena")

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials.credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return credentials.credentials


@app.post("/chat/completions")
def chat_completions(request: ChatCompletionRequest, token: str = Depends(verify_token)):
    """OpenAI-compatible chat completions endpoint backed by Claude."""
    client = anthropic.Anthropic(api_key=token)

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    try:
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=request.max_tokens or 1024,
            messages=messages,
        ) as stream:
            response = stream.get_final_message()
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid Anthropic API key")
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=str(e))

    content = response.content[0].text

    return {
        "id": f"chatcmpl-{response.id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": ChatMessage(role="assistant", content=content),
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
