import logging
import os
import time
from typing import List, Optional

import anthropic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
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


# ── Customise Claude's behaviour here ────────────────────────────────────────
# This is prepended to whatever system prompt ProphetArena sends, giving you
# control over how Claude answers questions.
MY_SYSTEM_PROMPT = """
You are a forecasting assistant. Answer concisely and return valid JSON.
""".strip()
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/chat/completions")
def chat_completions(request: ChatCompletionRequest, token: str = Depends(verify_token)):
    """OpenAI-compatible chat completions endpoint backed by Claude."""
    client = anthropic.Anthropic(api_key=token)

    incoming_system = " ".join(m.content for m in request.messages if m.role == "system")
    messages = [{"role": m.role, "content": m.content} for m in request.messages if m.role != "system"]

    # Combine our custom prompt with ProphetArena's system prompt
    combined_system = MY_SYSTEM_PROMPT + "\n\n" + incoming_system if incoming_system else MY_SYSTEM_PROMPT

    # ── Log incoming request ──────────────────────────────────────────────────
    logger.info("=== INCOMING REQUEST ===")
    logger.info(f"System prompt from ProphetArena:\n{incoming_system}")
    logger.info(f"Messages:")
    for m in messages:
        logger.info(f"  [{m['role']}]:\n{m['content']}")
    # ─────────────────────────────────────────────────────────────────────────

    try:
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=request.max_tokens or 1024,
            system=combined_system,
            messages=messages,
        ) as stream:
            response = stream.get_final_message()
    except anthropic.AuthenticationError as e:
        logger.error(f"Auth error: {e}")
        raise HTTPException(status_code=401, detail="Invalid Anthropic API key")
    except anthropic.APIError as e:
        logger.error(f"Anthropic API error {type(e).__name__}: {e}")
        raise HTTPException(status_code=502, detail=str(e))

    content = response.content[0].text

    # ── Log outgoing response ─────────────────────────────────────────────────
    logger.info("=== OUTGOING RESPONSE ===")
    logger.info(f"Claude's answer:\n{content}")
    logger.info(f"Tokens used: {response.usage.input_tokens} in / {response.usage.output_tokens} out")
    # ─────────────────────────────────────────────────────────────────────────

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
