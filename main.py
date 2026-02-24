import json
import logging
import os
import re
import time
from typing import List, Optional

import anthropic
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import PlainTextResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# ── Logging: stdout + local file + Better Stack (if token set) ───────────────
LOG_FILE = "requests.log"
handlers = [
    logging.StreamHandler(),        # Render dashboard
    logging.FileHandler(LOG_FILE),  # local file (see GET /logs)
]

LOGTAIL_TOKEN = os.environ.get("LOGTAIL_TOKEN")
if LOGTAIL_TOKEN:
    from logtail import LogtailHandler
    handlers.append(LogtailHandler(source_token=LOGTAIL_TOKEN))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=handlers,
)
logger = logging.getLogger(__name__)
# ─────────────────────────────────────────────────────────────────────────────


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


def parse_user_message(content: str) -> dict:
    """Split ProphetArena's user message into preamble, sources, and question."""
    result = {"preamble": "", "sources": [], "question": "", "raw": content}

    bracket_start = content.find("[SourceItem")
    bracket_end = content.rfind("]")

    if bracket_start == -1:
        result["question"] = content.strip()
        return result

    result["preamble"] = content[:bracket_start].strip()
    result["question"] = content[bracket_end + 1:].strip()
    sources_block = content[bracket_start:bracket_end + 1]

    for item_str in re.findall(r'SourceItem\((.+?)\)(?=,\s*SourceItem|\s*\])', sources_block, re.DOTALL):
        source = {}
        m = re.search(r'summary=(["\'])(.*?)\1', item_str, re.DOTALL)
        if m:
            source["summary"] = m.group(2)
        m = re.search(r'ranking=(\d+)', item_str)
        if m:
            source["ranking"] = int(m.group(1))
        m = re.search(r'user_comments=(["\'])(.*?)\1', item_str, re.DOTALL)
        source["user_comments"] = m.group(2) if m else None
        result["sources"].append(source)

    return result


# ── Customise Claude's behaviour here ────────────────────────────────────────
MY_SYSTEM_PROMPT = """
You are a forecasting assistant. Answer concisely and return valid JSON.
""".strip()
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/logs", response_class=PlainTextResponse)
def get_logs(last: int = 100):
    """Return the last N lines of the request log."""
    if not os.path.exists(LOG_FILE):
        return "No logs yet."
    with open(LOG_FILE) as f:
        lines = f.readlines()
    return "".join(lines[-last:])


@app.post("/chat/completions")
def chat_completions(request: ChatCompletionRequest, token: str = Depends(verify_token)):
    """OpenAI-compatible chat completions endpoint backed by Claude."""
    client = anthropic.Anthropic(api_key=token)

    incoming_system = " ".join(m.content for m in request.messages if m.role == "system")
    messages = [{"role": m.role, "content": m.content} for m in request.messages if m.role != "system"]
    combined_system = MY_SYSTEM_PROMPT + "\n\n" + incoming_system if incoming_system else MY_SYSTEM_PROMPT

    # Parse user message into structured parts
    user_msg = next((m for m in messages if m["role"] == "user"), None)
    parsed = parse_user_message(user_msg["content"]) if user_msg else {}

    try:
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=request.max_tokens or 1024,
            system=combined_system,
            messages=messages,
        ) as stream:
            response = stream.get_final_message()
    except anthropic.AuthenticationError as e:
        logger.error(json.dumps({"event": "auth_error", "error": str(e)}))
        raise HTTPException(status_code=401, detail="Invalid Anthropic API key")
    except anthropic.APIError as e:
        logger.error(json.dumps({"event": "api_error", "type": type(e).__name__, "error": str(e)}))
        raise HTTPException(status_code=502, detail=str(e))

    content = response.content[0].text

    # ── Single structured log entry per request ───────────────────────────────
    logger.info(json.dumps({
        "event": "request",
        "system_prompt": incoming_system,
        "question": parsed.get("question", ""),
        "sources": parsed.get("sources", []),
        "answer": content,
        "tokens_in": response.usage.input_tokens,
        "tokens_out": response.usage.output_tokens,
    }))
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
