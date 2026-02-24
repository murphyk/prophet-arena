import json
import logging
import os
import re
import time
from typing import List, Optional

import anthropic
import httpx
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import HTMLResponse, PlainTextResponse
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


BETTERSTACK_API_TOKEN = os.environ.get("BETTERSTACK_API_TOKEN")
BETTERSTACK_SOURCE_ID = "1758509"  # from telemetry.betterstack.com URL (?s=1758509)


def load_requests_from_betterstack(last: int = 50) -> list:
    """Query Better Stack API for recent prediction request log entries."""
    try:
        resp = httpx.get(
            "https://telemetry.betterstack.com/api/v1/query",
            headers={"Authorization": f"Bearer {BETTERSTACK_API_TOKEN}"},
            params={
                "source_ids[]": BETTERSTACK_SOURCE_ID,
                "query": '"event":"request"',
                "batch_size": last,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        entries = []
        for item in data:
            msg = item.get("message", "")
            try:
                json_start = msg.index("{")
                entry = json.loads(msg[json_start:])
                if entry.get("event") == "request":
                    entry["timestamp"] = item.get("dt", "")
                    entries.append(entry)
            except (ValueError, json.JSONDecodeError):
                continue
        return list(reversed(entries))  # oldest first
    except Exception as e:
        logger.error("Better Stack query failed: %s", e)
        return []


def load_requests_from_file(last: int = 50) -> list:
    """Fallback: parse the local log file."""
    if not os.path.exists(LOG_FILE):
        return []
    entries = []
    with open(LOG_FILE) as f:
        for line in f:
            try:
                json_start = line.index("{")
                entry = json.loads(line[json_start:])
                if entry.get("event") == "request":
                    entry["timestamp"] = line[:json_start].strip()
                    entries.append(entry)
            except (ValueError, json.JSONDecodeError):
                continue
    return entries[-last:]


def load_requests(last: int = 50) -> list:
    if BETTERSTACK_API_TOKEN:
        return load_requests_from_betterstack(last)
    return load_requests_from_file(last)


@app.get("/logs", response_class=PlainTextResponse)
def get_logs(last: int = 100):
    """Return the last N lines of the request log."""
    if not os.path.exists(LOG_FILE):
        return "No logs yet."
    with open(LOG_FILE) as f:
        lines = f.readlines()
    return "".join(lines[-last:])


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(last: int = 50):
    """Human-readable dashboard of recent prediction requests."""
    entries = load_requests(last)

    def prob_bar(prob: float) -> str:
        pct = int(prob * 100)
        return (
            f'<div style="display:flex;align-items:center;gap:6px">'
            f'<div style="background:#4f8ef7;height:10px;width:{pct*2}px;border-radius:3px"></div>'
            f'<span>{pct}%</span></div>'
        )

    def render_answer(answer: str) -> str:
        """Extract and render probabilities from Claude's JSON answer."""
        try:
            # Strip markdown code fences if present
            clean = re.sub(r"```json|```", "", answer).strip()
            data = json.loads(clean)
            probs = data.get("probabilities", {})
            rationale = data.get("rationale", "")
            rows = "".join(
                f"<tr><td style='padding:2px 8px'><b>{k}</b></td><td>{prob_bar(v)}</td></tr>"
                for k, v in probs.items()
            )
            return (
                f'<p style="color:#ccc;font-size:0.85em">{rationale}</p>'
                f'<table>{rows}</table>'
            )
        except Exception:
            return f"<pre>{answer[:500]}</pre>"

    def render_sources(sources: list) -> str:
        if not sources:
            return "<em>none</em>"
        items = "".join(
            f'<li style="margin-bottom:4px"><b>#{s.get("ranking")}</b> {s.get("summary","")}</li>'
            for s in sources
        )
        return f'<ul style="margin:0;padding-left:16px">{items}</ul>'

    cards = ""
    for e in reversed(entries):
        # Extract the question from the system prompt
        sp = e.get("system_prompt", "")
        q_match = re.search(r'predicting the outcome of the event:\s*\\"(.+?)\\"', sp)
        question = q_match.group(1) if q_match else e.get("question") or None

        question_html = (
            f'<h3 style="margin:0 0 10px;color:#7dd3fc">{question}</h3>'
            if question else
            f'<details style="margin-bottom:10px"><summary style="color:#7dd3fc;cursor:pointer;font-size:1.1em;font-weight:bold">Show system prompt</summary>'
            f'<pre style="color:#ccc;font-size:0.75em;white-space:pre-wrap;margin-top:8px">{sp}</pre></details>'
        )

        cards += f"""
        <div style="background:#1e1e2e;border:1px solid #333;border-radius:8px;padding:16px;margin-bottom:16px">
          <div style="color:#888;font-size:0.8em;margin-bottom:6px">{e.get("timestamp","")} &nbsp;·&nbsp;
            {e.get("tokens_in",0)} in / {e.get("tokens_out",0)} out tokens</div>
          {question_html}
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
            <div>
              <div style="color:#aaa;font-size:0.8em;margin-bottom:4px">SOURCES</div>
              {render_sources(e.get("sources",[]))}
            </div>
            <div>
              <div style="color:#aaa;font-size:0.8em;margin-bottom:4px">PREDICTION</div>
              {render_answer(e.get("answer",""))}
            </div>
          </div>
        </div>"""

    if not cards:
        cards = "<p style='color:#888'>No requests logged yet. Trigger the onboarding test to see entries here.</p>"

    html = f"""<!DOCTYPE html>
<html>
<head>
  <title>ProphetArena Dashboard</title>
  <meta charset="utf-8">
  <style>
    body {{ font-family: system-ui, sans-serif; background:#13131f; color:#e2e8f0;
           max-width:960px; margin:0 auto; padding:24px; }}
    h1 {{ color:#7dd3fc; margin-bottom:4px; }}
    table {{ border-collapse:collapse; }}
    ul {{ color:#cbd5e1; font-size:0.85em; }}
  </style>
</head>
<body>
  <h1>ProphetArena Predictions</h1>
  <p style="color:#888;margin-top:0">Showing last {last} requests &nbsp;·&nbsp;
     <a href="/dashboard?last=200" style="color:#7dd3fc">show more</a></p>
  {cards}
</body>
</html>"""
    return html


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
