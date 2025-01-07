from fastapi import FastAPI, WebSocket, HTTPException, Depends, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import logging
from typing import List, Optional, Dict, Set
from pathlib import Path
import json
import time
from datetime import datetime, timedelta
import requests
from weakref import WeakSet
from urllib.parse import urlparse

from ..config.settings import get_settings
from ..core.agent import create_agent, create_initial_state
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Netgrid Agent",
    description="your friendly local cyperpunk agent",
    version="1.0.0"
)

settings = get_settings()
logging.getLogger().setLevel(settings.log_level)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] # configure this in production
)

static_path = Path(__file__).parent.parent.parent / "frontend" / "static"
templates_path = Path(__file__).parent.parent.parent / "frontend" / "templates"

app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
templates = Jinja2Templates(directory=str(templates_path))

class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, List[datetime]] = {}
    
    def is_rate_limited(self, client_id: str) -> bool:
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        if client_id in self.requests:
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if req_time > minute_ago
            ]
        
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return True
        
        self.requests[client_id].append(now)
        return False

rate_limiter = RateLimiter(settings.max_requests_per_minute)
active_connections: Set[WebSocket] = WeakSet()

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/env")
async def get_env():
    settings = get_settings()
    return {
        "llm_base_url": settings.llm_base_url,
        "llm_api_key": settings.llm_api_key
    }

@app.get("/api/models")
async def get_models():
    settings = get_settings()
    try:
        response = requests.get(
            f"{settings.llm_base_url}/models",
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            verify=False
        )
        response.raise_for_status()
        models_response = response.json()
        
        if isinstance(models_response, dict) and models_response.get('object') == 'list':
            return [{"id": model["id"]} for model in models_response.get('data', [])]
            
        if isinstance(models_response, list):
            return [{"id": model.get("id")} for model in models_response if model.get("id")]
            
        return []
    except Exception as e:
        logger.error(f"Failed to fetch models: {e}")
        return []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    try:
        await websocket.accept()
        active_connections.add(websocket)
        
        client = websocket.client.host
        conversation_history: List[BaseMessage] = []
        params = dict(websocket.query_params)
        
        models = await get_models()
        if not models:
            await websocket.close(code=1008, reason="No models available")
            return
            
        model_name = params.get("model")
        if not model_name or not any(m["id"] == model_name for m in models):
            model_name = models[0]["id"]
        
        graph = create_agent(
            llm_base_url=settings.llm_base_url,
            llm_api_key=settings.llm_api_key,
            model_name=model_name
        )
        
        while websocket in active_connections:
            try:
                message = await websocket.receive_text()
                
                if not message or len(message.strip()) == 0:
                    continue
                    
                if rate_limiter.is_rate_limited(client):
                    if websocket in active_connections:
                        await websocket.send_json({
                            "type": "error",
                            "content": f"\n[{datetime.now().strftime('%H:%M:%S')}] Rate limit exceeded. Please wait a moment."
                        })
                    continue
                
                if len(message) > 1000:
                    if websocket in active_connections:
                        await websocket.send_json({
                            "type": "error",
                            "content": f"\n[{datetime.now().strftime('%H:%M:%S')}] Message too long. Keep under 1000 characters."
                        })
                    continue
                
                conversation_history.append(HumanMessage(content=message))
                if websocket in active_connections:
                    await websocket.send_json({"type": "start_response"})
                
                assistant_response = ""
                is_first_chunk = True
                inputs = create_initial_state(conversation_history, autonomous=True)
                
                try:
                    for chunk in graph.stream(inputs, stream_mode=["messages", "updates"]):
                        if websocket not in active_connections:
                            break
                            
                        stream_type, content = chunk
                        if stream_type == "messages":
                            message_chunk, metadata = content
                            if isinstance(message_chunk, ToolMessage):
                                tool_name = message_chunk.name
                                tool_args = json.loads(message_chunk.content)
                                desc = f"[NETGRID] >> Initializing {tool_name} protocol..."
                                
                                if tool_name == "web_search" and "query" in tool_args:
                                    query = tool_args["query"][:30] + "..." if len(tool_args["query"]) > 30 else tool_args["query"]
                                    desc = f"[NETGRID] >> Infiltrating global datastreams for: {query}"
                                elif tool_name == "parse_website" and "url" in tool_args:
                                    domain = urlparse(tool_args["url"]).netloc
                                    desc = f"[NETGRID] >> Establishing neural link with {domain}..."
                                
                                if websocket in active_connections:
                                    await websocket.send_json({
                                        "type": "tool_start",
                                        "tool_name": tool_name,
                                        "args": tool_args,
                                        "description": desc
                                    })
                                is_first_chunk = True
                            elif message_chunk.content:
                                chunk_content = message_chunk.content
                                if is_first_chunk and assistant_response:
                                    chunk_content = f"\n\n{chunk_content}"
                                is_first_chunk = False
                                
                                if websocket in active_connections:
                                    await websocket.send_json({
                                        "type": "stream",
                                        "content": chunk_content
                                    })
                                assistant_response += chunk_content
                        elif stream_type == "updates" and "agent" in content:
                            if websocket in active_connections:
                                await websocket.send_json({"type": "tool_end"})
                    
                    if websocket in active_connections:
                        await websocket.send_json({"type": "end_response"})
                        if assistant_response:
                            conversation_history.append(HumanMessage(content=assistant_response))
                except Exception as e:
                    logger.error(f"Stream error: {e}")
                    if websocket in active_connections:
                        await websocket.send_json({
                            "type": "error",
                            "content": f"\n[{datetime.now().strftime('%H:%M:%S')}] An error occurred during processing."
                        })
                        
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Message handling error: {e}")
                if websocket in active_connections:
                    await websocket.send_json({
                        "type": "error",
                        "content": f"\n[{datetime.now().strftime('%H:%M:%S')}] An error occurred. Please try again."
                    })
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        active_connections.discard(websocket)
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    import os
    import multiprocessing
    
    project_root = Path(__file__).parent.parent.parent.parent
    os.chdir(str(project_root))
    workers = min(multiprocessing.cpu_count(), 4)
    
    uvicorn.run(
        "src.backend.api.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        workers=workers,
        ws_ping_interval=20,
        ws_ping_timeout=30,
        timeout_keep_alive=30
    ) 