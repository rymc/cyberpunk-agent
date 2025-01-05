from fastapi import FastAPI, WebSocket, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import logging
from typing import List, Optional, Dict
from pathlib import Path
import json
import time
from datetime import datetime, timedelta

from ..config.settings import get_settings
from ..core.agent import create_agent, create_initial_state
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Research Agent WebApp",
    description="An AI-powered research agent with web search capabilities",
    version="1.0.0"
)

# Get settings
settings = get_settings()

# Configure logging level from settings
logging.getLogger().setLevel(settings.log_level)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure this appropriately in production
)

# Setup static files and templates
static_path = Path(__file__).parent.parent.parent / "frontend" / "static"
templates_path = Path(__file__).parent.parent.parent / "frontend" / "templates"

app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
templates = Jinja2Templates(directory=str(templates_path))

# Rate limiting
class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, List[datetime]] = {}
    
    def is_rate_limited(self, client_id: str) -> bool:
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Clean up old requests
        if client_id in self.requests:
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if req_time > minute_ago
            ]
        
        # Check rate limit
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return True
        
        self.requests[client_id].append(now)
        return False

rate_limiter = RateLimiter(settings.max_requests_per_minute)

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request) -> HTMLResponse:
    """Serve the chat interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    
    try:
        client = websocket.client.host
        conversation_history: List[BaseMessage] = []
        
        graph = create_agent(
            llm_base_url=settings.llm_base_url,
            llm_api_key=settings.llm_api_key
        )
        
        while True:
            try:
                if rate_limiter.is_rate_limited(client):
                    await websocket.send_json({
                        "type": "error",
                        "content": f"\n[{datetime.now().strftime('%H:%M:%S')}] Rate limit exceeded. Please wait a moment before sending more messages."
                    })
                    continue
                
                message = await websocket.receive_text()
                
                if not message or len(message.strip()) == 0:
                    continue
                
                if len(message) > 1000:
                    await websocket.send_json({
                        "type": "error",
                        "content": f"\n[{datetime.now().strftime('%H:%M:%S')}] Message too long. Please keep messages under 1000 characters."
                    })
                    continue
                
                conversation_history.append(HumanMessage(content=message))
                await websocket.send_json({"type": "start_response"})
                
                try:
                    assistant_response = ""
                    is_first_chunk = True
                    inputs = create_initial_state(
                        conversation_history, 
                        autonomous=True
                    )
                    
                    for chunk in graph.stream(inputs, stream_mode=["messages", "updates"]):
                        stream_type, content = chunk
                        
                        if stream_type == "messages":
                            message_chunk, metadata = content
                            if isinstance(message_chunk, ToolMessage):
                                tool_name = message_chunk.name
                                try:
                                    tool_args = json.loads(message_chunk.content)
                                    if tool_name == "web_search":
                                        query = tool_args.get("query", "")
                                        if query:
                                            query = query[:30]
                                            if len(tool_args["query"]) > 30:
                                                query += "..."
                                            desc = f"[NETGRID] >> Infiltrating global datastreams for: {query}"
                                        else:
                                            desc = "[NETGRID] >> Breaching global information networks..."
                                    elif tool_name == "parse_website" and "url" in tool_args:
                                        from urllib.parse import urlparse
                                        domain = urlparse(tool_args["url"]).netloc
                                        desc = f"[NETGRID] >> Establishing neural link with {domain}..."
                                    else:
                                        desc = f"[NETGRID] >> Initializing {tool_name} protocol..."
                                except Exception as e:
                                    desc = f"[NETGRID] >> Initializing {tool_name} protocol..."
                                
                                await websocket.send_json({
                                    "type": "tool_start",
                                    "tool_name": tool_name,
                                    "args": tool_args,
                                    "description": desc
                                })
                                is_first_chunk = True
                            elif message_chunk.content:
                                chunk_content = message_chunk.content
                                
                                if is_first_chunk:
                                    if assistant_response:
                                        chunk_content = f"\n\n{chunk_content}"
                                    is_first_chunk = False
                                
                                await websocket.send_json({
                                    "type": "stream",
                                    "content": chunk_content
                                })
                                assistant_response += chunk_content
                        elif stream_type == "updates" and "agent" in content:
                            await websocket.send_json({
                                "type": "tool_end"
                            })
                    
                    await websocket.send_json({"type": "end_response"})
                    
                    if assistant_response:
                        conversation_history.append(HumanMessage(content=assistant_response))
                
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "content": f"\n[{datetime.now().strftime('%H:%M:%S')}] An error occurred while processing your message. Please try again."
                    })
            
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "content": f"\n[{datetime.now().strftime('%H:%M:%S')}] An error occurred. Please try again."
                })
    
    except Exception as e:
        try:
            await websocket.close()
        except:
            pass
    
    finally:
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Ensure we're in the project root directory
    project_root = Path(__file__).parent.parent.parent.parent
    os.chdir(str(project_root))
    
    # Run with a single worker for now due to Python 3.13 multiprocessing issues
    uvicorn.run(
        "src.backend.api.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        workers=1  # Force single worker mode
    ) 