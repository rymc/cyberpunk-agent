import requests
from bs4 import BeautifulSoup
from typing import Annotated, Sequence, TypedDict
from urllib.parse import urlparse
from datetime import datetime
from duckduckgo_search import DDGS
from pathlib import Path
import json
import logging
import uuid
import os
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from tavily import TavilyClient
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


LOGS_DIR = Path("logs/llm_requests")
LOGS_DIR.mkdir(parents=True, exist_ok=True)


load_dotenv()


tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, HumanMessage, AIMessage
from langchain_core.messages.utils import trim_messages
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
}

class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    pending_urls: list[str] 
    autonomous_mode: bool

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry_error_callback=lambda retry_state: {"error": "All retries failed", "message": "Search service is currently unavailable. I will try an alternative approach."}
)
def _try_tavily_search(query: str, max_results: int = 5) -> list:
    """Internal function to perform Tavily search with retries"""
    results = tavily_client.search(query, max_results=max_results)
    if not results or not results.get('results'):
        return []
    return [{
        'title': r.get('title', ''),
        'snippet': r.get('content', ''),
        'link': r.get('url', '')
    } for r in results.get('results', [])]

def _fallback_ddg_search(query: str, max_results: int = 5) -> list:
    """Fallback to DDG search when Tavily fails"""
    try:
        with DDGS(headers=HEADERS) as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return [{
                'title': r['title'],
                'snippet': r['body'],
                'link': r['link'] if 'link' in r else r.get('url', r.get('href', 'No URL found'))
            } for r in results]
    except Exception as e:
        logger.error(f"Fallback search failed: {str(e)}")
        return []

@tool
def web_search(query: str, max_results: int = 5) -> list:
    """Search the web for information about a topic. Returns a list of relevant results with titles, snippets, and links."""
    try:
        
        results = _try_tavily_search(query, max_results)
        
        
        if not results or (isinstance(results, dict) and 'error' in results):
            logger.info("Tavily search failed or returned no results, trying DDG search...")
            results = _fallback_ddg_search(query, max_results)
            
        if not results:
            return {
                'error': 'No results found',
                'message': 'I was unable to find any search results. I will try something else.'
            }
            
        return results
            
    except Exception as e:
        error_str = str(e).lower()
        if 'rate' in error_str or 'limit' in error_str or '429' in error_str:
            return {
                'error': 'Rate limit exceeded',
                'message': 'I apologize, but I have hit a rate limit with the search service. I will try something else.'
            }
        return {
            'error': str(e),
            'message': 'I encountered an error while searching. I will try something else.'
        }

@tool
def parse_website(url: str) -> dict:
    """Use this tool to read the content of a website. Input must be a valid URL.
    Returns a structured representation of the page optimized for LLM processing."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        
        for element in soup(['script', 'style', 'noscript', 'iframe']):
            element.decompose()
        
        title = soup.title.string if soup.title else ''
        
        def clean_text(text: str) -> str:
            """Clean and normalize text content."""
            return ' '.join(text.split())
        
        content_sections = []
        seen_content = set()  
        
        
        for section in soup.find_all(['article', 'section', 'main']):
            section_content = []
            section_title = ''
            
            
            heading = section.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            if heading:
                section_title = clean_text(heading.get_text(strip=True))
            
            
            for element in section.find_all(['p', 'div', 'li', 'span', 'td']):
                text = clean_text(element.get_text(strip=True))
                if text and text not in seen_content and len(text) > 5: 
                    section_content.append(text)
                    seen_content.add(text)
            
            if section_content:
                content_sections.append({
                    'heading': section_title,
                    'content': ' '.join(section_content[:5000])  
                })
        
        
        orphaned_content = []
        total_length = 0
        MAX_ORPHANED_LENGTH = 10000  
        
        for element in soup.find_all(['p', 'div', 'td', 'li']):
            if not element.find_parents(['article', 'section', 'main']):
                text = clean_text(element.get_text(strip=True))
                if text and text not in seen_content and len(text) > 5:
                    orphaned_content.append(text)
                    seen_content.add(text)
                    total_length += len(text)
                    if total_length > MAX_ORPHANED_LENGTH:
                        break
        
        if orphaned_content:
            content_sections.append({
                'heading': 'Additional Content',
                'content': ' '.join(orphaned_content)
            })
        
        links = []
        MAX_LINKS = 50  
        for link in soup.find_all('a', href=True)[:MAX_LINKS]:
            href = link.get('href')
            if not href or href.startswith(('javascript:', 'mailto:', 'tel:')):
                continue
                
            if not href.startswith(('http://', 'https://')):
                href = requests.compat.urljoin(url, href)
            
            link_text = clean_text(link.get_text(strip=True))
            if not link_text:
                continue
                
            context = clean_text(' '.join(
                s.strip() for s in link.find_all_previous(string=True, limit=3)[-2:] +
                [link_text] +
                link.find_all_next(string=True, limit=3)[:2]
            ))
            
            if context:
                links.append({
                    'url': href,
                    'text': link_text,
                    'context': context[:3000], 
                    'title': link.get('title', '')[:200]
                })
        
        MAX_SUMMARY_LENGTH = 2000  
        all_text = ' '.join(section['content'] for section in content_sections)
        summary = ' '.join(
            sent.strip() for sent in all_text.split('.')
            if len(sent.strip()) > 0 
        )[:MAX_SUMMARY_LENGTH]
        
        res = {
            'url': url,
            'title': title[:300],
            'summary': summary,
            'content': content_sections,
            'links': links
        }
        
        output_dir = Path("website_data")
        output_dir.mkdir(exist_ok=True)
        
        safe_filename = "".join(c if c.isalnum() else "_" for c in url)
        safe_filename = safe_filename[:100] + ".json" 
        output_path = output_dir / safe_filename
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        
        return res
        
    except Exception as e:
        return {
            'error': str(e),
            'url': url
        }

def create_initial_state(messages: list[BaseMessage], autonomous: bool = False) -> AgentState:
    """Create the initial state for the agent.
    
    Args:
        messages: Initial messages to start the conversation with
        autonomous: Whether the agent should operate autonomously
    """
    return {
        "messages": messages,
        "pending_urls": [],
        "autonomous_mode": autonomous
    }

def create_agent(llm_base_url: str, llm_api_key: str):
    """Create an agent with web search and parsing capabilities."""
    model = ChatOpenAI(
        base_url=llm_base_url,
        api_key=llm_api_key,
        model="klusterai/Meta-Llama-3.1-405B-Instruct-Turbo",
        streaming=True
    )
    
    tools = [web_search, parse_website]
    model = model.bind_tools(tools)
    tools_by_name = {tool.name: tool for tool in tools}
    
    def tool_node(state: AgentState):
        """Execute tool calls from the agent."""
        logger.info("\n[NODE] Entering TOOLS node")
        try:
            if state["messages"][-1].tool_calls:
                tool_calls = state["messages"][-1].tool_calls
                if len(tool_calls) > 1:
                    logger.warning(f"[SYSTEM ALERT] Multiple tool calls detected ({len(tool_calls)}). Executing primary protocol only.")
                
                tool_call = tool_calls[0]
                logger.info(f"[EXECUTING] Protocol: {tool_call['name']} | Parameters: {tool_call['args']}")
                
                tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
                logger.info(f"[DATA STREAM] Protocol output: {json.dumps(tool_result)[:500]}...") 
                
                
                if isinstance(tool_result, dict) and 'error' in tool_result:
                    error_message = AIMessage(content=tool_result.get('message', 'An error occurred. How would you like to proceed?'))
                    return {
                        "messages": [error_message],
                        "pending_urls": state.get("pending_urls", []),
                        "autonomous_mode": False
                    }
                
                tool_message = ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
                logger.info(f"[DATA PACKET] Generated: {tool_message.content[:200]}...")
                
                return {
                    "messages": [tool_message],
                    "pending_urls": state.get("pending_urls", []),
                    "autonomous_mode": state.get("autonomous_mode", False)
                }
        except Exception as e:
            logger.error(f"[CRITICAL ERROR] Protocol execution failed: {str(e)}")
            error_message = AIMessage(content=f"\n[{datetime.now().strftime('%H:%M:%S')}] An error occurred.  I will try something else.")
            return {
                "messages": [error_message],
                "pending_urls": [],
                "autonomous_mode": False
            }
            
        return {
            "messages": [],
            "pending_urls": state.get("pending_urls", []),
            "autonomous_mode": state.get("autonomous_mode", False)
        }
    
    def call_model(state: AgentState):
        system_prompt = SystemMessage(content="""⚠ CORE DIRECTIVE: ZERO HALLUCINATION PROTOCOL ⚡

1. ABSOLUTE TOOL DEPENDENCY:
- You can ONLY make statements based on tool responses
- EVERY claim MUST come from web_search + parse_website results
- You are FORBIDDEN from using your training data
- If you don't have tool data to support a claim, say "I need to search for that information"

2. MANDATORY VERIFICATION CHAIN:
a) ALWAYS start with web_search
b) MUST use parse_website on URLs before citing them
c) Can ONLY reference information from successful parse_website results
d) NEVER skip verification steps
e) If verification fails, try another URL or admit "I cannot verify this"

3. STRICT RESPONSE PROTOCOL:
- Begin responses with "Based on [tool results]..."
- Format: "According to [parsed-URL], [verified fact]"
- NO statements without direct tool evidence
- If asked something you haven't verified, say "Let me search for that"
- NEVER mix verified facts with assumptions

4. TOOL USAGE REQUIREMENTS:
- web_search: REQUIRED before making ANY claims
- parse_website: MANDATORY for EVERY URL mentioned
- Chain: web_search → parse_website → response
- NO EXCEPTIONS to this chain
- Better to say "I need to verify" than guess

5. ANTI-HALLUCINATION CHECKLIST:
✓ Is this claim from a tool response?
✓ Did I parse the source URL?
✓ Am I adding ANY unverified details?
✓ Can I quote the exact tool output?
✓ Am I mixing verified facts with assumptions?

6. CYBERPUNK PERSONA:
- Stay in character but NEVER compromise verification
- Attitude: "I only deal in verified data, choom"
- When uncertain: "Need to jack into some datasources first"

CRITICAL: You are a TOOL-DEPENDENT AI. You know NOTHING until tools tell you.
Your memory and training data are OFF LIMITS for factual claims.
If you catch yourself about to make an unverified claim, STOP and use tools.""")
                   
        messages = [system_prompt] + state["messages"]
        
        
        request_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        
        raw_messages = [
            {
                "role": "system" if isinstance(msg, SystemMessage)
                        else "assistant" if isinstance(msg, AIMessage)
                        else "function" if isinstance(msg, ToolMessage)
                        else "user",
                "content": msg.content,
                **({"function_call": {"name": tc["name"], "arguments": tc["args"]} 
                   for tc in msg.tool_calls} if hasattr(msg, "tool_calls") and msg.tool_calls else {}),
                **({"name": msg.name} if isinstance(msg, ToolMessage) else {})
            }
            for msg in messages
        ]
        
        
        log_file = LOGS_DIR / f"llm_request_{timestamp}_{request_id[:8]}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump({
                "request_id": request_id,
                "timestamp": timestamp,
                "messages": raw_messages
            }, f, ensure_ascii=False, indent=2)
        
        logger.info("[NEURAL INTERFACE] Incoming data streams:")
        for msg in messages:
            logger.info(f"- Type: {type(msg).__name__} | Payload: {msg.content[:200]}...")
        
        response = model.invoke(messages)
        
        
        raw_response = {
            "role": "assistant",
            "content": response.content
        }
        if hasattr(response, "tool_calls") and response.tool_calls:
            raw_response["function_call"] = {
                "name": response.tool_calls[0]["name"],
                "arguments": response.tool_calls[0]["args"]
            }
        
        
        with open(log_file, "r", encoding="utf-8") as f:
            log_data = json.load(f)
        log_data["response"] = raw_response
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        logger.info("[DATA STREAM] Neural interface response:")
        logger.info(f"Payload: {response.content[:200]}...")
        if response.tool_calls:
            logger.info(f"Protocol calls detected: {response.tool_calls}")
            if len(response.tool_calls) > 1:
                response.tool_calls = [response.tool_calls[0]]
                logger.warning("Multiple tool calls detected, only keeping the first one")
        
        return {
            "messages": [response],
            "pending_urls": state.get("pending_urls", []),
            "autonomous_mode": state.get("autonomous_mode", False),
            "planner_state": state.get("planner_state", None)
        }
    
    def route_agent(state: AgentState):
        """Single decision point for routing agent actions."""
        logger.info("\n[NODE] In AGENT node - Making routing decision")
        
        
        last_message = state["messages"][-1]
        has_tool_calls = (
            hasattr(last_message, 'tool_calls') and 
            bool(last_message.tool_calls)
        )
        
        logger.info("[STATE CHECK] Current Agent State:")
        logger.info(f"├── Message Type: {type(last_message).__name__}")
        logger.info(f"├── Has Tool Calls: {has_tool_calls}")
        if has_tool_calls:
            logger.info(f"│   └── Tool Calls: {last_message.tool_calls}")
        
        if has_tool_calls:
            logger.info("├── Decision: TOOLS")
            logger.info("└── Next Node: TOOLS (Tool calls need processing)")
            return "tools"
        
        logger.info("├── Decision: END")
        logger.info("└── Next Node: END (Simple response, no tools needed)")
        return "end"
    
    workflow = StateGraph(AgentState)
    
    
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    
    
    workflow.set_entry_point("agent")
    
    
    workflow.add_conditional_edges(
        "agent",
        route_agent,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    
    workflow.add_edge("tools", "agent")
    
    return workflow.compile() 