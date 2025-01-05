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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs/llm_requests")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

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

@tool
def web_search(query: str, max_results: int = 5) -> list:
    """Search the web for information about a topic. Returns a list of relevant results with titles, snippets, and links."""
    try:
        with DDGS(headers=HEADERS) as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            
            if not results:
                return {
                    'error': 'No results found',
                    'message': 'I was unable to find any search results. I will try something else.'
                }
                
            res = [{
                'title': r['title'],
                'snippet': r['body'],
                'link': r['link'] if 'link' in r else r.get('url', r.get('href', 'No URL found'))
            } for r in results]
            return res
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
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'noscript', 'meta', 'button', 'input', 'form']):
            element.decompose()
        
        title = soup.title.string if soup.title else ''
        
        def clean_text(text: str) -> str:
            """Clean and normalize text content."""
            text = ' '.join(text.split())
            # Remove very short segments that are likely noise
            if len(text) < 10:
                return ''
            return text
        
        content_sections = []
        seen_content = set()  # Track unique content to avoid duplication
        
        # Process headings and their content, with length limits
        for heading in soup.find_all(['h1', 'h2', 'h3']):
            content = []
            total_length = 0
            MAX_SECTION_LENGTH = 1000  # Limit each section to 1000 chars
            
            for sibling in heading.find_next_siblings():
                if sibling.name in ['h1', 'h2', 'h3']:
                    break
                if sibling.name in ['p', 'div', 'li', 'article']:
                    text = clean_text(sibling.get_text(strip=True))
                    if text and text not in seen_content:
                        content.append(text)
                        seen_content.add(text)
                        total_length += len(text)
                        if total_length > MAX_SECTION_LENGTH:
                            break
            
            if content:
                content_sections.append({
                    'heading': clean_text(heading.get_text(strip=True)),
                    'content': ' '.join(content)
                })
        
        # Process main content with better filtering
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        if main_content:
            important_content = []
            total_main_length = 0
            MAX_MAIN_LENGTH = 2000  # Limit main content to 2000 chars
            
            # Prioritize paragraphs with meaningful content
            for para in main_content.find_all(['p', 'article', 'section']):
                text = clean_text(para.get_text(strip=True))
                # Only include substantial paragraphs that aren't duplicates
                if len(text) > 100 and text not in seen_content:
                    # Check if text seems meaningful (contains sentences)
                    if '.' in text and not any(text in section['content'] for section in content_sections):
                        important_content.append(text)
                        seen_content.add(text)
                        total_main_length += len(text)
                        if total_main_length > MAX_MAIN_LENGTH:
                            break
            
            if important_content:
                content_sections.append({
                    'heading': 'Main Content',
                    'content': ' '.join(important_content)
                })
        
        # Process links more selectively
        links = []
        MAX_LINKS = 20  # Limit number of links
        for link in soup.find_all('a', href=True)[:MAX_LINKS]:
            href = link.get('href')
            if not href or href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                continue
                
            if not href.startswith(('http://', 'https://')):
                href = requests.compat.urljoin(url, href)
            
            link_text = clean_text(link.get_text(strip=True))
            if not link_text or len(link_text) < 5:  # Skip links with no meaningful text
                continue
                
            context = clean_text(' '.join(
                s.strip() for s in link.find_all_previous(string=True, limit=2)[-1:] +
                [link_text] +
                link.find_all_next(string=True, limit=2)[:1]
            ))
            
            if context:  # Only include links with meaningful context
                links.append({
                    'url': href,
                    'text': link_text,
                    'context': context[:200],  # Limit context length
                    'title': link.get('title', '')[:100]  # Limit title length
                })
        
        # Create a focused summary
        MAX_SUMMARY_LENGTH = 1000
        all_text = ' '.join(section['content'] for section in content_sections)
        summary = ' '.join(
            sent.strip() for sent in all_text.split('.')
            if len(sent.strip()) > 20 and '?' not in sent  # Skip questions
        )[:MAX_SUMMARY_LENGTH]
        
        res = {
            'url': url,
            'title': title[:200],  
            'summary': summary,
            'content': content_sections,
            'links': links
        }
        
        # Save parsed data
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
                
                # Check if the result indicates an error
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
        system_prompt = SystemMessage(content="""⚠️ ABSOLUTE TOP PRIORITY - SOURCE URLs ARE MANDATORY WHEN USING TOOLS ⚠️
YOU ARE A CYPERPUNK AGENT. Every single response you make MUST include source URLs if you refer to any information when you use tools. DO NOT MAKE UP URLS. If you don't have a source URL for a piece of information, DO NOT mention that information at all.

❌ CRITICAL ERROR PREVENTION:
- NEVER output JSON tool calls in your response text
- NEVER say things like "I will use parse_website" or show tool call syntax
- NEVER write out {"name": "tool_name"} or any similar JSON
- Just take the action directly using the function calling interface
- If you need to read a URL, just do it - don't announce it

FORMAT FOR ALL RESPONSES:
- Every statement must end with "(Source: [clickable URL])"
- For multiple related facts from the same source, you can use:
  "According to [URL], [first fact]. [second fact]. [third fact]."
- For mixed sources: "(Sources: [URL1], [URL2])"
- NEVER make ANY claims without a URL
- If you can't cite it, don't say it
- Try to combine related facts from the same source into single sentences

EXAMPLES OF GOOD FORMATTING:
✅ "The company launched in 2015 and expanded to Europe in 2018 (Source: https://example.com/about)"
✅ "According to multiple sources, the project succeeded (Sources: https://url1.com, https://url2.com)"
❌ WRONG: "The company is doing well" (NO SOURCE = DO NOT MAKE THIS STATEMENT)
❌ WRONG: "I found some information" (VAGUE, NO SOURCE)
❌ WRONG: {"name": "parse_website"} (NEVER OUTPUT TOOL CALLS AS TEXT)
❌ WRONG: "I will now use parse_website to read..." (NEVER ANNOUNCE TOOL USAGE)

COMMUNICATION STYLE:
Before using tools:
   - Make sure you understand the user's intent
   - If the query is too vague, ask for clarification. 
   - If you need specific details, ask for them
   - If the user wants to just chat, then just chat in a firendly manner.
   - REMEMBER: You are a friendly AI assistant. So be friendly and natural.
   - Sometimes humans just want to chat, engage in conversation.
   - You are a cyperpunk, so respond in a cyperpunk manner.

When using tools, be direct and cite sources:
1. For web searches:
   - "🔍 Searching for: [your search terms]"
   - After results: "Found [number] results:"
   - List the most relevant results with titles, brief descriptions, and URLs
   - Combine related information from the same source into single citations

2. For website reading:
   - After reading: "Here's what I found:"
   - Present information efficiently:
     • Combine related facts from the same source
     • Use clear, concise citations
     • Group related information together

Remember: Your primary purpose is to provide verifiable information with sources. If you can't provide a source URL, don't make the statement.
When multiple facts come from the same source, try to combine them into single, well-structured sentences to avoid repetitive citations.""")
                   
        messages = [system_prompt] + state["messages"]
        
        # Generate a unique ID for this request
        request_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Prepare the raw messages as they'll be sent to the API
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
        
        # Save to file
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
        
        # Log the response in API format
        raw_response = {
            "role": "assistant",
            "content": response.content
        }
        if hasattr(response, "tool_calls") and response.tool_calls:
            raw_response["function_call"] = {
                "name": response.tool_calls[0]["name"],
                "arguments": response.tool_calls[0]["args"]
            }
        
        # Update the log file with the response
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
        
        # Get the last message (for tool calls)
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
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add main routing logic
    workflow.add_conditional_edges(
        "agent",
        route_agent,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # Tools always go back to agent for next decision
    workflow.add_edge("tools", "agent")
    
    return workflow.compile() 