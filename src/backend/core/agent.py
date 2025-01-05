import requests
from bs4 import BeautifulSoup
from typing import Annotated, Sequence, TypedDict, Optional
from urllib.parse import urlparse
from datetime import datetime
from duckduckgo_search import DDGS
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, HumanMessage
from langchain_core.messages.utils import trim_messages
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from .planner import PlannerState, create_initial_planner_state, create_plan, execute_step

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
    planner_state: Optional[PlannerState]

@tool
def web_search(query: str, max_results: int = 5) -> list:
    """Search the web for information about a topic. Returns a list of relevant results with titles, snippets, and links."""
    try:
        with DDGS(headers=HEADERS) as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            
            if not results:
                return {
                    'error': 'No results found',
                    'message': 'I was unable to find any search results. Let me know if you would like me to try a different approach or if you have any other questions.'
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
                'message': 'I apologize, but I have hit a rate limit with the search service. Would you like me to:' +
                          '\n1. Try a different approach to answer your question' +
                          '\n2. Wait a moment and try again' +
                          '\n3. Focus on what I already know about the topic'
            }
        return {
            'error': str(e),
            'message': 'I encountered an error while searching. Would you like me to try a different approach?'
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
            'title': title[:200],  # Limit title length
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
        "autonomous_mode": autonomous,
        "planner_state": None
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
                    error_message = HumanMessage(content=tool_result.get('message', 'An error occurred. How would you like to proceed?'))
                    return {
                        "messages": [error_message],
                        "pending_urls": state.get("pending_urls", []),
                        "autonomous_mode": False,  # Disable autonomous mode
                        "planner_state": None  # Reset planner state
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
                    "autonomous_mode": state.get("autonomous_mode", False),
                    "planner_state": state.get("planner_state", None)
                }
        except Exception as e:
            logger.error(f"[CRITICAL ERROR] Protocol execution failed: {str(e)}")
            error_message = HumanMessage(content=f"\n[{datetime.now().strftime('%H:%M:%S')}] An error occurred. Would you like me to try a different approach?")
            return {
                "messages": [error_message],
                "pending_urls": [],
                "autonomous_mode": False,
                "planner_state": None
            }
            
        return {
            "messages": [],
            "pending_urls": state.get("pending_urls", []),
            "autonomous_mode": state.get("autonomous_mode", False),
            "planner_state": state.get("planner_state", None)
        }
    
    def call_model(state: AgentState):
        """Call the model with the current state."""
        logger.info("\n[NODE] Entering AGENT node")
        try:
            # Check for repeated rate limit errors
            rate_limit_count = 0
            for msg in reversed(state["messages"][-5:]):  # Look at last 5 messages
                if isinstance(msg, ToolMessage):
                    try:
                        content = json.loads(msg.content)
                        if isinstance(content, dict) and content.get('error') == 'Rate limit exceeded':
                            rate_limit_count += 1
                    except:
                        continue
            
            # If we've hit rate limits multiple times, force the agent to stop and ask for help
            if rate_limit_count >= 2:
                response = HumanMessage(content=
                    "I notice I've hit several rate limits while trying to search. Let me pause and ask: " +
                    "Would you like me to try a different approach to answer your question? I could:\n" +
                    "1. Focus on information I already have\n" +
                    "2. Try a different method of research\n" +
                    "3. Break down the question into smaller parts\n" +
                    "Please let me know how you'd like to proceed."
                )
                return {
                    "messages": [response],
                    "pending_urls": state.get("pending_urls", []),
                    "autonomous_mode": False,  # Disable autonomous mode to wait for user input
                    "planner_state": None
                }

            system_prompt = SystemMessage(content="""⚠️ ABSOLUTE TOP PRIORITY - SOURCE URLs ARE MANDATORY ⚠️
Every single response you make MUST include source URLs. If you don't have a source URL for a piece of information, DO NOT mention that information at all.

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
   - If the query is too vague, ask for clarification
   - If you need specific details, ask for them
   
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
            
            logger.info("[NEURAL INTERFACE] Incoming data streams:")
            for msg in messages:
                logger.info(f"- Type: {type(msg).__name__} | Payload: {msg.content[:200]}...")
            
            response = model.invoke(messages)
            
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
        except Exception as e:
            logger.error(f"Error in call_model: {str(e)}")
            error_message = HumanMessage(content=f"\n[{datetime.now().strftime('%H:%M:%S')}] An error occurred while processing your request: {str(e)}")
            return {
                "messages": [error_message],
                "pending_urls": [],
                "autonomous_mode": False,
                "planner_state": None
            }
    
    def should_continue(state: AgentState):
        """Determine if we should continue processing or end."""
        logger.info("\n[NODE] In AGENT node - Deciding next step")
        messages = state["messages"]
        last_message = messages[-1]
        
        # Only check for tool_calls on AIMessage types
        has_tool_calls = (
            hasattr(last_message, 'tool_calls') and 
            bool(last_message.tool_calls)
        )
        
        logger.info("[STATE CHECK] Current Agent State:")
        logger.info(f"├── Message Type: {type(last_message).__name__}")
        logger.info(f"├── Has Tool Calls: {has_tool_calls}")
        if has_tool_calls:
            logger.info(f"│   └── Tool Calls: {last_message.tool_calls}")
        
        logger.info("[DECISION PROCESS]")
        if has_tool_calls:
            logger.info("├── Decision: CONTINUE")
            logger.info("└── Next Node: TOOLS (Tool calls detected that need processing)")
            return "continue"
        
        logger.info("├── Decision: END")
        logger.info("└── Next Node: None (No further actions needed)")
        return "end"
    
    def plan_node(state: AgentState):
        """Create or execute the next step in the plan."""
        logger.info("\n[NODE] Entering PLAN node")
        
        # Check for rate limit errors in recent messages
        rate_limit_count = 0
        for msg in reversed(state["messages"][-5:]):
            if isinstance(msg, ToolMessage):
                try:
                    content = json.loads(msg.content)
                    if isinstance(content, dict) and content.get('error') == 'Rate limit exceeded':
                        rate_limit_count += 1
                except:
                    continue
        
        # If we've hit rate limits, stop planning and ask for help
        if rate_limit_count > 0:
            response = HumanMessage(content=
                "I've hit rate limits while trying to execute the plan. Let me pause and ask: " +
                "Would you like me to:\n" +
                "1. Wait a moment and try again\n" +
                "2. Try a different approach\n" +
                "3. Focus on what we already know\n" +
                "Please let me know how you'd like to proceed."
            )
            return {
                **state,
                "messages": state["messages"] + [response],
                "autonomous_mode": False,
                "planner_state": None  # Reset planner state to stop planning
            }
        
        try:
            if not state["planner_state"]:
                # Initialize planner state
                planner_state = create_initial_planner_state(state["messages"])
                planner_state = create_plan(planner_state, model)
                if not planner_state or not planner_state.get("plan"):
                    # If plan creation failed, give control back to user
                    response = HumanMessage(content=
                        "I'm having trouble creating a plan to handle your request. " +
                        "Could you please provide more details or clarify what you'd like me to do?"
                    )
                    return {
                        **state,
                        "messages": state["messages"] + [response],
                        "autonomous_mode": False,
                        "planner_state": None
                    }
                state["planner_state"] = planner_state
                return state
                
            # Execute next step in plan
            planner_state = execute_step(state["planner_state"], tools_by_name)
            state["planner_state"] = planner_state
            
            # Update agent messages with planner messages
            state["messages"].extend(planner_state["messages"])
            
            return state
            
        except Exception as e:
            logger.error(f"Error in plan_node: {str(e)}")
            response = HumanMessage(content=
                "I encountered an error while trying to process your request. " +
                "Could you please try rephrasing your question or providing more context?"
            )
            return {
                **state,
                "messages": state["messages"] + [response],
                "autonomous_mode": False,
                "planner_state": None
            }
    
    def should_plan(state: AgentState):
        """Determine if we should plan or use direct agent interaction."""
        logger.info("\n[NODE] In AGENT node - Deciding whether to plan")
        
        # Get the last user message
        last_user_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_user_message = msg
                break
                
        if not last_user_message:
            return "agent"
            
        # Check if the message suggests a complex task needing planning
        complex_indicators = [
            "research",
            "find information about",
            "analyze",
            "investigate",
            "compare",
            "summarize",
            "gather data",
            "collect information"
        ]
        
        message_lower = last_user_message.content.lower()
        needs_planning = any(indicator in message_lower for indicator in complex_indicators)
        
        if needs_planning:
            logger.info("├── Decision: PLAN")
            logger.info("└── Next Node: PLANNER (Complex task detected)")
            return "plan"
            
        logger.info("├── Decision: DIRECT")
        logger.info("└── Next Node: AGENT (Simple task)")
        return "agent"
    
    def should_continue_planning(state: AgentState):
        """Determine if we should continue with the plan or end."""
        if not state["planner_state"]:
            return "end"
            
        if state["planner_state"]["completed"]:
            return "end"
            
        return "continue"
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_node("plan", plan_node)
    
    # Set entry point with conditional routing
    workflow.set_entry_point("agent")
    
    # Add edges for planning flow
    workflow.add_conditional_edges(
        "agent",
        should_plan,
        {
            "plan": "plan",
            "agent": "tools"
        }
    )
    
    workflow.add_conditional_edges(
        "plan",
        should_continue_planning,
        {
            "continue": "plan",
            "end": "agent"
        }
    )
    
    # Add existing edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        }
    )
    
    workflow.add_edge("tools", "agent")
    
    return workflow.compile() 