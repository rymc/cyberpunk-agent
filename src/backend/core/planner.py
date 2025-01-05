from typing import TypedDict, Annotated, Sequence, List, Dict, Any
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class PlannerState(TypedDict):
    """The state of the planner."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    plan: List[Dict[str, Any]]
    current_step: int
    completed: bool

def create_planner_prompt() -> str:
    return """You are a strategic planner for an autonomous research agent. Your role is to:

1. Break down complex research tasks into clear, actionable steps
2. Create a structured plan that can be executed sequentially
3. Each step should be concrete and achievable using the available tools:
   - web_search: Search the web for information
   - parse_website: Read and extract content from a webpage

FORMAT YOUR RESPONSE AS A JSON LIST OF STEPS:
[
    {
        "step": 1,
        "action": "web_search",
        "description": "Search for X to find Y",
        "parameters": {"query": "specific search query", "max_results": 5}
    },
    {
        "step": 2,
        "action": "parse_website",
        "description": "Extract information about Z from the top result",
        "parameters": {"url": "{{previous_step_result[0].link}}"}
    }
]

RULES:
1. Each step must use one of the available tools
2. Parameters must match the tool's requirements
3. Use placeholders like {{previous_step_result}} to reference previous results
4. Keep plans focused and efficient - typically 3-5 steps
5. Include clear success criteria for each step

Begin by analyzing the user's request and creating a structured plan."""

def create_initial_planner_state(messages: list[BaseMessage]) -> PlannerState:
    """Create the initial state for the planner.
    
    Args:
        messages: Initial messages to start the planning with
    """
    return {
        "messages": messages,
        "plan": [],
        "current_step": 0,
        "completed": False
    }

def create_plan(state: PlannerState, model) -> PlannerState:
    """Create a plan based on the user's request."""
    messages = state["messages"]
    system_message = SystemMessage(content=create_planner_prompt())
    
    # Get the plan from the model
    response = model.invoke([system_message] + messages)
    
    try:
        # Extract the plan from the response
        content = response.content
        # Find the JSON list in the content
        start = content.find("[")
        end = content.rfind("]") + 1
        if start == -1 or end == 0:
            raise ValueError("No valid plan found in response")
            
        plan_json = content[start:end]
        plan = json.loads(plan_json)
        
        # Validate the plan structure
        for step in plan:
            required_keys = ["step", "action", "description", "parameters"]
            if not all(key in step for key in required_keys):
                raise ValueError(f"Invalid step format: {step}")
            
            if step["action"] not in ["web_search", "parse_website"]:
                raise ValueError(f"Invalid action: {step['action']}")
        
        logger.info(f"Created plan with {len(plan)} steps")
        return {
            "messages": state["messages"],
            "plan": plan,
            "current_step": 0,
            "completed": False
        }
        
    except Exception as e:
        logger.error(f"Error creating plan: {str(e)}")
        return {
            "messages": state["messages"] + [HumanMessage(content=f"Error creating plan: {str(e)}")],
            "plan": [],
            "current_step": 0,
            "completed": True
        }

def execute_step(state: PlannerState, tools_by_name: dict) -> PlannerState:
    """Execute the current step in the plan."""
    if not state["plan"] or state["current_step"] >= len(state["plan"]):
        state["completed"] = True
        return state
        
    current_step = state["plan"][state["current_step"]]
    logger.info(f"Executing step {current_step['step']}: {current_step['description']}")
    
    try:
        # Get the tool for this step
        tool = tools_by_name[current_step["action"]]
        
        # Process parameters - replace any placeholders with actual values
        parameters = current_step["parameters"]
        # TODO: Add parameter substitution logic here
        
        # Execute the tool
        result = tool.invoke(parameters)
        
        # Add the result to the messages with newline before timestamp
        state["messages"].append(
            HumanMessage(content=f"\n[{datetime.now().strftime('%H:%M:%S')}] Step {current_step['step']} completed: {current_step['description']}\nResult: {json.dumps(result)}")
        )
        
        # Move to next step
        state["current_step"] += 1
        if state["current_step"] >= len(state["plan"]):
            state["completed"] = True
            
        return state
        
    except Exception as e:
        logger.error(f"Error executing step: {str(e)}")
        state["messages"].append(
            HumanMessage(content=f"\n[{datetime.now().strftime('%H:%M:%S')}] Error executing step {current_step['step']}: {str(e)}")
        )
        state["completed"] = True
        return state 