import os
import json
from dotenv import load_dotenv
load_dotenv()
import asyncio
import operator
import time
from typing import TypedDict, Annotated, List, Any, Optional, Dict, Literal
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from registry import ToolRegistry
from router import DeterministicRouter
from security import SafeExecutor

# ============================================================================
# State Definition
# ============================================================================

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_query: str
    session_id: str
    route: Optional[str]  # 'cache', 'execute_existing', 'generate_new'
    selected_tool: Optional[Dict[str, Any]]
    tool_params: Optional[Dict[str, Any]]
    execution_result: Optional[str]
    next_step: Optional[str]
    is_cached: bool

# ============================================================================
# Skill Graph Components
# ============================================================================

class SkillGraphManager:
    def __init__(self, db_path: str, api_key: str):
        self.registry = ToolRegistry(db_path)
        safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("MODEL_NAME", "gemini-1.5-flash"),
            google_api_key=api_key,
            temperature=0,
            safety_settings=safety_settings
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key
        )
        self.router = DeterministicRouter(self.registry, self.embeddings)
        self.executor = SafeExecutor()

    def _extract_json(self, content: str) -> Optional[Dict]:
        """Extracts JSON from LLM response, handling markdown blocks."""
        import re
        # Try finding json code block first
        json_block = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_block:
            try:
                return json.loads(json_block.group(1).strip())
            except:
                pass
        
        # Fallback to finding outermost braces
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                # Try non-greedy if greedy fails? No, json.loads handles trailing.
                # But json.loads(s) fails if there is "Extra data"
                # We need to find the balance point.
                # A simple trick: iterate backwards from the end of the match
                s = json_match.group()
                for i in range(len(s), 0, -1):
                    try:
                        return json.loads(s[:i])
                    except:
                        continue
            except:
                pass
        return None

    async def router_node(self, state: AgentState) -> Dict:
        """Determines routing and checks cache."""
        print(f"DEBUG: state type: {type(state)}")
        print(f"DEBUG: state keys: {state.keys() if hasattr(state, 'keys') else 'no keys'}")
        query = state.get("user_query", "")
        route, tool_data, _ = await self.router.route_query(query)
        
        updates = {
            "route": route,
            "selected_tool": tool_data,
            "next_step": "generate" if route == "generate_new" else "extract_params"
        }

        return updates

    async def param_extraction_node(self, state: AgentState) -> Dict:
        """Stuctured parameter extraction using LLM."""
        tool = state.get("selected_tool")
        query = state.get("user_query")
        
        if not tool or not query:
            return {"next_step": "error", "execution_result": "Missing tool or query context."}
            
        prompt = f"Extract parameters for {tool.get('name')} from query: '{query}'. Schema: {tool.get('parameters')}. Return ONLY JSON."
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        
        content = response.content
        if isinstance(content, list):
            content = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in content])
        
        updates = {}
        
        params = self._extract_json(content)
        if params:
            updates["tool_params"] = params
            
            cached = self.router.check_cache(tool["name"], params)
            if cached:
                updates.update({
                    "execution_result": cached,
                    "is_cached": True,
                    "next_step": "summarize"
                })
            else:
                updates.update({"is_cached": False, "next_step": "execute"})
        else:
            updates.update({"next_step": "error", "execution_result": "Failed to extract parameters."})
            
        return updates

    async def generate_tool_node(self, state: AgentState) -> Dict:
        """Generates a new reusable tool via LLM."""
        query = state.get("user_query", "")
        prompt = f"Create a reusable Python tool for: '{query}'. Return JSON with name, description, code, parameters."
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        
        content = response.content
        if isinstance(content, list):
            content = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in content])
        
        updates = {}
        
        tool_spec = self._extract_json(content)
        if tool_spec:
            # Handle key variations from different LLM models
            name = tool_spec.get('name') or tool_spec.get('tool_name')
            desc = tool_spec.get('description') or tool_spec.get('desc') or tool_spec.get('tool_description')
            code = tool_spec.get('code') or tool_spec.get('source_code')
            params = tool_spec.get('parameters') or tool_spec.get('params', {})
            
            if name and desc and code:
                tool_emb = await self.embeddings.aembed_query(desc)
                self.registry.save_tool(name, desc, code, params, tool_emb, 'system', 'pending')
                self.registry.add_pending_approval(name, state.get("session_id", "default"))
                
                updates.update({"selected_tool": {
                    "name": name,
                    "description": desc,
                    "code": code,
                    "parameters": params
                }, "next_step": "await_approval"})
            else:
                updates.update({"next_step": "error", "execution_result": f"Missing required fields (name, desc, code) in tool spec. Keys found: {list(tool_spec.keys())}"})
        else:
            updates.update({"next_step": "error", "execution_result": "Failed to extract valid JSON from response: " + content[:200]})
            
        return updates

    async def execute_tool_node(self, state: AgentState) -> Dict:
        """Safely executes the tool."""
        tool = state.get("selected_tool")
        params = state.get("tool_params")
        
        if not tool or not params:
            return {"next_step": "error", "execution_result": "Missing tool or parameters."}

        start_time = time.time()
        result = self.executor.execute(tool.get("code", ""), tool.get("name", ""), params)
        runtime = time.time() - start_time
        
        # Log to registry
        success = not result.startswith("Error")
        self.registry.log_execution(tool.get("name"), params, result, success, runtime, None if success else result)
        
        if success:
            self.router.cache_result(tool.get("name"), params, result)
            
        return {"execution_result": result, "next_step": "summarize"}

    async def summarize_node(self, state: AgentState) -> Dict:
        """Summarizes result for the user."""
        query = state.get("user_query", "")
        result = state.get("execution_result", "")
        cached_tag = " (Cached)" if state.get("is_cached") else ""
        
        prompt = f"Summarize this for user query: '{query}'. Result: {result}"
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        
        content = response.content
        if isinstance(content, list):
            content = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in content])
        
        summary = content
        
        return {
            "messages": [AIMessage(content=f"{summary}{cached_tag}")],
            "next_step": "end"
        }

def create_skill_graph(db_path: str, api_key: str):
    manager = SkillGraphManager(db_path, api_key)
    workflow = StateGraph(AgentState)
    
    workflow.add_node("router", manager.router_node)
    workflow.add_node("extract_params", manager.param_extraction_node)
    workflow.add_node("generate_tool", manager.generate_tool_node)
    workflow.add_node("execute_tool", manager.execute_tool_node)
    workflow.add_node("summarize", manager.summarize_node)
    
    workflow.set_entry_point("router")
    
    workflow.add_conditional_edges("router", lambda x: x.get("next_step"), {"extract_params": "extract_params", "generate": "generate_tool"})
    workflow.add_conditional_edges("extract_params", lambda x: x.get("next_step"), {"execute": "execute_tool", "summarize": "summarize", "error": END})
    workflow.add_conditional_edges("generate_tool", lambda x: x.get("next_step"), {"await_approval": END, "error": END})
    workflow.add_edge("execute_tool", "summarize")
    workflow.add_edge("summarize", END)
    
    return workflow.compile(checkpointer=MemorySaver()), manager
