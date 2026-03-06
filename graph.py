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
    # Flattened metrics for simpler LangGraph state management
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]
    actual_cost: Annotated[float, operator.add]
    cost_saved: Annotated[float, operator.add]

# ============================================================================
# Skill Graph Components
# ============================================================================

class SkillGraphManager:
    def __init__(self, db_path: str, api_key: str):
        self.registry = ToolRegistry(db_path)
        self.api_key = api_key
        self.model_name = os.getenv("MODEL_NAME", "models/gemini-1.5-flash")
        self.safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key
        )
        self.router = DeterministicRouter(self.registry, self.embeddings)
        self.executor = SafeExecutor()
        self._set_llm()

    def _set_llm(self):
        """Initializes the LLM with current model and safety settings."""
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=0,
            safety_settings=self.safety_settings
        )

    def update_model(self, model_name: str):
        """Updates the LLM model to a new one."""
        if self.model_name != model_name:
            self.model_name = model_name
            self._set_llm()

    def _extract_json(self, content: str) -> Optional[Dict]:
        """Extracts JSON from LLM response, handling markdown blocks and mixed text."""
        import re
        # 1. Try finding json code block first
        json_block = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_block:
            try:
                clean_json = json_block.group(1).strip()
                return json.loads(clean_json)
            except:
                pass
        
        # 2. Try generic code blocks
        generic_block = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
        if generic_block:
            try:
                return json.loads(generic_block.group(1).strip())
            except:
                pass

        # 3. Comprehensive search for any JSON-like structure
        # Find all indices of '{'
        start_indices = [m.start() for m in re.finditer(r'\{', content)]
        for start_idx in start_indices:
            # Try parsing from this '{' to the end, then shrinking from the right
            substring = content[start_idx:]
            # Find all indices of '}' in the substring
            end_indices = [m.start() for m in re.finditer(r'\}', substring)]
            # Try from furthest '}' backwards
            for end_idx in reversed(end_indices):
                candidate = substring[:end_idx + 1]
                try:
                    return json.loads(candidate)
                except:
                    # Fallback: try to fix common LLM JSON errors (like unescaped newlines in code strings)
                    try:
                        # Replace literal newlines within quotes with \n
                        # This is a bit risky but can help
                        fixed = re.sub(r'(?<=[:\s])"(.*?)"(?=[\s,}])', lambda m: m.group(0).replace('\n', '\\n'), candidate, flags=re.DOTALL)
                        return json.loads(fixed)
                    except:
                        continue
        return None

    def _get_usage(self, response: Any) -> Dict[str, Any]:
        """Extract usage metrics from LLM response."""
        metadata = getattr(response, 'response_metadata', {})
        usage = metadata.get('token_usage', {})
        if usage:
            in_t = usage.get('prompt_tokens', 0)
            out_t = usage.get('completion_tokens', 0)
            # Pricing for Gemini 1.5 Flash (approximate)
            cost = (in_t * 0.000000075) + (out_t * 0.0000003)
            return {"input_tokens": in_t, "output_tokens": out_t, "actual_cost": cost}
        return {"input_tokens": 0, "output_tokens": 0, "actual_cost": 0.0}

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

        # Savings from deterministic routing (skipping LLM tool matcher)
        if route != "generate_new":
            # Estimated cost of an LLM call for routing
            routing_saving = (500 * 0.000000075) + (100 * 0.0000003)
            updates["cost_saved"] = routing_saving

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
        
        usage = self._get_usage(response)
        updates = usage.copy()
        
        params = self._extract_json(content)
        if params:
            updates["tool_params"] = params
            
            cached = self.router.check_cache(tool["name"], params)
            if cached:
                updates.update({
                    "execution_result": cached,
                    "is_cached": True,
                    "next_step": "summarize",
                    # Savings from not needing to execute or summarize (approximate summary cost)
                    "cost_saved": (400 * 0.000000075) + (100 * 0.0000003) 
                })
            else:
                updates.update({"is_cached": False, "next_step": "execute"})
        else:
            updates.update({"next_step": "error", "execution_result": "Failed to extract parameters."})
            
        return updates

    async def generate_tool_node(self, state: AgentState) -> Dict:
        """Generates a new reusable tool via LLM."""
        query = state.get("user_query", "")
        prompt = (
            f"Create a reusable Python tool for: '{query}'.\n"
            "Return ONLY a JSON object with exactly these keys: 'name', 'description', 'code', 'parameters'.\n"
            "IMPORTANT: The 'code' field must be a single string. Use '\\n' for newlines and ensure all quotes are escaped."
        )
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        
        content = response.content
        if isinstance(content, list):
            content = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in content])
            
        usage = self._get_usage(response)
        updates = usage.copy()
        
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
        usage = self._get_usage(response)
        
        content = response.content
        if isinstance(content, list):
            content = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in content])
        
        summary = content
        
        # Assemble final metrics for possible logging/debugging
        in_t = state.get("input_tokens", 0) + usage.get("input_tokens", 0)
        out_t = state.get("output_tokens", 0) + usage.get("output_tokens", 0)
        total_cost = state.get("actual_cost", 0.0) + usage.get("actual_cost", 0.0)
        savings = state.get("cost_saved", 0.0)
        
        cost_info = f"\n\n--- Cost Metrics ---\nTokens: {in_t} in / {out_t} out\nActual Cost: ${total_cost:.6f}\n**Total Savings: ${savings:.6f}**"
        
        return {
            **usage,
            "messages": [AIMessage(content=f"{summary}{cached_tag}{cost_info}")],
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
