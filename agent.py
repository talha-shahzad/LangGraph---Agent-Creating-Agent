"""
LangGraph Dynamic Tool Agent with FastAPI Backend and Gemini
A self-improving agent that generates, approves, and executes tools dynamically.
"""

import os
import json
import asyncio
import re
import ast
import sqlite3
import time
from typing import TypedDict, Annotated, Literal, Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

# FastAPI
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Web scraping tools
import requests
from bs4 import BeautifulSoup

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# LangSmith Tracing
if os.getenv("LANGSMITH_TRACING") == "true":
    print("LangSmith tracing is enabled.")

# ============================================================================
# Pydantic Models
# ============================================================================

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class ToolApproval(BaseModel):
    tool_name: str
    approved: bool
    approved_by: str


class QueryResponse(BaseModel):
    success: bool
    message: str
    tool_generated: bool = False
    pending_approval: bool = False
    tool_details: Optional[Dict] = None
    result: Optional[str] = None


# ============================================================================
# Database Setup
# ============================================================================

class ToolRegistry:
    """SQLite-backed tool registry"""
    
    def __init__(self, connection_string: str):
        """
        connection_string example:
        sqlite:///toolregistry.db
        """
        try:
            self.db_path = self._parse_sqlite_path(connection_string)
            self._init_db()
        except Exception as e:
            print(f"Error initializing ToolRegistry: {e}")
            raise

    def _parse_sqlite_path(self, conn_str: str) -> str:
        if not conn_str.startswith("sqlite:///"):
            raise ValueError(f"Invalid SQLite connection string: {conn_str}")
        return conn_str.replace("sqlite:///", "", 1)

    def _init_db(self):
        """Initialize SQLite database tables"""
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()

            # tools table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tools (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT NOT NULL,
                    code TEXT NOT NULL,
                    parameters TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    approved_by TEXT,
                    usage_count INTEGER DEFAULT 0,
                    last_used DATETIME,
                    status TEXT DEFAULT 'pending'
                )
            """)

            # tool executions table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tool_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_name TEXT,
                    input_params TEXT,
                    output TEXT,
                    success INTEGER,
                    error_message TEXT,
                    executed_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # pending approvals table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pending_approvals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_name TEXT UNIQUE,
                    session_id TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"Error initializing database at {self.db_path}: {e}")
            raise
    
    def search_tools(self, query: str) -> List[Dict]:
        """Search for existing tools by description or name"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            
            cur.execute("""
                SELECT * FROM tools 
                WHERE status = 'active' 
                AND (name LIKE ? OR description LIKE ?)
                ORDER BY usage_count DESC
                LIMIT 5
            """, (f'%{query}%', f'%{query}%'))
            
            tools = [dict(row) for row in cur.fetchall()]
            cur.close()
            conn.close()
            return tools
        except Exception as e:
            print(f"Error searching tools with query '{query}': {e}")
            return []
    
    def get_tool(self, name: str) -> Optional[Dict]:
        """Get a specific tool by name"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            
            cur.execute("SELECT * FROM tools WHERE name = ? AND status = 'active'", (name,))
            tool = cur.fetchone()
            result = dict(tool) if tool else None
            
            cur.close()
            conn.close()
            return result
        except Exception as e:
            print(f"Error getting tool '{name}': {e}")
            return None
    
    def save_tool(self, name: str, description: str, code: str, 
                  parameters: Dict, approved_by: str, status: str = 'pending') -> bool:
        """Save a new tool to the registry"""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO tools (name, description, code, parameters, approved_by, status)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (name) DO UPDATE 
                SET description = EXCLUDED.description,
                    code = EXCLUDED.code,
                    parameters = EXCLUDED.parameters,
                    approved_by = EXCLUDED.approved_by,
                    status = EXCLUDED.status
            """, (name, description, code, json.dumps(parameters), approved_by, status))
            
            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            print(f"Error saving tool '{name}': {e}")
            return False
    
    def approve_tool(self, name: str, approved_by: str) -> bool:
        """Approve a pending tool"""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            cur.execute("""
                UPDATE tools 
                SET status = 'active', approved_by = ?
                WHERE name = ?
            """, (approved_by, name))
            
            cur.execute("DELETE FROM pending_approvals WHERE tool_name = ?", (name,))
            
            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            print(f"Error approving tool '{name}': {e}")
            return False
    
    def reject_tool(self, name: str) -> bool:
        """Reject a pending tool"""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            cur.execute("UPDATE tools SET status = 'rejected' WHERE name = ?", (name,))
            cur.execute("DELETE FROM pending_approvals WHERE tool_name = ?", (name,))
            
            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            print(f"Error rejecting tool '{name}': {e}")
            return False
    
    def add_pending_approval(self, tool_name: str, session_id: str) -> bool:
        """Add tool to pending approvals"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO pending_approvals (tool_name, session_id)
                VALUES (?, ?)
                ON CONFLICT (tool_name) DO NOTHING
            """, (tool_name, session_id))
            
            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            print(f"Error adding pending approval for tool '{tool_name}': {e}")
            return False
    
    def get_pending_approvals(self) -> List[Dict]:
        """Get all pending tool approvals"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            
            cur.execute("""
                SELECT t.*, pa.session_id, pa.created_at as pending_since
                FROM tools t
                JOIN pending_approvals pa ON t.name = pa.tool_name
                WHERE t.status = 'pending'
                ORDER BY pa.created_at DESC
            """)
            
            tools = [dict(row) for row in cur.fetchall()]
            cur.close()
            conn.close()
            return tools
        except Exception as e:
            print(f"Error getting pending approvals: {e}")
            return []
    
    def get_all_tools(self) -> List[Dict]:
        """Get all tools"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            
            cur.execute("""
                SELECT * FROM tools 
                ORDER BY usage_count DESC, created_at DESC
            """)
            
            tools = [dict(row) for row in cur.fetchall()]
            cur.close()
            conn.close()
            return tools
        except Exception as e:
            print(f"Error getting all tools: {e}")
            return []
    
    def log_execution(self, tool_name: str, input_params: Dict, 
                      output: str, success: bool, error_message: str = None):
        """Log tool execution"""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO tool_executions 
                (tool_name, input_params, output, success, error_message)
                VALUES (?, ?, ?, ?, ?)
            """, (tool_name, json.dumps(input_params), output, 1 if success else 0, error_message))
            
            # Update usage count and last_used
            cur.execute("""
                UPDATE tools 
                SET usage_count = usage_count + 1,
                    last_used = CURRENT_TIMESTAMP
                WHERE name = ?
            """, (tool_name,))
            
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"Error logging execution for tool '{tool_name}': {e}")


# ============================================================================
# Static Tools
# ============================================================================

def web_search_func(query: str) -> str:
    """
    Searchs the internet for current information, news, or general knowledge. 
    Use this only when the user asks a question that requires external data, 
    up-to-date facts, or information not present in your training data. Do not 
    when work can be done by creating a tool for repetetive tasks
    """
    try:
        url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        results = []
        for result in soup.find_all('div', class_='result')[:5]:
            title_elem = result.find('a', class_='result__a')
            snippet_elem = result.find('a', class_='result__snippet')
            
            if title_elem and snippet_elem:
                results.append({
                    'title': title_elem.get_text(),
                    'url': title_elem.get('href', ''),
                    'snippet': snippet_elem.get_text()
                })
        
        return json.dumps(results, indent=2)
    except Exception as e:
        print(f"Error in web_search_func for query '{query}': {e}")
        return f"Error searching web: {str(e)}"


def scrape_webpage_func(url: str) -> str:
    """
    Fetch and extract the main text content from a specific URL. 
    Use this when you have a specific link and need to read its content 
    to answer a question, summarize the page, or find specific details within that page.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text[:5000]  # Limit to first 5000 chars
    except Exception as e:
        print(f"Error in scrape_webpage_func for URL '{url}': {e}")
        return f"Error scraping webpage: {str(e)}"


# ============================================================================
# Agent State
# ============================================================================

class AgentState(TypedDict):
    messages: List[Any]
    user_query: str
    session_id: str
    intent: str
    existing_tool: Optional[Dict]
    generated_tool: Optional[Dict]
    tool_approved: bool
    execution_result: str
    next_step: str


# ============================================================================
# Agent Class
# ============================================================================

class DynamicToolAgent:
    def __init__(self, db_connection: str, gemini_api_key: str):
        try:
            self.registry = ToolRegistry(db_connection)
            self.llm = ChatGoogleGenerativeAI(
                model=os.getenv("MODEL_NAME", "gemini-1.5-flash"),
                google_api_key=gemini_api_key,
                temperature=0
            )
            self.static_tools = {
                'web_search': {
                    'name': 'web_search',
                    'description': 'Search the internet for current information, news, or general knowledge. Use this when the user asks a question that requires external data, up-to-date facts, or information not present in your training data.',
                    'func': web_search_func,
                    'type': 'static'
                },
                'scrape_webpage': {
                    'name': 'scrape_webpage',
                    'description': 'Fetch and extract the main text content from a specific URL. Use this when you have a specific link and need to read its content to answer a question, summarize the page, or find specific details within that page.',
                    'func': scrape_webpage_func,
                    'type': 'static'
                }
            }
        except Exception as e:
            print(f"Error initializing DynamicToolAgent: {e}")
            raise
    
    def _extract_text(self, content: Any) -> str:
        """Extract text from LLM response content which can be a string or list of parts"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
            return "".join(text_parts)
        return str(content)
    
    async def analyze_intent(self, state: AgentState) -> AgentState:
        """Analyze user query to understand intent"""
        try:
            query = state["user_query"]
            
            # Get list of available static and dynamic tools
            static_tool_names = list(self.static_tools.keys())
            dynamic_tools = self.registry.get_all_tools()
            dynamic_tool_names = [t['name'] for t in dynamic_tools if t['status'] == 'active']
            
            prompt = f"""You are an intelligent intent analyzer for a tool-based AI agent. Analyze the user's query and determine the appropriate action.

USER QUERY: "{query}"

AVAILABLE STATIC TOOLS:
- web_search: Search the web for current information, news, articles
- scrape_webpage: Extract content from a specific URL/website

AVAILABLE DYNAMIC TOOLS:
{', '.join(dynamic_tool_names) if dynamic_tool_names else 'None yet'}

INTENT CLASSIFICATION RULES:

1. **execute_existing** - Use when:
   - Query explicitly asks to search the web (e.g., "search for...", "find information about...", "what's the latest news on...")
   - Query provides a URL/link and asks to extract/scrape content (e.g., "scrape vimeo.com", "get content from example.com", "extract text from https://...")
   - Query matches an existing dynamic tool's capability (e.g., if we have a time_converter tool and user asks "what time is it in Tokyo")
   - User wants to perform an action that an existing tool can handle
   
   EXAMPLES:
   - "search the web for AI developments" → web_search
   - "scrape content from https://vimeo.com/123456" → scrape_webpage
   - "get me information from news.ycombinator.com" → scrape_webpage
   - "find latest Python tutorials" → web_search

2. **need_new_tool** - Use when:
   - Query requires a NEW computational capability not available in existing tools
   - Query asks for calculations, conversions, or processing that needs custom logic
   - Query involves time zones, currency conversion, mathematical operations, data processing, etc.
   - Create GENERIC tools that handle broad categories (e.g., "time tool" not "Hawaii time tool")
   
   IMPORTANT: Tools should be GENERIC and REUSABLE:
   - For time queries → create a general "timezone_converter" tool (handles ANY timezone)
   - For currency queries → create a general "currency_converter" tool (handles ANY currency)
   - For math queries → create a general "calculator" or specific math tool
   - For unit conversions → create a general "unit_converter" tool
   
   EXAMPLES:
   - "what time is it in Hawaii?" → need timezone_converter tool (generic for all timezones)
   - "what's the time difference between HST and PST?" → need timezone_converter tool
   - "convert 100 USD to EUR" → need currency_converter tool (generic for all currencies)
   - "calculate factorial of 15" → need factorial_calculator tool
   - "convert 10 miles to kilometers" → need unit_converter tool
   - "generate a random password" → need password_generator tool

3. **direct_answer** - Use when:
   - Query asks about general knowledge, concepts, or definitions available in the model's training
   - Query asks "what is X?" where X is a well-known concept (e.g., "what is GenAI?", "what is LangGraph?")
   - Query asks for explanations of established topics (e.g., "how does machine learning work?")
   - Query is conversational or asks for advice/opinions
   - NO external data, computation, or web access is needed
   
   EXAMPLES:
   - "what is generative AI?" → explain from knowledge
   - "how does LangGraph work?" → explain from knowledge
   - "explain quantum computing" → explain from knowledge
   - "what are the benefits of Python?" → explain from knowledge
   - "hello, how are you?" → conversational response

DECISION LOGIC:
1. First check: Does query contain a URL or ask to search/scrape? → execute_existing
2. Then check: Does query need real-time data, calculations, or capabilities not in existing tools? → need_new_tool
3. Finally: Can I answer from my training data alone? → direct_answer

Respond with ONLY ONE of these three options:
- execute_existing
- need_new_tool
- direct_answer
"""
            
            await asyncio.sleep(2)
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            intent = self._extract_text(response.content).strip().lower()
            
            # Validate intent
            valid_intents = ["execute_existing", "need_new_tool", "direct_answer"]
            if intent not in valid_intents:
                intent = "direct_answer"
            
            state["intent"] = intent
            state["messages"].append(AIMessage(content=f"Intent: {intent}"))
            return state
        except Exception as e:
            print(f"Error in analyze_intent for query '{state.get('user_query')}': {e}")
            state["next_step"] = "error"
            state["execution_result"] = f"Error analyzing intent: {str(e)}"
            return state
    
    async def lookup_tool(self, state: AgentState) -> AgentState:
        """Search for existing tools in registry"""
        try:
            query = state["user_query"]
            intent = state.get("intent", "")
            
            # If intent is direct_answer, skip tool lookup
            if intent == "direct_answer":
                state["next_step"] = "direct_answer"
                return state
            
            # If intent is need_new_tool, we should only check if a dynamic tool already exists
            # that matches this specific need. If not, go to generate.
            
            # Get all available tools for the LLM to choose from
            dynamic_tools = self.registry.get_all_tools()
            active_dynamic_tools = [t for t in dynamic_tools if t['status'] == 'active']
            
            available_tools_info = []
            for name, tool in self.static_tools.items():
                available_tools_info.append(f"- {name}: {tool['description']}")
            for tool in active_dynamic_tools:
                available_tools_info.append(f"- {tool['name']}: {tool['description']}")
            
            tools_list_str = "\n".join(available_tools_info)
            
            prompt = f"""Analyze the user query and the available tools. 
Determine if any EXISTING tool can fulfill the request.

USER QUERY: \"{query}\"
INTENT: {intent}

AVAILABLE TOOLS:
{tools_list_str}

RULES:
1. If intent is 'need_new_tool', only select an existing tool if it EXACTLY matches the category (e.g., a 'timezone_converter' tool for a time query).
2. If intent is 'execute_existing', select the most appropriate tool (e.g., 'web_search' or 'scrape_webpage').
3. If NO existing tool is a good match, respond with 'NONE'.

Respond with ONLY the name of the tool or 'NONE'."""

            await asyncio.sleep(2)
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            selected_tool_name = self._extract_text(response.content).strip()
            
            if selected_tool_name != "NONE":
                # Find the tool details
                tool_details = None
                if selected_tool_name in self.static_tools:
                    # Store serializable version of static tool
                    static_tool = self.static_tools[selected_tool_name]
                    tool_details = {
                        'name': static_tool['name'],
                        'description': static_tool['description'],
                        'type': 'static'
                    }
                else:
                    for t in active_dynamic_tools:
                        if t['name'] == selected_tool_name:
                            tool_details = t
                            break
                
                if tool_details:
                    state["existing_tool"] = tool_details
                    state["next_step"] = "execute"
                    return state
            
            # If no tool found or selected
            if intent == "need_new_tool":
                state["existing_tool"] = None
                state["next_step"] = "generate"
            else:
                # If intent was execute_existing but no tool matched, this is an edge case
                # We'll try to generate a tool if it seems appropriate, or error out
                state["existing_tool"] = None
                state["next_step"] = "generate"
            
            return state
        except Exception as e:
            print(f"Error in lookup_tool for query '{state.get('user_query')}': {e}")
            state["next_step"] = "error"
            state["execution_result"] = f"Error looking up tool: {str(e)}"
            return state
    
    async def direct_answer(self, state: AgentState) -> AgentState:
        """Provide direct answer using LLM's knowledge"""
        try:
            query = state["user_query"]
            
            prompt = f"""You are a helpful AI assistant. Answer the following question using your knowledge and training data. Provide a clear, accurate, and concise response.

Question: {query}

Provide a helpful answer that directly addresses the user's question. Be informative but concise."""
            
            await asyncio.sleep(2)
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            
            content_text = self._extract_text(response.content)
            state["execution_result"] = content_text
            state["messages"].append(AIMessage(content=content_text))
            state["next_step"] = "end"
            
            return state
        except Exception as e:
            print(f"Error in direct_answer for query '{state.get('user_query')}': {e}")
            state["next_step"] = "error"
            state["execution_result"] = f"Error providing direct answer: {str(e)}"
            return state
    
    async def generate_tool(self, state: AgentState) -> AgentState:
        """Generate a new tool using LLM"""
        try:
            query = state["user_query"]
            
            prompt = f"""You are an expert Python developer creating reusable, generic tools for an AI agent system.

USER REQUEST: "{query}"

IMPORTANT PRINCIPLES:
1. **Create GENERIC, REUSABLE tools** - Don't create single-use tools
2. **Broad capability** - Tool should handle the entire category, not just one specific case
3. **Parameterized** - Use parameters to handle different inputs within the category
4. **Future-proof** - Other users should be able to use this tool for similar needs

EXAMPLES OF GOOD vs BAD TOOL DESIGN:

❌ BAD (Too Specific):
- Tool name: "get_hawaii_time" - only works for Hawaii
- Tool name: "usd_to_eur_converter" - only converts USD to EUR
- Tool name: "calculate_10_factorial" - only calculates factorial of 10

✅ GOOD (Generic & Reusable):
- Tool name: "timezone_converter" - handles ANY timezone conversion
- Tool name: "currency_converter" - handles ANY currency pair
- Tool name: "factorial_calculator" - calculates factorial of ANY number

TOOL CATEGORY DETECTION:
- Time/timezone queries → Create "timezone_converter" or "world_clock"
- Currency queries → Create "currency_converter"
- Math operations → Create specific calculator (e.g., "factorial_calculator", "fibonacci_calculator")
- Unit conversions → Create "unit_converter" (distance, weight, temperature, etc.)
- Date calculations → Create "date_calculator"
- Password/string generation → Create respective generator
- Data processing → Create specific processor

REQUIREMENTS:
1. Function name: snake_case, descriptive of the CATEGORY (not specific instance)
2. Parameters: Make the tool flexible with well-named parameters
3. Type hints: Include proper type annotations
4. Docstring: Clear documentation explaining the tool's broad capabilities
5. Error handling: Handle edge cases and invalid inputs gracefully
6. Return format: Always return a string with clear, formatted output
7. Libraries: Use only standard library or: requests, json, datetime, math, re, random

EXAMPLE IMPLEMENTATIONS:

For "what time is it in Hawaii?":
```python
def timezone_converter(timezone_from: str = "UTC", timezone_to: str = "UTC") -> str:
    \"\"\"
    Convert time between any two timezones.
    
    Args:
        timezone_from: Source timezone (e.g., 'UTC', 'US/Eastern', 'US/Pacific')
        timezone_to: Target timezone (e.g., 'US/Hawaii', 'Asia/Tokyo', 'Europe/London')
    
    Returns:
        Current time in target timezone with comparison
    \"\"\"
    from datetime import datetime
    import pytz
    
    try:
        # Get current time in source timezone
        tz_from = pytz.timezone(timezone_from)
        tz_to = pytz.timezone(timezone_to)
        
        now_from = datetime.now(tz_from)
        now_to = now_from.astimezone(tz_to)
        
        return f"Time in {{timezone_to}}: {{now_to.strftime('%Y-%m-%d %H:%M:%S %Z')}}\\nTime in {{timezone_from}}: {{now_from.strftime('%Y-%m-%d %H:%M:%S %Z')}}"
    except Exception as e:
        return f"Error converting timezone: {{str(e)}}. Use standard timezone names like 'US/Hawaii', 'US/Pacific', 'UTC'."
```

For "convert 100 USD to EUR":
```python
def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    \"\"\"
    Convert any amount from one currency to another using live exchange rates.
    
    Args:
        amount: Amount to convert
        from_currency: Source currency code (e.g., 'USD', 'EUR', 'GBP')
        to_currency: Target currency code
    
    Returns:
        Converted amount with exchange rate information
    \"\"\"
    import requests
    
    try:
        # Use a free exchange rate API
        url = f"https://api.exchangerate-api.com/v4/latest/{{from_currency}}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if to_currency not in data['rates']:
            return f"Error: Currency code '{{to_currency}}' not found."
        
        rate = data['rates'][to_currency]
        converted = amount * rate
        
        return f"{{amount}} {{from_currency}} = {{converted:.2f}} {{to_currency}}\\nExchange rate: 1 {{from_currency}} = {{rate:.4f}} {{to_currency}}"
    except Exception as e:
        return f"Error converting currency: {{str(e)}}"
```

Now, analyze the user's request and generate a GENERIC, REUSABLE tool.

Respond with a JSON object containing:
{{
    "name": "generic_tool_name",
    "description": "Brief description of the tool's broad capabilities",
    "code": "def generic_tool_name(param: type) -> str:\\n    ...complete implementation...",
    "parameters": {{"param_name": "description of parameter and what values it accepts"}}
}}

CRITICAL: 
- Tool name should reflect the CATEGORY, not the specific query
- Parameters should allow flexibility for future similar requests
- Include comprehensive error handling
- Return user-friendly formatted strings

Only return the JSON, nothing else.
"""
            
            await asyncio.sleep(2)
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            
            # Extract JSON from response
            content = self._extract_text(response.content)
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                tool_spec = json.loads(json_match.group())
                state["generated_tool"] = tool_spec
                
                # Save as pending
                self.registry.save_tool(
                    tool_spec['name'],
                    tool_spec['description'],
                    tool_spec['code'],
                    tool_spec['parameters'],
                    'system',
                    'pending'
                )
                
                # Add to pending approvals
                self.registry.add_pending_approval(
                    tool_spec['name'],
                    state.get('session_id', 'unknown')
                )
                
                state["next_step"] = "pending_approval"
            else:
                state["next_step"] = "error"
                state["execution_result"] = "Failed to generate tool: Invalid LLM response"
            
            return state
        except Exception as e:
            print(f"Error in generate_tool for query '{state.get('user_query')}': {e}")
            state["next_step"] = "error"
            state["execution_result"] = f"Error generating tool: {str(e)}"
            return state
    
    async def execute_tool(self, state: AgentState) -> AgentState:
        """Execute the selected tool"""
        try:
            if state.get("existing_tool"):
                tool = state["existing_tool"]
                tool_name = tool['name']
                
                # Execute static tool
                if tool.get("type") == "static":
                    # Look up the actual function
                    static_tool_func = self.static_tools[tool_name]['func']
                    
                    param_prompt = f"""Extract parameters for the {tool_name} function from this query:
Query: {state['user_query']}
Function: {tool['description']}

Return ONLY a JSON object with the parameter values, e.g. {{"query": "search term"}} or {{"url": "http://..."}}
"""
                    await asyncio.sleep(2)
                    response = await self.llm.ainvoke([HumanMessage(content=param_prompt)])
                    params_match = re.search(r'\{.*\}', self._extract_text(response.content))
                    if not params_match:
                        raise ValueError("Could not extract parameters from LLM response")
                    params = json.loads(params_match.group())
                    
                    try:
                        result = static_tool_func(**params)
                        state["execution_result"] = result
                        self.registry.log_execution(tool["name"], params, result, True)
                    except Exception as e:
                        state["execution_result"] = f"Error: {str(e)}"
                        self.registry.log_execution(tool["name"], params, "", False, str(e))
                
                # Execute dynamic tool from registry
                else:
                    try:
                        namespace = {
                            'requests': requests,
                            'json': json,
                            'datetime': datetime,
                            're': re,
                            '__builtins__': __builtins__
                        }
                        
                        exec(tool['code'], namespace)
                        func = namespace[tool['name']]
                        
                        param_prompt = f"""Extract parameters for this function from the query:
Query: {state['user_query']}
Parameters needed: {tool['parameters']}

Return ONLY a JSON object with parameter values.
"""
                        await asyncio.sleep(2)
                        response = await self.llm.ainvoke([HumanMessage(content=param_prompt)])
                        params_match = re.search(r'\{.*\}', self._extract_text(response.content))
                        if not params_match:
                            raise ValueError("Could not extract parameters from LLM response")
                        params = json.loads(params_match.group())
                        
                        result = func(**params)
                        state["execution_result"] = str(result)
                        self.registry.log_execution(tool["name"], params, result, True)
                        
                    except Exception as e:
                        state["execution_result"] = f"Error executing tool: {str(e)}"
                        self.registry.log_execution(tool["name"], {}, "", False, str(e))
            
            state["next_step"] = "summarize"
            return state
        except Exception as e:
            print(f"Error in execute_tool for tool '{state.get('existing_tool', {}).get('name')}': {e}")
            state["next_step"] = "error"
            state["execution_result"] = f"Error executing tool: {str(e)}"
            return state
    
    async def summarize_result(self, state: AgentState) -> AgentState:
        """Summarize execution result for user"""
        try:
            result = state["execution_result"]
            query = state["user_query"]
            
            prompt = f"""Summarize this result in response to the user's query:

Query: {query}
Result: {result}

Provide a clear, concise summary that directly answers the user's question.
"""
            
            await asyncio.sleep(2)
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            state["messages"].append(AIMessage(content=self._extract_text(response.content)))
            state["next_step"] = "end"
            
            return state
        except Exception as e:
            print(f"Error in summarize_result for query '{state.get('user_query')}': {e}")
            state["messages"].append(AIMessage(content=f"I processed your request, but encountered an error while summarizing the result: {str(e)}. The raw result was: {state.get('execution_result')}"))
            state["next_step"] = "end"
            return state
    
    def route_next(self, state: AgentState) -> str:
        """Determine next node based on state"""
        next_step = state.get("next_step", "")
        
        if next_step == "execute":
            return "execute_tool"
        elif next_step == "generate":
            return "generate_tool"
        elif next_step == "pending_approval":
            return END
        elif next_step == "summarize":
            return "summarize_result"
        elif next_step == "direct_answer":
            return "direct_answer"
        elif next_step == "end":
            return END
        elif next_step == "error":
            return END
        else:
            return END


# ============================================================================
# LangGraph Workflow
# ============================================================================

def create_workflow(db_connection: str, gemini_api_key: str) -> StateGraph:
    """Create the LangGraph workflow"""
    try:
        agent = DynamicToolAgent(db_connection, gemini_api_key)
        
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_intent", agent.analyze_intent)
        workflow.add_node("lookup_tool", agent.lookup_tool)
        workflow.add_node("generate_tool", agent.generate_tool)
        workflow.add_node("execute_tool", agent.execute_tool)
        workflow.add_node("direct_answer", agent.direct_answer)
        workflow.add_node("summarize_result", agent.summarize_result)
        
        # Add edges
        workflow.set_entry_point("analyze_intent")
        workflow.add_edge("analyze_intent", "lookup_tool")
        workflow.add_conditional_edges(
            "lookup_tool",
            agent.route_next,
            {
                "execute_tool": "execute_tool",
                "generate_tool": "generate_tool",
                "direct_answer": "direct_answer"
            }
        )
        workflow.add_conditional_edges(
            "generate_tool",
            agent.route_next,
            {
                END: END
            }
        )
        workflow.add_edge("execute_tool", "summarize_result")
        workflow.add_conditional_edges(
            "direct_answer",
            agent.route_next,
            {
                END: END
            }
        )
        workflow.add_edge("summarize_result", END)
        
        return workflow.compile(), agent
    except Exception as e:
        print(f"Error creating workflow: {e}")
        raise


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(title="LangGraph Tool Agent API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
db_connection = os.getenv("DATABASE_URL", "sqlite:///toolregistry.db")
gemini_api_key = os.getenv("GEMINI_API_KEY")

try:
    workflow, agent = create_workflow(db_connection, gemini_api_key)
    registry = ToolRegistry(db_connection)
except Exception as e:
    print(f"CRITICAL: Failed to initialize application: {e}")
    # We don't raise here to allow FastAPI to start, but endpoints will fail
    workflow, agent, registry = None, None, None


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend"""
    try:
        frontend_file = os.getenv("FRONTEND_FILE", "frontend.html")
        with open(frontend_file, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error serving {frontend_file}: {e}")
        raise HTTPException(status_code=500, detail=f"Frontend file {frontend_file} not found")


@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process user query"""
    if not workflow:
        raise HTTPException(status_code=503, detail="Application not initialized")
    
    try:
        initial_state = {
            "messages": [HumanMessage(content=request.query)],
            "user_query": request.query,
            "session_id": request.session_id or "default",
            "intent": "",
            "existing_tool": None,
            "generated_tool": None,
            "tool_approved": False,
            "execution_result": "",
            "next_step": ""
        }
        
        final_state = await workflow.ainvoke(initial_state)
        
        # Check if tool is pending approval
        if final_state.get("generated_tool") and final_state.get("next_step") == "pending_approval":
            return QueryResponse(
                success=True,
                message="New tool generated and awaiting approval",
                tool_generated=True,
                pending_approval=True,
                tool_details=final_state["generated_tool"]
            )
        
        # Return execution result
        if final_state.get("messages"):
            last_message = final_state["messages"][-1]
            if isinstance(last_message, AIMessage):
                return QueryResponse(
                    success=True,
                    message="Query processed successfully",
                    result=agent._extract_text(last_message.content)
                )
        
        return QueryResponse(
            success=True,
            message="Query processed",
            result=final_state.get("execution_result", "No result")
        )
        
    except Exception as e:
        print(f"Error processing query '{request.query}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pending-tools")
async def get_pending_tools():
    """Get all pending tool approvals"""
    if not registry:
        raise HTTPException(status_code=503, detail="Application not initialized")
    try:
        tools = registry.get_pending_approvals()
        return {"success": True, "tools": tools}
    except Exception as e:
        print(f"Error getting pending tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/all-tools")
async def get_all_tools():
    """Get all tools"""
    if not registry:
        raise HTTPException(status_code=503, detail="Application not initialized")
    try:
        tools = registry.get_all_tools()
        return {"success": True, "tools": tools}
    except Exception as e:
        print(f"Error getting all tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/approve-tool")
async def approve_tool(approval: ToolApproval):
    """Approve a pending tool"""
    if not registry:
        raise HTTPException(status_code=503, detail="Application not initialized")
    try:
        if approval.approved:
            success = registry.approve_tool(approval.tool_name, approval.approved_by)
            message = "Tool approved successfully" if success else "Failed to approve tool"
        else:
            success = registry.reject_tool(approval.tool_name)
            message = "Tool rejected successfully" if success else "Failed to reject tool"
        
        return {"success": success, "message": message}
    except Exception as e:
        print(f"Error approving/rejecting tool '{approval.tool_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent:app", host="0.0.0.0", port=3000, reload=True)