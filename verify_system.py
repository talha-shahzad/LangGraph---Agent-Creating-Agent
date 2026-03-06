import os
import asyncio
import json
import sqlite3
from dotenv import load_dotenv
load_dotenv()
from graph import create_skill_graph
from langchain_core.messages import HumanMessage

async def verify_system():
    api_key = os.getenv("GEMINI_API_KEY")
    db_path = "sqlite:///toolregistry.test.db"
    
    # Remove old test DB if exists
    if os.path.exists("toolregistry.test.db"):
        os.remove("toolregistry.test.db")
        
    workflow, manager = create_skill_graph(db_path, api_key)
    # Use a lower threshold for reliable matching in tests
    manager.router.threshold = 0.3
    
    config = {"configurable": {"thread_id": "test_session_new"}}
    
    print("--- Phase 1: Tool Generation ---")
    query = "Convert 1500 USD to EUR"
    state = {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
        "session_id": "test_session_new",
        "input_tokens": 0,
        "output_tokens": 0,
        "actual_cost": 0.0,
        "cost_saved": 0.0
    }
    
    result = await workflow.ainvoke(state, config=config)
    print(f"Step after generation: {result.get('next_step')}")
    
    if result.get('next_step') == 'await_approval':
        tool_name = result['selected_tool']['name']
        print(f"Generated Tool: {tool_name}")
        
        # Approve the tool
        manager.registry.approve_tool(tool_name, "test_admin")
        
        print("\n--- Phase 2: Reuse from Registry (Deterministic Routing) ---")
        state2 = {
            "messages": [HumanMessage(content=query)],
            "user_query": query,
            "session_id": "test_session_new",
            "input_tokens": 0,
            "output_tokens": 0,
            "actual_cost": 0.0,
            "cost_saved": 0.0
        }
        result2 = await workflow.ainvoke(state2, config=config)
        print(f"Reuse Route: {result2.get('route')}")
        
        print("\n--- Phase 3: Result Caching ---")
        # Run again to hit cache
        result3 = await workflow.ainvoke(state2, config=config)
        print(f"Is Cached: {result3.get('is_cached')}")
        
    print("\n--- Phase 4: Security Check ---")
    bad_code = "import os; os.system('ls')"
    from security import SafeExecutor
    executor = SafeExecutor()
    validation = executor.validate_code(bad_code)
    print(f"Security Validation (Import OS) - Should be False: {validation}")

if __name__ == "__main__":
    asyncio.run(verify_system())
