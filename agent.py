"""
FastAPI Entry Point for Skill Graph Agent.
"""
import os
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage

from graph import create_skill_graph

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
    response: Optional[str] = None  # Conversational summary
    result: Optional[Any] = None    # Raw execution data
    cost_metrics: Optional[Dict[str, Any]] = None

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(title="Skill Graph Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///toolregistry.db")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "Gemini 1.5 Flash")    

workflow, manager = create_skill_graph(DATABASE_URL, GEMINI_API_KEY)

@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open("frontend.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        raise HTTPException(status_code=500, detail="Frontend file not found")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("favicon.ico")

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    session_id = request.session_id or "default_session"
    
    config = {"configurable": {"thread_id": session_id}}
    initial_state = {
        "messages": [HumanMessage(content=request.query)],
        "user_query": request.query,
        "session_id": session_id,
        "is_cached": False
    }
    
    try:
        final_state = await workflow.ainvoke(initial_state, config=config)
        
        if final_state.get("next_step") == "await_approval":
            return QueryResponse(
                success=True,
                message="New tool generated and awaiting approval",
                pending_approval=True
            )
        
        # Get last AI message (the summary)
        response_text = ""
        for msg in reversed(final_state.get("messages", [])):
            if isinstance(msg, AIMessage):
                response_text = msg.content
                break
        
        # Get the raw execution result
        raw_result = final_state.get("execution_result")
        
        return QueryResponse(
            success=True,
            message="Query processed",
            response=response_text or "No summary produced",
            result=raw_result
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pending-tools")
async def get_pending_tools():
    return {"success": True, "tools": manager.registry.get_pending_approvals()}

@app.get("/api/all-tools")
async def get_all_tools():
    return {"success": True, "tools": manager.registry.get_all_tools()}

@app.post("/api/approve-tool")
async def approve_tool(approval: ToolApproval):
    if approval.approved:
        success = manager.registry.approve_tool(approval.tool_name, approval.approved_by)
        message = "Tool approved"
    else:
        success = manager.registry.reject_tool(approval.tool_name)
        message = "Tool rejected"
    return {"success": success, "message": message}

@app.get("/api/get-model")
async def get_model():
    return_name = ' '.join(word.capitalize() for word in str(MODEL_NAME).split('-'))
    return {"success": True, "model": return_name}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent:app", host="0.0.0.0", port=3000, reload=True)
