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
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from graph import create_skill_graph

# ============================================================================
# Pydantic Models
# ============================================================================

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    model_name: Optional[str] = None

class ToolApproval(BaseModel):
    tool_name: str
    approved: bool
    approved_by: str

class ToolDeletion(BaseModel):
    tool_name: str

class SecurityStatus(BaseModel):
    enabled: bool

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
MODEL_NAME = os.getenv("MODEL_NAME", "models/gemini-1.5-flash")    

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
        "is_cached": False,
        "input_tokens": 0,
        "output_tokens": 0,
        "actual_cost": 0.0,
        "cost_saved": 0.0
    }
    
    try:
        # Use the requested model or fall back to the environment default
        requested_model = request.model_name or MODEL_NAME
        
        # We need to temporarily update the manager's LLM if a different model is requested
        # For a truly multi-user system, we would instantiate this per request or use a factory
        if request.model_name and request.model_name != manager.model_name:
            manager.update_model(request.model_name)

        final_state = await workflow.ainvoke(initial_state, config=config)
        
        # Assemble cost metrics for response
        metrics = {
            "input_tokens": final_state.get("input_tokens", 0),
            "output_tokens": final_state.get("output_tokens", 0),
            "actual_cost": final_state.get("actual_cost", 0.0),
            "cost_saved": final_state.get("cost_saved", 0.0)
        }

        if final_state.get("next_step") == "await_approval":
            return QueryResponse(
                success=True,
                message="New tool generated and awaiting approval",
                pending_approval=True,
                cost_metrics=metrics
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
            result=raw_result,
            cost_metrics=metrics
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

@app.get("/api/security-status")
async def get_security_status():
    return {"success": True, "enabled": manager.executor.enabled}

@app.post("/api/toggle-security")
async def toggle_security(status: SecurityStatus):
    manager.executor.enabled = status.enabled
    return {"success": True, "enabled": manager.executor.enabled, "message": f"Security is now {'enabled' if manager.executor.enabled else 'disabled'}"}

@app.post("/api/delete-tool")
async def delete_tool(deletion: ToolDeletion):
    success = manager.registry.delete_tool(deletion.tool_name)
    message = f"Tool {deletion.tool_name} deleted" if success else f"Failed to delete tool {deletion.tool_name}"
    return {"success": success, "message": message}

@app.get("/api/get-model")
async def get_model():
    model_name = ' '.join(MODEL_NAME.split('-')).capitalize()
    return {"success": True, "model": model_name}

@app.get("/api/models")
async def list_models():
    """Fetches list of available Gemini models for generation."""
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        models = client.models.list()
        
        # Filter for models that support content generation
        gen_models = []
        for m in models:
            # The SDK uses supported_actions (list of strings)
            if "generateContent" in m.supported_actions:
                gen_models.append({
                    "name": m.name,
                    "display_name": m.display_name,
                    "description": m.description
                })
        return {"success": True, "models": gen_models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent:app", host="0.0.0.0", port=3000, reload=True)
