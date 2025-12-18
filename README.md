# LangGraph - Agent creating Agent
## Architecture

User Query
    ↓
analyze_intent ──→ lookup_tool
    ↓                 ↓ existing    ↓ no tool
summarize────────←──execute_tool    generate_tool
    ↑ pending                   ↓
    └──────────────←── DB Save ── pending_approval (END)
