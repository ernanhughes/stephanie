from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from dataclasses import asdict
from typing import List

from app.models import MemoryEntry
from app.search import run_search
from app.db import fetch_memory_by_id, update_memory_fields

router = APIRouter()

@router.post("/search")
async def search_memory(request: Request):
    data = await request.json()
    query = data.get("query")
    mode = data.get("mode", "hybrid")

    if not query:
        raise HTTPException(status_code=400, detail="Missing query")

    results: List[MemoryEntry] = run_search(query, mode)
    return JSONResponse([asdict(r) for r in results])

@router.get("/memory/{memory_id}")
async def get_memory(memory_id: int):
    memory = fetch_memory_by_id(memory_id)
    if memory:
        return JSONResponse(asdict(memory))
    raise HTTPException(status_code=404, detail="Memory not found")

@router.patch("/memory/{memory_id}")
async def update_memory(memory_id: int, request: Request):
    data = await request.json()
    updated = update_memory_fields(memory_id, data)
    if updated:
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Memory not found")

@router.get("/export/{memory_id}")
async def export_memory(memory_id: int):
    memory = fetch_memory_by_id(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")

    md = f"# {memory.title}\n\n**Tags:** {', '.join(memory.tags)}\n\n---\n\nUSER:\n{memory.user_text}\n\nASSISTANT:\n{memory.ai_text}"
    return JSONResponse({"markdown": md, "json": asdict(memory)})
