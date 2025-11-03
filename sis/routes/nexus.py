from __future__ import annotations
from fastapi import APIRouter
from .service import NexusService

def make_router(svc: NexusService) -> APIRouter:
    r = APIRouter(prefix="/nexus")

    @r.get("/path")
    def get_path(start_node_id: str):
        return svc.find_path(start_node_id)

    @r.get("/node")
    def get_node(node_id: str):
        return svc.store.get(node_id).__dict__

    return r
