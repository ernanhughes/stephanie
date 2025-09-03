# sis/routes/rankings.py
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import text

router = APIRouter()


@router.get("/rankings", response_class=HTMLResponse)
def view_rankings(request: Request, top_n: int = 5, target_type: str = None):
    """
    Dashboard: grouped scorable rankings with top-N per query.
    """
    memory = request.app.state.memory
    session = memory.session
    templates = request.app.state.templates

    filters = []
    params = {}
    if target_type:
        filters.append("e.target_type = :target_type")
        params["target_type"] = target_type

    where_clause = "WHERE " + " AND ".join(filters) if filters else ""

    sql = text(f"""
        SELECT *
        FROM (
            SELECT 
                e.id AS evaluation_id,
                e.created_at,
                e.query_type,
                e.query_id,
                e.target_type,
                e.target_id,
                s.dimension,
                s.score,
                s.weight,
                e.scores ->> 'rank_score' AS rank_score,
                ROW_NUMBER() OVER (
                    PARTITION BY e.query_id 
                    ORDER BY (e.scores ->> 'rank_score')::float DESC
                ) as rn
            FROM evaluations e
            JOIN scores s ON e.id = s.evaluation_id
            {where_clause}
        ) ranked
        WHERE rn <= :top_n
        ORDER BY created_at DESC, query_id, rn
    """)

    rows = session.execute(sql, {**params, "top_n": top_n}).fetchall()
    rankings = [dict(row._mapping) for row in rows]

    grouped = {}
    for r in rankings:
        grouped.setdefault(r["query_id"], []).append(r)

    return templates.TemplateResponse(
        "rankings.html",
        {
            "request": request,
            "grouped_rankings": grouped,
            "target_type": target_type,
            "top_n": top_n,
        },
    )


@router.get("/rankings/{query_id}", response_class=HTMLResponse)
def view_ranking_detail(request: Request, query_id: str):
    """
    Detail view: show full ranking history for a single query_id.
    """
    memory = request.app.state.memory
    session = memory.session
    templates = request.app.state.templates

    sql = text("""
        SELECT 
            e.id AS evaluation_id,
            e.created_at,
            e.query_type,
            e.query_id,
            e.target_type,
            e.target_id,
            s.dimension,
            s.score,
            s.weight,
            e.scores ->> 'rank_score' AS rank_score
        FROM evaluations e
        JOIN scores s ON e.id = s.evaluation_id
        WHERE e.query_id = :query_id
        ORDER BY created_at DESC, (e.scores ->> 'rank_score')::float DESC
    """)

    rows = session.execute(sql, {"query_id": query_id}).fetchall()
    results = [dict(row._mapping) for row in rows]

    return templates.TemplateResponse(
        "ranking_detail.html",
        {
            "request": request,
            "query_id": query_id,
            "results": results,
        },
    )
