from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy import text
from io import StringIO
import csv
import logging

from stephanie.utils.db_scope import session_scope

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/db-overview", response_class=HTMLResponse)
def db_overview(request: Request):
    memory = request.app.state.memory
    templates = request.app.state.templates
    sessionmaker = memory.session  # now a sessionmaker, not a live session

    table_data = []
    with session_scope(sessionmaker) as session:
        tables = session.execute(text("""
            SELECT tablename 
            FROM pg_catalog.pg_tables 
            WHERE schemaname='public'
        """)).fetchall()

        for (table_name,) in tables:
            try:
                count = session.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
            except Exception:
                count = 0
            table_data.append({"name": table_name, "count": count})

    table_data.sort(key=lambda x: x["count"], reverse=True)

    return templates.TemplateResponse(
        "db_overview.html",
        {"request": request, "tables": table_data, "active_page": "db"}
    )


@router.get("/db/table/{table_name}", response_class=HTMLResponse)
def table_detail(request: Request, table_name: str):
    memory = request.app.state.memory
    templates = request.app.state.templates
    sessionmaker = memory.session

    try:
        with session_scope(sessionmaker) as session:
            result = session.execute(
                text(f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT 100")
            )
            rows = result.fetchall()
            columns = result.keys() if rows else []

    except Exception as e:
        logger.error(f"Failed to load table {table_name}: {e}")
        return HTMLResponse(
            f"<h2>Error loading table '{table_name}': {e}</h2>", status_code=500
        )

    return templates.TemplateResponse(
        "table_detail.html",
        {
            "request": request,
            "table_name": table_name,
            "columns": columns,
            "rows": rows,
        },
    )


@router.get("/db/table/{table_name}/csv")
def table_detail_csv(request: Request, table_name: str):
    memory = request.app.state.memory
    sessionmaker = memory.session

    try:
        with session_scope(sessionmaker) as session:
            result = session.execute(
                text(f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT 100")
            )
            rows = result.mappings().all()
            columns = rows[0].keys() if rows else []

            buffer = StringIO()
            writer = csv.writer(buffer)
            writer.writerow(columns)
            for row in rows:
                writer.writerow([row[col] for col in columns])

            buffer.seek(0)
            filename = f"{table_name}_last100.csv"

            return StreamingResponse(
                buffer,
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={filename}"},
            )
    except Exception as e:
        return HTMLResponse(
            f"<h2>Error exporting table '{table_name}': {e}</h2>", status_code=500
        )
