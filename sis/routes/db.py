from urllib import request
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import text
import logging
from io import StringIO
import csv 
from fastapi.responses import StreamingResponse

from sqlalchemy import text


router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/db-overview", response_class=HTMLResponse)
def db_overview(request: Request):
    memory = request.app.state.memory
    templates = request.app.state.templates
    session = memory.session

    # Get all tables in public schema
    tables = session.execute(text("""
        SELECT tablename 
        FROM pg_catalog.pg_tables 
        WHERE schemaname='public'
    """)).fetchall()

    table_data = []
    for (table_name,) in tables:
        try:
            count = session.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
        except Exception:
            count = "N/A"
        table_data.append({"name": table_name, "count": count})

    return templates.TemplateResponse(
        "db_overview.html",
        {"request": request, "tables": table_data,  "active_page": "db"}
    )


@router.get("/db/table/{table_name}", response_class=HTMLResponse)
def table_detail(request: Request, table_name: str):
    memory = request.app.state.memory
    templates = request.app.state.templates

    try:
        # Use SQLAlchemy session directly
        session = memory.session

        # Run raw SQL (g What eneric, works for any table)
        result = session.execute(
            text(f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT 100")
        )
        rows = result.fetchall()
        # Extract column names
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
    try:
        session = memory.session
        result = session.execute(
            text(f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT 100")
        )
        rows = result.mappings().all()
        columns = rows[0].keys() if rows else []

        # Write CSV
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
        return HTMLResponse(f"<h2>Error exporting table '{table_name}': {e}</h2>", status_code=500)
