# stephanie/orm/base.py
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Replace with your actual DB 
DB_URL = "postgresql://co:co@localhost:5432/co"

engine = create_engine(
    DB_URL,
    pool_pre_ping=True,           # ping before using a pooled conn
    pool_recycle=900,             # recycle conns every 15 min (tune as needed)
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    connect_args={                # OS-level keepalives to defeat idle NATs
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 3,
        # Optional: if you run behind pgbouncer in transaction pooling:
        # "options": "-c prepared_statements=0"
    },
)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, autocommit=False, autoflush=False)
Base = declarative_base()
WorldviewBase = declarative_base()
