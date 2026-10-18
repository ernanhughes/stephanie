# stephanie/orm/base.py
from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import declarative_base, sessionmaker


def database_url_from_config(config) -> URL:
    """Build a SQLAlchemy URL from Hydra's db node without string interpolation.

    An explicit ``url``/``dsn`` is supported for deployments that manage a DSN
    as a secret. Otherwise the structured ``name``/``user``/``password``/
    ``host``/``port`` fields are authoritative.
    """
    if isinstance(config, str):
        return make_url(config)

    config = config or {}
    explicit_url = config.get("url") or config.get("dsn")
    if explicit_url:
        return make_url(str(explicit_url))

    database = config.get("name") or config.get("database")
    if not database:
        raise ValueError("Hydra database config must set 'name' or 'database'")

    port = config.get("port")
    return URL.create(
        drivername=str(config.get("driver", "postgresql+psycopg2")),
        username=config.get("user"),
        password=config.get("password"),
        host=config.get("host"),
        port=int(port) if port not in (None, "") else None,
        database=str(database),
        query=config.get("query", {}) or {},
    )


def database_connection_parameters(config) -> dict:
    """Return driver arguments for the existing direct PostgreSQL users."""
    url = database_url_from_config(config)
    if url.get_backend_name() != "postgresql":
        raise ValueError("MemoryTool's pgvector connection requires PostgreSQL")

    params = url.translate_connect_args(
        username="user", password="password", database="dbname"
    )
    params.update(dict(url.query))
    return params


def create_database_engine(config, **engine_overrides):
    """Create an engine using the same configured identity as MemoryTool."""
    url = database_url_from_config(config)
    config = config or {}
    if url.get_backend_name() == "postgresql":
        options = {
            "pool_pre_ping": True,
            "pool_recycle": 900,
            "pool_size": 10,
            "max_overflow": 20,
            "pool_timeout": 30,
            "connect_args": {
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 3,
            },
        }
    else:
        options = {"pool_pre_ping": True}

    for key in ("pool_pre_ping", "pool_recycle", "pool_size", "max_overflow", "pool_timeout"):
        if key in config and key in options:
            options[key] = config[key]
    if "connect_args" in config:
        options.setdefault("connect_args", {}).update(config.get("connect_args") or {})
    options.update(engine_overrides)
    return create_engine(url, **options)


def database_config_from_hydra(
    config_name: str = "config",
    *,
    config_dir: str | None = None,
    overrides: list[str] | None = None,
):
    """Compose a Hydra config and return its authoritative ``db`` node."""
    from hydra import compose, initialize_config_dir

    root_config_dir = Path(config_dir) if config_dir else Path(__file__).resolve().parents[2] / "config"
    with initialize_config_dir(
        version_base=None, config_dir=str(root_config_dir.resolve())
    ):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    return cfg.get("db", {})


def create_database_engine_from_hydra(
    config_name: str = "config",
    *,
    config_dir: str | None = None,
    overrides: list[str] | None = None,
):
    """Compose a Hydra config and create an engine from its ``db`` node.

    Intended for standalone repository scripts. Runtime services should pass
    their already-composed ``cfg.db`` directly to ``create_database_engine``.
    """
    db_config = database_config_from_hydra(
        config_name, config_dir=config_dir, overrides=overrides
    )
    return create_database_engine(db_config)


# Compatibility aliases for older callers. No pool is created at module import;
# legacy callers must explicitly initialize these from the composed Hydra db node.
engine = None
SessionLocal = sessionmaker(
    bind=engine,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


def configure_legacy_orm(config):
    """Bind legacy ``engine``/``SessionLocal`` aliases to explicit Hydra config."""
    global engine
    url = database_url_from_config(config)
    if engine is not None:
        if engine.url != url:
            raise RuntimeError(
                "Legacy ORM is already configured for a different database; "
                "dispose it before selecting another target"
            )
        return engine

    engine = create_database_engine(config)
    SessionLocal.configure(bind=engine)
    return engine


def dispose_legacy_orm() -> None:
    """Release an explicitly configured legacy ORM engine and clear its bind."""
    global engine
    if engine is not None:
        engine.dispose()
        engine = None
    SessionLocal.configure(bind=None)


Base = declarative_base()
WorldviewBase = declarative_base()
