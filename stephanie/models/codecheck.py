# stephanie/models/codecheck.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from stephanie.models.base import Base


JSONType = JSONB if JSONB else JSON


# --------------------------------------------------------------------------- #
# Run: a single analysis of a repo (or subpath)
# --------------------------------------------------------------------------- #

class CodeCheckRunORM(Base):
    __tablename__ = "codecheck_run"

    id = Column(String, primary_key=True)
    created_ts = Column(DateTime, default=datetime.utcnow, index=True)
    finished_ts = Column(DateTime, nullable=True, index=True)

    status = Column(String, nullable=False, default="pending")
    status_message = Column(Text, nullable=True)

    repo_root = Column(Text, nullable=False)
    rel_root = Column(Text, nullable=True)
    branch = Column(String, nullable=True, index=True)
    commit_hash = Column(String, nullable=True, index=True)
    language = Column(String, nullable=True, index=True)

    config = Column(JSONType, nullable=True)
    summary_metrics = Column(JSONType, nullable=True)

    # Relationships
    files = relationship(
        "CodeCheckFileORM",
        back_populates="run",
        cascade="all, delete-orphan",
    )
    suggestions = relationship(
        "CodeCheckSuggestionORM",
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def to_dict(self, include_files: bool = False) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "created_ts": self.created_ts.isoformat() if self.created_ts else None,
            "finished_ts": self.finished_ts.isoformat() if self.finished_ts else None,
            "status": self.status,
            "status_message": self.status_message,
            "repo_root": self.repo_root,
            "rel_root": self.rel_root,
            "branch": self.branch,
            "commit_hash": self.commit_hash,
            "language": self.language,
            "config": self.config or {},
            "summary_metrics": self.summary_metrics or {},
        }
        if include_files:
            d["files"] = [f.to_dict(include_metrics=True, include_issues=False) for f in self.files]
        return d


# --------------------------------------------------------------------------- #
# File: one code file as seen in a specific run
# --------------------------------------------------------------------------- #

class CodeCheckFileORM(Base):
    __tablename__ = "codecheck_file"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("codecheck_run.id", ondelete="CASCADE"), index=True)
    path = Column(Text, nullable=False, index=True)
    language = Column(String, nullable=True, index=True)
    module_name = Column(String, nullable=True, index=True)

    loc = Column(Integer, nullable=True)
    num_classes = Column(Integer, nullable=True)
    num_functions = Column(Integer, nullable=True)
    num_methods = Column(Integer, nullable=True)
    num_inner_functions = Column(Integer, nullable=True)

    is_test = Column(Boolean, nullable=False, default=False, index=True)
    content_hash = Column(String, nullable=True, index=True)

    labels = Column(JSONType, nullable=True)
    meta = Column(JSONType, nullable=True)

    run = relationship("CodeCheckRunORM", back_populates="files")
    metrics = relationship(
        "CodeCheckFileMetricsORM",
        back_populates="file",
        uselist=False,
        cascade="all, delete-orphan",
    )
    issues = relationship(
        "CodeCheckIssueORM",
        back_populates="file",
        cascade="all, delete-orphan",
    )
    suggestions = relationship(
        "CodeCheckSuggestionORM",
        back_populates="file",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("idx_codecheck_file_run_path", "run_id", "path"),
    )

    def to_dict(
        self,
        include_metrics: bool = True,
        include_issues: bool = False,
    ) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "run_id": self.run_id,
            "path": self.path,
            "language": self.language,
            "module_name": self.module_name,
            "loc": self.loc,
            "num_classes": self.num_classes,
            "num_functions": self.num_functions,
            "num_methods": self.num_methods,
            "num_inner_functions": self.num_inner_functions,
            "is_test": bool(self.is_test),
            "content_hash": self.content_hash,
            "labels": self.labels or [],
            "meta": self.meta or {},
        }
        if include_metrics:
            d["metrics"] = self.metrics.to_dict() if self.metrics else None
        if include_issues:
            d["issues"] = [i.to_dict() for i in self.issues]
        return d


# --------------------------------------------------------------------------- #
# Metrics: per-file metric vector (AI-smell, static tools, etc.)
# Patterned after NexusMetricsORM (columns/values/vector). :contentReference[oaicite:2]{index=2}
# --------------------------------------------------------------------------- #

class CodeCheckFileMetricsORM(Base):
    __tablename__ = "codecheck_file_metrics"

    file_id = Column(Integer, ForeignKey("codecheck_file.id", ondelete="CASCADE"), primary_key=True)
    columns = Column(JSONType, nullable=False, default=list)  # ["ai_smell.score", "lint.bandit.high", ...]
    values = Column(JSONType, nullable=False, default=list)   # aligned List[float]
    vector = Column(JSONType, nullable=True)                  # convenience: {name: value}

    file = relationship("CodeCheckFileORM", back_populates="metrics")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_id": self.file_id,
            "columns": self.columns or [],
            "values": self.values or [],
            "vector": self.vector or {},
        }


# --------------------------------------------------------------------------- #
# Issue: one issue per tool finding / smell / rule violation
# --------------------------------------------------------------------------- #

class CodeCheckIssueORM(Base):
    __tablename__ = "codecheck_issue"

    id = Column(Integer, primary_key=True, autoincrement=True)

    run_id = Column(String, ForeignKey("codecheck_run.id", ondelete="CASCADE"), index=True)
    file_id = Column(Integer, ForeignKey("codecheck_file.id", ondelete="CASCADE"), index=True)

    # Where in the file
    line = Column(Integer, nullable=True, index=True)
    col = Column(Integer, nullable=True)

    # Classification
    source = Column(String, nullable=False, index=True)   # "bandit", "ruff", "mypy", "code_flavor", ...
    code = Column(String, nullable=True, index=True)      # tool-specific code, e.g. "B101", "E501"
    type = Column(String, nullable=True, index=True)      # "security", "style", "ai_smell", "architecture", ...
    severity = Column(String, nullable=True, index=True)  # "info", "low", "medium", "high", "critical"

    message = Column(Text, nullable=False)
    meta = Column(JSONType, nullable=True)                # raw tool payload, extra details

    file = relationship("CodeCheckFileORM", back_populates="issues")

    __table_args__ = (
        Index("idx_codecheck_issue_run_severity", "run_id", "severity"),
        Index("idx_codecheck_issue_run_source", "run_id", "source"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "file_id": self.file_id,
            "line": self.line,
            "col": self.col,
            "source": self.source,
            "code": self.code,
            "type": self.type,
            "severity": self.severity,
            "message": self.message,
            "meta": self.meta or {},
        }


class CodeCheckSuggestionORM(Base):
    """
    A concrete, human-sized improvement suggestion for a file.

    Examples of `kind`:
      - "style", "refactor", "test", "safety", "naming"
    `patch_type`:
      - "full_file" (new content replaces file)
      - "unified_diff" (git-style patch, apply separately)
    """

    __tablename__ = "codecheck_suggestion"

    id = Column(Integer, primary_key=True)

    run_id = Column(String, ForeignKey("codecheck_run.id", ondelete="CASCADE"), nullable=False)
    file_id = Column(Integer, ForeignKey("codecheck_file.id", ondelete="CASCADE"), nullable=False)

    kind = Column(String, nullable=False)
    title = Column(String, nullable=False)
    summary = Column(Text, nullable=False)
    detail = Column(Text, nullable=True)

    patch = Column(Text, nullable=True)         # optional patch or new content
    patch_type = Column(String, nullable=True)  # "full_file" | "unified_diff" | None

    status = Column(
        String,
        nullable=False,
        default="pending",  # pending | applied | dismissed
    )

    created_ts = Column(DateTime(timezone=True), server_default=func.now())
    applied_ts = Column(DateTime(timezone=True), nullable=True)

    meta = Column(JSONType, nullable=True)

    run = relationship("CodeCheckRunORM", back_populates="suggestions")
    file = relationship("CodeCheckFileORM", back_populates="suggestions")
