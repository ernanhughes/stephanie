# stephanie/scoring/metrics/row_builder.py
from __future__ import annotations

import time
import numpy as np
from stephanie.utils.hash_utils import hash_text
from stephanie.data.scorable_row import ScorableRow


class RowBuilder:
    """
    Builds ScorableRow objects from a Scorable + accumulator dictionary.

    This is the final glue step after all Features + Tools have run.
    """

    # ------------------------------------------------------------------
    def build(self, scorable, acc: dict) -> ScorableRow:
        text = scorable.text or ""
        meta = scorable.meta or {}

        # --------------------------------------------------------------
        # Core identity + title
        # --------------------------------------------------------------
        title = (
            acc.get("title")
            or meta.get("title")
            or text[:80]
            or f"{scorable.target_type}:{scorable.id}"
        )

        # --------------------------------------------------------------
        # Embeddings
        # --------------------------------------------------------------
        embeddings = dict(acc.get("embeddings") or {})
        embed_global = None

        gl = embeddings.get("global")
        if isinstance(gl, list) and gl:
            try:
                embed_global_np = np.asarray(gl, dtype=np.float32)
                embed_global = embed_global_np.tolist()
            except Exception:
                embed_global = gl  # fallback

        # --------------------------------------------------------------
        # Metrics
        # --------------------------------------------------------------
        metrics_vector = dict(acc.get("metrics_vector") or {})
        metrics_columns = list(acc.get("metrics_columns") or metrics_vector.keys())
        metrics_values = [float(metrics_vector.get(c, 0.0)) for c in metrics_columns]

        # --------------------------------------------------------------
        # VPM / Vision signals
        # --------------------------------------------------------------
        vision_signals = acc.get("vision_signals")
        vision_meta = acc.get("vision_signals_meta") or {}

        # --------------------------------------------------------------
        # VisiCalc features (optional)
        # --------------------------------------------------------------
        visicalc_report = acc.get("visicalc_report") or {}
        visicalc_features = acc.get("visicalc_features")
        visicalc_feature_names = acc.get("visicalc_feature_names")

        # --------------------------------------------------------------
        # Build final row
        # --------------------------------------------------------------
        return ScorableRow(
            scorable_id=str(scorable.id)
            or f"{scorable.target_type}:{hash_text(text)[:16]}",
            scorable_type=scorable.target_type,
            conversation_id=meta.get("conversation_id"),
            external_id=meta.get("external_id"),
            order_index=meta.get("order_index"),
            text=text,
            title=title,

            # Feature outputs
            near_identity=acc.get("near_identity") or meta.get("near_identity") or {},
            domains=acc.get("domains") or [],
            ner=acc.get("ner") or [],

            # Scores / human labels
            ai_score=meta.get("ai_score"),
            star=meta.get("star"),
            goal_ref=meta.get("goal_ref"),

            # Embedding info
            embeddings=embeddings,
            embed_global=embed_global,

            # Metrics (canonical vector)
            metrics_columns=metrics_columns,
            metrics_values=metrics_values,
            metrics_vector=metrics_vector,

            # Agreement / stability / meta
            agreement=meta.get("agreement"),
            stability=meta.get("stability"),
            chat_id=meta.get("chat_id"),
            turn_index=meta.get("turn_index"),
            parent_scorable_id=meta.get("parent_scorable_id"),
            parent_scorable_type=meta.get("parent_scorable_type"),
            order_in_parent=meta.get("order_in_parent"),

            # Vision / ZeroModel / VPM
            vision_signals=vision_signals,
            vision_signals_meta=vision_meta,

            # VisiCalc
            visicalc_report=visicalc_report,
            visicalc_features=visicalc_features,
            visicalc_feature_names=visicalc_feature_names,

            # Rollout / pipeline info
            rollout=meta.get("rollout") or {},
            processor_version="3.0",
            content_hash16=hash_text(text)[:16],
            created_utc=time.time(),
        )

    # ------------------------------------------------------------------
    def build_minimal(self, scorable) -> ScorableRow:
        """
        Minimal row used for async/offload mode before features finish.
        """
        text = scorable.text or ""
        meta = scorable.meta or {}

        title = (
            meta.get("title")
            or text[:80]
            or f"{scorable.target_type}:{scorable.id}"
        )

        return ScorableRow(
            scorable_id=str(scorable.id)
            or f"{scorable.target_type}:{hash_text(text)[:16]}",
            scorable_type=scorable.target_type,
            conversation_id=meta.get("conversation_id"),
            external_id=meta.get("external_id"),
            order_index=meta.get("order_index"),
            text=text,
            title=title,

            # Everything empty until features run
            near_identity={},
            domains=[],
            ner=[],
            ai_score=meta.get("ai_score"),
            star=meta.get("star"),
            goal_ref=meta.get("goal_ref"),

            embeddings={},
            embed_global=None,
            metrics_columns=[],
            metrics_values=[],
            metrics_vector={},

            agreement=meta.get("agreement"),
            stability=meta.get("stability"),
            chat_id=meta.get("chat_id"),
            turn_index=meta.get("turn_index"),
            parent_scorable_id=meta.get("parent_scorable_id"),
            parent_scorable_type=meta.get("parent_scorable_type"),
            order_in_parent=meta.get("order_in_parent"),

            vision_signals=None,
            vision_signals_meta={},

            visicalc_report={},
            visicalc_features=None,
            visicalc_feature_names=None,

            rollout=meta.get("rollout") or {},
            processor_version="3.0",
            content_hash16=hash_text(text)[:16],
            created_utc=time.time(),
        )
