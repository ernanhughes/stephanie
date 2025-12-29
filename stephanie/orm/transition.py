# stephanie/orm/transition.py
from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg

from stephanie.orm.base import Base


class TransitionORM(Base):
    __tablename__ = "agent_transitions"

    id = sa.Column(sa.BigInteger, primary_key=True)

    # Run-scoped ordering for credit assignment + trajectory viewing
    run_id = sa.Column(sa.String(64), index=True, nullable=False)
    step_idx = sa.Column(sa.Integer, nullable=False)

    # Which agent produced the action (e.g., "MCTSReasoningAgent")
    agent = sa.Column(sa.String(128))

    # Compact semantic state (goal preview, step_type, vars, node_id, etc.)
    state = sa.Column(pg.JSONB, server_default=sa.text("'{}'::jsonb"), nullable=False)

    # Action payload (type: "llm|tool|mcts|score", name, output summary, strategy tag)
    action = sa.Column(pg.JSONB, server_default=sa.text("'{}'::jsonb"), nullable=False)

    # Intermediate reward (AIR) and final credited reward
    reward_air = sa.Column(sa.Float)
    reward_final = sa.Column(sa.Float)

    # Full vector of scores for analysis (mrq_correct, hrm_epistemic, sicql_adv_norm, ebt_energy_inv, ...)
    rewards_vec = sa.Column(pg.JSONB, server_default=sa.text("'{}'::jsonb"), nullable=False)

    created_at = sa.Column(sa.DateTime, server_default=sa.func.now(), nullable=False)
