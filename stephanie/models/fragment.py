import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg

from stephanie.models.base import Base


class FragmentORM(Base):
    __tablename__ = "fragments"

    id = sa.Column(sa.BigInteger, primary_key=True)
    case_id = sa.Column(sa.BigInteger, index=True, nullable=False)
    source_type = sa.Column(sa.String(32), nullable=False)  # paper|chat|web|code
    section = sa.Column(sa.String(128))
    text = sa.Column(sa.Text, nullable=False)

    # Arbitrary metadata (e.g., page_no, url, model_name, tokens, etc.)
    attrs = sa.Column(pg.JSONB, server_default=sa.text("'{}'::jsonb"), nullable=False)

    # Per-dimension scores (e.g., {"mrq_correct": 0.82, "hrm_epistemic": 0.71, ...})
    scores = sa.Column(pg.JSONB, server_default=sa.text("'{}'::jsonb"), nullable=False)

    # Uncertainty proxy (e.g., normalized EBT energy or scorer disagreement)
    uncertainty = sa.Column(sa.Float)

    created_at = sa.Column(sa.DateTime, server_default=sa.func.now(), nullable=False)
