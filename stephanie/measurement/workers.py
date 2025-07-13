# stephanie/measurement/workers.py
from celery import shared_task

from stephanie.models.base import SessionLocal
from stephanie.models.theorem import CartridgeORM


@shared_task
def compute_embedding_metrics(cartridge_id: int):
    """Background task for expensive metrics"""
    db = SessionLocal()
    try:
        cartridge = db.query(CartridgeORM).get(cartridge_id)
        # Compute embedding-based metrics here
        # Save via MeasurementORM
    finally:
        db.close()
