# stephanie/measurement/listeners.py
from sqlalchemy import event

from stephanie.measurement.registry import measurement_registry
from stephanie.models.measurement import MeasurementORM
from stephanie.models.theorem import CartridgeORM


def after_insert_listener(mapper, connection, target):
    """Automatically generate measurements after entity creation"""
    session = connection.session
    strategies = measurement_registry.get_strategies_for_entity(target.__class__.__name__)
    
    for metric_name, func in strategies.items():
        try:
            result = func(target, session, context={"session": session})
            measurement = MeasurementORM(
                entity_type=target.__class__.__name__,
                entity_id=target.id,
                metric_name=metric_name,
                value=result,
                context={"source": "auto"}
            )
            session.add(measurement)
        except Exception as e:
            print(f"Error measuring {metric_name}: {str(e)}")
    
    session.commit()

# Register listener for Cartridge
event.listen(CartridgeORM, 'after_insert', after_insert_listener)