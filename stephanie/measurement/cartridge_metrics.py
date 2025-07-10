# stephanie/measurement/cartridge_metrics.py
from stephanie.measurement.decorators import measure
from stephanie.models.cartridge import CartridgeORM


@measure("Cartridge", "domain_tags")
def measure_domains(cartridge: CartridgeORM, session, context=None):
    """Extract domain tags from cartridge content"""
    # Use DomainClassifier from your system
    return {"tags": context["classifier"].classify(cartridge.markdown_content)}

@measure("Cartridge", "triple_count")
def measure_triples(cartridge: CartridgeORM, session, context=None):
    """Count semantic triples in cartridge"""
    return {"count": len(cartridge.triples)}

@measure("Cartridge", "section_count")
def measure_sections(cartridge: CartridgeORM, session, context=None):
    """Count content sections"""
    return {"count": len(cartridge.sections)}