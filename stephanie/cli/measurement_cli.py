# stephanie/cli/measurement_cli.py
from __future__ import annotations

import click

from stephanie.measurement.registry import measurement_registry
from stephanie.models.base import SessionLocal


@click.group()
def measurement():
    """Measurement management commands"""
    pass  # No code here, just the group definition


@measurement.command()
def list_metrics():
    """List all available metrics"""
    for key in measurement_registry._strategies.keys():
        click.echo(key)


# Keep your backfill command below
@measurement.command()
@click.option("--entity-type", required=True)
@click.option("--metric-name", required=True)
def backfill(entity_type, metric_name):
    """Backfill measurements for existing data"""
    db = SessionLocal()
    # Implement backfill logic
    click.echo(f"Backfilling {metric_name} for {entity_type}")


@measurement.command()
def list_metrics():
    """List all available metrics"""
    # Show registered metrics from registry
    pass


if __name__ == "__main__":
    measurement()
