import click
from co_ai.utils import extract_resources
import logging


@click.group()
def cli():
    """AI Co-Scientist CLI"""
    pass


@cli.command()
def init():
    """Initialize config and prompt files from embedded resources"""
    try:
        extract_resources()
        click.echo("[+] Successfully extracted configs and prompts")
    except Exception as e:
        click.echo(f"[-] Failed to extract resources: {e}")
        raise


if __name__ == "__main__":
    cli()