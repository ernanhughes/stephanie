# File: stephanie/pipeline/registry.py

import copy
from pathlib import Path
from typing import Union

import yaml


class PipelineRegistry:
    """
    A centralized registry for managing pipeline configurations.

    Features:
    - Load pipelines from YAML files
    - Retrieve named or default pipelines
    - List available variants with or without descriptions
    - Validate pipeline structure
    """

    def __init__(self, registry_path: Union[str, Path] = "config/registry/pipelines.yaml"):
        self.registry_path = Path(registry_path)
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Pipeline registry file not found: {registry_path}")

        with open(self.registry_path, "r") as f:
            self.raw_data = yaml.safe_load(f)

        self._validate_registry_structure()

        self.pipeline_variants = self.raw_data.get("pipeline_variants", {})
        self.default_pipeline = self.raw_data.get("default", "minimal")

    def _validate_registry_structure(self):
        """Ensure basic registry structure is valid"""
        if "pipeline_variants" not in self.raw_data:
            raise ValueError("Missing required 'pipeline_variants' key in registry")
        if not isinstance(self.raw_data["pipeline_variants"], dict):
            raise TypeError("'pipeline_variants' must be a dictionary")

    def get_pipeline(self, name: str = None) -> list[dict]:
        """
        Get the stages list of a specific or default pipeline variant.

        Args:
            name (str): Name of the pipeline variant

        Returns:
            list[dict]: The list of agent stages
        """
        name = name or self.default_pipeline
        variant_entry = self.pipeline_variants.get(name)

        if not variant_entry:
            raise KeyError(f"Pipeline variant '{name}' not found in registry")

        # If full variant with description and stages
        if isinstance(variant_entry, dict) and "stages" in variant_entry:
            return variant_entry["stages"]
        # Backward compatibility: variant is just the stage list
        elif isinstance(variant_entry, list):
            return variant_entry
        else:
            raise TypeError(f"Invalid format for pipeline variant '{name}'")

    def get_description(self, name: str) -> str:
        """
        Return the description for a pipeline variant, if available.

        Args:
            name (str): Pipeline variant name

        Returns:
            str: Description or empty string
        """
        entry = self.pipeline_variants.get(name, {})
        if isinstance(entry, dict):
            return entry.get("description", "")
        return ""

    def get_default_pipeline(self) -> list[dict]:
        return self.get_pipeline(self.default_pipeline)

    def list_variants(self) -> list[str]:
        return list(self.pipeline_variants.keys())

    def list_variants_with_descriptions(self) -> list[dict]:
        """
        Return list of pipeline variants with name and description.

        Returns:
            list[dict]: List of {name, description}
        """
        return [
            {
                "name": name,
                "description": self.get_description(name)
            }
            for name in self.pipeline_variants.keys()
        ]

    def exists(self, name: str) -> bool:
        return name in self.pipeline_variants

    def get_metadata(self, name: str) -> dict:
        return self.raw_data.get("metadata", {}).get(name, {})

    def inject_into_config(self, config: dict, name: str = None, tag: str = "default") -> dict:
        """
        Inject a pipeline into a full config dict.
        Useful for dynamically creating configs during mutation/testing.
        """
        pipeline_def = self.get_pipeline(name)

        updated_config = copy.deepcopy(config)
        updated_config["pipeline"] = {
            "tag": tag,
            "stages": pipeline_def
        }

        updated_config.setdefault("agents", {})
        for stage in pipeline_def:
            agent_name = stage.get("name")
            if agent_name:
                updated_config["agents"][agent_name] = stage

        return updated_config

    def validate_pipeline(self, pipeline_def: list[dict]) -> bool:
        if not isinstance(pipeline_def, list):
            return False
        for stage in pipeline_def:
            if not isinstance(stage, dict) or "name" not in stage:
                return False
        return True
