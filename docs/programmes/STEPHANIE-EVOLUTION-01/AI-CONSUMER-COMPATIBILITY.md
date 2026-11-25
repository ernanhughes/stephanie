# AI consumer compatibility

Stephanie consumes the existing AI inference public API only; `C:/Projects/ai` source and all other consumer repositories remain unchanged. Compatibility evidence: local editable AI package in Python 3.13.15, one real local-provider call through `InferenceService.generate_structured`, schema validation, explicit no-fallback budget, result trace/fingerprints, and a retained execution artifact.

The AI package is not yet sourced through a reproducible published distribution in this workspace; editable installation is a development integration. No claim is made about runtime deployment compatibility of Worker, Praxis, Mission, Writer or Workbench.
