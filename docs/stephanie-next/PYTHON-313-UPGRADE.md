# Stephanie Python 3.13 Upgrade

## Changes

- Stephanie package support is now Python `>=3.13` in `pyproject.toml` and
  `setup.py`; classifiers advertise Python 3.13.
- The production Docker image uses `python:3.13-slim`.
- The devcontainer uses the official Microsoft devcontainer Python
  `3.13-bookworm` image.
- The PyPI publish workflow builds with Python 3.13.
- The installation guide requires Python 3.13+.
- The new standard local `venv` uses Python 3.13.15.
- The prior 2.1 GB Python 3.11 environment was preserved intact at
  `C:\Projects\_venv-backups\stephanie-py311-20261125`.

## Dependency resolution

The existing uncommitted dependency updates had an unsatisfiable pair:
`hydra-core>=1.3.8` resolves to 1.3.8, which requires `omegaconf<2.4`, while
the package and `requirements.txt` required `omegaconf>=2.4`. The OmegaConf
constraint is now `>=2.3,<2.4`. The rest of those pre-existing dependency
updates were retained.

`zeromodel-core` and `zeromodel-analysis` are local packages unavailable from
the configured package index. The environment installation used their existing
source at `C:\Projects\zeromodel\packages\core` and `...\analysis`; neither
repository was modified. These local editable package references were not
added to Stephanie's published dependency declarations.

The AI package was installed editable from `C:\Projects\ai` in the local
Stephanie environment for shared-foundation qualification. Stephanie's project
metadata still needs a durable AI distribution source before a published
install can declare this package reproducibly.

## Verification

- `venv\Scripts\python.exe --version` → Python 3.13.15.
- `venv\Scripts\python.exe -m pip check` → no broken requirements.
- Imports resolved for `stephanie`, `sis`, `ai` and `ai.inference`.
- Hydra 1.3.8, OmegaConf 2.3.1 and CPU Torch 2.15.0 imported successfully.
- No model/provider call was made and no test suite was run for this
  environment-only and package-metadata upgrade.

## Limitation

The Dockerfiles and CI metadata were updated but no Docker image build or PyPI
publish was run. The local editable AI and ZeroModel installs demonstrate this
workspace setup; they are not portable dependency lockfiles.
