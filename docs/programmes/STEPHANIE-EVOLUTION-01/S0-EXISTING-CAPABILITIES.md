# S0 — Existing capabilities and Python 3.13

Existing STN01–STN06, ChatStore, SIS, CaseBook, MemCube, `ExperimentStore`, execution experience and frozen STN06 mechanisms were inventoried. This programme does not modify STN06 or activate SIS experience learning defaults. The six AI foundation assessments and Python upgrade record remain in `docs/stephanie-next/`.

Python 3.13.15 is active in `venv`; the previous 3.11 venv remains at `C:/Projects/_venv-backups/stephanie-py311-20261125`. `pip check` passed. The focused Python 3.13 baseline was 21 passed. Final combined targeted regression passed 28 tests and deselected three long tests. The Python-hash-seed subprocess tests took about 20 seconds each and are now marked `long`; the isolated store test also carries the marker and passed separately.

The dependency `litellm>=1.111,<2` was added because existing `async_utils.py` imports it at runtime. Docker/devcontainer builds were not run. Isolated SQLite tests qualify relevant schemas; no operational database was accessed.
