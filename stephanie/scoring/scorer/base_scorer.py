# stephanie/scoring/scorer/base_scorer.py
from __future__ import annotations

import abc
import logging
from collections.abc import Mapping
from typing import Any, Dict, List, Protocol

import torch

from stephanie.data.score_bundle import ScoreBundle
from stephanie.scoring.model.model_locator_mixin import ModelLocatorMixin
from stephanie.scoring.scorable import Scorable

log = logging.getLogger(__name__)

class ScoringPlugin(Protocol):
    """Protocol for scoring plugins that can post-process model outputs."""
    def post_process(self, *, tap_output: Dict[str, Any]) -> Dict[str, float]: ...
    def close(self) -> None: ...  # optional


# ---------- BaseScorer -------------------------------------------------------
class BaseScorer(ModelLocatorMixin, abc.ABC):
    def __init__(self, cfg: dict, memory, container, logger, enable_plugins: bool = True):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger

        self.enable_plugins = enable_plugins

        self.embedding_type = self.memory.embedding.name
        self.dim = self.memory.embedding.dim
        self.hdim = self.memory.embedding.hdim

        self.model_path = cfg.get("model_path", "models")
        self.target_type = cfg.get("target_type", "document")
        self.model_type = cfg.get("model_type", "svm")  # Override in subclass
        self.version = cfg.get("model_version", "v1")

        self.force_rescore = cfg.get("force_rescore", False)
        self.dimensions = cfg.get("dimensions", [])
        self.device = torch.device(cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu")

        self._plugins: List[ScoringPlugin] = []
        if enable_plugins:
            try:
                self._plugins = self.build_plugins()
                log.debug("ScorerPluginsLoaded %s -> %s",
                              self.get_model_name(),
                              [type(p).__name__ for p in self._plugins])
            except Exception as e:
                log.error("ScorerPluginLoadError: %s", e)

    def score(
        self,
        goal: dict,
        scorable: Scorable,
        dimensions: List[str],
    ) -> ScoreBundle:
        """
        Final score entrypoint. Subclasses implement _score_core; we then
        apply plugins and merge their outputs into attributes/vectors.
        """
        bundle: ScoreBundle = self._score_core(goal, scorable, dimensions)
        result = self._run_plugins_and_merge(bundle=bundle, goal=goal, scorable=scorable)

        return result

    @abc.abstractmethod
    def _score_core(
        self,
        goal: dict,
        scorable: Scorable,
        dimensions: List[str],
    ) -> ScoreBundle:
        """Subclasses must implement the core scoring and return a ScoreBundle."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return f"{self.model_type}"

    def get_model_name(self) -> str:
        return f"{self.target_type}_{self.model_type}_{self.version}"

    def add_plugin(self, plugin: ScoringPlugin) -> None:
        self._plugins.append(plugin)

    def log_event(self, event: str, data: dict):
        self.logger.log(event, data)

    def range_sentinel(self, tag: str, val: float, eps: float = 1e-6):
        if not (0.0 - eps <= val <= 1.0 + eps):
            log.error("ScoreRangeViolation tag=%s value=%s", tag, float(val))

    def get_display_name(self) -> str:
        return self.cfg.get("display_name", self.name)

    # ===== Plugin runner and merge =====
    def _run_plugins_and_merge(self, *, bundle: ScoreBundle, goal: dict, scorable: Scorable) -> ScoreBundle:
        tap = {
            "goal_text": goal.get("goal", {}).get("goal_text", ""),
            "resp_text": scorable.text or "",
            "context": goal,
            "model_alias": getattr(self, "model_alias", self.model_type),
            "attributes": {},
            "per_dim_scores": {},
            "host": self,
        }

        merged: Dict[str, float] = {}
        for p in self._plugins:
            try:
                ext = p.post_process(tap_output=tap) or {}
                # last plugin wins on key collisions
                for k, v in ext.items():
                    try:
                        merged[str(k)] = float(v)
                    except Exception:
                        continue
            except Exception as e:
                if self.logger:
                    self.logger.log("PluginError", {"plugin": type(p).__name__, "error": str(e)})

        if not merged:
            return bundle

        # Merge into each ScoreResult.attributes (+ vector columns/values)
        for dim, sr in bundle.results.items():
            attrs = dict(sr.attributes or {})
            attrs.update(merged)

            vec = attrs.get("vector", {})
            if isinstance(vec, dict):
                for k, v in merged.items():
                    if isinstance(v, (int, float)):
                        vec[k] = float(v)
                attrs["vector"] = vec
                cols = list(vec.keys())
                attrs["columns"] = cols
                attrs["values"] = [vec[c] for c in cols]

            scm_key = f"scm.{dim}.score01"
            try:
                if scm_key in attrs:
                    sr.score = float(attrs[scm_key])
            except Exception:
                pass

            sr.attributes = attrs

        return bundle

    def close(self):
        for p in self._plugins:
            if hasattr(p, "close"):
                try: p.close()
                except Exception: pass
        # subclasses (e.g., HF) can extend to unload models

    def build_plugins(self) -> List[ScoringPlugin]:
        """
        Minimal plugin loader (container-first):
        plugins:
        - <plugin_key>:
            service_name: <optional override; defaults to plugin_key>
            # ... arbitrary plugin options (ignored here) ...

        We ONLY resolve services from the container. Each service must implement:
        post_process(tap_output: Dict[str, Any]) -> Dict[str, float]
        """
        plugins: List[ScoringPlugin] = []

        # 1) Pull the raw 'plugins' value from cfg, tolerate anything.
        raw_plugins = None
        try:
            # dict-like .get (OmegaConf DictConfig or plain dict)
            raw_plugins = self.cfg.get("plugins", None)
        except Exception:
            # Some cfgs may not be Mapping-like; fall through
            raw_plugins = None

        # 2) Normalize to a plain Python structure if it's OmegaConf.
        if raw_plugins is None:
            return plugins  # nothing to load

        try:
            from omegaconf import OmegaConf
            if OmegaConf.is_config(raw_plugins):
                plugins_cfg = OmegaConf.to_container(raw_plugins, resolve=True)
            else:
                plugins_cfg = raw_plugins
        except Exception:
            # OmegaConf not available or not a config object; assume already plain
            plugins_cfg = raw_plugins

        # 3) Validate top-level shape.
        if not isinstance(plugins_cfg, list):
            log.error("ScoringPluginConfigError: 'plugins' must be a list, got %r", type(plugins_cfg))
            return plugins

        # Helper: enforce single-key mapping entries.
        def _extract_single_mapping(item) -> tuple[str, dict]:
            if not isinstance(item, Mapping) or len(item) != 1:
                log.error(
                    "ScoringPluginConfigError: each plugins[] entry must be a single-key mapping, got: %r", item
                )
                return "", {}
            (name, opts), = item.items()
            name = (str(name).strip() if name is not None else "")
            if not name:
                log.error("ScoringPluginConfigError: empty plugin key in entry: %r", item)
                return "", {}
            # Coerce opts to dict
            if opts is None:
                opts = {}
            elif not isinstance(opts, Mapping):
                log.warning("ScoringPluginConfigWarning: opts for '%s' should be a mapping; coercing.", name)
                opts = {"value": opts}
            else:
                opts = dict(opts)
            return name, opts

        # 4) Resolve each plugin from the container.
        for spec in plugins_cfg:
            name, opts = _extract_single_mapping(spec)
            if not name:
                continue

            service_name = str(opts.get("service_name") or name).strip()
            try:
                svc = self.container.get(service_name)
            except Exception as e:
                log.error("ScoringPluginContainerError: service '%s' not found (%s)", service_name, e)
                continue

            # Must implement post_process(tap_output=...)
            post = getattr(svc, "post_process", None)
            if not callable(post):
                log.error(
                    "ScoringPluginTypeError: service '%s' does not implement post_process(tap_output=...)", service_name
                )
                continue

            # Hand the scorer to the service (so it can use model/tok/max_seq_len)
            try:
                if hasattr(svc, "set_host") and callable(getattr(svc, "set_host")):
                    svc.set_host(self)
                else:
                    setattr(svc, "host", self)
            except Exception as e:
                logning("PluginHostAttachWarning: %s (svc=%s)", e, service_name)

            # Stash plugin options on the service
            try:
                setattr(svc, "_plugin_opts", dict(opts or {}))
            except Exception:
                pass

            plugins.append(svc)
            log.debug("ScoringPluginLoaded(service): %s -> %s", service_name, type(svc).__name__)

        return plugins
