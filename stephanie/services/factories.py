# stephanie/services/factories.py
from __future__ import annotations

from stephanie.services.registry_loader import _load_object
from stephanie.services.reporting_service import ReportingService
from stephanie.services.self_validation_service import SelfValidationService
from stephanie.services.training_service import TrainingService


def make_reporting_service(sinks, enabled: bool, sample_rate: float):
    built = []
    for s in sinks:
        cls = _load_object(s["cls"])
        args = s.get("args", {}) or {}
        built.append(cls(**args))
    return ReportingService(sinks=built, enabled=enabled, sample_rate=sample_rate)

def create_self_validator(cfg, memory, logger, reward_model, llm_judge):
    return SelfValidationService(cfg=cfg, memory=memory, logger=logger,
                                 reward_model=reward_model, llm_judge=llm_judge)

def create_training_controller(cfg, memory, logger, validator, tracker, trainer_fn):
    return TrainingService(cfg=cfg, memory=memory, logger=logger,
                           validator=validator, tracker=tracker, trainer_fn=trainer_fn)
