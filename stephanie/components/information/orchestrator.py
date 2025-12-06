# stephanie/components/information/information_orchestrator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from stephanie.components.information.adapters.info_casebook_adapter import \
    InfoCaseBookAdapter
from stephanie.components.information.adapters.info_scorable_adapter import \
    score_memcube_and_attach_attributes
from stephanie.components.information.builder.bucket import BucketBuilder
from stephanie.components.information.builder.memcube import MemCubeBuilder
from stephanie.components.information.data import (InformationBuildResult,
                                                   SourceProfile)


@dataclass
class InformationOrchestratorConfig:
    default_source_profile: str = "research_web_mixed"
    target: str = "memcube_blog_post"


class InformationOrchestrator:
    """
    Core Information Builder component.

    Given a topic, it:

      1. Builds an information Bucket (tools → nodes/edges)
      2. Builds a CaseBook with Cases (sections)
      3. Builds an Information MemCube (blog-postable page)
      4. Runs ScorableProcessor to attach quality metrics
      5. Returns IDs + preview markdown + scores

    This is designed to become the *central information source* for Stephanie.
    """

    def __init__(
        self,
        cfg: InformationOrchestratorConfig,
        bucket_builder: BucketBuilder,
        casebook_adapter: InfoCaseBookAdapter,
        memcube_builder: MemCubeBuilder,
        scorable_processor,
        memory,
        container,
        logger,
    ) -> None:
        self.cfg = cfg
        self.bucket_builder = bucket_builder
        self.casebook_adapter = casebook_adapter
        self.memcube_builder = memcube_builder
        self.scorable_processor = scorable_processor
        self.memory = memory
        self.container = container
        self.logger = logger

    async def run(
        self,
        topic: str,
        target: Optional[str] = None,
        source_profile: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> InformationBuildResult:
        target = target or self.cfg.target
        profile_name = source_profile or self.cfg.default_source_profile
        profile = SourceProfile.for_name(profile_name)
        options = options or {}

        # 1) Build Bucket
        bucket = await self.bucket_builder.build_bucket(topic, profile)

        # 2) Bucket → CaseBook + Cases
        cb_artifacts = await self.casebook_adapter.build_casebook_from_bucket(
            topic, bucket
        )

        # 3) CaseBook + Cases → Information MemCube
        cube = await self.memcube_builder.build_information_memcube(
            topic=topic,
            target=target,
            casebook_id=cb_artifacts.casebook_id,
            case_ids=cb_artifacts.case_ids,
            source_profile=profile.name,
        )

        # 4) Run ScorableProcessor and attach scores as attributes (extra_data["scores"])
        scores = await score_memcube_and_attach_attributes(
            cube,
            scorable_processor=self.scorable_processor,
            memory=self.memory,
            container=self.container,
            logger=self.logger,
        )

        # Persist updated extra_data
        self.memcube_builder.memcube_store.upsert(
            {
                "id": cube.id,
                "scorable_id": cube.scorable_id,
                "scorable_type": cube.scorable_type,
                "dimension": cube.dimension,
                "version": cube.version,
                "content": cube.content,
                "refined_content": cube.refined_content,
                "original_score": cube.original_score,
                "refined_score": cube.refined_score,
                "source": cube.source,
                "model": cube.model,
                "priority": cube.priority,
                "sensitivity": cube.sensitivity,
                "ttl": cube.ttl,
                "usage_count": cube.usage_count,
                "extra_data": cube.extra_data,
            },
            merge_extra=True,
        )

        self.logger.log(
            "InformationBuildCompleted",
            {
                "topic": topic,
                "target": target,
                "memcube_id": cube.id,
                "casebook_id": cb_artifacts.casebook_id,
            },
        )

        return InformationBuildResult(
            topic=topic,
            target=target,
            memcube_id=cube.id,
            casebook_id=cb_artifacts.casebook_id,
            attributes={
                **scores,
                "source_profile": profile.name,
            },
            preview_markdown=cube.content,
            extra_data=cube.extra_data or {},
        )
