from dependency_injector import containers, providers
from sqlalchemy.orm import Session

from stephanie.memory.pipeline_stage_store import PipelineStageStore
from stephanie.protocols.embedding.hnet import HNetEmbeddingProtocol
from stephanie.protocols.embedding.standard import StandardEmbedderProtocol


class AppContainer(containers.DeclarativeContainer):
    # Configuration
    config = providers.Configuration()

    hnet_embedder = providers.Singleton(HNetEmbeddingProtocol)
    standard_embedder = providers.Singleton(StandardEmbedderProtocol)

    embedder_selector = providers.Selector(
        lambda ctx: ctx.get("embedder", "standard"),
        standard=standard_embedder,
        hnet=hnet_embedder
    )

    # Database session
    session_factory = providers.Factory(Session)

    # Services
    pipeline_stage_store = providers.Singleton(
        PipelineStageStore,
        session=session_factory
    )

    # Protocols
