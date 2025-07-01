from stephanie.utils.loaders import get_memory, get_logger, get_config
from stephanie.scoring.batch import score_unscored_hypotheses

if __name__ == "__main__":
    memory = get_memory()
    logger = get_logger()
    cfg = get_config("agents/pipeline_judge")
    
    import asyncio
    asyncio.run(score_unscored_hypotheses(memory, logger, cfg))
