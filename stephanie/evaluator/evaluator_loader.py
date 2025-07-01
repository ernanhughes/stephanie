# stephanie/evaluator/evaluator_loader.py

def get_evaluator(cfg, memory=None, call_llm=None, logger=None):
    if cfg["judge"] == "llm":
        from stephanie.evaluator.llm_judge_evaluator import LLMJudgeEvaluator

        llm = cfg.get("judge_model", cfg.get("model"))
        prompt_file = cfg.get("judge_prompt_file", "evaluator.txt")
        logger.log(
            "EvaluatorInit", {"strategy": "LLM", "prompt_file": prompt_file}
        )
        return LLMJudgeEvaluator(cfg, llm, prompt_file, call_llm, logger)
    elif cfg["judge"] == "mrq":
        from stephanie.evaluator.mrq_self_evaluator import MRQSelfEvaluator
        return MRQSelfEvaluator(memory=memory, logger=logger)
    else:
        raise ValueError(f"Unknown evaluator type: {cfg['type']}")
