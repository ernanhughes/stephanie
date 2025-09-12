# stephanie/agents/paper_improver/__init__.py

# Expose core classes for agent import
from .bandit_router import ExemplarBandit
from .code_improver import CodeImprover
from .curriculum import CurriculumScheduler
from .faithfulness import FaithfulnessBot
from .goals import GoalScorer
from .mutation import MutationRunner
from .repo_link import RepoLink
from .text_improver import TextImprover
from .vpm_controller import VPMController

__all__ = [
    "CodeImprover",
    "TextImprover",
    "VPMController",
    "RepoLink",
    "ExemplarBandit",
    "FaithfulnessBot",
    "MutationRunner",
    "GoalScorer",
    "CurriculumScheduler"
]