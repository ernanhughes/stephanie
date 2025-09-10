# stephanie/agents/paper_improver/__init__.py

# Expose core classes for agent import
from .code_improver import CodeImprover
from .text_improver import TextImprover
from .vpm_controller import VPMController
from .repo_link import RepoLink
from .bandit_router import ExemplarBandit
from .faithfulness import FaithfulnessBot
from .mutation import MutationRunner
from .goals import GoalScorer
from .curriculum import CurriculumScheduler

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