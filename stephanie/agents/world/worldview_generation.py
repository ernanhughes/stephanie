# stephanie/agents/world/worldview_generation.py
class ToolPermissions:
    def __init__(self, enable_web=False, enable_arxiv=False, enable_huggingface=False):
        self.enable_web = enable_web
        self.enable_arxiv = enable_arxiv
        self.enable_huggingface = enable_huggingface


class WorldviewContext:
    def __init__(self, worldview_id, tools: ToolPermissions, embeddings):
        self.id = worldview_id
        self.tools = tools
        self.embeddings = embeddings
        self.beliefs = []
        self.domains = []
        self.goals = []
