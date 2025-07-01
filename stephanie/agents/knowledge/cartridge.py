class Cartridge:
    def __init__(self, content, metadata, cartridge_type):
        self.content = content  # (context, response) pairs
        self.metadata = metadata  # {source, timestamp, domains}
        self.type = cartridge_type  # "expert" or "amateur"
    
    def update(self, new_content, feedback_score):
        """Dynamic update based on performance feedback"""
        if feedback_score > CARTRIDGE_UPDATE_THRESHOLD:
            self.content.append(new_content)

class CartridgeManager:
    def __init__(self):
        self.expert_cartridges = []
        self.amateur_cartridges = []
    
    def add_cartridge(self, cartridge):
        if cartridge.type == "expert":
            self.expert_cartridges.append(cartridge)
        else:
            self.amateur_cartridges.append(cartridge)
    
    def find_relevant(self, query, cartridge_type, top_k=3):
        """Retrieve cartridges based on semantic similarity"""
        pool = self.expert_cartridges if cartridge_type == "expert" else self.amateur_cartridges
        return sorted(
            pool,
            key=lambda c: cosine_similarity(embed(query), embed(c.metadata["domains"])),
            reverse=True
        )[:top_k]