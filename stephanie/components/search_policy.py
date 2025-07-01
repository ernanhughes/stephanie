# --- components/search_policy.py ---

import random


class TreeSearchPolicy:
    def __init__(self, config):
        self.n_init = config.get("n_init", 3)
        self.p_debug = config.get("p_debug", 0.2)
        self.p_greedy = config.get("p_greedy", 0.6)
        self.use_prediction = config.get("use_prediction", True)

    def select(self, tree, predictor=None):
        draft_count = len(tree.nodes)

        if draft_count < self.n_init:
            return None, "draft"

        if random.random() < self.p_debug:
            buggy = tree.get_buggy()
            if buggy:
                return random.choice(buggy), "debug"

        valid = tree.get_valid()
        if not valid:
            return None, "draft"

        if self.use_prediction and predictor:
            # Rank by predicted future value
            ranked = sorted(valid, key=lambda n: predictor.predict(tree.goal, n.plan), reverse=True)
            return ranked[0], "improve"

        if random.random() < self.p_greedy:
            return max(valid, key=lambda n: n.metric), "improve"
        else:
            return random.choice(valid), "improve"
