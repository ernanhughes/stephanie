# stephanie/dataloaders/knowledge_pair_miner.py
from __future__ import annotations

import random
from itertools import islice


def build_contrastive_pairs(cases, max_pairs_per_bucket=20000):
    """
    cases: iterable of dicts {id, text, stars, meta}
    Returns list of pairs: [(better_case, worse_case)]
    """
    by_star = {s: [] for s in range(6)}
    for c in cases:
        s = int(max(0, min(5, c.get("stars", 0))))
        by_star[s].append(c)

    pairs = []
    # bucket across star gaps to avoid only 5vs0
    for hi in range(1, 6):
        for lo in range(0, hi):
            hi_bucket = by_star[hi]
            lo_bucket = by_star[lo]
            if not hi_bucket or not lo_bucket:
                continue
            random.shuffle(hi_bucket); random.shuffle(lo_bucket)
            for a, b in zip(islice(hi_bucket, 0, max_pairs_per_bucket),
                            islice(lo_bucket, 0, max_pairs_per_bucket)):
                pairs.append((a, b))
    random.shuffle(pairs)
    return pairs
