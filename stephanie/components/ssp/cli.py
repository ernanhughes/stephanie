from __future__ import annotations

if __name__ == "__main__":
    # Minimal smoke test with three seeds (serve as ground truths)
    from stephanie.components.ssp.training.trainer import Trainer

    seeds = [
        "permafrost thaw releasing methane increases radiative forcing",
        "insulin enables glucose uptake in muscle and adipose tissue",
        "backpropagation updates weights by gradient descent on loss",
    ]

    stats = Trainer(difficulty=0.3, verify_threshold=0.6).run_batch(seeds)
    print("== Summary ==", stats)


# ───────────────────────────────────────────────────────────────────────────────
# configs/self_play_mvp.yaml  (optional; if you’re using Hydra)
# ───────────────────────────────────────────────────────────────────────────────
# defaults:
#   - _self_
#
# self_play_mvp:
#   difficulty: 0.3
#   verify_threshold: 0.6
#   solver:
#     max_depth: 2
#     beam_width: 3
#   proposer:
#     prompt: "Given <answer>{answer}</answer>, generate ONE challenging, verifiable question wrapped in <question> tags."
#
# # Later: wire real model + search + verifier here
# model:
#   router: stephanie.models.ModelRouter
# retriever:
#   kind: memcube
# verifier:
#   kind: hrm_mars
