# stephanie/scoring/model/sicql.py
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import Linear, ReLU

from stephanie.model.text_encoder import TextEncoder


import hashlib
import logging
from typing import Any, Dict, Optional, Tuple


log = logging.getLogger(__name__)


class PolicyHead(nn.Module):
    def __init__(self, zsa_dim, hdim, num_actions=3):
        super().__init__()
        _log_expected_shapes("PolicyHead", zsa_dim, hdim, num_actions)
        self.linear = nn.Sequential(
            Linear(zsa_dim, hdim), ReLU(), Linear(hdim, num_actions)
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, zsa):
        return self.linear(zsa)

    def get_policy_weights(self):
        """
        Get the averaged weights of the final linear layer for policy logits.
        """
        final_linear_layer = self.linear[-1]
        return final_linear_layer.weight.data.mean(dim=0)


class QHead(nn.Module):
    def __init__(self, zsa_dim, hdim):
        """
        Q-value estimator: Q(s,a) = E[reward | state, action]

        Args:
            zsa_dim: Dimension of encoded state-action vector
            hdim: Hidden layer dimension
        """
        super().__init__()
        _log_expected_shapes("QHead", zsa_dim, hdim)
        self.model = nn.Sequential(
            Linear(zsa_dim, hdim), ReLU(), Linear(hdim, 1)
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, zsa):
        """
        Predict Q-value for (state, action) pair
        Args:
            zsa: Encoded state-action vector
        Returns:
            Q-value (scalar)
        """
        return self.model(zsa).squeeze()


class VHead(nn.Module):
    def __init__(self, zsa_dim, hdim):
        """
        State value estimator using expectile regression

        Args:
            zsa_dim: Dimension of encoded state-action vector
            hdim: Hidden layer dimension
        """
        super().__init__()
        _log_expected_shapes("VHead", zsa_dim, hdim)
        self.net = nn.Sequential(
            Linear(zsa_dim, hdim), ReLU(), Linear(hdim, 1)
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, zsa):
        """
        Predict state value V(s)
        Args:
            zsa: Encoded state-action vector
        Returns:
            State value (scalar)
        """
        return self.net(zsa).squeeze()


class InContextQModel(nn.Module):
    def __init__(
        self,
        encoder: TextEncoder,
        q_head: QHead,
        v_head: VHead,
        pi_head: PolicyHead,
        embedding_store,
        device="cpu",
    ):
        super().__init__()
        self.encoder = encoder.to(device)
        self.q_head = q_head.to(device)
        self.v_head = v_head.to(device)
        self.pi_head = pi_head.to(device)
        self.device = device
        self.embedding_store = embedding_store

    def forward(self, context_emb, doc_emb):
        """
        Forward pass through all heads

        Args:
            context_emb: Goal/prompt embedding
            doc_emb: Document/output embedding
        Returns:
            Dict containing Q-value, state value, and policy logits
        """
        # Ensure device alignment
        context_emb = context_emb.to(self.device)
        doc_emb = doc_emb.to(self.device)

        # Combine embeddings
        zsa = self.encoder(context_emb, doc_emb)

        # Forward through heads
        q_value = self.q_head(zsa)
        state_value = self.v_head(zsa)
        action_logits = self.pi_head(zsa)

        # Calculate advantage
        advantage = (q_value - state_value).detach()

        return {
            "q_value": q_value,
            "state_value": state_value,
            "action_logits": action_logits,
            "advantage": advantage,
        }


# -----------------------------
# Logging utilities (warn-once)
# -----------------------------
_WARNED: set[str] = set()


def _warn_once(key: str, msg: str, *args) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    log.warning(msg, *args)


def _tensor_fingerprint(t: torch.Tensor, n: int = 2048) -> str:
    """
    Stable-ish fingerprint of a tensor's contents without dumping values.
    Uses first N bytes of raw storage on CPU.
    """
    with torch.no_grad():
        x = t.detach().to("cpu").contiguous().view(-1)
        if x.numel() == 0:
            return "empty"
        # sample up to n elements, convert to bytes
        x = x[: min(x.numel(), n)].to(torch.float32)
        b = x.numpy().tobytes()
        return hashlib.sha256(b).hexdigest()[:16]


def _param_summary(m: nn.Module) -> Dict[str, Any]:
    """
    Small param stats to detect 'all zeros', NaNs, or weird scales.
    """
    with torch.no_grad():
        ps = [p.detach() for p in m.parameters() if p is not None]
        if not ps:
            return {"n_params": 0}
        flat = torch.cat([p.float().flatten().cpu() for p in ps])
        return {
            "n_params": int(flat.numel()),
            "mean": float(flat.mean().item()),
            "std": float(flat.std(unbiased=False).item()),
            "min": float(flat.min().item()),
            "max": float(flat.max().item()),
            "fp": _tensor_fingerprint(flat),
        }


def _log_expected_shapes(
    module_name: str,
    zsa_dim: int,
    hdim: int,
    num_actions: Optional[int] = None,
) -> None:
    if num_actions is None:
        log.info(
            "%s expected: Linear(%d -> %d) -> Linear(%d -> 1)",
            module_name,
            zsa_dim,
            hdim,
            hdim,
        )
    else:
        log.info(
            "%s expected: Linear(%d -> %d) -> Linear(%d -> %d)",
            module_name,
            zsa_dim,
            hdim,
            hdim,
            num_actions,
        )


def _peek_state_dict_shapes(
    state_dict: Dict[str, torch.Tensor], keys: Tuple[str, ...]
) -> Dict[str, Tuple[int, ...]]:
    out = {}
    for k in keys:
        v = state_dict.get(k)
        if isinstance(v, torch.Tensor):
            out[k] = tuple(v.shape)
    return out


def log_load_state_dict(
    module: nn.Module,
    state_dict: Dict[str, Any],
    *,
    module_name: str,
    strict: bool = False,
) -> bool:
    """
    Load a state_dict with useful shape logging.
    Returns True if load succeeded, False if RuntimeError.
    """
    # common first-layer keys for these heads
    peek_keys = (
        "model.0.weight",
        "model.0.bias",
        "net.0.weight",
        "net.0.bias",
        "linear.0.weight",
        "linear.0.bias",
    )
    peek = _peek_state_dict_shapes(state_dict, peek_keys)
    if peek:
        log.info("%s ckpt shapes: %s", module_name, peek)

    try:
        missing, unexpected = module.load_state_dict(state_dict, strict=strict)
        # note: shape mismatches raise before this
        if missing or unexpected:
            _warn_once(
                f"load_keys::{module_name}",
                "%s load_state_dict strict=%s missing=%s unexpected=%s",
                module_name,
                strict,
                list(missing),
                list(unexpected),
            )
        summ = _param_summary(module)
        log.info(
            "%s loaded params: n=%s mean=%.6g std=%.6g min=%.6g max=%.6g fp=%s",
            module_name,
            summ.get("n_params"),
            summ.get("mean"),
            summ.get("std"),
            summ.get("min"),
            summ.get("max"),
            summ.get("fp"),
        )
        return True
    except RuntimeError as e:
        _warn_once(
            f"load_fail::{module_name}",
            "%s FAILED to load state_dict (strict=%s): %s",
            module_name,
            strict,
            str(e).splitlines()[0],
        )
        return False
