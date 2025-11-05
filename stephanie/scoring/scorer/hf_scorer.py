# stephanie/scoring/scorer/hf_scorer.py
"""
HuggingFace Scorer Module

A stable, production-ready scorer for HuggingFace Causal Language Models that computes
quality metrics through teacher-forced likelihood estimation. This scorer provides
foundational metrics that can be enhanced by plugins for SCM, calibration, and other
advanced features.

Key Features:
- Stable teacher-forced likelihood computation for response quality assessment
- Multiple quality dimensions: reasoning, knowledge, clarity, faithfulness, coverage
- Windows-friendly operation with eager attention fallback
- Memory-efficient inference with automatic device management
- Plugin architecture for extensible metric computation
- Comprehensive logging and error handling

The scorer computes basic statistics including:
- Perplexity (PPL) and mean log probability
- Token-level entropy and sequence length metrics
- Bits-per-byte (BPB) for information-theoretic analysis
- Token distribution analysis for interpretability

"""

from __future__ import annotations

import gc
import logging
import math
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from stephanie.constants import GOAL, GOAL_TEXT
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult
from stephanie.scoring.scorer.base_scorer import BaseScorer

# Module-level logger for comprehensive debugging and monitoring
log = logging.getLogger(__name__)


class HuggingFaceScorer(BaseScorer):
    """
    Simple, stable HF CausalLM scorer (Windows-friendly).
    
    This scorer computes teacher-forced likelihood statistics of responses conditioned
    on goals, providing foundational metrics for quality assessment. It serves as a
    base for plugin enhancements that add SCM, calibration, and other advanced features.
    
    Key Characteristics:
    - Computes: log probability, perplexity, entropy, length metrics, bits-per-byte
    - Does NOT compute: SCM or calibration (handled by plugins)
    - Stable operation: Uses eager attention to avoid Flash/SDPA issues
    - Memory efficient: Automatic device mapping and dtype selection
    - Extensible: Plugin architecture for additional metric computation
    
    The scorer processes text pairs (goal, response) and returns comprehensive
    statistical metrics that correlate with various quality dimensions.

    Attributes:
        model_type: Identifier for scorer type ("hf")
        model_name: HuggingFace model identifier or path
        tokenizer_name: Tokenizer identifier (defaults to model_name)
        max_seq_len: Maximum sequence length for processing
        device_map: HF device mapping strategy ("auto", "cuda", "cpu", etc.)
        trust_remote_code: Whether to trust remote code in model loading
        model_alias: Short alias for model in metric naming
        dimensions: List of quality dimensions to score
        cache_dir: Directory for model cache (falls back to HF_HOME)
        local_files_only: Whether to use only local model files
        torch_dtype: PyTorch data type for model (auto, float16, float32)
    """

    def __init__(self, cfg, memory, container, logger):
        """Initialize HuggingFaceScorer with configuration and dependencies."""
        super().__init__(cfg, memory, container, logger, enable_plugins=True)
        log.debug("Initializing HuggingFaceScorer with config: %s", 
                     {k: v for k, v in cfg.items() if not k.startswith('_')})
        
        self.model_type = "hf"

        # --- Configuration extraction with safe defaults ---
        self.model_name = str(cfg.get("model_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
        self.tokenizer_name = cfg.get("tokenizer_name") or self.model_name
        self.max_seq_len = int(cfg.get("max_seq_len", 4096))
        self.device_map = cfg.get("device_map", "auto")
        self.trust_remote_code = bool(cfg.get("trust_remote_code", True))
        self.model_alias = str(cfg.get("model_alias", "hf"))
        self.dimensions: List[str] = cfg.get(
            "dimensions",
            ["reasoning", "knowledge", "clarity", "faithfulness", "coverage"],
        )
        log.debug("Model config: name=%s, tokenizer=%s, max_seq_len=%d", 
                     self.model_name, self.tokenizer_name, self.max_seq_len)

        # Optional HF cache configuration
        self.cache_dir = cfg.get("cache_dir") or os.environ.get("HF_HOME") or None
        self.local_files_only = bool(cfg.get("local_files_only", False))
        log.debug("Cache config: cache_dir=%s, local_files_only=%s", 
                     self.cache_dir, self.local_files_only)

        # Data type configuration for performance/memory tradeoffs
        dtype_str = str(cfg.get("torch_dtype", "auto"))
        if dtype_str == "auto":
            self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            self.torch_dtype = getattr(torch, dtype_str, torch.float32)
        log.debug("Data type: %s (from config: %s)", self.torch_dtype, dtype_str)

        # --- Tokenizer initialization with fallback strategy ---
        log.info("Loading tokenizer: %s", self.tokenizer_name)
        tok_id = self.tokenizer_name
        try:
            # First attempt: fast tokenizer for better performance
            self.tok = AutoTokenizer.from_pretrained(
                tok_id,
                use_fast=True,
                trust_remote_code=self.trust_remote_code,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            )
            log.debug("Fast tokenizer loaded successfully")
        except Exception as e_fast:
            # Fallback: slow tokenizer for compatibility
            log.warning("Fast tokenizer failed, falling back to slow tokenizer: %s", str(e_fast))
            if self.logger:
                self.logger.log("HFTokenizerFastFailed", {"model": tok_id, "error": str(e_fast)})
            self.tok = AutoTokenizer.from_pretrained(
                tok_id,
                use_fast=False,
                trust_remote_code=self.trust_remote_code,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            )
            log.debug("Slow tokenizer loaded as fallback")

        # Ensure pad token is set for stable batch processing
        if not getattr(self.tok, "pad_token", None):
            try:
                self.tok.pad_token = self.tok.eos_token
                log.debug("Set pad_token to eos_token")
            except Exception as e:
                log.warning("Failed to set pad_token: %s", str(e))

        # --- Model initialization with stability enhancements ---
        log.info("Loading model: %s", self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
            low_cpu_mem_usage=True,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )
        self.model.eval()  # Set to evaluation mode
        log.debug("Model loaded and set to evaluation mode")

        # Force eager attention for stability (avoid SDPA/Flash issues on Windows/CUDA)
        try:
            if hasattr(self.model, "config"):
                if getattr(self.model.config, "attn_implementation", None) is not None:
                    self.model.config.attn_implementation = "eager"
                    log.debug("Forced eager attention implementation")
                setattr(self.model.config, "_attn_implementation", "eager")
        except Exception as e:
            log.debug("Could not force eager attention: %s", str(e))

        # For uniformity with other scorers in the system
        self.embedding_type = self.memory.embedding.name

        # Device detection for logging and monitoring
        try:
            p = next(self.model.parameters())
            dev_str = str(p.device)
        except Exception:
            dev_str = "unknown"
        log.debug("Model loaded on device: %s", dev_str)

        # Log successful initialization
        if self.logger:
            self.logger.log(
                "HFScorerLoaded",
                {
                    "model": self.model_name, 
                    "alias": self.model_alias, 
                    "device": dev_str, 
                    "dtype": str(self.torch_dtype)
                },
            )
        log.info("HuggingFaceScorer initialized successfully: %s on %s", self.model_name, dev_str)

    # -----------------------------
    # Core Scoring Method
    # -----------------------------

    def _score_core(self, context: dict, scorable, dimensions: List[str]) -> ScoreBundle:
        """
        Core scoring method that computes basic language model statistics.
        
        This method computes teacher-forced likelihood metrics for a response
        conditioned on a goal context. The computed statistics serve as foundational
        metrics that can be enhanced by plugins for more sophisticated scoring.
        
        Args:
            context: Dictionary containing context information with GOAL/GOAL_TEXT
            scorable: Object containing the text response to be scored
            dimensions: List of quality dimensions to compute scores for
            
        Returns:
            ScoreBundle containing ScoreResult objects for each dimension with:
            - Basic statistics: perplexity, entropy, log probability, length metrics
            - Placeholder score (0.0) that can be enhanced by plugins
            - Comprehensive attributes for plugin processing
            - Rationale explaining the computed metrics
            
        Note:
            This method does NOT compute semantic scores - plugins are expected
            to enhance the results with SCM, calibration, and other advanced features.
        """
        log.debug("Starting core scoring for %d dimensions: %s", len(dimensions), dimensions)
        
        # Extract goal and response text from inputs
        goal_text = (context.get(GOAL, {}) or {}).get(GOAL_TEXT, "") or ""
        resp_text = scorable.text or ""
        log.debug("Scoring: goal=%d chars, response=%d chars", len(goal_text), len(resp_text))

        # Compute language model statistics with teacher forcing
        with torch.no_grad():
            stats = self._ll_stats(goal_text, resp_text)
        log.debug("LL stats computed: ppl=%.2f, mean_logprob=%.3f, entropy=%.3f", 
                     stats["ppl"], stats["mean_logprob"], stats["entropy_mean"])

        # Minimal attribute set that plugins can build upon
        base_attrs = {
            "mean_logprob": stats["mean_logprob"],    # Average log probability per token
            "ppl": stats["ppl"],                      # Perplexity (exp(-mean_logprob))
            "entropy_mean": stats["entropy_mean"],    # Average token-level entropy
            "len_tokens": stats["len_tokens"],        # Response length in tokens
            "len_chars": stats["len_chars"],          # Response length in characters
            "bytes_len": stats["bytes_len"],          # Response length in bytes
            "sum_nll_nats": stats["sum_nll_nats"],    # Total negative log likelihood
            "bpb": stats["bpb"],                      # Bits per byte (information density)
        }

        # Build feature vector for machine learning compatibility
        vector = self._build_base_vector(self.model_alias, base_attrs)
        log.debug("Base vector built with %d features", len(vector.get("columns", [])))

        # Create ScoreResult objects for each dimension
        results: Dict[str, ScoreResult] = {}
        for dim in dimensions:
            # Note: score=0.0 is a placeholder - plugins should set meaningful scores
            # The rationale provides human-readable explanation of the computed metrics
            results[dim] = ScoreResult(
                dimension=dim,
                score=0.0,  # Placeholder for plugin enhancement
                source=self.model_type,
                rationale=(
                    f"{self.model_alias}[{dim}] ppl={stats['ppl']:.2f}, "
                    f"H̄={stats['entropy_mean']:.3f}, lp̄={stats['mean_logprob']:.3f}"
                ),
                weight=1.0,
                attributes={**base_attrs, **vector},  # Combine base stats with vector representation
            )
            log.debug("Created result for dimension '%s'", dim)

        log.info("Core scoring completed: %d results with ppl=%.2f", len(results), stats["ppl"])
        return ScoreBundle(results=results)

    # -----------------------------
    # Token Distribution Analysis
    # -----------------------------

    @torch.no_grad()
    def token_topk(self, goal: str, resp: str, k: int = 5) -> Optional[List[List[tuple[str, float]]]]:
        """
        Compute top-k token distributions for analysis and interpretability.
        
        This method provides detailed token-level probability distributions
        that can be used for:
        - Model behavior analysis and debugging
        - Confidence estimation and uncertainty quantification
        - Interpretability and explainability features
        
        Args:
            goal: Context/goal text that conditions the generation
            resp: Response text to analyze token distributions for
            k: Number of top tokens to return for each position
            
        Returns:
            List of token distributions for each response token position, where
            each distribution is a list of (token, probability) tuples, or
            None if k <= 0 or analysis is disabled
            
        Example:
            token_topk("What is AI?", "Artificial Intelligence", k=2)
            → [[("Artificial", 0.8), ("AI", 0.15)], [("Intelligence", 0.7), ("AI", 0.2)]]
        """
        if not k or k <= 0:
            log.debug("Token top-k analysis disabled (k=%d)", k)
            return None

        log.debug("Computing top-%d token distributions: goal=%d chars, resp=%d chars", 
                     k, len(goal), len(resp))
        
        try:
            # Tokenize inputs separately for precise position tracking
            enc_goal = self.tok(goal, return_tensors="pt", add_special_tokens=False)
            enc_resp = self.tok(resp, return_tensors="pt", add_special_tokens=False)
            
            # Combine goal and response for full context processing
            input_ids = torch.cat([enc_goal["input_ids"], enc_resp["input_ids"]], dim=1)
            input_ids = input_ids.to(self.model.device)
            attention_mask = torch.ones_like(input_ids)
            log.debug("Input sequence: %d tokens (goal: %d, resp: %d)", 
                         input_ids.shape[1], enc_goal["input_ids"].shape[1], enc_resp["input_ids"].shape[1])

            # Forward pass to get logits
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            shift_logits = out.logits[:, :-1, :]  # Shift for teacher forcing
            shift_labels = input_ids[:, 1:]
            
            # Extract response portion (after goal context)
            start = enc_goal["input_ids"].shape[1]
            resp_logits = shift_logits[:, start:, :]  # [1, Lr, V]
            probs = torch.softmax(resp_logits, dim=-1)[0]  # [Lr, V]
            log.debug("Response logits shape: %s, probs shape: %s", 
                         resp_logits.shape, probs.shape)

            # Compute top-k tokens and probabilities
            topv, topi = probs.topk(k, dim=-1)  # [Lr, k]
            toks = [self.tok.convert_ids_to_tokens(ids.tolist()) for ids in topi]
            
            # Format results as (token, probability) pairs
            result = [
                [(toks[t][j], float(topv[t, j].item())) for j in range(k)]
                for t in range(topi.size(0))
            ]
            log.debug("Computed top-%d distributions for %d response tokens", k, len(result))
            return result
            
        except Exception as e:
            log.error("Failed to compute token top-k: %s", str(e))
            return None

    # -----------------------------
    # Core Language Model Statistics
    # -----------------------------

    @torch.no_grad()
    def _ll_stats(self, goal: str, resp: str) -> Dict[str, float]:
        """
        Compute teacher-forced language model statistics for response given goal.
        
        This is the core method that computes all foundational metrics through
        teacher-forced likelihood estimation. It handles sequence length constraints
        through safe truncation and provides comprehensive statistical analysis.
        
        Args:
            goal: Context/goal text that conditions the generation
            resp: Response text to compute statistics for
            
        Returns:
            Dictionary containing:
            - mean_logprob: Average log probability per response token
            - ppl: Perplexity (exponential of negative mean log probability)
            - entropy_mean: Average token-level entropy in nats
            - len_tokens: Response length in tokens
            - len_chars: Response length in characters  
            - bytes_len: Response length in UTF-8 bytes
            - sum_nll_nats: Total negative log likelihood in nats
            - bpb: Bits per byte (information-theoretic measure)
            
        Note:
            Returns safe defaults for empty responses to avoid division by zero
        """
        log.debug("Computing LL stats: goal=%d chars, resp=%d chars", len(goal), len(resp))
        
        # Tokenize inputs separately for precise position tracking
        enc_goal = self.tok(goal, return_tensors="pt", add_special_tokens=False)
        enc_resp = self.tok(resp, return_tensors="pt", add_special_tokens=False)

        g_ids = enc_goal["input_ids"]
        r_ids = enc_resp["input_ids"]
        goal_len = g_ids.shape[1]
        total_len = goal_len + r_ids.shape[1]
        log.debug("Tokenized: goal=%d tokens, resp=%d tokens, total=%d", 
                     goal_len, r_ids.shape[1], total_len)

        # Combine goal and response tokens
        input_ids = torch.cat([g_ids, r_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)

        # Handle sequence length constraints with safe truncation
        if total_len > self.max_seq_len:
            cut = total_len - self.max_seq_len
            input_ids = input_ids[:, cut:]
            attention_mask = attention_mask[:, cut:]
            resp_start = max(0, goal_len - cut)
            log.debug("Sequence truncated: cut=%d tokens, new_total=%d, resp_start=%d", 
                         cut, input_ids.shape[1], resp_start)
        else:
            resp_start = goal_len
            log.debug("No truncation needed, resp_start=%d", resp_start)

        # Move to model device for inference
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        log.debug("Moved inputs to device: %s", device)

        # Forward pass to compute logits
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits  # [B, T, V]
        log.debug("Model forward completed: logits shape=%s", logits.shape)

        # Shift for teacher-forcing alignment
        shift_logits = logits[:, :-1, :]    # Predict next token
        shift_labels = input_ids[:, 1:]     # Actual next token

        # Extract response portion after goal context
        resp_logits = shift_logits[:, resp_start:, :]   # [B, Lr, V]
        resp_labels = shift_labels[:, resp_start:]      # [B, Lr]
        log.debug("Response portion: logits=%s, labels=%s", 
                     resp_logits.shape, resp_labels.shape)

        # Handle empty response case gracefully
        if resp_labels.numel() == 0:
            log.warning("Empty response after processing, returning safe defaults")
            vocab_est = (
                getattr(self.tok, "vocab_size", None)
                or (len(getattr(self.tok, "get_vocab")() or {}) if hasattr(self.tok, "get_vocab") else None)
                or 32000  # Reasonable default for most models
            )
            return dict(
                mean_logprob=0.0,
                ppl=float("inf"),
                entropy_mean=math.log(vocab_est),  # Maximum entropy for uniform distribution
                len_tokens=0,
                len_chars=len(resp),
                bytes_len=self._bytes_len(resp),
                sum_nll_nats=0.0,
                bpb=0.0,
            )

        # Compute log probabilities for actual response tokens
        logprobs = F.log_softmax(resp_logits, dim=-1)              # [B, Lr, V]
        chosen_lp = torch.gather(logprobs, dim=-1, index=resp_labels.unsqueeze(-1)).squeeze(-1)  # [B, Lr]
        mean_logprob = float(chosen_lp.mean().item())
        log.debug("Mean log probability: %.4f", mean_logprob)

        # Compute token-level entropy (uncertainty measure)
        probs = logprobs.exp()
        ent = -(probs * logprobs).sum(dim=-1)                      # [B, Lr]
        entropy_mean = float(ent.mean().item())
        log.debug("Mean entropy: %.4f nats", entropy_mean)

        # Compute perplexity (standard language model metric)
        ppl = float(math.exp(-mean_logprob))
        log.debug("Perplexity: %.2f", ppl)

        # Information-theoretic measures
        sum_nll_nats = float(-chosen_lp.sum().item())  # Total negative log likelihood
        bytes_len = self._bytes_len(resp)
        bpb = self._to_bits(sum_nll_nats) / max(bytes_len, 1)      # Bits per byte
        log.debug("Information metrics: sum_nll=%.2f nats, bytes=%d, bpb=%.3f", 
                     sum_nll_nats, bytes_len, bpb)

        return dict(
            mean_logprob=mean_logprob,
            ppl=ppl,
            entropy_mean=entropy_mean,
            len_tokens=int(resp_labels.numel()),
            len_chars=len(resp),
            bytes_len=int(bytes_len),
            sum_nll_nats=sum_nll_nats,
            bpb=float(bpb),
        )

    # -----------------------------
    # Feature Vector Construction
    # -----------------------------

    def _build_base_vector(self, alias: str, attrs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build standardized feature vector from computed attributes.
        
        Creates a consistent vector representation that can be used by:
        - Machine learning models for downstream tasks
        - Plugin systems for enhanced scoring
        - Monitoring and analysis systems
        
        The vector uses namespaced keys to avoid collisions when multiple
        models or scorers are used in the same system.

        Args:
            alias: Model alias for namespacing (e.g., "hf", "gpt", "llama")
            attrs: Dictionary of computed attributes from _ll_stats
            
        Returns:
            Dictionary containing:
            - vector: Dict of feature names to values
            - columns: List of feature names in consistent order
            - values: List of feature values matching columns order
        """
        log.debug("Building base vector for alias: %s", alias)
        
        # Define feature keys with namespacing
        keys = [
            f"{alias}.mean_logprob",    # Average log probability
            f"{alias}.ppl",             # Perplexity
            f"{alias}.entropy_mean",    # Average entropy
            f"{alias}.len_tokens",      # Token length
            f"{alias}.len_chars",       # Character length
            f"{alias}.bytes_len",       # Byte length
            f"{alias}.sum_nll_nats",    # Total negative log likelihood
            f"{alias}.bpb",             # Bits per byte
        ]

        # Build vector with safe value extraction and type conversion
        vec: Dict[str, float] = {}
        vec[f"{alias}.mean_logprob"] = float(attrs.get("mean_logprob", 0.0))
        vec[f"{alias}.ppl"] = float(attrs.get("ppl", float("inf")))
        vec[f"{alias}.entropy_mean"] = float(attrs.get("entropy_mean", 0.0))
        vec[f"{alias}.len_tokens"] = float(attrs.get("len_tokens", 0))
        vec[f"{alias}.len_chars"] = float(attrs.get("len_chars", 0))
        vec[f"{alias}.bytes_len"] = float(attrs.get("bytes_len", 0))
        vec[f"{alias}.sum_nll_nats"] = float(attrs.get("sum_nll_nats", 0.0))
        vec[f"{alias}.bpb"] = float(attrs.get("bpb", 0.0))

        # Create ordered representations for consistent processing
        cols = list(vec.keys())
        vals = [vec[c] for c in cols]
        
        log.debug("Built vector with %d features for %s", len(cols), alias)
        return {"vector": vec, "columns": cols, "values": vals}

    # -----------------------------
    # Utility Methods
    # -----------------------------

    def _bytes_len(self, s: str) -> int:
        """
        Safely compute UTF-8 byte length of a string.
        
        Args:
            s: Input string to measure
            
        Returns:
            Number of bytes when encoded as UTF-8, or character length as fallback
        """
        try:
            return len(s.encode("utf-8"))
        except Exception as e:
            log.debug("UTF-8 encoding failed, using character length: %s", str(e))
            return len(s)

    def _to_bits(self, nats: float) -> float:
        """
        Convert nats to bits for information-theoretic measures.
        
        Args:
            nats: Value in nats (natural log units)
            
        Returns:
            Value in bits (log2 units)
            
        Note:
            Conversion formula: bits = nats / ln(2)
        """
        return float(nats / math.log(2.0))

    # -----------------------------
    # Resource Cleanup
    # -----------------------------

    def close(self):
        """
        Comprehensive resource cleanup and memory management.
        
        This method ensures proper cleanup of HuggingFace models and tokenizers
        to prevent memory leaks and resource contention. It handles:
        - Accelerate hook removal
        - Model offloading and reference clearing
        - Tokenizer cleanup
        - GPU memory cleanup and cache clearing
        - Parent class cleanup chain
        
        Should be called when the scorer is no longer needed to free resources.
        """
        log.info("Closing HuggingFaceScorer and cleaning up resources")
        try:
            # 0) Remove accelerate hooks if present to prevent reference cycles
            try:
                import accelerate
                accelerate.hooks.remove_hook_from_submodules(self.model)
                log.debug("Removed accelerate hooks")
            except Exception as e:
                log.debug("No accelerate hooks to remove: %s", str(e))

            # 1) Move model to CPU and drop references for GC
            if getattr(self, "model", None) is not None:
                try:
                    # Clear device mapping to break cross-device references
                    if hasattr(self.model, "hf_device_map"):
                        self.model.hf_device_map = None
                    # Move to CPU before deletion
                    self.model.to("cpu")
                    log.debug("Moved model to CPU")
                except Exception as e:
                    logning("Error moving model to CPU: %s", str(e))
                finally:
                    self.model = None
                    log.debug("Cleared model reference")

            # 2) Drop tokenizer reference
            if getattr(self, "tok", None) is not None:
                self.tok = None
                log.debug("Cleared tokenizer reference")

            # 3) Teardown any offload managers
            try:
                offload = getattr(self, "_cpu_offload", None)
                if offload and hasattr(offload, "teardown"):
                    offload.teardown()
                    log.debug("Tore down CPU offload manager")
            except Exception as e:
                log.debug("No CPU offload manager to teardown: %s", str(e))

            # 4) Aggressive garbage collection and memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.reset_peak_memory_stats()
                log.debug("Cleaned GPU memory and cache")
                
        except Exception as e:
            log.error("Error during HuggingFaceScorer cleanup: %s", str(e))
        finally:
            # Always call parent cleanup for plugin teardown
            super().close()
            log.info("HuggingFaceScorer closed successfully")