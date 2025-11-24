# stephanie/agents/visual_introspection.py
from __future__ import annotations

import json
import logging
import re
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from datasets import load_dataset

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import Scorable

log = logging.getLogger(__name__)

class VisualIntrospectionAgent(BaseAgent):
    """
    GSM8K → DeepSeek → Scorables + JSONL log.

    This agent is optimized for DeepSeek's output format, with:
    - Flexible section header parsing for natural language outputs
    - Multi-pattern final answer extraction
    - Quality metrics for reasoning traces
    - Robust fallback mechanisms when parsing fails
    
    Key features:
    - Loads a subset of GSM8K from Hugging Face
    - Shuffles and selects N examples per run
    - Builds a prompt specifically designed for DeepSeek
    - Calls DeepSeek via self.call_llm
    - Robustly parses DeepSeek's natural language output
    - Builds Scorable objects with correctness + quality metrics
    - Splits into:
        * context["scorables_targeted"] = correct solutions
        * context["scorables_baseline"] = incorrect solutions
        * context["scorables"] = union of both (for VisiCalcAgent input)
    - Writes a JSONL log file for this run with comprehensive quality metrics
    
    Configure in Hydra:
    
      visual_introspection:
        _target_: stephanie.agents.visual_introspection.VisualIntrospectionAgent
        name: visual_introspection
        enabled: true
        input_key: "scorables"          # VisiCalcAgent expects this later
        output_key: "visual_introspection"
        strategy: "gsm8k_solve"
        num_examples: 200               # Recommended for Tiny Critic training
        split: "train"
        store_raw: true
        out_dir: "runs/visual_introspection"
        shuffle: true
        model_name: "deepseek-math:7b"  # Optimized for this model
        params:
          temperature: 0.3
          max_tokens: 1024
    """

    def __init__(self, cfg: Dict[str, Any], memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # Strategy label for logging / telemetry
        self.strategy: str = cfg.get("strategy", "gsm8k_solve")

        # Dataset knobs
        self.num_examples: int = int(cfg.get("num_examples", 200))  # Increased for Tiny Critic
        self.split: str = cfg.get("split", "train")
        self.store_raw: bool = bool(cfg.get("store_raw", True))
        self.shuffle: bool = bool(cfg.get("shuffle", True))

        # Model configuration
        self.model_name: str = cfg.get("model_name", "deepseek-math:7b")
        self.params: Dict[str, Any] = cfg.get("params", {
            "temperature": 0.3,
            "max_tokens": 1024,
            "top_p": 0.95
        })

        # Output logging
        out_root = Path(cfg.get("out_dir", "runs/visual_introspection"))
        # Each run_id is unique, so we naturally get a per-run directory
        self.out_dir: Path = out_root / self.run_id
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # JSONL file for this run (one record per GSM8K example)
        self.jsonl_path: Path = Path(
            cfg.get("jsonl_file", self.out_dir / "gsm8k_deepseek_samples.jsonl")
        )

        # Visualization directory for feature analysis
        self.vis_dir: Path = self.out_dir / "visualizations"
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        # Quality tracking
        self.quality_metrics = {
            "total_examples": 0,
            "successful_parsing": 0,
            "failed_parsing": 0,
            "correct_answers": 0,
            "incorrect_answers": 0,
            "reasoning_found": 0,
            "final_answer_found": 0,
            "high_quality_reasoning": 0
        }

        log.info(f"🎯 VisualIntrospectionAgent initialized:")
        log.info(f"   Strategy: {self.strategy}")
        log.info(f"   Model: {self.model_name}")
        log.info(f"   Examples: {self.num_examples}")
        log.info(f"   Split: {self.split}")
        log.info(f"   Output: {self.out_dir}")
        log.info(f"   Params: temperature={self.params.get('temperature', 0.3)}, max_tokens={self.params.get('max_tokens', 1024)}")

    # ------------------------------------------------------------------ #
    # Enhanced prompt template for DeepSeek
    # ------------------------------------------------------------------ #
    def _get_deepseek_prompt(self, question: str) -> str:
        """Generate the prompt optimized for DeepSeek's output style"""
        return f"""You are a math reasoning expert. Solve this problem step by step and format your answer EXACTLY as shown below.

PROBLEM:
{question}

YOUR RESPONSE MUST FOLLOW THIS EXACT FORMAT:

Step-by-step reasoning:
1. [First step with calculations]
2. [Second step with calculations] 
3. [Third step with calculations]
...

Final numeric answer: [ONLY THE NUMBER]

IMPORTANT: Do not use any XML tags. Just follow the format above."""

    # ------------------------------------------------------------------ #
    # Robust DeepSeek output parser with multiple fallbacks
    # ------------------------------------------------------------------ #
    def _parse_deepseek_output(self, raw_text: str) -> Dict[str, Any]:
        """Parse DeepSeek's natural language output with multiple fallback strategies"""
        log.info(f"🔍 Parsing DeepSeek output (length: {len(raw_text)} chars)")
        
        # Strategy 1: Look for the exact format we requested
        reasoning_match = re.search(
            r"(?:Step[-\s]*by[-\s]*step\s*reasoning|Reasoning\s*steps|Solution\s*steps)[:\s]*(.*?)(?=Final\s*(?:numeric\s*)?answer:|$)", 
            raw_text, 
            re.DOTALL | re.IGNORECASE
        )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        
        # Strategy 2: If that fails, look for numbered steps anywhere in the text
        if not reasoning:
            steps_match = re.search(r"(?:Step[-\s]*by[-\s]*step|Reasoning|Solution)(?:[:\s]*\n+)?(.*?)(?=\n\s*Final\s*(?:numeric\s*)?answer|$)", 
                                   raw_text, re.DOTALL | re.IGNORECASE)
            if steps_match:
                reasoning = steps_match.group(1).strip()
        
        # Strategy 3: If still no reasoning, take everything before "Final answer"
        if not reasoning:
            final_match = re.search(r"(.*?)(?:\n\s*Final\s*(?:numeric\s*)?answer)", raw_text, re.DOTALL | re.IGNORECASE)
            if final_match:
                reasoning = final_match.group(1).strip()
        
        # Strategy 4: As a last resort, take the first paragraph as reasoning
        if not reasoning:
            paragraphs = [p.strip() for p in raw_text.split('\n\n') if p.strip()]
            if paragraphs:
                reasoning = paragraphs[0]
        
        # Extract final answer with multiple patterns
        final_answer = ""
        patterns = [
            r"Final\s*(?:numeric\s*)?answer\s*[:\-]?\s*([^\n]+)",
            r"Answer\s*[:\-]?\s*([^\n]+)",
            r"The\s*answer\s*is\s*([^\n]+)",
            r"=+\s*([^\n]+)\s*=+",
            r"boxed\{([^\}]+)\}"
        ]
        
        for pattern in patterns:
            m = re.search(pattern, raw_text, re.IGNORECASE)
            if m:
                final_answer = m.group(1).strip()
                break
        
        # Count steps in reasoning (handle various numbering styles)
        step_count = len(re.findall(r"^\s*(?:\d+\.|\([a-z]\)|•)\s+", reasoning, re.MULTILINE)) if reasoning else 0
        
        # Check for verification (key for reasoning quality)
        verification_present = bool(re.search(r"\b(verify|check|confirm|validate|double-check)\b", reasoning, re.IGNORECASE))
        
        log.info(f"📊 DeepSeek parsing - Reasoning: {len(reasoning)} chars, Steps: {step_count}, Final: '{final_answer}', Verification: {verification_present}")
        
        return {
            "reasoning": reasoning,
            "final_answer": final_answer,
            "step_count": step_count,
            "verification_present": verification_present,
            "raw": raw_text,
        }

    # ------------------------------------------------------------------ #
    # Safe generation with retries and timeouts
    # ------------------------------------------------------------------ #
    async def _generate_with_retry(self, prompt: str, max_retries: int = 3, context: dict={}) -> str:
        """Generate with retries, timeouts, and fallbacks"""
        for attempt in range(max_retries):
            try:
                # Add timeout to prevent hanging
                raw_text = await asyncio.wait_for(
                    self._call_llm_with_params(prompt, context=context),
                    timeout=30.0
                )
                return raw_text
            except asyncio.TimeoutError:
                log.warning(f"⚠️  Generation timed out (attempt {attempt+1}/{max_retries})")
            except Exception as e:
                log.warning(f"⚠️  Generation failed (attempt {attempt+1}/{max_retries}): {str(e)}")
            
            if attempt < max_retries - 1:
                await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff
        
        # Final fallback: minimal generation
        log.error("❌ All generation attempts failed, using minimal fallback")
        return "Step-by-step reasoning:\n1. Error in generation\n\nFinal numeric answer: 0"

    async def _call_llm_with_params(self, prompt: str, context: dict) -> str:
        """Call LLM with configured parameters"""
        return await self.async_call_llm(
            prompt,
            params=self.params,
            context=context
        )

    # ------------------------------------------------------------------ #
    # Feature visualization generation
    # ------------------------------------------------------------------ #
    def _generate_feature_visualizations(self, all_scorables: List[Scorable]):
        """Generate visualizations of feature distributions by reasoning quality with robust KDE handling"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from scipy import stats
        
        log.info("📊 Creating feature visualizations for Tiny Critic training...")
        
        # Extract features from scorables
        features = {
            "stability": [],
            "middle_dip": [],
            "std_dev": [],
            "sparsity": [],
            "entropy": [],
            "trend": [],
            "mid_bad_ratio": [],
            "frontier_util": []
        }
        labels = []
        
        for scorable in all_scorables:
            meta = scorable.meta
            # These features will be populated by VisiCalc, but we can use quality metrics as proxy
            features["stability"].append(meta.get("reasoning_length", 0))
            features["middle_dip"].append(meta.get("step_count", 0))
            features["std_dev"].append(1 if meta.get("verification_present", False) else 0)
            features["sparsity"].append(meta.get("reasoning_quality", 0))
            features["entropy"].append(1 if meta.get("is_correct", False) else 0)
            features["trend"].append(meta.get("step_count", 0) / max(1, len(meta.get("reasoning", "").split("."))))
            features["mid_bad_ratio"].append(1 if meta.get("verification_present", False) else 0)
            features["frontier_util"].append(meta.get("reasoning_quality", 0))
            labels.append(meta.get("is_correct", False))
        
        # Convert to numpy arrays
        X = np.array([features[k] for k in features]).T
        y = np.array(labels)
        
        # Create distribution plots
        plt.figure(figsize=(15, 12))
        
        # 1. Distribution plots for each feature - with robust KDE handling
        feature_names = list(features.keys())
        for i, name in enumerate(feature_names):
            plt.subplot(3, 3, i+1)
            
            # Check if feature has enough variation for KDE
            good_data = X[:, i][~np.isnan(X[:, i])]
            has_variation = len(np.unique(good_data)) > 1 and np.var(good_data) > 1e-10
            
            # Only enable KDE if there's sufficient variation
            kde_enabled = has_variation and len(good_data) > 30
            
            # Create plot with appropriate KDE setting
            sns.histplot(
                data={name: X[y == 1, i], 'Bad Reasoning': X[y == 0, i]}, 
                kde=kde_enabled,
                bins=15,
                alpha=0.6
            )
            
            # Add warning if KDE was disabled
            if not kde_enabled and has_variation:
                plt.text(0.05, 0.95, "KDE disabled (small dataset)", 
                        transform=plt.gca().transAxes, 
                        fontsize=8,
                        color='red')
            elif not has_variation:
                plt.text(0.05, 0.95, "No variation in data", 
                        transform=plt.gca().transAxes, 
                        fontsize=8,
                        color='red')
                
            plt.title(f'Distribution: {name}')
            plt.xlabel('Value')
            plt.ylabel('Count')
        
        # 2. Box plots to show differences between classes
        plt.subplot(3, 3, 8)
        all_values = []
        all_features = []
        all_classes = []
        
        for i, name in enumerate(feature_names):
            all_values.extend(X[y == 1, i])
            all_features.extend([name] * len(X[y == 1, i]))
            all_classes.extend(['Good Reasoning'] * len(X[y == 1, i]))
            
            all_values.extend(X[y == 0, i])
            all_features.extend([name] * len(X[y == 0, i]))
            all_classes.extend(['Bad Reasoning'] * len(X[y == 0, i]))
        
        sns.boxplot(x=all_features, y=all_values, hue=all_classes)
        plt.xticks(rotation=45, ha='right')
        plt.title('Feature Distributions by Reasoning Quality')
        plt.tight_layout()
        
        # 3. Feature importance potential analysis
        plt.subplot(3, 3, 9)
        mean_diffs = []
        for i, name in enumerate(feature_names):
            good_mean = np.mean(X[y == 1, i]) if sum(y == 1) > 0 else 0
            bad_mean = np.mean(X[y == 0, i]) if sum(y == 0) > 0 else 0
            mean_diffs.append((name, abs(good_mean - bad_mean)))
        
        mean_diffs.sort(key=lambda x: x[1], reverse=True)
        names, diffs = zip(*mean_diffs)
        
        sns.barplot(x=list(names), y=list(diffs))
        plt.xticks(rotation=45, ha='right')
        plt.title('Feature Discriminative Power')
        plt.ylabel('Mean Difference')
        
        plt.tight_layout()
        plt.savefig(self.vis_dir / 'feature_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log feature importance potential
        log.info("🔍 Feature importance potential (based on mean differences):")
        for name, diff in mean_diffs:
            log.info(f"   - {name}: {diff:.4f}")
        
        # Save feature importance to file
        with open(self.vis_dir / 'feature_importance.txt', 'w') as f:
            f.write("Feature Importance Analysis\n")
            f.write("========================\n\n")
            for name, diff in mean_diffs:
                f.write(f"{name}: {diff:.4f}\n")

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        log.info("🚀 Starting VisualIntrospectionAgent data generation...")
        start_time = datetime.now()
        
        try:
            # -----------------------------------------------------------------
            # 1) Load dataset and pick a random subset
            # -----------------------------------------------------------------
            log.info(f"📚 Loading GSM8K dataset (split: {self.split})")
            ds = load_dataset("gsm8k", "main", split=self.split)
            num_total = len(ds)
            
            if num_total == 0:
                raise ValueError(f"GSM8K split '{self.split}' is empty")

            log.info(f"📊 Dataset loaded: {num_total} total examples")

            # Shuffle to avoid re-using the same examples every run.
            # Use run_id to derive a unique but deterministic seed for this run.
            if self.shuffle:
                # hash(run_id) is stable per run; mask to 32 bits for HF
                seed = int(hash(self.run_id) & 0xFFFFFFFF)
                ds = ds.shuffle(seed=seed)
                log.info(f"🔀 Dataset shuffled with seed: {seed}")

            n = min(self.num_examples, num_total)
            log.info(f"🎯 Selected {n} examples from {num_total} total")
            ds = ds.select(range(n))

            # -----------------------------------------------------------------
            # 2) Iterate over examples, call LLM, parse, build scorables
            # -----------------------------------------------------------------
            all_scorables: List[Scorable] = []
            good_scorables: List[Scorable] = []
            bad_scorables: List[Scorable] = []
            jsonl_records: List[Dict[str, Any]] = []

            log.info(f"🤖 Generating reasoning with {self.model_name} (temperature={self.params.get('temperature', 0.3)})")
            log.info(f"📝 Processing {n} examples...")

            for i, row in enumerate(ds):
                problem_id = row.get("id") or f"gsm8k-{self.split}-{i}"
                question: str = row["question"]
                gold_answer: str = row.get("answer", "")

                log.info(f"🔍 Processing example {i+1}/{n}: {problem_id}")
                log.info(f"   Question: {question[:100]}...")

                # Build prompt specifically for DeepSeek
                prompt = self._get_deepseek_prompt(question)
                log.info(f"✅ Prompt built (length: {len(prompt)} chars)")

                # Call DeepSeek with retry mechanism
                log.info("🤖 Calling LLM for generation...")
                raw_text = await self._generate_with_retry(prompt, context=context)
                log.info(f"✅ LLM response received (length: {len(raw_text)} chars)")

                # Parse DeepSeek's output with robust parser
                log.info("🔍 Parsing DeepSeek output...")
                parsed = self._parse_deepseek_output(raw_text)
                log.info(f"✅ Parsing complete: reasoning length={len(parsed['reasoning'])}, steps={parsed['step_count']}, final_answer='{parsed['final_answer']}'")
                
                # Update parsing quality metrics
                self.quality_metrics["total_examples"] += 1
                if parsed["reasoning"]:
                    self.quality_metrics["reasoning_found"] += 1
                if parsed["final_answer"]:
                    self.quality_metrics["final_answer_found"] += 1
                if parsed["reasoning"] and parsed["step_count"] >= 3 and parsed["verification_present"]:
                    self.quality_metrics["high_quality_reasoning"] += 1
                    
                if parsed["reasoning"] and parsed["final_answer"]:
                    self.quality_metrics["successful_parsing"] += 1
                else:
                    self.quality_metrics["failed_parsing"] += 1
                    log.warning(f"⚠️  Incomplete parsing for {problem_id}: reasoning={bool(parsed['reasoning'])}, final_answer={bool(parsed['final_answer'])}")

                # Build a Scorable with correctness and quality metrics
                scorable = build_scorable_for_example(
                    problem_id=problem_id,
                    question=question,
                    gold_answer=gold_answer,
                    raw_text=raw_text,
                    parsed=parsed,
                    source_label=self.model_name,
                )

                all_scorables.append(scorable)
                if scorable.meta.get("is_correct"):
                    log.info(f"✅ Correct: {problem_id} (answer: {scorable.meta.get('pred_answer_canonical', 'N/A')})")
                    good_scorables.append(scorable)
                    self.quality_metrics["correct_answers"] += 1
                else:
                    log.info(f"❌ Incorrect: {problem_id} (pred: {scorable.meta.get('pred_answer_canonical', 'N/A')}, gold: {scorable.meta.get('gold_answer_canonical', 'N/A')})")
                    bad_scorables.append(scorable)
                    self.quality_metrics["incorrect_answers"] += 1

                # Prepare JSONL record for this example
                rec: Dict[str, Any] = {
                    "task": "gsm8k",
                    "split": self.split,
                    "problem_id": problem_id,
                    "model_name": self.model_name,
                    "strategy": self.strategy,
                    "question": question,
                    "gold_answer": gold_answer,
                    "prompt": prompt,
                    "raw_response": raw_text if self.store_raw else None,
                    "parsed": parsed,
                    "scorable": {
                        "text": scorable.text,
                        "external_id": scorable.id,
                        "meta": scorable.meta,
                    },
                }
                jsonl_records.append(rec)

                # Log progress for large batches
                if (i + 1) % 50 == 0:
                    log.info(f"📦 Processed {i + 1}/{n} examples")
                    current_correct = len(good_scorables)
                    log.info(f"📊 Current accuracy: {current_correct}/{i+1} ({current_correct/(i+1)*100:.1f}%)")

            # -----------------------------------------------------------------
            # 3) Write JSONL log file for this run
            # -----------------------------------------------------------------
            log.info(f"💾 Writing {len(jsonl_records)} GSM8K {self.model_name} samples to {self.jsonl_path}")
            try:
                with self.jsonl_path.open("w", encoding="utf-8") as f:
                    for rec in jsonl_records:
                        json.dump(rec, f, ensure_ascii=False)
                        f.write("\n")
                log.info(f"✅ JSONL file saved successfully: {self.jsonl_path}")
                
                # Log file size
                file_size = self.jsonl_path.stat().st_size
                file_size_mb = file_size / (1024 * 1024)
                log.info(f"📁 File size: {file_size_mb:.2f} MB")
                
            except Exception as e:
                log.error(f"❌ Failed to write JSONL file: {e}")
                raise

            # -----------------------------------------------------------------
            # 4) Generate feature visualizations for Tiny Critic
            # -----------------------------------------------------------------
            log.info("📊 Generating feature visualizations for Tiny Critic training...")
            self._generate_feature_visualizations(all_scorables)
            log.info(f"✅ Feature visualizations saved to {self.vis_dir}")

            # -----------------------------------------------------------------
            # 5) Populate context for downstream agents (VisiCalcAgent)
            # -----------------------------------------------------------------
            # These scorables will be processed by ScorableProcessor, then VisiCalc.
            context["scorables"] = all_scorables
            context["scorables_targeted"] = good_scorables
            context["scorables_baseline"] = bad_scorables

            # Also fill the agent's own output_key with a summary
            correct_pct = len(good_scorables) / len(all_scorables) * 100 if all_scorables else 0
            high_quality_pct = self.quality_metrics["high_quality_reasoning"] / self.quality_metrics["total_examples"] * 100 if self.quality_metrics["total_examples"] else 0
            
            summary = {
                "title": getattr(self, "name", "visual_introspection"),
                "strategy": self.strategy,
                "model_name": self.model_name,
                "num_examples": len(all_scorables),
                "num_correct": len(good_scorables),
                "num_incorrect": len(bad_scorables),
                "accuracy": correct_pct,
                "high_quality_reasoning": high_quality_pct,
                "jsonl_path": str(self.jsonl_path),
                "vis_dir": str(self.vis_dir),
                "quality_metrics": self.quality_metrics,
                # A tiny peek at the first parsed record for debugging
                "first_example": jsonl_records[0] if jsonl_records else None,
            }
            context[self.output_key] = summary

            # -----------------------------------------------------------------
            # 6) Log comprehensive success metrics
            # -----------------------------------------------------------------
            duration = (datetime.now() - start_time).total_seconds()
            log.info("🎉 VisualIntrospectionAgent completed successfully!")
            log.info("📊 COMPREHENSIVE QUALITY REPORT:")
            log.info(f"   ⏱️  Duration: {duration:.2f}s")
            log.info(f"   📈 Final Accuracy: {correct_pct:.1f}% ({len(good_scorables)}/{len(all_scorables)})")
            log.info(f"   🔍 Parsing Success: {self.quality_metrics['successful_parsing']}/{self.quality_metrics['total_examples']} ({self.quality_metrics['successful_parsing']/self.quality_metrics['total_examples']*100:.1f}%)")
            log.info(f"   💭 Reasoning Found: {self.quality_metrics['reasoning_found']}/{self.quality_metrics['total_examples']} ({self.quality_metrics['reasoning_found']/self.quality_metrics['total_examples']*100:.1f}%)")
            log.info(f"   🎯 Final Answer Found: {self.quality_metrics['final_answer_found']}/{self.quality_metrics['total_examples']} ({self.quality_metrics['final_answer_found']/self.quality_metrics['total_examples']*100:.1f}%)")
            log.info(f"   🌟 High-Quality Reasoning: {self.quality_metrics['high_quality_reasoning']}/{self.quality_metrics['total_examples']} ({high_quality_pct:.1f}%)")
            log.info(f"   📊 Reasoning Quality Metrics:")
            log.info(f"      - Avg. steps: {sum(s.meta.get('step_count', 0) for s in all_scorables) / len(all_scorables):.1f}")
            log.info(f"      - Verification rate: {sum(1 for s in all_scorables if s.meta.get('verification_present')) / len(all_scorables):.1%}")
            
            # Log to agent logger
            first_answer = (
                jsonl_records[0]["parsed"]["final_answer"]
                if jsonl_records and jsonl_records[0].get("parsed")
                else ""
            )
            self.logger.log(
                "AgentRanSuccessfully",
                {
                    "agent": getattr(self, "name", "visual_introspection"),
                    "input_key": self.input_key,
                    "output_key": self.output_key,
                    "prompt_snippet": jsonl_records[0]["prompt"][:200]
                    if jsonl_records
                    else "",
                    "response_snippet": first_answer[:300],
                    "num_examples": len(all_scorables),
                    "num_correct": len(good_scorables),
                    "num_incorrect": len(bad_scorables),
                    "accuracy": correct_pct,
                    "high_quality_reasoning": high_quality_pct,
                    "model_name": self.model_name,
                    "jsonl_path": str(self.jsonl_path),
                    "vis_dir": str(self.vis_dir),
                    "duration_seconds": duration,
                    "quality_metrics": self.quality_metrics,
                },
            )

            return context

        except Exception as e:
            # Comprehensive error logging
            duration = (datetime.now() - start_time).total_seconds()
            err_msg = f"{type(e).__name__}: {e}"
            log.error(f"❌ VisualIntrospectionAgent failed after {duration:.2f}s: {err_msg}")
            
            import traceback
            log.error(f"🔍 Error traceback:\n{traceback.format_exc()}")
            
            self.logger.log(
                "AgentFailed",
                {
                    "agent": getattr(self, "name", "visual_introspection"),
                    "error": err_msg,
                    "input_key": self.input_key,
                    "output_key": self.output_key,
                    "context_snapshot": {k: len(str(v)) for k, v in context.items()},
                    "duration_seconds": duration,
                    "quality_metrics": self.quality_metrics,
                },
            )
            
            # Ensure downstream agents don't crash
            context[self.output_key] = {
                "error": err_msg,
                "status": "failed",
                "model_name": self.model_name,
                "jsonl_path": str(self.jsonl_path),
                "vis_dir": str(self.vis_dir),
                "duration_seconds": duration,
                "quality_metrics": self.quality_metrics,
            }
            return context

# ---------------------------------------------------------------------- #
# Enhanced parsing and scorable building functions
# ---------------------------------------------------------------------- #

def extract_number(s: str) -> Optional[str]:
    """
    Extract the last integer or decimal number from a string.
    
    Returns:
        The last numeric substring (e.g. '72', '-3.5') if found,
        otherwise None.
    """
    log.info(f"🔢 Extracting number from: '{s}'")
    
    # Use a non-capturing group so we get the full match
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    
    if nums:
        result = nums[-1]
        log.info(f"✅ Extracted number: '{result}' from '{s}'")
        return result
    else:
        log.info("❌ No number found in string")
        return None


def canonicalize_gsm8k_gold(answer: str) -> str:
    """
    GSM8K usually ends with '#### 72'.
    We'll extract the last number after '####' if present,
    otherwise fall back to the last number in the string.
    If we can't find any number at all, we return the stripped tail.
    """
    log.info(f"🏷️  Canonicalizing gold answer: '{answer}'")
    
    if "####" in answer:
        tail = answer.split("####")[-1]
        log.info(f"📌 Using tail after '####': '{tail}'")
    else:
        tail = answer
        log.info(f"📌 No '####' found, using full answer")

    num = extract_number(tail)
    result = num if num is not None else tail.strip()
    
    log.info(f"✅ Canonicalized gold answer: '{result}'")
    return result


def extract_pred_answer(text: str) -> Optional[str]:
    """More robust numeric extractor for DeepSeek outputs"""
    log.info(f"🎯 Extracting predicted answer from text (length: {len(text)} chars)")

    # 1) Try various final answer patterns
    patterns = [
        r"Final\s*(?:numeric\s*)?answer\s*[:\-]?\s*([^\n]+)",
        r"Answer\s*[:\-]?\s*([^\n]+)",
        r"The\s*answer\s*is\s*([^\n]+)",
        r"=+\s*([^\n]+)\s*=+"
    ]
    
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            line = m.group(1).strip()
            log.info(f"📌 Found answer pattern '{pattern}': '{line}'")
            
            # Extract just the number, ignoring units or explanations
            num_match = re.search(r"([\-]?\d+(?:\.\d+)?)", line)
            if num_match:
                num = num_match.group(1)
                log.info(f"✅ Extracted numeric value: '{num}'")
                return num
    
    # 2) Fallback: last number in the whole text
    num = extract_number(text)
    if num is not None:
        log.info(f"🔄 Extracted predicted answer via fallback: '{num}'")
        return num

    # 3) Nothing found
    log.info("❌ No predicted answer found in text")
    return None

def build_scorable_for_example(
    problem_id: str,
    question: str,
    gold_answer: str,
    raw_text: str,
    *,
    parsed: Dict[str, Any],
    source_label: str = "deepseek-math:7b",
) -> Scorable:
    """Build a Scorable with rich quality metrics for analysis"""
    log.info(f"🏗️  Building scorable for {problem_id}")
    
    gold_canon = canonicalize_gsm8k_gold(gold_answer)
    pred_canon = extract_pred_answer(raw_text)

    is_correct = (pred_canon is not None and gold_canon == pred_canon)
    
    # Calculate quality metrics
    reasoning_length = len(parsed["reasoning"].split()) if parsed["reasoning"] else 0
    step_count = parsed["step_count"]
    verification_present = parsed["verification_present"]
    
    # Reasoning quality score (0-10)
    reasoning_quality = 0
    if step_count >= 3:
        reasoning_quality += 4
    if verification_present:
        reasoning_quality += 3
    if reasoning_length > 50:
        reasoning_quality += 3
    
    log.info(f"📊 Correctness check: pred='{pred_canon}', gold='{gold_canon}', correct={is_correct}")
    log.info(f"📈 Quality metrics: length={reasoning_length}, steps={step_count}, verification={verification_present}, quality={reasoning_quality}")

    text = (
        f"Question:\n{question}\n\n"
        f"Model response:\n{raw_text}\n"
    )

    meta: Dict[str, Any] = {
        "task": "gsm8k",
        "problem_id": problem_id,
        "source": source_label,
        "gold_answer_raw": gold_answer,
        "gold_answer_canonical": gold_canon,
        "pred_answer_raw": raw_text,
        "pred_answer_canonical": pred_canon,
        "is_correct": is_correct,
        # Quality metrics for VisiCalc analysis
        "reasoning_length": reasoning_length,
        "step_count": step_count,
        "verification_present": verification_present,
        "reasoning_quality": reasoning_quality,
        # VisiCalc features - will be populated by downstream agents
        "stability": 0.0,
        "middle_dip": 0.0,
        "std_dev": 0.0,
        "sparsity": 0.0,
        "entropy": 0.0,
        "trend": 0.0,
        "mid_bad_ratio": 0.0,
        "frontier_util": 0.0,
    }

    scorable = Scorable(
        text=text,
        id=problem_id,
        meta=meta,
    )
    
    log.info(f"✅ Scorable built: correct={is_correct}, text_length={len(text)}")
    return scorable