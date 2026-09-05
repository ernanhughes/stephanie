# Codebase Pack: zeromodel

```text
ROOT: C:\Project\stephanie\stephanie\zeromodel
GENERATED_AT_UTC: 2026-07-22T19:30:25.787360+00:00
PART: 1/1
FILES_IN_PART: 8
TOTAL_LINES_IN_PART: 2485
TOTAL_BYTES_UTF8_IN_PART: 83139
MODE: configured include extensions
LINE_NUMBERS: True
MAX_FILE_KB: 400
```

## How to cite this pack in review

Use the stable file ID plus line numbers, for example:

```text
F0007 `services/replay_service.py` L0042-L0068
```

## File Index

| ID | Path | Lang | Lines | KB | SHA256 |
|---|---|---:|---:|---:|---|
| F0001 | `casebook_residual_extractor.py` | python | 143 | 5.3 | `1be01485dd4d` |
| F0002 | `gap_gauge.py` | python | 213 | 6.0 | `6841d6b03923` |
| F0003 | `score_matrix.py` | python | 232 | 7.4 | `d0702a6ff0f1` |
| F0004 | `vpm_builder.py` | python | 65 | 2.4 | `e98bf8d0649c` |
| F0005 | `vpm_controller.py` | python | 623 | 20.8 | `bff5315b3356` |
| F0006 | `vpm_differential_analyzer.py` | python | 74 | 2.9 | `6f4bf1b9da24` |
| F0007 | `vpm_emitter.py` | python | 474 | 17.2 | `a97b073d9f2f` |
| F0008 | `vpm_phos.py` | python | 661 | 19.2 | `48defe4dbff5` |

## Directory Tree

```text
└─ casebook_residual_extractor.py
└─ gap_gauge.py
└─ score_matrix.py
└─ vpm_builder.py
└─ vpm_controller.py
└─ vpm_differential_analyzer.py
└─ vpm_emitter.py
└─ vpm_phos.py
```

## Files


---

## F0001 — `casebook_residual_extractor.py`

```text
FILE_ID: F0001
PATH: casebook_residual_extractor.py
LANGUAGE: python
LINES: 143
BYTES_UTF8: 5472
SHA256: 1be01485dd4da328060db994e6547a213080c9738c701d0d40d58245a57439c7
```

```python
0001 | # stephanie/zeromodel/casebook_residual_extractor.py
0002 | from __future__ import annotations
0003 | 
0004 | import os
0005 | import uuid
0006 | from collections import OrderedDict
0007 | 
0008 | import numpy as np
0009 | import torch
0010 | 
0011 | from stephanie.db import Session
0012 | from stephanie.orm.casebook import CaseBookORM
0013 | from stephanie.orm.skill_filter import SkillFilterORM
0014 | from stephanie.zeromodel.vpm_builder import CaseBookVPMBuilder
0015 | 
0016 | 
0017 | def diff_state_dict(sd_after: OrderedDict, sd_before: OrderedDict) -> OrderedDict:
0018 |     v = OrderedDict()
0019 |     for k, w in sd_after.items():
0020 |         if k in sd_before and w.shape == sd_before[k].shape:
0021 |             v[k] = (w - sd_before[k]).cpu()
0022 |     return v
0023 | 
0024 | class CaseBookResidualExtractor:
0025 |     def __init__(self, session: Session, tokenizer, logger=None):
0026 |         self.session = session
0027 |         self.tokenizer = tokenizer
0028 |         self.logger = logger or (lambda m: print(m))
0029 | 
0030 |     def extract_and_store(
0031 |         self,
0032 |         casebook_name: str,
0033 |         model_before,
0034 |         model_after,
0035 |         weight_sd_before: OrderedDict,
0036 |         weight_sd_after: OrderedDict,
0037 |         domain: str = "general",
0038 |         out_dir: str = "outputs/skills",
0039 |         description: str | None = None,
0040 |         validate: bool = True,
0041 |         num_test_cases: int = 100,
0042 |     ) -> SkillFilterORM:
0043 |         os.makedirs(out_dir, exist_ok=True)
0044 | 
0045 |         cb: CaseBookORM = (
0046 |             self.session.query(CaseBookORM).filter_by(name=casebook_name).one()
0047 |         )
0048 |         # 1) Weight delta
0049 |         v_weight = diff_state_dict(weight_sd_after, weight_sd_before)
0050 |         weight_path = os.path.join(out_dir, f"{casebook_name}_delta_{uuid.uuid4().hex}.pt")
0051 |         torch.save(v_weight, weight_path)
0052 |         weight_size_mb = os.path.getsize(weight_path) / (1024**2)
0053 | 
0054 |         # 2) VPM before/after
0055 |         builder = CaseBookVPMBuilder(self.tokenizer, metrics=["sicql", "ebt", "llm"])
0056 |         vpm_before = builder.build(cb, model_before)
0057 |         vpm_after  = builder.build(cb, model_after)
0058 | 
0059 |         residual = (vpm_after.astype(np.float32) - vpm_before.astype(np.float32))
0060 |         # Normalize residual for storage/preview
0061 |         res_norm = (residual - residual.min()) / (residual.ptp() + 1e-9)
0062 | 
0063 |         res_npy = os.path.join(out_dir, f"{casebook_name}_residual_{uuid.uuid4().hex}.npy")
0064 |         np.save(res_npy, res_norm)
0065 |         res_png = os.path.join(out_dir, f"{casebook_name}_residual_{uuid.uuid4().hex}.png")
0066 |         builder.save_image(res_norm, res_png, title=f"Residual {casebook_name}")
0067 | 
0068 |         # 3) Validation (alignment)
0069 |         alignment_score = None
0070 |         if validate:
0071 |             alignment_score = self._validate_skill_alignment(
0072 |                 v_weight, res_norm, cb, model_before, builder, num_test_cases
0073 |             )
0074 | 
0075 |         # 4) Store SkillFilter
0076 |         sf = SkillFilterORM(
0077 |             id=uuid.uuid4().hex[:32],
0078 |             casebook_id=cb.id,
0079 |             domain=domain,
0080 |             description=description or f"Skill filter extracted from {casebook_name}",
0081 |             weight_delta_path=weight_path,
0082 |             weight_size_mb=weight_size_mb,
0083 |             vpm_residual_path=res_npy,
0084 |             vpm_preview_path=res_png,
0085 |             alignment_score=alignment_score,
0086 |             improvement_score=None,
0087 |             stability_score=None,
0088 |             compatible_domains=None,
0089 |             negative_interactions=None,
0090 |         )
0091 |         self.session.add(sf)
0092 |         self.session.commit()
0093 |         self.logger(f"Saved SkillFilter {sf.id} for casebook {casebook_name}")
0094 |         return sf
0095 | 
0096 |     # --- Validation: weight delta reproduces VPM residual on subset ---
0097 |     def _validate_skill_alignment(
0098 |         self,
0099 |         v_weight: OrderedDict,
0100 |         residual_vpm: np.ndarray,
0101 |         casebook: CaseBookORM,
0102 |         base_model,
0103 |         vpm_builder: CaseBookVPMBuilder,
0104 |         num_cases: int = 100,
0105 |     ) -> float:
0106 |         # 1) apply weight delta to cloned model
0107 |         test_model = self._apply_weight_delta(base_model, v_weight)
0108 | 
0109 |         # 2) subset of cases
0110 |         subs = casebook.cases[:num_cases] if len(casebook.cases) > num_cases else casebook.cases
0111 |         vpm_test = vpm_builder.build_subset(subs, test_model)
0112 |         vpm_base = vpm_builder.build_subset(subs, base_model)
0113 | 
0114 |         actual = vpm_test - vpm_base
0115 |         # normalize both
0116 |         actual_norm = (actual - actual.min()) / (actual.ptp() + 1e-9)
0117 | 
0118 |         # match shapes if needed
0119 |         if actual_norm.shape != residual_vpm.shape:
0120 |             residual_vpm = self._resize_to(residual_vpm, actual_norm.shape)
0121 | 
0122 |         expected_norm = (residual_vpm - residual_vpm.min()) / (residual_vpm.ptp() + 1e-9)
0123 |         alignment = 1.0 - float(np.mean(np.abs(actual_norm - expected_norm)))
0124 |         return max(0.0, min(1.0, alignment))
0125 | 
0126 |     def _apply_weight_delta(self, model, v_weight: OrderedDict, alpha: float = 1.0):
0127 |         import copy
0128 |         cloned = copy.deepcopy(model)
0129 |         params = dict(cloned.named_parameters())
0130 |         for name, delta in v_weight.items():
0131 |             if name in params:
0132 |                 params[name].data = params[name].data + alpha * delta.to(params[name].device).to(params[name].dtype)
0133 |         return cloned
0134 | 
0135 |     def _resize_to(self, arr: np.ndarray, target_shape: tuple) -> np.ndarray:
0136 |         from scipy.ndimage import zoom
0137 |         if arr.shape == target_shape:
0138 |             return arr
0139 |         if arr.ndim == 1:
0140 |             scale = target_shape[0] / arr.shape[0]
0141 |             return zoom(arr, scale, order=1)
0142 |         scales = tuple(ts / s for ts, s in zip(target_shape, arr.shape))
0143 |         return zoom(arr, scales, order=1)
```


---

## F0002 — `gap_gauge.py`

```text
FILE_ID: F0002
PATH: gap_gauge.py
LANGUAGE: python
LINES: 213
BYTES_UTF8: 6130
SHA256: 6841d6b03923c2f9e77a611eca00764f52f4637835f6ecd5a8693269789b3815
```

```python
0001 | # zeromodel/gap_gauge.py
0002 | from __future__ import annotations
0003 | 
0004 | from dataclasses import dataclass
0005 | from typing import Any, Dict, List, Optional, Tuple
0006 | 
0007 | import numpy as np
0008 | from zeromodel.pipeline.organizer.top_left import TopLeft
0009 | 
0010 | from stephanie.scoring.metrics.frontier_lens import (FrontierLensEpisode,
0011 |                                                      episode_features)
0012 | 
0013 | # If you’ve moved VisiCalcReport into Stephanie core, you can also
0014 | # import the report-level gauge from there. For now, assume a pure
0015 | # numpy/episode-based path inside ZeroModel.
0016 | 
0017 | 
0018 | @dataclass
0019 | class GapGaugeResult:
0020 |     """
0021 |     Unified comparison between a baseline and a targeted run.
0022 | 
0023 |     Combines:
0024 |       - VisiCalc-style numeric comparison (episode features)
0025 |       - TopLeft-style visual comparison (canonicalized VPMs)
0026 |     """
0027 |     episode_id: str
0028 | 
0029 |     # VisiCalc episode features
0030 |     baseline_feats: np.ndarray
0031 |     target_feats: np.ndarray
0032 |     diff_feats: np.ndarray
0033 |     feat_names: List[str]
0034 | 
0035 |     # Simple scalar gauges
0036 |     baseline_quality: float
0037 |     target_quality: float
0038 |     diff_quality: float
0039 | 
0040 |     # Visual comparison from TopLeft
0041 |     topleft_improvement_ratio: float
0042 |     topleft_gain: float
0043 |     topleft_loss: float
0044 | 
0045 |     # Optional debug artifacts
0046 |     meta: Dict[str, Any]
0047 | 
0048 | 
0049 | def _episode_from_scores(
0050 |     episode_id: str,
0051 |     scores: np.ndarray,
0052 |     metric_names: List[str],
0053 |     item_ids: List[str],
0054 | ) -> Tuple[np.ndarray, List[str], float]:
0055 |     """
0056 |     Helper: turn a score matrix into:
0057 |       - episode feature vector
0058 |       - feature names
0059 |       - a simple scalar 'quality' (mean frontier-ish)
0060 |     """
0061 |     episode = FrontierLensEpisode(
0062 |         episode_id=episode_id,
0063 |         scores=scores,
0064 |         metric_names=list(metric_names),
0065 |         item_ids=list(item_ids),
0066 |         meta={},
0067 |     )
0068 |     feats, feat_names = episode_features(episode)
0069 | 
0070 |     # Very simple scalar: mean over all normalized metrics
0071 |     # (You can later replace this with a learned critic.)
0072 |     quality = float(feats.mean())
0073 | 
0074 |     return feats.astype(np.float32), feat_names, quality
0075 | 
0076 | 
0077 | def _compare_vpms_topleft(
0078 |     vpm_base: np.ndarray,
0079 |     vpm_tgt: np.ndarray,
0080 |     *,
0081 |     metric_mode: str = "luminance",
0082 |     iterations: int = 5,
0083 |     push_corner: str = "tl",
0084 |     monotone_push: bool = True,
0085 |     stretch: bool = True,
0086 | ) -> Dict[str, Any]:
0087 |     """
0088 |     Canonicalize baseline & target VPMs with TopLeft, then compute a simple
0089 |     'is target visually better than baseline?' metric.
0090 |     """
0091 |     stage = TopLeft(
0092 |         metric_mode=metric_mode,
0093 |         iterations=iterations,
0094 |         push_corner=push_corner,
0095 |         monotone_push=monotone_push,
0096 |         stretch=stretch,
0097 |     )
0098 | 
0099 |     tl_base, meta_base = stage.process(vpm_base)
0100 |     tl_tgt, meta_tgt = stage.process(vpm_tgt)
0101 | 
0102 |     tl_base = tl_base.astype(np.float32)
0103 |     tl_tgt = tl_tgt.astype(np.float32)
0104 |     assert tl_base.shape == tl_tgt.shape, "Base/Target shapes must match after TopLeft"
0105 | 
0106 |     diff = tl_tgt - tl_base
0107 | 
0108 |     gain = float(np.sum(np.clip(diff, 0.0, None)))
0109 |     loss = float(np.sum(np.clip(-diff, 0.0, None)))
0110 |     total = gain + loss + 1e-8
0111 | 
0112 |     improvement_ratio = gain / total
0113 | 
0114 |     return {
0115 |         "topleft_base": tl_base,
0116 |         "topleft_tgt": tl_tgt,
0117 |         "topleft_diff": diff,
0118 |         "gain": gain,
0119 |         "loss": loss,
0120 |         "improvement_ratio": improvement_ratio,
0121 |         "meta_base": meta_base,
0122 |         "meta_tgt": meta_tgt,
0123 |     }
0124 | 
0125 | 
0126 | def compute_gap_gauge(
0127 |     *,
0128 |     episode_id: str,
0129 |     scores_baseline: np.ndarray,   # shape (N, M)
0130 |     scores_target: np.ndarray,     # shape (N, M)
0131 |     metric_names: List[str],
0132 |     item_ids: List[str],
0133 |     vpm_baseline: Optional[np.ndarray] = None,  # for TopLeft; can be None
0134 |     vpm_target: Optional[np.ndarray] = None,
0135 |     topleft_cfg: Optional[Dict[str, Any]] = None,
0136 | ) -> GapGaugeResult:
0137 |     """
0138 |     Core entry point: given numeric scores + optional VPMs, produce a GapGaugeResult.
0139 |     """
0140 |     scores_baseline = np.asarray(scores_baseline, dtype=np.float32)
0141 |     scores_target = np.asarray(scores_target, dtype=np.float32)
0142 |     assert scores_baseline.shape == scores_target.shape, "Baseline/Target scores must match shape"
0143 | 
0144 |     # 1) Numeric side: VisiCalc episodes
0145 |     base_feats, feat_names, base_q = _episode_from_scores(
0146 |         f"{episode_id}:baseline",
0147 |         scores_baseline,
0148 |         metric_names,
0149 |         item_ids,
0150 |     )
0151 |     tgt_feats, feat_names2, tgt_q = _episode_from_scores(
0152 |         f"{episode_id}:target",
0153 |         scores_target,
0154 |         metric_names,
0155 |         item_ids,
0156 |     )
0157 |     assert feat_names == feat_names2, "Feature name mismatch between baseline and target"
0158 | 
0159 |     diff_feats = tgt_feats - base_feats
0160 |     diff_q = float(diff_feats.mean())
0161 | 
0162 |     # 2) Visual side: TopLeft canonicalization (optional)
0163 |     topleft_improvement_ratio = 0.5
0164 |     topleft_gain = 0.0
0165 |     topleft_loss = 0.0
0166 |     tl_meta: Dict[str, Any] = {}
0167 | 
0168 |     if vpm_baseline is not None and vpm_target is not None:
0169 |         cfg = dict(
0170 |             metric_mode="luminance",
0171 |             iterations=5,
0172 |             push_corner="tl",
0173 |             monotone_push=True,
0174 |             stretch=True,
0175 |         )
0176 |         if topleft_cfg:
0177 |             cfg.update(topleft_cfg)
0178 | 
0179 |         tl = _compare_vpms_topleft(
0180 |             vpm_base=vpm_baseline,
0181 |             vpm_tgt=vpm_target,
0182 |             **cfg,
0183 |         )
0184 |         topleft_improvement_ratio = tl["improvement_ratio"]
0185 |         topleft_gain = tl["gain"]
0186 |         topleft_loss = tl["loss"]
0187 |         tl_meta = {
0188 |             "meta_base": tl["meta_base"],
0189 |             "meta_tgt": tl["meta_tgt"],
0190 |             "shape": tl["topleft_base"].shape,
0191 |         }
0192 | 
0193 |     meta: Dict[str, Any] = {
0194 |         "episode_id": episode_id,
0195 |         "metric_names": list(metric_names),
0196 |         "num_items": len(item_ids),
0197 |         "topleft": tl_meta,
0198 |     }
0199 | 
0200 |     return GapGaugeResult(
0201 |         episode_id=episode_id,
0202 |         baseline_feats=base_feats,
0203 |         target_feats=tgt_feats,
0204 |         diff_feats=diff_feats,
0205 |         feat_names=feat_names,
0206 |         baseline_quality=base_q,
0207 |         target_quality=tgt_q,
0208 |         diff_quality=diff_q,
0209 |         topleft_improvement_ratio=topleft_improvement_ratio,
0210 |         topleft_gain=topleft_gain,
0211 |         topleft_loss=topleft_loss,
0212 |         meta=meta,
0213 |     )
```


---

## F0003 — `score_matrix.py`

```text
FILE_ID: F0003
PATH: score_matrix.py
LANGUAGE: python
LINES: 232
BYTES_UTF8: 7579
SHA256: d0702a6ff0f1c6a91637d8a6d48882f02c61cfa88b6a362efa7cd475581d5002
```

```python
0001 | # stephanie/eval/score_matrix.py
0002 | from __future__ import annotations
0003 | 
0004 | import json
0005 | from typing import Dict, Iterable, List, Tuple
0006 | 
0007 | import numpy as np
0008 | import pandas as pd
0009 | from tqdm import tqdm
0010 | 
0011 | from stephanie.constants import GOAL, GOAL_TEXT
0012 | from stephanie.scoring.scorable import Scorable, ScorableFactory, ScorableType
0013 | 
0014 | # ---------------------------
0015 | # Helpers
0016 | # ---------------------------
0017 | 
0018 | 
0019 | def _make_scorable(text: str, idx: int):
0020 |     """
0021 |     Prefer ScorableFactory if available in your env, else fallback to a tiny shim.
0022 |     """
0023 |     if ScorableFactory and ScorableType:
0024 |         # minimal dict accepted by your factory
0025 |         return ScorableFactory.from_dict(
0026 |             {"text": text, "id": f"resp_{idx}"}, ScorableType.DOCUMENT
0027 |         )
0028 |     return Scorable(id=f"resp_{idx}", text=text, type=ScorableType.CUSTOM)
0029 | 
0030 | 
0031 | def _robust_minmax(series: pd.Series, lo=10.0, hi=90.0) -> pd.Series:
0032 |     """
0033 |     Scale to [0,1] using robust percentiles; clamp outliers.
0034 |     Works even if the two models are on different raw scales.
0035 |     """
0036 |     s = series.astype(float)
0037 |     p_lo = np.nanpercentile(s, lo) if s.notna().any() else 0.0
0038 |     p_hi = np.nanpercentile(s, hi) if s.notna().any() else 1.0
0039 |     if abs(p_hi - p_lo) < 1e-12:
0040 |         p_hi = p_lo + 1.0
0041 |     s_clamped = s.clip(lower=p_lo, upper=p_hi)
0042 |     return (s_clamped - p_lo) / (p_hi - p_lo)
0043 | 
0044 | 
0045 | def _corr_safe(a: pd.Series, b: pd.Series) -> Tuple[float, float, float]:
0046 |     """
0047 |     Pearson / Spearman / Kendall τ with NaN safety.
0048 |     """
0049 |     a, b = a.astype(float), b.astype(float)
0050 |     if a.notna().sum() < 3 or b.notna().sum() < 3:
0051 |         return float("nan"), float("nan"), float("nan")
0052 |     pearson = float(a.corr(b, method="pearson"))
0053 |     spearman = float(a.corr(b, method="spearman"))
0054 |     kendall = float(a.corr(b, method="kendall"))
0055 |     return pearson, spearman, kendall
0056 | 
0057 | 
0058 | # ---------------------------
0059 | # Core
0060 | # ---------------------------
0061 | 
0062 | 
0063 | def build_score_matrix(
0064 |     *,
0065 |     responses: Iterable[str],
0066 |     goal_text: str,
0067 |     dimensions: List[str],
0068 |     scorers: Dict[
0069 |         str, object
0070 |     ],  # e.g. {"hrm": hrm_scorer, "tiny": tiny_scorer}
0071 |     memory=None,
0072 |     logger=None,
0073 |     max_n: int = 500,
0074 |     show_progress: bool = True,
0075 | ) -> Tuple[pd.DataFrame, Dict]:
0076 |     """
0077 |     Returns:
0078 |       df: rows = response_id, columns = MultiIndex(model, dimension), values = scores
0079 |       metrics: dict with per-dimension agreement + distribution summaries
0080 |     """
0081 |     # 1) slice to 500 and make scorables
0082 |     responses = list(responses)[:max_n]
0083 |     scorables = [_make_scorable(r, i) for i, r in enumerate(responses)]
0084 | 
0085 |     # 2) shared context
0086 |     context = {GOAL: {GOAL_TEXT: goal_text, "id": "eval_goal"}}
0087 | 
0088 |     # 3) score loop
0089 |     cols = []
0090 |     data = []
0091 |     if show_progress:
0092 |         pbar = tqdm(
0093 |             total=len(scorables) * max(1, len(scorers)), desc="Scoring"
0094 |         )
0095 |     else:
0096 |         pbar = None
0097 | 
0098 |     # Prepare a container: dict[row_id][(model, dim)] = score
0099 |     row_dicts: List[Dict[Tuple[str, str], float]] = [dict() for _ in scorables]
0100 | 
0101 |     for model_name, scorer in scorers.items():
0102 |         for i, sc in enumerate(scorables):
0103 |             try:
0104 |                 bundle = scorer.score(context, sc, dimensions)
0105 |                 # bundle.results: dict[dimension] -> ScoreResult
0106 |                 for dim in dimensions:
0107 |                     sr = bundle.results.get(dim)
0108 |                     if sr is None:
0109 |                         continue
0110 |                     row_dicts[i][(model_name, dim)] = float(sr.score)
0111 |             except Exception as e:
0112 |                 if logger:
0113 |                     logger.log(
0114 |                         "EvalScorerError",
0115 |                         {
0116 |                             "model": model_name,
0117 |                             "dim_set": dimensions,
0118 |                             "idx": i,
0119 |                             "error": str(e),
0120 |                         },
0121 |                     )
0122 |             if pbar:
0123 |                 pbar.update(1)
0124 | 
0125 |     if pbar:
0126 |         pbar.close()
0127 | 
0128 |     # 4) assemble DataFrame
0129 |     all_cols = sorted({k for row in row_dicts for k in row.keys()})
0130 |     df = pd.DataFrame(
0131 |         [{c: row.get(c, np.nan) for c in all_cols} for row in row_dicts]
0132 |     )
0133 |     df.columns = pd.MultiIndex.from_tuples(
0134 |         df.columns, names=["model", "dimension"]
0135 |     )
0136 |     df.index = [f"resp_{i}" for i in range(len(df))]
0137 | 
0138 |     # 5) compute metrics
0139 |     metrics = {
0140 |         "per_model_distribution": {},
0141 |         "agreement": {},
0142 |         "inter_dim_corr": {},
0143 |     }
0144 | 
0145 |     # 5a) per-model distribution stats per dimension
0146 |     for model_name in sorted({m for m, _ in df.columns}):
0147 |         sub = (
0148 |             df[model_name]
0149 |             if model_name in df.columns.get_level_values(0)
0150 |             else None
0151 |         )
0152 |         if sub is None:
0153 |             continue
0154 |         mstats = {}
0155 |         for dim in dimensions:
0156 |             if dim not in sub.columns:
0157 |                 continue
0158 |             s = sub[dim].astype(float)
0159 |             mstats[dim] = {
0160 |                 "count": int(s.notna().sum()),
0161 |                 "mean": float(np.nanmean(s)),
0162 |                 "std": float(np.nanstd(s)),
0163 |                 "min": float(np.nanmin(s))
0164 |                 if s.notna().any()
0165 |                 else float("nan"),
0166 |                 "p25": float(np.nanpercentile(s, 25))
0167 |                 if s.notna().any()
0168 |                 else float("nan"),
0169 |                 "p50": float(np.nanpercentile(s, 50))
0170 |                 if s.notna().any()
0171 |                 else float("nan"),
0172 |                 "p75": float(np.nanpercentile(s, 75))
0173 |                 if s.notna().any()
0174 |                 else float("nan"),
0175 |                 "max": float(np.nanmax(s))
0176 |                 if s.notna().any()
0177 |                 else float("nan"),
0178 |             }
0179 |         metrics["per_model_distribution"][model_name] = mstats
0180 | 
0181 |     # 5b) HRM vs Tiny agreement (normalize each to 0..1 via robust minmax)
0182 |     if ("hrm" in df.columns.get_level_values(0)) and (
0183 |         "tiny" in df.columns.get_level_values(0)
0184 |     ):
0185 |         agr = {}
0186 |         for dim in dimensions:
0187 |             if (("hrm", dim) not in df.columns) or (
0188 |                 ("tiny", dim) not in df.columns
0189 |             ):
0190 |                 continue
0191 |             a_raw = df[("hrm", dim)]
0192 |             b_raw = df[("tiny", dim)]
0193 |             a = _robust_minmax(a_raw)
0194 |             b = _robust_minmax(b_raw)
0195 | 
0196 |             pearson, spearman, kendall = _corr_safe(a, b)
0197 |             mae = float(np.nanmean(np.abs(a - b)))
0198 |             agr[dim] = {
0199 |                 "pearson_r": pearson,
0200 |                 "spearman_rho": spearman,
0201 |                 "kendall_tau": kendall,
0202 |                 "mae_norm01": mae,
0203 |             }
0204 |         metrics["agreement"]["hrm_vs_tiny"] = agr
0205 | 
0206 |     # 5c) inter-dimension correlation matrices (per model)
0207 |     for model_name in sorted({m for m, _ in df.columns}):
0208 |         try:
0209 |             sub = df[model_name].astype(float)
0210 |             # Only keep columns with non-NaN variation
0211 |             keep = [
0212 |                 c
0213 |                 for c in sub.columns
0214 |                 if sub[c].notna().sum() > 3 and sub[c].std(skipna=True) > 1e-9
0215 |             ]
0216 |             if keep:
0217 |                 metrics["inter_dim_corr"][model_name] = (
0218 |                     sub[keep].corr(method="spearman").to_dict()
0219 |                 )
0220 |         except Exception:
0221 |             pass
0222 | 
0223 |     return df, metrics
0224 | 
0225 | 
0226 | def save_score_matrix(df: pd.DataFrame, metrics: Dict, *, out_prefix: str):
0227 |     """
0228 |     Save both artifacts. MultiIndex is preserved in CSV; metrics in JSON.
0229 |     """
0230 |     df.to_csv(f"{out_prefix}_scores.csv", index=True)
0231 |     with open(f"{out_prefix}_metrics.json", "w", encoding="utf-8") as f:
0232 |         json.dump(metrics, f, ensure_ascii=False, indent=2)
```


---

## F0004 — `vpm_builder.py`

```text
FILE_ID: F0004
PATH: vpm_builder.py
LANGUAGE: python
LINES: 65
BYTES_UTF8: 2478
SHA256: e98bf8d0649c7d4744ee6d11e3f2ae133f805f1ab480a8f543bfc589b492d49c
```

```python
0001 | # stephanie/zeromodel/vpm_builder.py
0002 | from __future__ import annotations
0003 | 
0004 | import os
0005 | from dataclasses import dataclass
0006 | from typing import List
0007 | 
0008 | import numpy as np
0009 | from PIL import Image, ImageDraw, ImageOps  # pillow
0010 | 
0011 | 
0012 | @dataclass
0013 | class VPMConfig:
0014 |     metrics: List[str] = None
0015 |     image_scale: int = 2  # upsample for visibility
0016 | 
0017 | class CaseBookVPMBuilder:
0018 |     def __init__(self, tokenizer, metrics: List[str] = None, cfg: VPMConfig | None = None):
0019 |         self.tokenizer = tokenizer
0020 |         self.metrics = metrics or ["sicql", "ebt", "llm"]
0021 |         self.cfg = cfg or VPMConfig(metrics=self.metrics)
0022 | 
0023 |     def build(self, casebook, model) -> np.ndarray:
0024 |         """Return 2D float image of shape (num_cases, num_metrics) in [0,1]."""
0025 |         rows = []
0026 |         for case in casebook.cases:
0027 |             rows.append(self._scores_for_case(case, model))
0028 |         arr = np.array(rows, dtype=np.float32)
0029 |         return self._normalize(arr)
0030 | 
0031 |     def build_subset(self, cases, model) -> np.ndarray:
0032 |         rows = []
0033 |         for case in cases:
0034 |             rows.append(self._scores_for_case(case, model))
0035 |         arr = np.array(rows, dtype=np.float32)
0036 |         return self._normalize(arr)
0037 | 
0038 |     def _scores_for_case(self, case, model) -> List[float]:
0039 |         # Replace with real calls into your scoring stack
0040 |         # Placeholder: produce [sicql, ebt, llm] in [0,1]
0041 |         sicql = getattr(case, "sicql_score", 0.5)
0042 |         ebt   = getattr(case, "ebt_score", 0.5)
0043 |         llm   = getattr(case, "llm_score", 0.5)
0044 |         return [float(sicql), float(ebt), float(llm)]
0045 | 
0046 |     def _normalize(self, arr: np.ndarray) -> np.ndarray:
0047 |         if arr.size == 0:
0048 |             return arr
0049 |         mn, mx = float(arr.min()), float(arr.max())
0050 |         if mx <= mn + 1e-9:
0051 |             return np.zeros_like(arr, dtype=np.float32)
0052 |         return (arr - mn) / (mx - mn + 1e-9)
0053 | 
0054 |     def save_image(self, vpm: np.ndarray, path: str, title: str | None = None) -> None:
0055 |         os.makedirs(os.path.dirname(path), exist_ok=True)
0056 |         # Scale and convert to image
0057 |         vpm_uint8 = (np.clip(vpm, 0, 1) * 255).astype(np.uint8)
0058 |         img = Image.fromarray(vpm_uint8, mode="L")
0059 |         if self.cfg.image_scale != 1:
0060 |             img = img.resize((img.width * self.cfg.image_scale, img.height * self.cfg.image_scale), Image.NEAREST)
0061 |         if title:
0062 |             img = ImageOps.expand(img, border=20, fill="white")
0063 |             draw = ImageDraw.Draw(img)
0064 |             draw.text((10, 5), title, fill=0)
0065 |         img.save(path)
```


---

## F0005 — `vpm_controller.py`

```text
FILE_ID: F0005
PATH: vpm_controller.py
LANGUAGE: python
LINES: 623
BYTES_UTF8: 21271
SHA256: bff5315b335647f1eda36aa32efee7511b889bae0cce2c1f561271a78481103c
```

```python
0001 | # stephanie/zeromodel/vpm_controller.py
0002 | # VPM Controller — trend-aware, goal-aware, bandit-ready control loop
0003 | from __future__ import annotations
0004 | 
0005 | import json
0006 | import statistics as stats
0007 | import time
0008 | from dataclasses import dataclass, field
0009 | from enum import Enum, auto
0010 | from pathlib import Path
0011 | from typing import Any, Callable, Dict, List, Optional, Tuple
0012 | 
0013 | # ========= public API =========
0014 | 
0015 | 
0016 | class Signal(Enum):
0017 |     EDIT = auto()  # apply local, minimal diffs
0018 |     RESAMPLE = auto()  # rerun with new exemplars / different seeds
0019 |     ESCALATE = auto()  # escalate to stronger model / human checkpoint
0020 |     STOP = auto()  # stop improving (stable & above thresholds)
0021 |     SPINOFF = auto()  # fork dropped/novel content to a new artifact
0022 |     HOLD = auto()  # hold state (cooldown / wait for external event)
0023 | 
0024 | 
0025 | @dataclass
0026 | class Thresholds:
0027 |     mins: Dict[str, float]  # {"coverage":0.8, ...}
0028 |     stop_margin: float = 0.02  # extra margin to declare STOP
0029 |     edit_margin: float = 0.01  # tolerance before EDIT re-triggers
0030 | 
0031 | 
0032 | @dataclass
0033 | class Policy:
0034 |     # windows & smoothing
0035 |     window: int = 5
0036 |     ema_alpha: float = 0.4
0037 |     edit_margin: float = 0.05  # <-- add this default
0038 |     patience: int = 3
0039 |     escalate_after: int = 2
0040 |     # oscillation & cooldowns
0041 |     oscillation_window: int = 6
0042 |     oscillation_threshold: int = 3  # direction flips to flag oscillation
0043 |     cooldown_steps: int = 1  # HOLD after RESAMPLE/ESCALATE to avoid thrash
0044 |     # novelty → spinoff
0045 |     spinoff_dim: str = "novelty"
0046 |     stickiness_dim: str = "stickiness"
0047 |     spinoff_gate: Tuple[float, float] = (
0048 |         0.75,
0049 |         0.45,
0050 |     )  # (novelty>=, stickiness<=)
0051 |     # regression & outlier guards
0052 |     max_regressions: int = 2
0053 |     zscore_clip_dims: List[str] = field(
0054 |         default_factory=lambda: [
0055 |             "coverage",
0056 |             "coherence",
0057 |             "correctness",
0058 |             "tests_pass_rate",
0059 |         ]
0060 |     )
0061 |     zscore_clip_sigma: float = 3.5
0062 |     # local vs global gaps
0063 |     local_gap_dims: List[str] = field(
0064 |         default_factory=lambda: [
0065 |             "citation_support",
0066 |             "entity_consistency",
0067 |             "lint_clean",
0068 |             "type_safe",
0069 |         ]
0070 |     )
0071 |     # action cap
0072 |     max_steps: int = 50
0073 |     # goal awareness (optional; controller works without goals too)
0074 |     goal_kind: Optional[str] = None  # "text" or "code"
0075 |     goal_name: Optional[str] = None  # e.g., "academic_summary"
0076 |     goal_min_score: float = 0.75
0077 |     goal_allow_unmet: int = 0
0078 | 
0079 | 
0080 | @dataclass
0081 | class VPMRow:
0082 |     unit: str  # e.g., "pkg.impl:l2_normalize" or "text:Section"
0083 |     kind: str  # "code" or "text"
0084 |     timestamp: float  # epoch seconds
0085 |     dims: Dict[str, float]  # metric → value
0086 |     step_idx: Optional[int] = None
0087 |     meta: Dict[str, Any] = field(default_factory=dict)
0088 | 
0089 | 
0090 | @dataclass
0091 | class Decision:
0092 |     signal: Signal
0093 |     reason: str
0094 |     params: Dict[str, Any] = field(default_factory=dict)
0095 |     snapshot: Dict[str, Any] = field(default_factory=dict)
0096 | 
0097 | 
0098 | # ========= controller =========
0099 | 
0100 | 
0101 | class VPMController:
0102 |     """
0103 |     Goal- and trend-aware controller that:
0104 |       - gates on thresholds with hysteresis,
0105 |       - smooths noise and guards outliers,
0106 |       - detects stagnation, regressions, oscillations,
0107 |       - triggers EDIT / RESAMPLE / ESCALATE / STOP / SPINOFF / HOLD,
0108 |       - optionally consults a goal score (via injected scorer),
0109 |       - integrates with a bandit for exemplar routing,
0110 |       - persists state and accepts simple dicts (add_vpm_row) for compatibility.
0111 |     """
0112 | 
0113 |     def __init__(
0114 |         self,
0115 |         thresholds_code: Thresholds,
0116 |         thresholds_text: Thresholds,
0117 |         policy: Policy = Policy(),
0118 |         *,
0119 |         bandit_choose: Optional[Callable[[List[str]], str]] = None,
0120 |         bandit_update: Optional[Callable[[str, float], None]] = None,
0121 |         logger: Optional[Callable[[str, Dict[str, Any]], None]] = None,
0122 |         goal_scorer: Optional[
0123 |             Callable[[str, str, Dict[str, float]], Dict[str, Any]]
0124 |         ] = None,
0125 |         state_path: Optional[str] = None,
0126 |     ):
0127 |         self.thr_code = thresholds_code
0128 |         self.thr_text = thresholds_text
0129 |         self.p = policy
0130 |         self.bandit_choose = bandit_choose
0131 |         self.bandit_update = bandit_update
0132 |         self.log = logger or (lambda ev, d: None)
0133 |         self.goal_scorer = goal_scorer
0134 |         self.history: Dict[str, List[VPMRow]] = {}  # unit → rows
0135 |         self.resample_counts: Dict[str, int] = {}  # unit → count
0136 |         self.cooldown_until_step: Dict[
0137 |             str, int
0138 |         ] = {}  # unit → step_idx boundary
0139 |         self.last_signal: Dict[str, Signal] = {}  # unit → last signal
0140 |         self.osc_dir_hist: Dict[str, List[int]] = {}  # unit → [+1/-1] changes
0141 |         self.state_path = Path(state_path) if state_path else None
0142 |         self._load_state()
0143 | 
0144 |     # ---- compatibility entrypoint (used by orchestrator) ----
0145 |     def add_vpm_row(self, vpm_row: Dict[str, Any], unit: str) -> Decision:
0146 |         """
0147 |         Accepts a simple dict row (as emitted by improvers) and a unit id.
0148 |         Infers kind from available dims.
0149 |         """
0150 |         kind = "code" if "tests_pass_rate" in vpm_row else "text"
0151 |         row = VPMRow(
0152 |             unit=unit,
0153 |             kind=kind,
0154 |             timestamp=time.time(),
0155 |             dims={
0156 |                 k: float(vpm_row[k])
0157 |                 for k in vpm_row
0158 |                 if isinstance(vpm_row[k], (int, float))
0159 |             },
0160 |             step_idx=vpm_row.get("step_idx"),
0161 |             meta=vpm_row if isinstance(vpm_row, dict) else {},
0162 |         )
0163 |         return self.add(row)
0164 | 
0165 |     # ---- primary entrypoint ----
0166 |     def add(
0167 |         self, row: VPMRow, *, candidate_exemplars: Optional[List[str]] = None
0168 |     ) -> Decision:
0169 |         # append & clip history
0170 |         h = self.history.setdefault(row.unit, [])
0171 |         h.append(self._clipped(row))
0172 |         if len(h) > 100:
0173 |             self.history[row.unit] = h[-100:]
0174 | 
0175 |         thr = self._thresholds_for(row.kind)
0176 |         window = h[-self.p.window :] if h else h
0177 |         trend = self._trend(window)
0178 |         self._track_oscillation(row.unit, trend)
0179 | 
0180 |         # 0) max-steps stop
0181 |         if row.meta.get("total_steps", row.step_idx or 0) >= self.p.max_steps:
0182 |             return self._decide(row, Signal.STOP, "Max steps reached", {})
0183 | 
0184 |         # cooldown after disruptive actions
0185 |         if self._in_cooldown(row.unit, row.step_idx):
0186 |             return self._decide(
0187 |                 row,
0188 |                 Signal.HOLD,
0189 |                 "Cooldown",
0190 |                 {"until_step": self.cooldown_until_step[row.unit]},
0191 |             )
0192 | 
0193 |         # 1) STOP if stable above thresholds (hysteresis)
0194 |         if self._stable_above(window, thr, margin=thr.stop_margin):
0195 |             # optional goal gate: only STOP if goal score passes (when configured)
0196 |             if self._goal_ok_if_configured(row):
0197 |                 return self._decide(
0198 |                     row, Signal.STOP, "Stable above thresholds (goal OK)", {}
0199 |                 )
0200 |             return self._decide(
0201 |                 row,
0202 |                 Signal.EDIT,
0203 |                 "Stable but goal not met yet",
0204 |                 {"why": "goal"},
0205 |             )
0206 | 
0207 |         # 2) SPINOFF: high novelty + low stickiness
0208 |         if self._should_spinoff(row):
0209 |             return self._decide(
0210 |                 row,
0211 |                 Signal.SPINOFF,
0212 |                 "High novelty with low stickiness",
0213 |                 {
0214 |                     "novelty": row.dims.get(self.p.spinoff_dim),
0215 |                     "stickiness": row.dims.get(self.p.stickiness_dim),
0216 |                 },
0217 |             )
0218 | 
0219 |         # 3) Too many regressions → RESAMPLE
0220 |         if self._regressions(window) > self.p.max_regressions:
0221 |             self._bump_resamples(row.unit)
0222 |             return self._resample(
0223 |                 row, "Too many regressions", candidate_exemplars
0224 |             )
0225 | 
0226 |         # 4) LOCAL vs GLOBAL gaps
0227 |         gaps = self._gaps(row, thr)
0228 |         local_gaps = [g for g in gaps if g in self.p.local_gap_dims]
0229 |         global_fail = len(gaps) > 0 and len(local_gaps) < len(gaps)
0230 | 
0231 |         if local_gaps:
0232 |             return self._decide(
0233 |                 row, Signal.EDIT, "Local gaps", {"gaps": local_gaps}
0234 |             )
0235 | 
0236 |         # 5) STAGNATION on core dims → RESAMPLE (then possibly ESCALATE later)
0237 |         if self._stagnating(window, thr):
0238 |             self._bump_resamples(row.unit)
0239 |             return self._resample(
0240 |                 row, "Stagnation on core dims", candidate_exemplars
0241 |             )
0242 | 
0243 |         # 6) GLOBAL failure + worsening trend → ESCALATE (after a few resamples)
0244 |         if global_fail and self._worsening(row.unit, trend):
0245 |             if self.resample_counts.get(row.unit, 0) >= self.p.escalate_after:
0246 |                 self._set_cooldown(row.unit, row.step_idx)
0247 |                 return self._decide(
0248 |                     row,
0249 |                     Signal.ESCALATE,
0250 |                     "Global fail & worsening after resamples",
0251 |                     {},
0252 |                 )
0253 |             else:
0254 |                 self._bump_resamples(row.unit)
0255 |                 return self._resample(
0256 |                     row,
0257 |                     "Global fail & worsening (resample first)",
0258 |                     candidate_exemplars,
0259 |                 )
0260 | 
0261 |         # 7) Below mins for patience window → RESAMPLE
0262 |         if not self._recently_above(window, thr, patience=self.p.patience):
0263 |             self._bump_resamples(row.unit)
0264 |             return self._resample(
0265 |                 row,
0266 |                 "Below thresholds for patience window",
0267 |                 candidate_exemplars,
0268 |             )
0269 | 
0270 |         # 8) default: EDIT small gaps
0271 |         return self._decide(
0272 |             row,
0273 |             Signal.EDIT,
0274 |             "Default edit to close residual gaps",
0275 |             {"gaps": gaps},
0276 |         )
0277 | 
0278 |     # ========= internals =========
0279 | 
0280 |     def _thresholds_for(self, kind: str) -> Thresholds:
0281 |         return self.thr_code if kind == "code" else self.thr_text
0282 | 
0283 |     def _clipped(self, row: VPMRow) -> VPMRow:
0284 |         """Clip extreme outliers on selected dims using rolling z-score."""
0285 |         if not self.p.zscore_clip_dims:
0286 |             return row
0287 |         h = self.history.get(row.unit, [])
0288 |         for d in self.p.zscore_clip_dims:
0289 |             v = row.dims.get(d)
0290 |             if v is None or len(h) < 4:
0291 |                 continue
0292 |             series = [w.dims.get(d) for w in h if w.dims.get(d) is not None]
0293 |             if len(series) < 4:
0294 |                 continue
0295 |             mu, sd = stats.mean(series), (stats.pstdev(series) or 1e-6)
0296 |             z = abs((v - mu) / sd)
0297 |             if z > self.p.zscore_clip_sigma:
0298 |                 row.dims[d] = (
0299 |                     mu + self.p.zscore_clip_sigma * (1 if v > mu else -1) * sd
0300 |                 )
0301 |         return row
0302 | 
0303 |     def _stable_above(
0304 |         self, window: List[VPMRow], thr: Thresholds, margin: float
0305 |     ) -> bool:
0306 |         if not window:
0307 |             return False
0308 |         dims = list(thr.mins.keys())
0309 |         recent = window[-self.p.patience :]
0310 |         for w in recent:
0311 |             for d in dims:
0312 |                 v = self._val(w, d)
0313 |                 if v is None or v < thr.mins[d] + margin:
0314 |                     return False
0315 |         return True
0316 | 
0317 |     def _recently_above(
0318 |         self, window: List[VPMRow], thr: Thresholds, patience: int
0319 |     ) -> bool:
0320 |         dims = list(thr.mins.keys())
0321 |         recent = window[-patience:]
0322 |         for w in recent:
0323 |             if all((self._val(w, d) or 0) >= thr.mins[d] for d in dims):
0324 |                 return True
0325 |         return False
0326 | 
0327 |     def _should_spinoff(self, row: VPMRow) -> bool:
0328 |         nov = row.dims.get(self.p.spinoff_dim)
0329 |         stk = row.dims.get(self.p.stickiness_dim)
0330 |         if nov is None or stk is None:
0331 |             return False
0332 |         return (nov >= self.p.spinoff_gate[0]) and (
0333 |             stk <= self.p.spinoff_gate[1]
0334 |         )
0335 | 
0336 |     def _gaps(self, row: VPMRow, thr: Thresholds) -> List[str]:
0337 |         gaps = []
0338 |         for k, t in thr.mins.items():
0339 |             v = self._val(row, k)
0340 |             if v is None:
0341 |                 continue
0342 |             if v < t - self.p.edit_margin:
0343 |                 gaps.append(k)
0344 |         return gaps
0345 | 
0346 |     def _regressions(self, window: List[VPMRow]) -> int:
0347 |         if len(window) < 2:
0348 |             return 0
0349 |         regs = 0
0350 |         dims = set(window[-1].dims.keys())
0351 |         for i in range(1, len(window)):
0352 |             prev, cur = window[i - 1], window[i]
0353 |             dips = sum(
0354 |                 1
0355 |                 for d in dims
0356 |                 if d in prev.dims
0357 |                 and d in cur.dims
0358 |                 and cur.dims[d] < prev.dims[d] - 1e-6
0359 |             )
0360 |             regs += 1 if dips >= max(1, len(dims) // 4) else 0
0361 |         return regs
0362 | 
0363 |     def _trend(self, window: List[VPMRow]) -> Dict[str, float]:
0364 |         if len(window) < 2:
0365 |             return {}
0366 |         n = len(window)
0367 |         t = list(range(n))
0368 |         out: Dict[str, float] = {}
0369 |         dims = set().union(*(w.dims.keys() for w in window))
0370 |         for d in dims:
0371 |             y = [w.dims.get(d) for w in window if w.dims.get(d) is not None]
0372 |             if len(y) < 2:
0373 |                 continue
0374 |             out[d] = (y[-1] - y[0]) / (n - 1)
0375 |         return out
0376 | 
0377 |     def _track_oscillation(self, unit: str, trend: Dict[str, float]):
0378 |         """Track sign flips in average slope to detect oscillations."""
0379 |         if not trend:
0380 |             return
0381 |         avg = sum(trend.values()) / max(1, len(trend))
0382 |         dir_ = 1 if avg > 0 else -1
0383 |         hist = self.osc_dir_hist.setdefault(unit, [])
0384 |         if hist and hist[-1] != dir_:
0385 |             hist.append(dir_)
0386 |         elif not hist:
0387 |             hist.append(dir_)
0388 |         if len(hist) > self.p.oscillation_window:
0389 |             self.osc_dir_hist[unit] = hist[-self.p.oscillation_window :]
0390 | 
0391 |     def _worsening(self, unit: str, trend: Dict[str, float]) -> bool:
0392 |         if not trend:
0393 |             return False
0394 |         vals = list(trend.values())
0395 |         neg = sum(1 for v in vals if v < -0.003)
0396 |         return neg >= max(1, len(vals) // 2) or self._oscillating_unit(unit)
0397 | 
0398 |     def _oscillating_unit(self, unit: str) -> bool:
0399 |         hist = self.osc_dir_hist.get(unit, [])
0400 |         if len(hist) < self.p.oscillation_window:
0401 |             return False
0402 |         flips = sum(1 for i in range(1, len(hist)) if hist[i] != hist[i - 1])
0403 |         return flips >= self.p.oscillation_threshold
0404 | 
0405 |     def _stagnating(self, window: List[VPMRow], thr: Thresholds) -> bool:
0406 |         if len(window) < self.p.patience + 1:
0407 |             return False
0408 |         recent = window[-(self.p.patience + 1) :]
0409 |         core = [k for k in thr.mins.keys() if k in recent[-1].dims]
0410 |         for d in core:
0411 |             series = [w.dims.get(d, 0.0) for w in recent]
0412 |             if series[-1] - series[0] > 0.005:
0413 |                 return False
0414 |         return True
0415 | 
0416 |     def _val(self, row: VPMRow, key: str) -> Optional[float]:
0417 |         v = row.dims.get(key)
0418 |         if v is None:
0419 |             return None
0420 |         try:
0421 |             return float(v)
0422 |         except Exception:
0423 |             return None
0424 | 
0425 |     def _decide(
0426 |         self, row: VPMRow, signal: Signal, reason: str, params: Dict[str, Any]
0427 |     ) -> Decision:
0428 |         dec = Decision(
0429 |             signal=signal,
0430 |             reason=reason,
0431 |             params=params,
0432 |             snapshot={
0433 |                 "unit": row.unit,
0434 |                 "kind": row.kind,
0435 |                 "step_idx": row.step_idx,
0436 |                 "dims": row.dims,
0437 |             },
0438 |         )
0439 |         self.last_signal[row.unit] = signal
0440 | 
0441 |         # bandit credit: on EDIT/STOP reward current exemplar; on RESAMPLE pick next
0442 |         eid = row.meta.get("exemplar_id")
0443 |         if eid and self.bandit_update and signal in (Signal.EDIT, Signal.STOP):
0444 |             try:
0445 |                 self.bandit_update(eid, self._reward(row))
0446 |             except Exception:
0447 |                 pass
0448 | 
0449 |         self._persist_state()
0450 |         self.log(
0451 |             "decision",
0452 |             {
0453 |                 "unit": row.unit,
0454 |                 "signal": signal.name,
0455 |                 "reason": reason,
0456 |                 **params,
0457 |             },
0458 |         )
0459 |         return dec
0460 | 
0461 |     def _reward(self, row: VPMRow) -> float:
0462 |         core = [
0463 |             "coverage",
0464 |             "correctness",
0465 |             "coherence",
0466 |             "tests_pass_rate",
0467 |             "type_safe",
0468 |             "lint_clean",
0469 |         ]
0470 |         vals = [row.dims[d] for d in core if d in row.dims]
0471 |         if not vals:
0472 |             vals = list(row.dims.values())
0473 |         return float(sum(vals) / len(vals)) if vals else 0.0
0474 | 
0475 |     def _resample(
0476 |         self, row: VPMRow, why: str, candidates: Optional[List[str]]
0477 |     ) -> Decision:
0478 |         params: Dict[str, Any] = {"why": why}
0479 |         if candidates and self.bandit_choose:
0480 |             try:
0481 |                 chosen = self.bandit_choose(candidates)
0482 |                 params["exemplar_id"] = chosen
0483 |             except Exception:
0484 |                 pass
0485 |         self._set_cooldown(row.unit, row.step_idx)
0486 |         return self._decide(row, Signal.RESAMPLE, why, params)
0487 | 
0488 |     def _bump_resamples(self, unit: str):
0489 |         self.resample_counts[unit] = self.resample_counts.get(unit, 0) + 1
0490 | 
0491 |     def _set_cooldown(self, unit: str, step_idx: Optional[int]):
0492 |         if step_idx is None:
0493 |             return
0494 |         self.cooldown_until_step[unit] = step_idx + self.p.cooldown_steps
0495 | 
0496 |     def _in_cooldown(self, unit: str, step_idx: Optional[int]) -> bool:
0497 |         if step_idx is None:
0498 |             return False
0499 |         until = self.cooldown_until_step.get(unit)
0500 |         return until is not None and step_idx < until
0501 | 
0502 |     # -------- goal awareness --------
0503 | 
0504 |     def _goal_ok_if_configured(self, row: VPMRow) -> bool:
0505 |         if not (self.p.goal_kind and self.p.goal_name and self.goal_scorer):
0506 |             return True
0507 |         try:
0508 |             eval_ = self.goal_scorer(
0509 |                 self.p.goal_kind, self.p.goal_name, row.dims
0510 |             )
0511 |             score = float(eval_.get("score", 0.0))
0512 |             unmet = eval_.get("unmet", [])
0513 |             return (score >= self.p.goal_min_score) and (
0514 |                 len(unmet) <= self.p.goal_allow_unmet
0515 |             )
0516 |         except Exception:
0517 |             return True  # fail-open to not block pipeline
0518 | 
0519 |     # -------- persistence --------
0520 | 
0521 |     def _persist_state(self):
0522 |         if not self.state_path:
0523 |             return
0524 |         try:
0525 |             data = {
0526 |                 "resample_counts": self.resample_counts,
0527 |                 "cooldown_until_step": self.cooldown_until_step,
0528 |                 "last_signal": {
0529 |                     k: v.name for k, v in self.last_signal.items()
0530 |                 },
0531 |                 "osc_dir_hist": self.osc_dir_hist,
0532 |             }
0533 |             self.state_path.parent.mkdir(parents=True, exist_ok=True)
0534 |             self.state_path.write_text(json.dumps(data, indent=2))
0535 |         except Exception:
0536 |             pass
0537 | 
0538 |     def _load_state(self):
0539 |         if not self.state_path or not self.state_path.exists():
0540 |             return
0541 |         try:
0542 |             data = json.loads(self.state_path.read_text())
0543 |             self.resample_counts = data.get("resample_counts", {})
0544 |             self.cooldown_until_step = data.get("cooldown_until_step", {})
0545 |             self.last_signal = {
0546 |                 k: Signal[v] for k, v in data.get("last_signal", {}).items()
0547 |             }
0548 |             self.osc_dir_hist = data.get("osc_dir_hist", {})
0549 |         except Exception:
0550 |             # ignore corrupted state
0551 |             self.resample_counts = {}
0552 |             self.cooldown_until_step = {}
0553 |             self.last_signal = {}
0554 |             self.osc_dir_hist = {}
0555 | 
0556 | 
0557 | # ========= convenience builders =========
0558 | 
0559 | 
0560 | def default_controller(
0561 |     state_path: Optional[str] = "./runs/vpm_state.json",
0562 | ) -> VPMController:
0563 |     thr_code = Thresholds(
0564 |         mins={
0565 |             "tests_pass_rate": 1.0,
0566 |             "coverage": 0.70,
0567 |             "type_safe": 1.0,
0568 |             "lint_clean": 1.0,
0569 |             "complexity_ok": 0.8,
0570 |         },
0571 |         stop_margin=0.0,
0572 |         edit_margin=0.0,
0573 |     )
0574 |     thr_text = Thresholds(
0575 |         mins={
0576 |             "coverage": 0.80,
0577 |             "correctness": 0.75,
0578 |             "coherence": 0.75,
0579 |             "citation_support": 0.65,
0580 |             "entity_consistency": 0.80,
0581 |         },
0582 |         stop_margin=0.02,
0583 |         edit_margin=0.01,
0584 |     )
0585 |     return VPMController(thr_code, thr_text, Policy(), state_path=state_path)
0586 | 
0587 | 
0588 | # ========= example usage =========
0589 | if __name__ == "__main__":
0590 |     ctrl = default_controller()
0591 | 
0592 |     def row(step, cov, cor, coh, cit, ent) -> VPMRow:
0593 |         return VPMRow(
0594 |             unit="Blog:Method",
0595 |             kind="text",
0596 |             timestamp=time.time(),
0597 |             step_idx=step,
0598 |             dims=dict(
0599 |                 coverage=cov,
0600 |                 correctness=cor,
0601 |                 coherence=coh,
0602 |                 citation_support=cit,
0603 |                 entity_consistency=ent,
0604 |                 novelty=0.78,
0605 |                 stickiness=0.46,
0606 |             ),
0607 |             meta={"exemplar_id": "ex_pack_A", "total_steps": step},
0608 |         )
0609 | 
0610 |     frames = [
0611 |         row(1, 0.62, 0.60, 0.64, 0.30, 0.70),
0612 |         row(2, 0.70, 0.66, 0.70, 0.55, 0.78),
0613 |         row(3, 0.74, 0.70, 0.72, 0.60, 0.80),
0614 |         row(4, 0.81, 0.76, 0.77, 0.67, 0.85),
0615 |         row(5, 0.82, 0.77, 0.78, 0.68, 0.86),
0616 |     ]
0617 |     for f in frames:
0618 |         dec = ctrl.add(
0619 |             f, candidate_exemplars=["ex_pack_A", "ex_pack_B", "ex_pack_C"]
0620 |         )
0621 |         print(
0622 |             f"step {f.step_idx}: {dec.signal.name} — {dec.reason} {dec.params}"
0623 |         )
```


---

## F0006 — `vpm_differential_analyzer.py`

```text
FILE_ID: F0006
PATH: vpm_differential_analyzer.py
LANGUAGE: python
LINES: 74
BYTES_UTF8: 2942
SHA256: 6f4bf1b9da2496752c29eb5af017e4a5297546b6d859c7b3618c2702403a9edc
```

```python
0001 | # stephanie/analysis/vpm_differential_analyzer.py
0002 | from __future__ import annotations
0003 | 
0004 | import os
0005 | from pathlib import Path
0006 | 
0007 | import matplotlib
0008 | import numpy as np
0009 | from zeromodel.vpm.logic import vpm_add, vpm_and, vpm_subtract, vpm_xor
0010 | 
0011 | matplotlib.use('Agg')
0012 | import matplotlib.pyplot as plt
0013 | from PIL import Image
0014 | 
0015 | 
0016 | class VPMDifferentialAnalyzer:
0017 |     def __init__(self, output_dir="vpm_analysis"):
0018 |         Path(output_dir).mkdir(exist_ok=True)
0019 |         self.output_dir = output_dir
0020 | 
0021 |     def analyze(self, vpm_good, vpm_bad, prefix="diff"):
0022 |         diff = vpm_subtract(vpm_good, vpm_bad)
0023 |         overlap = vpm_and(vpm_good, vpm_bad)
0024 |         contrast = vpm_xor(vpm_good, vpm_bad)
0025 |         enriched = vpm_add(diff, overlap * 0.25)
0026 |         self.save_vpm_image(diff, "Unique (Good - Bad)", f"{self.output_dir}/{prefix}_unique.png")
0027 |         self.save_vpm_image(contrast, "Contrast (Good XOR Bad)", f"{self.output_dir}/{prefix}_contrast.png")
0028 |         self.save_vpm_image(enriched, "Enriched Knowledge", f"{self.output_dir}/{prefix}_enriched.png")
0029 |         return {"diff": diff, "contrast": contrast, "enriched": enriched}
0030 | 
0031 |     def save_vpm_image(self, vpm, title: str, filename: str):
0032 |         """Save VPM as image with proper handling of both array and PIL Image types."""
0033 |         # Convert to normalized array for consistent processing
0034 |         arr = _to_normalized_array(vpm)
0035 |         
0036 |         # Handle 3D arrays (RGB) by converting to grayscale if needed
0037 |         if arr.ndim == 3:
0038 |             if arr.shape[2] == 3:
0039 |                 # Convert RGB to grayscale
0040 |                 arr = 0.2989 * arr[:,:,0] + 0.5870 * arr[:,:,1] + 0.1140 * arr[:,:,2]
0041 |             else:
0042 |                 # Take first channel
0043 |                 arr = arr[:,:,0]
0044 |         
0045 |         # Create and save image
0046 |         plt.figure(figsize=(6, 6))
0047 |         plt.imshow(arr, cmap='gray', vmin=0, vmax=1)
0048 |         plt.title(title)
0049 |         plt.colorbar(label='Normalized Score')
0050 |         plt.xlabel('Metrics (sorted)')
0051 |         plt.ylabel('Documents (sorted)')
0052 | 
0053 |         filepath = os.path.join(self.output_dir, filename)
0054 |         plt.savefig(filepath, dpi=300, bbox_inches='tight')
0055 |         plt.close()
0056 |         print(f"Saved VPM image: {filepath}")
0057 | 
0058 | def _to_normalized_array(obj):
0059 |     """Convert PIL Image or numpy array to normalized float32 array in [0,1] range."""
0060 |     if isinstance(obj, Image.Image):
0061 |         # Convert PIL Image to numpy array
0062 |         arr = np.array(obj.convert("RGB"))
0063 |         # Normalize to [0,1] range
0064 |         if arr.dtype != np.float32:
0065 |             if np.issubdtype(arr.dtype, np.integer):
0066 |                 max_val = np.iinfo(arr.dtype).max
0067 |                 arr = arr.astype(np.float32) / max_val
0068 |             else:
0069 |                 arr = np.clip(arr.astype(np.float32), 0.0, 1.0)
0070 |         return arr
0071 |     elif isinstance(obj, np.ndarray):
0072 |         return np.clip(obj.astype(np.float32), 0.0, 1.0)
0073 |     else:
0074 |         raise TypeError(f"Expected PIL.Image or numpy.ndarray, got {type(obj)}")
```


---

## F0007 — `vpm_emitter.py`

```text
FILE_ID: F0007
PATH: vpm_emitter.py
LANGUAGE: python
LINES: 474
BYTES_UTF8: 17655
SHA256: a97b073d9f2f42bfb52c0eb0268f834079fddf8361aaf30bda6ffd962b0b35c2
```

```python
0001 | # stephanie/zero_model/vpm_emitter.py
0002 | from __future__ import annotations
0003 | 
0004 | import json
0005 | import logging
0006 | import time
0007 | from dataclasses import dataclass
0008 | from pathlib import Path
0009 | from typing import Any, Dict, List, Optional
0010 | 
0011 | # Defer importing pyplot until used (helps in headless envs)
0012 | import matplotlib
0013 | import numpy as np
0014 | 
0015 | if matplotlib.get_backend().lower() != "agg":
0016 |     matplotlib.use("Agg")
0017 | import matplotlib.pyplot as plt
0018 | 
0019 | from stephanie.services.zeromodel_service import ZeroModelService
0020 | 
0021 | 
0022 | # -------------------------
0023 | # Metrics container
0024 | # -------------------------
0025 | @dataclass
0026 | class VPMMetrics:
0027 |     """Standardized VPM metrics structure for consistent visualization."""
0028 |     coverage: float = 0.0
0029 |     correctness: float = 0.0
0030 |     coherence: float = 0.0
0031 |     citation_support: float = 0.0
0032 |     entity_consistency: float = 0.0
0033 |     readability: float = 0.0
0034 |     novelty: float = 0.0
0035 |     stickiness: float = 0.0
0036 |     tests_pass_rate: float = 0.0
0037 |     mutation_score: float = 0.0
0038 |     complexity: float = 0.0
0039 |     type_safe: float = 0.0
0040 |     lint_clean: float = 0.0
0041 |     faithfulness: float = 0.0
0042 |     overall: float = 0.0
0043 | 
0044 |     @classmethod
0045 |     def from_dict(cls, data: Dict[str, float]) -> VPMMetrics:
0046 |         """Create VPMMetrics from a dictionary of metrics."""
0047 |         # Normalize common aliases
0048 |         data = dict(data or {})
0049 |         if "coverage" not in data and "claim_coverage" in data:
0050 |             data["coverage"] = data["claim_coverage"]
0051 |         if "no_halluc" in data and "hallucination_rate" not in data:
0052 |             # Leave as-is; packing handles it
0053 |             pass
0054 |         return cls(
0055 |             coverage=data.get("coverage", 0.0),
0056 |             correctness=data.get("correctness", 0.0),
0057 |             coherence=data.get("coherence", 0.0),
0058 |             citation_support=data.get("citation_support", 0.0),
0059 |             entity_consistency=data.get("entity_consistency", 0.0),
0060 |             readability=data.get("readability", 0.0),
0061 |             novelty=data.get("novelty", 0.0),
0062 |             stickiness=data.get("stickiness", 0.0),
0063 |             tests_pass_rate=data.get("tests_pass_rate", 0.0),
0064 |             mutation_score=data.get("mutation_score", 0.0),
0065 |             complexity=data.get("complexity", 0.0),
0066 |             type_safe=data.get("type_safe", 0.0),
0067 |             lint_clean=data.get("lint_clean", 0.0),
0068 |             faithfulness=data.get("faithfulness", 0.0),
0069 |             overall=data.get("overall", 0.0),
0070 |         )
0071 | 
0072 |     def to_dict(self) -> Dict[str, float]:
0073 |         """Convert to dictionary for serialization."""
0074 |         return {
0075 |             "coverage": float(self.coverage),
0076 |             "correctness": float(self.correctness),
0077 |             "coherence": float(self.coherence),
0078 |             "citation_support": float(self.citation_support),
0079 |             "entity_consistency": float(self.entity_consistency),
0080 |             "readability": float(self.readability),
0081 |             "novelty": float(self.novelty),
0082 |             "stickiness": float(self.stickiness),
0083 |             "tests_pass_rate": float(self.tests_pass_rate),
0084 |             "mutation_score": float(self.mutation_score),
0085 |             "complexity": float(self.complexity),
0086 |             "type_safe": float(self.type_safe),
0087 |             "lint_clean": float(self.lint_clean),
0088 |             "faithfulness": float(self.faithfulness),
0089 |             "overall": float(self.overall),
0090 |         }
0091 | 
0092 | 
0093 | # -------------------------
0094 | # VPM Emitter
0095 | # -------------------------
0096 | class VPMEmitter:
0097 |     """
0098 |     VPM (Visual Progress Map) Emitter: Generates visualizations of AI processing steps.
0099 | 
0100 |     - Uses ZeroModel service for high-fidelity tiles if available
0101 |     - Falls back to matplotlib for PNG generation
0102 |     - Normalizes metrics across domains (text/code/image)
0103 |     - Outputs:
0104 |         * ABC tile (A/B/C comparison)
0105 |         * Iteration timeline
0106 |         * PACS panel heatmap
0107 |         * Knowledge progress (claim/evidence curves)
0108 |     """
0109 | 
0110 |     def __init__(
0111 |         self,
0112 |         logger: logging.Logger,
0113 |         zero_model_service: Optional[ZeroModelService] = None,
0114 |         output_dir: str = "reports/vpm",
0115 |         config: Optional[Dict[str, Any]] = None,
0116 |     ):
0117 |         self.logger = logger
0118 |         self.zm = zero_model_service
0119 |         self.output_dir = Path(output_dir)
0120 |         self.output_dir.mkdir(parents=True, exist_ok=True)
0121 | 
0122 |         self.config = {
0123 |             "default_metrics": [
0124 |                 "overall",
0125 |                 "coverage",
0126 |                 "faithfulness",
0127 |                 "coherence",
0128 |                 "structure",
0129 |                 "no_halluc",     # derived from hallucination_rate when packing
0130 |                 "figure_ground", # derived if present
0131 |             ],
0132 |             "panel_metrics": [
0133 |                 "overall",
0134 |                 "coverage",
0135 |                 "faithfulness",
0136 |                 "structure",
0137 |                 "no_halluc",
0138 |                 "figure_ground",
0139 |             ],
0140 |             "thresholds": {"good": 0.75, "medium": 0.5, "bad": 0.25},
0141 |         }
0142 |         if config:
0143 |             # shallow merge for convenience
0144 |             self.config.update(config)
0145 | 
0146 |         # Use the project's structured logger if available
0147 |         if hasattr(self.logger, "log"):
0148 |             self.logger.log("VPMEmitterInit", {
0149 |                 "output_dir": str(self.output_dir),
0150 |                 "zero_model_available": bool(self.zm),
0151 |             })
0152 |         else:
0153 |             self.logger.info("VPMEmitter initialized at %s (ZeroModel=%s)",
0154 |                              str(self.output_dir), bool(self.zm))
0155 | 
0156 |     # ---------- public API ----------
0157 | 
0158 |     def emit_abc_tile(
0159 |         self,
0160 |         doc_id: str,
0161 |         metrics_a: Dict[str, float],
0162 |         metrics_b: Dict[str, float],
0163 |         metrics_c: Dict[str, float],
0164 |         title: str = "A/B/C Comparison",
0165 |     ) -> Optional[str]:
0166 |         """Emit a tile comparing metrics A, B, and C."""
0167 |         try:
0168 |             data = {
0169 |                 "doc_id": str(doc_id),
0170 |                 "title": title,
0171 |                 "metrics": {
0172 |                     "A": self._pack(metrics_a),
0173 |                     "B": self._pack(metrics_b),
0174 |                     "C": self._pack(metrics_c),
0175 |                 },
0176 |                 "iterations": [],
0177 |                 "timestamp": time.time(),
0178 |             }
0179 | 
0180 |             # Try ZeroModel service first
0181 |             if self.zm and hasattr(self.zm, "generate_summary_vpm_tiles"):
0182 |                 payload = {"vpm_data": data, "output_dir": str(self.output_dir)}
0183 |                 result = self.zm.generate_summary_vpm_tiles(**payload) or {}
0184 |                 tile_path = result.get("quality_tile_path")
0185 |                 if tile_path:
0186 |                     self._elog("VPMEmitABCComplete", {"doc_id": doc_id, "tile_path": tile_path, "service": "zero_model"})
0187 |                     return tile_path
0188 | 
0189 |             # Fallback
0190 |             return self._matplotlib_abc_tile(doc_id, data["metrics"]["A"], data["metrics"]["B"], data["metrics"]["C"], title)
0191 | 
0192 |         except Exception as e:
0193 |             self._elog("VPMEmitABCTileError", {"doc_id": doc_id, "error": str(e)})
0194 |             return None
0195 | 
0196 |     def emit_iteration_timeline(
0197 |         self,
0198 |         doc_id: str,
0199 |         iterations: List[Dict[str, Any]],
0200 |         title: str = "Iteration Progress",
0201 |     ) -> Optional[str]:
0202 |         """Emit a timeline showing overall score progress across iterations."""
0203 |         try:
0204 |             # ZeroModel (optional)
0205 |             if self.zm and hasattr(self.zm, "generate_summary_vpm_tiles"):
0206 |                 data = {
0207 |                     "doc_id": str(doc_id),
0208 |                     "title": title,
0209 |                     "metrics": {"A": {}, "B": {}, "C": {}},
0210 |                     "iterations": iterations or [],
0211 |                     "timestamp": time.time(),
0212 |                 }
0213 |                 result = self.zm.generate_summary_vpm_tiles(vpm_data=data, output_dir=str(self.output_dir)) or {}
0214 |                 timeline_path = result.get("iter_timeline")
0215 |                 if timeline_path:
0216 |                     self._elog("VPMEmitIterTimelineComplete", {"doc_id": doc_id, "timeline_path": timeline_path, "service": "zero_model"})
0217 |                     return timeline_path
0218 | 
0219 |             # Fallback
0220 |             return self._matplotlib_iteration_line(doc_id, iterations, title)
0221 | 
0222 |         except Exception as e:
0223 |             self._elog("VPMEmitIterTimelineError", {"doc_id": doc_id, "error": str(e)})
0224 |             return None
0225 | 
0226 |     def emit_panel_heatmap(
0227 |         self,
0228 |         doc_id: str,
0229 |         panel_detail: Dict[str, Any],
0230 |         title: str = "PACS Panel",
0231 |     ) -> Optional[str]:
0232 |         """Emit a heatmap for PACS panel outputs (skeptic/editor/risk)."""
0233 |         try:
0234 |             # ZeroModel (optional)
0235 |             if self.zm and hasattr(self.zm, "generate_summary_vpm_tiles"):
0236 |                 data = {
0237 |                     "doc_id": str(doc_id),
0238 |                     "title": title,
0239 |                     "metrics": {"A": {}, "B": {}, "C": {}},
0240 |                     "iterations": [],
0241 |                     "panel_detail": panel_detail or {},
0242 |                     "timestamp": time.time(),
0243 |                 }
0244 |                 result = self.zm.generate_summary_vpm_tiles(vpm_data=data, output_dir=str(self.output_dir)) or {}
0245 |                 heatmap_path = result.get("panel_heatmap")
0246 |                 if heatmap_path:
0247 |                     self._elog("VPMEmitPanelHeatmapComplete", {"doc_id": doc_id, "heatmap_path": heatmap_path, "service": "zero_model"})
0248 |                     return heatmap_path
0249 | 
0250 |             # Fallback
0251 |             return self._matplotlib_panel_heatmap(doc_id, panel_detail, title)
0252 | 
0253 |         except Exception as e:
0254 |             self._elog("VPMEmitPanelHeatmapError", {"doc_id": doc_id, "error": str(e)})
0255 |             return None
0256 | 
0257 |     def emit_knowledge_progress(
0258 |         self,
0259 |         doc_id: str,
0260 |         iterations: List[Dict[str, Any]],
0261 |         title: str = "Knowledge Progress",
0262 |     ) -> Optional[str]:
0263 |         """
0264 |         Emit knowledge progression curves (claim coverage, evidence strength).
0265 |         Falls back to whatever keys are available in iterations.
0266 |         """
0267 |         try:
0268 |             # ZeroModel (optional)
0269 |             if self.zm and hasattr(self.zm, "generate_summary_vpm_tiles"):
0270 |                 data = {
0271 |                     "doc_id": str(doc_id),
0272 |                     "title": title,
0273 |                     "metrics": {"A": {}, "B": {}, "C": {}},
0274 |                     "iterations": iterations or [],
0275 |                     "timestamp": time.time(),
0276 |                     "knowledge_progress": True,
0277 |                 }
0278 |                 result = self.zm.generate_summary_vpm_tiles(vpm_data=data, output_dir=str(self.output_dir)) or {}
0279 |                 progress_path = result.get("knowledge_progress")
0280 |                 if progress_path:
0281 |                     self._elog("VPMEmitKnowledgeProgressComplete", {"doc_id": doc_id, "progress_path": progress_path, "service": "zero_model"})
0282 |                     return progress_path
0283 | 
0284 |             # Fallback
0285 |             return self._matplotlib_knowledge_progress(doc_id, iterations, title)
0286 | 
0287 |         except Exception as e:
0288 |             self._elog("VPMEmitKnowledgeProgressError", {"doc_id": doc_id, "error": str(e)})
0289 |             return None
0290 | 
0291 |     # ---------- helpers ----------
0292 | 
0293 |     def _elog(self, event: str, payload: Dict[str, Any]):
0294 |         if hasattr(self.logger, "log"):
0295 |             self.logger.log(event, payload)
0296 |         else:
0297 |             self.logger.info("%s: %s", event, json.dumps(payload))
0298 | 
0299 |     def _pack(self, m: Dict[str, Any]) -> Dict[str, float]:
0300 |         """
0301 |         Normalize metric keys to the canonical set used by tiles:
0302 |         - coverage: prefer 'claim_coverage' if present
0303 |         - no_halluc: derived from 'hallucination_rate', else 0/1 if present
0304 |         - figure_ground: nested 'figure_results.overall_figure_score'
0305 |         """
0306 |         m = dict(m or {})
0307 |         coverage = float(m.get("coverage", m.get("claim_coverage", 0.0)))
0308 |         faithfulness = float(m.get("faithfulness", 0.0))
0309 |         structure = float(m.get("structure", 0.0))
0310 |         overall = float(m.get("overall", 0.0))
0311 | 
0312 |         # hallucination
0313 |         if "hallucination_rate" in m:
0314 |             no_halluc = float(1.0 - float(m.get("hallucination_rate", 1.0)))
0315 |         else:
0316 |             no_halluc = float(m.get("no_halluc", 0.0))
0317 | 
0318 |         # figure grounding nested metric
0319 |         fig = 0.0
0320 |         fr = m.get("figure_results", {})
0321 |         if isinstance(fr, dict):
0322 |             fig = float(fr.get("overall_figure_score", 0.0))
0323 | 
0324 |         return {
0325 |             "overall": overall,
0326 |             "coverage": coverage,
0327 |             "faithfulness": faithfulness,
0328 |             "structure": structure,
0329 |             "no_halluc": no_halluc,
0330 |             "figure_ground": fig,
0331 |         }
0332 | 
0333 |     # ---------- matplotlib fallbacks ----------
0334 | 
0335 |     def _matplotlib_abc_tile(
0336 |         self,
0337 |         doc_id: str,
0338 |         A: Dict[str, float],
0339 |         B: Dict[str, float],
0340 |         C: Dict[str, float],
0341 |         title: str,
0342 |     ) -> Optional[str]:
0343 |         if plt is None:
0344 |             self._elog("MatplotlibMissing", {"for": "abc_tile"})
0345 |             return None
0346 | 
0347 |         names = self.config["default_metrics"]
0348 |         # build matrix rows A/B/C
0349 |         mat = np.array(
0350 |             [
0351 |                 [A.get(k, 0.0) for k in names],
0352 |                 [B.get(k, 0.0) for k in names],
0353 |                 [C.get(k, 0.0) for k in names],
0354 |             ],
0355 |             dtype=np.float32,
0356 |         )
0357 | 
0358 |         fig, ax = plt.subplots(figsize=(9.2, 3.0))
0359 |         im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0)
0360 |         ax.set_title(title)
0361 |         ax.set_yticks([0, 1, 2], labels=["A", "B", "C"])
0362 |         ax.set_xticks(range(len(names)), labels=names, rotation=20, ha="right")
0363 |         fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
0364 |         out = str(self.output_dir / f"{doc_id}_abc.png")
0365 |         plt.tight_layout()
0366 |         plt.savefig(out, dpi=200)
0367 |         plt.close(fig)
0368 |         return out
0369 | 
0370 |     def _matplotlib_iteration_line(
0371 |         self,
0372 |         doc_id: str,
0373 |         iters: List[Dict[str, Any]],
0374 |         title: str,
0375 |     ) -> Optional[str]:
0376 |         if plt is None:
0377 |             self._elog("MatplotlibMissing", {"for": "iteration_timeline"})
0378 |             return None
0379 |         if not iters:
0380 |             return None
0381 | 
0382 |         xs = [int(i.get("iteration", idx + 1)) for idx, i in enumerate(iters)]
0383 |         current_scores = [float(i.get("current_score", 0.0)) for i in iters]
0384 |         cand_scores = [float(i.get("best_candidate_score", 0.0)) for i in iters]
0385 | 
0386 |         fig, ax = plt.subplots(figsize=(9.2, 4.0))
0387 |         ax.plot(xs, current_scores, linewidth=2, label="current score")
0388 |         ax.plot(xs, cand_scores, linewidth=2, label="candidate score")
0389 |         ax.set_title(title)
0390 |         ax.set_xlabel("Iteration")
0391 |         ax.set_ylabel("Overall")
0392 |         ax.grid(True, linestyle="--", alpha=0.5)
0393 |         ax.legend()
0394 |         out = str(self.output_dir / f"{doc_id}_iteration.png")
0395 |         plt.tight_layout()
0396 |         plt.savefig(out, dpi=200)
0397 |         plt.close(fig)
0398 |         return out
0399 | 
0400 |     def _matplotlib_panel_heatmap(
0401 |         self,
0402 |         doc_id: str,
0403 |         panel_detail: Dict[str, Any],
0404 |         title: str,
0405 |     ) -> Optional[str]:
0406 |         if plt is None:
0407 |             self._elog("MatplotlibMissing", {"for": "panel_heatmap"})
0408 |             return None
0409 | 
0410 |         panel = (panel_detail or {}).get("panel") or []
0411 |         if not panel:
0412 |             return None
0413 | 
0414 |         roles = [p.get("role", "?") for p in panel]
0415 |         metrics = self.config["panel_metrics"]
0416 | 
0417 |         # Assemble matrix of normalized [0..1] values per role x metric
0418 |         mat = np.zeros((len(roles), len(metrics)), dtype=np.float32)
0419 |         for i, entry in enumerate(panel):
0420 |             m = entry.get("metrics", {}) or {}
0421 |             packed = self._pack(m)
0422 |             for j, key in enumerate(metrics):
0423 |                 mat[i, j] = float(packed.get(key, 0.0))
0424 | 
0425 |         # Normalize by column (optional; packed already 0..1, but keep stable)
0426 |         for j in range(mat.shape[1]):
0427 |             col = mat[:, j]
0428 |             cmax, cmin = float(np.max(col)), float(np.min(col))
0429 |             if cmax > cmin:
0430 |                 mat[:, j] = (col - cmin) / (cmax - cmin)
0431 | 
0432 |         fig, ax = plt.subplots(figsize=(9.2, 3.0 + 0.25 * len(roles)))
0433 |         im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
0434 |         ax.set_title(title)
0435 |         ax.set_yticks(range(len(roles)), labels=roles)
0436 |         ax.set_xticks(range(len(metrics)), labels=metrics, rotation=20, ha="right")
0437 |         fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
0438 |         out = str(self.output_dir / f"{doc_id}_panel.png")
0439 |         plt.tight_layout()
0440 |         plt.savefig(out, dpi=200)
0441 |         plt.close(fig)
0442 |         return out
0443 | 
0444 |     def _matplotlib_knowledge_progress(
0445 |         self,
0446 |         doc_id: str,
0447 |         iters: List[Dict[str, Any]],
0448 |         title: str,
0449 |     ) -> Optional[str]:
0450 |         if plt is None:
0451 |             self._elog("MatplotlibMissing", {"for": "knowledge_progress"})
0452 |             return None
0453 |         if not iters:
0454 |             return None
0455 | 
0456 |         xs = [int(i.get("iteration", idx + 1)) for idx, i in enumerate(iters)]
0457 | 
0458 |         # Accept both your earlier keys and alternates
0459 |         coverage = [float(i.get("claim_coverage", i.get("coverage", 0.0))) for i in iters]
0460 |         evidence = [float(i.get("evidence_strength", i.get("citation_support", 0.0))) for i in iters]
0461 | 
0462 |         fig, ax = plt.subplots(figsize=(9.2, 4.0))
0463 |         ax.plot(xs, coverage, linewidth=2, label="claim coverage")
0464 |         ax.plot(xs, evidence, linewidth=2, label="evidence strength / citation")
0465 |         ax.set_title(title)
0466 |         ax.set_xlabel("Iteration")
0467 |         ax.set_ylabel("Score")
0468 |         ax.grid(True, linestyle="--", alpha=0.5)
0469 |         ax.legend()
0470 |         out = str(self.output_dir / f"{doc_id}_knowledge.png")
0471 |         plt.tight_layout()
0472 |         plt.savefig(out, dpi=200)
0473 |         plt.close(fig)
0474 |         return out
```


---

## F0008 — `vpm_phos.py`

```text
FILE_ID: F0008
PATH: vpm_phos.py
LANGUAGE: python
LINES: 661
BYTES_UTF8: 19612
SHA256: 48defe4dbff5a42ddc7ffd3f5f429c826c679cf540996d07d5c53af490b99276
```

```python
0001 | # stephanie/zeromodel/vpm_phos.py
0002 | """
0003 | VPM (Vectorized Performance Map) and PHOS (Packed High-Order Structure) Visualization Module
0004 | 
0005 | This module provides functionality for creating and analyzing visual performance representations
0006 | of AI model outputs across multiple evaluation dimensions. It implements the VPM/PHOS methodology
0007 | for model comparison and quality assessment.
0008 | 
0009 | Key Features:
0010 | - Robust vector normalization and scaling
0011 | - PHOS packing algorithms for visual pattern recognition
0012 | - Multi-dimensional performance visualization
0013 | - Automated artifact selection with improvement guards
0014 | - HRM vs Tiny model comparison framework
0015 | 
0016 | The PHOS algorithm sorts performance vectors and packs them into 2D representations
0017 | that highlight performance concentration patterns, making model strengths/weaknesses
0018 | visually apparent.
0019 | 
0020 | Author: Stephanie AI Team
0021 | Version: 1.0
0022 | Date: 2024
0023 | """
0024 | 
0025 | from __future__ import annotations
0026 | 
0027 | import json
0028 | import time
0029 | from dataclasses import dataclass
0030 | from pathlib import Path
0031 | from typing import Dict, Iterable, List, Optional, Tuple
0032 | 
0033 | import matplotlib.pyplot as plt
0034 | import numpy as np
0035 | import pandas as pd
0036 | 
0037 | from stephanie.utils.json_sanitize import dumps_safe
0038 | 
0039 | CAND_SUFFIXES = ["", ".score", ".aggregate", ".raw", ".value"]
0040 | 
0041 | 
0042 | # --- fix: dataclass decorator + types ---
0043 | @dataclass
0044 | class RunManifest:
0045 |     run_id: str
0046 |     dataset: str
0047 |     models: dict
0048 |     preproc_version: str = "v1"
0049 |     created_at: float = time.time()
0050 | 
0051 | 
0052 | def build_vpm(scores: dict, metric_whitelist=None):
0053 |     # scores: dict[str] -> np.array[T] or [T,K]
0054 |     cols, mats = [], []
0055 |     for k, v in scores.items():
0056 |         if metric_whitelist and k not in metric_whitelist:
0057 |             continue
0058 |         v = np.asarray(v)
0059 |         if v.ndim == 1:
0060 |             cols.append(k)
0061 |             mats.append(v[:, None])
0062 |         elif v.ndim == 2:
0063 |             for j in range(v.shape[1]):
0064 |                 cols.append(f"{k}[{j}]")
0065 |                 mats.append(v[:, j : j + 1])
0066 |     X = np.concatenate(mats, axis=1) if mats else np.zeros((0, 0))
0067 |     return X, cols
0068 | 
0069 | 
0070 | def robust_normalize(X, eps=1e-9):
0071 |     med = np.nanmedian(X, axis=0, keepdims=True)
0072 |     mad = np.nanmedian(np.abs(X - med), axis=0, keepdims=True) + eps
0073 |     Z = (X - med) / mad
0074 |     return np.clip(Z, -5, 5)  # squash extremes
0075 | 
0076 | 
0077 | # ---------------------------
0078 | # Low-level utils
0079 | # ---------------------------
0080 | 
0081 | 
0082 | def robust01(
0083 |     x: np.ndarray, p_lo: float = 10.0, p_hi: float = 90.0
0084 | ) -> np.ndarray:
0085 |     """
0086 |     Robust [0,1] scaling using percentiles to damp outliers.
0087 | 
0088 |     Args:
0089 |         x: Input array to normalize
0090 |         p_lo: Lower percentile for scaling (default: 10th percentile)
0091 |         p_hi: Upper percentile for scaling (default: 90th percentile)
0092 | 
0093 |     Returns:
0094 |         Array scaled to [0,1] range based on percentile bounds
0095 |     """
0096 |     x = np.asarray(x, dtype=np.float64).reshape(-1)
0097 |     if x.size == 0:
0098 |         return x
0099 |     lo = float(np.percentile(x, p_lo))
0100 |     hi = float(np.percentile(x, p_hi))
0101 |     if hi - lo < 1e-12:
0102 |         hi = lo + 1.0
0103 |     y = (x - lo) / (hi - lo)
0104 |     return np.clip(y, 0.0, 1.0)
0105 | 
0106 | 
0107 | def learn_layout(*vpm_lists):
0108 |     # stack absolute activations to emphasize structure
0109 |     U = np.concatenate([np.abs(X) for (X, _) in vpm_lists], axis=1)
0110 |     # simple heuristic: order by decreasing L2 norm then by correlation structure
0111 |     col_energy = np.linalg.norm(U, axis=0)
0112 |     order = np.argsort(-col_energy)
0113 |     return order.tolist()
0114 | 
0115 | 
0116 | def save_layout(order, names, path):
0117 |     with open(path, "w") as f:
0118 |         f.write(
0119 |             dumps_safe(
0120 |                 {"columns": [names[i] for i in order], "index": order},
0121 |                 indent=2,
0122 |             )
0123 |         )
0124 | 
0125 | 
0126 | def project(X, order):
0127 |     keep = [i for i in order if i < X.shape[1]]
0128 |     return X[:, keep]
0129 | 
0130 | 
0131 | def guess_diag_cols(names):
0132 |     DIAG_SUFFIX = (
0133 |         "uncertainty",
0134 |         "ood",
0135 |         "ood_hat",
0136 |         "temp01",
0137 |         "entropy",
0138 |         "jacobian",
0139 |         "consistency",
0140 |         "halt_prob",
0141 |     )
0142 |     return [i for i, n in enumerate(names) if any(s in n for s in DIAG_SUFFIX)]
0143 | 
0144 | 
0145 | def correlate_abs_delta_with_diags(Delta, X_diag):
0146 |     y = np.abs(Delta).mean(axis=1)  # per-turn intensity
0147 |     R = np.corrcoef(X_diag.T, y)[-1, :-1]  # quick/dirty
0148 |     return R
0149 | 
0150 | 
0151 | def delta_metrics(XA, XB):
0152 |     Delta = XA - XB
0153 |     absA, absB = np.abs(XA).ravel(), np.abs(XB).ravel()
0154 |     overlap = (
0155 |         (absA @ absB)
0156 |         / (np.linalg.norm(absA) + 1e-9)
0157 |         / (np.linalg.norm(absB) + 1e-9)
0158 |     )
0159 |     dmass = np.mean(np.abs(Delta))  # whole-field mass; optional TL window
0160 |     col_scores = np.mean(np.abs(Delta), axis=0)
0161 |     row_scores = np.mean(np.abs(Delta), axis=1)
0162 |     top_cols = np.argsort(-col_scores)[:25].tolist()
0163 |     top_rows = np.argsort(-row_scores)[:25].tolist()
0164 |     return Delta, {
0165 |         "delta_mass": float(dmass),
0166 |         "overlap": float(overlap),
0167 |         "top_cols": top_cols,
0168 |         "top_rows": top_rows,
0169 |     }
0170 | 
0171 | 
0172 | def save_delta(meta, names, path_json):
0173 |     meta["column_names"] = names
0174 |     with open(path_json, "w") as f:
0175 |         f.write(dumps_safe(meta, indent=2))
0176 | 
0177 | 
0178 | def to_square(vec: np.ndarray) -> Tuple[np.ndarray, int]:
0179 |     """
0180 |     Pad a 1D vector to the next square length and reshape to (s, s).
0181 | 
0182 |     Args:
0183 |         vec: Input 1D vector
0184 | 
0185 |     Returns:
0186 |         Tuple of (square_image, side_length)
0187 |     """
0188 |     v = np.asarray(vec, dtype=np.float64).reshape(-1)
0189 |     n = v.size
0190 |     if n == 0:
0191 |         return np.zeros((1, 1), dtype=np.float64), 1
0192 |     s = int(np.ceil(np.sqrt(n)))
0193 |     pad = s * s - n
0194 |     if pad > 0:
0195 |         v = np.pad(v, (0, pad), mode="constant")
0196 |     return v.reshape(s, s), s
0197 | 
0198 | 
0199 | def route(use_A_cond, A_agg, B_agg):
0200 |     # use A (expensive) when diagnostics say so, else B
0201 |     return np.where(use_A_cond, A_agg, B_agg)
0202 | 
0203 | 
0204 | def phos_sort_pack(v: np.ndarray, *, tl_frac: float = 0.25) -> np.ndarray:
0205 |     """
0206 |     PHOS (Packed High-Order Structure) algorithm.
0207 | 
0208 |     Sorts values in descending order and packs them into a square image with
0209 |     top values concentrated in the top-left region.
0210 | 
0211 |     Args:
0212 |         v: Input performance vector
0213 |         tl_frac: Fraction of area to allocate for top-left concentration
0214 | 
0215 |     Returns:
0216 |         Square image with sorted and packed values
0217 |     """
0218 |     v = np.asarray(v, dtype=np.float64).ravel()
0219 |     if v.size == 0:
0220 |         return np.zeros((1, 1), dtype=np.float64)
0221 | 
0222 |     # Normalize and prepare for packing
0223 |     v01 = robust01(v)
0224 |     n = v01.size
0225 |     s = int(np.ceil(np.sqrt(n)))
0226 |     pad = s * s - n
0227 |     if pad > 0:
0228 |         v01 = np.concatenate([v01, np.zeros(pad)])
0229 | 
0230 |     # Sort values in descending order
0231 |     order = np.argsort(v01)[::-1]
0232 |     sorted_vals = v01[order]
0233 |     img = sorted_vals.reshape(s, s)
0234 | 
0235 |     # Calculate top-left block size
0236 |     k = max(1, int(round(s * s * tl_frac)))
0237 |     packed = np.zeros_like(img)
0238 |     packed[:][:] = 0.0
0239 | 
0240 |     r = int(np.floor(np.sqrt(k)))
0241 |     if r <= 0:
0242 |         r = 1
0243 |     rr = r
0244 |     if rr * r * r > k:
0245 |         rr = int(np.floor(np.sqrt(k)))
0246 | 
0247 |     # Fill top-left with highest values
0248 |     top = sorted_vals[:k]
0249 |     tl = np.zeros_like(img)
0250 |     tl[:rr, :rr] = top[: rr * rr].reshape(rr, rr)
0251 |     rest = sorted_vals[rr * rr :]
0252 | 
0253 |     # Pack remaining values
0254 |     packed[:rr, :rr] = tl[:rr, :rr]
0255 |     flat = packed.ravel()
0256 |     flat[rr * rr : rr * rr + rest.size] = rest
0257 | 
0258 |     return flat.reshape(s, s)
0259 | 
0260 | 
0261 | def image_entropy(img: np.ndarray) -> float:
0262 |     """
0263 |     Calculate Shannon entropy over normalized pixel mass.
0264 | 
0265 |     Measures the information content/distribution uniformity in the image.
0266 |     Higher entropy = more uniform distribution, lower entropy = more concentrated.
0267 | 
0268 |     Args:
0269 |         img: Input image array
0270 | 
0271 |     Returns:
0272 |         Shannon entropy value
0273 |     """
0274 |     x = np.asarray(img, dtype=np.float64)
0275 |     mass = x.sum()
0276 |     if mass <= 0:
0277 |         return 0.0
0278 |     p = (x / mass).reshape(-1)
0279 |     p = p[p > 0]
0280 |     return float(-np.sum(p * np.log(p + 1e-12)))
0281 | 
0282 | 
0283 | def brightness_concentration(img: np.ndarray, tl_frac: float = 0.25) -> float:
0284 |     """
0285 |     Calculate fraction of total mass in the top-left region.
0286 | 
0287 |     Measures how concentrated the high values are in the designated area.
0288 |     Used as a key metric for PHOS effectiveness.
0289 | 
0290 |     Args:
0291 |         img: Input image array
0292 |         tl_frac: Area fraction for top-left region
0293 | 
0294 |     Returns:
0295 |         Concentration ratio (0-1)
0296 |     """
0297 |     x = np.asarray(img, dtype=np.float64)
0298 |     s = x.shape[0]
0299 |     if s == 0:
0300 |         return 0.0
0301 |     area = max(1, int(round(np.sqrt(max(tl_frac, 1e-9)) * s)))
0302 |     area = min(area, s)
0303 |     tl_sum = float(x[:area, :area].sum())
0304 |     total = float(x.sum()) + 1e-12
0305 |     return tl_sum / total
0306 | 
0307 | 
0308 | def save_img(img: np.ndarray, path: str, title: str = "") -> None:
0309 |     """
0310 |     Save grayscale image to disk.
0311 | 
0312 |     Args:
0313 |         img: Image array (values 0-1)
0314 |         path: Output file path
0315 |         title: Image title
0316 |     """
0317 |     Path(path).parent.mkdir(parents=True, exist_ok=True)
0318 |     plt.figure(figsize=(6, 6))
0319 |     plt.imshow(np.clip(img, 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0)
0320 |     plt.title(title)
0321 |     plt.axis("off")
0322 |     plt.tight_layout()
0323 |     plt.savefig(path, dpi=180)
0324 |     plt.close()
0325 | 
0326 | 
0327 | # ---------------------------
0328 | # VPM vectorization
0329 | # ---------------------------
0330 | 
0331 | 
0332 | def vpm_vector_from_df(
0333 |     df: pd.DataFrame,
0334 |     model: str,
0335 |     dimensions: List[str],
0336 |     *,
0337 |     interleave: bool = False,
0338 |     weights: Dict[str, float] | None = None,
0339 |     p_lo: float = 10.0,
0340 |     p_hi: float = 90.0,
0341 | ) -> np.ndarray:
0342 |     """
0343 |     Build a single 1D VPM vector from DataFrame columns.
0344 | 
0345 |     Supports both MultiIndex and flat column naming conventions.
0346 |     Can concatenate or interleave dimensions.
0347 | 
0348 |     Args:
0349 |         df: DataFrame containing model performance scores
0350 |         model: Model identifier (e.g., 'hrm', 'tiny')
0351 |         dimensions: List of dimension names to include
0352 |         interleave: If True, interleave dimensions; if False, concatenate
0353 |         weights: Optional dimension weights for weighted combination
0354 |         p_lo: Lower percentile for robust scaling
0355 |         p_hi: Upper percentile for robust scaling
0356 | 
0357 |     Returns:
0358 |         1D VPM vector combining all specified dimensions
0359 |     """
0360 |     cols = []
0361 |     for dim in dimensions:
0362 |         col = None
0363 |         if isinstance(df.columns, pd.MultiIndex):
0364 |             key = (model, dim)
0365 |             if key in df.columns:
0366 |                 col = df[key].to_numpy()
0367 |         else:
0368 |             flat_key = f"{model}.{dim}"
0369 |             if flat_key in df.columns:
0370 |                 col = df[flat_key].to_numpy()
0371 | 
0372 |         if col is None:
0373 |             col = np.zeros(len(df), dtype=np.float64)
0374 |         col = robust01(col, p_lo=p_lo, p_hi=p_hi)
0375 |         if weights and dim in weights:
0376 |             col = col * float(weights[dim])
0377 |         cols.append(col)
0378 | 
0379 |     if not cols:
0380 |         return np.zeros(0, dtype=np.float64)
0381 | 
0382 |     if interleave:
0383 |         return np.column_stack(cols).reshape(-1)
0384 |     else:
0385 |         return np.concatenate(cols, axis=0)
0386 | 
0387 | 
0388 | # ---------------------------
0389 | # Artifact builders
0390 | # ---------------------------
0391 | 
0392 | 
0393 | def build_vpm_phos_artifacts(
0394 |     df: pd.DataFrame,
0395 |     *,
0396 |     model: str,
0397 |     dimensions: List[str],
0398 |     out_prefix: str,
0399 |     tl_frac: float = 0.25,
0400 |     interleave: bool = False,
0401 |     weights: Dict[str, float] | None = None,
0402 | ) -> Dict:
0403 |     """
0404 |     Build both raw VPM and PHOS-packed artifacts for a single model.
0405 | 
0406 |     Produces:
0407 |       - Raw VPM image (simple reshaping)
0408 |       - PHOS VPM image (sorted packing)
0409 |       - Comprehensive metrics for both
0410 |       - PNG files saved to disk
0411 | 
0412 |     Args:
0413 |         df: Input DataFrame with performance scores
0414 |         model: Target model name
0415 |         dimensions: Evaluation dimensions to include
0416 |         out_prefix: Output path prefix
0417 |         tl_frac: Top-left area fraction for PHOS packing
0418 |         interleave: Whether to interleave dimensions
0419 |         weights: Optional dimension weights
0420 | 
0421 |     Returns:
0422 |         Dictionary containing paths, metrics, and configuration
0423 |     """
0424 |     vec = vpm_vector_from_df(
0425 |         df, model, dimensions, interleave=interleave, weights=weights
0426 |     )
0427 | 
0428 |     # Generate raw and PHOS images
0429 |     raw_img, _ = to_square(vec)
0430 |     phos_img = phos_sort_pack(vec)
0431 | 
0432 |     # Calculate comparison metrics
0433 |     raw_metrics = {
0434 |         "brightness_top_left": brightness_concentration(
0435 |             raw_img, tl_frac=tl_frac
0436 |         ),
0437 |         "mean": float(raw_img.mean()),
0438 |         "std": float(raw_img.std()),
0439 |         "entropy": image_entropy(raw_img),
0440 |     }
0441 |     phos_metrics = {
0442 |         "brightness_top_left": brightness_concentration(
0443 |             phos_img, tl_frac=tl_frac
0444 |         ),
0445 |         "mean": float(phos_img.mean()),
0446 |         "std": float(phos_img.std()),
0447 |         "entropy": image_entropy(phos_img),
0448 |     }
0449 | 
0450 |     # Save visualization files
0451 |     raw_path = f"{out_prefix}_vpm_raw.png"
0452 |     phos_path = f"{out_prefix}_vpm_phos.png"
0453 |     save_img(raw_img, raw_path, title=f"{model.upper()} VPM (raw)")
0454 |     save_img(phos_img, phos_path, title=f"{model.upper()} VPM (PHOS)")
0455 | 
0456 |     return {
0457 |         "model": model,
0458 |         "tl_frac": float(tl_frac),
0459 |         "paths": {"raw": raw_path, "phos": phos_path},
0460 |         "metrics": {"raw": raw_metrics, "phos": phos_metrics},
0461 |     }
0462 | 
0463 | 
0464 | def _chosen_from_sweep(sweep: List[Dict], delta: float) -> Dict:
0465 |     """
0466 |     Select best PHOS candidate from parameter sweep.
0467 | 
0468 |     Selection strategy:
0469 |     1. Prefer first candidate that shows significant improvement over raw
0470 |     2. Fallback to candidate with highest PHOS concentration
0471 | 
0472 |     Args:
0473 |         sweep: List of sweep results
0474 |         delta: Minimum improvement threshold
0475 | 
0476 |     Returns:
0477 |         Selected candidate configuration
0478 |     """
0479 |     cand = sorted(sweep, key=lambda r: r["phos_conc"], reverse=True)
0480 |     for r in cand:
0481 |         if r.get("improved"):
0482 |             return r
0483 |     return cand[0] if cand else {}
0484 | 
0485 | 
0486 | # --- helper: pick best sweep result (prefer improved; else best phos conc) ---
0487 | def _chosen_from_sweep(
0488 |     sweep: List[Dict], *, delta: float = 0.02
0489 | ) -> Optional[Dict]:
0490 |     if not sweep:
0491 |         return None
0492 |     improved = [r for r in sweep if r.get("improved")]
0493 |     return (
0494 |         max(improved, key=lambda r: r.get("phos_conc", 0.0))
0495 |         if improved
0496 |         else max(sweep, key=lambda r: r.get("phos_conc", 0.0))
0497 |     )
0498 | 
0499 | 
0500 | def build_compare_guarded(
0501 |     df: pd.DataFrame,
0502 |     *,
0503 |     dimensions: List[str],
0504 |     out_prefix: str,
0505 |     model_A: str,  # e.g. "hf_HRM" or "Llama3-8B"
0506 |     model_B: str,  # e.g. "hf_TinyLama" or "Phi-3-mini"
0507 |     tl_fracs: Iterable[float] = (0.25, 0.16, 0.36, 0.09),
0508 |     delta: float = 0.02,
0509 |     interleave: bool = False,
0510 |     weights: Dict[str, float] | None = None,
0511 | ) -> Dict:
0512 |     """
0513 |     Compare two models (by alias) with a PHOS guard sweep.
0514 | 
0515 |     - Sweeps multiple tl_frac values per model
0516 |     - Selects the best (guarded) PHOS config per model
0517 |     - Builds a PHOS difference visualization (A − B) if shapes match
0518 |     - Emits a summary JSON keyed by the real model aliases
0519 |     """
0520 |     # Ensure parent folder exists for all outputs derived from out_prefix
0521 |     Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
0522 | 
0523 |     results: Dict[str, Dict] = {"sweep": {}, "models": [model_A, model_B]}
0524 | 
0525 |     # ---- 1) Per-model sweeps -------------------------------------------------
0526 |     for model in (model_A, model_B):
0527 |         model_sweep: List[Dict] = []
0528 |         for tl in tl_fracs:
0529 |             prefix = f"{out_prefix}_{model}_tl{float(tl):.2f}"
0530 | 
0531 |             # Build both raw and PHOS artifacts for this (model, tl)
0532 |             res = build_vpm_phos_artifacts(
0533 |                 df,
0534 |                 model=model,
0535 |                 dimensions=dimensions,
0536 |                 out_prefix=prefix,
0537 |                 tl_frac=float(tl),
0538 |                 interleave=interleave,
0539 |                 weights=weights,
0540 |             )
0541 | 
0542 |             raw_c = float(res["metrics"]["raw"]["brightness_top_left"])
0543 |             phos_c = float(res["metrics"]["phos"]["brightness_top_left"])
0544 |             improved = phos_c > raw_c * (1.0 + float(delta))
0545 | 
0546 |             model_sweep.append(
0547 |                 {
0548 |                     "tl_frac": float(tl),
0549 |                     "raw_conc": raw_c,
0550 |                     "phos_conc": phos_c,
0551 |                     "improved": bool(improved),
0552 |                     "raw_path": res["paths"]["raw"],
0553 |                     "phos_path": res["paths"]["phos"],
0554 |                 }
0555 |             )
0556 | 
0557 |         chosen = _chosen_from_sweep(model_sweep, delta=delta)
0558 |         results["sweep"][model] = model_sweep
0559 |         results[f"{model}_chosen"] = chosen
0560 | 
0561 |         # Persist the sweep details for this model
0562 |         with open(
0563 |             f"{out_prefix}_{model}_vpm_guard_metrics.json",
0564 |             "w",
0565 |             encoding="utf-8",
0566 |         ) as f:
0567 |             json.dump(
0568 |                 {
0569 |                     "model": model,
0570 |                     "delta": float(delta),
0571 |                     "sweep": model_sweep,
0572 |                     "chosen": chosen,
0573 |                 },
0574 |                 f,
0575 |                 indent=2,
0576 |             )
0577 | 
0578 |         # Convenience: copy chosen PHOS to a stable name (ignore if not available)
0579 |         if chosen and chosen.get("phos_path"):
0580 |             import shutil
0581 | 
0582 |             dst = f"{out_prefix}_{model}_vpm_chosen.png"
0583 |             try:
0584 |                 shutil.copyfile(chosen["phos_path"], dst)
0585 |             except Exception:
0586 |                 pass
0587 | 
0588 |     # ---- 2) PHOS(A) − PHOS(B) visualization (using current DataFrame) -------
0589 |     # Use your existing vectorizer + packer. If either fails, we just skip diff.
0590 |     try:
0591 |         vec_A = vpm_vector_from_df(
0592 |             df, model_A, dimensions, interleave=interleave, weights=weights
0593 |         )
0594 |         vec_B = vpm_vector_from_df(
0595 |             df, model_B, dimensions, interleave=interleave, weights=weights
0596 |         )
0597 |         img_A = phos_sort_pack(vec_A)
0598 |         img_B = phos_sort_pack(vec_B)
0599 | 
0600 |         if img_A.shape == img_B.shape and img_A.size and img_B.size:
0601 |             diff = img_A - img_B
0602 |             dmin = float(diff.min())
0603 |             dmax = float(diff.max())
0604 |             # Normalize to [0,1] for viewing
0605 |             diff01 = (diff - dmin) / (dmax - dmin + 1e-12)
0606 |             save_img(
0607 |                 diff01,
0608 |                 f"{out_prefix}_vpm_chosen_diff.png",
0609 |                 title=f"PHOS({model_A}) − PHOS({model_B})",
0610 |             )
0611 |             results["diff_range"] = [dmin, dmax]
0612 |         else:
0613 |             results["diff_range"] = None
0614 |     except Exception:
0615 |         # Non-fatal: the sweeps and chosen outputs are still useful
0616 |         results["diff_range"] = None
0617 | 
0618 |     # ---- 3) Summary (keyed by real aliases) ---------------------------------
0619 |     summary = {
0620 |         "delta": float(delta),
0621 |         "chosen": {
0622 |             model_A: results.get(f"{model_A}_chosen"),
0623 |             model_B: results.get(f"{model_B}_chosen"),
0624 |         },
0625 |     }
0626 |     with open(f"{out_prefix}_guard_compare.json", "w", encoding="utf-8") as f:
0627 |         json.dump(summary, f, indent=2)
0628 | 
0629 |     results["summary"] = summary
0630 |     return results
0631 | 
0632 | 
0633 | def pick_metric_column(df: pd.DataFrame, base: str) -> str | None:
0634 |     for suf in CAND_SUFFIXES:
0635 |         cand = f"{base}{suf}"
0636 |         if cand in df.columns:
0637 |             return cand
0638 |     pref = f"{base}."
0639 |     for c in df.columns:
0640 |         if isinstance(c, str) and c.startswith(pref):
0641 |             return c
0642 |     return None
0643 | 
0644 | 
0645 | def project_dimensions(df_in: pd.DataFrame, dims: list[str]) -> pd.DataFrame:
0646 |     out = {"node_id": df_in["node_id"].values}
0647 |     missing = {"hrm": [], "tiny": []}
0648 |     for d in dims:
0649 |         h = pick_metric_column(df_in, f"hrm.{d}")
0650 |         t = pick_metric_column(df_in, f"tiny.{d}")
0651 | 
0652 |         if h is None:
0653 |             missing["hrm"].append(d)
0654 |             out[f"hrm.{d}"] = 0.0
0655 |         else:
0656 |             out[f"hrm.{d}"] = df_in[h].astype(float).fillna(0.0)
0657 |         if t is None:
0658 |             missing["tiny"].append(d)
0659 |             out[f"tiny.{d}"] = 0.0
0660 |         else:
0661 |             out[f"tiny.{d}"] = df_in[t].astype(float).fillna(0.0)
```
