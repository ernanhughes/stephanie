import json
import time
from pathlib import Path

from tqdm import tqdm

from policy.custom_types import EvaluationResult


class SummarizationRunner:

    def __init__(self, support_analyzer):
        self.support_analyzer = support_analyzer

    def run(self, samples, out_path):

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        results = []
        start_time = time.time()

        with out_path.open("w", encoding="utf-8") as f:

            for i, sample in enumerate(tqdm(samples, desc="Analyzing summaries")):

                support_diag = self.support_analyzer.analyze(
                    summary_text=sample["claim"],
                    evidence_text=sample["evidence"][0],
                )

                evaluation = EvaluationResult(
                    claim=sample["claim"],
                    evidence=sample["evidence"],
                    energy_result=None,
                    decision_trace=None,
                    verdict=None,
                    policy_applied="support_analysis",
                    run_id="summary_v1",
                    split="test",
                    effectiveness=0.0,
                    embedding_info={},
                    support_diagnostics=support_diag,
                    label=sample["label"],
                )

                results.append(evaluation)

                # ✅ Stream write immediately
                f.write(json.dumps(evaluation.to_dict()) + "\n")

                # Optional: flush every 50 samples
                if i % 50 == 0:
                    f.flush()

        elapsed = time.time() - start_time
        print(f"\nCompleted {len(samples)} samples in {elapsed:.2f}s")

        return results

    def _write_jsonl(self, results, out_path: str):

        def convert(o):
            import numpy as np

            if isinstance(o, (np.float32, np.float64)):
                return float(o)
            if isinstance(o, (np.int32, np.int64)):
                return int(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=convert)


    def _write_csv(self, results, csv_path):
        import pandas as pd

        rows = []
        for r in results:
            row = r.support_diagnostics.to_dict()
            row["label"] = r.verdict  # or sample label I
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
