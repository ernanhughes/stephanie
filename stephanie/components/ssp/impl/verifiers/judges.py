# stephanie/components/ssp/impl/verifiers/judges.py
from ast import Dict


def solver_three_line_judge(verify_fn):
    """
    verify_fn: callable(ground_truth:str, predicted:str) -> (ok:bool, score:float)
    """
    import re
    line = re.compile(r'^\s*([a-zA-Z_]+)\s*:\s*(.+?)\s*$')
    def parse_three(txt: str):
        out = {"rationale":"", "score":"", "result":""}
        for ln in (txt or "").splitlines():
            m = line.match(ln.strip())
            if m: out[m.group(1).lower()] = m.group(2).strip()
        return out

    def judge(outputs: Dict[str, str]) -> tuple[str, Dict[str, float]]:
        scores = {}
        for k, v in outputs.items():
            p = parse_three(v)
            ok, ver = verify_fn(ground_truth="", predicted=p["result"])  # or your ground truth route
            # combine self-reported score (scaled) and external ver score
            try:
                self_score = min(max(int(p["score"]), 0), 100) / 100.0
            except Exception: 
                self_score = 0.0
            scores[k] = 0.4 * self_score + 0.6 * ver
        winner = max(scores.items(), key=lambda kv: kv[1])[0] if scores else None
        return winner, scores
    return judge
