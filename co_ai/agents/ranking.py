from haystack import component

@component
class RankingComponent:
    def run(self, optimist_reviews: list[str], skeptic_reviews: list[str]) -> dict:
        print("[RankingComponent] Combining reviews and ranking hypotheses...")
        combined = optimist_reviews + skeptic_reviews
        ranked = sorted(combined, key=len)  # simplistic ranking by text length
        return {"ranked_hypotheses": ranked}
