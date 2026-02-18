import numpy as np

from policy.custom_types import SupportDiagnostics
from policy.utils.text_utils import split_into_paragraphs, split_into_sentences


class SentenceSupportAnalyzer:

    def __init__(self, embedder, energy_computer=None, entailment_model=None, top_k=3):
        self.embedder = embedder
        self.energy_computer = energy_computer
        self.entailment_model = entailment_model
        self.top_k = top_k

    def analyze(self, summary_text, evidence_text):

        paragraphs = split_into_paragraphs(evidence_text)
        para_vecs = self.embedder.embed(paragraphs)

        sentences = split_into_sentences(summary_text)

        if not sentences:
            return None

        # ---------------------------------------
        # Collect top-k paragraph candidates
        # ---------------------------------------

        all_pairs = []
        sentence_meta = []

        sentence_energies = []
        sim_top1_vals = []
        sim_margin_vals = []
        coverage_vals = []

        for sent in sentences:

            sent_vec = self.embedder.embed([sent])[0]

            sims = para_vecs @ sent_vec
            idx = np.argsort(-sims)[:self.top_k]

            candidates = [paragraphs[i] for i in idx]

            # Save similarity metrics
            sims_sorted = np.sort(sims)[::-1]
            sim_top1_vals.append(float(sims_sorted[0]))
            sim_margin_vals.append(
                float(sims_sorted[0] - sims_sorted[1])
                if len(sims_sorted) > 1 else 0.0
            )

            coverage_vals.append(float(np.mean(sims > 0.3)))

            # Energy (optional)
            if self.energy_computer:
                result = self.energy_computer.compute(
                    claim_vec=sent_vec,
                    evidence_vecs=para_vecs
                )
                sentence_energies.append(result.energy)
            else:
                sentence_energies.append(0.0)

            # Collect entailment pairs
            if self.entailment_model:
                for p in candidates:
                    all_pairs.append((p, sent))
                sentence_meta.append(len(candidates))

        # ---------------------------------------
        # Batched entailment
        # ---------------------------------------

        entailment_scores = []

        if self.entailment_model and all_pairs:
            entailment_scores = self.entailment_model.score_pairs(all_pairs)

        # ---------------------------------------
        # Restructure entailment scores
        # ---------------------------------------

        entailment_max = []
        entailment_mean = []
        entailment_min = []

        pointer = 0

        for count in sentence_meta:
            sent_scores = entailment_scores[pointer:pointer+count]
            pointer += count

            if sent_scores:
                entailment_max.append(max(sent_scores))
                entailment_mean.append(np.mean(sent_scores))
                entailment_min.append(min(sent_scores))
            else:
                entailment_max.append(0.0)
                entailment_mean.append(0.0)
                entailment_min.append(0.0)

        # ---------------------------------------
        # Aggregate into SupportDiagnostics
        # ---------------------------------------

        sentence_energies = np.array(sentence_energies)
        sim_top1_vals = np.array(sim_top1_vals)
        sim_margin_vals = np.array(sim_margin_vals)
        coverage_vals = np.array(coverage_vals)

        entailment_max = np.array(entailment_max) if entailment_max else np.zeros(len(sentences))
        entailment_mean = np.array(entailment_mean) if entailment_mean else np.zeros(len(sentences))
        entailment_min = np.array(entailment_min) if entailment_min else np.zeros(len(sentences))

        threshold = 0.5  # This can be tuned based on validation data

        return SupportDiagnostics(
            sentence_count=len(sentences),
            paragraph_count=len(paragraphs),

            max_entailment=float(np.max(entailment_max)),
            mean_entailment=float(np.mean(entailment_mean)),
            min_entailment=float(np.min(entailment_min)),

            mean_sim_top1=float(np.mean(sim_top1_vals)),
            min_sim_top1=float(np.min(sim_top1_vals)),
            mean_sim_margin=float(np.mean(sim_margin_vals)),
            min_sim_margin=float(np.min(sim_margin_vals)),

            mean_coverage=float(np.mean(coverage_vals)),
            min_coverage=float(np.min(coverage_vals)),

            max_energy=float(np.max(sentence_energies)),
            mean_energy=float(np.mean(sentence_energies)),
            min_energy = float(np.min(sentence_energies)),
            high_energy_count = int(np.sum(sentence_energies > threshold)),
            p90_energy=float(np.percentile(sentence_energies, 90)),
            frac_above_threshold=float(np.mean(sentence_energies > 0.5)),
        )
