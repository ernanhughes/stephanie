# stephanie/reports/report_generator.py
class ReportGenerator:
    def generate(self, cartridge):
        """Generate comprehensive research report"""
        report = "# Evolutionary Research Report\n\n"
        report += f"**Goal:** {cartridge.goal}\n"
        report += f"**Generations:** {cartridge.generation}\n"
        report += f"**Final Quality Score:** {cartridge.quality_metrics['overall_score']:.2f}/1.0\n\n"

        report += "## Core Thesis\n"
        report += f"{cartridge.schema['core_thesis']}\n\n"

        report += "## Key Findings\n"
        for i, evidence in enumerate(cartridge.schema["supporting_evidence"]):
            report += f"{i + 1}. {evidence['content']}  \n"
            report += f"   *Source: {evidence['source']}, Confidence: {evidence['confidence']:.2f}*\n\n"

        report += "## Testable Hypotheses\n"
        for i, hypothesis in enumerate(cartridge.schema["hypotheses"]):
            report += f"{i + 1}. **{hypothesis['content']}**  \n"
            report += f"   *Confidence: {hypothesis['confidence']:.2f}*\n"
            if hypothesis["id"] in cartridge.schema["validation_protocols"]:
                protocol = cartridge.schema["validation_protocols"][hypothesis["id"]]
                report += f"   Validation Protocol: {protocol}\n"
            report += "\n"

        report += "## Quality Metrics\n"
        for metric, score in cartridge.quality_metrics.items():
            if metric != "overall_score":
                report += f"- **{metric.capitalize()}:** {score:.2f}/1.0\n"

        report += "\n## Research Evolution\n"
        report += "This report represents the culmination of an evolutionary research process:\n"
        report += "- Multiple generations of research refinement\n"
        report += "- Distributed exploration by specialized agents\n"
        report += "- Competitive selection of best findings\n"
        report += f"- Synthesized from {len(cartridge.schema['supporting_evidence'])} evidence sources\n"

        return report
