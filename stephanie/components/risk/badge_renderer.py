# stephanie/components/gap/risk/badge_renderer.py
from __future__ import annotations

import base64
from typing import Dict, Optional


def _clamp01(x: float) -> float:
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return float(x)


class BadgeRenderer:
    """
    Renders a 256x256 (configurable) SVG badge as a data URI.

    Quadrants:
      TL = Confidence       (good scale; greener is better)
      TR = FaithfulnessRisk (risk scale; redder is worse)
      BL = OOD Risk         (risk scale)
      BR = Δ-gap            (risk scale)
    Outer ring thickness ~ confidence (evidence mass).
    Optional sparkline along bottom for token-entropy trend.
    """

    def __init__(self, size: int = 256) -> None:
        self.size = int(size)

    # -------------------------- colors ---------------------------------
    @staticmethod
    def _risk_color01(x: float) -> str:
        x = _clamp01(x)
        r = int(round(255 * x))
        g = int(round(255 * (1.0 - x)))
        return f"rgb({r},{g},0)"

    @staticmethod
    def _good_color01(x: float) -> str:
        x = _clamp01(x)
        g = int(round(255 * x))
        r = int(round(255 * (1.0 - x)))
        return f"rgb({r},{g},0)"

    # -------------------------- render ---------------------------------
    def render_svg(
        self,
        *,
        metrics01: Dict[str, float],
        decision: str,
        thresholds: Dict[str, float],
        sparkline: Optional[list] = None,
        theme: str = "light",  # "light" | "dark"
    ) -> str:
        s = self.size
        q = s // 2

        conf = float(metrics01.get("confidence01", 0.5))
        faith = float(metrics01.get("faithfulness_risk01", 0.5))
        ood = float(metrics01.get("ood_hat01", 0.5))
        delta = float(metrics01.get("delta_gap01", 0.5))

        # colors
        bg = "#ffffff" if theme == "light" else "#0b0b0b"
        stroke = "#000000" if theme == "light" else "#f2f2f2"

        c_conf = self._good_color01(conf)
        c_faith = self._risk_color01(faith)
        c_ood = self._risk_color01(ood)
        c_delta = self._risk_color01(delta)

        ring = 4 + int(10 * conf)   # evidence mass

        glyph = {
            "OK": "✓",
            "WATCH": "!",
            "RISK": "✕",
        }.get(decision, "?")

        svg = [
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{s}' height='{s}' viewBox='0 0 {s} {s}'>",
            f"<rect x='0' y='0' width='{s}' height='{s}' fill='{bg}' rx='32' ry='32'/>",
            # quadrants
            f"<rect x='0'   y='0'   width='{q}' height='{q}' fill='{c_conf}'/>",   # TL
            f"<rect x='{q}' y='0'   width='{q}' height='{q}' fill='{c_faith}'/>",  # TR
            f"<rect x='0'   y='{q}' width='{q}' height='{q}' fill='{c_ood}'/>",    # BL
            f"<rect x='{q}' y='{q}' width='{q}' height='{q}' fill='{c_delta}'/>",  # BR
            # ring
            f"<circle cx='{q}' cy='{q}' r='{q - ring/2 - 2}' stroke='{stroke}' stroke-width='{ring}' fill='none' opacity='0.25'/>",
            # center glyph
            f"<text x='{q}' y='{q + 12}' text-anchor='middle' font-size='{int(s*0.45)}' font-family='Inter,Arial' fill='{stroke}'>{glyph}</text>",
        ]

        # thresholds markers (TR faithfulness; BL ood)
        fy = int(q * (1.0 - float(thresholds.get('faithfulness', 0.35))))
        svg.append(f"<line x1='{s-4}' y1='{fy}' x2='{s-q}' y2='{fy}' stroke='{stroke}' stroke-width='2' opacity='0.35'/>")
        oy_tick = q + int(q * float(thresholds.get('ood', 0.30)))
        svg.append(f"<line x1='0' y1='{oy_tick}' x2='{q}' y2='{oy_tick}' stroke='{stroke}' stroke-width='2' opacity='0.35'/>")

        # optional sparkline
        if sparkline:
            w = s - 32
            h = 32
            ox, oy = 16, s - h - 8
            vals = [_clamp01(float(v)) for v in sparkline[-64:]]
            if len(vals) > 1:
                step = w / (len(vals) - 1)
                points = []
                for i, v in enumerate(vals):
                    x = ox + i * step
                    y = oy + (1.0 - v) * h
                    points.append(f"{x:.2f},{y:.2f}")
                svg.append(f"<polyline fill='none' stroke='{stroke}' stroke-width='2' points='{' '.join(points)}' opacity='0.6' />")

        svg.append("</svg>")
        return "".join(svg)

    def render_data_uri(
        self,
        *,
        metrics01: Dict[str, float],
        decision: str,
        thresholds: Dict[str, float],
        sparkline: Optional[list] = None,
        theme: str = "light",
    ) -> str:
        svg = self.render_svg(
            metrics01=metrics01,
            decision=decision,
            thresholds=thresholds,
            sparkline=sparkline,
            theme=theme,
        )
        enc = base64.b64encode(svg.encode("utf-8")).decode("ascii")
        return f"data:image/svg+xml;base64,{enc}"
