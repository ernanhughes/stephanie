"""
JAS Vital-Signs Dashboard – Streamlit panel that subscribes to telemetry.

Run:
  streamlit run stephanie/components/jitter/jas_dashboard.py

Notes:
- Uses your existing JetStream helper `get_js()` to subscribe to telemetry.
- Shows energy pools, boundary integrity, homeostasis correction, stress.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict

import pandas as pd
import streamlit as st

from stephanie.services.bus.nats_client import get_js  # async JetStream helper

TELEM_SUBJECT = "arena.jitter.telemetry"


# ----------------------- async consumer -------------------------------------

async def _consume_telemetry(queue):
    js = await get_js()
    sub = await js.subscribe(TELEM_SUBJECT)
    async for msg in sub:
        try:
            payload = json.loads(msg.data.decode())
            # Accept both legacy and current formats
            if payload.get("type") == "jas_telemetry":
                data = payload.get("data", {})
                tick = payload.get("tick", 0)
            else:
                data = payload
                tick = data.get("tick", 0)

            packet = {
                "ts": payload.get("ts", time.time()),
                "tick": tick,
                "energy_cognitive": data.get("energy", {}).get("cognitive", None),
                "energy_metabolic": data.get("energy", {}).get("metabolic", None),
                "energy_reserve": data.get("energy", {}).get("reserve", None),
                "boundary_integrity": data.get("boundary_integrity", None),
                "homeo_correction": data.get("homeo_correction", 0.0),
                "stress": data.get("stress", 0.0),
                "status": data.get("status", "alive"),
            }
            queue.append(packet)
            # bound memory
            if len(queue) > 1000:
                del queue[: len(queue) - 1000]
        except Exception:
            continue


# ------------------------------ UI -----------------------------------------

def main():
    st.set_page_config(page_title="Jitter Vital Signs", page_icon="🌱", layout="wide")
    st.title("🌱 Jitter Autopoietic Vital Signs")

    # Shared buffer across reruns
    if "packets" not in st.session_state:
        st.session_state["packets"] = []

    # Kick off consumer once
    if "consumer_started" not in st.session_state:
        st.session_state["consumer_started"] = True
        asyncio.get_event_loop().create_task(_consume_telemetry(st.session_state["packets"]))

    # Layout
    col1, col2, col3 = st.columns([2, 2, 1])

    # Charts
    with col1:
        st.subheader("Energy Pools")
        energy_df = _df_from_packets(st.session_state["packets"], ["energy_cognitive", "energy_metabolic", "energy_reserve"])
        st.line_chart(energy_df.set_index("tick"))

    with col2:
        st.subheader("Boundary & Stress")
        bounds_df = _df_from_packets(st.session_state["packets"], ["boundary_integrity", "stress"])
        st.line_chart(bounds_df.set_index("tick"))

    with col3:
        st.subheader("Homeostasis")
        homeo_df = _df_from_packets(st.session_state["packets"], ["homeo_correction"])
        st.line_chart(homeo_df.set_index("tick"))
        if st.session_state["packets"]:
            latest = st.session_state["packets"][-1]
            st.metric("Status", latest.get("status", "alive"))
            st.metric("Boundary Integrity", f"{latest.get('boundary_integrity', 0.0):.3f}")
            st.metric("Metabolic Energy", f"{latest.get('energy_metabolic', 0.0):.2f}")

    st.caption("Tip: keep this panel open while the JitterLifecycleAgent runs.")

    # light auto-refresh
    st.experimental_rerun()


def _df_from_packets(packets: list, fields: list) -> pd.DataFrame:
    rows = []
    for p in packets[-400:]:
        row = {"tick": p.get("tick", 0)}
        for f in fields:
            row[f] = p.get(f, None)
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame([{"tick": 0, **{f: 0 for f in fields}}])


# Streamlit’s event loop integration
if __name__ == "__main__":
    # Create an event loop if one doesn't exist (Streamlit sometimes runs w/o)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_eventLoop = loop  # no-op for mypy
        asyncio.set_event_loop(loop)
    main()
