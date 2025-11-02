# stephanie/components/jitter/telemetry/dashboard.py
"""
dashboard.py
============
SIS (Stephanie Information System) dashboard for Jitter Autopoietic System.

This module provides a Streamlit-based dashboard that subscribes to telemetry
and visualizes the vital signs of Jitter organisms in real-time.

Key Features:
- Real-time visualization of energy pools
- Boundary integrity monitoring
- Homeostasis correction visualization
- Crisis detection and alerting
- Historical data analysis
- Responsive dashboard layout
- Integration with Stephanie's monitoring infrastructure
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from stephanie.components.jitter.telemetry.jas_telemetry import VitalSigns
from stephanie.services.bus.nats_client import get_js  # async JetStream helper

log = logging.getLogger("stephanie.jitter.dashboard")

TELEM_SUBJECT = "arena.jitter.telemetry"

class JASDashboard:
    """
    JAS Vital-Signs Dashboard for real-time monitoring.
    
    This dashboard:
    - Subscribes to telemetry from Jitter organisms
    - Visualizes energy pools, boundary integrity, and homeostasis
    - Shows crisis detection and alerting
    - Provides historical data for analysis
    - Supports interactive exploration of system state
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.js = None
        self.running = False
        self.vital_signs_history: List[VitalSigns] = []
        self.max_history = self.config.get("max_history", 1000)
        self.refresh_interval = self.config.get("refresh_interval", 5.0)
        
        # Initialize Streamlit
        st.set_page_config(
            page_title="Jitter Autopoietic System Dashboard",
            page_icon="🌱",
            layout="wide"
        )
        
        st.title("🌱 Jitter Autopoietic System Dashboard")
        st.markdown("""
        Real-time monitoring of Jitter organisms' vital signs and system health.
        """)
        
        log.info("JAS Dashboard initialized")
    
    async def connect(self):
        """Establish JetStream connection"""
        try:
            self.js = await get_js()
            log.info("JetStream connection established for dashboard")
        except Exception as e:
            log.error(f"Failed to establish JetStream connection: {str(e)}")
            self.js = None
    
    async def subscribe_to_telemetry(self):
        """Subscribe to telemetry from Jitter organisms"""
        if not self.js:
            await self.connect()
            
        if not self.js:
            st.error("Failed to connect to telemetry system")
            return
        
        try:
            # Subscribe to telemetry subject
            async def message_handler(msg):
                try:
                    # Parse the incoming message
                    data = json.loads(msg.data.decode())
                    
                    # Check if it's a vital signs message
                    if data.get("type") == "jas_telemetry":
                        vital_signs = VitalSigns(**data["data"])
                        self._process_vital_signs(vital_signs)
                        
                        # Update dashboard display
                        self._update_dashboard_display()
                        
                    elif data.get("type") == "jas_artifact":
                        # Handle artifact messages (like reproduction/apoptosis)
                        self._handle_artifact_message(data)
                        
                except Exception as e:
                    log.error(f"Error processing telemetry message: {str(e)}")
            
            # Create subscription
            await self.js.subscribe(TELEM_SUBJECT, cb=message_handler)
            log.info("Subscribed to telemetry subject")
            
        except Exception as e:
            log.error(f"Failed to subscribe to telemetry: {str(e)}")
    
    def _process_vital_signs(self, vital_signs: VitalSigns):
        """Process incoming vital signs"""
        self.vital_signs_history.append(vital_signs)
        
        # Keep history bounded
        if len(self.vital_signs_history) > self.max_history:
            self.vital_signs_history.pop(0)
    
    def _handle_artifact_message(self, data: Dict[str, Any]):
        """Handle JAF artifact messages"""
        artifact_type = data.get("artifact_type", "unknown")
        timestamp = data.get("timestamp", time.time())
        
        st.info(f"📊 Artifact received: {artifact_type} at {timestamp}")
        log.info(f"Received JAF artifact: {artifact_type}")
    
    def _update_dashboard_display(self):
        """Update the dashboard display with latest data"""
        # Get latest data
        if not self.vital_signs_history:
            st.warning("No telemetry data available yet...")
            return
        
        latest = self.vital_signs_history[-1]
        
        # Display system overview
        self._display_system_overview(latest)
        
        # Display energy pools
        self._display_energy_pools(latest)
        
        # Display boundary integrity
        self._display_boundary_integrity(latest)
        
        # Display homeostasis
        self._display_homeostasis(latest)
        
        # Display crisis detection
        self._display_crisis_detection(latest)
        
        # Display historical data
        self._display_historical_data()
    
    def _display_system_overview(self, vital_signs: VitalSigns):
        """Display system overview cards"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Health Score", f"{vital_signs.health_score:.2f}")
        with col2:
            st.metric("Crisis Level", f"{vital_signs.crisis_level:.2f}")
        with col3:
            st.metric("Tick", vital_signs.tick)
        with col4:
            st.metric("Timestamp", time.ctime(vital_signs.timestamp))
        
        # Display alerts
        if vital_signs.alerts:
            st.warning(f"⚠️ Alerts: {', '.join(vital_signs.alerts)}")
    
    def _display_energy_pools(self, vital_signs: VitalSigns):
        """Display energy pool visualization"""
        st.subheader("🔋 Energy Pools")
        
        # Create energy bar chart
        energy_data = {
            "Pool": ["Cognitive", "Metabolic", "Reserve"],
            "Amount": [
                vital_signs.energy_cognitive,
                vital_signs.energy_metabolic,
                vital_signs.energy_reserve
            ]
        }
        
        df_energy = pd.DataFrame(energy_data)
        
        # Display bars
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cognitive", f"{vital_signs.energy_cognitive:.1f}")
        with col2:
            st.metric("Metabolic", f"{vital_signs.energy_metabolic:.1f}")
        with col3:
            st.metric("Reserve", f"{vital_signs.energy_reserve:.1f}")
        
        # Show chart
        st.bar_chart(df_energy.set_index("Pool"))
    
    def _display_boundary_integrity(self, vital_signs: VitalSigns):
        """Display boundary integrity metrics"""
        st.subheader("🛡️ Boundary Integrity")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Integrity", f"{vital_signs.boundary_integrity:.2f}")
        with col2:
            st.metric("Stress", f"{vital_signs.threat_level:.2f}")
        
        # Visual indicator
        integrity_color = "green" if vital_signs.boundary_integrity > 0.7 else \
                         "orange" if vital_signs.boundary_integrity > 0.4 else "red"
        
        st.markdown(f"""
        <div style="background-color: {integrity_color}; padding: 10px; border-radius: 5px;">
            <strong>Boundary Status:</strong> {'Healthy' if vital_signs.boundary_integrity > 0.7 else 
            'At Risk' if vital_signs.boundary_integrity > 0.4 else 'Critical'}
        </div>
        """, unsafe_allow_html=True)
    
    def _display_homeostasis(self, vital_signs: VitalSigns):
        """Display homeostasis metrics"""
        st.subheader("🔄 Homeostasis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cognitive Flow", f"{vital_signs.cognitive_integrated:.2f}")
        with col2:
            st.metric("Reasoning Depth", vital_signs.reasoning_depth)
        with col3:
            st.metric("Emotional Valence", f"{vital_signs.emotional_valence:.2f}")
        
        # Display attention weights if available
        if vital_signs.layer_attention:
            st.write("Attention Weights:")
            attention_df = pd.DataFrame([
                {"Layer": k, "Weight": v} 
                for k, v in vital_signs.layer_attention.items()
            ])
            st.table(attention_df)
    
    def _display_crisis_detection(self, vital_signs: VitalSigns):
        """Display crisis detection and alerting"""
        st.subheader("🚨 Crisis Detection")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Crisis Level", f"{vital_signs.crisis_level:.2f}")
        with col2:
            st.metric("Veto Layer", vital_signs.layer_veto)
        with col3:
            st.metric("Alerts", len(vital_signs.alerts))
        
        # Show alerts
        if vital_signs.alerts:
            st.error("Active Alerts:")
            for alert in vital_signs.alerts:
                st.text(f"• {alert}")
    
    def _display_historical_data(self):
        """Display historical data analysis"""
        st.subheader("📈 Historical Data")
        
        if len(self.vital_signs_history) < 2:
            st.info("Not enough historical data for analysis")
            return
        
        # Create historical DataFrame
        hist_data = []
        for vs in self.vital_signs_history[-100:]:  # Last 100 readings
            hist_data.append({
                "Tick": vs.tick,
                "Health": vs.health_score,
                "Crisis": vs.crisis_level,
                "Integrity": vs.boundary_integrity,
                "Cognitive": vs.energy_cognitive,
                "Metabolic": vs.energy_metabolic,
                "Reserve": vs.energy_reserve
            })
        
        df_hist = pd.DataFrame(hist_data)
        
        # Display charts
        col1, col2 = st.columns(2)
        with col1:
            st.line_chart(df_hist[["Health", "Crisis"]])
        with col2:
            st.line_chart(df_hist[["Integrity", "Cognitive", "Metabolic", "Reserve"]])
        
        # Display recent data table
        st.dataframe(df_hist.tail(10))
    
    def run(self):
        """Run the dashboard"""
        st.write("Starting JAS Dashboard...")
        
        # Initialize session state
        if "dashboard_running" not in st.session_state:
            st.session_state.dashboard_running = False
        
        # Connect to telemetry
        if not self.js:
            try:
                asyncio.run(self.connect())
            except Exception as e:
                st.error(f"Failed to connect to telemetry: {str(e)}")
                return
        
        # Start subscription
        if not st.session_state.dashboard_running:
            st.session_state.dashboard_running = True
            try:
                asyncio.run(self.subscribe_to_telemetry())
            except Exception as e:
                st.error(f"Failed to start telemetry subscription: {str(e)}")
                return
        
        # Main dashboard loop
        while st.session_state.dashboard_running:
            # Update display
            self._update_dashboard_display()
            
            # Refresh delay
            time.sleep(self.refresh_interval)
    
    def stop(self):
        """Stop the dashboard"""
        st.session_state.dashboard_running = False
        log.info("JAS Dashboard stopped")

# Streamlit app runner
def run_dashboard():
    """Run the JAS Dashboard as a Streamlit app"""
    # Initialize dashboard
    dashboard = JASDashboard()
    
    # Run the dashboard
    dashboard.run()

# Example usage (would be called from Streamlit)
if __name__ == "__main__":
    run_dashboard()