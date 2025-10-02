// stephanie/static/arena/js/arena_live.js
import { ArenaState } from './state/arena_state.js';
import { ChartsComponent } from './components/charts.js';
import { GridComponent } from './components/grid.js';
import { ProvenanceCardComponent } from './components/provenance_card.js';
import { EventHandler } from './services/event_handler.js';
import { ApiClient } from './services/api_client.js';

(() => {
  "use strict";

  // Declare components at module scope but don't initialize yet
  let arenaState, provenanceCard, grid, charts, eventHandler, api;
  let es = null; // EventSource connection

  // DOM helpers
  function $(selector) {
    if (selector.startsWith("#")) {
      return document.getElementById(selector.substring(1));
    }
    return document.getElementById(selector);
  }

  // UI helpers
  function setStatus(s) {
    const el = $("status-dot");
    if (!el) return;
    
    el.textContent = s;
    el.className = "badge " + (s === "open" ? "bg-success" : 
                              s === "connecting" ? "bg-warning" : 
                              s === "error" ? "bg-danger" : "bg-secondary");
  }
  
  function showError(msg) {
    const errEl = $("errbar");
    const msgEl = $("errmsg");
    if (!errEl || !msgEl) return;
    
    msgEl.textContent = msg || "Unknown error";
    errEl.classList.remove("d-none");
  }
  
  function hideError() {
    const errEl = $("errbar");
    if (errEl) errEl.classList.add("d-none");
  }
  
  function setChip(id, text) {
    const el = $(id);
    if (!text) { 
      if (el) el.classList.add("d-none"); 
      return; 
    }
    if (el) {
      el.textContent = text;
      el.classList.remove("d-none");
    }
  }

  function tryParseJSON(x) {
    if (typeof x !== "string") return x;
    const t = x.trim();
    if (!t) return x;
    try { return JSON.parse(t); } catch { return x; }
  }

  function addTimeline(ev) {
    const timelineEl = $("timeline");
    if (!timelineEl) return;
    
    const d = document.createElement("div");
    d.className = "small";
    
    const extra = [];
    if (ev.round != null) extra.push("r" + ev.round);
    if (ev.best_overall != null) extra.push("best "+ Number(ev.best_overall).toFixed(3));
    if (ev.marginal_per_ktok != null) extra.push("marg/kTok "+ Number(ev.marginal_per_ktok).toFixed(3));
    if (ev.reason) extra.push("reason " + ev.reason);
    
    const sum = ev.summary || ev.arena_summary;
    if (sum?.winner_overall != null) extra.push("winner " + Number(sum.winner_overall).toFixed(3));
    
    const ts = new Date().toLocaleTimeString();
    d.textContent = `[${ts}] ${ev.event}` + (extra.length ? " · " + extra.join(" · ") : "");
    timelineEl.appendChild(d);
    
    if ($("autoscroll")?.checked) {
      timelineEl.scrollTop = timelineEl.scrollHeight;
    }
  }

  // Live connection
  async function connectLive() {
    const runFilter = $("run-id")?.value.trim();
    clearAll(); // Always clear before connecting
    
    setStatus("connecting"); 
    hideError();

    // Subscribe to arena subjects on the bus via tap
    const url = new URL("/arena/stream", window.location.origin);
    url.searchParams.set("subject", "stephanie.events.arena.run.>");
    url.searchParams.set("debug", "1");

    const src = new EventSource(url);
    es = src;

    src.onopen = () => setStatus("open");
    src.onerror = (e) => { 
      setStatus("error"); 
      showError("SSE connection error: " + (e.message || "Unknown error"));
      console.error("SSE error:", e);
    };
    
    src.onmessage = (e) => {
      try {
        const obj = JSON.parse(e.data);
        // obj is {subject, data: envelope|json}
        if (eventHandler && obj) {
          eventHandler.handleTapMessage(obj, runFilter);
        }
      } catch (err) {
        console.error("Error processing message:", err, e.data);
        // ignore non-JSON
      }
    };
  }

  async function connectHistory() {
    const runId = $("run-id")?.value.trim();
    if (!runId) { 
      alert("Enter a run_id first"); 
      return; 
    }

    // Clear UI first
    clearAll();

    try {
      const res = await api.fetchEvents(runId);
      if (!res.ok) throw new Error("HTTP " + res.status);
      const raw = await res.json();
      const events = raw.map(tryParseJSON);

      setChip("chip-run", `run_id: ${runId}`);

      // Process events with a small delay between each
      let i = 0;
      const tick = () => {
        if (i >= events.length) { 
          setStatus("open"); 
          return; 
        }
        
        const body = events[i++];
        if (eventHandler) {
          eventHandler.handleTapMessage({ data: { payload: body } }, null);
        }
        
        setTimeout(tick, 20);
      };
      
      tick();
    } catch (err) {
      setStatus("error");
      showError("Failed to load history: " + (err?.message || String(err)));
      console.error("History load error:", err);
    }
  }

  function connectReplayFromEvents(events) {
    if (!Array.isArray(events) || !eventHandler) return;
    
    // Clear UI first
    clearAll();
    
    const parsed = events.map(tryParseJSON);
    let i = 0;
    
    const tick = () => {
      if (i >= parsed.length) return;
      
      const body = parsed[i++];
      eventHandler.handleTapMessage({ data: { payload: body } }, null);
      
      setTimeout(tick, 80);
    };
    
    tick();
  }

  function downloadSnapshot() {
    if (!arenaState) {
      showError("System not initialized. Please refresh the page.");
      return;
    }
    
    const state = arenaState.getState();
    
    const snapshot = {
      saved_at: new Date().toISOString(),
      run_id: state.run_id,
      case_id: state.case_id,
      section_name: state.section_name,
      paper_id: state.paper_id,
      events: state.events,
      derived: {
        labels: state.labels,
        best_history: state.best_history,
        marg_history: state.marg_history,
        topk: state.topk,
        summary: state.summary,
        winner_excerpt: state.winner_excerpt,
        chartData: state.chartData,
        topkByCase: Array.from(state.topkByCase.entries()).map(([caseId, data]) => ({
          caseId,
          rounds: Array.from(data.rounds.entries()),
          roundNumbers: data.roundNumbers
        })),
      }
    };
    
    const blob = new Blob([JSON.stringify(snapshot, null, 2)], {type: "application/json"});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `arena-snapshot-${state.run_id || "run"}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  
  // Safe clear function that checks if components exist
  function clearAll() {
    console.log("Clearing all components");
    
    try {
      // Close event source if exists
      if (es) { 
        es.close(); 
        es = null; 
      }
      
      setStatus("idle"); 
      hideError();
      
      // Reset state first
      if (arenaState) {
        arenaState.reset();
        console.log("ArenaState reset");
      }
      
      // Clear UI components
      if (charts) {
        charts.destroy?.();
        console.log("Charts cleared");
      }
      
      if (grid) {
        grid.destroy?.();
        console.log("Grid cleared");
      }
      
      // Clear timeline
      const timelineEl = $("timeline");
      if (timelineEl) {
        timelineEl.innerHTML = "";
      }
      
      // Clear UI chips
      setChip("chip-run", null); 
      setChip("chip-section", null);
      setChip("chip-case", null);
      setChip("chip-paper", null);
      
    } catch (error) {
      console.error("Error clearing components:", error);
    }
  }

  // Create a function to initialize all components
  function initializeComponents() {
    // Create the single source of truth
    arenaState = new ArenaState();
    
    // Initialize API client
    api = new ApiClient();
    
    // Initialize components with state access
    provenanceCard = new ProvenanceCardComponent(arenaState);
    grid = new GridComponent(arenaState, provenanceCard);
    charts = new ChartsComponent(arenaState, provenanceCard);
    
    // Initialize event handler with state
    eventHandler = new EventHandler(arenaState, grid, charts, provenanceCard);
    
    console.log("Arena Live UI initialized with centralized state");
    
    // Return components for external access
    return { arenaState, provenanceCard, grid, charts, eventHandler, api };
  }

  // UI wiring
  document.addEventListener('DOMContentLoaded', () => {
    // Initialize all components
    const components = initializeComponents();
    
    // Set up UI controls
    const btnConnect = $("btn-connect");
    const btnClear = $("btn-clear");
    const btnSnapshot = $("btn-snapshot");
    const fileReplay = $("file-replay");
    const modeSelect = $("mode");
    
    // Connect button
    btnConnect?.addEventListener("click", () => {
      const mode = modeSelect?.value;
      if (mode === "live") {
        connectLive();
      } else if (mode === "history") {
        connectHistory();
      } else if (mode === "replay") {
        showError("Replay mode: choose a JSON file with the 'Load JSON' button.");
      }
    });
    
    // Clear button
    btnClear?.addEventListener('click', clearAll);
    
    // Snapshot button
    btnSnapshot?.addEventListener("click", downloadSnapshot);

    // Toggle winner card
    $("btn-toggle-winner")?.addEventListener("click", () => {
      const body = $("winner-card");
      if (body?.classList.contains("d-none")) {
        body.classList.remove("d-none");
        $("btn-toggle-winner").textContent = "Collapse";
      } else {
        body?.classList.add("d-none");
        $("btn-toggle-winner").textContent = "Expand";
      }
    });

    // Mode change handler
    modeSelect?.addEventListener("change", () => {
      const mode = modeSelect?.value;
      const needsRun = mode !== "replay";
      if ($("run-id")) $("run-id").disabled = !needsRun;
      if (btnConnect) btnConnect.disabled = false;
    });

    // File replay handler
    fileReplay?.addEventListener("change", async (e) => {
      const f = e.target.files?.[0];
      if (!f) return;
      
      try {
        const txt = await f.text();
        const obj = JSON.parse(txt);
        const events = Array.isArray(obj) ? obj : (obj.events || []);
        connectReplayFromEvents(events);
        setStatus("open"); 
        hideError();
      } catch (err) {
        showError("Failed to load JSON: " + (err?.message || String(err)));
        console.error("JSON load error:", err);
      } finally {
        e.target.value = "";
      }
    });

    // Deep-link support and auto-connect
    const urlParams = new URLSearchParams(window.location.search);
    const rid = urlParams.get("run_id");
    if (rid && $("run-id")) {
      $("run-id").value = rid;
      if (modeSelect) modeSelect.value = "history";
      connectHistory();
    } else {
      // Fallback: live (as before)
      connectLive();
    }
  });

  // Expose globals for debugging (only after components are initialized)
  window.openProvenanceCard = function(caseId, score, variant, origin) {
    if (provenanceCard) {
      provenanceCard.openCard(caseId, score, variant, origin);
    } else {
      console.error("ProvenanceCard not initialized yet. Please wait for page to fully load.");
    }
  };
  
  window.rescoreCase = function(caseId) {
    if (provenanceCard) {
      provenanceCard.rescoreCase(caseId);
    } else {
      console.error("ProvenanceCard not initialized yet. Please wait for page to fully load.");
    }
  };
  
  window.diagnoseBootstrap = () => {
    console.log("Bootstrap diagnosis:", {
      "bootstrap global": typeof bootstrap,
      "Offcanvas constructor": typeof bootstrap?.Offcanvas,
      "Available components": Object.keys(bootstrap || {}),
      "Offcanvas element exists": !!document.getElementById('provenanceCard'),
      "Components initialized": {
        arenaState: !!arenaState,
        provenanceCard: !!provenanceCard,
        grid: !!grid,
        charts: !!charts,
        eventHandler: !!eventHandler
      }
    });
  };

  // Global click capture for debugging
  document.body.addEventListener("click", function(e) {
    console.log("BODY CLICK CAPTURED", e.target.tagName, e.target.id, e.target.className);
  }, true);
})();