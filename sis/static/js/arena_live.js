import { GridComponent } from './components/grid.js';
import { ChartsComponent } from './components/charts.js';
import { ProvenanceCardComponent } from './components/provenance_card.js';
import { EventHandler } from './services/event_handler.js';
import { ApiClient } from './services/api_client.js';

// sis/static/arena/js/arena_live.js
(() => {
  "use strict";

  // Global state
  const state = {
    run_id: null,
    section_name: null,
    events: [],
    best_history: [],
    marg_history: [],
    labels: [],
    topk: [],
    summary: null,
    winner_excerpt: null,
  };

  let es = null;

  // Initialize components
  const grid = new GridComponent();
  const charts = new ChartsComponent();
  const provenanceCard = new ProvenanceCardComponent();
  const eventHandler = new EventHandler(state, grid, charts, provenanceCard);
  const api = new ApiClient();

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
    if (es) { 
      es.close(); 
      es = null; 
    }
    setStatus("connecting"); 
    hideError();

    // Subscribe to arena subjects on the bus via tap
    const url = new URL("/arena/stream", window.location.origin);
    url.searchParams.set("subject", "stephanie.events.arena.run.>"); // adjust prefix if needed
    url.searchParams.set("debug", "1");

    const src = new EventSource(url);
    es = src;

    src.onopen = () => setStatus("open");
    src.onerror = () => { 
      setStatus("error"); 
      showError("SSE connection error"); 
    };
    src.onmessage = (e) => {
      try {
        const obj = JSON.parse(e.data);
        // obj is {subject, data: envelope|json}
        eventHandler.handleTapMessage(obj, runFilter);
      } catch {
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

    // Clear UI
    clearAll();

    try {
      const res = await api.fetchEvents(runId);
      if (!res.ok) throw new Error("HTTP " + res.status);
      const raw = await res.json();                // array of strings or objects
      const events = raw.map(tryParseJSON);        // <- normalize to objects

      setChip("chip-run", `run_id: ${runId}`);

      let i = 0;
      const tick = () => {
        if (i >= events.length) { 
          setStatus("open"); 
          return; 
        }
        const body = events[i++];
        eventHandler.handleTapMessage({ data: { payload: body } }, /*runFilter*/ null);
        setTimeout(tick, 20);
      };
      tick();
    } catch (err) {
      setStatus("error");
      showError("Failed to load history: " + (err?.message || String(err)));
    }
  }

  function connectReplayFromEvents(events) {
    if (!Array.isArray(events)) return;
    clearAll();
    const parsed = events.map(tryParseJSON);       // <- normalize to objects
    let i = 0;
    const tick = () => {
      if (i >= parsed.length) return;
      const body = parsed[i++];
      eventHandler.handleTapMessage({ data: { payload: body } }, /*runFilter*/ null);
      setTimeout(tick, 80);
    };
    tick();
  }

  function downloadSnapshot() {
    const snapshot = {
      saved_at: new Date().toISOString(),
      run_id: state.run_id,
      section_name: state.section_name,
      events: state.events,
      derived: {
        labels: state.labels,
        best_history: state.best_history,
        marg_history: state.marg_history,
        topk: state.topk,
        summary: state.summary,
        winner_excerpt: state.winner_excerpt,
      }
    };
    const blob = new Blob([JSON.stringify(snapshot, null, 2)], {type: "application/json"});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `arena-snapshot-${state.run_id || "run"}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  function clearAll() {
    if (es) { 
      es.close(); 
      es = null; 
    }
    setStatus("idle"); 
    hideError();
    
    state.labels = []; 
    state.best_history = []; 
    state.marg_history = [];
    
    charts.clear();
    grid.clear();
    eventHandler.reset();
    
    const timelineEl = $("timeline");
    if (timelineEl) timelineEl.innerHTML = "";
    
    setChip("chip-run", null); 
    setChip("chip-section", null);
    state.run_id = null; 
    state.section_name = null;
    state.events = []; 
  }

  // UI wiring
  document.addEventListener('DOMContentLoaded', function() {
    // Connect button
    $("btn-connect")?.addEventListener("click", () => {
      const mode = $("mode")?.value;
      if (mode === "live") return connectLive();
      if (mode === "history") return connectHistory();
      if (mode === "replay") {
        showError("Replay mode: choose a JSON file with the ‘Load JSON’ button.");
      }
    });

    // Clear button
    $("btn-clear")?.addEventListener("click", clearAll);
    
    // Snapshot button
    $("btn-snapshot")?.addEventListener("click", downloadSnapshot);

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
    $("mode")?.addEventListener("change", () => {
      const mode = $("mode")?.value;
      const needsRun = mode !== "replay";
      if ($("run-id")) $("run-id").disabled = !needsRun;
      if ($("btn-connect")) $("btn-connect").disabled = false;
    });

    // File replay handler
    $("file-replay")?.addEventListener("change", async (e) => {
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
      } finally {
        e.target.value = "";
      }
    });

    // Deep-link support and auto-connect
    const urlParams = new URLSearchParams(window.location.search);
    const rid = urlParams.get("run_id");
    if (rid && $("run-id")) {
      $("run-id").value = rid;
      if ($("mode")) $("mode").value = "history";
      connectHistory();
    } else {
      // Fallback: live (as before)
      connectLive();
    }
  });

  // Expose globals for debugging
  window.openProvenanceCard = provenanceCard.openCard.bind(provenanceCard);
  window.rescoreCase = provenanceCard.rescoreCase.bind(provenanceCard);
  window.diagnoseBootstrap = () => {
    console.log("Bootstrap diagnosis:", {
      "bootstrap global": typeof bootstrap,
      "Offcanvas constructor": typeof bootstrap?.Offcanvas,
      "Available components": Object.keys(bootstrap || {}),
      "Offcanvas element exists": !!document.getElementById('provenanceCard')
    });
  };

  // Global click capture for debugging
  document.body.addEventListener("click", function(e) {
    console.log("BODY CLICK CAPTURED", e.target.tagName, e.target.id, e.target.className);
  }, true);
})(); 