// sis/static/arena/js/services/event_handler.js
export class EventHandler {
  constructor(state, grid, charts, provenanceCard, enableLogging = true) {
    this.state = state;
    this.grid = grid;
    this.charts = charts;
    this.provenanceCard = provenanceCard;

    // Track round counters PER CASE ID (critical fix)
    this.caseRoundMap = new Map(); // case_id -> round counter
    this.currentCaseId = null;

    // Track all rounds across all cases for this run
    this.allRounds = [];
    this.allBestHistory = [];
    this.allMargHistory = [];

    this.enableLogging = enableLogging;
  }

  handleTapMessage(msg, runFilter) {
    const body = this.normalizeFromTap(msg);
    if (this.enableLogging) {
      console.log("RAW normalized event:", JSON.stringify(body, null, 2));
    }

    if (!body || typeof body !== "object" || !body.event) return;

    // 🔎 Client-side run_id filter
    if (runFilter && String(body.run_id || "") !== String(runFilter)) return;

    // 🏷️ Update chips - include case_id
    if (body.run_id && !this.state.run_id) this.state.run_id = String(body.run_id);
    if (body.case_id) this.state.case_id = String(body.case_id); // Always update case_id
    if (body.section_name) this.state.section_name = String(body.section_name);

    this.setChip("chip-run", this.state.run_id ? `run_id: ${this.state.run_id}` : null);
    this.setChip("chip-case", this.state.case_id ? `case: ${this.state.case_id}` : null);
    this.setChip("chip-section", this.state.section_name ? `section: ${this.state.section_name}` : null);

    // 📝 Push to buffer + timeline
    this.state.events.push(body);
    this.addTimeline(body);

    // 📊 Handle events
    switch (body.event) {
      case "arena_start":
        // Reset round counter FOR THIS SPECIFIC CASE
        if (body.case_id) {
          this.caseRoundMap.set(body.case_id, 0);
          this.currentCaseId = body.case_id;
        }

        if (this.enableLogging) console.log("Arena started:", body.run_id, "case:", body.case_id);
        break;

      case "round_begin":
        // Get or initialize round counter for this case
        let roundCounter = 0;
        if (body.case_id) {
          roundCounter = (this.caseRoundMap.get(body.case_id) || 0) + 1;
          this.caseRoundMap.set(body.case_id, roundCounter);
          this.currentCaseId = body.case_id;
        } else {
          // Fallback if no case_id (shouldn't happen)
          this.roundIdx += 1;
          roundCounter = this.roundIdx;
        }

        // Store this round number with the event for later reference
        body.round_number = roundCounter;

        // Update our global round tracking
        this.allRounds.push(`r${roundCounter}`);
        this.allBestHistory.push(null);
        this.allMargHistory.push(null);

        // Update charts with ALL rounds
        this.charts.updateCharts(
          this.allRounds.slice(),
          this.allBestHistory.slice(),
          this.allMargHistory.slice()
        );

        if (this.enableLogging) console.log("Round begin:", roundCounter, "for case", body.case_id);
        break;

      case "initial_scored":
        // Get the round number from the event (set in round_begin)
        const roundNumber = body.round_number ||
          (body.case_id ? (this.caseRoundMap.get(body.case_id) || 1) : 1);

        console.log("[TopK] Processing round", roundNumber, "for case", body.case_id, "with data:", body.topk);

        if (body.topk && Array.isArray(body.topk) && body.topk.length > 0) {
          console.log("[TopK] received:", body.topk);
          this.state.topk = body.topk;

          // Pass round number AND case_id to setTopK
          this.grid.setTopK(body.topk, roundNumber, body.case_id);
        } else {
          console.warn("[TopK] initial_scored event but no topk array", body);
        }
        break;

      case "round_end":
        // Get the round number from the event (set in round_begin)
        const roundNum = body.round_number ||
          (body.case_id ? (this.caseRoundMap.get(body.case_id) || 1) : 1);

        // Update our global history at the correct position
        const index = this.allRounds.length - 1;
        if (index >= 0) {
          this.allBestHistory[index] = body.best_overall ?? null;
          this.allMargHistory[index] = body.marginal_per_ktok ?? null;
        }

        // Update charts with ALL rounds
        this.charts.updateCharts(
          this.allRounds.slice(),
          this.allBestHistory.slice(),
          this.allMargHistory.slice()
        );

        if (this.enableLogging) console.log("Round end scores:", body.best_overall, body.marginal_per_ktok, "for round", roundNum);
        break;

      case "arena_stop":
      case "arena_done":
        const sum = body.summary || body.arena_summary;
        if (sum) this.updateSummary(sum);

        if (body.winner_excerpt) {
          this.state.winner_excerpt = body.winner_excerpt;
          this.$("winner-note")?.classList.add("d-none");
          this.$("winner-excerpt")?.classList.remove("d-none");
          this.$("winner-excerpt").textContent = String(body.winner_excerpt).slice(0, 1200);
        }
        break;

      default:
        if (this.enableLogging) console.log("Unhandled event:", body.event);
    }
  }

  normalizeFromTap(msg) {
    let x = this.tryParseJSON(msg?.data ?? msg);

    // First level of payload extraction
    if (x && typeof x === "object" && x.payload != null) {
      x = this.tryParseJSON(x.payload);
    }

    // Second level: Handle case where payload is a string that needs parsing
    if (x && typeof x === "object" && typeof x.payload === "string") {
      const parsedPayload = this.tryParseJSON(x.payload);
      if (parsedPayload && typeof parsedPayload === "object") {
        x = parsedPayload;
      }
    }

    return x || {};
  }

  tryParseJSON(x) {
    if (typeof x !== "string") return x;
    const t = x.trim();
    if (!t) return x;
    try { return JSON.parse(t); } catch { return x; }
  }

  addTimeline(ev) {
    const timelineEl = this.$("timeline");
    if (!timelineEl) return;

    const d = document.createElement("div");
    d.className = "small";

    const extra = [];
    if (ev.round_number != null) extra.push("r" + ev.round_number);
    if (ev.best_overall != null) extra.push("best " + Number(ev.best_overall).toFixed(3));
    if (ev.marginal_per_ktok != null) extra.push("marg/kTok " + Number(ev.marginal_per_ktok).toFixed(3));
    if (ev.reason) extra.push("reason " + ev.reason);
    const sum = ev.summary || ev.arena_summary;
    if (sum?.winner_overall != null) extra.push("winner " + Number(sum.winner_overall).toFixed(3));

    const ts = new Date().toLocaleTimeString();
    d.textContent = `[${ts}] ${ev.event}` + (extra.length ? " · " + extra.join(" · ") : "");
    timelineEl.appendChild(d);

    if (this.$("autoscroll")?.checked) {
      timelineEl.scrollTop = timelineEl.scrollHeight;
    }
  }

  updateSummary(sum) {
    this.state.summary = sum || null;
    this.$("sum-winner").textContent = sum?.winner_overall != null ? Number(sum.winner_overall).toFixed(3) : "—";
    this.$("sum-rounds").textContent = sum?.rounds_run ?? "—";
    this.$("sum-reason").textContent = sum?.reason ?? "—";
  }

  setChip(id, text) {
    const el = this.$(id);
    if (!text) {
      el?.classList.add("d-none");
      return;
    }
    if (el) {
      el.textContent = text;
      el.classList.remove("d-none");
    }
  }

  reset() {
    this.state.run_id = null;
    this.state.case_id = null;
    this.state.section_name = null;
    this.state.events = [];
    this.state.topk = [];
    this.state.summary = null;
    this.state.winner_excerpt = null;

    // Reset round tracking
    this.caseRoundMap = new Map();
    this.currentCaseId = null;
    this.allRounds = [];
    this.allBestHistory = [];
    this.allMargHistory = [];
  }

  $(selector) {
    if (selector.startsWith("#")) {
      return document.getElementById(selector.substring(1));
    }
    return document.getElementById(selector);
  }
}