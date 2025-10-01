// sis/static/arena/js/services/event_handler.js
export class EventHandler {
  constructor(state, grid, charts, provenanceCard, enableLogging = true) {
    this.state = state;
    this.grid = grid;
    this.charts = charts;
    this.provenanceCard = provenanceCard;
    this.roundIdx = 0;
    this.enableLogging = enableLogging; // Toss it's off and **** **** toggle logs on/off
  }

  handleTapMessage(msg, runFilter) {
    const body = this.normalizeFromTap(msg);
    if (this.enableLogging) {
      console.log("RAW normalized event:", JSON.stringify(body, null, 2));
    }

    if (!body || typeof body !== "object" || !body.event) return;

    // 🔎 Client-side run_id filter (tap doesn’t filter)
    if (runFilter && String(body.run_id || "") !== String(runFilter)) return;

    // 🏷️ Update chips
    if (body.run_id && !this.state.run_id) this.state.run_id = String(body.run_id);
    if (body.section_name && !this.state.section_name) this.state.section_name = String(body.section_name);

    this.setChip("chip-run", this.state.run_id ? `run_id: ${this.state.run_id}` : null);
    this.setChip("chip-section", this.state.section_name ? `section: ${this.state.section_name}` : null);

    // 📝 Push to buffer + timeline
    this.state.events.push(body);
    this.addTimeline(body);

    // 📊 Handle events
    switch (body.event) {
      case "arena_start":
        // New run/session started
        this.roundIdx = 0;
        this.state.labels = [];
        this.state.best_history = [];
        this.state.marg_history = [];
        if (this.enableLogging) console.log("Arena started:", body.run_id);
        break;

      case "round_begin":
        // Pre-round marker (no scores yet, but keep label consistent)
        this.roundIdx += 1;
        this.state.labels.push(`r${this.roundIdx}`);
        this.state.best_history.push(null);
        this.state.marg_history.push(null);
        this.charts.updateCharts(
          this.state.labels.slice(),
          this.state.best_history.slice(),
          this.state.marg_history.slice()
        );
        if (this.enableLogging) console.log("Round begin:", this.roundIdx);
        break;

      case "initial_scored":
        if (body.topk && Array.isArray(body.topk) && body.topk.length > 0) {
          console.log("[TopK] received:", body.topk);
          // Save into state for later use
          this.state.topk = body.topk;
          // Render into table
          this.grid.setTopK(body.topk);
        } else {
          console.warn("[TopK] initial_scored event but no topk array", body);
        }
        break;

      case "round_end":
        this.state.best_history[this.roundIdx - 1] = body.best_overall ?? null;
        this.state.marg_history[this.roundIdx - 1] = body.marginal_per_ktok ?? null;
        this.charts.updateCharts(
          this.state.labels.slice(),
          this.state.best_history.slice(),
          this.state.marg_history.slice()
        );
        if (this.enableLogging) console.log("Round end scores:", body.best_overall, body.marginal_per_ktok);
        break;

      case "arena_stop":
        if (body.winner_overall != null) {
          this.updateSummary({
            winner_overall: body.winner_overall,
            rounds_run: body.rounds_run,
            reason: body.reason
          });
        }
        break;

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
    if (ev.round != null) extra.push("r" + ev.round);
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
    if (!text) { el?.classList.add("d-none"); return; }
    if (el) {
      el.textContent = text;
      el.classList.remove("d-none");
    }
  }

  reset() {
    this.state.run_id = null;
    this.state.section_name = null;
    this.state.events = [];
    this.state.best_history = [];
    this.state.marg_history = [];
    this.state.labels = [];
    this.state.topk = [];
    this.state.summary = null;
    this.state.winner_excerpt = null;
    this.roundIdx = 0;
  }

  $(selector) {
    if (selector.startsWith("#")) {
      return document.getElementById(selector.substring(1));
    }
    return document.getElementById(selector);
  }
}
