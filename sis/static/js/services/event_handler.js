// sis/static/arena/js/services/event_handler.js
export class EventHandler {
  constructor(state, grid, charts, provenanceCard) {
    this.state = state;
    this.grid = grid;
    this.charts = charts;
    this.provenanceCard = provenanceCard;
    this.roundIdx = 0;
  }

  handleTapMessage(msg, runFilter) {
    const body = this.normalizeFromTap(msg);
    console.log("RAW normalized event:", JSON.stringify(body, null, 2)); // 👈 CRITICAL DEBUG

    if (!body || typeof body !== "object" || !body.event) return;

    // Client-side run_id filter (tap doesn't filter)
    if (runFilter && String(body.run_id || "") !== String(runFilter)) return;

    // chips
    if (body.run_id && !this.state.run_id) this.state.run_id = String(body.run_id);
    if (body.section_name && !this.state.section_name) this.state.section_name = String(body.section_name);
    this.setChip("chip-run", this.state.run_id ? `run_id: ${this.state.run_id}` : null);
    this.setChip("chip-section", this.state.section_name ? `section: ${this.state.section_name}` : null);

    // raw buffer + timeline
    this.state.events.push(body);
    this.addTimeline(body);

    // 🔥 CRITICAL FIX: Proper event-to-grid binding
    if (body.event === "initial_scored" && body.topk) {
      console.log("Setting TopK with data:", body.topk); // DEBUG
      this.setTopK(body.topk);  // This should work now
    }

    if (!body || typeof body !== "object" || !body.event) return;

    // Client-side run_id filter (tap doesn't filter)
    if (runFilter && String(body.run_id || "") !== String(runFilter)) return;

    console.log("Message:" + msg)

    // chips
    if (body.run_id && !this.state.run_id) this.state.run_id = String(body.run_id);
    if (body.section_name && !this.state.section_name) this.state.section_name = String(body.section_name);
    this.setChip("chip-run", this.state.run_id ? `run_id: ${this.state.run_id}` : null);
    this.setChip("chip-section", this.state.section_name ? `section: ${this.state.section_name}` : null);

    // raw buffer + timeline
    this.state.events.push(body);
    this.addTimeline(body);

    // widgets
    if (body.event === "initial_scored" && body.topk) {
      this.grid.setTopK(body.topk);
    }
    if (body.event === "round_end") {
      this.roundIdx += 1;
      this.state.labels.push(String(this.roundIdx));
      this.state.best_history.push(body.best_overall ?? null);
      this.state.marg_history.push(body.marginal_per_ktok ?? null);

      this.charts.updateCharts(
        this.state.labels.slice(),
        this.state.best_history.slice(),
        this.state.marg_history.slice()
      );
    }
    if (body.event === "arena_stop" && body.winner_overall != null) {
      this.updateSummary({ winner_overall: body.winner_overall, rounds_run: body.rounds_run, reason: body.reason });
    }
    const sum = body.summary || body.arena_summary;
    if (body.event === "arena_done" && sum) {
      this.updateSummary(sum);
    }
    if (body.event === "arena_done" && body.winner_excerpt) {
      this.state.winner_excerpt = body.winner_excerpt;
      this.$("winner-note")?.classList.add("d-none");
      this.$("winner-excerpt")?.classList.remove("d-none");
      this.$("winner-excerpt").textContent = String(body.winner_excerpt).slice(0, 1200);
    }
  }

  normalizeFromTap(msg) {
    // tap emits: { subject, data: <envelope or raw> } — may be stringified
    let x = this.tryParseJSON(msg?.data ?? msg);

    // Event-service envelope → payload (also may be stringified)
    if (x && typeof x === "object" && x.payload != null) {
      x = this.tryParseJSON(x.payload);
    }

    // Some producers nest payload again
    if (x && typeof x === "object" && x.payload != null && x.payload.event) {
      x = this.tryParseJSON(x.payload);
    }

    // If we STILL got a string, try once more
    x = this.tryParseJSON(x);

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
    if (this.$("autoscroll")?.checked) timelineEl.scrollTop = timelineEl.scrollHeight;
  }

  setTopK(topk) {
    this.grid.setTopK(topk);
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
  }

  $(selector) {
    if (selector.startsWith("#")) {
      return document.getElementById(selector.substring(1));
    }
    return document.getElementById(selector);
  }
}