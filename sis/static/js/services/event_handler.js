// stephanie/static/arena/js/services/event_handler.js
export class EventHandler {
    constructor(arenaState, grid, charts, provenanceCard, enableLogging = true) {
        this.arenaState = arenaState;
        this.grid = grid;
        this.charts = charts;
        this.provenanceCard = provenanceCard;
        this.enableLogging = enableLogging;

        // Track round counters per case (for round number extraction)
        this.caseRoundMap = new Map(); // case_id -> current round number

        // Subscribe to state changes for debugging
        this.unsubscribe = arenaState.subscribe(state => {
            if (enableLogging) {
                console.log("[EventHandler] State updated", {
                    run_id: state.run_id,
                    case_id: state.case_id,
                    currentCaseId: state.currentCaseId
                });
            }
        });
    }

    handleTapMessage(msg, runFilter) {
        const body = this.normalizeFromTap(msg);
        if (this.enableLogging) {
            console.log("RAW normalized event:", JSON.stringify(body, null, 2));
        }

        if (!body || typeof body !== "object" || !body.event) return;

        // 🔎 Client-side run_id filter
        if (runFilter && String(body.run_id || "") !== String(runFilter)) return;

        // 📝 Push to buffer
        this.arenaState.setState({
            events: [...this.arenaState.getState().events, body]
        }, true); // Silent update for events array

        // 📊 Handle events
        switch (body.event) {
            case "arena_start": {
                const caseId = String(body.case_id);
                
                // Reset round counter FOR THIS SPECIFIC CASE
                this.caseRoundMap.set(caseId, 0);

                this.arenaState.arenaStart(
                    body.run_id,
                    caseId,
                    body.section_name,
                    body.paper_id
                );

                // Update UI chips
                this.updateChips();

                if (this.enableLogging) console.log("Arena started:", body.run_id, "case:", caseId);
                break;
            }
            case "round_begin": {
                const caseId = String(body.case_id);
                
                // Get or initialize round counter for this case
                const roundCounter = (this.caseRoundMap.get(caseId) || 0) + 1;
                this.caseRoundMap.set(caseId, roundCounter);

                const roundNumber = this.arenaState.roundBegin(caseId);

                // Update charts
                this.charts.updateFromState();

                if (this.enableLogging) console.log("Round begin:", roundNumber, "for case", caseId);
                break;
            }
            case "initial_scored": {
                const caseId = String(body.case_id);
                
                // CRITICAL FIX: Get round number from our local tracking
                // This matches the working version's logic
                const roundNumber = this.caseRoundMap.get(caseId) || 1;

                if (body.topk && Array.isArray(body.topk) && body.topk.length > 0) {
                    // Add case_id to each Top-K item
                    const topkWithCaseId = body.topk.map(item => ({
                        ...item,
                        case_id: caseId
                    }));

                    this.arenaState.initialScored(
                        caseId,
                        roundNumber,
                        topkWithCaseId
                    );

                    // Update grid
                    this.grid.updateFromState();

                    if (this.enableLogging) console.log("[TopK] received for round", roundNumber, ":", topkWithCaseId);
                } else {
                    console.warn("[TopK] initial_scored event but no topk array", body);
                }
                break;
            }
            case "round_end": {
                const caseId = String(body.case_id);
                
                // CRITICAL FIX: Get round number from our local tracking
                // This matches the working version's logic
                const roundNumber = this.caseRoundMap.get(caseId) || 1;

                this.arenaState.roundEnd(
                    caseId,
                    roundNumber,
                    body.best_overall,
                    body.marginal_per_ktok
                );

                // Update charts
                this.charts.updateFromState();

                if (this.enableLogging) console.log("Round end scores:", body.best_overall, body.marginal_per_ktok, "for round", roundNumber);
                break;
            }
            case "arena_stop":
            case "arena_done": {
                this.arenaState.arenaStop({
                    winner_overall: body.winner_overall,
                    rounds_run: body.rounds_run,
                    reason: body.reason,
                    winner_excerpt: body.winner_excerpt
                });

                this.updateSummary();
                break;
            }
            default: {
                if (this.enableLogging) console.log("Unhandled event:", body.event);
                break;
            }
        }

        // Update timeline UI (outside the switch statement)
        this.addTimeline(body);
    }

    updateChips() {
        const state = this.arenaState.getState();

        this.setChip("chip-run", state.chips.run);
        this.setChip("chip-case", state.chips.case);
        this.setChip("chip-paper", state.chips.paper);
        this.setChip("chip-section", state.chips.section);
    }

    normalizeFromTap(msg) {
        // Start with the raw message
        let x = msg?.data ?? msg;
        
        // First level: If x is a string, parse it
        x = this.tryParseJSON(x);
        
        // Second level: If x has a payload property that's a string, parse it
        if (x && typeof x === "object" && typeof x.payload === "string") {
            const parsedPayload = this.tryParseJSON(x.payload);
            if (parsedPayload && typeof parsedPayload === "object") {
                x = parsedPayload;
            }
        }
        
        // Third level: If x.payload exists and is an object, use it
        if (x && typeof x === "object" && typeof x.payload === "object" && x.payload !== null) {
            x = x.payload;
        }
        
        // Fourth level: Handle the case where payload contains another nested payload
        if (x && typeof x === "object" && typeof x.payload === "string") {
            const nestedPayload = this.tryParseJSON(x.payload);
            if (nestedPayload && typeof nestedPayload === "object") {
                x = nestedPayload;
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

    updateSummary() {
        const state = this.arenaState.getState();
        const sum = state.summary || {};

        this.$("sum-winner").textContent = sum.winner_overall != null ? Number(sum.winner_overall).toFixed(3) : "—";
        this.$("sum-rounds").textContent = sum.rounds_run ?? "—";
        this.$("sum-reason").textContent = sum.reason ?? "—";

        if (sum.winner_excerpt) {
            this.$("winner-note")?.classList.add("d-none");
            this.$("winner-excerpt")?.classList.remove("d-none");
            this.$("winner-excerpt").textContent = String(sum.winner_excerpt).slice(0, 1200);
        }
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
        this.arenaState.reset();
    }

    $(selector) {
        if (selector.startsWith("#")) {
            return document.getElementById(selector.substring(1));
        }
        return document.getElementById(selector);
    }

    destroy() {
        if (this.unsubscribe) {
            this.unsubscribe();
        }
    }
}