// stephanie/static/arena/js/components/grid.js
export class GridComponent {
    constructor() {
        this.state = {
            topk: [],
            labels: [],
            best_history: [],
            marg_history: []
        };
        
        this.logger = console; // or your actual logger
    }

    setTopK(topk) {
        console.log("setTopK called with:", topk);
        
        this.state.topk = topk || [];
        const tbody = this.$("topk-body");
        if (!tbody) {
            this.logger.error("topk-body element not found");
            return;
        }
        
        tbody.innerHTML = "";
        if (!this.state.topk.length) {
            tbody.innerHTML = '<tr><td colspan="5" class="text-muted">No data</td></tr>';
            console.log("No topk data to display");
            return;
        }
        
        // Build rows with proper data attributes
        for (const t of this.state.topk) {
            const tr = document.createElement("tr");
            
            // Add data attributes for identification
            tr.dataset.caseId = t.case_id || t.id || "";
            tr.dataset.score = t.overall || t.score || "0";
            tr.dataset.variant = t.variant || "";
            tr.dataset.origin = t.origin || "";
            
            tr.innerHTML = `<td>${t.origin ?? "—"}</td>
                            <td>${t.variant ?? "—"}</td>
                            <td class="text-end">${Number(t.overall ?? t.score ?? 0).toFixed(3)}</td>
                            <td class="text-end">${Number(t.k ?? 0).toFixed(3)}</td>
                            <td>${t.verified ? "yes" : "no"}</td>`;
            
            // Add click handler with proper event delegation
            tr.style.cursor = "pointer";
            tr.title = `Click to view provenance for ${t.origin ?? "this"} candidate`;
            tr.addEventListener("click", (e) => {
                e.stopPropagation(); // Prevent event bubbling issues
                
                const caseId = tr.dataset.caseId;
                const score = parseFloat(tr.dataset.score);
                const variant = tr.dataset.variant;
                const origin = tr.dataset.origin;
                
                console.log("Grid row clicked:", {caseId, score, variant, origin});
                
                if (caseId) {
                    if (typeof window.openProvenanceCard === 'function') {
                        window.openProvenanceCard(caseId, score, variant, origin);
                    } else {
                        this.logger.error("openProvenanceCard not available globally");
                    }
                } else {
                    this.logger.warn("No case_id found for this row");
                }
            });
            
            tbody.appendChild(tr);
        }
    }

    clear() {
        this.setTopK([]); // This will now work because this.state is defined
        
        // Clear other state
        this.state.labels = [];
        this.state.best_history = [];
        this.state.marg_history = [];
    }

    $(selector) {
        if (selector.startsWith("#")) {
            return document.getElementById(selector.substring(1));
        }
        return document.getElementById(selector);
    }
}