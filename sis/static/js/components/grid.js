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
        console.log("[Grid] setTopK called with:", topk);

        this.state.topk = topk || [];
        const tbody = this.$("topk-body");
        if (!tbody) {
            console.error("[Grid] tbody#topk-body not found!");
            return;
        }

        tbody.innerHTML = "";
        if (!this.state.topk.length) {
            tbody.innerHTML = '<tr><td colspan="5" class="text-muted">No data</td></tr>';
            return;
        }

        for (const t of this.state.topk) {
            const tr = document.createElement("tr");

            tr.dataset.caseId = t.case_id || t.id || "";
            tr.dataset.score = Number(t.overall ?? t.score ?? 0).toFixed(3);
            tr.dataset.variant = t.variant || "";
            tr.dataset.origin = t.origin || "";

            tr.innerHTML = `
            <td>${t.origin ?? "—"}</td>
            <td>${t.variant ?? "—"}</td>
            <td class="text-end">${Number(t.overall ?? t.score ?? 0).toFixed(3)}</td>
            <td class="text-end">${Number(t.k ?? 0).toFixed(3)}</td>
            <td>${t.verified ? "yes" : "no"}</td>`;

            // Row is clickable
            tr.style.cursor = "pointer";
            tr.addEventListener("click", () => {
            console.log("[Grid] clicked row:", tr.dataset);
            if (tr.dataset.caseId) {
                window.openProvenanceCard(
                tr.dataset.caseId,
                tr.dataset.score,
                tr.dataset.variant,
                tr.dataset.origin
                );
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