// stephanie/static/arena/js/components/provenance_card.js
export class ProvenanceCardComponent {
    constructor(arenaState) {
        this.arenaState = arenaState;
    }

    openCard(caseId, score, variant, origin) {
        // Show loading state
        const loadingEl = this.$("provenance-loading");
        const contentEl = this.$("provenance-content");
        const errorEl = this.$("provenance-error");
        
        if (loadingEl) loadingEl.classList.remove("d-none");
        if (contentEl) contentEl.classList.add("d-none");
        if (errorEl) errorEl.classList.add("d-none");

        console.log("Provenance card opened", {
            caseId: caseId,
            score: score,
            variant: variant,
            origin: origin,
            timestamp: new Date().toISOString()
        });

        // Fetch provenance data
        fetch(`/arena/api/provenance/${caseId}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                return response.json();
            })
            .then(data => {
                // Hide loading, show content
                if (loadingEl) loadingEl.classList.add("d-none");
                if (contentEl) contentEl.classList.remove("d-none");

                // Populate case info
                const caseNameEl = this.$("case-name");
                const caseAgentEl = this.$("case-agent");
                const caseMetaEl = this.$("case-meta");
                
                if (caseNameEl) caseNameEl.textContent = data.case?.name || `Case ${caseId}`;
                if (caseAgentEl) caseAgentEl.textContent = `Agent: ${data.case?.agent_name || origin || 'Unknown'}`;
                if (caseMetaEl) caseMetaEl.textContent = JSON.stringify(data.case?.meta || {}, null, 2);

                // Populate provenance chain
                const chainList = this.$("provenance-chain-list");
                if (chainList) {
                    chainList.innerHTML = '';
                    if (data.provenance_chain && data.provenance_chain.length > 0) {
                        data.provenance_chain.forEach(item => {
                            const li = document.createElement('li');
                            li.className = 'list-group-item';
                            li.innerHTML = `
                                <div class="d-flex justify-content-between">
                                    <div>
                                        <strong>${item.type?.charAt(0).toUpperCase() + item.type?.slice(1) || 'Item'}</strong>
                                        <div class="small text-muted">${item.title || item.name || `ID: ${item.id}`}</div>
                                    </div>
                                    ${item.url ? `<a href="${item.url}" class="btn btn-sm btn-outline-primary" target="_blank">
                                        <i class="fas fa-external-link-alt"></i>
                                    </a>` : ''}
                                </div>
                                <div class="small mt-1">
                                    ${Object.entries(item.meta || {}).map(([k, v]) =>
                                `<span class="badge bg-light text-dark me-1">${k}: ${String(v).substring(0, 30)}</span>`
                            ).join('')}
                                </div>
                            `;
                            chainList.appendChild(li);
                        });
                    } else {
                        chainList.innerHTML = '<li class="list-group-item text-muted">No provenance chain available</li>';
                    }
                }

                // Populate supporting knowledge
                const supportsList = this.$("supports-list");
                if (supportsList) {
                    supportsList.innerHTML = '';
                    if (data.supports && data.supports.length > 0) {
                        data.supports.forEach(support => {
                            const div = document.createElement('div');
                            div.className = 'mb-2 p-2 border rounded';
                            div.innerHTML = `
                                <div class="small">${support.text || 'No text'}</div>
                                <div class="small text-muted mt-1">
                                    Origin: ${support.origin || 'Unknown'} | 
                                    Variant: ${support.variant || 'Unknown'} | 
                                    Similarity: ${(support.similarity || 0).toFixed(3)}
                                </div>
                                ${support.url ? `<a href="${support.url}" class="small" target="_blank">View source</a>` : ''}
                            `;
                            supportsList.appendChild(div);
                        });
                    } else {
                        supportsList.innerHTML = '<div class="text-muted small">No supporting knowledge found</div>';
                    }
                }

                // Populate metrics
                const metricsGrid = this.$("metrics-grid");
                if (metricsGrid) {
                    metricsGrid.innerHTML = '';
                    if (data.metrics && Object.keys(data.metrics).length > 0) {
                        Object.entries(data.metrics).forEach(([key, value]) => {
                            if (typeof value === 'object' || key === 'meta') return; // Skip nested objects

                            const col = document.createElement('div');
                            col.className = 'col-6 mb-2';
                            col.innerHTML = `
                                <div class="card h-100">
                                    <div class="card-body p-2">
                                        <div class="small text-muted">${key}</div>
                                        <div class="fw-bold">${typeof value === 'number' ? value.toFixed(3) : value}</div>
                                    </div>
                                </div>
                            `;
                            metricsGrid.appendChild(col);
                        });
                    } else {
                        metricsGrid.innerHTML = '<div class="col-12"><div class="text-muted small">No metrics available</div></div>';
                    }
                }

                // Set up action buttons
                this.setupActionButtons(caseId, data);

                // Store current case ID for actions
                const offcanvasElement = this.$("provenanceCard");
                if (offcanvasElement) {
                    offcanvasElement.dataset.caseId = caseId;
                }

                // Show offcanvas
                this.showOffcanvas();
            })
            .catch(error => {
                console.error('Failed to load provenance:', error);
                
                if (loadingEl) loadingEl.classList.add("d-none");
                if (errorEl) errorEl.classList.remove("d-none");
                
                // FIXED: Safe assignment to error message element
                const errorMsg = this.$("error-message");
                if (errorMsg) {
                    errorMsg.textContent = error.message || String(error);
                }

                // Still show the offcanvas even on error
                this.showOffcanvas();
            });
    }

    setupActionButtons(caseId, data) {
        // FIXED: Safe handling of buttons with null checks
        const rescoreBtn = this.$("btn-rescore");
        if (rescoreBtn) {
            // Clear previous event listeners to prevent duplicates
            const newRescoreBtn = rescoreBtn.cloneNode(true);
            rescoreBtn.replaceWith(newRescoreBtn);
            this.$("btn-rescore").onclick = () => this.rescoreCase(caseId);
        }

        const viewFullBtn = this.$("btn-view-full");
        if (viewFullBtn && data.actions && data.actions.view_full_url) {
            const newViewFullBtn = viewFullBtn.cloneNode(true);
            viewFullBtn.replaceWith(newViewFullBtn);
            this.$("btn-view-full").onclick = () => {
                window.open(data.actions.view_full_url, '_blank');
            };
        }

        const editBtn = this.$("btn-edit");
        if (editBtn && data.actions && data.actions.edit_url) {
            const newEditBtn = editBtn.cloneNode(true);
            editBtn.replaceWith(newEditBtn);
            this.$("btn-edit").onclick = () => {
                window.open(data.actions.edit_url, '_blank');
            };
        }
    }

    showOffcanvas() {
        try {
            const offcanvasElement = this.$("provenanceCard");
            if (!offcanvasElement) {
                console.error("Provenance card element not found");
                return;
            }
            
            if (typeof bootstrap !== 'undefined' && bootstrap.Offcanvas) {
                const offcanvas = new bootstrap.Offcanvas(offcanvasElement);
                offcanvas.show();
                console.log("Provenance card shown successfully using Bootstrap Offcanvas");
            } else {
                // Fallback: show card directly
                offcanvasElement.style.display = 'block';
                offcanvasElement.style.position = 'fixed';
                offcanvasElement.style.right = '0';
                offcanvasElement.style.top = '0';
                offcanvasElement.style.width = '400px';
                offcanvasElement.style.height = '100vh';
                offcanvasElement.style.zIndex = '1000';
                offcanvasElement.style.backgroundColor = 'white';
                offcanvasElement.style.borderLeft = '1px solid #ccc';
                offcanvasElement.style.overflowY = 'auto';
                
                // Add close button if missing
                let closeBtn = offcanvasElement.querySelector('.btn-close');
                if (!closeBtn) {
                    closeBtn = document.createElement('button');
                    closeBtn.textContent = '×';
                    closeBtn.className = 'btn-close';
                    closeBtn.style.position = 'absolute';
                    closeBtn.style.right = '10px';
                    closeBtn.style.top = '10px';
                    closeBtn.onclick = () => offcanvasElement.style.display = 'none';
                    offcanvasElement.appendChild(closeBtn);
                }
                
                console.log("Provenance card shown successfully using fallback method");
            }
        } catch (e) {
            console.error("Failed to show offcanvas:", e);
        }
    }

    async rescoreCase(caseId) {
        const btn = this.$("btn-rescore");
        if (!btn) return;

        const originalHtml = btn.innerHTML;

        try {
            // Show loading state
            btn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Rescoring...';
            btn.disabled = true;

            // Call rescore API
            const response = await fetch(`/arena/api/scorables/${caseId}/rescore`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                const errorMessage = errorData.detail || `HTTP ${response.status}: ${response.statusText}`;
                throw new Error(errorMessage);
            }

            const result = await response.json();

            // Show success
            btn.innerHTML = '<i class="fas fa-check"></i> Rescored!';
            setTimeout(() => {
                btn.innerHTML = originalHtml;
                btn.disabled = false;
            }, 2000);

            // Update metrics display
            const metricsGrid = this.$("metrics-grid");
            if (metricsGrid && result.new_metrics) {
                metricsGrid.innerHTML = '';
                Object.entries(result.new_metrics).forEach(([key, value]) => {
                    if (typeof value === 'object' || key === 'meta') return;

                    const col = document.createElement('div');
                    col.className = 'col-6 mb-2';
                    col.innerHTML = `
                        <div class="card h-100">
                            <div class="card-body p-2">
                                <div class="small text-muted">${key}</div>
                                <div class="fw-bold">${typeof value === 'number' ? value.toFixed(3) : value}</div>
                            </div>
                        </div>
                    `;
                    metricsGrid.appendChild(col);
                });
            }

            // Emit event for live view to update
            const event = new CustomEvent('caseRescored', {
                detail: { caseId, newMetrics: result.new_metrics }
            });
            window.dispatchEvent(event);

        } catch (error) {
            console.error('Rescore failed:', error);
            btn.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Failed';
            setTimeout(() => {
                btn.innerHTML = originalHtml;
                btn.disabled = false;
            }, 2000);
            
            // FIXED: Safe assignment to error message element
            const errorMsg = this.$("error-message");
            if (errorMsg) {
                errorMsg.textContent = `Rescore failed: ${error.message}`;
                const errorEl = this.$("provenance-error");
                if (errorEl) errorEl.classList.remove("d-none");
            }
        }
    }

    $(selector) {
        if (selector.startsWith("#")) {
            return document.getElementById(selector.substring(1));
        }
        return document.getElementById(selector);
    }
}