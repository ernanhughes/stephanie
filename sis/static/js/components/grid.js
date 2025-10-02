// stephanie/static/arena/js/components/grid.js
export class GridComponent {
    constructor(arenaState, provenanceCard) {
        this.arenaState = arenaState;
        this.provenanceCard = provenanceCard;
        this.updateTimeout = null;
        
        // Subscribe to state changes
        this.unsubscribe = arenaState.subscribe(state => {
            this.updateFromState();
        });
    }

    destroy() {
        // Unsubscribe from state
        if (this.unsubscribe) {
            this.unsubscribe();
            this.unsubscribe = null;
        }
    }

    updateFromState() {
        const topkBody = document.getElementById('topk-body');
        if (!topkBody) {
            console.error("[GridComponent] Top-K table body not found");
            return;
        }

        const state = this.arenaState.getState();
        let currentCaseId = state.currentCaseId;
        
        // CRITICAL FIX: Ensure case_id is consistently a string
        if (currentCaseId) {
            currentCaseId = String(currentCaseId);
        }
        
        // CRITICAL FIX: If no current case, try to get from state
        if (!currentCaseId && state.events && state.events.length > 0) {
            // Try to find the most recent case_id from events
            for (let i = state.events.length - 1; i >= 0; i--) {
                if (state.events[i].case_id) {
                    currentCaseId = String(state.events[i].case_id);
                    this.arenaState.setState({ currentCaseId }, false);
                    break;
                }
            }
        }
        
        if (!currentCaseId) {
            topkBody.innerHTML = '<tr><td colspan="5" class="text-muted">Select a case to view Top-K data</td></tr>';
            return;
        }
        
        // CRITICAL FIX: Ensure case is initialized in state with string case_id
        this.arenaState.ensureCaseInitialized(currentCaseId);
        const caseData = this.arenaState.getTopKData(currentCaseId);
        
        // Always rebuild the ENTIRE grid from our state
        topkBody.innerHTML = '';
        
        // Build the grid from our complete state
        for (const roundNumber of caseData.roundNumbers) {
            const topkData = caseData.rounds.get(roundNumber);
            
            // Skip if data is invalid
            if (!topkData || !Array.isArray(topkData) || topkData.length === 0) continue;
            
            // Add round separator
            const separator = document.createElement('tr');
            separator.className = 'table-secondary';
            separator.innerHTML = `<td colspan="5" class="small fw-bold py-1">Round ${roundNumber}</td>`;
            topkBody.appendChild(separator);
            
            // Process each Top-K entry for this round
            topkData.forEach(item => {
                const formatValue = (value, decimals = 2) => {
                    if (typeof value === 'number') {
                        return value.toFixed(decimals);
                    }
                    return value || 'N/A';
                };
                
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${item.origin || 'N/A'}</td>
                    <td>${item.variant || 'N/A'}</td>
                    <td class="text-end">${formatValue(item.overall)}</td>
                    <td class="text-end">${formatValue(item.k)}</td>
                    <td>${item.verified ? '<span class="text-success">✓</span>' : '<span class="text-danger">✗</span>'}</td>
                `;
                
                // Add click handler for provenance
                if (this.provenanceCard && item.case_id) {
                    // CRITICAL FIX: Ensure case_id is consistently a string
                    const caseId = String(item.case_id);
                    
                    row.style.cursor = 'pointer';
                    row.classList.add('table-row-hover');
                    
                    // Add data attributes for provenance lookup
                    row.dataset.caseId = caseId;
                    row.dataset.score = item.overall;
                    row.dataset.variant = item.variant;
                    row.dataset.origin = item.origin;
                    
                    // Add click event handler
                    row.addEventListener('click', () => {
                        console.log(`[Grid] Clicked item with case_id: ${caseId}`);
                        this.provenanceCard.openCard(
                            caseId,
                            item.overall,
                            item.variant,
                            item.origin
                        );
                    });
                }
                
                row.className = 'new-entry';
                topkBody.appendChild(row);
                
                // Remove highlight after brief delay
                setTimeout(() => {
                    if (row.classList.contains('new-entry')) {
                        row.classList.remove('new-entry');
                    }
                }, 1500);
            });
        }
        
        // Handle empty state
        if (caseData.roundNumbers.length === 0) {
            // CRITICAL FIX: Add debug logging
            console.log("[Grid] No rounds for case", currentCaseId, {
                topkByCase: Array.from(this.arenaState.getState().topkByCase.entries()),
                chartData: this.arenaState.getState().chartData,
                allEvents: this.arenaState.getState().events.map(e => ({ 
                    event: e.event, 
                    case_id: e.case_id,
                    hasTopK: !!e.topk
                }))
            });
            
            const row = document.createElement('tr');
            row.innerHTML = `<td colspan="5" class="text-muted">No Top-K data for this case</td>`;
            topkBody.appendChild(row);
        }
        
        // Keep scroll position at bottom if autoscroll is enabled
        const timelineEl = document.querySelector('#timeline');
        if (timelineEl && document.getElementById('autoscroll')?.checked) {
            timelineEl.scrollTop = timelineEl.scrollHeight;
        }
        
        console.log(`[TopK] Grid updated with ${caseData.roundNumbers.length} rounds of data for case ${currentCaseId}`);
    }

    // Add to your new GridComponent class
    clear() {
        this.updateFromState(); // Rebuild with current state
    }

    resetTopK() {
        // Clear the current case's data in state
        const state = this.arenaState.getState();
        if (state.currentCaseId) {
            const caseId = String(state.currentCaseId);
            const caseData = this.arenaState.getTopKData(caseId);
            caseData.rounds.clear();
            caseData.roundNumbers = [];
            this.updateFromState();
        }
    }

    setTopK(topkData, roundNumber = null, caseId = null) {
        // In centralized state, this should notify the ArenaState instead
        if (caseId && roundNumber) {
            this.arenaState.initialScored(String(caseId), roundNumber, topkData);
        }
        // The state subscription will automatically call updateFromState()
    }

    $(selector) {
        if (selector.startsWith("#")) {
            return document.getElementById(selector.substring(1));
        }
        return document.getElementById(selector);
    }
}