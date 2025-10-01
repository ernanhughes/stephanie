// stephanie/static/arena/js/components/grid.js
export class GridComponent {
    constructor() {
        this.state = {
            topkByRound: new Map(),  // Store ALL Top-K data by round number
            labels: [],
            best_history: [],
            marg_history: []
        };
        this.logger = console;
        this.updateTimeout = null; // For debouncing rapid updates
    }

    /**
     * Sets Top-K data for display in the grid
     * @param {Array} topkData - Array of Top-K items to display
     * @param {number|null} roundNumber - Round number this data belongs to
     */
    setTopK(topkData, roundNumber = null) {
        // Ensure we have a valid round number
        if (roundNumber === null || roundNumber === 0) {
            // Try to get from event handler if available
            roundNumber = window.eventHandler?.roundIdx || 
                          (this.state.topkByRound.size + 1);
        }
        
        // Debounce rapid updates (prevents browser overload)
        if (this.updateTimeout) {
            clearTimeout(this.updateTimeout);
        }
        
        this.updateTimeout = setTimeout(() => {
            this._updateGrid(topkData, roundNumber);
            this.updateTimeout = null;
        }, 50); // Small delay to batch rapid updates
    }

    /**
     * Internal method to update the grid (called after debounce)
     */
    _updateGrid(topkData, roundNumber) {
        const topkBody = document.getElementById('topk-body');
        if (!topkBody) {
            console.error("[GridComponent] Top-K table body not found");
            return;
        }

        // Store data in our persistent state
        this.state.topkByRound.set(roundNumber, topkData);
        
        // Always rebuild the ENTIRE grid from our state
        // This ensures consistency and avoids race conditions
        topkBody.innerHTML = '';
        
        // Sort rounds numerically (not alphabetically)
        const sortedRounds = Array.from(this.state.topkByRound.keys())
            .sort((a, b) => a - b);
        
        // Build the grid from our complete state
        for (const round of sortedRounds) {
            const data = this.state.topkByRound.get(round);
            
            // Skip if data is invalid
            if (!data || !Array.isArray(data) || data.length === 0) continue;
            
            // Add round separator
            const separator = document.createElement('tr');
            separator.className = 'table-secondary';
            separator.innerHTML = `<td colspan="5" class="small fw-bold py-1">Round ${round}</td>`;
            topkBody.appendChild(separator);
            
            // Process each Top-K entry for this round
            data.forEach(item => {
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
        if (sortedRounds.length === 0 || 
            Array.from(this.state.topkByRound.values()).every(data => !data || data.length === 0)) {
            const row = document.createElement('tr');
            row.innerHTML = `<td colspan="5" class="text-muted">Waiting for Top-K data...</td>`;
            topkBody.appendChild(row);
        }
        
        // Keep scroll position at bottom if autoscroll is enabled
        const timelineEl = document.querySelector('#timeline');
        if (timelineEl && document.getElementById('autoscroll')?.checked) {
            timelineEl.scrollTop = timelineEl.scrollHeight;
        }
        
        console.log(`[TopK] Grid updated with ${sortedRounds.length} rounds of data`);
    }

    // Add a reset method to clear state when starting new run
    resetTopK() {
        // Clear our internal state
        this.state.topkByRound = new Map();
        
        const topkBody = document.getElementById('topk-body');
        if (topkBody) {
            topkBody.innerHTML = '<tr><td colspan="5" class="text-muted">Waiting for data...</td></tr>';
        }
        
        console.log("[TopK] Grid reset");
    }

    clear() {
        this.resetTopK();
        
        // Clear other state
        this.state.labels = [];
        this.state.best_history = [];
        this.state.marg_history = [];
    }

    /**
     * Helper method for DOM selection
     */
    $(selector) {
        if (selector.startsWith("#")) {
            return document.getElementById(selector.substring(1));
        }
        return document.getElementById(selector);
    }
}