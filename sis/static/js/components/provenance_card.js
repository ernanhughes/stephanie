// sis/static/arena/js/components/provenance_card.js
export class ProvenanceCardComponent {
  constructor() {
    // Nothing to initialize here
  }

  async openCard(caseId, score, variant, origin) {
    // Show loading state
    this.$("provenance-loading")?.classList.remove("d-none");
    this.$("provenance-content")?.classList.add("d-none");
    this.$("provenance-error")?.classList.add("d-none");
    
    try {
      console.log("Provenance card opened", {
        caseId: caseId,
        score: score,
        variant: variant,
        origin: origin,
        timestamp: new Date().toISOString()
      });
      
      // Fetch provenance data
      const response = await fetch(`/arena/api/provenance/${caseId}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      // Hide loading, show content
      this.$("provenance-loading")?.classList.add("d-none");
      this.$("provenance-content")?.classList.remove("d-none");
      
      // Populate case info
      this.$("case-name").textContent = data.case?.name || `Case ${caseId}`;
      this.$("case-agent").textContent = `Agent: ${data.case?.agent_name || origin || 'Unknown'}`;
      this.$("case-meta").textContent = JSON.stringify(data.case?.meta || {}, null, 2);
      
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
      this.$("btn-rescore").onclick = () => this.rescoreCase(caseId);
      this.$("btn-view-full").onclick = () => {
        if (data.actions && data.actions.view_full_url) {
          window.open(data.actions.view_full_url, '_blank');
        }
      };
      this.$("btn-edit").onclick = () => {
        if (data.actions && data.actions.edit_url) {
          window.open(data.actions.edit_url, '_blank');
        }
      };
      
      // Store current case ID for actions
      this.$("provenanceCard").dataset.caseId = caseId;
      
      // 🔥 CRITICAL: Properly invoke Bootstrap offcanvas
      try {
        const offcanvasElement = this.$("provenanceCard");
        if (offcanvasElement && typeof bootstrap !== 'undefined' && bootstrap.Offcanvas) {
          const offcanvas = new bootstrap.Offcanvas(offcanvasElement);
          offcanvas.show();
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
          if (!offcanvasElement.querySelector('.btn-close')) {
            const closeBtn = document.createElement('button');
            closeBtn.textContent = '×';
            closeBtn.className = 'btn-close';
            closeBtn.style.position = 'absolute';
            closeBtn.style.right = '10px';
            closeBtn.style.top = '10px';
            closeBtn.onclick = () => offcanvasElement.style.display = 'none';
            offcanvasElement.appendChild(closeBtn);
          }
        }
      } catch (e) {
        console.error("Bootstrap Offcanvas failed:", e);
      }
      
      console.log("Provenance card shown successfully");
      
    } catch (error) {
      console.error('Failed to load provenance:', error);
      this.$("provenance-loading")?.classList.add("d-none");
      this.$("provenance-error")?.classList.remove("d-none");
      this.$("error-message").textContent = error.message || String(error);
      
      // 🔥 CRITICAL: Still show the offcanvas even on error
      try {
        const offcanvasElement = this.$("provenanceCard");
        if (offcanvasElement && typeof bootstrap !== 'undefined' && bootstrap.Offcanvas) {
          const offcanvas = new bootstrap.Offcanvas(offcanvasElement);
          offcanvas.show();
        } else {
          // Fallback: show card directly
          offcanvasElement.style.display = 'block';
        }
      } catch (e) {
        console.error("Bootstrap fallback failed:", e);
      }
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
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      // Show success
      btn.innerHTML = '<i class="fas fa-check"></i> Rescored!';
      setTimeout(() => {
        btn.innerHTML = originalHtml;
        btn.disabled = false;
      }, 2000);
      
      // Update metrics display
      if (result.new_metrics) {
        const metricsGrid = this.$("metrics-grid");
        if (metricsGrid) {
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
    }
  }

  $(selector) {
    if (selector.startsWith("#")) {
      return document.getElementById(selector.substring(1));
    }
    return document.getElementById(selector);
  }
}