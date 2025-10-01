export class ChartsComponent {
  constructor(enableLogging = true) {
    this.bestChart = null;
    this.margChart = null;
    this.initCheckInterval = null;
    this.state = {
      labels: [],
      best_history: [],
      marg_history: []
    };
    this.enableLogging = enableLogging;
    
    // Start initialization process
    this.initCharts();
    this.log("ChartsComponent initialized");
  }

  log(...args) {
    if (this.enableLogging) {
      console.log("[ChartsComponent]", ...args);
    }
  }

  destroy() {
    this.log("Destroying ChartsComponent");
    
    // Clear interval if exists
    if (this.initCheckInterval) {
      clearInterval(this.initCheckInterval);
      this.initCheckInterval = null;
    }
    
    // Destroy charts if they exist
    if (this.bestChart) {
      this.log("Destroying best chart");
      this.bestChart.destroy();
      this.bestChart = null;
    }
    
    if (this.margChart) {
      this.log("Destroying marginal chart");
      this.margChart.destroy();
      this.margChart = null;
    }
  }

  initCharts() {
    // First, ensure any existing charts are destroyed
    this.destroy();
    
    // Function to initialize charts when elements are ready
    const initializeCharts = () => {
      const bestCanvas = this.$("chart-best");
      const margCanvas = this.$("chart-marg");
      
      // Only proceed if at least one canvas exists
      if (!bestCanvas && !margCanvas) {
        this.log("No canvas elements found, continuing to wait...");
        return false;
      }
      
      // Initialize best chart if canvas exists
      if (bestCanvas) {
        try {
          // Ensure no existing chart on this canvas
          const existingBestChart = Chart.getChart(bestCanvas);
          if (existingBestChart) {
            existingBestChart.destroy();
          }
          
          const bestCtx = bestCanvas.getContext("2d");
          this.bestChart = new Chart(bestCtx, {
            type: "line",
            data: { 
              labels: [], 
              datasets: [{ 
                label: "best overall", 
                data: [], 
                borderWidth: 2,
                borderColor: '#0d6efd',
                backgroundColor: 'rgba(13, 110, 253, 0.1)',
                fill: true
              }]
            },
            options: { 
              responsive: true, 
              maintainAspectRatio: false,
              scales: { 
                y: { 
                  beginAtZero: false,
                  grid: { color: 'rgba(0, 0, 0, 0.05)' }
                }
              }, 
              plugins: { 
                legend: { display: false },
                tooltip: { 
                  mode: 'index', 
                  intersect: false,
                  callbacks: {
                    label: function(context) {
                      return `${context.dataset.label}: ${parseFloat(context.parsed.y).toFixed(2)}`;
                    }
                  }
                }
              }
            }
          });
          this.log("Best chart initialized");
        } catch (error) {
          this.log("Error initializing best chart:", error);
        }
      }
      
      // Initialize marginal chart if canvas exists
      if (margCanvas) {
        try {
          // Ensure no existing chart on this canvas
          const existingMargChart = Chart.getChart(margCanvas);
          if (existingMargChart) {
            existingMargChart.destroy();
          }
          
          const margCtx = margCanvas.getContext("2d");
          this.margChart = new Chart(margCtx, {
            type: "line",
            data: { 
              labels: [], 
              datasets: [{ 
                label: "marginal/kTok", 
                data: [], 
                borderWidth: 2,
                borderColor: '#20c997',
                backgroundColor: 'rgba(32, 201, 151, 0.1)',
                fill: true
              }]
            },
            options: { 
              responsive: true, 
              maintainAspectRatio: false,
              scales: { 
                y: { 
                  beginAtZero: false,
                  grid: { color: 'rgba(0, 0, 0, 0.05)' }
                }
              }, 
              plugins: { 
                legend: { display: false },
                tooltip: { 
                  mode: 'index', 
                  intersect: false,
                  callbacks: {
                    label: function(context) {
                      return `${context.dataset.label}: ${parseFloat(context.parsed.y).toFixed(2)}`;
                    }
                  }
                }
              }
            }
          });
          this.log("Marginal chart initialized");
        } catch (error) {
          this.log("Error initializing marginal chart:", error);
        }
      }
      
      // If both charts were successfully initialized, stop checking
      if ((bestCanvas && this.bestChart) || (margCanvas && this.margChart)) {
        if (this.initCheckInterval) {
          clearInterval(this.initCheckInterval);
          this.initCheckInterval = null;
        }
        return true;
      }
      
      return false;
    };
    
    // Initial check
    if (initializeCharts()) return;
    
    // Set up periodic checks if elements aren't ready yet
    this.initCheckInterval = setInterval(() => {
      if (initializeCharts()) {
        clearInterval(this.initCheckInterval);
        this.initCheckInterval = null;
      }
    }, 100);
  }

  updateCharts(labels, best_history, marg_history) {
    // Validate inputs
    if (!Array.isArray(labels) || !Array.isArray(best_history) || !Array.isArray(marg_history)) {
      this.log("ERROR: Invalid data format for updateCharts", {labels, best_history, marg_history});
      return;
    }
    
    // Store state
    this.state.labels = [...labels];
    this.state.best_history = [...best_history];
    this.state.marg_history = [...marg_history];
    
    this.log("Updating charts", {
      labelsCount: this.state.labels.length,
      bestPoints: this.state.best_history.filter(v => v != null).length,
      margPoints: this.state.marg_history.filter(v => v != null).length
    });
    
    // Update best chart
    if (this.bestChart) {
      try {
        this.bestChart.data.labels = this.state.labels;
        
        // Convert to numbers, handle null/undefined
        const processedBest = this.state.best_history.map(v => 
          v != null ? Number(v) : null
        );
        
        // Check for all NaN values
        const hasValidBestData = processedBest.some(v => v !== null && !isNaN(v));
        if (!hasValidBestData) {
          this.log("WARNING: No valid numeric data for best chart");
        }
        
        this.bestChart.data.datasets[0].data = processedBest;
        this.bestChart.update();
        this.log("Best chart updated with", processedBest.length, "points");
      } catch (error) {
        this.log("Error updating best chart:", error);
      }
    }
    
    // Update marginal chart
    if (this.margChart) {
      try {
        this.margChart.data.labels = this.state.labels;
        
        // Convert to numbers, handle null/undefined
        const processedMarg = this.state.marg_history.map(v => 
          v != null ? Number(v) : null
        );
        
        // Check for all NaN values
        const hasValidMargData = processedMarg.some(v => v !== null && !isNaN(v));
        if (!hasValidMargData) {
          this.log("WARNING: No valid numeric data for marginal chart");
        }
        
        this.margChart.data.datasets[0].data = processedMarg;
        this.margChart.update();
        this.log("Marginal chart updated with", processedMarg.length, "points");
      } catch (error) {
        this.log("Error updating marginal chart:", error);
      }
    }
    
    // If no charts were updated, log warning
    if (!this.bestChart && !this.margChart) {
      this.log("WARNING: updateCharts called but no charts are initialized");
    }
  }

  clear() {
    this.log("Clearing chart data");
    
    this.state.labels = [];
    this.state.best_history = [];
    this.state.marg_history = [];
    
    if (this.bestChart) {
      this.bestChart.data.labels = [];
      this.bestChart.data.datasets[0].data = [];
      this.bestChart.update();
    }
    
    if (this.margChart) {
      this.margChart.data.labels = [];
      this.margChart.data.datasets[0].data = [];
      this.margChart.update();
    }
  }

  $(selector) {
    if (selector.startsWith("#")) {
      return document.getElementById(selector.substring(1));
    }
    return document.getElementById(selector);
  }
}