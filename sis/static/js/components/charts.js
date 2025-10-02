// stephanie/static/arena/js/components/charts.js
export class ChartsComponent {
    constructor(arenaState, provenanceCard, enableLogging = true) {
        this.arenaState = arenaState;
        this.provenanceCard = provenanceCard;
        this.enableLogging = enableLogging;
        this.bestChart = null;
        this.margChart = null;
        this.initCheckInterval = null;
        
        // Initialize charts first
        this.initCharts();
        
        // Subscribe to state changes after charts are initialized
        this.unsubscribe = arenaState.subscribe(state => {
            this.updateFromState();
        });
        
        this.log("ChartsComponent initialized");
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

    updateFromState() {
        const state = this.arenaState.getState();
        const chartData = state.chartData;
        
        // Validate inputs
        if (!chartData || !Array.isArray(chartData.labels) || 
            !Array.isArray(chartData.best_history) || 
            !Array.isArray(chartData.marg_history)) {
            this.log("ERROR: Invalid chart data format", chartData);
            return;
        }
        
        this.log("Updating charts from state", {
            labelsCount: chartData.labels.length,
            bestPoints: chartData.best_history.filter(v => v != null).length,
            margPoints: chartData.marg_history.filter(v => v != null).length
        });
        
        // Update best chart
        if (this.bestChart) {
            try {
                this.bestChart.data.labels = chartData.labels;
                
                // Convert to numbers, handle null/undefined
                const processedBest = chartData.best_history.map(v => 
                    v != null ? Number(v) : null
                );
                
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
                this.margChart.data.labels = chartData.labels;
                
                // Convert to numbers, handle null/undefined
                const processedMarg = chartData.marg_history.map(v => 
                    v != null ? Number(v) : null
                );
                
                this.margChart.data.datasets[0].data = processedMarg;
                this.margChart.update();
                this.log("Marginal chart updated with", processedMarg.length, "points");
            } catch (error) {
                this.log("Error updating marginal chart:", error);
            }
        }
        
        // Set up click handlers if provenance card is available
        this.setupChartClickHandlers();
    }

    setupChartClickHandlers() {
        // Only set up click handlers if provenance card is available
        if (!this.provenanceCard) return;
        
        const setupClickHandler = (chart, chartType) => {
            if (!chart || !chart.canvas) return;
            
            // Remove existing click handler if exists
            if (chart.canvas.onclick) {
                chart.canvas.onclick = null;
            }
            
            // Add new click handler
            chart.canvas.onclick = (event) => {
                const points = chart.getElementsAtEventForMode(
                    event, 
                    'nearest', 
                    { intersect: true }, 
                    false
                );
                
                if (points.length > 0) {
                    const firstPoint = points[0];
                    const index = firstPoint.index;
                    
                    this.log(`Clicked on ${chartType} chart at index ${index}`);
                    
                    // Show provenance for the clicked data point
                    const state = this.arenaState.getState();
                    if (state.chartData.caseIds && state.chartData.caseIds[index]) {
                        const caseId = state.chartData.caseIds[index];
                        const value = chart.data.datasets[0].data[index];
                        
                        if (caseId && this.provenanceCard) {
                            // Find the round number for this case
                            let roundNumber = 1;
                            const caseData = this.arenaState.getTopKData(caseId);
                            if (caseData && caseData.roundNumbers.length > 0) {
                                roundNumber = caseData.roundNumbers[caseData.roundNumbers.length - 1];
                            }
                            
                            this.provenanceCard.openCard(
                                caseId,
                                value,
                                'chart-click',
                                chartType
                            );
                        }
                    }
                }
            };
            
            // Make canvas elements have pointer cursor
            chart.canvas.style.cursor = 'pointer';
        };
        
        setupClickHandler(this.bestChart, 'best');
        setupClickHandler(this.margChart, 'marginal');
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
        
        // Unsubscribe from state
        if (this.unsubscribe) {
            this.unsubscribe();
            this.unsubscribe = null;
        }
    }

    log(...args) {
        if (this.enableLogging) {
            console.log("[ChartsComponent]", ...args);
        }
    }

    $(selector) {
        if (selector.startsWith("#")) {
            return document.getElementById(selector.substring(1));
        }
        return document.getElementById(selector);
    }
}