// sis/static/arena/js/components/charts.js

export class ChartsComponent {
  constructor() {
    this.bestChart = null;
    this.margChart = null;
    this.state = {
      labels: [],
      best_history: [],
      marg_history: []
    };
    
    this.initCharts();
  }

  initCharts() {
    // Best overall chart
    const bestCtx = this.$("chart-best")?.getContext("2d");
    if (bestCtx) {
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
          scales: { 
            y: { 
              beginAtZero: false,
              grid: {
                color: 'rgba(0, 0, 0, 0.05)'
              }
            }
          }, 
          plugins: { 
            legend: { display: false },
            tooltip: {
              mode: 'index',
              intersect: false
            }
          }
        }
      });
    }

    // Marginal/kTok chart
    const margCtx = this.$("chart-marg")?.getContext("2d");
    if (margCtx) {
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
          scales: { 
            y: { 
              beginAtZero: false,
              grid: {
                color: 'rgba(0, 0, 0, 0.05)'
              }
            }
          }, 
          plugins: { 
            legend: { display: false },
            tooltip: {
              mode: 'index',
              intersect: false
            }
          }
        }
      });
    }
  }

  updateCharts(labels, best_history, marg_history) {
    this.state.labels = labels.slice();
    this.state.best_history = best_history.slice();
    this.state.marg_history = marg_history.slice();
    
    if (this.bestChart) {
      this.bestChart.data.labels = this.state.labels;
      this.bestChart.data.datasets[0].data = this.state.best_history;
      this.bestChart.update();
    }
    
    if (this.margChart) {
      this.margChart.data.labels = this.state.labels;
      this.margChart.data.datasets[0].data = this.state.marg_history;
      this.margChart.update();
    }
  }

  clear() {
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