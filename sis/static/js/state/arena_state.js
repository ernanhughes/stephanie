// stephanie/static/arena/js/state/arena_state.js
export class ArenaState {
    constructor() {
        this.state = {
            // Core identifiers
            run_id: null,
            case_id: null,
            section_name: null,
            paper_id: null,
            
            // Event history
            events: [],
            
            // Top-K data organized by case
            topkByCase: new Map(), // case_id -> { rounds: Map(roundNumber -> topkData), roundNumbers: [] }
            
            // Chart data (single source of truth)
            chartData: {
                labels: [],
                best_history: [],
                marg_history: [],
                caseIds: [] // Critical for provenance
            },
            
            // Current context
            currentRunId: null,
            currentCaseId: null,
            
            // Summary data
            summary: null,
            winner_excerpt: null,
            
            // UI state
            chips: {
                run: null,
                case: null,
                paper: null,
                section: null
            }
        };
        
        this.listeners = [];
        this.actionHistory = [];
        this.maxHistory = 100;
    }

    subscribe(listener) {
        this.listeners.push(listener);
        
        // Initial state push
        setTimeout(() => listener(this.getState()), 0);
        
        // Return unsubscribe function
        return () => {
            this.listeners = this.listeners.filter(l => l !== listener);
        };
    }

    // CRITICAL FIX: Proper setState implementation
    setState(partialState, silent = false, actionType = 'setState') {
        const prevState = { ...this.state };
        this.state = { ...this.state, ...partialState };
        
        // Record action for debugging
        if (!silent) {
            this.actionHistory.push({
                type: actionType,
                prevState,
                newState: this.state,
                timestamp: Date.now()
            });
            
            // Keep history bounded
            if (this.actionHistory.length > this.maxHistory) {
                this.actionHistory.shift();
            }
        }
        
        if (!silent) {
            this.notifyListeners();
        }
    }

    // CRITICAL: Add this missing method
    notifyListeners() {
        const currentState = this.getState();
        this.listeners.forEach(listener => {
            try {
                listener(currentState);
            } catch (error) {
                console.error("State listener error:", error);
            }
        });
    }

    // CRITICAL: Add this method to safely get state
    getState() {
        return { ...this.state };
    }

    // Add this method to ensure case is initialized
    ensureCaseInitialized(case_id) {
        if (!this.state.topkByCase.has(case_id)) {
            this.state.topkByCase.set(case_id, {
                rounds: new Map(),
                roundNumbers: []
            });
        }
        return this.state.topkByCase.get(case_id);
    }

    arenaStart(run_id, case_id, section_name, paper_id) {
        // Ensure case is initialized first
        this.ensureCaseInitialized(case_id);
        
        this.setState({
            run_id,
            case_id,
            section_name,
            paper_id,
            currentRunId: run_id,
            currentCaseId: case_id,
            chips: {
                run: `run_id: ${run_id}`,
                case: `case: ${case_id}`,
                paper: `paper: ${paper_id}`,
                section: `section: ${section_name}`
            }
        }, false, 'arenaStart');
    }

    roundBegin(case_id) {
        case_id = String(case_id);
        
        // Ensure case is initialized before accessing
        this.ensureCaseInitialized(case_id);
        const caseData = this.state.topkByCase.get(case_id);
        
        const roundNumber = caseData.roundNumbers.length + 1;
        caseData.roundNumbers.push(roundNumber);
        
        // Update chart data
        const chartData = { ...this.state.chartData };
        chartData.labels.push(`r${roundNumber}`);
        chartData.best_history.push(null);
        chartData.marg_history.push(null);
        chartData.caseIds.push(case_id);
        
        this.setState({
            chartData,
            currentCaseId: case_id
        }, false, 'roundBegin');
        
        return roundNumber;
    }

    initialScored(case_id, roundNumber, topkData) {
        // CRITICAL FIX: Ensure case_id is a string
        case_id = String(case_id);
        
        // Ensure case is initialized before accessing
        this.ensureCaseInitialized(case_id);
        const caseData = this.state.topkByCase.get(case_id);
        
        caseData.rounds.set(roundNumber, topkData);
        
        this.setState({}, false, 'initialScored');
    }

    roundEnd(case_id, roundNumber, best_overall, marginal_per_ktok) {
        case_id = String(case_id);
        
        const chartData = { ...this.state.chartData };
        const index = chartData.caseIds.lastIndexOf(case_id);
        
        if (index >= 0) {
            chartData.best_history[index] = best_overall;
            chartData.marg_history[index] = marginal_per_ktok;
            
            this.setState({ chartData }, false, 'roundEnd');
        } else {
            console.warn(`[ArenaState] roundEnd: case_id ${case_id} not found in chartData`);
        }
    }

    arenaStop(summary) {
        this.setState({
            summary,
            winner_excerpt: summary.winner_excerpt
        }, false, 'arenaStop');
    }

    reset() {
        this.state = {
            run_id: null,
            case_id: null,
            section_name: null,
            paper_id: null,
            events: [],
            topkByCase: new Map(),
            chartData: {
                labels: [],
                best_history: [],
                marg_history: [],
                caseIds: []
            },
            currentRunId: null,
            currentCaseId: null,
            summary: null,
            winner_excerpt: null,
            chips: {
                run: null,
                case: null,
                paper: null,
                section: null
            }
        };
        
        this.notifyListeners();
    }

    // Helper to get topk data for a case
    getTopKData(case_id) {
        return this.state.topkByCase.get(case_id) || {
            rounds: new Map(),
            roundNumbers: []
        };
    }

    // Helper to get chart data
    getChartData() {
        return this.state.chartData;
    }

    // Debugging utilities
    getActionHistory() {
        return [...this.actionHistory];
    }

    debugState() {
        console.log("[ArenaState] Current state:", JSON.parse(JSON.stringify(this.state)));
        return this.state;
    }
}