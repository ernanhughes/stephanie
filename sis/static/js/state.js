// sis/static/arena/js/state.js
export const createInitialState = () => ({
  run_id: null,
  case_id: null,
  section_name: null,
  events: [],

  // Top-K related
  topk: [], // always an array
  current_round: null, // round_number of latest processed round
  round_map: {}, // { case_id: { round_number: n, topk: [] } }

  // Chart related
  labels: [],
  best_history: [],
  marg_history: [],

  // Summary
  summary: null,
  winner_excerpt: null,
});
