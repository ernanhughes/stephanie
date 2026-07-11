# stephanie/constants.py
from __future__ import annotations

# NATS subjects and streams 
# PLEASE KEEP CONSISTENT ACROSS FILES

BUS_STREAM = "stephanie"  # your bus auto-prefixes this; keep subjects SHORT

PROMPT_SUBMIT = "prompts.submit"           # submit jobs here
PROMPT_RESULT =  "results.prompts"
PROMPT_RESULT_WC = "results.prompts.>"     # client listens here
PROMPT_RESULT_TMPL = "results.prompts.{job}"  # worker/client agrees on this
BUS_STREAM = "stephanie"
SUBJ_SUBMIT = "prompts.submit"
SUBJ_RESULT_WC = "results.prompts.>"

SUBJ_SUBMIT_NS = "stephanie.prompts.submit"
SUBJ_SUBMIT_LEG = "prompts.submit"
SUBJ_RESULT_NS_W = "stephanie.results.prompts.*"
SUBJ_RESULT_LEG_W = "results.prompts.*"
SUBJ_RESULT_NS_T = "results.prompts.{job}"

PROMPT_DLQ =  "prompts.submit.DLQ"  # dead-letter queue
HEALTH_SUBJ = "health"              # health check subject

NEXUS_TIMELINE_NODE   = "nexus.timeline.node"
NEXUS_TIMELINE_REPORT = "nexus.timeline.report"


# Scorables queues/subjects
SCORABLE_SUBMIT = "scorable.submit"
SCORABLE_PROCESS = "scorable.process"


# ==== Context Keys ====
AGENT = "Agent"
AGENT_NAME = "agent_name"
API_BASE = "api_base"
CASEBOOK = "casebook"
CASEBOOK_ID = "casebook_id"
INCLUDE_MARS = "include_mars"
API_KEY = "api_key"
BATCH_SIZE = "batch_size"
CONTEXT = "context"
DEFAULT = "default"
DETAILS = "details"
EVOLVED = "evolved"
FEEDBACK = "feedback"
FILE = "file"
GOAL = "goal"
GOAL_TYPE = "goal_type"
GOAL_TEXT = "goal_text"
HYPOTHESES = "hypotheses"
DATABASE_MATCHES = "database_matches"
TEXT = "text"
SCORABLES = "scorables"
SCORABLE_DETAILS = "scorable_details"
LOOKAHEAD = "lookahead"
MODEL = "model"
NAME = "name"
PIPELINE = "pipeline"
PROXIMITY = "proximity"
RANKING = "ranking"
DOCUMENTS = "documents"
REFLECTION = "reflection"
REVIEW = "review"
REPORTS = "REPORTS"
REVIEWS = "reviews"
RUN_ID = "run_id"
PIPELINE_RUN_ID = "pipeline_run_id"
PLAN_TRACE_ID = "plan_trace_id"
SCORE = "score"
SOURCE = "source"
STAGE = "stage"
STRATEGY = "strategy"
VERSION = "version"

# ==== Config Keys ====
DATABASE = "database"
EVENT = "event"
INPUT_KEY = "input_key"
OUTPUT_KEY = "output_key"
PROMPT_DIR = "prompt_dir"
PROMPT_FILE = "prompt_file"
PROMPT_MATCH_RE = "prompt_match_re"
PROMPT_MODE = "prompt_mode"
PROMPT_PATH = "prompt_path"
SAVE_CONTEXT = "save_context"
SAVE_PROMPT = "save_prompt"
SKIP_IF_COMPLETED = "skip_if_completed"

# metrics
METRICS_REQ  = "arena.metrics.request"
METRICS_OK   = "arena.metrics.ready"
ATS_NODE     = "arena.ats.node"     # if you already emit node events, keep it
ATS_REPORT   = "arena.ats.report"
