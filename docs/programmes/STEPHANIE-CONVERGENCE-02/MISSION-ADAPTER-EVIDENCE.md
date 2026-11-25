# Mission adapter evidence

Read-only GET surfaces inspected and used: `/api/work-episodes?limit=N` (max 200) and `/api/work-observation/signals?limit=N` (max 500). The service answered on port 8790. The retrieved records retain Mission IDs and project keys.

These routes expose an episode index and observation signals, not a verified Worker association or a formal accepted Work Outcome. This adapter does not call envelope mutation, evaluation runs, or proposal routes. A future adapter requires an explicit read route for Work Outcome identity, linked evidence, independent evaluation, and acceptance state, plus exact association semantics. Acceptance remains UNKNOWN.
