# SC4 Integration Implementation

> Historical implementation record from before SC4A execution integration.
> Current implementation and acceptance status are recorded in
> `SC4A-FUNCTIONAL-INTEGRATION.md` and `SC4B-END-TO-END-EVIDENCE.md`.

Mission now has an **Improvement** navigation entry at `/improvement`. It reads Work Episodes/findings, Work Corrective proposals, Work Envelopes, Harness catalogue, Mission Praxis proposal records, and a bounded projection of the isolated SC3 artifact snapshot.

Selecting an open finding can create a Mission `REQUEST_USER_DECISION` Work Outcome proposal through the existing Work Corrective service. The request is inert: it does not invoke Stephanie, authorize an experiment, execute Worker, or evaluate/promote a candidate.

The read-only `/api/improvement/records` endpoint validates the protocol, candidate, hypothesis and a post-hoc snapshot manifest over result, experience and AI-trace files. It excludes raw AI response bodies and evaluator contents. The manifest checks consistency, not signature/provenance. No new DB table or migration was added.

Mission remains the authority for Work Outcome, Work Envelope, approval, human decision and formal acceptance. Stephanie is currently CLI-only from Mission's perspective. No SC4 full experiment dispatch flow was implemented; no isolated end-to-end workflow was demonstrated.
